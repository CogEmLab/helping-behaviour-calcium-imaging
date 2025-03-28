import os
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Dict, Tuple, List
from src.plotting import plot_correlated_traces, plot_cross_correlation_matrix, plot_distance_matrix
import yaml
from scipy.spatial.distance import cdist
import networkx as nx
import datetime
import neurokit2 as nk
import logging
from cdlib import evaluation, NodeClustering
import leidenalg
import igraph as ig

def compute_event_vs_session_entropy(dff_signal, victim_lib_timecode, fps, window_seconds=15):
    """Compute entropy for event window vs rest of session"""
    if victim_lib_timecode is None:
        return None
        
    window_frames = int(window_seconds * fps)
    event_frame = int(victim_lib_timecode * fps)
    
    # Check if event window would exceed signal boundaries
    if event_frame - window_frames < 0 or event_frame + window_frames > dff_signal.shape[1]:
        print(f"Warning: Event window ({event_frame}±{window_frames}) exceeds signal boundaries (0-{dff_signal.shape[1]})")
        return None
    
    # Extract event window and rest of session
    event_window = dff_signal[:, event_frame-window_frames:event_frame+window_frames]
    pre_event = dff_signal[:, event_frame-window_frames:event_frame]
    post_event = dff_signal[:, event_frame:event_frame+window_frames]
    rest_session = np.concatenate([
        dff_signal[:, :event_frame-window_frames],
        dff_signal[:, event_frame+window_frames:]
    ], axis=1)
    
    # Verify extracted segments are not empty
    if event_window.size == 0 or rest_session.size == 0:
        print("Warning: Empty signal segments detected")
        return None
    
    entropy_scores = {
        'pre_event': {'individual_mse': {}, 'valid_scales': []},
        'post_event': {'individual_mse': {}, 'valid_scales': []},
        'event_window': {'individual_mse': {}},
        'rest_session': {'individual_mse': {}},
        'neuron_scores': {}
    }
    
    # Calculate entropy for all periods
    entropy_changes = []
    activity_scores = []
    
    for neuron_id in range(dff_signal.shape[0]):
        # Calculate event window activity
        event_signal = event_window[neuron_id, :]
        rest_signal = rest_session[neuron_id, :]
        
        # Safe calculation of activity scores
        event_peak = np.max(np.abs(event_signal)) if event_signal.size > 0 else 0
        rest_std = np.std(rest_signal) if rest_signal.size > 0 else 1
        rest_mean = np.mean(rest_signal) if rest_signal.size > 0 else 0
        
        # Calculate activity score using peak amplitude during event
        if rest_std != 0:
            activity_score = (event_peak - rest_mean) / rest_std
        else:
            activity_score = 0
        activity_scores.append(activity_score)
        
        # Calculate entropies
        pre_mse, valid_scales = compute_multiscale_entropy(pre_event[neuron_id, :], scales=5)
        post_mse, _ = compute_multiscale_entropy(post_event[neuron_id, :], scales=5)
        
        entropy_scores['pre_event']['individual_mse'][neuron_id] = pre_mse
        entropy_scores['pre_event']['valid_scales'] = valid_scales
        entropy_scores['post_event']['individual_mse'][neuron_id] = post_mse
        entropy_scores['post_event']['valid_scales'] = valid_scales
        
        # Calculate entropy changes
        event_mse, _ = compute_multiscale_entropy(event_signal, scales=5)
        rest_mse, _ = compute_multiscale_entropy(rest_signal, scales=5)
        
        event_mean = np.nanmean(event_mse)
        rest_mean = np.nanmean(rest_mse)
        
        if not np.isnan(rest_mean) and rest_mean != 0:
            entropy_change = (event_mean - rest_mean) / rest_mean
        else:
            entropy_change = 0
        entropy_changes.append(entropy_change)
    
    # Calculate z-scores
    entropy_changes = np.array(entropy_changes)
    activity_scores = np.array(activity_scores)
    
    entropy_z = (entropy_changes - np.nanmean(entropy_changes)) / np.nanstd(entropy_changes)
    activity_z = (activity_scores - np.nanmean(activity_scores)) / np.nanstd(activity_scores)
    
    # Store results
    for neuron_id in range(dff_signal.shape[0]):
        entropy_scores['neuron_scores'][neuron_id] = {
            'event_rest_z': entropy_z[neuron_id],
            'activity_z': activity_z[neuron_id],
            'entropy_change': entropy_changes[neuron_id],
            'activity_change': activity_scores[neuron_id],
            'is_event_related': (entropy_z[neuron_id] > 0.5 and 
                               activity_z[neuron_id] > 0.2 and
                               entropy_changes[neuron_id] > 0 and
                               activity_scores[neuron_id] > 0)
        }
    
    return entropy_scores

def plot_event_entropy_analysis(entropy_results, session, output_dir, dff_signal, victim_lib_timecode, fps):
    if entropy_results is None:
        print("No entropy results to plot")
        return

    try:
        # Create figure and subplots
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(2, 2, height_ratios=[1, 1])

        # Original entropy plots
        ax1 = plt.subplot(gs[0, 0])
        ax2 = plt.subplot(gs[0, 1])
        ax3 = plt.subplot(gs[1, :])

        # Plot individual neuron changes
        pre_means = [np.mean(mse) for mse in entropy_results['pre_event']['individual_mse'].values()]
        post_means = [np.mean(mse) for mse in entropy_results['post_event']['individual_mse'].values()]

        ax1.scatter(pre_means, post_means, alpha=0.5)
        ax1.plot([0, max(pre_means)], [0, max(pre_means)], 'k--')
        ax1.set_xlabel('Pre-event Mean Entropy')
        ax1.set_ylabel('Post-event Mean Entropy')
        ax1.set_title('Individual Neuron Entropy Changes')

        # Plot mean entropy traces
        all_scales = np.arange(15)
        pre_mse = np.zeros(len(all_scales))
        post_mse = np.zeros(len(all_scales))
        counts = np.zeros(len(all_scales))

        for pre_vals in entropy_results['pre_event']['individual_mse'].values():
            pre_vals = np.array(pre_vals)
            if len(pre_vals) > 0:
                pre_mse[:len(pre_vals)] += pre_vals
                counts[:len(pre_vals)] += 1

        for post_vals in entropy_results['post_event']['individual_mse'].values():
            post_vals = np.array(post_vals)
            if len(post_vals) > 0:
                post_mse[:len(post_vals)] += post_vals

        # Calculate means
        valid_mask = counts > 0
        valid_indices = np.where(valid_mask)[0]
        pre_mse[valid_mask] /= counts[valid_mask]
        post_mse[valid_mask] /= counts[valid_mask]

        ax2.plot(valid_indices, pre_mse[valid_mask], label='Pre-event', color='blue')
        ax2.plot(valid_indices, post_mse[valid_mask], label='Post-event', color='red')
        ax2.set_xlabel('Scale Factor')
        ax2.set_ylabel('Mean Sample Entropy')
        ax2.set_title('Population Entropy Changes')
        ax2.legend()

        # Find and plot event cells
        event_cells = [neuron_id for neuron_id, scores in 
                    entropy_results['neuron_scores'].items() 
                    if scores['is_event_related']]
        print(f"Number of event cells found: {len(event_cells)}")
        print(f"Event cells: {event_cells}")

        if len(event_cells) > 0:
            top_cells = sorted(event_cells, 
                             key=lambda x: entropy_results['neuron_scores'][x]['entropy_change'],
                             reverse=True)[:5]

            time_vector = np.arange(dff_signal.shape[1]) / fps
            event_time = victim_lib_timecode
            offset = 10

            for i, cell_id in enumerate(top_cells):
                signal = dff_signal[cell_id, :]
                baseline = i * offset
                smooth_signal = np.convolve(signal, np.ones(5)/5, mode='same')
                ax3.plot(time_vector, smooth_signal + baseline, label=f'Neuron {cell_id}')
                ax3.axhline(y=baseline, color='gray', linestyle=':', alpha=0.3)
                # Add label on the right
                ax3.text(time_vector[-1] + 10, baseline, f'Neuron {cell_id}', 
                        verticalalignment='center')

        # Add scale bar (top-left corner)
        ax3.plot([20, 20], [offset*len(top_cells), offset*len(top_cells)+2], 'k-', lw=2)
        ax3.text(25, offset*len(top_cells)+1, '2 ΔF/F', verticalalignment='center')

        # Add shading for the event window
        window_seconds = 20
        ax3.axvspan(event_time - window_seconds, 
                    event_time + window_seconds, 
                    color='r', alpha=0.1)
        
        # Make event line more prominent
        ax3.axvline(x=event_time, color='r', linestyle='--', label='Event', lw=2, alpha=0.8)

        # Customize axes
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('ΔF/F (offset)')
        ax3.set_yticks([])  # Remove y-ticks since we have baselines
        ax3.set_title('Event-Related Cell Signals')
        ax3.legend()

        # Save figure
        plt.tight_layout()
        filepath = os.path.join(output_dir, session.cohort, session.session, 
                              f'event_entropy_analysis_{session.animal_id}_{session.session}.png')
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Figure saved to {filepath}")
        
    except Exception as e:
        print(f"Error in plotting: {str(e)}")
        plt.close('all')  # Clean up any open figures

def compute_multiscale_entropy(data, scales=20, m=2, r=0.15):
    def coarse_grain(data, scale):
        reshaped = data[:len(data) - len(data) % scale]
        return np.mean(reshaped.reshape(-1, scale), axis=1)
    
    # Calculate maximum possible scale based on data length
    max_scale = min(scales, len(data) // (m + 1))
    mse = np.zeros(max_scale)
    valid_scales = []
    
    for i in range(max_scale):
        scale = i + 1
        try:
            coarse_data = coarse_grain(data, scale)
            if len(coarse_data) > m + 1:  # Ensure enough data points
                entropy_value = nk.entropy_sample(coarse_data, dimension=m, tolerance=r)[0]
                if not np.isinf(entropy_value) and not np.isnan(entropy_value):
                    mse[i] = entropy_value
                    valid_scales.append(i)
        except:
            continue
    
    if len(valid_scales) == 0:
        # Return at least one valid value to prevent errors
        return np.array([0.0]), [0]
    
    return mse[valid_scales], valid_scales

def compute_entropy_metrics(dff_signal, communities, session):
    def compute_multiscale_entropy(data, scales=20, m=2, r=0.15):
        """
        Calculate multiscale entropy for calcium imaging data
        """
        def coarse_grain(data, scale):
            reshaped = data[:len(data) - len(data) % scale]
            return np.mean(reshaped.reshape(-1, scale), axis=1)
        
        mse = np.zeros(scales)
        for i in range(scales):
            scale = i + 1
            coarse_data = coarse_grain(data, scale)
            try:
                entropy_value = nk.entropy_sample(coarse_data, dimension=m, tolerance=r)[0]
                mse[i] = entropy_value
            except (ValueError, IndexError):
                mse[i] = 0
                print(f"Warning: Entropy calculation failed for scale {scale}")
        
        return mse

    entropy_results = {
        'individual_mse': {},
        'community_mse': {},
        'community_sizes': {}
    }

    # Individual neuron entropy
    for neuron_id in range(dff_signal.shape[0]):
        entropy_results['individual_mse'][neuron_id] = compute_multiscale_entropy(dff_signal[neuron_id, :])

    # Community-level entropy
    for comm_id, neurons in communities.items():
        try:
            if len(neurons) >= 3:
                comm_signal = np.mean(dff_signal[neurons, :], axis=0)
                if not np.isnan(comm_signal).any():  # Check for NaN values
                    entropy_results['community_mse'][comm_id] = compute_multiscale_entropy(comm_signal)
                    entropy_results['community_sizes'][comm_id] = len(neurons)
        except Exception as e:
            print(f"Warning: Failed to compute entropy for community {comm_id}: {str(e)}")
            continue

    return entropy_results

def plot_entropy_analysis(entropy_results, session, output_dir):
    """Plot entropy analysis results"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot individual neuron MSE
    for neuron_id, mse in entropy_results['individual_mse'].items():
        ax1.plot(mse, alpha=0.1, color='gray')
    ax1.plot(np.mean([mse for mse in entropy_results['individual_mse'].values()], axis=0),
             color='black', linewidth=2, label='Mean')
    ax1.set_title('Individual Neuron MSE')
    ax1.set_xlabel('Scale Factor')
    ax1.set_ylabel('Sample Entropy')
    
    # Plot community MSE
    for comm_id, mse in entropy_results['community_mse'].items():
        size = entropy_results['community_sizes'][comm_id]
        ax2.plot(mse, alpha=0.5, linewidth=2, 
                 label=f'Community {comm_id} (n={size})')
    ax2.set_title('Community-level MSE')
    ax2.set_xlabel('Scale Factor')
    ax2.set_ylabel('Sample Entropy')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    filename = f'entropy_analysis_{session.animal_id}_{session.session}.png'
    filepath = os.path.join(output_dir, session.cohort, session.session, filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

def analyze_communities(G):
    """
    Performs Louvain community detection on the graph and returns community assignments.
    
    Parameters:
    -----------
    G : networkx.Graph
        The graph object representing functional connectivity
        
    Returns:
    --------
    communities : dict
        Dictionary mapping node IDs to their community assignments
    modularity : float
        Modularity score of the community structure
    n_communities : int
        Number of communities detected
    community_sizes : dict
        Dictionary containing sizes of each community
    conductances : list
        List of conductance values for each community
    """

    # Convert networkx graph to igraph
    edges = list(G.edges())

    # Check if the graph has any edges
    if len(edges) == 0:
        # If no edges, each node is its own community
        communities = {node: idx for idx, node in enumerate(G.nodes())}
        modularity = 0.0  # Modularity is 0 for a graph with no edges
        community_sizes = {idx: 1 for idx in range(len(G.nodes()))}
        n_communities = len(G.nodes())
        conductances = [1.0] * len(G.nodes())
        print(f"Warning: Graph has no edges. Each node assigned to its own community.")
        return communities, modularity, n_communities, community_sizes, conductances

    # Create igraph object with correct number of vertices
    weights = [G.edges[edge]['weight'] for edge in edges]
    g_ig = ig.Graph(n=G.number_of_nodes(), edges=edges, directed=False)
    g_ig.es['weight'] = weights
  
    # Perform Leiden community detection
    partition = leidenalg.find_partition(
        g_ig,
        leidenalg.RBConfigurationVertexPartition,
        weights='weight'
    )

    # Convert results back to dictionary format
    communities = {v: c for v, c in enumerate(partition.membership)}

    # Convert communities dict to list of lists format required by NodeClustering
    community_to_nodes = {}
    for node, comm_id in communities.items():
        if comm_id not in community_to_nodes:
            community_to_nodes[comm_id] = []
        community_to_nodes[comm_id].append(node)
    
    community_list = list(community_to_nodes.values())
    
    # Create NodeClustering object
    node_clustering = NodeClustering(
        communities=community_list,
        graph=G,
        method_name="leiden",
        method_parameters={},
        overlap=False
    )

    # Calculate modularity
    modularity = partition.quality()

    # Calculate conductance for each community
    conductances = evaluation.conductance(G, node_clustering, summary=False)
    
    # Count communities and their sizes
    community_sizes = {}
    for node, community_id in communities.items():
        if community_id not in community_sizes:
            community_sizes[community_id] = 0
        community_sizes[community_id] += 1
    
    n_communities = len(community_sizes)
    
    return communities, modularity, n_communities, community_sizes, conductances

def save_community_data(session, communities, G, output_dir, liberation_cells_ids=None):
    """
    Saves community data to CSV files with additional liberation cell metrics.
    """
    logging.info(f"Starting save_community_data for session {session.session}")
    logging.debug(f"Input communities format: {type(communities)}")
    logging.debug(f"Input communities content: {communities}")

    # Create directory structure
    session_dir = os.path.join(output_dir, session.cohort, session.session)
    os.makedirs(session_dir, exist_ok=True)
    
    # Filter communities with at least 3 neurons
    large_communities = {comm_id: nodes for comm_id, nodes 
                        in communities.items() 
                        if len(nodes) >= 2}
    logging.info(f"Number of large communities: {len(large_communities)}")
    
    # Create DataFrame with community information
    community_data = []
    
    # Process each community
    for comm_id, nodes in communities.items():
        # Calculate community metrics
        comm_subgraph = G.subgraph(nodes)
        
        # Convert communities dict to list format for conductance calculation
        community_list = []
        for c_id, c_nodes in communities.items():
            if isinstance(c_nodes, (list, np.ndarray)):
                community_list.append(list(c_nodes))
            else:
                community_list.append([c_nodes])
        
        # Create NodeClustering object
        node_clustering = NodeClustering(
            communities=community_list,
            graph=G,
            method_name="leiden",
            method_parameters={},
            overlap=False
        )
        
        # Calculate conductance
        conductances = evaluation.conductance(G, node_clustering, summary=False)
        conductance_map = {
            tuple(sorted(comm)): cond 
            for comm, cond in zip(community_list, conductances)
        }
        
        # Get conductance for this community
        nodes_tuple = tuple(sorted(nodes) if isinstance(nodes, (list, np.ndarray)) else [nodes])
        conductance = conductance_map.get(nodes_tuple, 0.0)
        
        # Check if community has any liberation cells
        has_liberation_cells = False
        n_liberation_cells = 0
        if liberation_cells_ids is not None:
            liberation_cells_in_comm = [node for node in nodes if node in liberation_cells_ids]
            has_liberation_cells = len(liberation_cells_in_comm) > 0
            n_liberation_cells = len(liberation_cells_in_comm)
        
        # Process each node in the community
        for node in nodes:
            community_data.append({
                'Mouse': session.animal_id,
                'Session': session.session,
                'Cohort': session.cohort,
                'DayOfHelping': session.day_of_helping,
                'Community_ID': comm_id,
                'Neuron_ID': node,
                'Degree': G.degree(node),
                'Community_Size': len(nodes),
                'Community_Conductance': conductance,
                'Has_Liberation_Cells': has_liberation_cells,
                'N_Liberation_Cells': n_liberation_cells,
                'Liberation_Cells_Ratio': n_liberation_cells / len(nodes),
                'Is_Liberation_Cell': node in liberation_cells_ids if liberation_cells_ids is not None else False
            })
    
    if community_data:
        community_df = pd.DataFrame(community_data)
        filename = f"community_data_{session.animal_id}_{session.session}.csv"
        filepath = os.path.join(session_dir, filename)
        try:
            community_df.to_csv(filepath, index=False)
            logging.info(f"Saved community data to {filepath}")
            return community_df
        except Exception as e:
            logging.error(f"Error saving community data: {str(e)}")
            return None
    else:
        logging.warning("No community data to save")
        return None

def setup_plot(fig_size: Tuple[float, float] = (12, 9),
               grid_spec: Tuple[int, int] = (3, 6)) -> Tuple[plt.Figure, Dict[str, plt.Axes]]:
    """Set up the plot with a specific style and grid layout."""
    plt.style.use('seaborn-v0_8-talk')
    fig = plt.figure(figsize=fig_size)
    gs = GridSpec(*grid_spec, figure=fig)
    
    # Set larger font sizes
    plt.rcParams.update({
        'font.size': 14,          # General font size
        'axes.titlesize': 18,     # Font size for axes titles
        'axes.labelsize': 16,     # Font size for axes labels
        'xtick.labelsize': 14,    # Font size for x-axis tick labels
        'ytick.labelsize': 14,    # Font size for y-axis tick labels
        'legend.fontsize': 14,    # Font size for legend
        'figure.titlesize': 20    # Font size for figure title
    })

    axes = {
        'cross-correlation-matrix': fig.add_subplot(gs[0, 0]),
        'distance-matrix': fig.add_subplot(gs[1, 0]),
        'correlation-vs-distance': fig.add_subplot(gs[2, 0]),
        'image-example': fig.add_subplot(gs[0:2, 1:3]),
        'traces-example': fig.add_subplot(gs[0:2, 3:5]),
        'clustering-coefficient-over-days': fig.add_subplot(gs[2, 1:3]),
        'clustering-coefficient-over-days-of-helping': fig.add_subplot(gs[2, 3:5]),
        'lib-cells-percentage': fig.add_subplot(gs[0, 5]),
        'pcc-distribution': fig.add_subplot(gs[1, 5]),
        'pcc-cumulative-distribution': fig.add_subplot(gs[2, 5])
    }

    return fig, axes

def parse_timecode(timecode):
    if pd.isna(timecode):
        return None
    try:
        return int(timecode)
    except ValueError:
        return None

def load_config(config_file='functional_connectivity_config.yaml'):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def plot_session_example(axes, session, pnr, accepted_dff_signal, distances, correlation_matrix, correlated_pairs_new, position_dict, G, colors, labels, liberation_associated_cells_ids):
    # axes['image-example'].imshow(pnr, cmap='grey')
    nx.draw_networkx_labels(G, pos=position_dict, ax=axes['image-example'], labels=labels, font_size=8, font_color='white')

    # colors = ['white' if c == 'black' else c for c in colors]
    nx.draw(G, pos=position_dict, node_size=50, node_color=colors, ax=axes['image-example'], edge_color='black', node_shape='*')
    axes['image-example'].set_title('Graph example, animal: ' + str(session.animal_id) + ', session: ' + session.session)

    # Plot the cross-correlation matrix, sort the values before plotting
    plot_cross_correlation_matrix(axes['cross-correlation-matrix'], correlation_matrix, 'Cross-correlation matrix, animal: ' + str(session.animal_id) + ', session: ' + session.session, cbar=False)

    # Plot the distance matrix
    plot_distance_matrix(axes['distance-matrix'], distances, 'Distance matrix, animal: ' + str(session.animal_id) + ', session: ' + session.session, cbar=False)

    # Plot the traces of highly correlated components
    plot_correlated_traces(axes['traces-example'], accepted_dff_signal, correlated_pairs_new, distances=distances, correlation_matrix=correlation_matrix,liberation_cell_ids=liberation_associated_cells_ids, fps=session.calcium_fps)
    axes['traces-example'].set_title('Traces of highly correlated components')
    axes['traces-example'].axvline(session.victim_lib_timecode * session.calcium_fps, color='k', linestyle='dashed', linewidth=2, label='Victim crossing')

def sample_values(distances_values: List[float], correlation_values: List[float], n_samples: int = 50) -> Tuple[List[float], List[float]]:
    """
    Sample n_samples random elements from distances_values and correlation_values.

    Args:
        distances_values (List[float]): List of distances.
        correlation_values (List[float]): List of correlation values.
        n_samples (int, optional): Number of samples to take. Defaults to 50.

    Returns:
        Tuple[List[float], List[float]]: Sampled distances and correlations.
    """
    sample_idx = np.random.RandomState(0).permutation(len(distances_values))[:n_samples]
    return distances_values[sample_idx], correlation_values[sample_idx]

def compute_correlations(positions, signal, threshold, 
                         bin_size_seconds, sampling_rate,
                         distance_metric='euclidean', correlation_method='pearson'):
    """
    Compute correlations and related metrics from the model and DFF data.

    Parameters:
        positions (np.array): Positions of components (in pixels) as a 2D (x, y of the centroid) array where each row corresponds to a component.
        signal (np.array): Z-scored delta F/F values as a 2D array where each row corresponds to a component and each column corresponds to a time step. First column corresponds to the original indices of the components.
        threshold (float): Threshold for the adjacency matrix.
        bin_size_seconds (float): Size of each time bin in seconds. Used to calculate cross-correlation in smaller intervals to get rid of a session lenght effect.
        sampling_rate (float): Sampling rate of the data in Hz.
        distance_metric (str): Metric to use for distance calculation. Default is 'euclidean'.
        correlation_method (str): Method to use for correlation calculation. Default is 'pearson'.

    Returns:
        tuple: Contains the following elements:
            - distances (np.array): Distance matrix between components (in pixels) as a 2D array where each entry (i, j) is the distance between components i and j.
            - cross_correlation_bin (np.array): Cross-correlation matrix as a 2D array where each entry (i, j) is the Pearson correlation between components i and j.
            - adjacency (np.array): Adjacency matrix as a 2D array where each entry (i, j) is 1 if components i and j are correlated and 0 otherwise.
            - correlated_pairs (tuple): Indices of highly correlated components as a tuple of two arrays, each containing the indices of the first and second components in the pair. Returns indexes as in original CNMFe model (before filtering only accepted components)
    """

    # Remove the original indices from the signal
    signal = signal[:, 1:]

    # Input validation
    if not isinstance(signal, np.ndarray):
        raise TypeError("signal must be numpy array")
    if signal is None or np.isnan(signal).any():
        return None, None, None, None

    # Calculate pairwise distances between component centers
    try:
        distances = cdist(positions, positions, metric=distance_metric)
    except Exception as e:
        print(f"Error calculating distances: {e}")
        return None, None, None, None

    # Calculate the number of time steps in each bin
    bin_size_steps = int(bin_size_seconds * sampling_rate)
    
    # Calculate the total number of bins
    n_bins = signal.shape[1] // bin_size_steps

    if n_bins == 0:
        print(f"Warning: Specified bin size ({bin_size_steps} steps) is larger than the total signal length ({signal.shape[1]} steps).")
        print("Computing cross-correlation using the entire signal.")
        if correlation_method == 'pearson':
            cross_correlation_bin = np.corrcoef(signal)
        else:
            raise ValueError("Unsupported correlation method")
    else:
        cross_correlation_bin = np.zeros((signal.shape[0], signal.shape[0]))
        for i in range(n_bins):
            bin_start = i * bin_size_steps
            bin_end = (i+1) * bin_size_steps
            signal_bin = signal[:, bin_start:bin_end]

            if correlation_method == 'pearson':
                cross_correlation_bin += np.corrcoef(signal_bin)
            else:
                raise ValueError("Unsupported correlation method")

        # Average the cross-correlation matrices across all time-window bins
        cross_correlation_bin /= n_bins

    # Create the adjacency matrix by thresholding the cross-correlation matrix
    adjacency_matrix = (cross_correlation_bin > threshold)
    # Set the diagonal elements to 0 to avoid counting the correlation of a component with itself
    adjacency_matrix = np.triu(adjacency_matrix, k=1) # k==1 to avoid the main diagonal

    # Find highly correlated components
    correlated_pairs = np.where(adjacency_matrix)
    # Map the correlated pairs back to the original index; DON'T MAKE, ACCEPTED COMPONENTS ARE PRE-SELECTED
    correlated_pairs = (correlated_pairs[0], correlated_pairs[1])

    return distances, cross_correlation_bin, adjacency_matrix, correlated_pairs

def build_graph(session, neuron_positions, adjacency_matrix, liberation_associated_cells_ids=None, random_event_associated_cells_ids=None, suffix=None, communities=None, correlation_matrix=None):    
    """ Create a weighted graph from the adjacency matrix and correlation matrix

    Parameters:
        session (Session): Session object containing information about the session.
        neuron_positions (list of tuples): List of (x, y) positions of neurons.
        adjacency_matrix (np.array): Adjacency matrix representing the graph.
        liberation_cells_ids (list): List of IDs of liberation cells, mapped to (new) IDs.
        random_event_cells_ids (list): List of IDs of random event cells, mapped to (new) IDs.
        suffix (str): Optional suffix to add to the filename.
        communities (xxx): Optional List of communities
        correlation_matrix (np.array): Correlation matrix

    Returns:
        tuple: Contains the following elements:
            - position_dict (dict): Mapping between node IDs and their positions.
            - G (nx.Graph): Graph object representing the graph.
            - colors (list): List of colors (white, red, blue or violet) for each node.
            - labels (dict): Mapping between node IDs and their labels.
    """
    G = nx.Graph()
    
    # Add nodes
    for i in range(len(adjacency_matrix)):
        G.add_node(i)
    
    # Add edges with weights from correlation matrix
    for i in range(len(adjacency_matrix)):
        for j in range(i+1, len(adjacency_matrix)):
            if adjacency_matrix[i,j]:
                    if correlation_matrix is not None:
                        G.add_edge(i, j, weight=correlation_matrix[i,j])
                    else:
                        # Add edge with default weight if correlation matrix is not provided
                        G.add_edge(i, j, weight=1.0)
    
    # Create a mapping from node IDs to their positions
    position_dict = {i: (y, x) for i, (x, y) in enumerate(neuron_positions)}
    
    # Create a list of colors (white or red) for each node
    colors = ['black'] * len(neuron_positions)
    alphas = [1] * len(neuron_positions)
    labels = {}

    degrees = G.degree()

    # If degree is 0, set alpha to 0.4, else set alpha to 1
    for node, degree in degrees:
        if degree == 0:
            alphas[node] = 0.4
        else:
            continue
    
    # If a cell is both liberation and random event, make it violet
    if liberation_associated_cells_ids is not None and random_event_associated_cells_ids is not None:
        for i in range(len(neuron_positions)):
            if i in random_event_associated_cells_ids and i in liberation_associated_cells_ids:
                colors[i] = 'violet'

    # If liberation cells are present, highlight them
    if liberation_associated_cells_ids is not None:
        colors = ['tab:red' if i in liberation_associated_cells_ids else c for i, c in enumerate(colors)]
        labels.update({i: f'{str(i)}' for i in range(len(neuron_positions)) if i in liberation_associated_cells_ids})

    # If random event cells are present, highlight them
    if random_event_associated_cells_ids is not None:
        colors = ['tab:blue' if i in random_event_associated_cells_ids and not i in liberation_associated_cells_ids else c for i, c in enumerate(colors)]
        labels.update({i: f'{str(i)}' for i in range(len(neuron_positions)) if i in random_event_associated_cells_ids})
            
    # If communities are provided, use them for coloring
    if communities is not None:
        try:
            # Generate a color map for communities
            unique_communities = sorted(set(communities.values()))
            community_colors = plt.cm.tab20(np.linspace(0, 1, len(unique_communities)))
            
            # Create a mapping of community ID to color
            community_color_map = dict(zip(unique_communities, community_colors))
            
            # Create two figures: one for communities, one for cell types
            # Community visualization
            # Create community visualization
            fig_comm = plt.figure(figsize=(8, 6))
            ax_comm = fig_comm.add_subplot(111)
            community_node_colors = []
            for node in G.nodes():
                if node in communities:
                    community_node_colors.append(community_color_map[communities[node]])
                else:
                    # Assign a default color for nodes without community
                    community_node_colors.append([0.7, 0.7, 0.7, 1.0])  # Gray color
                
            nx.draw(G, pos=position_dict, node_size=100,
                node_color=community_node_colors,
                edge_color='lightgray', ax=ax_comm)
        
            # Save community visualization
            output_dir = os.path.join('results', 'figures', 'functional_connectivity', 'communities_graphs', 
                                    session.cohort, session.session)
            os.makedirs(output_dir, exist_ok=True)
            now = datetime.datetime.now()
            filename_comm = os.path.join(output_dir, 
                f'COMMUNITIES_{now.strftime("%Y-%m-%d_%H-%M-%S")}_{str(session.animal_id)}{suffix}.png')
            fig_comm.savefig(filename_comm, dpi=300)
            plt.close(fig_comm)
        except Exception as e:
            print(f"Warning: Error in community visualization: {str(e)}")
   
    # Plot the graph
    __, ax = plt.subplots(1, 1, figsize=(8, 6))
    nx.draw_networkx_labels(G, pos=position_dict, ax=ax, labels=labels, font_size=6, font_color='k')
    nx.draw(G, pos=position_dict, node_size=100, node_color=colors, alpha=alphas, edge_color='lightgray', ax=ax)
    
    # Save the figure
    now = datetime.datetime.now()
    output_dir = os.path.join('results', 'figures', 'functional_connectivity', 'graphs', session.cohort, session.session)
    os.makedirs(output_dir, exist_ok=True)
    filename_png = os.path.join(output_dir, f'NETWORK_{now.strftime("%Y-%m-%d_%H-%M-%S")}_{os.path.basename(__file__)[:-3]}_{str(session.animal_id)}{suffix}.png')
    filename_eps = os.path.join(output_dir, f'NETWORK_{now.strftime("%Y-%m-%d_%H-%M-%S")}_{os.path.basename(__file__)[:-3]}_{str(session.animal_id)}{suffix}.eps')
    filename_svg = os.path.join(output_dir, f'NETWORK_{now.strftime("%Y-%m-%d_%H-%M-%S")}_{os.path.basename(__file__)[:-3]}_{str(session.animal_id)}{suffix}.svg')
    plt.savefig(filename_png, dpi=300)
    plt.savefig(filename_eps, format='eps')
    plt.savefig(filename_svg, format='svg')
    plt.close()
    
    return position_dict, G, colors, labels

def get_values(matrix: np.ndarray) -> np.ndarray:
    """
    Get the values from the upper triangle of a matrix.

    Parameters
    ----------
    matrix : np.ndarray
        The input matrix.

    Returns
    -------
    values_list : np.ndarray
        The values from the upper triangle of the matrix.
    """
    # Use the triu function from numpy to get the upper triangle of the matrix.
    upper_triangle_matrix_indices = np.triu_indices(matrix.shape[0], k=1) # Should be below diagonal and not include same neuron data
    # Take only the values from the upper triangle
    value_list = matrix[upper_triangle_matrix_indices]
    return value_list

def calculate_psth(dff_signal, border, fps=30):

    # Remove the original indices from the signal
    dff_signal = dff_signal[:, 1:]

    # Input validation
    if not isinstance(dff_signal, np.ndarray):
        raise TypeError("signal must be numpy array")
    if dff_signal is None or np.isnan(dff_signal).any():
        return None, None, None, None
    
    """Calculates the PSTH for a given event timecode."""
    pre_event_duration = 15  # seconds
    post_event_duration = 15  # seconds

    # Calculate frame indices based on border
    start_pre = border
    end_pre = border + pre_event_duration * fps
    
    end_post = border + post_event_duration * fps + pre_event_duration * fps

    # Ensure indices are within bounds
    if start_pre < 0 or end_post > dff_signal.shape[1]:
        print("Warning: Index out of bounds.")
        return np.zeros(0), 0  # Return empty array and length 0

    # Extract signals
    pre_event_signal = dff_signal[:, start_pre:end_pre]
    post_event_signal = dff_signal[:, end_pre:end_post]

    # Normalize by baseline mean (using a baseline period if needed)
    baseline_mean = np.mean(dff_signal[:, start_pre - (10 * fps):start_pre], axis=1, keepdims=True)

    # Average across neurons for each time bin
    avg_pre_event = np.mean((pre_event_signal - baseline_mean) / baseline_mean, axis=0)
    avg_post_event = np.mean((post_event_signal - baseline_mean) / baseline_mean, axis=0)

    # Combine pre and post event averages
    psth_combined = np.concatenate((avg_pre_event, avg_post_event))

    return psth_combined, len(psth_combined)  # Return combined PSTH and its length