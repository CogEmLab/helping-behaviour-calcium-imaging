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

def parse_victim_lib_timecode(timecode):
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

def build_graph(session, neuron_positions, adjacency_matrix, liberation_associated_cells_ids, random_event_associated_cells_ids, suffix=None):    
    """Build and plot a graph from the adjacency matrix.

    Parameters:
        session (Session): Session object containing information about the session.
        neuron_positions (list of tuples): List of (x, y) positions of neurons.
        adjacency_matrix (np.array): Adjacency matrix representing the graph.
        liberation_cells_ids (list): List of IDs of liberation cells, mapped to (new) IDs.
        random_event_cells_ids (list): List of IDs of random event cells, mapped to (new) IDs.
        suffix (str): Optional suffix to add to the filename.

    Returns:
        tuple: Contains the following elements:
            - position_dict (dict): Mapping between node IDs and their positions.
            - G (nx.Graph): Graph object representing the graph.
            - colors (list): List of colors (white, red, blue or violet) for each node.
            - labels (dict): Mapping between node IDs and their labels.
    """
    # Create a graph from the adjacency matrix
    G = nx.from_numpy_array(adjacency_matrix)
    
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
    
    # If liberation cells are present, highlight them
    if liberation_associated_cells_ids is not None:
        colors = ['red' if i in liberation_associated_cells_ids else 'black' for i in range(len(neuron_positions))]

    # If random event cells are present, highlight them
    if random_event_associated_cells_ids is not None:
        colors = ['blue' if i in random_event_associated_cells_ids else c for i, c in enumerate(colors)]

    # If a cell is both liberation and random event, make it violet
    if liberation_associated_cells_ids is not None and random_event_associated_cells_ids is not None:
        for i in range(len(neuron_positions)):
            if i in random_event_associated_cells_ids and i in liberation_associated_cells_ids:
                colors[i] = 'violet'
    
    # Save liberation and random event cells to the labels 
    if liberation_associated_cells_ids is not None:
        labels.update({i: f'{str(i)} Lib' for i in range(len(neuron_positions)) if i in liberation_associated_cells_ids})
    if random_event_associated_cells_ids is not None:
        labels.update({i: f'{str(i)} Rand' for i in range(len(neuron_positions)) if i in random_event_associated_cells_ids})

    # Plot the graph
    __, ax = plt.subplots(1, 1, figsize=(8, 6))
    nx.draw(G, pos=position_dict, node_size=100, node_color=colors, alpha=alphas, edge_color='lightgray', ax=ax)

    # Add legend for the colors; if black==Rest, if Red ==Lib, if Blue ==Rand, if Violet == Both
    legend_elements = [Patch(facecolor='black', edgecolor='black', label='Rest'),
                       Patch(facecolor='red', edgecolor='black', label='Liberation-associated'),
                       Patch(facecolor='blue', edgecolor='black', label='Random-Event-associated'),
                       Patch(facecolor='violet', edgecolor='black', label='Both')]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Save the figure
    now = datetime.datetime.now()
    output_dir = os.path.join('figures', 'functional_connectivity', 'graphs', session.cohort, session.session)
    os.makedirs(output_dir, exist_ok=True)
    filename_png = os.path.join(output_dir, f'NETWORK_{now.strftime("%Y-%m-%d_%H-%M-%S")}_{os.path.basename(__file__)[:-3]}_{str(session.animal_id)}{suffix}.png')
    filename_eps = os.path.join(output_dir, f'NETWORK_{now.strftime("%Y-%m-%d_%H-%M-%S")}_{os.path.basename(__file__)[:-3]}_{str(session.animal_id)}{suffix}.eps')
    plt.savefig(filename_png, dpi=300)
    plt.savefig(filename_eps, format='eps')
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