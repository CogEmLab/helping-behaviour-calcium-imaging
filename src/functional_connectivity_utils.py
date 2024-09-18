import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Dict, Tuple, List
from src.plotting import plot_correlated_traces, plot_cross_correlation_matrix, plot_distance_matrix
import yaml
from scipy.spatial.distance import cdist
import networkx as nx

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

def plot_session_example(axes, session, pnr, accepted_dff_signal, distances, correlation_matrix, correlated_pairs_new, position_dict, G, colors, labels):
    axes['image-example'].imshow(pnr, cmap='grey')
    nx.draw_networkx_labels(G, pos=position_dict, ax=axes['image-example'], labels=labels, font_size=8, font_color='white')

        # colors = ['white' if c == 'black' else c for c in colors]
    nx.draw(G, pos=position_dict, node_size=50, node_color=colors, ax=axes['image-example'], edge_color='black', node_shape='*')
    axes['image-example'].set_title('Graph example, animal: ' + str(session.animal_id) + ', session: ' + session.session)

        # Plot the cross-correlation matrix, sort the values before plotting
    plot_cross_correlation_matrix(axes['cross-correlation-matrix'], correlation_matrix, 'Cross-correlation matrix, animal: ' + str(session.animal_id) + ', session: ' + session.session, cbar=False)

        # Plot the distance matrix
    plot_distance_matrix(axes['distance-matrix'], distances, 'Distance matrix, animal: ' + str(session.animal_id) + ', session: ' + session.session, cbar=False)

        # Plot the traces of highly correlated components
    plot_correlated_traces(axes['traces-example'], accepted_dff_signal, correlated_pairs_new, distances=distances)
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

def compute_correlations(positions, DFF_z_scored, threshold, 
                         distance_metric='euclidean', correlation_method='pearson'):
    """
    Compute correlations and related metrics from the model and DFF data.

    Parameters:
        estimates_spatial (np.array): Spatial estimates from the model.
        DFF_z_scored (np.array): Z-scored delta F/F values.
        img_dims (tuple): Dimensions of the image.
        threshold (float): Threshold for the adjacency matrix.
        distance_metric (str): Metric to use for distance calculation. Default is 'euclidean'.
        correlation_method (str): Method to use for correlation calculation. Default is 'pearson'.

    Returns:
        tuple: Contains the following elements:
            - distances (np.array): Distance matrix between components (in pixels).
            - cross_correlation (np.array): Cross-correlation matrix.
            - correlated_pairs (tuple): Indices of highly correlated components.
    """

    original_idx = DFF_z_scored[:, 0].astype(int)

    DFF_z_scored = DFF_z_scored[:, 1:]

    # Input validation
    if not isinstance(DFF_z_scored, np.ndarray):
        raise TypeError("DFF_z_scored must be numpy array")
    if DFF_z_scored is None or np.isnan(DFF_z_scored).any():
        return None, None, None
    
    # Calculate pairwise distances between component centers
    try:
        distances = cdist(positions, positions, metric=distance_metric)
    except Exception as e:
        print(f"Error calculating distances: {e}")
        return None, None, None

    # Calculate cross-correlation and adjacency matrix
    try:
        if correlation_method == 'pearson':
            cross_correlation = np.corrcoef(DFF_z_scored)
        else:
            raise ValueError("Unsupported correlation method")
    except Exception as e:
        print(f"Error calculating correlations: {e}")
        return None, None, None

    adjacency_matrix = (cross_correlation > threshold)
    adjacency_matrix = np.tril(adjacency_matrix, k=-1)

    # Identify highly correlated components
    correlated_pairs = np.where(adjacency_matrix)
    correlated_pairs = (original_idx[correlated_pairs[0]], original_idx[correlated_pairs[1]])

    return distances, cross_correlation, adjacency_matrix, correlated_pairs

def build_graph(session, neuron_positions, adjacency_matrix, liberation_cell_ids_mapped, suffix=''):
    """Build and plot a graph from the adjacency matrix.

    Parameters:
        session (Session): Session object containing information about the session.
        neuron_positions (list of tuples): List of (x, y) positions of neurons.
        adjacency_matrix (np.array): Adjacency matrix representing the graph.
        liberation_cells_ids (list): List of IDs of liberation cells, mapped to (new) IDs.
        suffix (str): Optional suffix to add to the filename.

    Returns:
        tuple: Contains the following elements:
            - position_dict (dict): Mapping between node IDs and their positions.
            - G (nx.Graph): Graph object representing the graph.
            - colors (list): List of colors (white or red) for each node.
            - labels (dict): Mapping between node IDs and their labels.
    """
    # Create a graph from the adjacency matrix
    G = nx.from_numpy_array(adjacency_matrix)
    
    # Create a mapping from node IDs to their positions
    position_dict = {i: (y, x) for i, (x, y) in enumerate(neuron_positions)}
    
    # Create a list of colors (white or red) for each node
    colors = ['black'] * len(neuron_positions)
    
    # Create a mapping between node IDs and their labels
    labels = {}
    
    # If liberation cells are present, highlight them
    if liberation_cell_ids_mapped is not None:
        colors = ['red' if i in liberation_cell_ids_mapped else 'black' for i in range(len(neuron_positions))]
        labels.update({i: str(i) for i in liberation_cell_ids_mapped})

        # Compare number of 'red' color to the number of liberation cells in session.number_of_liberation_cells;
        # if not matching print out
        if len(liberation_cell_ids_mapped) != session.number_of_liberation_cells:
            print(f'Number of cells in {session.session} does not match number of cells in Liberation Data')
    
    # Plot the graph
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    nx.draw(G, pos=position_dict, node_size=100, node_color=colors, ax=ax)
    
    # Save the figure
    output_dir = os.path.join('figures', 'graphs', session.cohort, session.session)
    os.makedirs(output_dir, exist_ok=True)
    filename_png = os.path.join(output_dir, f'graph_{str(session.animal_id)}{suffix}.png')
    filename_eps = os.path.join(output_dir, f'graph_{str(session.animal_id)}{suffix}.eps')
    plt.savefig(filename_png, dpi=300)
    plt.savefig(filename_eps, format='eps')
    plt.close()
    
    return position_dict, G, colors, labels

def get_values(matrix: np.ndarray) -> np.ndarray:
    """
    Get the values from the lower triangle of a matrix.

    Parameters
    ----------
    matrix : np.ndarray
        The input matrix.

    Returns
    -------
    values_list : np.ndarray
        The values from the lower triangle of the matrix.
    """
    # Use the tril function from numpy to get the lower triangle of the matrix.
    lower_triangle_matrix = np.tril(matrix, k=-1)
    
    # Use the flatten function to flatten the lower triangle matrix into a 1D array.
    values_list = lower_triangle_matrix.flatten()
    
    return values_list