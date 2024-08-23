import matplotlib
matplotlib.use('Agg')  # Set the backend to 'Agg' before importing pyplot

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from collections import defaultdict
import networkx as nx
from caiman.source_extraction.cnmf.cnmf import load_CNMF
from src.utils import get_behaviour_timestamps
from tqdm import tqdm
from matplotlib.gridspec import GridSpec
from typing import Dict, Tuple
import seaborn as sns
from scipy.optimize import curve_fit
from tifffile import imread
from src.plotting import plot_boxplots

# Constants
SESSION_FILE = 'data/session_file_MiniscopeCW_all_Moises_latest_liberation_cells.csv'
METADATA_FILE = 'processed-data/metadata_cross_comp_updated.csv'
ANNOTATIONS_DIR = 'annotations'
IMG_DIMS = (400, 640)
THRESHOLD = 0.7
FIG_SIZE = (25, 15)
GRID_SPEC = (3, 6)

def setup_plot() -> Tuple[plt.Figure, Dict[str, plt.Axes]]:
    """Set up the plot with a specific style and grid layout."""
    plt.style.use('seaborn-v0_8-talk')
    fig = plt.figure(figsize=FIG_SIZE)
    gs = GridSpec(*GRID_SPEC, figure=fig)
    
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

def load_data(model_path, annotations_path):
    """Loads npy data and annotations based on the session row information."""
    npy_path = model_path.replace('.hdf5', '_Dff_z_scored.npy').replace('cnmfe_model', 'npy_data')
    try:
        npy_data = np.load(npy_path)
        model = load_CNMF(model_path)
        annotations = pd.read_csv(annotations_path.replace('_regrouped', '_manual_annotations'))
        return model, npy_data, annotations
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None
    
def plot_traces(ax, DFF_z_scored, distances, highly_correlated_components, behaviour_timestamps):
    """
    Plot the traces of highly correlated components on a provided axis.

    Parameters:
        ax (matplotlib.axes.Axes): The axis to plot on.
        DFF_z_scored (np.array): Z-scored delta F/F values.
        distances (np.array): Distance matrix between components.
        highly_correlated_components (tuple): Indices of highly correlated components.
        behaviour_timestamps (dict): Dictionary containing behaviour names as keys and timestamps as values.
        animal_id (str): Identifier for the animal (used in saving the plot).
        model_path (str): Path to the model (used in constructing the save path).

    Returns:
        None
    """
    n_to_plot = min(8, len(highly_correlated_components[0]))

    # Generate a color palette based on the number of behaviours
    color_set = sns.color_palette('husl', len(behaviour_timestamps))

    for i in range(n_to_plot):
        idx1, idx2 = highly_correlated_components[0][i], highly_correlated_components[1][i]
        distance = distances[idx1, idx2]

        # Plot behavior spans
        for j, (behaviour, timestamps) in enumerate(behaviour_timestamps.items()):
            for k, timestamp in enumerate(timestamps):
                ax.axvspan(timestamp[0], timestamp[1], label=behaviour if i == 0 and k == 0 else None, color=color_set[j], alpha=0.3)

        # Plot traces
        ax.plot(DFF_z_scored[idx1] + 10*i, color='k', alpha=0.7)
        ax.plot(DFF_z_scored[idx2] + 10*i, alpha=0.6)
        ax.text(10, 10*i + 5, f'Distance: {distance:.2f}', fontsize=12)

    ax.legend()

def process_session(session_df, idx, row, plot, axes):
    """Processes a session to calculate mean correlations and return a DataFrame."""
    cohort, session, animal_id, model_path = row['Cohort'], row['Session'], row['Mouse'], row['CNMFE Model Path']
    day_of_helping = row['DayofHelping']

    annotations_filename = row['ManualAnnotationSide']
    annotations_path = os.path.join(ANNOTATIONS_DIR, row['Cohort'], row['Session'], f'{annotations_filename}.csv')

    model, DFF_z_scored, annotations = load_data(model_path, annotations_path)
    if annotations is None:
        return pd.DataFrame(), None, None, None, None, session_df

    pnr_path = row['Peak-to-Noise Ratio Path']
    pnr = imread(pnr_path)

    unique_behaviours = annotations['Behavior'].unique()
    all_behaviour_timestamps = {behaviour: get_behaviour_timestamps(annotations, behaviour) for behaviour in unique_behaviours}

    session_metadata = metadata_df[(metadata_df['Cohort'] == cohort) & (metadata_df['Session'] == session) & (metadata_df['Mouse'] == animal_id)]

    accepted_components = model.estimates.idx_components
    liberation_cells_ids = session_metadata['Component ID']

    D, D_val, cross_correlation, corr_comps, G, pccs = compute_correlations(model, DFF_z_scored, accepted_components)
    __, D_val_lib, __, corr_comps_lib, __, pccs_lib = compute_correlations(model, DFF_z_scored, np.intersect1d(liberation_cells_ids, accepted_components))
    
    corr_comps_unique = np.unique(np.concatenate(corr_comps))

    # Check that corr_comps_unique is not empty
    if len(corr_comps_unique) == 0:
        return (pd.DataFrame(), None, None, None, None, session_df)

    percentage_of_liberation_cells = len(corr_comps_lib) / len(corr_comps_unique) * 100

    # Update the session DataFrame with the percentage of liberation cells in highly correlated components
    session_df.at[idx, '% of liberation cells in highly correlated components'] = percentage_of_liberation_cells

    # TODO: Check how many of these are liberation cells; plot vienn diagram with that information

    if plot:
        axes['image-example'].imshow(pnr, cmap='viridis')
        axes['image-example'].set_title(f'Peak-to-Noise Ratio', fontsize=16)

        # Plot the cross-correlation matrix
        axes['cross-correlation-matrix'].imshow(cross_correlation, aspect='equal')
        axes['cross-correlation-matrix'].set_title('Cross-correlation matrix')

        # Plot the distance matrix
        axes['distance-matrix'].imshow(D, cmap='viridis')
        axes['distance-matrix'].set_title('Distance matrix')

        # Plot the traces of highly correlated components
        plot_traces(axes['traces-example'], DFF_z_scored, D, corr_comps, all_behaviour_timestamps)
        axes['traces-example'].set_title('Traces of highly correlated components')    
        
        # Plot the correlation vs distance plot
        axes['correlation-vs-distance'].scatter(D_val, pccs, alpha=0.5, s=1, c='k')
        axes['correlation-vs-distance'].set_xlabel('Distance (px)')
        axes['correlation-vs-distance'].set_ylabel('Pearson correlation coefficient')
        axes['correlation-vs-distance'].set_ylim(-1, 1)
        axes['correlation-vs-distance'].set_title(f'Correlation vs distance, {session}')

        # # take only positive values
        # pccs_pos, D_val_pos = zip(*[(p, d) for p, d in zip(pccs, D_val) if p > 0])

        # # fit exponential curve
        # def exponential(x, a, b, c):
        #     return a * np.exp(-b * x) + c
        
        # popt, pcov = curve_fit(exponential, D_val_pos, pccs_pos)
        # print('Fitted parameters:', popt)

        # # Plot the exponential fit
        # x = np.linspace(0, 400, 100)
        # y = exponential(x, *popt)

        # axes['correlation-vs-distance'].plot(x, y, '--', label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
        # axes['correlation-vs-distance'].legend()


    results = {
        'Cohort': cohort,
        'Session': session,
        'Mouse': animal_id,
        'DayOfHelping': day_of_helping,
        'Average degree (all)': round(np.mean([d for _, d in G.degree()]), 2),
        # 'Average degree (liberation cells)': round(np.mean([d for _, d in G_liberation.degree()]), 2) if len(G_liberation) > 0 else 0,
        'Average distance (px)': round(np.mean(D_val), 2),
        'Number of correlated pairs (all)': len(corr_comps[0]),
        # 'Percentage of liberation cells': round(percentage_of_liberation_cells, 2),
        # 'Number of correlated pairs (liberation cells)': len(highly_correlated_components_liberation[0]),
        'Clustering coefficient (all)': round(nx.average_clustering(G), 2),
        # 'Clustering coefficient (liberation cells)': round(nx.average_clustering(G_liberation), 2) if len(G_liberation) > 0 else 0,
    }

    return (pd.DataFrame(results, index=[0]), pccs, pccs_lib, D_val, D_val_lib, session_df)

def compute_correlations(model, DFF_z_scored, accepted_components):
    """
    Compute correlations and related metrics from the model and DFF data.

    Parameters:
        model: The model containing the estimates.
        DFF_z_scored (np.array): Z-scored delta F/F values.
        accepted_components: List or array of accepted components.

    Returns:
        distances (np.array): Distance matrix between components.
        distance_values (np.array): Values extracted from the distance matrix.
        cross_correlation (np.array): Cross-correlation matrix.
        highly_correlated_components (tuple): Indices of highly correlated components.
        G (networkx.Graph): Graph constructed from adjacency matrix.
        correlation_values (np.array): Values extracted from the cross-correlation matrix.
    """

    # Validate input data
    if DFF_z_scored is None or np.isnan(DFF_z_scored).any():
        return None, None, None, None, None, None

    # Compute binary activity map and centers of mass (coms)
    A_binary = np.stack([
        np.where(model.estimates.A[:, i].toarray().reshape(IMG_DIMS, order='F') > 0, i + 1, 0)
        for i in range(DFF_z_scored.shape[0])
    ], axis=0)
    coms = np.array([np.mean(np.where(A_binary[i] > 0), axis=1) for i in range(A_binary.shape[0])])

    # Calculate pairwise distances between component centers
    distances = cdist(coms, coms)

    # Calculate cross-correlation and adjacency matrix
    cross_correlation = np.corrcoef(DFF_z_scored)
    adjacency_matrix = (cross_correlation > THRESHOLD)
    np.fill_diagonal(adjacency_matrix, 0)
    adjacency_matrix = np.tril(adjacency_matrix, k=-1)

    # Identify highly correlated components
    highly_correlated_components = np.where(adjacency_matrix)

    # Construct a graph from the adjacency matrix
    G = nx.from_numpy_array(adjacency_matrix)

    # Filter results based on accepted components
    accepted_components = np.array(accepted_components)
    distances = distances[np.ix_(accepted_components, accepted_components)]
    cross_correlation = cross_correlation[np.ix_(accepted_components, accepted_components)]

    # Filter highly correlated components based on accepted components
    highly_correlated_components_accepted = (np.array([i for i in highly_correlated_components[0] if i in accepted_components]), np.array([i for i in highly_correlated_components[1] if i in accepted_components]))

    # Construct a graph from the filtered adjacency matrix
    filtered_adjacency_matrix = adjacency_matrix[np.ix_(accepted_components, accepted_components)]
    G = nx.from_numpy_array(filtered_adjacency_matrix)

    # Extract values for distances and correlations
    distance_values = get_values(distances)
    correlation_values = get_values(cross_correlation)

    return distances, distance_values, cross_correlation, highly_correlated_components_accepted, G, correlation_values

def get_values(matrix):
    lower_triangle_matrix = matrix[np.tril_indices(matrix.shape[0], k=-1)]
    values_list = lower_triangle_matrix.tolist()
    return values_list

def main():
    fig, axes = setup_plot()
    global metadata_df
    metadata_df = pd.read_csv(METADATA_FILE)
    session_df = pd.read_csv(SESSION_FILE)
    session_df = session_df[~session_df['Session'].str.contains('Hab')]
    # Filter out only Mar_2024 cohort
    session_df = session_df[session_df['Cohort'] == 'Mar_2024']
    # session_df = session_df[session_df['Session'] == 'HBTD6']

    # Add new column with % of liberation cells in highly correlated components
    session_df['% of liberation cells in highly correlated components'] = np.nan

    # Reindex the DataFrame
    session_df = session_df.reset_index(drop=True)

    correlation_values_accepted = []
    correlation_values_accepted_lib = []
    D = []
    D_lib = []
    all_data = pd.DataFrame()

    # Get random index for plotting 
    idx = np.random.randint(0, len(session_df))
    print('Random index:', idx)
    for i, row in tqdm(session_df.iterrows()):
        print('Processing session:', i, row['Session'])
        if i == idx:
            plot=True
            print('Plotting session:', row['Session'])
        else:
            plot=False

        corr_data, pccs, pccs_lib, dist_accepted, dist_accepted_lib, session_df = process_session(session_df, i, row, plot, axes)
        if not corr_data.empty:
            all_data = pd.concat([all_data, corr_data], ignore_index=True)
            if pccs is not None and dist_accepted is not None:
                correlation_values_accepted.extend(pccs)
                D.extend(dist_accepted)
            if pccs_lib is not None and dist_accepted_lib is not None:
                correlation_values_accepted_lib.extend(pccs_lib)
                D_lib.extend(dist_accepted_lib)
    assert len(correlation_values_accepted) == len(D)

    # Plot the clustering coefficient over days
    sns.barplot(x='Session', y='Clustering coefficient (all)', data=all_data, ax=axes['clustering-coefficient-over-days'])
    axes['clustering-coefficient-over-days'].set_title('Clustering coefficient over days')

    # Plot the clustering coefficient over days of helping
    sns.barplot(x='DayOfHelping', y='Clustering coefficient (all)', data=all_data, ax=axes['clustering-coefficient-over-days-of-helping'])
    axes['clustering-coefficient-over-days-of-helping'].set_title('Clustering coefficient over days of helping')

    # Plot histogram of PCC values for all accepted components and liberation cells
    sns.histplot(correlation_values_accepted, ax=axes['pcc-distribution'], label='All components', color='k', bins=50, fill=False, alpha=0.5)
    sns.histplot(correlation_values_accepted_lib, ax=axes['pcc-distribution'], label='Liberation cells', color='r', bins=50, fill=False, alpha=0.5)
    axes['pcc-distribution'].set_title('PCC distribution')
    axes['pcc-distribution'].legend()

    # Plot cumulative distribution of PCC values for all accepted components and liberation cells
    sns.ecdfplot(correlation_values_accepted, ax=axes['pcc-cumulative-distribution'], label='All components', color='k')
    sns.ecdfplot(correlation_values_accepted_lib, ax=axes['pcc-cumulative-distribution'], label='Liberation cells', color='r')
    axes['pcc-cumulative-distribution'].set_title('PCC cumulative distribution')
    axes['pcc-cumulative-distribution'].legend()

    # Remove NaN values
    session_df = session_df.dropna(subset=['% of liberation cells in highly correlated components'])
    print(session_df.head())
    # plot 2 boxplots: one for % of liberation cells in highly correlated components, and one for just % of liberation cells
    plot_boxplots(session_df['% of liberation cells in highly correlated components'].values, session_df['% of liberation cells'].values, axes['lib-cells-percentage'], '', ('Correlated cells', 'All cells'), 'Percentage of liberation cells')

    # Finalize plot
    for ax in axes.values():
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    all_data.to_csv('processed-data/functional_connectivity.csv', index=False)

    # Save the plot
    plt.savefig('tech_plots/functional_connectivity.pdf')

if __name__ == "__main__":
    main()