import matplotlib
matplotlib.use('Agg')  # Set the backend to 'Agg' before importing pyplot

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import networkx as nx
from caiman.source_extraction.cnmf.cnmf import load_CNMF
from tqdm import tqdm
from matplotlib.gridspec import GridSpec
from typing import Dict, Tuple
import seaborn as snsaccepted_components
from tifffile import imread
from src.plotting import plot_boxplots, plot_correlated_traces, plot_cross_correlation_matrix, plot_distance_matrix
from src.utils import get_values, compute_correlations, build_graph, save_matrix
import seaborn as sns
from src.session_loader import Session

# Constants
SESSION_FILE = 'data/session_file_MiniscopeCW_all_Moises_latest_liberation_cells.csv'
METADATA_FILE = 'processed-data/metadata_cross_comp_updated.csv'
ANNOTATIONS_DIR = 'annotations'
IMG_DIMS = (400, 640)
THRESHOLD = 0.42
FIG_SIZE = (30, 15)
GRID_SPEC = (3, 6)
SESSION_ORDER = ['HBTD1', 'HBTD2', 'HBTD3', 'HBTD4', 'HBTD5', 'HBTD6', 'RECNvlCtx', 'Stranger', 'Toy', 'Empty', 'Pool', 'REMNvlCtx', 'REMHBTD']
DAYS_OF_HELPING_ORDER = [0, 1, 2, 3, 4, 'RECNvlCtx', 'REMNvlCtx', 'REMHBTD']

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

def process_session(session_df: pd.DataFrame, row: pd.Series, plot: bool, axes: Dict[str, plt.Axes]) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame]:
    """Processes a session to calculate mean correlations and return a DataFrame."""

    session = Session(row)

    pnr = session.load_pnr_image()
    session.get_neuron_labels()
    dff_signal = session.load_dff_signal()
    accepted_components = session.load_accepted_components()
    accepted_dff_signal = session.filter_dff_signal(dff_signal, accepted_components)
    # Create a mapping from original indices to new indices
    component_mapping = {orig: new for new, orig in enumerate(accepted_components)}

    neuron_positions = session.neuron_positions[accepted_components]

    distances, correlation_matrix, adjacency_matrix, correlated_pairs = compute_correlations(neuron_positions, accepted_dff_signal, THRESHOLD)
    
    __ = session.load_liberation_metadata('processed-data/metadata_cross_comp_updated.csv')
    liberation_cells_ids = session.get_liberation_cells_ids()
    correlated_pairs_new = []
    correlated_pairs_not_lib = []
    correlated_pairs_lib_to_all = []
    correlated_pairs_only_lib = []
    liberation_cell_ids_mapped = None

    if liberation_cells_ids is not None and len(liberation_cells_ids) > 2:
        liberation_cells_set = set(liberation_cells_ids)

        for pair in zip(correlated_pairs[0], correlated_pairs[1]):
            cell1, cell2 = pair

            correlated_pairs_new.append((component_mapping[cell1], component_mapping[cell2]))
            if cell1 in liberation_cells_set and cell2 in liberation_cells_set:
                correlated_pairs_only_lib.append((component_mapping[cell1], component_mapping[cell2]))
            elif cell1 in liberation_cells_set or cell2 in liberation_cells_set:
                correlated_pairs_lib_to_all.append((component_mapping[cell1], component_mapping[cell2]))
            else:
                correlated_pairs_not_lib.append((component_mapping[cell1], component_mapping[cell2]))

        cross_correlation_liberation_only = np.corrcoef(session.filter_dff_signal(dff_signal, liberation_cells_ids)[:, 1:])
        liberation_cell_ids_mapped = {component_mapping[i] for i in liberation_cells_ids}
    else:
        correlated_pairs_new = [(component_mapping[cell1], component_mapping[cell2]) for cell1, cell2 in zip(correlated_pairs[0], correlated_pairs[1])]
        cross_correlation_liberation_only = None

    if liberation_cells_ids is not None and len(liberation_cells_ids) > 2 and len(correlated_pairs[0]) > 0:
        unique_cells_correlated_lib = len(set(cell1 for pair in correlated_pairs_only_lib for cell1 in pair))
        unique_cells_correlated_no_lib = len(set(cell1 for pair in correlated_pairs_not_lib for cell1 in pair))
        session_df.at[session_df.index[session_df['Session'] == session.session].tolist()[0], '% of liberation cells in highly correlated components'] = unique_cells_correlated_lib / session.number_of_liberation_cells
        # Percentage of non-liberation cells in highly correlated components
        session_df.at[session_df.index[session_df['Session'] == session.session].tolist()[0], '% of other cells in highly correlated components'] = unique_cells_correlated_no_lib / (session.number_of_accepted_components - session.number_of_liberation_cells)
    # Create a graph from the adjacency matrix
    position_dict, G, colors, labels = build_graph(session, component_mapping, neuron_positions, adjacency_matrix, liberation_cells_ids)

    save_matrix(session, correlation_matrix, f'cross_correlation_matrix_{str(session.animal_id)}', plot_cross_correlation_matrix)
    save_matrix(session, distances, f'distance_matrix_{str(session.animal_id)}', plot_distance_matrix)
    
    if plot:          
        # Plot graph on image_example axis, overlay with PNR image
        # axes['image-example'].imshow(pnr, cmap='viridis')
        nx.draw_networkx_labels(G, pos=position_dict, ax=axes['image-example'], labels=labels, font_size=8, font_color='black')
        # colors = ['white' if c == 'black' else c for c in colors]
        nx.draw(G, pos=position_dict, node_size=50, node_color=colors, ax=axes['image-example'], edge_color='black', node_shape='*')
        axes['image-example'].set_title('Graph example, animal: ' + str(session.animal_id) + ', session: ' + session.session)

        # Plot the cross-correlation matrix, sort the values before plotting
        plot_cross_correlation_matrix(axes['cross-correlation-matrix'], correlation_matrix, 'Cross-correlation matrix, animal: ' + str(session.animal_id) + ', session: ' + session.session)

        # Plot the distance matrix
        plot_distance_matrix(axes['distance-matrix'], distances, 'Distance matrix, animal: ' + str(session.animal_id) + ', session: ' + session.session)

        # Plot the traces of highly correlated components
        plot_correlated_traces(axes['traces-example'], accepted_dff_signal, correlated_pairs_new, distances=distances)
        axes['traces-example'].set_title('Traces of highly correlated components')
        axes['traces-example'].axvline(session.victim_lib_timecode * session.calcium_fps, color='k', linestyle='dashed', linewidth=2, label='Victim crossing')

    results = {
        'Cohort': session.cohort,
        'Session': session.session,
        'Mouse': session.animal_id,
        'DayOfHelping': session.day_of_helping,
        'Latency (min)': session.latency,
        'Final outcome': session.final_outcome,
        'Session outcome': session.session_outcome,
        'Average degree (all)': round(np.mean([d for _, d in G.degree()]), 2),
        'Average distance (px)': round(np.mean(get_values(distances)), 2),
        'Number of correlated pairs (all)': len(correlated_pairs[0]),
        'Number of correlated pairs (lib to all)': len(correlated_pairs_lib_to_all),
        'Number of correlated pairs (only lib)': len(correlated_pairs_only_lib),
        'Clustering coefficient (all)': round(nx.average_clustering(G), 2),
        'Clustering coefficient (lib)': round(nx.average_clustering(G.subgraph(list(liberation_cell_ids_mapped))), 2) if liberation_cell_ids_mapped is not None else None,
        'Global Efficiency (all)': round(nx.global_efficiency(G), 2),
        'Global Efficiency (lib)': round(nx.global_efficiency(G.subgraph(list(liberation_cell_ids_mapped))), 2) if liberation_cell_ids_mapped is not None else None,
        'Density (all)': round(nx.density(G), 2),
        'Density (lib)': round(nx.density(G.subgraph(list(liberation_cell_ids_mapped))), 2) if liberation_cell_ids_mapped is not None else None,
        'Has liberation cells': liberation_cells_ids is not None and len(liberation_cells_ids) > 0
    }

    correlation_values = get_values(correlation_matrix)
    correlation_liberation_only_values = get_values(cross_correlation_liberation_only) if cross_correlation_liberation_only is not None else None

    # Instead of plotting these values here, sample random 40 of them and return
    n_samples = 50
    distances_values = get_values(distances)
    correlations_values = get_values(correlation_matrix)
    sample_idx = np.random.choice(len(distances_values), n_samples, replace=False)
    
    sampled_distances = distances_values[sample_idx]
    sampled_correlations = correlations_values[sample_idx.astype(int)]
    
    return (pd.DataFrame(results, index=[0]), correlation_values, correlation_liberation_only_values, sampled_distances, sampled_correlations, session_df)

def main():
    fig, axes = setup_plot()
    global metadata_df
    metadata_df = pd.read_csv(METADATA_FILE)
    session_df = pd.read_csv(SESSION_FILE)
    session_df = session_df[~session_df['Session'].str.contains('Hab')]
    # Filter out only Mar_2024 cohort
    # session_df = session_df[session_df['Cohort'] == 'Jan_2024']
    # session_df = session_df[session_df['Session'] == 'HBTD3']

    # Add new columns
    session_df['% of liberation cells in highly correlated components'] = np.nan
    session_df['% of other cells in highly correlated components'] = np.nan

    # Reindex the DataFrame
    session_df = session_df.reset_index(drop=True)

    correlation_values_accepted = []
    correlation_values_accepted_lib = []

    sampled_distances_all, sampled_correlations_all = [], []

    all_data = pd.DataFrame()

    # Get random index for plotting 
    idx = np.random.randint(0, len(session_df))
    for i, row in tqdm(session_df.iterrows()):
        if i == idx:
            print('Plotting session with index:', i)
            plot=True
        else:
            plot=False

        corr_data, pccs, pccs_lib, sampled_distances, sampled_correlations, session_df = process_session(session_df, row, plot, axes)
        
        if not corr_data.empty:
            # Add correlation data to the DataFrame
            all_data = pd.concat([all_data, corr_data], ignore_index=True)
            sampled_correlations_all.extend(sampled_correlations)
            sampled_distances_all.extend(sampled_distances)
            if pccs is not None:
                correlation_values_accepted.extend(pccs)
            if pccs_lib is not None:
                correlation_values_accepted_lib.extend(pccs_lib)

    # Plot the clustering coefficient over days
    sns.barplot(x='Session', y='Clustering coefficient (all)', data=all_data, ax=axes['clustering-coefficient-over-days'], fill=False, 
                capsize=0.1, err_kws={'linewidth': 1.5}, color='black', order=SESSION_ORDER)
    sns.swarmplot(x='Session', y='Clustering coefficient (all)', data=all_data, hue='Mouse', size=8,
                    palette='Set1', dodge=True, alpha=0.7, ax=axes['clustering-coefficient-over-days'], order=SESSION_ORDER, linewidth=1, legend=False)
    
    axes['clustering-coefficient-over-days'].set_title('Clustering coefficient over days')

    # Plot the clustering coefficient over days of helping
    sns.barplot(x='DayOfHelping', y='Clustering coefficient (all)', data=all_data, ax=axes['clustering-coefficient-over-days-of-helping'], fill=False,
                capsize=0.1, err_kws={'linewidth': 1.5}, color='black', order=DAYS_OF_HELPING_ORDER)
    sns.swarmplot(x='DayOfHelping', y='Clustering coefficient (all)', data=all_data, hue='Mouse', size=8,
                    palette='Set1', dodge=True, alpha=0.7, ax=axes['clustering-coefficient-over-days-of-helping'], order=DAYS_OF_HELPING_ORDER, linewidth=1, legend=False)
    
    axes['clustering-coefficient-over-days-of-helping'].set_title('Clustering coefficient over days of helping')
        
    # Plot the correlation vs distance plot
    axes['correlation-vs-distance'].scatter(sampled_distances_all, sampled_correlations_all, alpha=0.5, s=1, c='k')
    axes['correlation-vs-distance'].set_xlabel('Distance (px)')
    axes['correlation-vs-distance'].set_ylabel('Pearson correlation coefficient')
    axes['correlation-vs-distance'].set_ylim(-1, 1)
    axes['correlation-vs-distance'].set_title(f'Correlation vs distance')

    df = pd.DataFrame({'Label': 'All cells', 'PCC': correlation_values_accepted})
    df_lib = pd.DataFrame({'Label': 'Liberation cells', 'PCC': correlation_values_accepted_lib})
    # Concatenate the two dataframes
    df = pd.concat([df, df_lib], ignore_index=True)

    # Plot histogram of PCC values for all accepted components and liberation cells
    sns.histplot(data=df, x='PCC', hue='Label', ax=axes['pcc-distribution'], bins=100, common_norm=False, stat='density', fill=False, alpha=0.5, palette={'All cells': 'black', 'Liberation cells': 'tab:red'}, kde=False)
    axes['pcc-distribution'].set_xlim(-1, 1)

    # Plot cumulative distribution of PCC values for all accepted components and liberation cells
    sns.ecdfplot(correlation_values_accepted, ax=axes['pcc-cumulative-distribution'], label='All components', color='k')
    sns.ecdfplot(correlation_values_accepted_lib, ax=axes['pcc-cumulative-distribution'], label='Liberation cells', color='r')
    axes['pcc-cumulative-distribution'].set_title('PCC cumulative distribution')
    axes['pcc-cumulative-distribution'].legend()

    # Remove NaN values
    session_df = session_df.dropna(subset=['% of liberation cells in highly correlated components', '% of other cells in highly correlated components'])
    
    # plot 2 boxplots: one for % of liberation cells in highly correlated components, and one for just % of liberation cells
    plot_boxplots(session_df['% of liberation cells in highly correlated components'].values, session_df['% of other cells in highly correlated components'].values, axes['lib-cells-percentage'], '', ('Liberation cells', 'Other cells'), 'Percentage of correlated components')

    # Finalize plot
    for ax in axes.values():
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)

    all_data.to_csv('processed-data/functional_connectivity.csv', index=False)

    # Save the plot
    plt.tight_layout()
    plt.savefig('tech_plots/functional_connectivity.pdf')

if __name__ == "__main__":
    main()