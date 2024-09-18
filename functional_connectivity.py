import matplotlib
matplotlib.use('Agg')  # Set the backend to 'Agg' before importing pyplot

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
from src.plotting import *
from src.session_loader import Session
from src.functional_connectivity_utils import *
import networkx as nx

# Load the configuration
config = load_config('tech_docs/functional_connectivity_config.yaml')

# Access the configuration values
SESSION_FILE = config['session_file']
METADATA_FILE = config['metadata_file']
ANNOTATIONS_DIR = config['annotations_dir']

IMG_DIMS = tuple(config['img_dims'])
THRESHOLD = config['threshold']
FIG_SIZE = tuple(config['fig_size'])
GRID_SPEC = tuple(config['grid_spec'])
SESSION_ORDER = config['session_order']
DAYS_OF_HELPING_ORDER = config['days_of_helping_order']

def process_session(session_df, row, plot, axes, METADATA_FILE):
    """Processes a session to calculate mean correlations and return a DataFrame."""

    # Create a session object
    session = Session(row)

    # Load the data
    pnr = session.load_pnr_image()
    session.get_neuron_labels()
    dff_signal = session.load_dff_signal()
    accepted_components = session.load_accepted_components()
    accepted_dff_signal = session.filter_dff_signal(dff_signal, accepted_components)
    neuron_positions = session.neuron_positions[list(accepted_components)]

    # Load the metadata for liberation components
    metadata = session.load_liberation_metadata(METADATA_FILE)
    if metadata is not None:
        liberation_cells_ids = session.get_liberation_cells_ids() # old indexing
    else:
        liberation_cells_ids = None

    # Get the timecode of the victim liberation
    victim_lib_timecode = parse_victim_lib_timecode(session.victim_lib_timecode)
    border = int((victim_lib_timecode * session.calcium_fps) - 15 * session.calcium_fps) if victim_lib_timecode is not None else None #* The formula calculates a time point 15 seconds before the victim liberation time, converted to frame numbers in the calcium imaging data.

    # Create a mapping from original indices to new indices
    component_mapping = {orig: new for new, orig in enumerate(accepted_components)}

    # Compute correlations
    distances, correlation_matrix, adjacency_matrix, correlated_pairs = compute_correlations(neuron_positions, accepted_dff_signal[:,:border], THRESHOLD) #* Returns correlated pairs indexed according to old scheme before filtering
    
    correlated_pairs_new = []
    correlated_pairs_lib_to_all = []
    correlated_pairs_only_lib = []
    liberation_cell_ids_mapped = None

    if liberation_cells_ids is not None and len(liberation_cells_ids) > 2:

        for pair in zip(correlated_pairs[0], correlated_pairs[1]):
            cell1, cell2 = pair # old indexing

            correlated_pairs_new.append((component_mapping[cell1], component_mapping[cell2])) # from old to new; for plotting purposes only
            if cell1 in liberation_cells_ids and cell2 in liberation_cells_ids:
                correlated_pairs_only_lib.append((cell1, cell2))
            if cell1 in liberation_cells_ids:
                correlated_pairs_lib_to_all.append((cell1, cell2))
            elif cell2 in liberation_cells_ids:
                correlated_pairs_lib_to_all.append((cell2, cell1))

        cross_correlation_liberation_only = np.corrcoef(session.filter_dff_signal(dff_signal, liberation_cells_ids)[:, 1:border])
        liberation_cell_ids_mapped = {component_mapping[i] for i in liberation_cells_ids} # new indexing, to e used for graphs only
    else:
        correlated_pairs_new = [(component_mapping[cell1], component_mapping[cell2]) for cell1, cell2 in zip(correlated_pairs[0], correlated_pairs[1])]
        cross_correlation_liberation_only = None


    # Filter out response scores and identities for unique cells participating in correlations
    if metadata is not None:
        correlated_liberation_cells_lib_to_all = set({pair[0] for pair in correlated_pairs_lib_to_all})

        assert correlated_liberation_cells_lib_to_all.issubset(liberation_cells_ids)

        filtered_response_scores_lib_to_all = metadata.loc[metadata['Component ID'].isin(correlated_liberation_cells_lib_to_all), 'Response Score'].values
        filtered_identities_lib_to_all = metadata.loc[metadata['Component ID'].isin(correlated_liberation_cells_lib_to_all), 'Group'].values
        
        correlated_liberation_cells_only_lib = set({cell for pair in correlated_pairs_only_lib for cell in pair})
        assert correlated_liberation_cells_only_lib.issubset(liberation_cells_ids)

        filtered_response_scores_only_lib = metadata.loc[metadata['Component ID'].isin(correlated_liberation_cells_only_lib), 'Response Score'].values
        filtered_identities_only_lib = metadata.loc[metadata['Component ID'].isin(correlated_liberation_cells_only_lib), 'Group'].values

        avg_response_score_lib_to_all = filtered_response_scores_lib_to_all.mean()
        identity_counts_lib_to_all = pd.Series(filtered_identities_lib_to_all).value_counts().to_dict()
        
        avg_response_score_only_lib = filtered_response_scores_only_lib.mean()
        identity_counts_only_lib = pd.Series(filtered_identities_only_lib).value_counts().to_dict()

    # Create a graph from the adjacency matrix
    position_dict, G, colors, labels = build_graph(session, neuron_positions, adjacency_matrix, liberation_cell_ids_mapped)

    fig_filename = f"{session.animal_id}_{session.session}_cross_correlation_matrix.png"
    fig_path = os.path.join(f"figures", "cross-correlations", session.cohort, session.session, fig_filename)
    plot_cross_correlation_matrix(
        None, correlation_matrix,
        title=f"Cross-correlation matrix, animal: {session.animal_id}, session: {session.session}",
        save_path=fig_path, vmin=-0.5, vmax=0.5
    )

    fig_filename = f"{session.animal_id}_{session.session}_distance_matrix.png"
    fig_path = os.path.join(f"figures", "distances", session.cohort, session.session, fig_filename)
    plot_distance_matrix(
        None, distances,
        title=f"Distance matrix, animal: {session.animal_id}, session: {session.session}",
        save_path=fig_path,
    )

    if plot:          
        # Plot graph on image_example axis, overlay with PNR image
        plot_session_example(axes, session, pnr, accepted_dff_signal, distances, correlation_matrix, correlated_pairs_new, position_dict, G, colors, labels)

    results = {
        'Cohort': session.cohort,
        'Session': session.session,
        'Mouse': session.animal_id,
        'DayOfHelping': session.day_of_helping,
        'Latency (min)': session.latency,
        'Final outcome': session.final_outcome,
        'Session outcome': session.session_outcome,
        'Average degree (all)': round(np.mean([d for _, d in G.degree()]), 2),
        'Average degree (lib)': round(np.mean([d for _, d in G.subgraph(list(liberation_cell_ids_mapped)).degree()]), 2) if liberation_cell_ids_mapped is not None else None,
        'Average distance (px)': round(np.mean(get_values(distances)), 2),
        'Number of correlated pairs (all)': len(correlated_pairs[0]),
        'Number of correlated pairs (lib -> all)': len(correlated_pairs_lib_to_all),
        'Number of correlated pairs (lib -> lib)': len(correlated_pairs_only_lib),
        'Clustering coefficient (all)': round(nx.average_clustering(G), 2),
        'Clustering coefficient (lib)': round(nx.average_clustering(G.subgraph(list(liberation_cell_ids_mapped))), 2) if liberation_cell_ids_mapped is not None else None,
        'Global Efficiency (all)': round(nx.global_efficiency(G), 2),
        'Global Efficiency (lib)': round(nx.global_efficiency(G.subgraph(list(liberation_cell_ids_mapped))), 2) if liberation_cell_ids_mapped is not None else None,
        'Density (all)': round(nx.density(G), 2),
        'Density (lib)': round(nx.density(G.subgraph(list(liberation_cell_ids_mapped))), 2) if liberation_cell_ids_mapped is not None else None,
        'Has liberation cells': liberation_cells_ids is not None and len(liberation_cells_ids) > 0,
        'Average response score (lib -> all)': avg_response_score_lib_to_all if metadata is not None else None,
        'Average response score (lib -> lib)': avg_response_score_only_lib if metadata is not None else None,
    }

    # Add identity counts to results
    if metadata is not None:
        for identity, count in identity_counts_lib_to_all.items():
            results[f'Number of {identity} cells in correlations (lib -> all)'] = count
        for identity, count in identity_counts_only_lib.items():
            results[f'Number of {identity} cells in correlations (lib -> lib)'] = count

    correlation_values = get_values(correlation_matrix)
    correlation_liberation_only_values = get_values(cross_correlation_liberation_only) if cross_correlation_liberation_only is not None else None

    # Instead of plotting these values here, sample random 40 of them and return
    sampled_distances, sampled_correlations = sample_values(get_values(distances), correlation_values)
    
    return (pd.DataFrame(results, index=[0]), correlation_values, correlation_liberation_only_values, sampled_distances, sampled_correlations, session_df)

def main():
    fig, axes = setup_plot(FIG_SIZE, GRID_SPEC)
    global metadata_df
    metadata_df = pd.read_csv(METADATA_FILE)
    session_df = pd.read_csv(SESSION_FILE)
    session_df = session_df[~session_df['Session'].str.contains('Hab')]
    session_df = session_df[session_df['Mouse'] != 7964]

    # Reindex the DataFrame
    session_df = session_df.reset_index(drop=True)

    correlation_values_accepted, correlation_values_accepted_lib = [], []
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

        corr_data, pccs, pccs_lib, sampled_distances, sampled_correlations, session_df = process_session(session_df, row, plot, axes, METADATA_FILE)
        
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
    plot_values_over_days(all_data, 'Session', 'Clustering coefficient (all)', ax=axes['clustering-coefficient-over-days'], order=SESSION_ORDER, hue_column='Mouse')
    # Plot the clustering coefficient over days of helping
    plot_values_over_days(all_data, 'DayOfHelping', 'Clustering coefficient (all)', ax=axes['clustering-coefficient-over-days-of-helping'], order=DAYS_OF_HELPING_ORDER, hue_column='Mouse')
    
    # Plot correlation vs distance
    plot_scatter(
        sampled_distances_all,
        sampled_correlations_all,
        axes['correlation-vs-distance'],
        title='Correlation vs distance',
        alpha=0.5,
        s=1,
        c='k',
        ylim=(-1, 1),
        fit_type='exponential',
        perform_test=True
    )

    # Plot correlation vs distance (separate figure)
    plot_scatter(
        sampled_distances_all,
        sampled_correlations_all,
        None,
        title='Correlation vs distance',
        xlabel='Distance (px)',
        ylabel='Pearson correlation coefficient',
        ylim=(-1, 1),
        fit_type='exponential',
        perform_test=True,
        save_path=os.path.join('figures', 'functional_connectivity', 'general', 'pcc_vs_distance.png')
    )

    # Plot histogram of PCC values for all accepted components and Liberation cells
    df = pd.DataFrame({'Label': 'All cells', 'PCC': correlation_values_accepted})
    df_lib = pd.DataFrame({'Label': 'Liberation cells', 'PCC': correlation_values_accepted_lib})
    df = pd.concat([df, df_lib], ignore_index=True)

    # Plot histogram of PCC values for all accepted components and liberation cells
    sns.histplot(data=df, x='PCC', hue='Label', ax=axes['pcc-distribution'], bins=100, common_norm=False, stat='density', fill=False, alpha=0.5, palette={'All cells': 'black', 'Liberation cells': 'tab:red'}, kde=False)
    axes['pcc-distribution'].set_xlim(-1, 1)

    # Plot cumulative distribution of PCC values for all accepted components and liberation cells
    sns.ecdfplot(correlation_values_accepted, ax=axes['pcc-cumulative-distribution'], label='All components', color='k')
    sns.ecdfplot(correlation_values_accepted_lib, ax=axes['pcc-cumulative-distribution'], label='Liberation cells', color='r')
    axes['pcc-cumulative-distribution'].set_title('PCC cumulative distribution')
    axes['pcc-cumulative-distribution'].legend()

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
