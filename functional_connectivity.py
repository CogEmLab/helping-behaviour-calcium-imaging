import matplotlib
matplotlib.use('Agg')  # Set the backend to 'Agg' before importing pyplot
matplotlib.rcParams['axes.unicode_minus'] = False
import datetime
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
from src.utils import load_session_file

# Load the configuration
config = load_config('tech_docs/functional_connectivity_config.yaml')

# Access the configuration values
SESSION_FILE = load_session_file(config['session_file_dir'])
METADATA_FILE = config['metadata_file']
ANNOTATIONS_DIR = config['annotations_dir']

IMG_DIMS = tuple(config['img_dims'])
THRESHOLD = config['threshold']
FIG_SIZE = tuple(config['fig_size'])
GRID_SPEC = tuple(config['grid_spec'])
SESSION_ORDER = config['session_order']
DAYS_OF_HELPING_ORDER = config['days_of_helping_order']

def process_session(SESSION_FILE, row, plot, axes, METADATA_FILE, random_event=False):
    """Processes a session to calculate mean correlations and return a DataFrame."""

    # Create a session object
    session = Session(row)

    # Load the data
    pnr = session.load_pnr_image()
    session.get_neuron_labels()
    dff_signal = session.load_dff_signal()
    neuron_positions = session.neuron_positions
    
    # Load the metadata for liberation components
    liberation_associated_metadata = session.load_liberation_metadata('Liberation')
    random_event_metadata = session.load_liberation_metadata('Random')

    # Get the IDs of the liberation cells
    if liberation_associated_metadata is not None:
        liberation_cells_ids = session.get_liberation_cells_ids()
    else:
        liberation_cells_ids = None

    if random_event_metadata is not None:
        random_event_cells_ids = session.get_liberation_cells_ids()
    else:
        random_event_cells_ids = None

    # Get the timecode of the victim liberation
    victim_lib_timecode = parse_victim_lib_timecode(session.victim_lib_timecode)
    border = int((victim_lib_timecode * session.calcium_fps) - 15 * session.calcium_fps) if victim_lib_timecode is not None else None #* The formula calculates a time point 15 seconds before the victim liberation time, converted to frame numbers in the calcium imaging data.

    # Compute correlations
    distances, correlation_matrix, adjacency_matrix, correlated_pairs = compute_correlations(neuron_positions, dff_signal[:,:border], THRESHOLD, bin_size_seconds=60, sampling_rate=session.calcium_fps)

    # Create a graph from the adjacency matrix
    position_dict, G, colors, labels = build_graph(session, neuron_positions, adjacency_matrix, liberation_cells_ids)

    # Separate correlation pairs depending on the neuron identity; only do it if there are at least some liberation cells
    correlated_pairs_lib_to_all = []
    correlated_pairs_only_lib = []

    # Defined empty objects for correlation results of liberation cells
    correlation_matrix_liberation_only = None
    H_lib = None

    # Create a graph from the adjacency matrix for only the liberation cells
    if liberation_cells_ids is not None and len(liberation_cells_ids) > 2:

        for (cell1, cell2) in zip(correlated_pairs[0], correlated_pairs[1]):

            #! I don't really need this, it serves only a purpose of making a list of tuples
            if cell1 in liberation_cells_ids and cell2 in liberation_cells_ids:
                correlated_pairs_only_lib.append((cell1, cell2))
            elif cell1 in liberation_cells_ids:
                correlated_pairs_lib_to_all.append((cell1, cell2))
            elif cell2 in liberation_cells_ids:
                correlated_pairs_lib_to_all.append((cell2, cell1))

        # Compute correlations for only the liberation cells
        __, correlation_matrix_liberation_only, adjacency_matrix_liberation_only, correlated_pairs_liberation_only = compute_correlations(neuron_positions[liberation_cells_ids], dff_signal[liberation_cells_ids, :border], THRESHOLD, bin_size_seconds=60, sampling_rate=session.calcium_fps)

        # Convert correlation pairs to list of tuples
        correlated_pairs_liberation_only = list(zip(*correlated_pairs_liberation_only))

        # Check that the subset of liberation cells participating in correlations is same independent on how we computed it
        if len(list(set(correlated_pairs_liberation_only))) != len(list(set(correlated_pairs_only_lib))):
            # Raise a warning with numbers of correlated cells + session, animal and cohort
            print('Something went wrong with the correlation computation! Check the numbers of correlated cells in session: ' + str(session.session) + ', animal: ' + str(session.animal_id) + ', cohort: ' + str(session.cohort) + '\n' + 'Number from general graph ' + str(len(correlated_pairs_only_lib)) + ', number from lib only graph ' + str(len(correlated_pairs_liberation_only)))
        
        # Create a graph from the adjacency matrix for only the liberation cells
        __, H_lib, __, __ = build_graph(session, neuron_positions[liberation_cells_ids], adjacency_matrix_liberation_only, liberation_cells_ids, suffix='_liberation_only')

    # Convert correlation pairs to list of tuples
    correlated_pairs_tuple_list = list(zip(*correlated_pairs))
    
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

    # Plot correlation matrix        
    fig_filename = f"{session.animal_id}_{session.session}_cross_correlation_matrix.png"
    fig_path = os.path.join(f"figures", 'functional_connectivity', "cross-correlations", session.cohort, session.session, fig_filename)
    plot_cross_correlation_matrix(
        None, correlation_matrix,
        title=f"Cross-correlation matrix, animal: {session.animal_id}, session: {session.session}",
        save_path=fig_path, vmin=-0.5, vmax=0.5
    )

    fig_filename = f"{session.animal_id}_{session.session}_distance_matrix.png"
    fig_path = os.path.join(f"figures", 'functional_connectivity', "distances", session.cohort, session.session, fig_filename)
    plot_distance_matrix(
        None, distances,
        title=f"Distance matrix, animal: {session.animal_id}, session: {session.session}",
        save_path=fig_path,
    )

    if plot:          
        plot_session_example(axes, session, pnr, dff_signal, distances, correlation_matrix, correlated_pairs_tuple_list, position_dict, G, colors, labels, liberation_cells_ids)

    # Plot the traces of highly correlated components
    fig_filename = f"{session.animal_id}_{session.session}_highly_correlated_traces.png"
    fig_path = os.path.join(f"figures", 'functional_connectivity', "general", session.cohort, session.session, fig_filename)

    # plot_correlated_traces(ax=None, traces=accepted_dff_signal, component_pairs=correlated_pairs_tuple_list, distances=distances, liberation_cell_ids=liberation_cells_ids, correlation_matrix=correlation_matrix, title='Traces of highly correlated pairs (overlaid)', filename=fig_path, victim_lib_time=session.victim_lib_timecode * session.calcium_fps, fps=session.calcium_fps)
    plot_correlated_traces(ax=None, traces=dff_signal, component_pairs=correlated_pairs_tuple_list, distances=distances, liberation_cell_ids=liberation_cells_ids, correlation_matrix=correlation_matrix, title='Traces of highly correlated pairs (overlaid)', filename=fig_path, victim_lib_time=session.victim_lib_timecode * session.calcium_fps, fps=session.calcium_fps)

    results = {
        'Cohort': session.cohort,
        'Session': session.session,
        'Mouse': session.animal_id,
        'DayOfHelping': session.day_of_helping,
        'Latency (min)': session.latency,
        'Final outcome': session.final_outcome,
        'Session outcome': session.session_outcome,
        'Average degree (all)': round(np.mean([d for _, d in G.degree()]), 2),
        'Average degree (lib)': round(np.mean([d for _, d in G.subgraph(list(liberation_cells_ids)).degree()]), 2) if liberation_cells_ids is not None else None,
        'Average distance (px)': round(np.mean(get_values(distances)), 2),
        'Number of correlated pairs (all)': len(correlated_pairs[0]),
        'Number of correlated pairs (lib -> all)': len(correlated_pairs_lib_to_all),
        'Number of correlated pairs (lib -> lib)': len(correlated_pairs_only_lib),
        'Clustering coefficient (all)': round(nx.average_clustering(G), 2),
        'Clustering coefficient (lib)': round(nx.average_clustering(G.subgraph(list(liberation_cells_ids))), 2) if liberation_cells_ids is not None else None,
        'Clustering coefficient from full graph (lib)': round(nx.average_clustering(H_lib), 2) if H_lib is not None else None,
        'Global Efficiency (all)': round(nx.global_efficiency(G), 2),
        'Global Efficiency (lib)': round(nx.global_efficiency(G.subgraph(list(liberation_cells_ids))), 2) if liberation_cells_ids is not None else None,
        'Global Efficiency from full graph (lib)': round(nx.global_efficiency(H_lib), 2) if H_lib is not None else None,
        'Density (all)': round(nx.density(G), 2),
        'Density (lib)': round(nx.density(G.subgraph(list(liberation_cells_ids))), 2) if liberation_cells_ids is not None else None,
        'Density from full graph (lib)': round(nx.density(H_lib), 2) if H_lib is not None else None,
        'Has liberation cells': liberation_cells_ids is not None and len(liberation_cells_ids) > 0,
        'Number of liberation cells': len(liberation_cells_ids) if liberation_cells_ids is not None else 0,
        'Average response score (lib -> all)': avg_response_score_lib_to_all if metadata is not None else None,
        'Average response score (lib -> lib)': avg_response_score_only_lib if metadata is not None else None,
    }

    # Add identity counts to results
    if metadata is not None:
        for identity, count in identity_counts_lib_to_all.items():
            results[f'Number of {identity} cells in correlations (lib -> all)'] = count
        for identity, count in identity_counts_only_lib.items():
            results[f'Number of {identity} cells in correlations (lib -> lib)'] = count

    if liberation_cells_ids is not None and len(liberation_cells_ids) > 2:
        correlation_values = get_values(correlation_matrix)
        correlation_liberation_only_values = get_values(correlation_matrix_liberation_only) 
    else:
        correlation_values = []
        correlation_liberation_only_values = []

    # Instead of plotting these values here, sample random 40 of them and return
    sampled_distances, sampled_correlations = sample_values(get_values(distances), get_values(correlation_matrix))
    
    return (pd.DataFrame(results, index=[0]), correlation_values, correlation_liberation_only_values, sampled_distances, sampled_correlations, SESSION_FILE)

def main(random_event=False):
    fig, axes = setup_plot(FIG_SIZE, GRID_SPEC)

    # Filter the session file
    SESSION_FILE = SESSION_FILE[SESSION_FILE['Excluded'] != True]

    # # Reindex the DataFrame
    # SESSION_FILE = SESSION_FILE.reset_index(drop=True)

    correlation_values_accepted, correlation_values_accepted_lib = [], []
    sampled_distances_all, sampled_correlations_all = [], []

    all_data = pd.DataFrame()

    # Get random index for plotting 
    idx = np.random.randint(0, len(SESSION_FILE))
    for i, row in tqdm(SESSION_FILE.iterrows()):
        if i == idx:
            print('Plotting session with index:', i)
            plot=True
        else:
            plot=False

        corr_data, pccs, pccs_lib, sampled_distances, sampled_correlations, SESSION_FILE = process_session(SESSION_FILE, row, plot, axes, METADATA_FILE, random_event=random_event)
        
        if not corr_data.empty:
            # Add correlation data to the DataFrame
            all_data = pd.concat([all_data, corr_data], ignore_index=True)
            sampled_correlations_all.extend(sampled_correlations)
            sampled_distances_all.extend(sampled_distances)
            if pccs is not None:
                correlation_values_accepted.extend(pccs)
            if pccs_lib is not None:
                correlation_values_accepted_lib.extend(pccs_lib)

    # Remove 847 Mouse from plotting
    all_data = all_data[all_data['Mouse'] != 847]
        
    # Plot the clustering coefficient over days
    plot_values_over_days(all_data, 'Session', 'Clustering coefficient (all)', ax=axes['clustering-coefficient-over-days'], order=SESSION_ORDER, hue_column='Mouse')
    # Plot the clustering coefficient over days of helping
    plot_values_over_days(all_data, 'DayOfHelping', 'Clustering coefficient (all)', ax=axes['clustering-coefficient-over-days-of-helping'], order=DAYS_OF_HELPING_ORDER, hue_column='Mouse')
    
    # Plot correlation vs distance
    axes['correlation-vs-distance'] = plot_scatter(
        sampled_distances_all,
        sampled_correlations_all,
        ax=axes['correlation-vs-distance'],
        title='Correlation vs distance',
        ylim=(-1, 1),
        perform_test=True,
        filename=None
    )

    # Plot correlation vs distance (separate figure)
    now = datetime.datetime.now()
    plot_scatter(
        sampled_distances_all,
        sampled_correlations_all,
        title='Correlation vs distance',
        xlabel='Distance (px)',
        ylabel='Pearson correlation coefficient',
        ylim=(-1, 1),
        perform_test=True,
        filename=os.path.join('figures', 'functional_connectivity', 'general', f'pcc_vs_distance_{os.path.basename(__file__)[:-3]}_{now.strftime("%Y-%m-%d_%H-%M-%S")}.png')
    )

    # Plot histogram of PCC values for all accepted components and Liberation cells
    df = pd.DataFrame({'Label': 'All cells', 'PCC': correlation_values_accepted})
    df_lib = pd.DataFrame({'Label': 'Liberation cells', 'PCC': correlation_values_accepted_lib})
    df = pd.concat([df, df_lib], ignore_index=True)

    # Plot histogram of PCC values for all accepted components and liberation cells
    sns.histplot(data=df, x='PCC', hue='Label', ax=axes['pcc-distribution'], bins=100, common_norm=False, stat='density', fill=False, alpha=0.5, palette={'All cells': 'black', 'Liberation cells': 'tab:red'}, kde=False)
    axes['pcc-distribution'].set_xlim(-1, 1)

    # Plot histogram of PCC values for all accepted components and liberation cells (separate figure)
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.histplot(data=df, x='PCC', hue='Label', ax=ax, bins=50, common_norm=False, stat='density', fill=False, alpha=0.7, palette={'All cells': 'black', 'Liberation cells': 'tab:red'}, kde=False)
    ax.set_xlim(-1, 1)
    ax.set_title('PCC distribution')
    ax.set_xlabel('Pearson correlation coefficient')
    ax.set_ylabel('Probability density')
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join('figures', 'functional_connectivity', 'general', f'pcc_distribution_{os.path.basename(__file__)[:-3]}_{now.strftime("%Y-%m-%d_%H-%M-%S")}.png'), dpi=300)
    fig.savefig(os.path.join('figures', 'functional_connectivity', 'general', f'pcc_distribution_{os.path.basename(__file__)[:-3]}_{now.strftime("%Y-%m-%d_%H-%M-%S")}.eps'), dpi=300, format='eps', bbox_inches='tight')
    plt.close(fig)

    # Plot cumulative distribution of PCC values for all accepted components and liberation cells
    sns.ecdfplot(correlation_values_accepted, ax=axes['pcc-cumulative-distribution'], label='All components', color='k')
    sns.ecdfplot(correlation_values_accepted_lib, ax=axes['pcc-cumulative-distribution'], label='Liberation cells', color='r')
    axes['pcc-cumulative-distribution'].set_title('PCC cumulative distribution')
    axes['pcc-cumulative-distribution'].legend()

    if random_event:
        suffix='_random_event'
    else:
        suffix=''
    
    # Plot cumulative distribution of PCC values for all accepted components and liberation cells (separate figure)
    plot_ecdf_with_ks_test(
        correlation_values_accepted,
        correlation_values_accepted_lib,
        filename=os.path.join('figures', 'functional_connectivity', 'general', f'pcc_cumulative_distribution{suffix}_{os.path.basename(__file__)[:-3]}_{now.strftime("%Y-%m-%d_%H-%M-%S")}.png'),
        shuffled_signal_color='red')
    
    # Save correlation_values_accepted and correlation_values_accepted_lib as npy files
    np.save(os.path.join('processed-data', f'correlation_values{suffix}_{os.path.basename(__file__)[:-3]}_{now.strftime("%Y-%m-%d_%H-%M-%S")}.npy'), correlation_values_accepted)
    np.save(os.path.join('processed-data', f'correlation_values_lib{suffix}_{os.path.basename(__file__)[:-3]}_{now.strftime("%Y-%m-%d_%H-%M-%S")}.npy'), correlation_values_accepted_lib)

    # Finalize plot
    for ax in axes.values():
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)

    all_data.to_csv(os.path.join('processed-data', f'functional_connectivity_{os.path.basename(__file__)[:-3]}_{now.strftime("%Y-%m-%d_%H-%M-%S")}.csv'), index=False)

    # Save the plot
    plt.tight_layout()
    plt.savefig(os.path.join('tech_plots', f'functional_connectivity_{os.path.basename(__file__)[:-3]}_{now.strftime("%Y-%m-%d_%H-%M-%S")}.pdf'))
 
if __name__ == "__main__":
    main(random_event=True)
