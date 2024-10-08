import matplotlib
matplotlib.use('Agg')  # Set the backend to 'Agg' before importing pyplot

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.session_loader import Session
from src.functional_connectivity_utils import *
from src.utils import shuffle_signal
from collections import defaultdict
from src.plotting import *
from scipy.stats import ks_2samp

# Load the configuration
config = load_config('tech_docs/functional_connectivity_config.yaml')

# Access the configuration values
SESSION_FILE = config['session_file']
IMG_DIMS = tuple(config['img_dims'])
THRESHOLD = config['threshold']
SESSION_ORDER = config['session_order']
DAYS_OF_HELPING_ORDER = config['days_of_helping_order']

# Access the color palette
scatter_color = config['color_palette']['scatter_plot_points']
significance_color = config['color_palette']['significance_star']
correlated_trace_color = config['color_palette']['correlated_trace_example']
shuffled_signal_color = config['color_palette']['shuffled_signal_example']

# Define figure size in inches for Portrait Orientation
fig_width = 210 / 25.4  # Width in inches
fig_height = 148.5 / 25.4  # Height in inches

# Set up the supplementary figure dimensions
GRID_SPEC = (3, 4)
FIG_SIZE = (fig_width * 2, fig_height * 2)

def setup_plot(fig_size, grid_spec):
    plt.style.use('seaborn-v0_8-paper') # Setting up "paper" style to use figure as supplementary
    fig = plt.figure(figsize=fig_size)
    gs = GridSpec(*grid_spec, figure=fig)
    
    # Set larger font sizes
    plt.rcParams.update({
        'font.size': 14,          # General font size
        'axes.titlesize': 14,     # Font size for axes titles
        'axes.labelsize': 14,     # Font size for axes labels
        'xtick.labelsize': 14,    # Font size for x-axis tick labels
        'ytick.labelsize': 14,    # Font size for y-axis tick labels
        'legend.fontsize': 14,    # Font size for legend
        'figure.titlesize': 16    # Font size for figure title
    })

    axes = {
        'correlations_example': fig.add_subplot(gs[0:2, 0:3]),
        'distance_exp': fig.add_subplot(gs[0, 3]),
        'quality_exp': fig.add_subplot(gs[1, 3]),
        'shuffling_example': fig.add_subplot(gs[2, 0:3]),
        'cumulative_dist': fig.add_subplot(gs[2, 3])
    }
    
    # Plots with traces are going to share the same x-axis
    axes['correlations_example'].sharex(axes['shuffling_example'])

    return fig, axes

def process_session(row, results, plot, axes):
    """Processes a session to calculate mean correlations and return a DataFrame."""

    # Create a session object
    session = Session(row)

    # Load the data
    session.get_neuron_labels()
    dff_signal = session.load_dff_signal()
    neuron_positions = session.neuron_positions
    model = session.load_caiman_model()
    quality = model.estimates.r_values # values of spatial correlation

    # Check that positions len == dff_signal len
    if len(neuron_positions) != len(dff_signal):
        raise ValueError('Neuron positions and DFF signal have different number of neurons.')

    # Check that SNR_list and dff_signal have same number of neurons
    if len(quality) != len(dff_signal):
        raise ValueError('SNR list and DFF signal have different number of neurons.')
    
    # Compute correlations for true dataset (not shuffled)
    distances, cross_correlation_bin, __, correlated_pairs = compute_correlations(neuron_positions, dff_signal, THRESHOLD, 
                         bin_size_seconds=30, sampling_rate=30)
    
    # Get all the indecies pairs for cross_correlation_bin and flattened array of the corresponding values
    corr_indices_true = np.triu_indices(cross_correlation_bin.shape[0], k=1)
    # For each pair of corr_indices, compute average SNR for respective neurons
    corr_snr_true = np.array([np.nan if np.isnan(quality[corr_indices_true[0][i]]) or np.isnan(quality[corr_indices_true[1][i]]) or np.isinf(quality[corr_indices_true[0][i]]) or np.isinf(quality[corr_indices_true[1][i]]) else np.mean([quality[corr_indices_true[0][i]],quality[corr_indices_true[1][i]]]) for i in range(len(corr_indices_true[0]))])
    
    # Extract values from the matrix for plotting
    true_correlation_values = get_values(cross_correlation_bin)
    true_distances_values = get_values(distances)

    # Compute shuffled correlations
    shuffled_correlations_values_all, shuffled_distances_values_all, shuffled_quality_all = [], [], []

    for _ in range(100):
        # Shuffle the signal using 6 seconds of time bins
        shuffled_signal = shuffle_signal(dff_signal, bin_time_seconds=6)
        # For each shuffled signal, compute correlations in interval-based manner
        shuffled_distances, shuffled_correlation_matrix, __, __ = compute_correlations(neuron_positions, shuffled_signal, THRESHOLD, 
                         bin_size_seconds=30, sampling_rate=30)
        
        # Get all the indecies pairs for shuffled_correlation_matrix and flattened array of the corresponding values
        shuffled_corr_indices = np.triu_indices(shuffled_correlation_matrix.shape[0], k=1)
        shuffled_corr_snr = np.array([np.nan if np.isnan(quality[shuffled_corr_indices[0][i]]) or np.isnan(quality[shuffled_corr_indices[1][i]]) or np.isinf(quality[shuffled_corr_indices[0][i]]) or np.isinf(quality[shuffled_corr_indices[1][i]]) else np.mean([quality[shuffled_corr_indices[0][i]],quality[shuffled_corr_indices[1][i]]]) for i in range(len(shuffled_corr_indices[0]))])

        # Get values from correlation matrix
        shuffled_correlations_values = get_values(shuffled_correlation_matrix)
        shuffled_distances_values = get_values(shuffled_distances)

        # Add values to lists
        shuffled_correlations_values_all.extend(shuffled_correlations_values)
        shuffled_distances_values_all.extend(shuffled_distances_values)
        shuffled_quality_all.extend(shuffled_corr_snr)

    # Merge the list of shuffled correlations and distances 
    shuffled_correlations_values_all = np.array(shuffled_correlations_values_all)
    shuffled_distances_values_all = np.array(shuffled_distances_values_all)
    shuffled_quality_all = np.array(shuffled_quality_all)

    # Sample number of values from shuffled_correlations that is equal to number of true correlation values
    n_values = len(true_correlation_values)
    sample_idx = np.random.RandomState(0).permutation(len(shuffled_correlations_values_all))[:n_values]
    
    # Sample only selected values
    sampled_shuffled_correlations = [shuffled_correlations_values_all[i] for i in sample_idx]
    sampled_shuffled_distances = [shuffled_distances_values_all[i] for i in sample_idx]
    sampled_shuffled_quality = [shuffled_quality_all[i] for i in sample_idx]

    # Check if sampled shuffled correlations have same len as true correlations
    assert len(sampled_shuffled_correlations) == len(true_correlation_values)

    # Check that the shuffled and true values are not identical
    assert not np.array_equal(sampled_shuffled_correlations, true_correlation_values)

    if plot:
        # Convert to the format for plotting
        correlated_pairs_new = [(cell1, cell2) for cell1, cell2 in zip(correlated_pairs[0], correlated_pairs[1])]
        # plot_correlated_traces(ax=None, traces=accepted_dff_signal, component_pairs=correlated_pairs_new, distances=distances, liberation_cell_ids=liberation_cell_ids_mapped, correlation_matrix=correlation_matrix, title='Traces of highly correlated pairs (overlaid)', filename=fig_path, victim_lib_time=session.victim_lib_timecode * session.calcium_fps, fps=session.calcium_fps)
        plot_correlated_traces(ax=axes['correlations_example'], traces=dff_signal, component_pairs=correlated_pairs_new, distances=distances, correlation_matrix=cross_correlation_bin, fps=session.calcium_fps)
        
        # Remove x axis ticks and labels from correlations_example
        axes['correlations_example'].set_xticks([])
        axes['correlations_example'].set_xlabel(' ')

        
        axes['shuffling_example'].plot(dff_signal[0], color='k', label='Original signal', linewidth=1)
        axes['shuffling_example'].plot(shuffled_signal[0] + 10, color=shuffled_signal_color, label='Shuffled signal', linewidth=1)
        axes['shuffling_example'].legend(title_fontsize='large', fontsize='large')
        
        # Set xticks to be in seconds
        tick_locs = np.linspace(-20, dff_signal.shape[1], 20, endpoint=False).astype(int)
        axes['shuffling_example'].set_xticks(tick_locs)
        axes['shuffling_example'].set_xticklabels([f'{int(tick/session.calcium_fps):d}' for tick in tick_locs])
        axes['shuffling_example'].set_yticks([])
        axes['shuffling_example'].set_ylabel(' ')
        axes['shuffling_example'].set_xlabel('Time (s)')

        # # If signal > 200s, set axes lim to 0,200s
        # if dff_signal.shape[1] / session.calcium_fps > 200:
        #     axes['shuffling_example'].set_xlim(0, 200 * session.calcium_fps)
        # else:
        #     axes['shuffling_example'].set_xlim(0, dff_signal.shape[1])

        # Turn top and right spines off
        axes['shuffling_example'].spines['top'].set_visible(False)
        axes['shuffling_example'].spines['right'].set_visible(False)
        axes['shuffling_example'].spines['left'].set_visible(False)

    # Return a DataFrame with the results containning session + day of helping labels and list of correlation values for original and shuffled
    results['Session'].extend([session.session] * (2 * n_values))
    results['Mouse'].extend([session.animal_id] * (2 * n_values))
    results['Day of Helping'].extend([session.day_of_helping] * (2 * n_values))
    results['PCC'].extend(list(true_correlation_values) + list(sampled_shuffled_correlations))
    results['Distance'].extend(list(true_distances_values) + list(sampled_shuffled_distances))
    results['Spatial Correlation'].extend(list(corr_snr_true) + list(sampled_shuffled_quality))
    results['Label'].extend(['True'] * n_values + ['Shuffled'] * n_values)
    
    # Check the len of each list to make sure they are equal
    assert len(results['PCC']) == len(results['Label']) == len(results['Session']) == len(results['Day of Helping']) == len(results['Distance']) == len(results['Spatial Correlation'])
    return results

def main():
    # Set up the plot 
    fig, axes = setup_plot(FIG_SIZE, GRID_SPEC)

    # Load the session file
    session_df = pd.read_csv(SESSION_FILE)

    # Remove habituation sessions
    session_df = session_df[~session_df['Session'].str.contains('Hab')]
    # session_df = session_df[session_df['Mouse'] == 5592]

    # Reindex the DataFrame
    session_df = session_df.reset_index(drop=True)

    # Initialize the dictionary to store the results
    results = defaultdict(list)

    # Get random index for plotting 
    idx_to_plot = np.random.randint(0, len(session_df))
    idx_to_plot = 99
    # Process each session
    for i, row in tqdm(session_df.iterrows()):
        if i == idx_to_plot:
            print('Plotting session with index:', i)
            plot=True
        else:
            plot=False

        results = process_session(row, results, plot, axes)
        
    # Save the results
    results = pd.DataFrame(results)

    # Sample 1,000 rows only from true distribution (we want to plot how it is for the actual data)
    only_true_label = results[results['Label'] == 'True']
    only_true_label = results[results['Spatial Correlation'] > 0]
    sampled_results = only_true_label.sample(n=6000)

    # Plot PCC values dependence for Distance between pair of neurons and Spatial Correlation
    plot_scatter(x=sampled_results['Distance'].to_numpy(), y=sampled_results['PCC'].to_numpy(), ax=axes['distance_exp'], xlabel='Distance', ylabel='Pearson Correlation', perform_test=True, ylim=(-1, 1), c=scatter_color, significance_color=significance_color, s=2)
    plot_scatter(x=sampled_results['Spatial Correlation'].to_numpy(), y=sampled_results['PCC'].to_numpy(), ax=axes['quality_exp'], xlabel='Spatial Correlation', ylabel='Pearson Correlation', perform_test=True, ylim=(-1, 1), c=scatter_color, significance_color=significance_color, s=2)
    
    # Sample 10,000 random rows for cumulative distribution plot (to compare real PPC distribution and shuffled PPC distribution)
    sampled_results = results.sample(n=10000)
    # Remove nan
    sampled_results = sampled_results[sampled_results['PCC'].notna()]
    
    # Split the results into true and shuffled
    true_dist = sampled_results[sampled_results['Label'] == 'True']['PCC'].to_numpy()
    shuffled_dist = sampled_results[sampled_results['Label'] == 'Shuffled']['PCC'].to_numpy()

    # Plot and perform the KS test
    plot_ecdf_with_ks_test(true_dist, shuffled_dist, ax=axes['cumulative_dist'],
                       significance_color=significance_color, shuffled_signal_color=shuffled_signal_color)

    # Add letters A to E to the figure
    for ax, letter in zip(axes.values(), ['A', 'B', 'C', 'D', 'E']):
        ax.text(0.05, 1.1, f'{letter}', transform=ax.transAxes, verticalalignment='top', fontsize=20)

    # Tight layout
    fig.tight_layout()

    # Save the supplementary figure
    fig.savefig('figures/supplementary_functional_connectivity.pdf')
    # Save in svg and eps formats
    fig.savefig('figures/supplementary_functional_connectivity.svg')
    fig.savefig('figures/supplementary_functional_connectivity.eps')
    
if __name__ == "__main__":
    main()
