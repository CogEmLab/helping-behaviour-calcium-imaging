import numpy as np
import pandas as pd
import os
from caiman.source_extraction.cnmf.cnmf import load_CNMF
from scipy.stats import binned_statistic_2d, wilcoxon
import matplotlib.pyplot as plt
from src.plotting import plot_boxplots
from src.place_cells_utils import *
from src.place_cells_plotting import *
from tqdm import tqdm
import yaml
import json
# don't show any plots
import matplotlib as mpl
mpl.use('Agg')

# Experiment parameters

# Load configuration
with open("tech_docs/place_cells_pipeline_params.yml", "r") as f:
    config = yaml.safe_load(f)

# Extract parameters
arena_size = tuple(config["arena_size"])
n_bins_x = config["n_bins_x"]
n_bins_y = config["n_bins_y"]
n_shuffles = config["n_shuffles"]
min_entries = config["min_entries"]
quiet_window = config["quiet_window"]
quiet_var_thresh = config["quiet_var_thresh"]
k_thresh = config["k_thresh"]
session_file = config["session_file"]
output_folder = config["output_folder"]

# Create results folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

output_summary_file = os.path.join(output_folder, "place_cell_summary.csv")
plot_output_path = os.path.join(output_folder, "place_cell_boxplots.png")
scatter_output_path = os.path.join(output_folder, "place_cell_scatter.png")
temp_output_path = os.path.join(output_folder, "temp_summary.json")

# Load session metadata
try:
    sessions = pd.read_csv(session_file)
    sessions = sessions[~sessions['cnmfe model path'].isna() & ~sessions['DLC results'].isna()]
except pd.errors.EmptyDataError:
    print(f"Error: Unable to read session file at {session_file}. Skipping...")
    exit()

def compute_place_cells(
    signal_data, x_data, y_data, occupancy, entry_map, arena_size, n_bins_x, n_bins_y, output_dir, n_shuffles=1000, min_entries=5
):
    """
    Compute place cells based on spatial information and shuffle test.

    Args:
        signal_data (ndarray): shape (n_neurons, n_timepoints)
        x_data, y_data (ndarray): position coordinates
        occupancy (ndarray): occupancy map (normalized)
        arena_size (tuple): (width, height) in cm
        n_bins_x, n_bins_y (int): spatial bin counts
        n_shuffles (int): number of temporal shuffles

    Returns:
        info_scores (ndarray): spatial information per neuron
        shuffled_info (ndarray): shuffled info scores [n_neurons, n_shuffles]
        place_cells (list[int]): indices of neurons classified as place cells
    """
    n_neurons, n_timepoints = signal_data.shape
    tuning_maps = []
    
    # Compute spatial tuning maps
    for n in range(n_neurons):
        stat, _, _, _ = binned_statistic_2d(
            x_data, y_data, signal_data[n, :],
            statistic='mean', bins=[n_bins_x, n_bins_y],
            range=[[0, arena_size[0]], [0, arena_size[1]]]
        )
        tuning_maps.append(stat)

    # Compute spatial information per neuron
    info_scores = [
        spatial_information(tm, occupancy, entry_map=entry_map, min_entries=min_entries)
        for tm in tuning_maps
    ]

    # Shuffle-based null distribution
    shuffled_info = np.zeros((n_neurons, n_shuffles))
    for n in range(n_neurons):
        for s in range(n_shuffles):
            shift = np.random.randint(100, n_timepoints - 100)
            shuffled = np.roll(signal_data[n, :], shift)
            stat, _, _, _ = binned_statistic_2d(
                x_data, y_data, shuffled,
                statistic='mean', bins=[n_bins_x, n_bins_y],
                range=[[0, arena_size[0]], [0, arena_size[1]]]
            )
            shuffled_info[n, s] = spatial_information(stat, occupancy, entry_map=entry_map, min_entries=min_entries)

        if n == 0:
            os.makedirs(output_dir, exist_ok=True)
            plt.figure(figsize=(6, 4))
            plt.hist(shuffled_info[0], bins=30, alpha=0.7, label='Shuffled')
            plt.axvline(info_scores[0], color='red', linestyle='--', label='Observed')
            plt.xlabel('Spatial Information (bits/event)')
            plt.ylabel('Count')
            plt.title(f'Shuffled Distribution (Neuron 0)')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "shuffled_distribution.png"), dpi=300)
            plt.close()

    # Determine place cells (p < 0.05)
    place_cells = [
        n for n in range(n_neurons)
        if np.mean(shuffled_info[n] >= info_scores[n]) < 0.01
    ]

    return info_scores, shuffled_info, place_cells

def preprocess_spikes(C, S, x_data, quiet_window, quiet_var_thresh, k_thresh):
    n_neurons = S.shape[0]
    S_binary_data = np.zeros_like(S)
    for n in range(n_neurons):
        quiet_mask = get_quiet_period_mask(C[n, :len(x_data)], window=quiet_window, var_thresh=quiet_var_thresh)
        binary_S, _ = threshold_and_binarize(S[n, :], quiet_mask, k=k_thresh)
        S_binary_data[n, :] = binary_S
    return S_binary_data

# Initialize summary storage
summary_rows = []

# Loop through sessions
for idx, row in tqdm(sessions.iterrows()):
    cnmfe_path = str(row['cnmfe model path'])
    movement_path = str(row['DLC results'])

    # if row['cohort'] != 'Oct_2021':
    #     continue
    
    # Check that both are valid paths
    if not os.path.isfile(cnmfe_path) or not os.path.isfile(movement_path):
        print(f"Skipping session {row['session']} due to invalid paths")
        continue
    
    session_output_dir = os.path.join(output_folder, f"{row['mouse']}_{row['session']}")
    os.makedirs(session_output_dir, exist_ok=True)

    # Load movement data
    movement_data = pd.read_csv(movement_path)
    x_data = movement_data['X Center'].to_numpy()
    y_data = movement_data['Y Center'].to_numpy()

    # merge to one array
    # movement_data = np.column_stack((x_data, y_data))

    # Load CNMFe model
    cnmfe_model = load_CNMF(cnmfe_path)
    
    # Set A_thr field to None before selecting components (threshold step of reevaluate_comps creates a dictionary of different shape of A)
    cnmfe_model.estimates.A_thr = None
    print("Set A_thr to None.")

    # Select only the accepted components
    cnmfe_model.estimates.select_components(use_object=True, save_discarded_components=True)
    
    # Get dF/F; detrend only (1p protocol)
    cnmfe_model.estimates.detrend_df_f(detrend_only=True)
    
    # Get dF/F
    df_f = cnmfe_model.estimates.F_dff
    S_data = cnmfe_model.estimates.S # inferred spikes
    n_neurons, n_timepoints = df_f.shape

    # Check that data are not empty
    if n_neurons == 0 or n_timepoints == 0:
        print(f"Skipping session {row['session']} due to empty data")
        continue

    # Crop calcium data according to the movement data shape 1
    df_f = df_f[:, :len(x_data)]
    S_data = S_data[:, :len(x_data)]

    print('Data for session {}, cohort {} and animal {} loaded successfully. Data shapes are: df_f ({}, {}), S ({}, {}), x_data {}, y_data {}'.format(row['session'], row['cohort'], row['mouse'], df_f.shape[0], df_f.shape[1], S_data.shape[0], S_data.shape[1], len(x_data), len(y_data)))

    # Compute occupancy map
    occupancy, _, _, _ = binned_statistic_2d(
        x_data, y_data, None, statistic='count', 
        bins=[n_bins_x, n_bins_y], range=[[0, arena_size[0]], [0, arena_size[1]]]
    )
    occupancy = occupancy / np.sum(occupancy)

    # Compute entries
    entry_map = compute_entries(x_data, y_data, n_bins_x, n_bins_y, arena_size=arena_size)

     # --- Plot occupancy and entry maps ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    im0 = axes[0].imshow(occupancy.T, origin='lower', cmap='viridis')
    axes[0].set_title("Occupancy map")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(entry_map.T, origin='lower', cmap='magma')
    axes[1].set_title("Entry map")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.set_xlabel('X bins')
        ax.set_ylabel('Y bins')

    plt.tight_layout()
    plt.savefig(os.path.join(session_output_dir, "occupancy_entry_maps.png"), dpi=300)
    plt.close()

    print('Computing place cells using dF/F...')
    # --- ΔF/F ---
    df_info_scores, df_shuffled_info, df_place_cells = compute_place_cells(
        df_f, x_data, y_data, occupancy, entry_map, arena_size, n_bins_x, n_bins_y, os.path.join(session_output_dir, "dfF"), n_shuffles, min_entries
    )

    plot_tuning_maps(
    signal_data=df_f,
    x_data=x_data,
    y_data=y_data,
    place_cells=df_place_cells,
    scores=df_info_scores,
    arena_size=arena_size,
    n_bins_x=n_bins_x,
    n_bins_y=n_bins_y,
    output_dir=session_output_dir,
    title_prefix=f"{row['mouse']}_{row['session']}_dfF"
)

    # --- Spikes ---
    S_binary_data = preprocess_spikes(cnmfe_model.estimates.C, S_data, x_data, quiet_window, quiet_var_thresh, k_thresh)
    print('Computing place cells using inferred spikes...')
    S_info_scores, S_shuffled_info, S_place_cells = compute_place_cells(
        S_binary_data, x_data, y_data, occupancy, entry_map, arena_size, n_bins_x, n_bins_y, os.path.join(session_output_dir, "spikes"), n_shuffles, min_entries
    )


    if len(S_place_cells) > 0:

        # Plot tuning maps
        plot_tuning_maps(
        signal_data=S_binary_data,
        x_data=x_data,
        y_data=y_data,
        place_cells=S_place_cells,
        scores=S_info_scores,
        arena_size=arena_size,
        n_bins_x=n_bins_x,
        n_bins_y=n_bins_y,
        output_dir=session_output_dir,
        title_prefix=f"{row['mouse']}_{row['session']}_spikes")

        # --- Trajectories with spike events for each place cell ---
        fig, axes = plt.subplots(
            nrows=int(np.ceil(len(S_place_cells) / 4)), ncols=4,
            figsize=(12, 3 * int(np.ceil(len(S_place_cells) / 4)))
        )
        axes = axes.flatten()

        for i, n in enumerate(S_place_cells):
            spike_frames = np.where(S_binary_data[n, :] > 0)[0]
            axes[i].plot(x_data, y_data, color='lightgray', lw=0.5)
            axes[i].scatter(x_data[spike_frames], y_data[spike_frames], c='r', s=2)
            axes[i].set_title(f'Place cell {n} - Info: {S_info_scores[n]:.2f}')
            axes[i].invert_yaxis()  # optional if your coordinate system is flipped
            axes[i].axis('off')

        for j in range(i+1, len(axes)):
            axes[j].axis('off')

        fig.suptitle(f"Spike Trajectories - Mouse {row['mouse']}, Session {row['session']}")
        plt.tight_layout()
        plt.savefig(os.path.join(session_output_dir, "spike_trajectories.png"), dpi=300)
        plt.close()

    plot_si_histogram(
        df_info_scores=df_info_scores,
        S_info_scores=S_info_scores,
        output_dir=session_output_dir,
        title_prefix=f"{row['mouse']}_{row['session']}"
    )

    # --- Paired statistical test on info scores ---
    # Wilcoxon signed-rank test for paired comparisons
    try:
        stat_test, p_info = wilcoxon(df_info_scores, S_info_scores)
    except ValueError:
        p_info = np.nan

    # --- Save summary statistics ---
    summary_rows.append({
        'session': row['session'],
        'cohort': row['cohort'],
        'mouse': row['mouse'],
        'total_neurons': n_neurons,
        'df_place_cells_count': len(df_place_cells),
        'df_place_cells_percent': 100 * len(df_place_cells)/n_neurons,
        'S_place_cells_count': len(S_place_cells),
        'S_place_cells_percent': 100 * len(S_place_cells)/n_neurons,
        'df_mean_info': np.mean(df_info_scores),
        'S_mean_info': np.mean(S_info_scores),
        'info_p_value': p_info,
        'session_length_s': row.get('session length(s)', np.nan),
        'latency_min': row.get('latencyminconverted', np.nan)
    })

    print(f"Processed session {row['session']} successfully. Place cells: dF/F: {len(df_place_cells)}, inferred spikes: {len(S_place_cells)}")
    # Save current results to temporary txt file
    with open(temp_output_path, 'w') as f:
        f.write(json.dumps(summary_rows))
    
# --- Convert summary to DataFrame and save ---
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(output_summary_file, index=False)
print(f"Saved summary to {output_summary_file}")

# --- Boxplot comparison of % place cells ---
fig, ax = plt.subplots(figsize=(6,6))
plot_boxplots(summary_df['df_place_cells_percent'].to_numpy(),
              summary_df['S_place_cells_percent'].to_numpy(),
              ax, title='Place cells: ΔF/F vs inferred spikes',
              labels=('ΔF/F', 'Spikes'), ylabel='% of neurons')
plt.tight_layout()
plt.savefig(plot_output_path)
plt.show()
print(f"Saved boxplot to {plot_output_path}")

# --- Scatter plot of ΔF/F vs spike place cells ---
fig, ax = plt.subplots(figsize=(6,6))
ax.scatter(summary_df['df_place_cells_percent'], summary_df['S_place_cells_percent'], alpha=0.7)
ax.plot([0,100],[0,100],'r--', lw=1)  # identity line
ax.set_xlabel('ΔF/F % place cells')
ax.set_ylabel('Spikes % place cells')
ax.set_title('Per-session comparison of place cell detection')
plt.tight_layout()
plt.savefig(scatter_output_path)
plt.show()
print(f"Saved scatter plot to {scatter_output_path}")
