import numpy as np
from scipy.stats import binned_statistic_2d
from scipy.ndimage import uniform_filter
from statsmodels.stats.multitest import multipletests
import time
from src.place_cells_utils import *
import numpy as np
from scipy.stats import binned_statistic_2d
from scipy.ndimage import uniform_filter
from statsmodels.stats.multitest import multipletests
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix

def simulate_data(n_neurons, n_place_cells, duration_s=600, sampling_rate=10):
    """Generates simulated behavioral and neural data (Unchanged)."""
    # ... (code from previous response, no changes needed here) ...
    arena_size = (100, 100); n_bins = 20; n_timepoints = duration_s * sampling_rate
    x_pos, y_pos = np.zeros(n_timepoints), np.zeros(n_timepoints)
    x_pos[0], y_pos[0] = arena_size[0] / 2, arena_size[1] / 2
    for t in range(1, n_timepoints):
        x_pos[t] = np.clip(x_pos[t-1] + np.random.randn() * 2, 0, arena_size[0])
        y_pos[t] = np.clip(y_pos[t-1] + np.random.randn() * 2, 0, arena_size[1])
    occupancy, _, _, _ = binned_statistic_2d(x_pos, y_pos, None, statistic='count', bins=n_bins, range=[[0, arena_size[0]], [0, arena_size[1]]])
    occupancy = occupancy / sampling_rate
    bin_x = np.clip(np.floor(x_pos / (arena_size[0] / n_bins)).astype(int), 0, n_bins - 1)
    bin_y = np.clip(np.floor(y_pos / (arena_size[1] / n_bins)).astype(int), 0, n_bins - 1)
    entry_map = np.zeros((n_bins, n_bins))
    for t in range(1, n_timepoints):
        if bin_x[t] != bin_x[t-1] or bin_y[t] != bin_y[t-1]:
            entry_map[bin_y[t], bin_x[t]] += 1
    signal_data = np.zeros((n_neurons, n_timepoints))
    ground_truth_pcs = list(range(n_place_cells))
    place_field_centers = np.random.rand(n_place_cells, 2) * arena_size
    field_size = 15
    for n in range(n_neurons):
        if n in ground_truth_pcs:
            dist = np.sqrt((x_pos - place_field_centers[n, 0])**2 + (y_pos - place_field_centers[n, 1])**2)
            firing_prob = 0.5 * np.exp(-dist**2 / (2 * field_size**2))
        else:
            firing_prob = np.full(n_timepoints, 0.01)
        signal_data[n, :] = np.random.poisson(firing_prob)
    return signal_data, x_pos, y_pos, occupancy, entry_map, arena_size, n_bins, ground_truth_pcs


def plot_evaluation_metrics(p_values, ground_truth_labels, title_prefix=""):
    """Generates and plots the ROC curve and confusion matrix."""
    
    # --- 1. ROC Curve ---
    # The "score" for the ROC curve is how likely a cell is a place cell.
    # A low p-value means it's MORE likely to be a place cell. So we use 1 - p_value.
    fpr, tpr, thresholds = roc_curve(ground_truth_labels, np.subtract(1, p_values))
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(14, 6))
    
    # Plot ROC
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Chance')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title(f'{title_prefix} ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.6)

    # --- 2. Confusion Matrix ---
    # Using the FDR-corrected p-values with alpha=0.01 for a fixed decision point
    reject, pvals_corrected, _, _ = multipletests(p_values, alpha=0.01, method='fdr_bh')
    predicted_labels = reject.astype(int)

    cm = confusion_matrix(ground_truth_labels, predicted_labels)
    
    plt.subplot(1, 2, 2)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Noise', 'Place Cell'],
                yticklabels=['Noise', 'Place Cell'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'{title_prefix} Confusion Matrix (alpha=0.01)')
    
    plt.tight_layout()
    plt.savefig(f'figures/evaluation_metrics_{title_prefix}.png', dpi=300)

if __name__ == "__main__":
    
    # --- Test Case 1: Mixed population ---
    print("\n" + "="*50)
    print(">>> Test Case 1: Mixed Population (1000 neurons, 200 are place cells)")
    print("="*50)
    
    output_dir=None
    N_NEURONS_1, N_PCS_1 = 1000, 200
    signal_1, x, y, occ, entry, size, n_bins, truth_idx_1 = simulate_data(N_NEURONS_1, N_PCS_1)
    
    # Create ground truth labels (1 for place cell, 0 for noise cell)
    truth_labels_1 = np.zeros(N_NEURONS_1)
    truth_labels_1[truth_idx_1] = 1
    
    start_time = time.time()

    _, p_vals_1, _, place_cells_pred = compute_place_cells_fast(
        signal_1, x, y, occ, entry, size, n_bins, n_bins, output_dir, n_shuffles=200, min_entries=3
    )
    
    end_time = time.time()

    print(f"\n--- Results for Test Case 1 (took {end_time - start_time:.2f}s) ---")
    print(f"Ground truth place cells: {truth_idx_1}")
    print(f"Function identified place cells: {place_cells_pred}")
    
    # Validate results
    correctly_identified = set(truth_idx_1).intersection(set(place_cells_pred))
    missed = set(truth_idx_1) - set(place_cells_pred)
    false_positives = set(place_cells_pred) - set(truth_idx_1)

    print(f"\nAnalysis:")
    print(f"  - Correctly identified: {len(correctly_identified)}/{N_PCS_1} ({list(correctly_identified)})")
    print(f"  - Missed (False Negatives): {len(missed)}/{N_PCS_1} ({list(missed)})")
    print(f"  - False Positives: {len(false_positives)}/{N_NEURONS_1 - N_PCS_1} ({list(false_positives)})")


    plot_evaluation_metrics(p_vals_1, truth_labels_1, title_prefix="Mixed Population")
    
    # # --- Test Case 2: Noise-only population ---
    # print("\n" + "="*50)
    # print(">>> Test Case 2: Noise-Only Population (50 neurons, 0 place cells)")
    # print("="*50)

    # N_NEURONS_2, N_PCS_2 = 1000, 0
    # signal_2, x, y, occ, entry, size, n_bins, truth_idx_2 = simulate_data(N_NEURONS_2, N_PCS_2)
    # truth_labels_2 = np.zeros(N_NEURONS_2)
    
    # start_time = time.time()
    # _, p_vals_2, _, place_cells_2_pred = compute_place_cells_fast(
    #     signal_2, x, y, occ, entry, size, n_bins, n_bins, output_dir, n_shuffles=200, min_entries=3
    # )

    # end_time = time.time()

    # print(f"\n--- Results for Test Case 1 (took {end_time - start_time:.2f}s) ---")
    # print(f"Ground truth place cells: {truth_idx_2}")
    # print(f"Function identified place cells: {place_cells_2_pred}")
    
    # # Validate results
    # correctly_identified = set(truth_idx_2).intersection(set(place_cells_2_pred))
    # missed = set(truth_idx_2) - set(place_cells_2_pred)
    # false_positives = set(place_cells_2_pred) - set(truth_idx_2)

    # print(f"\nAnalysis:")
    # print(f"  - Correctly identified: {len(correctly_identified)}/{N_PCS_2} ({list(correctly_identified)})")
    # print(f"  - Missed (False Negatives): {len(missed)}/{N_PCS_2} ({list(missed)})")
    # print(f"  - False Positives: {len(false_positives)}/{N_NEURONS_2 - N_PCS_2} ({list(false_positives)})")


    # # plot_evaluation_metrics(p_vals_2, truth_labels_2, title_prefix="Noise-Only Population")