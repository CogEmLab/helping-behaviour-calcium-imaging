import numpy as np
import pandas as pd
import os
from caiman.source_extraction.cnmf.cnmf import load_CNMF
from scipy.stats import binned_statistic_2d, stats
import matplotlib.pyplot as plt
from src.plotting import plot_boxplots
from statsmodels.stats.multitest import multipletests
from scipy.ndimage.filters import uniform_filter
from joblib import Parallel, delayed
from scipy.ndimage import convolve
from scipy.stats import binned_statistic_2d
from statsmodels.stats.multitest import multipletests
import numpy as np

# Experiment parameters
quiet_window = 200
quiet_var_thresh = 0.5
k_thresh = 2.5  # threshold multiplier for binarization

def compute_entries(x, y, n_bins_x, n_bins_y, arena_size):
    """
    Count how many times the animal entered each spatial bin.
    An entry is defined as a transition from a different bin index.
    """
    # Bin indices for each timepoint
    x_bins = np.digitize(x, bins=np.linspace(0, arena_size[0], n_bins_x + 1)) - 1
    y_bins = np.digitize(y, bins=np.linspace(0, arena_size[1], n_bins_y + 1)) - 1

    # Mask invalid positions
    valid = (x_bins >= 0) & (x_bins < n_bins_x) & (y_bins >= 0) & (y_bins < n_bins_y)
    x_bins, y_bins = x_bins[valid], y_bins[valid]

    # Initialize entry map
    entries = np.zeros((n_bins_x, n_bins_y))

    # Count transitions: when (x_i, y_i) != (x_{i-1}, y_{i-1})
    for i in range(1, len(x_bins)):
        if (x_bins[i] != x_bins[i - 1]) or (y_bins[i] != y_bins[i - 1]):
            entries[x_bins[i], y_bins[i]] += 1

    return entries

import numpy as np

def spatial_information(tuning_map, occupancy, entry_map=None, min_entries=2, return_mask=False, occupancy_threshold=0, firing_rate_threshold=0):
    """
    Compute Skaggs spatial information (SI) in bits per event for one neuron,
    robustly handling empty bins, smoothing artifacts, and minimum entry filtering.

    Parameters
    ----------
    tuning_map : np.ndarray
        Neuron firing rate map (2D spatial bins).
    occupancy : np.ndarray
        Time spent in each spatial bin (same shape as tuning_map).
    entry_map : np.ndarray, optional
        Number of entries per bin. Bins with fewer than min_entries are ignored.
    min_entries : int, default=2
        Minimum entries required per bin to include it in the calculation.
    return_mask : bool, default=False
        If True, also return the boolean mask of valid bins.
    occupancy_threshold : float, default=0
        Minimum occupancy required per bin to include it in the calculation.
    firing_rate_threshold : float, default=0
        Minimum firing rate required per bin to include it in the calculation.

    Returns
    -------
    info : float
        Spatial information in bits per event. Returns np.nan if insufficient data.
    valid : np.ndarray (optional)
        Boolean mask of valid bins used for calculation.

    Notes
    -----
    Spatial information is computed using the following formula:
        info = sum(p_i * log2(r_i / p_i)) where r_i is the firing rate in each bin
        and p_i is the occupancy probability in each bin.
    """

    # Flatten 2D maps to 1D for convenience
    r_i = tuning_map.flatten()
    p_i = occupancy.flatten()

    # Assert occupancy is probability distribution
    total_occupancy = np.sum(p_i)

    assert np.isclose(total_occupancy, 1.0, atol=1e-6), \
            f"Occupancy must be normalized before calling spatial_information(). Got sum={total_occupancy}"

    # Entry map filtering
    if entry_map is not None:
        entry_mask = entry_map.flatten() >= min_entries
    else:
        entry_mask = np.ones_like(r_i, dtype=bool)  # Include all bins

    # Ensure firing rates are non-negative (smoothing may produce tiny negatives)
    r_i = np.maximum(r_i, 0)

    # Define valid bins: sufficient occupancy, positive activity, enough entries
    valid = (p_i > occupancy_threshold) & (r_i > firing_rate_threshold) & entry_mask #! implement average firing rate per entry plot the map

    if not np.any(valid):
        # No bins meet criteria → cannot compute SI
        return (np.nan, valid) if return_mask else np.nan

    # Compute mean firing rate across valid bins
    r_bar = np.sum(r_i[valid] * p_i[valid]) 

    # Compute Skaggs spatial information
    # Add tiny epsilon to avoid division by zero or log(0)
    eps = 1e-12
    ratio = (r_i[valid] + eps) / (r_bar + eps)
    info = np.sum(p_i[valid] * ratio * np.log2(ratio))

    if return_mask:
        return info, valid
    return info

def preprocess_spikes(C, S, x_data, quiet_window, quiet_var_thresh, k_thresh):
    n_neurons = S.shape[0]
    S_binary_data = np.zeros_like(S)
    for n in range(n_neurons):
        quiet_mask = get_quiet_period_mask(C[n, :len(x_data)], window=quiet_window, var_thresh=quiet_var_thresh)
        binary_S, _ = threshold_and_binarize(S[n, :], quiet_mask, k=k_thresh)
        S_binary_data[n, :] = binary_S
    return S_binary_data

def compute_place_cells(
    signal_data, x_data, y_data, occupancy, entry_map, arena_size, n_bins_x, n_bins_y, output_dir, n_shuffles=1000, min_entries=5, smoothing_filter=2, alpha=0.05, occupancy_threshold=0, firing_rate_threshold=0
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
        min_entries (int): minimum number of entries per bin
        smoothing_filter (int): smoothing filter size
        alpha (float): significance threshold

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
        
        # replace nan with 0
        stat[np.isnan(stat)] = 0
        stat_smooth = uniform_filter(stat, size=smoothing_filter)

        tuning_maps.append(stat_smooth)

    # Compute spatial information per neuron
    info_scores = [
        spatial_information(tm, occupancy, entry_map=entry_map, min_entries=min_entries, occupancy_threshold=occupancy_threshold, firing_rate_threshold=firing_rate_threshold)
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
            
            # replace nan with 0
            stat[np.isnan(stat)] = 0
            stat_smooth = uniform_filter(stat, size=smoothing_filter)

            shuffled_info[n, s] = spatial_information(stat_smooth, occupancy, entry_map=entry_map, min_entries=min_entries, occupancy_threshold=occupancy_threshold, firing_rate_threshold=firing_rate_threshold)

    # Compute raw p-values
    p_values = [np.mean(shuffled_info[n] >= info_scores[n]) for n in range(n_neurons)]

    # Apply FDR correction
    reject, pvals_corrected, _, _ = multipletests(p_values, alpha=alpha, method='fdr_bh')

    # Select place cells
    place_cells = [n for n, r in enumerate(reject) if r and not np.isnan(info_scores[n])]

    # return info_scores, shuffled_info, place_cells
    return info_scores, p_values, pvals_corrected, place_cells


def compute_for_neuron(n, signal_data, occupancy, entry_map, n_bins_x, n_bins_y, n_shuffles, min_entries, kernel, valid_mask, flat_idx, n_bins, n_timepoints, occupancy_threshold, firing_rate_threshold):
    # --- Compute tuning map via bincount (fast) ---
    signal = signal_data[n, valid_mask]
    sum_per_bin = np.bincount(flat_idx, weights=signal, minlength=n_bins)
    count_per_bin = np.bincount(flat_idx, minlength=n_bins)
    mean_map = np.divide(sum_per_bin, count_per_bin, out=np.zeros_like(sum_per_bin), where=count_per_bin > 0)
    mean_map = mean_map.reshape((n_bins_x, n_bins_y))

    # --- Smoothing ---
    mean_map = convolve(mean_map, kernel, mode='reflect')

    # --- Compute real spatial info ---
    info_real = spatial_information(mean_map, occupancy, entry_map=entry_map, min_entries=min_entries, occupancy_threshold=occupancy_threshold, firing_rate_threshold=firing_rate_threshold)

    # --- Shuffles ---
    shuffled_vals = np.zeros(n_shuffles)
    for s in range(n_shuffles):
        # shift = min(100, )

        

        if n_timepoints - 100 > 100:
            shift = np.random.randint(100, n_timepoints - 100)
        else:
            half_signal = n_timepoints // 2
            shift = half_signal

        # print('shift: ', shift)
        rolled = np.roll(signal_data[n, :], shift)[valid_mask]
        sum_per_bin = np.bincount(flat_idx, weights=rolled, minlength=n_bins)
        count_per_bin = np.bincount(flat_idx, minlength=n_bins)
        mean_map_s = np.divide(sum_per_bin, count_per_bin, out=np.zeros_like(sum_per_bin), where=count_per_bin > 0)
        mean_map_s = mean_map_s.reshape((n_bins_x, n_bins_y))
        mean_map_s = convolve(mean_map_s, kernel, mode='reflect')
        shuffled_vals[s] = spatial_information(mean_map_s, occupancy, entry_map=entry_map, min_entries=min_entries, occupancy_threshold=occupancy_threshold, firing_rate_threshold=firing_rate_threshold)

    return info_real, shuffled_vals

def compute_place_cells_fast(
    signal_data, x_data, y_data, occupancy, entry_map, arena_size,
    n_bins_x, n_bins_y, output_dir=None, n_shuffles=1000,
    min_entries=5, smoothing_filter=2, n_jobs=-1, alpha=0.05, occupancy_threshold=0, firing_rate_threshold=0
):
    """
    Optimized version of compute_place_cells for speed.
    """
    n_neurons, n_timepoints = signal_data.shape
    info_scores = np.zeros(n_neurons)
    shuffled_info = np.zeros((n_neurons, n_shuffles))

    # --- Precompute bin indices (faster than binned_statistic_2d) ---
    x_edges = np.linspace(0, arena_size[0], n_bins_x + 1)
    y_edges = np.linspace(0, arena_size[1], n_bins_y + 1)
    x_bin = np.digitize(x_data, x_edges) - 1
    y_bin = np.digitize(y_data, y_edges) - 1
    valid_mask = (x_bin >= 0) & (x_bin < n_bins_x) & (y_bin >= 0) & (y_bin < n_bins_y)
    flat_idx = x_bin[valid_mask] * n_bins_y + y_bin[valid_mask]
    n_bins = n_bins_x * n_bins_y

    kernel = np.ones((smoothing_filter, smoothing_filter)) / (smoothing_filter ** 2)

    # Parallelize across neurons
    results = Parallel(n_jobs=n_jobs)(delayed(compute_for_neuron)(n, signal_data, occupancy, entry_map, n_bins_x, n_bins_y, n_shuffles, 
                                                                  min_entries, kernel, valid_mask, flat_idx, n_bins, n_timepoints, occupancy_threshold, firing_rate_threshold) for n in range(n_neurons))
    for n, (info_real, shuf) in enumerate(results):
        info_scores[n] = info_real
        shuffled_info[n, :] = shuf

    # --- Statistical test (FDR) ---
    p_values = [np.mean(shuffled_info[n] >= info_scores[n]) for n in range(n_neurons)]
    reject, pvals_corrected, _, _ = multipletests(p_values, alpha=alpha, method='fdr_bh')
    place_cells = [n for n, r in enumerate(reject) if r and not np.isnan(info_scores[n])]

    return info_scores, p_values, pvals_corrected, place_cells

def get_quiet_period_mask(C, window=quiet_window, var_thresh=quiet_var_thresh):
    T = len(C)
    mask = np.zeros(T, dtype=bool)
    half = window // 2
    window_vars = np.full(T, np.nan)
    for t in range(half, T-half):
        w = C[t-half : t+half]
        window_vars[t] = np.var(w)
    valid_vars = window_vars[half:T-half]
    med_var = np.median(valid_vars)
    mask = window_vars < (var_thresh * med_var)
    return mask

def threshold_and_binarize(S, quiet_mask, k=k_thresh):
    if not np.any(quiet_mask):
        mu = np.mean(S)
        sigma = np.std(S)
    else:
        mu = np.mean(S[quiet_mask])
        sigma = np.std(S[quiet_mask])
    thresh = mu + k * sigma
    binary = (S > thresh).astype(int)
    return binary, thresh

