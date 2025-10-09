import numpy as np
import pandas as pd
import os
from caiman.source_extraction.cnmf.cnmf import load_CNMF
from scipy.stats import binned_statistic_2d, stats
import matplotlib.pyplot as plt
from src.plotting import plot_boxplots


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

def spatial_information(tuning_map, occupancy, entry_map=None, min_entries=2):
    """
    Compute spatial information (SI) in bits per event for one neuron,
    applying Gaussian smoothing and minimum entry filtering.
    """

    # Flatten maps
    r_i = tuning_map.flatten()
    p_i = occupancy.flatten()

    # Normalize occupancy to be probability distribution
    p_i = p_i / np.sum(p_i)

    # If entry map is given, mask bins with < min_entries
    if entry_map is not None:
        entry_mask = (entry_map.flatten() >= min_entries)
    else:
        entry_mask = np.ones_like(r_i, dtype=bool)

    # Mask bins with sufficient occupancy, activity, and entries
    valid = (p_i > 0) & (r_i > 0) & entry_mask

    if not np.any(valid):
        return np.nan

    # Mean firing rate across visited bins
    r_bar = np.sum(r_i[valid] * p_i[valid])

    # Avoid log(0)
    ratio = (r_i[valid] + 1e-12) / (r_bar + 1e-12)

    # Skaggs spatial information
    info = np.sum(p_i[valid] * ratio * np.log2(ratio))
    return info

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