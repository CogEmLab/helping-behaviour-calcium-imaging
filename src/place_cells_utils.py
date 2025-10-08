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

def spatial_information(tuning_map, occupancy):
    r_i = tuning_map.flatten()
    p_i = occupancy.flatten()
    p_i = p_i / np.sum(p_i)
    valid = (p_i > 0) & (r_i > 0)
    r_bar = np.sum(r_i[valid] * p_i[valid])
    ratio = (r_i[valid] + 1e-12) / (r_bar + 1e-12)
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