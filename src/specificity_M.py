import logging
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from scipy import stats
from elephant.statistics import mean_firing_rate
import logging
import os

def calculate_fixed_threshold(signal):
    """Calculate fixed threshold based on signal noise level"""
    # Get baseline by taking lower portion of signal
    baseline = np.percentile(signal, 65)  
    noise_level = signal[signal <= baseline]
    
    # Calculate noise statistics
    noise_std = np.std(noise_level)
    noise_median = np.median(noise_level)
    
    # Set threshold above noise
    return noise_median + 2 * noise_std


def detect_peaks_with_features(signal, threshold, fps):
    """Detect peaks with prominence and width considerations"""

    # Convert signal and threshold to numpy arrays
    signal = np.array(signal)

    # Minimum peak requirements
    min_prominence = 0.35
    min_width = 0.4 * fps  # 400ms minimum width
    max_width = 10.0 * fps  # 2s maximum width
    min_distance = 0.25 * fps
  
    peaks, properties = find_peaks(
        signal,
        height=threshold,
        prominence=min_prominence,
        width=(min_width, max_width),
        distance=min_distance,
        rel_height=0.6
    )

    return peaks, properties

def calculate_auc_rate(signal_segment, peaks, threshold, fps):
    if len(peaks) == 0:
        return float(0) + 1e-10
        
    total_auc = 0
    peak_aucs = []  # Store individual peak AUCs

    fixed_threshold = threshold

    for peak in peaks:
        # Find left and right boundaries of the peak
        left_idx = peak
        right_idx = peak

        # Move left until signal drops below threshold or start of signal
        while left_idx > 0 and signal_segment[left_idx] > fixed_threshold:
            left_idx -= 1
            
        # Move right until signal drops below threshold or end of signal
        while right_idx < len(signal_segment)-1 and signal_segment[right_idx] > fixed_threshold:
            right_idx += 1
        
        # Calculate AUC only for values above noise_level
        peak_signal = signal_segment[left_idx:right_idx+1]
        peak_signal = np.where(peak_signal > fixed_threshold, peak_signal - fixed_threshold, 0)
        auc = np.trapz(peak_signal)
        peak_aucs.append(auc)
    
    # Calculate amplitude-weighted AUC
    if len(peak_aucs) > 0:
        peak_heights = signal_segment[peaks] - fixed_threshold  
        weights = peak_heights / np.max(peak_heights) if np.max(peak_heights) > 0 else np.ones_like(peak_heights)
        total_auc = np.sum(np.array(peak_aucs) * weights)

    # Calculate rate as AUC per second
    duration = len(signal_segment) / fps
    return total_auc / duration if duration > 0 else 0

def calculate_event_specificity(signal, event_window, event_duration, fps, plot=False):
    """
    Calculate the p-value for event specificity of a calcium imaging signal.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Starting calculation of event specificity.")

    # Normalize the signal
    signal = (signal - np.min(signal)) / (np.max(signal) - np.min(signal))

    # Extract signal segments
    event_signal = signal[event_window[0]:event_window[1]]
    background_signal = np.concatenate([
        signal[:event_window[0]],
        signal[event_window[1]:]
    ])

    # Calculate fixed threshold using noise statistics
    threshold = calculate_fixed_threshold(signal)
    
    background_peaks, _ = detect_peaks_with_features(background_signal, threshold, fps)
    
    # Filter peaks based on fixed threshold
    valid_peaks = background_peaks[background_signal[background_peaks] > threshold]
    logging.info("Detecting peaks in the background signal.")

    # Calculate background duration in seconds
    background_duration = len(background_signal) / fps
    
    # Split background into non-overlapping windows of event_duration length
    n_windows = int(background_duration // event_duration)
    if n_windows == 0:  # Ensure at least one window
        n_windows = 1
    
    # Calculate background rate for each window
    background_rates = []
    for i in range(n_windows):
        start_idx = int(i * event_duration * fps)
        end_idx = int((i + 1) * event_duration * fps)
        if end_idx > len(background_signal):
            end_idx = len(background_signal)
        
        window_signal = background_signal[start_idx:end_idx]
        window_peaks, _ = detect_peaks_with_features(window_signal, threshold, fps)
        valid_window_peaks = window_peaks[window_signal[window_peaks] > threshold]
        
        if len(valid_window_peaks) > 0:
            window_rate = calculate_auc_rate(window_signal, valid_window_peaks, threshold, fps)
            background_rates.append(window_rate)
    
    # Calculate mean background rate from windows
    if background_rates:
        lambda_background = np.mean(background_rates)
    else:
        lambda_background = float(0) + 1e-10

    # Detect event peaks
    event_peaks, _ = detect_peaks_with_features(event_signal, threshold, fps)
    valid_event_peaks = event_peaks[event_peaks < len(event_signal)]

    if len(valid_event_peaks) > 0:
        valid_event_peaks = valid_event_peaks[event_signal[valid_event_peaks] > threshold]
    else:
        valid_event_peaks = np.array([], dtype=int)

    # Calculate event rate as before
    event_rate = calculate_auc_rate(event_signal, valid_event_peaks, threshold, fps)

    # Calculate AUCs and p-value using window-based background rate
    expected_auc = lambda_background * event_duration    
    actual_auc = event_rate * event_duration
    
    # Add check for no peaks in event window
    if actual_auc == 0:
        # If no peaks in event window, p-value should be high
        k = 0
        p_value = stats.poisson.cdf(k, expected_auc)    
    else:
        k = int(np.ceil(actual_auc / lambda_background)) if lambda_background > 0 else 0
        p_value = 1 - stats.poisson.cdf(k - 1, expected_auc)

    logging.info(f"Calculated p-value: {p_value:.5f}.")

    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(signal, label='Signal', alpha=0.7)
        plt.axhline(y=threshold, linestyle='--', color='orange', label='Threshold')
        
        plt.scatter(valid_peaks[valid_peaks < event_window[0]], 
                   background_signal[valid_peaks[valid_peaks < event_window[0]]], 
                   color='red', label='Background Peaks')
        plt.scatter(valid_peaks[valid_peaks >= event_window[0]] + event_duration*fps, 
                   background_signal[valid_peaks[valid_peaks >= event_window[0]]], 
                   color='red')
        plt.scatter(valid_event_peaks + event_window[0], 
                   signal[valid_event_peaks + event_window[0]], 
                   color='blue', label='Event Peaks')
        plt.axvspan(event_window[0], event_window[1], 
                   color='yellow', alpha=0.3, label='Event Window')
        
        plt.title(f"Calcium Imaging Signal with Detected Peaks\np-value: {p_value:.5f}, Threshold: {threshold:.2f}")
        plt.xlabel("Frame")
        plt.ylabel("Signal Amplitude")
        plt.legend()

        output_dir = 'figures/specificity'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(os.path.join(output_dir, f'signal_example_{p_value:.5f}_{threshold:.2f}.png'), dpi=300)

    return p_value, k


if __name__ == "__main__":
    # Example usage
    signal = np.random.normal(0, 1, 1000)  # Example signal
    signal[450:550] += 2  # Add an "event"
    event_window = (450, 550)
    event_duration = 1.0  # Assuming each sample represents 0.01 time units

    p_value = calculate_event_specificity(signal, event_window, event_duration, 100, plot=True)
    print(f"P-value: {p_value}")

