import logging
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from scipy import stats
from elephant.statistics import mean_firing_rate
import logging

def calculate_event_specificity(signal, event_window, event_duration, fps, noise_level, plot=False):
    """
    Calculate the p-value for event specificity of a calcium imaging signal.

    Parameters:
    signal (array-like): The full calcium imaging signal. This is the time-series data of calcium activity.
    event_window (tuple): Start and end indices of the event window within the signal (e.g., (start_frame, end_frame)).
    event_duration (float): Duration of the event window in time units (e.g., seconds).
    fps (int): Frames per second of the calcium imaging data, used to convert between frame indices and time.
    noise_level (float): Threshold for peak detection in the background signal. Get from SNR of the signal.
    plot (bool): If True, plots the signal with detected peaks for visual evaluation.

    Returns:
    float: p-value indicating the probability of observing the number of events
           in the window given the background event rate.
    """
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    logging.info("Starting calculation of event specificity.")

    # Extract the segment of the signal corresponding to the event window
    event_signal = signal[event_window[0]:event_window[1]]

    # Combine all signal outside the event window into the background signal
    background_signal = np.concatenate([
        signal[:event_window[0]],  # Signal before the event window
        signal[event_window[1]:]  # Signal after the event window
    ])

    logging.info("Event window and background signal extracted.")

    # Define the duration of a calcium transient based on imaging frame rate
    transient_duration = 0.4 * fps  # Assume transient lasts ~0.4 seconds

    # Set the noise level threshold for detecting peaks
    noise_level = noise_level * 0.9

    logging.info("Detecting peaks in the background signal.")

    # Detect peaks in the background signal
    peaks, _ = find_peaks(
        background_signal,
        distance=3 * transient_duration,  # Enforce minimum separation between peaks
        width=transient_duration,         # Ensure peak width matches transient duration
        height=noise_level          # Set threshold for peak height relative to noise
    )

    if len(peaks) == 0:
        logging.warning("No peaks detected in the background signal. Returning NaN.")
        lambda_background = float(0) + 1e-10

    else:
        # Convert detected peak positions to a time-based event train (in seconds)
        event_train = peaks / fps

        # Calculate the mean firing rate (events per second) in the background signal
        lambda_background = mean_firing_rate(event_train)  # Assumes `mean_firing_rate` is defined elsewhere

    logging.info(f"Mean firing rate in the background: {lambda_background:.3f} events/sec.")
    logging.info("Detecting peaks in the event window.")

    # Detect peaks in the event window using the same parameters as for the background
    k = len(find_peaks(
        event_signal,
        distance=3 * transient_duration,  # Enforce minimum separation between peaks
        width=transient_duration,         # Ensure peak width matches transient duration
        height=noise_level          # Set threshold for peak height relative to noise
    )[0])

    logging.info(f"Number of peaks detected in the event window: {k}.")

    # Calculate the p-value for observing `k` or more events in the event window
    # given the Poisson-distributed background event rate
    p_value = 1 - stats.poisson.cdf(k - 1, lambda_background * event_duration)

    logging.info(f"Calculated p-value: {p_value:.5f}.")

    if p_value < 0.01:
        plot=False

    # Optional plotting
    if plot:
        logging.info("Plotting the signal and detected peaks.")
        plt.figure(figsize=(10, 6))
        plt.plot(signal, label='Signal', alpha=0.7)
        plt.scatter(peaks[peaks < event_window[0]], background_signal[peaks[peaks < event_window[0]]], color='red', label='Background Peaks')
        plt.scatter(peaks[peaks >= event_window[0]] + event_duration*fps, background_signal[peaks[peaks >= event_window[0]]], color='red')
        event_peaks, _ = find_peaks(
            event_signal,
            distance=3 * transient_duration,
            width=transient_duration,
            height=noise_level
        )
        event_peaks += event_window[0]  # Adjust indices to match full signal
        plt.scatter(event_peaks, signal[event_peaks], color='blue', label='Event Peaks')
        plt.axvspan(event_window[0], event_window[1], color='yellow', alpha=0.3, label='Event Window')
        plt.title("Calcium Imaging Signal with Detected Peaks\np-value: {:.5f}, Noise Level: {:.2f}".format(p_value, noise_level))
        plt.xlabel("Frame")
        plt.ylabel("Signal Amplitude")
        plt.legend()
        plt.show(block=True)

    return p_value

if __name__ == "__main__":
    # Example usage
    signal = np.random.normal(0, 1, 1000)  # Example signal
    signal[450:550] += 2  # Add an "event"
    event_window = (450, 550)
    event_duration = 1.0  # Assuming each sample represents 0.01 time units

    p_value = calculate_event_specificity(signal, event_window, event_duration, 100, plot=True)
    print(f"P-value: {p_value}")

