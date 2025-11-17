import numpy as np
import pandas as pd
import caiman as cm
import os
from typing import List
import logging
from caiman.source_extraction.cnmf import params as params
from src.setup_for_caiman import *
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction import cnmf
from caiman.source_extraction.cnmf.cnmf import load_CNMF
from scipy import ndimage
from src.plotting import *
import cv2
from elephant.statistics import mean_firing_rate, isi
from scipy.signal import find_peaks

def compute_event_rates(dff, cohort, animal, session, calcium_fps, distance, width):
    # Ensure the directory exists
    os.makedirs(f'tech_plots/snr/{cohort}/{animal}/{session}', exist_ok=True)

    # Compute event rate for each neuron
    event_rates = []
    isis = []

    for i, signal in enumerate(dff):
        # Min-max normalize the signal and zero the mean
        signal = (signal - np.min(signal)) / (np.max(signal) - np.min(signal))
        signal = signal - np.mean(signal)

        # Plot 20 random signals
        if i % len(dff) // 20 == 0:
            fig, ax = plt.subplots(figsize=(16, 4))
            ax.plot(signal, color='black', label='Signal')

            # Detect peaks
            peaks, _ = find_peaks(signal, distance=distance, width=width, height=0.2)
            if len(peaks) == 0:
                event_rates.append(0)
                isis.append(0)
                continue

            event_train = peaks / calcium_fps
            rate = mean_firing_rate(event_train)
            mean_isi = np.mean(isi(event_train))

            event_rates.append(rate)
            isis.append(mean_isi)

            # Plot peaks
            ax.plot(peaks, signal[peaks], '*', color='red', label=f'Peaks (height=0.2)', alpha=1)

            ax.set_title(f'Signal with Peaks - Neuron {i}')
            ax.set_xlabel('Time (frames)')
            ax.set_ylabel('Signal Amplitude')
            ax.legend()
            ax.set_aspect('auto')
            plt.savefig(f'tech_plots/snr/{cohort}/{animal}/{session}/neuron_{i}_peaks.png', bbox_inches='tight')
            plt.close()
        else:
            # Detect peaks
            peaks, _ = find_peaks(signal, distance=distance, width=width, height=0.2)
            if len(peaks) == 0:
                event_rates.append(0)
                isis.append(0)
                continue

            event_train = peaks / calcium_fps
            rate = mean_firing_rate(event_train)
            mean_isi = np.mean(isi(event_train))

            event_rates.append(rate)
            isis.append(mean_isi)

    return event_rates, isis

def shuffle_signal(DF_F_Z_scored, bin_time_seconds=3, fps=30):
    """
    Shuffle the temporal structure of cellular signals while preserving local patterns.

    This function takes a 2D array of z-scored ΔF/F signals, where each row represents
    a cell's activity over time. It divides each cell's signal into time bins of a 
    specified duration, randomly shuffles these bins, and reconstructs the signal.
    Any remaining frames at the end are randomly placed within the shuffled signal.

    Parameters:
    -----------
    DF_F_Z_scored : numpy.ndarray
        A 2D array of shape (n_cells, n_timepoints) containing z-scored ΔF/F signals.
    bin_time_seconds : float, optional
        The duration of each time bin in seconds (default is 3 seconds).
    fps : int, optional
        Frames per second of the recording (default is 30 fps).

    Returns:
    --------
    shuffled_DFF : numpy.ndarray
        A 2D array of the same shape as the input, containing the shuffled signals.

    Notes:
    ------
    - The function calculates the number of frames per bin based on the provided
      bin_time_seconds and fps.
    - Any remaining frames that don't fit into a full bin are randomly placed
      within the shuffled signal.
    - This function is useful for creating control datasets in neuroscience experiments
      to assess the significance of observed temporal patterns.

    Raises:
    -------
    AssertionError:
        If the output shape doesn't match the input shape or if no shuffling occurred.
    ValueError:
        If the calculated frames per bin is less than 1.
    """

    # Calculate the number of frames per bin
    frames_per_bin = int(bin_time_seconds * fps)
    if frames_per_bin < 1:
        raise ValueError(f"The combination of bin_time_seconds={bin_time_seconds} and fps={fps} results in less than one frame per bin. Please adjust these parameters.")

    # Calculate the number of bins and the size of the leftover piece
    num_frames = DF_F_Z_scored.shape[1]
    num_bins = num_frames // frames_per_bin
    leftover_size = num_frames % frames_per_bin

    # Initialize the output array with zeros, matching the input shape
    shuffled_DFF = np.zeros_like(DF_F_Z_scored)

    # Iterate over each cell (each row in the input array)
    for i in range(DF_F_Z_scored.shape[0]):
        # Create an array of bin indices and shuffle it
        idx = np.arange(num_bins)
        np.random.shuffle(idx)

        # Shuffle the main bins
        for k, j in enumerate(idx):
            shuffled_start = j * frames_per_bin
            shuffled_end = (j + 1) * frames_per_bin
            positional_start = k * frames_per_bin
            positional_end = (k + 1) * frames_per_bin
            shuffled_DFF[i, positional_start:positional_end] = DF_F_Z_scored[i, shuffled_start:shuffled_end]

        # Handle the leftover piece
        if leftover_size > 0:
            # Get the leftover piece from the end of the original signal
            leftover_piece = DF_F_Z_scored[i, -leftover_size:]
            
            # Choose a random position to insert the leftover piece
            insert_position = np.random.randint(0, num_frames - leftover_size + 1)
            
            # Shift the data after the insert position
            shuffled_DFF[i, insert_position + leftover_size:] = shuffled_DFF[i, insert_position:-leftover_size]
            
            # Insert the leftover piece
            shuffled_DFF[i, insert_position:insert_position + leftover_size] = leftover_piece

    # Verify that the output has the same shape as the input
    assert shuffled_DFF.shape == DF_F_Z_scored.shape, "Output shape does not match input shape"

    # Verify that shuffling actually occurred (output is different from input)
    assert not np.array_equal(shuffled_DFF, DF_F_Z_scored), "No shuffling occurred"

    return shuffled_DFF

def smooth_and_shrink_label(label, smooth_sigma=1, shrink_pixels=1):
    """
    Smooth and shrink a binary label.

    :param label: Binary label array.
    :param smooth_sigma: Sigma for Gaussian smoothing.
    :param shrink_pixels: Number of pixels to erode.
    :return: Smoothed and shrunken label array.
    """
    if smooth_sigma > 0:
        # Apply Gaussian smoothing only if smooth_sigma is greater than 0
        smoothed = ndimage.gaussian_filter(label.astype(float), sigma=smooth_sigma)
        # Use a binary threshold to create a binary mask again
        label = (smoothed > 0.5).astype(int)
    
    if shrink_pixels > 0:
        # Apply binary erosion only if shrink_pixels is greater than 0
        label = ndimage.binary_erosion(label, iterations=shrink_pixels).astype(int)

    return label

def calculate_binary_behaviors(annotations, unique_behaviors, npy_data):
    """Calculates binary vectors indicating the presence of behaviors."""
    behaviors = {behavior: get_behaviour_timestamps(annotations, behavior) for behavior in unique_behaviors}
    binary_behaviors = {behavior: np.zeros(npy_data.shape[1]) for behavior in unique_behaviors}
    
    for behavior in unique_behaviors:
        for start, end in behaviors[behavior]:
            binary_behaviors[behavior][start:end] = 1
    return binary_behaviors

def select_rows(summary_file, day, mouse):
    """
    Selects rows from a summary file based on the given day and mouse.

    Args:
        summary_file (DataFrame): The summary file containing the data.
        day (int): The day to filter the rows by.
        mouse (str): The mouse to filter the rows by.

    Returns:
        DataFrame: The filtered summary file.

    """
    # Parse day from "day_{int}" into int 
    #summary_file['Day'] = summary_file['Day'].str.extract('(\d+)').astype(int)
    # In Mouse column, if there are only 3 characters, add a 0 in front
    #summary_file['Mouse'] = summary_file['Mouse'].apply(lambda x: str(x).zfill(4))
    # Select the rows that match the day and mouse
    summary_file = summary_file[(summary_file['Day'] == int(day)) & (summary_file['Mouse'] == str(mouse))]
    
    return summary_file

def load_manual_annotations(session_file, ci_path):
    logging.info('Loading manual annotations...')
    # Parse session file to get behaviour video
    __, side_camera_name = parse_session_file(session_file, ci_path)
    # Remove .mp4 from behaviour_video_name
    behaviour_video_name = side_camera_name.replace('.mp4', '')
    # Remove last last words (separated by '_')
    behaviour_video_name = behaviour_video_name.rsplit('_', 1)[0]
    
    # Add manual_annotations to behaviour_video_name
    manual_annotations_path = behaviour_video_name + '_manual_annotations.csv'

    # Replace data with manual_annotations
    manual_annotations_path = manual_annotations_path.replace('data', 'manual_annotations_behaviour')
    # Remove bottom_camera from path
    manual_annotations_path = manual_annotations_path.replace('/bottom_camera', '')
    # Load manual annotations
    manual_annotations = pd.read_csv(manual_annotations_path)

    return manual_annotations

def get_behaviour_timestamps(manual_annotations, behavior_type):
    # Get all behavior_timestamps for the specified behavior type
    manual_annotations_filtered = manual_annotations[manual_annotations['Behavior'] == behavior_type]
    
    # Initialize list to store behavior_timestamps
    behavior_timestamps = []
    
    # Iterate over each door interaction and get start and end frame
    for index, row in manual_annotations_filtered.iterrows():
        start_frame = int(row['Start (frames)'])
        end_frame = int(row['Stop (frames)'])
        behavior_timestamps.append((start_frame, end_frame))
    
    return behavior_timestamps

def load_behaviour_video(behaviour_video_path, manual_annotations):
    logging.info('Loading behaviour video...')
    """
    Load a behaviour video and annotate it with manual annotations.

    Args:
        behaviour_video_path (str): The path to the behaviour video file.
        manual_annotations (pandas.DataFrame): The manual annotations for the video.

    Returns:
        numpy.ndarray: The annotated behaviour video as a numpy array.
    """
    # get all frames from behaviour videos
    frames = []

    cap = cv2.VideoCapture(behaviour_video_path)
    ret = True
    
    # Check for equal length of manual_annotations and video
    length_manual_annotations = manual_annotations['Stop (frames)'].max()

    if length_manual_annotations != int(cap.get(cv2.CAP_PROP_FRAME_COUNT)):
        # print length of manual_annotations and video but don't stop the code
        logging.warning(f'Length of manual_annotations ({length_manual_annotations}) and video ({int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}) do not match.')

    while ret:
        ret, img = cap.read()
        if ret:
            frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            behavior = manual_annotations[(manual_annotations['Start (frames)'] <= frame_number) & (manual_annotations['Stop (frames)'] >= frame_number)]['Behavior'].values
            if len(behavior) > 0:
                behavior_text = behavior[0]
                cv2.putText(img, behavior_text, (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            frames.append(img)
    behaviour_video = np.stack(frames, axis=0)

    return behaviour_video

def get_time_victim_crossing(session_file_path: str, fps: int, ci_path: str) -> int:
    """
    Get the TimeVictimCrossing for a session.

    Args:
        session_file_path (str): Path to the session file.
        fps (int): Frames per second.

    Returns:
        int: TimeVictimCrossing in frames.
    """
    # Load session file
    session_file = pd.read_csv(session_file_path)

    # Parse session file to get behaviour video
    bottom_camera_filename, __ = parse_session_file(session_file, ci_path)

    # Get the corresponding session file entry
    session_file_entry = session_file[session_file['Bottom camera'] == bottom_camera_filename.split('/')[-1].split('.')[0]]

    # Get the TimeVictimCrossing
    time_victim_crossing = session_file_entry['TimeVictimCrossing'].values[0] * fps  # in frames

    return time_victim_crossing

def parse_session_file(session_file, ci_path):
    logging.info('Parsing session file...')
    # Parse ci_path to get the video name
    path, ci_filename = os.path.split(ci_path)

    # Replace miniscope with bottom_camera in path
    path = path.replace('miniscope', 'bottom_camera')
    
    # Parse session file to get behaviour video
    current_session = session_file[session_file['Miniscope video'] == ci_filename]
    
    bottom_camera_name = current_session['Bottom camera'].values[0]
    bottom_camera_name = bottom_camera_name + '.mp4'

    side_camera_name = current_session['Side camera'].values[0]
    side_camera_name = side_camera_name + '.mp4'

    return os.path.join(path, bottom_camera_name), os.path.join(path, side_camera_name)

def load_data(ci_path, session_file, fps=30):
    """
    Load data including calcium imaging data, session file, manual annotations, and behaviour video.

    Parameters:
    - ci_path (str): Path to the calcium imaging movie file.
    - session_file (str): Path to the session file.
    - fps (int): Frame rate of the calcium imaging data.

    Returns:
    - List: List of tuples containing loaded data, where each tuple consists of the data and its name.
    """ 
    global ci_path_global
    ci_path_global = ci_path
    
    logging.info('Loading data...')
    # Load calcium imaging data
    ci_movie = cm.load(ci_path, fr=fps, outtype='uint16')

    # Load session file
    session_file = pd.read_csv(session_file)
    # Parse session file to get behaviour video
    behaviour_video_path, ___ = parse_session_file(session_file, ci_path)

    # Load manual annotations
    manual_annotations = load_manual_annotations(session_file, ci_path)

    # Load behaviour video
    behaviour_video = load_behaviour_video(behaviour_video_path, manual_annotations) 

    return ci_movie, behaviour_video

def visualize_analysis(ci_movie, motion_correct=False, plot=False, already_set=False):
    """
    Visualize the analysis results.

    Parameters:
        ci_movie: The calcium imaging movie.
        motion_correct: Flag indicating whether to perform motion correction.
        plot: Flag indicating whether to plot rigid shifts.
        already_set: Flag indicating whether the analysis has already been set.

    Returns:
        A list of napari layers.

    """
    cnmfe_model, correlation_image, peak_to_noise_ratio, dims = analyze_data(ci_movie, motion_correct, plot, already_set)

    masks = []
    for i in range(cnmfe_model.estimates.A.shape[1]):
        masks.append(cnmfe_model.estimates.A[:, i].toarray().reshape(dims, order='F'))
        masks[i] = masks[i] > 0
        masks[i] = masks[i] * i

    masks = np.stack(masks, axis=0)
    components_labels = np.max(masks, axis=0)

    duration = cnmfe_model.estimates.C.shape[1]

    frames = []

    for f in range(duration):
        fig = plt.figure(figsize=(10,10))

        for i, estimate in enumerate(cnmfe_model.estimates.C):
            if i > 10:
                continue
            plt.plot(estimate - 500*i)
            plt.axvline(f, color='r')

        plt.title(f'Frame {f}')
        frames.append(fig)
        # Close the figure to avoid memory leak
        plt.close(fig)

    # Create array from the frames
    frames = np.stack(frames, axis=0)
        
    return [(correlation_image, {'name': 'Correlation Image'}), (components_labels, {'name': 'Components Labels'}), (frames, {'name': 'Traces'})]

def plot_rigid_shifts(motion_corrector, save_dir, save_name):
    """
    Plot the rigid shifts during motion correction.

    Parameters:
        motion_corrector: The motion corrector object.
        save_dir: The directory to save the plot.

    """
    plt.plot(motion_corrector.shifts_rig)  # plot rigid shifts
    plt.legend(['x shifts', 'y shifts'])
    plt.xlabel('frames')
    plt.ylabel('pixels')
    plt.gcf().set_size_inches(6, 3)

    # Save plot
    plt.savefig(f'{save_dir}/rigid_shifts_{save_name}.png')
    # Close the plot to avoid memory leak
    plt.close()
    
    logging.info('Rigid shifts plot saved.')

def load_session_file(dir_path):
    """
    Load a session file and prepare it for analysis.

    Parameters
    ----------
    file_path : str
        The path to the directory where session files are stored.
    event_type : str
        The event type to create proper columns for.

    Returns
    -------
    session_file : pandas.DataFrame
        The prepared session file.
    """

    # Check directory to find all csv files starting with SESSION_FILE
    csv_files = [file for file in os.listdir(dir_path) if file.startswith('SESSION_FILE')]
    # In all csv files sort according to date, extracted using 'data/SESSION_FILE_{os.path.basename(__file__)[:-3]}_{current_datetime}.csv' pattern
    csv_files.sort(key=lambda x: os.path.getmtime(os.path.join(dir_path, x)))
    # Select the most recent csv file
    file_path = os.path.join(dir_path, csv_files[-1])

    # Load the csv file
    session_file = pd.read_csv(file_path)

    # Convert Mouse column to str with 4 digits
    # session_file['Mouse'] = session_file['Mouse'].astype(str).str.zfill(4)

    return session_file