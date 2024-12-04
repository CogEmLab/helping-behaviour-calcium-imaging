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
from collections import defaultdict

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

def reevaluate_components(model_path, fname_new, fps=30):
    """
    Analyze calcium imaging data using CNMF-E algorithm.

    Parameters:
        ci_movie_path (str): Path to the calcium imaging movie file.
        save_dir_docs (str): Directory to save the technical documentation files.
        motion_correct (bool): Flag indicating whether to perform motion correction. Default is False.
        plot (bool): Flag indicating whether to plot the rigid shifts. Default is False.

    Returns:
        cnmfe_model: The CNMF-E model object.
        ci_movie: The calcium imaging movie.
        fname_new: The filename of the memory-mappable file.
        correlation_image: The correlation image.
        peak_to_noise_ratio: The peak to noise ratio movie.
        dims: The dimensions of the calcium imaging movie.
        total_components: The total number of components.
        num_accepted: The number of accepted components.
        num_rejected: The number of rejected components.
    """
    logging.info('Analyzing data...')
    # # Get the name of the file

    # Perform setup for caiman
    cluster, __ = general_setup()

    # load memory mappable file
    Yr, dims, T = cm.load_memmap(fname_new)
    ci_movie = Yr.T.reshape((T,) + dims, order='F')

    cnmfe_model = load_CNMF(model_path)

    # # Upload the new parameters
    cnmfe_model.params.quality['SNR_lowest'] = 2.0
    cnmfe_model.params.quality['rval_lowest'] = 0.3
    cnmfe_model.params.quality['min_SNR'] = 3.0

    cnmfe_model.estimates.evaluate_components(ci_movie, cnmfe_model.params)

    total_components = len(cnmfe_model.estimates.C)
    num_accepted = len(cnmfe_model.estimates.idx_components)
    num_rejected = len(cnmfe_model.estimates.idx_components_bad)

    # Compute some summary images (correlation and peak to noise)
    gsig_tmp = (3,3)
    correlation_image, peak_to_noise_ratio = cm.summary_images.correlation_pnr(ci_movie[::max(T//1000, 1)], # subsample if needed
                                                                            gSig=gsig_tmp[0], # used for filter
                                                                            swap_dim=False) # change swap dim if output looks weird, it is a problem with tiffile
    
    cm.stop_server(dview=cluster)
    
    return cnmfe_model, ci_movie, fname_new, correlation_image, peak_to_noise_ratio, dims, total_components, num_accepted, num_rejected


def analyze_data(ci_movie_path, save_dir_docs, config_filepath, fps=30, motion_correct=False, plot=False):
    """
    Analyze calcium imaging data using CNMF-E algorithm.

    Parameters:
        ci_movie_path (str): Path to the calcium imaging movie file.
        save_dir_docs (str): Directory to save the technical documentation files.
        motion_correct (bool): Flag indicating whether to perform motion correction. Default is False.
        plot (bool): Flag indicating whether to plot the rigid shifts. Default is False.

    Returns:
        cnmfe_model: The CNMF-E model object.
        ci_movie: The calcium imaging movie.
        fname_new: The filename of the memory-mappable file.
        correlation_image: The correlation image.
        peak_to_noise_ratio: The peak to noise ratio movie.
        dims: The dimensions of the calcium imaging movie.
        total_components: The total number of components.
        num_accepted: The number of accepted components.
        num_rejected: The number of rejected components.
        border_px: Maximum shift in motion correction.
    """
    logging.info('Analyzing data...')
    
    # Load config file 
    config = load_config(config_filepath)
    
    # Get the name of the file
    save_name = os.path.basename(ci_movie_path).split('.')[0]

    # Perform setup for caiman
    cluster, n_processes = general_setup()

    # Get parameters for motion correction
    logging.info('Getting parameters for motion correction')
    params_mc = get_params_mot_corr(ci_movie_path, config=config)

    # Perform motion correction if True   
    if motion_correct:
        logging.info('Performing motion correction')
        # do motion correction rigid
        motion_corrector = MotionCorrect(ci_movie_path, dview=cluster, **params_mc.get_group('motion'))
        motion_corrector.motion_correct(save_movie=True)

        f_order_file = motion_corrector.fname_tot_rig[0]

        # # Save motion corrected movie as memory-mappable file
        pw_rigid = params_mc.get('motion', 'pw_rigid')  # Check if motion correction is rigid or piecewise rigid
          # Get the filename of the motion corrected movie

        if pw_rigid:
            # If motion correction is rigid, calculate the maximum absolute shifts in x and y directions
            logging.info('Calculating maximum absolute shifts in x and y directions')
            border_px = np.ceil(np.maximum(np.max(np.abs(motion_corrector.x_shifts_els)),
                                           np.max(np.abs(motion_corrector.y_shifts_els)))).astype(int)
        else:
            # If motion correction is piecewise rigid, calculate the maximum absolute shifts
            logging.info('Calculating maximum absolute shifts')
            border_px = np.ceil(np.max(np.abs(motion_corrector.shifts_rig))).astype(int)

        border_nan = params_mc.get('motion', 'border_nan')  # Get the border_nan parameter

        border_px = 0 if border_nan == 'copy' else border_px  # Set border_px to 0 if border_nan is 'copy'

        logging.info('Saving memory-mappable file')
        fname_new = cm.save_memmap([ci_movie_path], base_name=f'memmap_{save_name}', order='C', border_to_0=border_px)  # Save memory-mappable file
        
    else:  # if no motion correction, just memory map the file
        # save memory file to save_path
        logging.info('Saving memory-mappable file without motion correction')
        fname_new = cm.save_memmap([ci_movie_path], base_name=f'memmap_{save_name}',
                                order='C', border_to_0=0, dview=cluster)
    
    # load memory mappable file
    logging.info('Loading memory-mappable file')
    Yr, dims, T = cm.load_memmap(fname_new)
    ci_movie = Yr.T.reshape((T,) + dims, order='F')

    # get parameters for cnmfe
    logging.info('Getting parameters for CNMF-E')
    cnmfe_params = get_params_cnmfe(params_mc, config=config)
    
    # run cnmfe
    logging.info('Running CNMF-E')
    cnmfe_model = cnmf.CNMF(n_processes=n_processes, 
                        dview=cluster, 
                        params=cnmfe_params, center_psf=True, method_init='corr_pnr', only_init_patch=True)
    cnmfe_model.fit(ci_movie)

    if plot:
        logging.info('Plotting rigid shifts')
        plot_rigid_shifts(motion_corrector, save_dir_docs, save_name)

    logging.info('CNMF-E analysis completed.')

    quality_params = {'use_cnn' : False,
                    'rval_thr' : 0.85,
                    'SNR_lowest' : 2.0,
                    'rval_lowest' : 0.3,
                    'min_SNR' : 3.0,
                    }
    
    cnmfe_model.params.change_params(params_dict=quality_params)

    cnmfe_model.estimates.evaluate_components(ci_movie, cnmfe_model.params)

    total_components = len(cnmfe_model.estimates.C)
    num_accepted = len(cnmfe_model.estimates.idx_components)
    num_rejected = len(cnmfe_model.estimates.idx_components_bad)

    # Compute some summary images (correlation and peak to noise)
    gsig_tmp = (3,3)
    logging.info('Computing summary images')
    correlation_image, peak_to_noise_ratio = cm.summary_images.correlation_pnr(ci_movie[::max(T//1000, 1)], # subsample if needed
                                                                            gSig=gsig_tmp[0], # used for filter
                                                                            swap_dim=False) # change swap dim if output looks weird, it is a problem with tiffile
    # ^ Just out of curiosity, which is subsampling being used in above function when passing the ci_movie array?
    cm.stop_server(dview=cluster)
    
    # Delete one of the motion corrected file
    os.system(f'rm {f_order_file}')
    return cnmfe_model, ci_movie, fname_new, correlation_image, peak_to_noise_ratio, dims, total_components, num_accepted, num_rejected, border_px

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