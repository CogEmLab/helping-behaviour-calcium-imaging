#TODO: Also add deeplabcut coords to the behaviour video!

import napari
import numpy as np
import pandas as pd
from napari.layers import Image
import pathlib
import caiman as cm
import os
from napari.types import LayerDataTuple, ImageData
from typing import List
import cv2
import logging
from caiman.source_extraction.cnmf import params as params
from src.setup_for_caiman import *
from caiman.motion_correction import MotionCorrect
import matplotlib.pyplot as plt
from caiman.source_extraction import cnmf
from src.plotting import plot_histogram

############################################################################################################
# Configure logging
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

############################################################################################################
# core functions
def get_most_active_components(DF_F_Z_scored, path, cohort, session, animal, plot_example=False, ax=None):
    """
    Get the most active components in the DF_F_Z_scored based on k*sigma threshold and save a histogram of the sum of the components with a threshold.

    Parameters:
        DF_F_Z_scored (np.DF_F_Z_scored): The input DF_F_Z_scored of shape (n_components, time) cropped around the window of interest.
        path (str): The path to the file to save the histogram.
        cohort (str): The cohort of the mouse.
        session (str): The session of the mouse.
        animal (str): The animal ID.
        plot_example (bool): Whether to plot an example of the histogram.
        ax (matplotlib.axis): The axis to plot the histogram on.

    Returns:
        DF_F_Z_scored_selected (np.DF_F_Z_scored): The components that are above the threshold.
        selected_ids (np.array): The ids of the selected components.

    Raises:
        None
    """
    
    # Get only components that are the most active in the window 
    dff_sum = np.sum(DF_F_Z_scored, axis=1)
    # Sort the components by the sum of their activity
    sorted_idxs = np.argsort(dff_sum)[::-1] #! [::-1] to get the descending order
    array_sorted = DF_F_Z_scored[sorted_idxs]

    # Compute the threshold as 1 sigma above the mean
    threshold = np.mean(dff_sum) + np.std(dff_sum)
    # Get the components that are above the threshold
    mask = dff_sum[sorted_idxs] > threshold
    DF_F_Z_scored_selected = array_sorted[mask]

    # Get a vector with selected ids
    selected_ids = sorted_idxs[mask] #^ To use for registration search

    # Also plot a histogram of the sum of the components with a threshold
    if plot_example:
        filename = os.path.join(path, 'histogram.png')
        print('-----------------------------------')
        print('Plotting example')
        print('-----------------------------------')
        plot_histogram(dff_sum, threshold, filename, ax=ax)
    
    # Add cohort, session and mouse_id to the path
    path = os.path.join(path, cohort, session)
    os.makedirs(path, exist_ok=True)
    filename = os.path.join(path, f'histogram_{animal}.png')
    plot_histogram(dff_sum, threshold, filename)
    
    return DF_F_Z_scored_selected, selected_ids

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

def analyze_data(ci_movie_path, save_dir_docs, fps=30, motion_correct=False, plot=False):
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
    # Get the name of the file
    save_name = os.path.basename(ci_movie_path).split('.')[0]

    # Perform setup for caiman
    cluster, n_processes = general_setup()

    # Get parameters for motion correction
    params_mc = get_params_mot_corr(ci_movie_path, frate=fps)

    # Perform motion correction if True
    if motion_correct:
        
        # do motion correction rigid
        motion_corrector = MotionCorrect(ci_movie_path, dview=cluster, **params_mc.get_group('motion'))
        motion_corrector.motion_correct(save_movie=True)

        # Save motion corrected movie as memory-mappable file
        pw_rigid = params_mc.get('motion', 'pw_rigid')  # Check if motion correction is rigid or piecewise rigid
        fname_mc = motion_corrector.fname_tot_rig  # Get the filename of the motion corrected movie

        if pw_rigid:
            # If motion correction is rigid, calculate the maximum absolute shifts in x and y directions
            border_px = np.ceil(np.maximum(np.max(np.abs(motion_corrector.x_shifts_els)),
                                           np.max(np.abs(motion_corrector.y_shifts_els)))).astype(int)
        else:
            # If motion correction is piecewise rigid, calculate the maximum absolute shifts
            border_px = np.ceil(np.max(np.abs(motion_corrector.shifts_rig))).astype(int)

        border_nan = params_mc.get('motion', 'border_nan')  # Get the border_nan parameter

        border_px = 0 if border_nan == 'copy' else border_px  # Set border_px to 0 if border_nan is 'copy'

        fname_new = cm.save_memmap(fname_mc, base_name=f'memmap_{save_name}', order='C', border_to_0=border_px)  # Save memory-mappable file
        
    else:  # if no motion correction, just memory map the file
        # save memory file to save_path

        fname_new = cm.save_memmap([ci_movie_path], base_name=f'memmap_{save_name}',
                                order='C', border_to_0=0, dview=cluster)
            
    # load memory mappable file
    Yr, dims, T = cm.load_memmap(fname_new)
    ci_movie = Yr.T.reshape((T,) + dims, order='F')

    # get parameters for cnmfe
    cnmfe_params = get_params_cnmfe(params_mc, gSig=np.array([4, 4]), gSiz= 2*np.array([4, 4])+1,  rf=24, stride_cnmf=8,  min_pnr=7, min_SNR=3.0, SNR_lowest=0.5)
    
    # run cnmfe
    cnmfe_model = cnmf.CNMF(n_processes=n_processes, 
                        dview=cluster, 
                        params=cnmfe_params, center_psf=True, method_init='corr_pnr', only_init_patch=True)
    
    cnmfe_model.fit(ci_movie)

    if plot:
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
    correlation_image, peak_to_noise_ratio = cm.summary_images.correlation_pnr(ci_movie[::max(T//1000, 1)], # subsample if needed
                                                                            gSig=gsig_tmp[0], # used for filter
                                                                            swap_dim=False) # change swap dim if output looks weird, it is a problem with tiffile
    
    cm.stop_server(dview=cluster)
    
    return cnmfe_model, ci_movie, fname_new, correlation_image, peak_to_noise_ratio, dims, total_components, num_accepted, num_rejected

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


############################################################################################################
def run_viewer():
    # start napari
    viewer = napari.Viewer()
    
    # # widgets section
    # # CREATE A WIDGET TO LOAD THE DATA
    # @magicgui(call_button="Load Data",
    #             file={"label": "Calcium Imaging Data"},
    #             session_file={"label": "Session File"},
    #             fps={"label": "Frames per second"})
    #            # set labels + default values
    # def load_data_widget(file = pathlib.Path('data/day_1/miniscope/tiff_data_5592_day1.tiff'), session_file = pathlib.Path('data/session_file_MiniscopeCW_10_2021_Moises.csv'), fps = int(30)) -> List[LayerDataTuple]:
    #     logging.info('Load Data button clicked.')
    #     return load_data(file, session_file, fps)
    
    # # CREATE A WIDGET TO ANALYZE THE DATA
    # @magicgui(call_button="Analyze Data",
    #             motion_correct={"label": "Motion Correct"},
    #             plot={"label": "Plot Rigid Shifts"})
    # def analyze_data_widget(motion_correct=False, plot=False) -> List[LayerDataTuple]:
    #     logging.info('Analyze Data button clicked.')
    #     return visualize_analysis(ci_path_global, motion_correct, plot)
    
    # dw1 = viewer.window.add_dock_widget(load_data_widget, area='right', name='Load Data')
    # dw2 = viewer.window.add_dock_widget(analyze_data_widget, area='right', name='Analyze Data')
    
    return viewer

############################################################################################################
# main function

if __name__ == "__main__":
    ci_path_global = None
    # start napari
    logging.info('Starting Napari viewer...')
    run_viewer()
    napari.run()
