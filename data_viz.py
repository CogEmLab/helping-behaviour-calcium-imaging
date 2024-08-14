# Call using: ```python save_data_viz.py 5592 HBTD3 --downsample```
#TODO: Add a reg number to the napari vizualization
import os
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import pynapple as nap
import cv2 as cv
from tifffile import imread, imwrite
import caiman as cm
from caiman.source_extraction.cnmf.cnmf import load_CNMF
import argparse
from src.utils import get_behaviour_timestamps
import napari
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops_table


COHORT = 'Oct_2021'
ROOT = '/home/aagafono/linstore/CogEm/Projects/helping_behavior/Analysis/Calcium-Imaging-Analysis'
SUBROOT = f'data/dHPC/{COHORT}'

CAIMAN_OUTPUT_SUMMARY_FILE_PATH = os.path.join(ROOT, SUBROOT, 'Oct_2021_cnmfe_output.csv')

OUTPUT_DIR = os.path.join(ROOT, 'test_viz')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def make_bbox(bbox_extents):
    """Get the coordinates of the corners of a
    bounding box from the extents

    Parameters
    ----------
    bbox_extents : list (4xN)
        List of the extents of the bounding boxes for each of the N regions.
        Should be ordered: [min_row, min_column, max_row, max_column]

    Returns
    -------
    bbox_rect : np.ndarray
        The corners of the bounding box. Can be input directly into a
        napari Shapes layer.
    """
    minr = bbox_extents[0]
    minc = bbox_extents[1]
    maxr = bbox_extents[2]
    maxc = bbox_extents[3]

    bbox_rect = np.array(
        [[minr, minc], [maxr, minc], [maxr, maxc], [minr, maxc]]
    )
    bbox_rect = np.moveaxis(bbox_rect, 2, 0)

    return bbox_rect

def load_caiman_data(caiman_output_summary_file_path, animal, session):
    """
    Load CaImAn output summary data (it is the file that batch_cnmfe script creates) for a specific animal and session.

    Args:
        caiman_output_summary_file_path (str): Path to the CaImAn output summary file.
        animal (int): Animal number.
        session (str): Session name.

    Returns:
        pd.DataFrame: Filtered CaImAn output summary data.
    """
    caiman_output_summary_file = pd.read_csv(caiman_output_summary_file_path)
    return caiman_output_summary_file[(caiman_output_summary_file['Mouse'] == animal) & (caiman_output_summary_file['Session'] == session)]

def load_cnmfe_model(caiman_output_summary_file, root):
    """
    Load CNMF-E model from the specified path.

    Args:
        caiman_output_summary_file (pd.DataFrame): CaImAn output summary data.
        root (str): Root directory.

    Returns:
        CNMF: Loaded CNMF-E model.
    """
    caiman_path = os.path.join(root, caiman_output_summary_file['CNMFE Model Path'].values[0])
    return load_CNMF(caiman_path)

def load_image(file_path, root):
    """
    Load an image file.

    Args:
        file_path (str): Path to the image file.
        root (str): Root directory.

    Returns:
        np.ndarray: Loaded image.
    """
    full_path = os.path.join(root, file_path)
    return imread(full_path)

def load_motion_corrected_video(file_path):
    """
    Load motion-corrected video.

    Args:
        file_path (str): Path to the motion-corrected video file.

    Returns:
        np.ndarray: Loaded video data.
    """
    Yr, dims, T = cm.load_memmap(file_path)
    return Yr.T.reshape((T,) + dims, order='F')

def load_behavior_video(file_path, root, subroot, day):
    """
    Load behavior video.

    Args:
        file_path (str): Path to the behavior video file.
        root (str): Root directory.

    Returns:
        np.ndarray: Loaded behavior video data.
    """
    video_file_path = os.path.join(root, subroot, f'day_{day}', 'bottom_camera', f'{file_path}.mp4')
    
    cap = cv.VideoCapture(video_file_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    frames = np.array(frames)[:, :, :, 0]
    return np.stack(frames, axis=0)

def process_annotations(caiman_output_summary_file, root):
    """
    Process and load annotations.

    Args:
        caiman_output_summary_file (pd.DataFrame): CaImAn output summary data.
        root (str): Root directory.

    Returns:
        pd.DataFrame: Loaded annotations.
    """
    subroot = os.path.dirname(caiman_output_summary_file['CNMFE Model Path'].values[0]).replace('data', 'annotations').replace('cnmfe', '')
    annotation_file_path = os.path.join(root, subroot)
    video_file_path = Path(caiman_output_summary_file['Side Camera'].values[0])
    video_name = os.path.join(annotation_file_path, '_'.join(video_file_path.stem.split('_')[:-1]) + '_manual_annotations.csv')
    return pd.read_csv(video_name)

def get_activity_movie(ci_movie, pnr_image):
    """
    Create a peak-to-noise ratio (PNR) movie by overlaying the PNR image with the calcium imaging movie.

    Args:
        ci_movie (np.ndarray): Calcium imaging movie.
        pnr_image (np.ndarray): Peak-to-noise ratio image.

    Returns:
        np.ndarray: PNR movie.
    """
    # subtract background from ci_movie
    # First subtract the min projection of the movie
    # min_projection = np.min(ci_movie, axis=0)
    # ci_movie_subtracted = np.zeros_like(ci_movie)
    # for i in range(ci_movie.shape[0]):
    #     ci_movie_subtracted[i] = cv.subtract(ci_movie[i], min_projection)
    
    # get maxprojection of the movie
    max_projection = np.max(ci_movie, axis=0)

    # run gaussian filter
    max_projection = cv.GaussianBlur(max_projection, (35, 35), 0)

    # subtract background from each frame
    ci_movie_subtracted = np.zeros_like(ci_movie)

    for i in range(ci_movie.shape[0]):
        ci_movie_subtracted[i] = cv.subtract(ci_movie[i], max_projection)

        # also perform a bit of gaussian smoothing
        ci_movie_subtracted[i] = cv.GaussianBlur(ci_movie_subtracted[i], (5, 5), 0)
    
    # activity_movie = np.zeros_like(ci_movie)

    # for i in range(ci_movie.shape[0]):
    #     activity_movie[i] = cv.addWeighted(ci_movie_subtracted[i], 0.5, pnr_image, 0.5, 0)


    return ci_movie_subtracted


def main(animal, session, downsample=False):
    """
    Main function to process calcium imaging and behavior data.

    Args:
        animal (int): Animal number.
        session (str): Session name.
        downsample (bool): Whether to downsample data.
    """
    # Load session data
    caiman_output_summary_file = load_caiman_data(CAIMAN_OUTPUT_SUMMARY_FILE_PATH, animal, session)
    logging.info(f"Loaded CaImAn output summary data for animal {animal} and session {session}")

    # Set global variables
    FPS = caiman_output_summary_file['FPS'].values[0]
    DAY = caiman_output_summary_file['Day'].values[0]

    # Load CNMF-E model and other data files
    cnmfe_model = load_cnmfe_model(caiman_output_summary_file, ROOT)
    pnr_image = load_image(caiman_output_summary_file['Peak-to-Noise Ratio Path'].values[0], ROOT)
    ci_movie = load_motion_corrected_video(caiman_output_summary_file['Motion Corrected Path'].values[0])
    behaviour_video = load_behavior_video(caiman_output_summary_file['Bottom Camera'].values[0], ROOT, SUBROOT, DAY)
    logging.info("Loaded CNMF-E model and other data files")

    # Get good components
    # GOOD_COMPS_IDS = cnmfe_model.estimates.idx_components

    crossing_comp = pd.read_csv('data/dHPC/metadata_cross_comp.csv')
    crossing_comp = crossing_comp[crossing_comp['Session'] == session]
    crossing_comp = crossing_comp[crossing_comp['Mouse'] == animal]
    crossing_comp = crossing_comp['Component ID'].to_numpy()
    crossing_comp = crossing_comp[:20]

    print(len(crossing_comp))
    # Rmove components from crossing components if they are not in good components
    # COMPS_IDS = crossing_comp[np.isin(crossing_comp, GOOD_COMPS_IDS)]

    print(crossing_comp)

    # Convert spatial traces to mask
    accepted_components_labels = np.stack(
        [np.where(cnmfe_model.estimates.A[:, i].toarray().reshape(pnr_image.shape, order='F') > 0, i+1, 0)
         for i in COMPS_IDS], axis=0)
    logging.info("Converted spatial traces to mask")

    # Process annotations
    annotations = process_annotations(caiman_output_summary_file, ROOT)
    beh_types = annotations['Behavior'].unique()
    beh_values = [nap.IntervalSet(starts, ends, time_units='s') for starts, ends in
                  [(get_behaviour_timestamps(annotations, beh_type) / FPS).T for beh_type in beh_types]]
    logging.info("Processed and loaded annotations")

    # Detrend and normalize DF/F traces
    cnmfe_model.estimates.detrend_df_f(detrend_only=True)
    DF_F = cnmfe_model.estimates.F_dff[COMPS_IDS]
    DF_F_Z_scored = (DF_F - DF_F.mean(axis=1)[:, None]) / DF_F.std(axis=1)[:, None]
    logging.info("Detrended and normalized DF/F traces")

    # Create PNR movie
    activity_movie = get_activity_movie(ci_movie, pnr_image)
    logging.info("Created PNR movie")

    if downsample:
        # Downsample data if required
        behaviour_video = behaviour_video[:, ::2, ::2]
        ci_movie = ci_movie[:, ::2, ::2]
        accepted_components_labels = accepted_components_labels[:, ::2, ::2]
        activity_movie = activity_movie[:, ::2, ::2]
        pnr_image = pnr_image[::2, ::2]
        logging.info("Downsampled data")

    
    components_labels_napari = np.max(accepted_components_labels, axis=0)
    # GET BOUNDING BOXES AND ANNOTATE WITH SNR VALUES
    # create the properties dictionary
    properties = regionprops_table(
        components_labels_napari, properties=('label', 'bbox', 'perimeter', 'area')
    )

    properties['IDS'] = COMPS_IDS

    # create the bounding box rectangles
    bbox_rects = make_bbox([properties[f'bbox-{j}'] for j in range(4)])

    # specify the display parameters for the text
    text_parameters = {
        'string': 'label: {label}\nIDS: {IDS}',
        'size': 12,
        'color': 'green',
        'anchor': 'upper_left',
        'translation': [-3, 0],
    }

    # Open napari viewer to eximine data 
    with napari.gui_qt():

        #TODO: make png of DF_F_Z_scored first 10 calcium traces plot and add it to the napari viewer
        fig, ax = plt.subplots()
        # Convert to seconds
        TIME = np.arange(0, ci_movie.shape[0]) / FPS

        # plot first 10 traces of DF/F Z-scored one by one beneath each other
        for i in range(19):
            ax.plot(TIME, DF_F_Z_scored[i] + i*10, label=f'Component {COMPS_IDS[i]}')
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('DF/F Z-scored')
        ax.set_title('First 10 DF/F Z-scored traces')
        fig.tight_layout()
        # Make transparent background
        fig.patch.set_alpha(0)
        # Turn off axis
        ax.axis('off')

        # convert to numpy array
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)

        viewer = napari.Viewer()
        #viewer.add_image(ci_movie, name='ci_movie', colormap='gray', blending='additive')
        # Add pnr image
        viewer.add_image(pnr_image, name='pnr_image', colormap='gray', blending='additive')
        viewer.add_image(behaviour_video, name='behaviour_video', colormap='gray', blending='additive')
        viewer.add_image(activity_movie, name='activity_movie', colormap='red', blending='additive')
        viewer.add_labels(components_labels_napari, name='components_labels')
        # Add first 10 DF/F Z-scored traces plot
        viewer.add_image(data, name='DF_F_Z_scored_plot')

        viewer.add_shapes(
            bbox_rects,
            face_color='transparent',
            edge_color='green',
            properties=properties,
            text=text_parameters,
            name='SNR values',
        )
        
        logging.info("Opened napari viewer")

    # # Get time vector
    # TIME = np.arange(0, ci_movie.shape[0]) / FPS

    # # Convert to pynapple objects
    # ci_movie_pynapple = nap.TsdTensor(TIME, ci_movie, time_units='s')
    # behaviour_video_pynapple = nap.TsdTensor(TIME, behaviour_video, time_units='s')
    # DF_F_pynapple = nap.TsdFrame(TIME, DF_F_Z_scored.T, time_units='s')
    # activity_movie_pynapple = nap.TsdTensor(TIME, activity_movie, time_units='s')

    # logging.info("Converted data to pynapple objects")

    # # Create output directory if it doesn't exist
    # os.makedirs(OUTPUT_DIR, exist_ok=True)
    # logging.info(f"Created output directory: {OUTPUT_DIR}")

    # # Save data
    # ci_movie_pynapple.save(f'{OUTPUT_DIR}/ci_movie_pynapple_{animal}_{session}.npz')
    # behaviour_video_pynapple.save(f'{OUTPUT_DIR}/behaviour_video_pynapple_{animal}_{session}.npz')
    # DF_F_pynapple.save(f'{OUTPUT_DIR}/DF_F_pynapple_{animal}_{session}.npz')
    # activity_movie_pynapple.save(f'{OUTPUT_DIR}/activity_movie_pynapple_{animal}_{session}.npz')
    
    # # Save activity movie as tiff
    # imwrite(f'{OUTPUT_DIR}/activity_movie_{animal}_{session}.tif', activity_movie)
    
    # np.save(f'{OUTPUT_DIR}/components_labels_{animal}_{session}.npy', accepted_components_labels)
    # np.save(f'{OUTPUT_DIR}/pnr_image_{animal}_{session}.npy', pnr_image)
    # for i, beh_type in enumerate(beh_types):
    #     beh_values[i].save(f'{OUTPUT_DIR}/beh_type_{beh_type}_{animal}_{session}.npz')
    # logging.info("Saved data")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process calcium imaging and behavior data.')
    parser.add_argument('animal', type=int, help='Animal number')
    parser.add_argument('session', type=str, help='Session name')
    parser.add_argument('--downsample', action='store_true', help='Flag to downsample data')
    
    args = parser.parse_args()
    
    main(args.animal, args.session, args.downsample)
