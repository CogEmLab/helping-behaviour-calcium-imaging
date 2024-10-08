import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from src.plotting import *

def sample_random_event_cells(signal, session_loader, window, fps, plot_example=False, suffix="", ax=None, n_samples=10, n_iterations=10):
    """
    Sample random event windows from a pre-loaded dataset, compute the most active components, and select a sample.

    This function randomly samples event windows from the pre-loaded `DF_F_Z_scored` dataset, computes the most active components
    within each window, and returns a random sample of the selected components based on their activity.

    Parameters
    ----------
    signal : np.ndarray
        A pre-loaded 2D numpy array (components x time points) representing the activity data (e.g., DF/F, Z-scored signal).
    session_loader : object
        A data loader object that provides methods for computing the most active components.
    window : float
        The window size in seconds to sample around the random event.
    fps : float
        The frame rate (frames per second) of the data. Used to convert time to frames.
    plot_example : bool, optional
        Whether to plot an example of the selected event window and components (default is False).
    suffix : str, optional
        A suffix to append to filenames or plot titles (default is an empty string).
    ax : matplotlib.axis, optional
        A matplotlib axis to plot the example if `plot_example` is True. If None, no plot is created (default is None).
    n_samples : int, optional
        The number of samples to return from the selected components (default is 10).
    n_iterations : int, optional
        The number of iterations or random event windows to sample from (default is 10).

    Returns
    -------
    selected_ids : numpy.ndarray
        A numpy array of randomly selected component ids from the most active components.
    selected_data : numpy.ndarray
        A numpy array of the corresponding data for the selected components.
    """

    # Initialize lists to store selected component IDs and their corresponding data
    selected_ids_all = []
    selected_data_all = []

    # Perform the sampling for a specified number of iterations (default is 10)
    for i in range(n_iterations):
        # Randomly select a time point within the valid range, ensuring the event window fits within the session
        session_length = session_loader.session_length  # Get total session length (in seconds or frames)
        time_victim_crossing = np.random.randint(window * 2, session_length - window * 2)
        
        # Convert the selected time point to the frame index using fps
        time_victim_crossing_frames = int(time_victim_crossing * fps)

        # Calculate the start and end indices of the window (frames)
        start_frame = int(time_victim_crossing_frames - window * fps)
        end_frame = int(time_victim_crossing_frames + window * fps)

        # Extract the data within the specified time window from the pre-loaded DF_F_Z_scored
        signal_window = signal[:, start_frame:end_frame]

        # Sum the activity across time for each component within the window
        sum_data = np.sum(signal_window, axis=1)

        # Get the most active components within this window
        active_data, active_ids = get_most_active_components(signal_window, sum_data, f'figures/extraction_lib_cells/threshold_plots{suffix}', cohort=session_loader.cohort, session=session_loader.session, animal=session_loader.animal_id, ax=ax, suffix=suffix, plot_example=plot_example if i == 0 else False)
        # Randomly sample up to 10 components from the most active ones, ensuring no duplicates
        sampled_ids = np.random.choice(active_ids, min(len(active_ids), 10), replace=False)

        # Append the selected component IDs and corresponding data to the global lists
        selected_ids_all.extend(sampled_ids)
        selected_data_all.extend(active_data)

    # Convert the collected component IDs and data to numpy arrays for easier processing
    selected_ids_all = np.array(selected_ids_all)
    selected_data_all = np.array(selected_data_all)

    # Randomly select n_samples from the global list of selected IDs and their corresponding data
    selected_indices = np.random.choice(len(selected_ids_all), n_samples, replace=False)
    selected_ids = selected_ids_all[selected_indices]
    selected_data = selected_data_all[selected_indices]

    return selected_ids, selected_data

def get_most_active_components(DF_F_Z_scored, dff_sum, path, cohort, session, animal, plot=False, plot_example=False, ax=None, suffix=None):
    """
    Get the most active components in the DF_F_Z_scored based on k*sigma threshold and save a histogram of the sum of the components with a threshold.

    Parameters:
        DF_F_Z_scored (np.DF_F_Z_scored): The input DF_F_Z_scored of shape (n_components, time) cropped around the window of interest.
        dff_sum: The vector of the sums inside the window
        path (str): The path to the file to save the histogram.
        cohort (str): The cohort of the mouse.
        session (str): The session of the mouse.
        animal (str): The animal ID.
        plot (bool): Whether to plot the histogram as separate figure.
        plot_example (bool): Whether to plot an example of the histogram on a bigger grid.
        ax (matplotlib.axis): The axis to plot the histogram on.

    Returns:
        DF_F_Z_scored_selected (np.DF_F_Z_scored): The components that are above the threshold.
        selected_ids (np.array): The ids of the selected components.

    Raises:
        None
    """
    
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
        plot_histogram(dff_sum, threshold, filename, ax=ax, suffix=suffix)
    
    if plot:
        # Add cohort, session and mouse_id to the path
        path = os.path.join(path, cohort, session)
        os.makedirs(path, exist_ok=True)
        filename = os.path.join(path, f'histogram_{animal}.png')
        plot_histogram(dff_sum, threshold, filename, suffix=suffix, legend=False)
    
    return DF_F_Z_scored_selected, selected_ids

def setup_plot(fig_size, grid_spec):
    plt.style.use('seaborn-v0_8-talk')
    fig = plt.figure(figsize=fig_size)
    gs = GridSpec(*grid_spec, figure=fig)
    
    
    # Set larger font sizes
    plt.rcParams.update({
        'font.size': 14,          # General font size
        'axes.titlesize': 18,     # Font size for axes titles
        'axes.labelsize': 16,     # Font size for axes labels
        'xtick.labelsize': 14,    # Font size for x-axis tick labels
        'ytick.labelsize': 14,    # Font size for y-axis tick labels
        'legend.fontsize': 14,    # Font size for legend
        'figure.titlesize': 20    # Font size for figure title
    })

    axes = {
        'heatmap': fig.add_subplot(gs[0:2, 0:2]),
        'histogram': fig.add_subplot(gs[0, 2]),
        'boxplots': fig.add_subplot(gs[0, 3]),
        'components': fig.add_subplot(gs[1, 2:4])
    }
    
    return fig, axes

def load_and_preprocess_data(file_path):
    """
    Load a CSV file and preprocess the data by filtering out specific sessions and cohorts.

    Parameters
    ----------
    file_path : str
        The path to the CSV file to load.

    Returns
    -------
    data : pandas.DataFrame
        The preprocessed data.
    """
    data = pd.read_csv(file_path)

    # # Filter out specific sessions and cohorts
    # excluded_sessions = ['Hab1', 'Hab2', 'Hab3', 'Toy', 'Pool', 'Empty']
    # excluded_cohorts = []

    # data = data[~data['Session'].isin(excluded_sessions) & ~data['Cohort'].isin(excluded_cohorts)] #! This is bad. I am removing these sessions from the main session file. TODO: rewrite

    # Convert Mouse column to str with 4 digits
    data['Mouse'] = data['Mouse'].astype(str).str.zfill(4)

    # Add new columns if they don't exist
    columns_to_add = [
        'Num of liberation cells',
        '% of liberation cells',
        '% of randomd liberation cells',
        'Num of randomd liberation cells'
    ]

    #TODO: need first to check if such a column exist or not already 
    for column in columns_to_add:
        if column not in data.columns:
            data[column] = np.nan

    return data
