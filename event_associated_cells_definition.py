#TODO: implemenet clean function, it should check for already existing outputs of this function and put them to the log folder
"""
This script extracts the most active cells in a time window around the moment
of liberation of the victim mouse. It processes each session in the provided
session file, computes the sum of dff in the time window for each component,
and selects the components with the highest sum.

The selected components are then plotted as a heatmap, and the response scores
are calculated for each component. The response scores are then plotted as a
boxplot.

The script also saves the results to a CSV file, and plots an overview figure
showing the percentage of liberation cells for each mouse and day of helping.

The script can be run in two modes: with the random flag set to True, it will
sample random events and most active components for these events. With
the random flag set to False, it will compute the response scores for the
actual liberation events.

The script assumes that the session file is in the format produced by the
session_loader script, and that the annotations are in the format produced by
the annotation script."""

import logging

import matplotlib
matplotlib.use('Agg')  # Set the backend to 'Agg' before importing pyplot

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from src.session_loader import Session
from src.plotting import plot_heatmap, plot_boxplots, plot_values_over_days
from src.event_associated_cells_definition_utils import *
from src.update_registration_data import update_metadata
import datetime

# Constants
FIG_SIZE = (30, 12)
GRID_SPEC = (2, 4)
EXCLUDED_SESSIONS = ['Hab1', 'Hab2', 'Hab3', 'Toy', 'Pool', 'Empty']
WINDOW = 30 # seconds

def process_session(row, WINDOW, random=False, plot_example=False, axes=None):
    #TODO: also check the column "excluded", if it is there, remove this session from the analysis
    '''
    window given in seconds and full
    '''
    # Extract necessary data from the row
    session_loader = Session(row)
    
    # Skip if session outcome is not 'H'. Neural activity for helping and non-helping sessions can be assumed to be different, no need to mix these data together. (Duch's comment 2024 MidTermEvaluation)
    if session_loader.session in EXCLUDED_SESSIONS or session_loader.session_outcome != 'H' or np.isnan(session_loader.victim_lib_timecode):
        logging.warning(f"Skipping session {session_loader.session} with outcome {session_loader.session_outcome}")
        return np.nan, np.nan, np.nan

    # Load CNMF model and process data; only accepted components here 
    DF_F_Z_scored = session_loader.load_dff_signal()

    # Load registration data
    reg_matrix, reg_indexing = session_loader.load_registration_files() 
    reg_vector = reg_matrix[:, reg_indexing[session_loader.session]]

    # Load the neuron labels to extract the position of the cells
    labels = session_loader.get_neuron_labels()
    positions = session_loader.extract_neuron_positions(labels)

    # Get start and end of the window
    start, end = generate_event_window(WINDOW, session_loader.victim_lib_timecode, session_loader.calcium_fps)
    logging.info('Using window size of {} seconds for data with FPS of {}'.format(WINDOW, session_loader.calcium_fps))

    # Ensure indices are within bounds
    if start < 0 or end > DF_F_Z_scored.shape[1]:
        logging.warning(f"Skipping session {session_loader.session} with victim liberation timecode {session_loader.victim_lib_timecode}")
        return np.nan, np.nan, np.nan
    # ^instead of returning nan, you could go to the maximum possible window length, that could give you more data
    
    logging.info(f"Processing window: {start} - {end} in {DF_F_Z_scored.shape[1]} frames")

    # Cut DF_F_Z_scored to the time window
    DF_F_Z_scored_window = DF_F_Z_scored[:, start:end]

    logging.info(f'Current shape of DF_F_Z_scored: {DF_F_Z_scored.shape}')

    # Compute the sum of dff inside the window
    sum_dff = np.sum(DF_F_Z_scored_window, axis=1)

    # Get most active components
    if random:
        suffix = '_random_event'

        # Compute the real data around liberation window to predict the correct number of cells required
        ___, selected_ids_true = get_most_active_components(
            DF_F_Z_scored_window, sum_dff, f'figures/extraction_lib_cells/threshold_plots{suffix}', cohort=session_loader.cohort, session=session_loader.session, animal=session_loader.animal_id, plot_example=False, ax=axes['histogram'], suffix=suffix)

        logging.info(f"Sampling {len(selected_ids_true)} components from random event time")
        
        # Sample random components
        selected_ids, DF_F_Z_scored_selected = sample_random_event_cells(DF_F_Z_scored, session_loader, session_loader.calcium_fps, WINDOW, plot_example=plot_example, suffix=suffix, ax=axes['histogram'], n_samples=len(selected_ids_true))
        
        # Make sure that random selected ids has the same length as selected_ids_true
        assert len(selected_ids) == len(selected_ids_true)

    else:
        suffix = ''
        # Get most active components in the liberation window
        DF_F_Z_scored_selected, selected_ids = get_most_active_components(
                DF_F_Z_scored_window, sum_dff, f'figures/extraction_lib_cells/threshold_plots{suffix}', cohort=session_loader.cohort, session=session_loader.session, animal=session_loader.animal_id, plot=True, plot_example=plot_example, ax=axes['histogram'], suffix=suffix)

    logging.info(f"Selected {len(selected_ids)} components")

    # Take only reg IDs where reg_vector values are in selected components 
    reg_ids = np.array([np.where(reg_vector == i)[0][0] for i in selected_ids])

    # Check that reg_ids len is equal to selected_ids len
    assert len(reg_ids) == len(selected_ids)

    # Plot heatmaps for selected components
    filename = f'figures/extraction_lib_cells/heatmaps{suffix}/{session_loader.cohort}/{session_loader.session}/{session_loader.animal_id}_heatmap{suffix}.png'
    # Create a directory for the cohort and session
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    # Plot heatmap
    plot_heatmap(DF_F_Z_scored_selected, filename=filename, ax=None, window_to_use=WINDOW, FPS=session_loader.calcium_fps, animal=session_loader.animal_id, day_of_helping=row['DayofHelping'])

    # Plot again if plot_example is True on heatmap axis of the setup_plot
    if plot_example:
        filename = None
        plot_heatmap(DF_F_Z_scored_selected, filename=filename, ax=axes['heatmap'], window_to_use=WINDOW, FPS=session_loader.calcium_fps, animal=session_loader.animal_id, day_of_helping=row['DayofHelping'])
    
    # Calculate response scores
    print("Calculating response scores")
    print(DF_F_Z_scored_selected.shape)
    
    pre_response = np.absolute(np.sum(DF_F_Z_scored_selected[:, :int((WINDOW*session_loader.calcium_fps) / 2)], axis=1)) # Add absolute value getting out of the bounds
    post_response = np.absolute(np.sum(DF_F_Z_scored_selected[:, int((WINDOW*session_loader.calcium_fps) / 2):], axis=1))
    total_response = np.add(pre_response, post_response)

    # Remove components with zero total response
    mask = total_response != 0
    if np.any(mask):
        response_score = (post_response[mask] - pre_response[mask]) / total_response[mask]
        # ^ check if absolute value is not needed here
    else:
        response_score = np.nan

    logging.info(f"Mask shape: {mask.shape}, response score shape: {response_score.shape}")

    # Check again that objects exist and not empty before returning
    if selected_ids.size == 0:
        return np.nan, np.nan, np.nan

    # Separate sum_dff into selected and not selected components (for plotting)
    sum_dff_selected = sum_dff[selected_ids]
    sum_dff_not_selected = sum_dff[~np.isin(np.arange(len(sum_dff)), selected_ids)]

    # Extract position data
    positions_selected = positions[selected_ids]

    return ({
        'selected_ids': selected_ids[mask] if selected_ids.size > 0 else np.nan,
        'reg_ids': reg_ids[mask] if reg_ids.size > 0 else np.nan,
        'response_score': response_score if selected_ids.size > 0 else np.nan,
        'num_liberation_cells': len(selected_ids[mask]),
        'percent_liberation_cells': (len(selected_ids[mask]) / session_loader.number_of_accepted_components) * 100 if session_loader.number_of_accepted_components > 0 else np.nan,
        'sum_dff_selected': sum_dff_selected[mask] if sum_dff_selected.size > 0 else np.nan,
        'positions_selected': positions_selected[mask] if positions_selected.size > 0 else np.nan
    }, sum_dff_selected, sum_dff_not_selected)

def main(random=False):
    """
    Main function to process sessions and extract event associated cells.
    
    Parameters
    ----------
    random : bool, optional
        If True, process sessions using random event extraction. Random event is generated by taking the window around random moment (n times) and then performing random sampling to get same number of cells as otherwise would be for the liberation event. 
        If False, process sessions using liberation event extraction. Default is False.
    """

    # If random mode of processing is True, set respective suffix to save all results
    if random:
        event_type = 'Random'
        suffix = '_random'
    else:
        event_type = 'Liberation'
        suffix = ''

    fig, axes = setup_plot(FIG_SIZE, GRID_SPEC) # Setup of the overview plot

    # Load the session file
    session_file = load_and_preprocess_session_file(dir_path='data', event_type=event_type)

    logging.info('Loading session file')

    # Initialize lists for metadata and components; extending it by information about the position and the sum of activity
    metadata = {
        'Cohort': [], 'Mouse': [], 'Session': [], 'Component ID': [],
        'Reg ID': [], 'Response Score': [], 'DayofHelping': [],
        'CNMFE Model Path': [], 'Position (x)' : [], 'Position (y)': [], 'Sum of activity': []
    }
    
    # Initialize lists for sum of activity in the selected and not selected components (this will only be needed for plotting the control image)
    sum_components = {'selected': [], 'not_selected': []}
    
    # Make a random choice for the example plot
    random_idx_to_plot = np.random.choice(session_file.index)

    # Process each session
    for i, row in tqdm(session_file.iterrows(), total=len(session_file)):
        
        logging.info('Plotting index', random_idx_to_plot, ', Current index:', i)

        # Plot example if random index
        plot_example = int(i) == int(random_idx_to_plot) # boolian flag to define whether to plot the example
        
        row_results, sum_dff_selected, sum_dff_not_selected = process_session(row, WINDOW, random=random, plot_example=plot_example, axes=axes)  # Get the results
                
        if row_results is not np.nan:
            # Update metadata
            for key in ['Cohort', 'Mouse', 'Session', 'DayofHelping', 'CNMFE Model Path']:
                metadata[key].extend([row[key]] * len(row_results['selected_ids']))
                        
            metadata['Component ID'].extend(row_results['selected_ids']) # Upload IDs as from the model to specify which cell was selected
            metadata['Reg ID'].extend(row_results['reg_ids']) # Upload Reg IDs to specify at which row one can find this cell in the registration matrix
            metadata['Response Score'].extend(row_results['response_score']) # Upload response scores
            
            # Fill in the position x, position y and sum of activity columns
            metadata['Position (x)'].extend(row_results['positions_selected'][:, 0])
            metadata['Position (y)'].extend(row_results['positions_selected'][:, 1])
            metadata['Sum of activity'].extend(row_results['sum_dff_selected'])

            # Update session file with liberation cell information
            session_file.loc[session_file.index == row.name, f'Num of {event_type} cells'] = row_results['num_liberation_cells']
            session_file.loc[session_file.index == row.name, f'% of {event_type} cells'] = row_results['percent_liberation_cells']
            
            # Update sum of components with mean values for control plotting
            sum_components['selected'].append(np.mean(sum_dff_selected))
            sum_components['not_selected'].append(np.mean(sum_dff_not_selected))

        else:
            # If plotting was true, but the session was not processed, set the random plotting index to the next session
            if plot_example==True:
                print('Skipping plotting session with index:', i)
                random_idx_to_plot = i + 1

        # Reset the plot_example flag
        plot_example=False   

    logging.info('Finished processing all sessions')

    # Convert metadata to DataFrame and save to CSV
    metadata_df = pd.DataFrame(metadata)
    # Create the value for Group to simplify later analysis

    metadata_df['Group'] = pd.cut(metadata_df['Response Score'], 
                                  bins=[-np.inf, -0.33, 0.33, np.inf],
                                  labels=['Pre-responsive', 'Mixed', 'Post-responsive'])

    # Add an index column and make it the first column
    metadata_df.insert(0, 'Index', range(1, 1 + len(metadata_df)))

    # Update metadata
    metadata_df = update_metadata(metadata_df)
    
    # Save metadata to CSV
    current_datetime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    metadata_df_path = f'processed-data/METADATA_{event_type}_{os.path.basename(__file__)[:-3]}_{current_datetime}.csv'
    metadata_df.to_csv(metadata_df_path, index=False)

    # # Update session file with normalized liberation cells
    # col_name_3 = 'Normalized components' if not random else 'Normalized random event components'

    # # Normalize using the save formula K_d = \frac{n_d}{\sum_{i}^S n_{i}} \times 100 (($n_{d}$) is normalized number of components in day)
    # session_file[col_name_3] = session_file[col_name_1] / session_file['Ca Imaging Length(s)']

    # # K_d = \frac{n_d}{\sum_{i}^S n_{i}} \times 100 implementation
    # sum_normalized = session_file.groupby('Mouse')[col_name_3].sum()
    # session_file = session_file.join(sum_normalized, on='Mouse', rsuffix='_sum')
    # session_file[col_name_3] = (session_file[col_name_3] / session_file[f'{col_name_3}_sum']) * 100

    # # Remove the sum of normalized components
    # session_file = session_file.drop(columns=[f'{col_name_3}_sum'])

    # Add new column to session file "{EVENT} Associated Cells Metadata Path" if it doesn't exist yet. If it does, overwrite it with the currect path. Replace Event with Liberation if suffix = '', otherwise with Random Event
    col_name = f'{event_type} Associated Cells Metadata Path'
    if col_name not in session_file.columns:
        session_file[col_name] = ''
    session_file[col_name] = metadata_df_path #The metadata path is common for all sessions, however, it is better to make it easier to work with the loader

    # Save updated session file
    session_file.to_csv(f'data/SESSION_FILE_{os.path.basename(__file__)[:-3]}_{current_datetime}.csv', index=False)

    # Plot boxplots to show the difference between selected and not selected cells in terms of sum of the activity in the window
    plot_boxplots(sum_components['selected'], sum_components['not_selected'], axes['boxplots'], '', ('Selected', 'Not selected'), 'Sum of $\\Delta F/F$ values in the window')
    # Plot the values over days for each mouse
    plot_values_over_days(session_file, 'DayofHelping', f'% of {event_type} cells', ax=axes['components'], order=[1, 2, 3, 4, '5+', 'Stranger', 'RECNvlCtx', 'REMNvlCtx', 'REMHBTD'], title='Liberation Cells', hue_column='Mouse', filename=None)

    # Save the figure
    plt.tight_layout()

    # Save the overview figure
    plt.savefig(f'tech_plots/OVERVIEW_FIGURE_{os.path.basename(__file__)[:-3]}_{current_datetime}{suffix}.pdf')
    # Also save as PNG
    plt.savefig(f'tech_plots/OVERVIEW_FIGURE_{os.path.basename(__file__)[:-3]}_{current_datetime}{suffix}.png')

if __name__ == "__main__":
    main(random=True)