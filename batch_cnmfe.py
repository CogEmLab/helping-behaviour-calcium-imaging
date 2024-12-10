import os
import logging
import pandas as pd
from caiman.source_extraction.cnmf import params as params
from src.utils import *
from glob import glob 
from tqdm import tqdm
from tifffile import imwrite
import os
from collections import defaultdict
import datetime
import yaml
from src.plotting import quality_control_plotting

with open('batch_cnmfe_params.yaml', 'r') as f:
    params = yaml.safe_load(f)

paths = [os.path.normpath(p) for p in params['paths']]
session_file_paths = [os.path.normpath(p) for p in params['session_file_paths']]
config_filepath = os.path.normpath(params['config_filepath'])
skip_if_exists = params['skip_if_exists']

# Define required columns
required_columns = ['session', 'pair', 'sex', 'miniscope video', 'ci_fps_downsampled', 'timevideoend(s)', 'session outcome', 'final outcome', 'timevictiminsertion(s)']

for GLOBAL_PATH, session_file_path in zip(paths, session_file_paths):

    # Configure logging, move logging to tech docs, save the path to it and add there time
    log_filename = os.path.join('tech_docs', f'batch_cnmfe_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    # Check if miniscope folder exist, if yes, seach for files
    miniscope_paths = glob(f'{GLOBAL_PATH}/**/miniscope/**.tiff', recursive=True)

    # Define the path to the session file
    session_file_path  = os.path.normpath(os.path.join(GLOBAL_PATH, session_file_path))

    # Read the session file
    session_df = pd.read_csv(session_file_path)
    # Convert all column names to small lettercase
    session_df.columns = [x.lower() for x in session_df.columns]
    
    # Check if all required columns are present
    if not all(x in session_df.columns for x in required_columns):
        logging.warning(f'Session file {session_file_path} is missing one or more of the required columns: {required_columns}. Skipping it.')
        
        
    SNR_dict = defaultdict(list)
    max_shifts_dict = defaultdict(list)

    for path in tqdm(miniscope_paths):
        
        # Extract session and mouse name from path
        session = path.split('/')[2]
        filename = os.path.splitext(os.path.basename(path))[0]

        # Log the filename and path
        logging.info(f'Processing {filename} in {session}')

        # Read FPS from the session file
        session_entry = session_df[session_df['miniscope video'] == filename]
        if not session_entry.empty:
            FPS = int(session_entry['ci_fps_downsampled'].values[0])
            logging.info(f'Setting FPS to {FPS} based on session file.')
        else:
            FPS = 30 # Assuming default=30
            logging.info(f'No FPS found in session file, setting to default=30.')

        # Find index of session file entry with the given filename in Miniscope video TIFF column
        idx = session_df.index[session_df['miniscope video'] == filename].tolist()
        if len(idx) != 1:
            logging.error(f'No single entry found for {filename} in the session file. Cannot extract needed metadata.')
            continue
        session_entry = session_df.loc[idx[0]]
        
        # Assert that session in sesison entry corresponds to session
        assert session_entry['session'].values[0] == session

        if session_entry.shape[0] == 0:
            logging.error(f'No entry found for {filename} in the session file. Cannot extract needed metadata.')
            # If no entry found, skip to the next file but write to the txt file the filename and path
            with open('no_entry_found.txt', 'a') as f:
                f.write(f'{filename} {path}\n')
            continue

        # Create save directory
        OUTPUT_DIR = os.path.normpath(GLOBAL_PATH.replace('data', 'processed-data'))

        save_dir_model = os.path.join(OUTPUT_DIR, session, 'cnmfe_model')
        save_dir_data = os.path.join(OUTPUT_DIR, session, 'cnmfe_data')
        save_dir_docs = os.path.join(OUTPUT_DIR, session, 'cnmfe_docs')
        save_dir_npy = os.path.join(OUTPUT_DIR, session, 'npy_data')
        save_dir_quality_control = os.path.join(OUTPUT_DIR, session, 'quality_control')

        os.makedirs(save_dir_model, exist_ok=True)
        os.makedirs(save_dir_data, exist_ok=True)
        os.makedirs(save_dir_docs, exist_ok=True)
        os.makedirs(save_dir_npy, exist_ok=True)
        os.makedirs(save_dir_quality_control, exist_ok=True)

        # Extract the mouse name from the session file
        mouse = str(session_entry['pair'].values[0].split('-')[0])

        # Extract the session duration from the session file
        if ':' in str(session_entry['timevideoend(s)'].values[0]):
            session_duration = float(f'{int(session_entry["timevideoend(s)"].values[0].split(":")[0])*60 + int(session_entry["timevideoend(s)"].values[0].split(":")[1]):.2f}')
        else:
            session_duration = float(session_entry['timevideoend(s)'].values[0])

        # Extract outcome of the session from the session file
        session_outcome = session_entry['session outcome'].values[0]
        final_outcome = session_entry['final outcome'].values[0]
        
        # Extract the time of victim insertion from the session file
        time_victim_inserted = session_entry['timevictiminsertion(s)'].values[0]
        if ':' in str(time_victim_inserted):
            time_victim_inserted = float(f'{int(time_victim_inserted.split(":")[0])*60 + int(time_victim_inserted.split(":")[1]):.2f}')
        else:
            time_victim_inserted = float(time_victim_inserted)
        
        # Extract sex of the animal 
        sex = session_entry['sex'].values[0]

        logging.info(f'Processing {session} {mouse}, Sex: {sex}, Session duration: {session_duration}, Session outcome: {session_outcome}, Final outcome: {final_outcome}, Time victim inserted: {time_victim_inserted}')

        # Check if the file has already been processed only
        if os.path.exists(os.path.join(save_dir_model, f'{filename}.hdf5')) and skip_if_exists:
            logging.info(f'File {filename} has already been processed. Skipping it.')
            continue

        try:
            cnmfe_model, ci_movie, fname_new, correlation_image, peak_to_noise_ratio, dims, total_components, num_accepted, num_rejected, border_px = analyze_data(path, save_dir_docs, config_filepath, fps=FPS, motion_correct=True, plot=True)
        except Exception as e:
            logging.error(f'Error processing {session} {mouse}: {str(e)}')

         # Get dF/F; detrend only (1p protocol)
        cnmfe_model.estimates.detrend_df_f(detrend_only=True)
        # Get dF/F
        DF_F = cnmfe_model.estimates.F_dff
        # Check if dF/F shape matches the number of accepted components after component selection
        assert DF_F.shape[0] == num_accepted
        # ^ Good that you check this!

        # This is the work-around for the non-helper animal (848) where session is damaged after certain moment.
        if '2021-10-27-10-25-03_video' in filename:
            logging.info('For file %s, cutting dF/F to 6000 frames', filename)
            DF_F = DF_F[:, :6000]
        elif '2021-10-29-10-43-11_video' in filename:
            logging.info('For file %s, cutting dF/F to 8000 frames', filename)
            DF_F = DF_F[:, :8000]

        # Normalize dF/F to have a mean of 0 and a standard deviation of 1
        DF_F_Z_scored = (DF_F - DF_F.mean(axis=1)[:, None]) / DF_F.std(axis=1)[:, None]
        # ^ I saw your traces and this seems to be OK but when I did my analysis earlier where I only used the S Matrix for later steps
        # ^ I have normalized using the SNR value that you also obtain from Caiman. This could help you to filter out noisy data.
        # ^ Maybe keep it as an option for later, or to experiment with it.

        # Save cnmfe model
        cnmfe_save_dir = os.path.join(save_dir_model, f'{filename}.hdf5')
        cnmfe_model.save(cnmfe_save_dir)
        
        # Save correlation image
        correlation_image_path = os.path.join(save_dir_data, f'correlation_image_{filename}.tif')
        imwrite(correlation_image_path, correlation_image)
        
        # Save peak-to-noise ratio image
        peak_to_noise_ratio_path = os.path.join(save_dir_data, f'peak_to_noise_ratio_{filename}.tif')
        imwrite(peak_to_noise_ratio_path, peak_to_noise_ratio)
        
        # Save dF/F
        np.save(os.path.join(save_dir_npy, f'{filename}.npy'), DF_F_Z_scored)
        
        # Copy the motion corrected file to the save directory
        save_motion_corrected = os.path.join(os.path.dirname(path).replace('data', 'processed-data').replace('miniscope', 'cnmfe_data'), os.path.basename(fname_new))
        print(save_motion_corrected)
        os.makedirs(os.path.dirname(save_motion_corrected), exist_ok=True)
        os.system(f'cp {fname_new} {save_motion_corrected}')

        # Delete the motion corrected file
        os.system(f'rm {fname_new}')

        # Get SNR values (for accepted components only)
        SNR = cnmfe_model.estimates.SNR_comp

        # Create a dictionary to save SNR values for each animal and session
        SNR_dict[mouse].extend(SNR)
    
        # Similarly, get the maximum shift values
        max_shifts_dict[mouse].extend(border_px)

        # Check if SNR values contain only int, if not, remove that component from accepted_components
        if not all(isinstance(x, int) for x in SNR):

            try:
                plot_random_traces(DF_F_Z_scored, snr=SNR, filename=os.path.join(save_dir_docs, f'{filename}_random_traces.png'))
                logging.info('Plot saved to: %s', save_dir_docs)
            except Exception as e:
                logging.error('Error: %s', e)
                logging.error('Could not plot random traces for %s', filename)
                
        session_df.loc[idx[0], 'Total Components'] = total_components
        session_df.loc[idx[0], 'Num Accepted'] = num_accepted
        session_df.loc[idx[0], 'Num Rejected'] = num_rejected
        session_df.loc[idx[0], 'Dimensions'] = dims
        session_df.loc[idx[0], 'Correlation Image Path'] = correlation_image_path
        session_df.loc[idx[0], 'Peak-to-Noise Ratio Path'] = peak_to_noise_ratio_path
        session_df.loc[idx[0], 'CNMFE Model Path'] = cnmfe_save_dir
        session_df.loc[idx[0], 'Motion Corrected Path'] = fname_new
        session_df.loc[idx[0], 'Maximum Shift'] = float(border_px)
        session_df.loc[idx[0], 'CNMFE Log File Location'] = log_filename

        now = datetime.now().strftime('%Y-%m-%d')
        session_df.to_csv(f'{GLOBAL_PATH}/SESSION_FILE_{os.path.basename(__file__)[:-3]}_{now}.csv', index=False)

    logging.info(f'Batch CNMFE processing complete for cohort {GLOBAL_PATH}')

    quality_control_plotting(SNR_dict, max_shifts_dict, save_dir_quality_control)

    logging.info(f'Quality control plots saved to {save_dir_quality_control}')

    # Normalize number of components
    session_df['Normalized components'] = session_df['Num Accepted'] / session_df['Ca Imaging Length(s)']
    # For each unique mouse, sum the normalized components over all days
    sum_normalized_components = session_df.groupby(['Mouse'])['Normalized components'].sum()
    # Add the sum of normalized components to the data
    session_df = session_df.join(sum_normalized_components, on='Mouse', rsuffix='_sum')
    # Compute K_id(Num Accepted components for animal i and day d) as k_id (normalized by Session Duration) /sum(k_il over all days) for each animal i and day d
    session_df['Normalized components'] = (session_df['Normalized components'] / session_df['Normalized components_sum']) * 100
    # Remove the sum of normalized components
    session_df = session_df.drop(columns=['Normalized components_sum'])
    # Round Ca Imaging Length(s) and Normalized components to 2 decimal places
    session_df['Ca Imaging Length(s)'] = session_df['Ca Imaging Length(s)'].round(2)
    session_df['Normalized components'] = session_df['Normalized components'].round(2)

    # Save session file
    now = datetime.now().strftime('%Y-%m-%d')
    session_df.to_csv(f'{GLOBAL_PATH}/SESSION_FILE_{os.path.basename(__file__)[:-3]}_{now}.csv', index=False)

    logging.info(f'Session file saved to {GLOBAL_PATH}')