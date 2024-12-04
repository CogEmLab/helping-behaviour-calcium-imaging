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

# Input directories
paths = [os.path.normpath(p) for p in ['data/Oct_2024_synthetic']]
session_file_paths = [os.path.normpath(p) for p in ['Oct_2024_synthetic_session_file.csv']]
config_filepath = os.path.normpath('tech_docs/caiman_parameters.yaml')

# Define required columns
required_columns = ['session', 'pair', 'sex', 'miniscope video', 'ci_fps_downsampled', 'timevideoend(s)', 'session outcome', 'final outcome', 'timevictiminsertion(s)']

for GLOBAL_PATH, session_file_path in zip(paths, session_file_paths):

    # Configure logging
    logging.basicConfig(filename='batch_cnmfe_synthetic.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Check if miniscope folder exist, if yes, seach for files
    miniscope_paths = glob(f'{GLOBAL_PATH}/**/miniscope/**.tiff', recursive=True)

    # Create an output doc path, we will save all metadata information there
    output_file_path = os.path.normpath(os.path.join(GLOBAL_PATH, 'cnmfe_output_updated.csv'))

    if os.path.isfile(output_file_path):
        # Read the existing output file
        output_df = pd.read_csv(output_file_path)
        # In Mouse column, if there are only 3 characters, add a 0 in front
        output_df['Mouse'] = [str(x).zfill(4) for x in output_df['Mouse']]
        # Convert the dataframe to a defaultdict list
        output_dict = output_df.to_dict(orient='list', into=defaultdict(list))
    else:
        # Create an empty defaultdict list
        output_dict = defaultdict(list)

    # Define the path to the session file
    session_file_path  = os.path.normpath(os.path.join(GLOBAL_PATH, session_file_path))

    # Read the session file
    session_df = pd.read_csv(session_file_path)
    # Convert all column names to small lettercase
    session_df.columns = [x.lower() for x in session_df.columns]
    
    # Check if all required columns are present
    if not all(x in session_df.columns for x in required_columns):
        logging.warning(f'Session file {session_file_path} is missing one or more of the required columns: {required_columns}. Skipping it.')
        continue
    
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

        # Find in session file entry with the given filename in Miniscope video TIFF column
        session_entry = session_df[session_df['miniscope video'] == filename]
        
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

        os.makedirs(save_dir_model, exist_ok=True)
        os.makedirs(save_dir_data, exist_ok=True)
        os.makedirs(save_dir_docs, exist_ok=True)

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

        # Check if the file has already been processed only if there is an output file
        if os.path.isfile(output_file_path):
            if output_df[(output_df['Session'] == session) & (output_df['Mouse'] == mouse)].shape[0] > 0: 
                logging.info(f'{session} {mouse} has already been processed')
                continue
            else:
                logging.info(f'{session} {mouse} has not been processed yet')

        try:
            cnmfe_model, ci_movie, fname_new, correlation_image, peak_to_noise_ratio, dims, total_components, num_accepted, num_rejected, border_px = analyze_data(path, save_dir_docs, config_filepath, fps=FPS, motion_correct=True, plot=True)
        except Exception as e:
            logging.error(f'Error processing {session} {mouse}: {str(e)}')

        # Save cnmfe model
        cnmfe_save_dir = os.path.join(save_dir_model, f'{filename}.hdf5')
        cnmfe_model.save(cnmfe_save_dir)
        
        # Save correlation image
        correlation_image_path = os.path.join(save_dir_data, f'correlation_image_{filename}.tif')
        imwrite(correlation_image_path, correlation_image)
        
        # Save peak-to-noise ratio image
        peak_to_noise_ratio_path = os.path.join(save_dir_data, f'peak_to_noise_ratio_{filename}.tif')
        imwrite(peak_to_noise_ratio_path, peak_to_noise_ratio)
        
        # Copy the motion corrected file to the save directory
        save_motion_corrected = os.path.join(os.path.dirname(path).replace('data', 'processed-data').replace('miniscope', 'cnmfe_data'), os.path.basename(fname_new))
        print(save_motion_corrected)
        os.makedirs(os.path.dirname(save_motion_corrected), exist_ok=True)
        os.system(f'cp {fname_new} {save_motion_corrected}')

        # Delete the motion corrected file
        os.system(f'rm {fname_new}')

        # Append the information to each column accordingly
        output_dict['Session'].append(session)
        output_dict['Mouse'].append(str(mouse).zfill(4))
        output_dict['Total Components'].append(total_components)
        output_dict['Num Accepted'].append(num_accepted)
        output_dict['Num Rejected'].append(num_rejected)
        output_dict['FPS'].append(FPS)
        output_dict['Dimensions'].append(dims)
        output_dict['Correlation Image Path'].append(correlation_image_path)
        output_dict['Peak-to-Noise Ratio Path'].append(peak_to_noise_ratio_path)
        output_dict['CNMFE Model Path'].append(cnmfe_save_dir)
        output_dict['Motion Corrected Path'].append(fname_new)
        output_dict['Session Duration'].append(session_duration)
        output_dict['Session Outcome'].append(session_outcome)
        output_dict['Final Outcome'].append(final_outcome)
        output_dict['Time Victim Inserted'].append(time_victim_inserted)
        output_dict['Sex'].append(sex)
        output_dict['Maximum Shift'].append(float(border_px))

        # Assert that all lists have the same size
        assert all(len(x) == len(output_dict['Session']) for x in output_dict.values()), 'All lists should have the same size'
        
        # Save the output dataframe
        output_df = pd.DataFrame(output_dict)
        output_df.to_csv(f'{GLOBAL_PATH}/cnmfe_output_updated.csv', index=False)
        
    logging.info(f'Batch CNMFE processing complete for cohort {GLOBAL_PATH}')

