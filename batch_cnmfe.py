# TODO: also read a session file and extract total duration of the session from there
# TODO: save outputs to a separate folder but similar structure
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

# Set the global path
#GLOBAL_PATH = 'data/dHPC/May_2024'

paths = ['data/Oct_2021']
session_file_paths = ['session_file_MiniscopeCW_10_2021_Moises.csv']

for GLOBAL_PATH, session_file_path in zip(paths, session_file_paths):

    # Configure logging
    logging.basicConfig(filename='batch_cnmfe.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    miniscope_paths = glob(f'{GLOBAL_PATH}/**/miniscope/**.tiff', recursive=True)

    output_file_path = f'{GLOBAL_PATH}/cnmfe_output_updated.csv'

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

    #session_file_path = 'data/dHPC/Jan_2024/session_file_MiniscopeCW_01_2024_Moises.csv'
    session_file_path  = os.path.join(GLOBAL_PATH, session_file_path)
    # Read the session file
    session_df = pd.read_csv(session_file_path)
    
    for path in tqdm(miniscope_paths):

        if path == f'{GLOBAL_PATH}/HBTD1/miniscope/2021-10-23-13-10-46_video.tiff' or path == f'{GLOBAL_PATH}/HBTD1/miniscope/2021-10-23-13-36-07_video.tiff' or path == f'{GLOBAL_PATH}/HBTD1/miniscope/2021-10-23-14-33-10_video.tiff':
            FPS = 20
        else:
            FPS = 30
        # ^ Not highest priority, but I would recommend to load the FPS from the session file, to not rely on hardcoded values in single scripts
        
        # Extract session and mouse name from path
        session = path.split('/')[2]
        filename = os.path.splitext(os.path.basename(path))[0]

        # Log the filename and path
        logging.info(f'Processing {filename} in {session}')

        # Find in session file entry with the given filename in Miniscope video TIFF column
        session_entry = session_df[session_df['Miniscope video'] == filename]
        
        # Assert that session in sesison entry corresponds to session
        assert session_entry['Session'].values[0] == session

        # Create save directory
        OUTPUT_DIR = GLOBAL_PATH.replace('data', 'processed-data')

        save_dir_model = os.path.join(OUTPUT_DIR, session, 'cnmfe_model')
        save_dir_data = os.path.join(OUTPUT_DIR, session, 'cnmfe_data')
        save_dir_docs = os.path.join(OUTPUT_DIR, session, 'cnmfe_docs')

        # # First check if the directories exist, if they do, skip to the next file
        # if os.path.isdir(save_dir_model) and os.path.isdir(save_dir_data) and os.path.isdir(save_dir_docs):
        #     logging.info(f'{filename} has already been processed')
        #     continue

        os.makedirs(save_dir_model, exist_ok=True)
        os.makedirs(save_dir_data, exist_ok=True)
        os.makedirs(save_dir_docs, exist_ok=True)

        if session_entry.shape[0] == 0:
            logging.error(f'No entry found for {filename} in session file')
            # If no entry found, skip to the next file but write to the txt file the filename and path
            with open('no_entry_found.txt', 'a') as f:
                f.write(f'{filename} {path}\n')
            continue

        else:
             # Extract the mouse name from the session file
            mouse = str(session_entry['Pair'].values[0].split('-')[0])
            # Extract the session duration from the session file
            session_duration = session_entry['TimeVideoEnd'].values[0] 
            # If not nan, convert mm:ss to seconds format and to float with 2 decimal points
            if not pd.isna(session_duration):
                session_duration = float(f'{int(session_duration.split(":")[0])*60 + int(session_duration.split(":")[1]):.2f}')
            logging.info(f'Session duration: {session_duration}')
            session_outcome = session_entry['Session outcome'].values[0]
            final_outcome = session_entry['Final outcome'].values[0]
            time_victim_inserted = session_entry['TimeVictimInsertion'].values[0] 
            # If not nan, convert mm:ss to seconds format and to float with 2 decimal points
            if not pd.isna(time_victim_inserted):
                time_victim_inserted = float(f'{int(time_victim_inserted.split(":")[0])*60 + int(time_victim_inserted.split(":")[1]):.2f}')
            sex = session_entry['Sex'].values[0]

        logging.info(f'Processing {session} {mouse}')

        # Check if the file has already been processed only if there is an output file
        if os.path.isfile(output_file_path):
            if output_df[(output_df['Session'] == session) & (output_df['Mouse'] == mouse)].shape[0] > 0: 
                logging.info(f'{session} {mouse} has already been processed')
                continue
            else:
                logging.info(f'{session} {mouse} has not been processed yet')

        try:
            cnmfe_model, ci_movie, fname_new, correlation_image, peak_to_noise_ratio, dims, total_components, num_accepted, num_rejected = analyze_data(path, save_dir_docs, fps=FPS, motion_correct=True, plot=True)
        except Exception as e:
            logging.error(f'Error processing {session} {mouse}: {str(e)}')
        # ^ I would recommend you to save also the border_px variable (max shift in x/y during registration) so you can later 
        # ^ remove areas with very high shift values where likely you could have some motion artefacts from the experiment. 

        # Save cnmfe model
        cnmfe_save_dir = os.path.join(save_dir_model, f'{filename}.hdf5')
        cnmfe_model.save(cnmfe_save_dir)
        
        # Save correlation image
        correlation_image_path = os.path.join(save_dir_data, f'correlation_image_{filename}.tif')
        imwrite(correlation_image_path, correlation_image)
        
        # Save peak-to-noise ratio image
        peak_to_noise_ratio_path = os.path.join(save_dir_data, f'peak_to_noise_ratio_{filename}.tif')
        imwrite(peak_to_noise_ratio_path, peak_to_noise_ratio)
        
        # copy the motion corrected file to the save directory
        save_motion_corrected = fname_new.replace('data', 'processed-data').replace('miniscope', 'cnmfe_data')
        os.makedirs(os.path.dirname(save_motion_corrected), exist_ok=True)
        os.system(f'cp {fname_new} {save_motion_corrected}')

        # delete the motion corrected file
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
        output_dict['Motion Corrected Path'].append(save_motion_corrected)
        output_dict['Session Duration'].append(session_duration)
        output_dict['Session Outcome'].append(session_outcome)
        output_dict['Final Outcome'].append(final_outcome)
        output_dict['Time Victim Inserted'].append(time_victim_inserted)
        output_dict['Sex'].append(sex)

        print(output_dict)

        # Print all lengths of the lists
        print(f'Session: {len(output_dict["Session"])}')
        print(f'Mouse: {len(output_dict["Mouse"])}')
        print(f'Total Components: {len(output_dict["Total Components"])}')
        print(f'Num Accepted: {len(output_dict["Num Accepted"])}')
        print(f'Num Rejected: {len(output_dict["Num Rejected"])}')
        print(f'FPS: {len(output_dict["FPS"])}')
        print(f'Dimensions: {len(output_dict["Dimensions"])}')
        print(f'Correlation Image Path: {len(output_dict["Correlation Image Path"])}')
        print(f'Peak-to-Noise Ratio Path: {len(output_dict["Peak-to-Noise Ratio Path"])}')
        print(f'CNMFE Model Path: {len(output_dict["CNMFE Model Path"])}')
        print(f'Motion Corrected Path: {len(output_dict["Motion Corrected Path"])}')
        print(f'Session Duration: {len(output_dict["Session Duration"])}')
        print(f'Session Outcome: {len(output_dict["Session Outcome"])}')
        print(f'Final Outcome: {len(output_dict["Final Outcome"])}')
        print(f'Time Victim Inserted: {len(output_dict["Time Victim Inserted"])}')
        print(f'Sex: {len(output_dict["Sex"])}')

        
    # Save the output dataframe
    output_df = pd.DataFrame(output_dict)
    output_df.to_csv(f'{GLOBAL_PATH}/cnmfe_output_updated.csv', index=False)
    logging.info('Batch CNMFE processing complete')
