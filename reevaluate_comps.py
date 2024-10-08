import pandas as pd
from src.utils import reevaluate_components
from tqdm import tqdm
session_file = 'data/updated_session_file_MiniscopeCW_all_Moises.csv'

session_file = pd.read_csv(session_file)

# Take only Oct_2021 cohort
session_file = session_file[session_file['Cohort'] == 'Oct_2021']

# Iterate through the session file
for index, row in tqdm(session_file.iterrows()):
    model_path = row['CNMFE Model Path'].replace('processed-data', 'processed_data')
    motion_corrected_path = row['Motion Corrected Path'].replace('processed-data', 'processed_data')

    FPS = row['FPS (CI downsampled)']
    cnmfe_model, ci_movie, fname_new, correlation_image, peak_to_noise_ratio, dims, total_components, num_accepted, num_rejected = reevaluate_components(model_path, motion_corrected_path, FPS)

    # Update the session file
    # session_file.at[index, 'CNMFE Model Path'] = fname_new
    # Save the model to the same path
    cnmfe_model.save(model_path)
    # Update Num Accepted,Num Rejected columns
    session_file.at[index, 'Num Accepted'] = num_accepted
    session_file.at[index, 'Num Rejected'] = num_rejected
    print(f'Updated session file for {row["Session"]}')
    print(f'Num Accepted: {num_accepted}')
    print(f'Num Rejected: {num_rejected}')
    # Saved model to the same path
    print(f'Saved model to {model_path}')
    
# Save the updated session file
session_file.to_csv('data/updated_session_file_MiniscopeCW_all_Moises_Oct2021only.csv', index=False)