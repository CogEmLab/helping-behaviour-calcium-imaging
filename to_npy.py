import numpy as np
import os
from glob import glob
from caiman.source_extraction.cnmf.cnmf import load_CNMF
import matplotlib
matplotlib.use('Agg')  # Set the backend to 'Agg' to prevent displaying figures
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from src.plotting import plot_random_traces


GLOBAL_DIR = 'processed-data/'

# Find all files in the directory and subdirectories with the extension .hdf5
files = glob(GLOBAL_DIR + '**/**/**/*.hdf5', recursive=True)

for file in files:

    # # Skip if output already exists
    output_filename_full = file.replace('.hdf5', '_Dff_z_scored.npy').replace('cnmfe_model', 'npy_data')
    if os.path.exists(output_filename_full):
        print('Skipping ', file)
        continue
    
    cnmfe_model = load_CNMF(file)
    
    # Get SNR values
    SNR = cnmfe_model.estimates.SNR_comp

    # Get ids of accepted components 
    accepted_components = cnmfe_model.estimates.idx_components

    print('Accepted components: ', accepted_components.shape)
    # Get dF/F 
    cnmfe_model.estimates.detrend_df_f(detrend_only=True)
    DF_F = cnmfe_model.estimates.F_dff
    
    # Normalize dF/F
    DF_F_Z_scored = (DF_F - DF_F.mean(axis=1)[:, None]) / DF_F.std(axis=1)[:, None]
    
    # Create output directories
    output_filename_full = file.replace('.hdf5', '_Dff_z_scored.npy').replace('cnmfe_model', 'npy_data')
    output_dir = os.path.dirname(output_filename_full)
    output_dir_transformer = output_dir.replace('processed-data', 'transformer-data')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir_transformer, exist_ok=True)

    # Save the dF/F
    np.save(output_filename_full, DF_F_Z_scored)
    np.save(output_filename_full.replace('processed-data', 'transformer-data'), DF_F_Z_scored)
    print('Saved to: ', output_dir)

    # Also save the accepted components vector
    accepted_components_output_path = os.path.join(output_dir, 'accepted_components_{}.npy'.format(os.path.basename(file).replace('.hdf5', '')))
    np.save(accepted_components_output_path, accepted_components)
    np.save(accepted_components_output_path.replace('processed-data', 'transformer-data'), accepted_components)
    print('Saved to: ', accepted_components_output_path)
    
    # Create cnmfe_docs folder and plot random traces
    cohort_folder = os.path.dirname(os.path.dirname(output_dir))
    session_folder = os.path.dirname(output_dir)
    cnmfe_docs_folder = os.path.join(cohort_folder, os.path.basename(session_folder), 'cnmfe_docs')
    os.makedirs(cnmfe_docs_folder, exist_ok=True)
    
    plot_output_path = os.path.join(cnmfe_docs_folder, f'{os.path.basename(file)}_random_traces.png')

    # # Check if SNR values contain only int, if not, remove that component from accepted_components
    # if not all(isinstance(x, int) for x in SNR[accepted_components]):
    # #     accepted_components = [x for x in accepted_components if isinstance(SNR[x], int)]

    # try:
    #     plot_random_traces(DF_F_Z_scored[accepted_components], plot_output_path, SNR[accepted_components])
    #     print('Plot saved to: ', plot_output_path)
    # except Exception as e:
    #     print('Error: ', e)
    #     print('Could not plot random traces for ', file)

print("Processing complete.")