"""
This script takes in a folder containing CNMFE models, 
and saves the dF/F and accepted components of each model to a new folder.

The dF/F is normalized to have a mean of 0 and a standard deviation of 1.

The accepted components vector is saved to the same folder as the dF/F.

The script also saves the CNMFE model with only the accepted components to a new folder.

(Maybe in the future we can rewrite it to only save the accepted components model)
Finally, the script plots random traces of the dF/F for each model and saves them to a new folder.

"""

import logging
import matplotlib
matplotlib.use('Agg') # To avoid displaying plots
import os
import numpy as np
from glob import glob
from caiman.source_extraction.cnmf.cnmf import load_CNMF
from plotting import plot_random_traces


# Move one directory up, because scrip is located in subfolder 'preprocessing-scripts'
os.chdir('..')

GLOBAL_DIR = 'processed-data/'
SKIP = True

# Find all files in the directory and subdirectories with the extension .hdf5; 
# as of 10.2024 it will find all full models and all models with only accepted components. Be careful here!
files = glob(GLOBAL_DIR + '**/**/**/*video.hdf5', recursive=True)

for file in files:
   
    # Get the name of the output file
    output_filename_full = file.replace('.hdf5', '_Dff_z_scored.npy').replace('cnmfe_model', 'npy_data')
    # Create output directories
    output_dir = os.path.dirname(output_filename_full)
    os.makedirs(output_dir, exist_ok=True)
    
    # Skip if output already exists and if we want to skip (SKIP = True)
    if SKIP and os.path.exists(file):
        logging.info('Skipping %s', file)
        continue
    
    cnmfe_model = load_CNMF(file) # Load the model

    # Get ids of accepted components in original indexing
    accepted_components = cnmfe_model.estimates.idx_components
    logging.info('Accepted components: %s', accepted_components.shape)

    # Select only the accepted components, onte it will change the indexing everywhere in the model. Discarded components could be restored.
    cnmfe_model.estimates.select_components(use_object=True, save_discarded_components=True)

    # Get dF/F; detrend only (1p protocol)
    cnmfe_model.estimates.detrend_df_f(detrend_only=True)
    # Get dF/F
    DF_F = cnmfe_model.estimates.F_dff
    # Check if dF/F shape matches the number of accepted components after component selection
    assert DF_F.shape[0] == accepted_components.shape[0]
    # ^ Good that you check this!

    # This is the work-around for the non-helper animal (848) where session is damaged after certain moment.
    if '2021-10-27-10-25-03_video' in file:
        logging.info('For file %s, cutting dF/F to 6000 frames', file)
        DF_F = DF_F[:, :6000]
    elif '2021-10-29-10-43-11_video' in file:
        logging.info('For file %s, cutting dF/F to 8000 frames', file)
        DF_F = DF_F[:, :8000]

    # Normalize dF/F to have a mean of 0 and a standard deviation of 1
    DF_F_Z_scored = (DF_F - DF_F.mean(axis=1)[:, None]) / DF_F.std(axis=1)[:, None]
    # ^ I saw your traces and this seems to be OK but when I did my analysis earlier where I only used the S Matrix for later steps
    # ^ I have normalized using the SNR value that you also obtain from Caiman. This could help you to filter out noisy data.
    # ^ Maybe keep it as an option for later, or to experiment with it.
    
    # Save the dF/F
    np.save(output_filename_full, DF_F_Z_scored) # Saving the df_f with only accepted components
    logging.info('Saved to: %s', output_dir)

    # Also save the accepted components vector
    accepted_components_output_path = os.path.join(output_dir, 'accepted_components_{}.npy'.format(os.path.basename(file).replace('.hdf5', '')))
    np.save(accepted_components_output_path, accepted_components)
    logging.info('Saved to: %s', accepted_components_output_path)
    
    # Save the accepted components only model, model after select_components function
    cnmfe_model.save(file.replace('.hdf5', '_accepted_only.hdf5'))
    
    # Create cnmfe_docs folder and plot random traces
    cnmfe_docs_folder = output_dir.replace('npy_data', 'cnmfe_docs')
    os.makedirs(cnmfe_docs_folder, exist_ok=True)    
    plot_output_path = os.path.join(cnmfe_docs_folder, f'{os.path.basename(file).replace(".hdf5", "")}_random_traces.png')

    # Get SNR values (for accepted components only)
    SNR = cnmfe_model.estimates.SNR_comp
    
    # Check if SNR values contain only int, if not, remove that component from accepted_components
    if not all(isinstance(x, int) for x in SNR):

        try:
            plot_random_traces(DF_F_Z_scored, snr=SNR, filename=plot_output_path)
            logging.info('Plot saved to: %s', plot_output_path)
        except Exception as e:
            logging.error('Error: %s', e)
            logging.error('Could not plot random traces for %s', file)

logging.info("Processing complete.")


