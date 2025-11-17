import numpy as np
import pandas as pd
import caiman as cm
import os
import logging
from caiman.source_extraction.cnmf import params as params
from src.setup_for_caiman import *
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction import cnmf
from caiman.source_extraction.cnmf.cnmf import load_CNMF
from src.plotting import *

def reevaluate_components(model_path, fname_new, fps=30):
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
    # # Get the name of the file

    # Perform setup for caiman
    cluster, __ = general_setup()

    # load memory mappable file
    Yr, dims, T = cm.load_memmap(fname_new)
    ci_movie = Yr.T.reshape((T,) + dims, order='F')

    cnmfe_model = load_CNMF(model_path)

    # # Upload the new parameters
    cnmfe_model.params.quality['SNR_lowest'] = 2.5
    cnmfe_model.params.quality['rval_lowest'] = 0.4
    cnmfe_model.params.quality['min_SNR'] = 3.0
    cnmfe_model.params.quality['rval_thr'] = 0.87

    # Compute thresholded components
    logging.info('Computing thresholded components...')
    cnmfe_model.estimates.threshold_spatial_components(maxthr=0.90)  # Adjust parameters as needed
    # Check if thresholded components were computed
    if cnmfe_model.estimates.idx_components.size == 0:
       raise ValueError("No components were thresholded. Please check your input data and parameters.")

    cnmfe_model.estimates.evaluate_components(ci_movie, cnmfe_model.params)
    print(len(cnmfe_model.estimates.idx_components),' after removing bad param')    

    # Remove duplicates
    logging.info('Removing duplicate components...')
    cnmfe_model.estimates.remove_duplicates(min_dist=12, select_comp=False)
    print(len(cnmfe_model.estimates.idx_components),' accepted after removing duplicates')   
    print(len(cnmfe_model.estimates.idx_components_bad),' rejected after removing duplicates')

    # Remove small and large neurons
    logging.info('Removing small and large neurons...')
    cnmfe_model.estimates.remove_small_large_neurons(max_size_neuro=45, min_size_neuro=5, select_comp=False)
    print(len(cnmfe_model.estimates.idx_components),' accepted after removing weird size neurons')   
    print(len(cnmfe_model.estimates.idx_components_bad),' rejected after removing duplicates')

    total_components = len(cnmfe_model.estimates.C)
    num_accepted = len(cnmfe_model.estimates.idx_components)
    num_rejected = len(cnmfe_model.estimates.idx_components_bad)

    # Compute some summary images (correlation and peak to noise)
    gsig_tmp = (3,3)
    correlation_image, peak_to_noise_ratio = cm.summary_images.correlation_pnr(ci_movie[::max(T//1000, 1)], # subsample if needed
                                                                            gSig=gsig_tmp[0], # used for filter
                                                                            swap_dim=False) # change swap dim if output looks weird, it is a problem with tiffile
    
    cm.stop_server(dview=cluster)
    
    # Set A_thr field to None before selecting components (threshold step of reevaluate_comps creates a dictionary of different shape of A)
    cnmfe_model.estimates.A_thr = None

    # Select only the accepted components
    cnmfe_model.estimates.select_components(use_object=True, save_discarded_components=True)

    return (cnmfe_model, ci_movie, fname_new, correlation_image, peak_to_noise_ratio, dims, total_components, num_accepted, num_rejected)

def analyze_data(ci_movie_path, save_dir_docs, config, fps=30, plot=False):
    """
    Analyze calcium imaging data using CNMF-E algorithm.

    Parameters:
        ci_movie_path (str): Path to the calcium imaging movie file.
        save_dir_docs (str): Directory to save the technical documentation files.
        config (dict): Configuration dictionary.
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
    params_mc = get_params_mot_corr(ci_movie_path, config=config)

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
    cnmfe_params = get_params_cnmfe(params_mc, config=config)
    
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
    # ^ Just out of curiosity, which is subsampling being used in above function when passing the ci_movie array?
    cm.stop_server(dview=cluster)
    
    return cnmfe_model, ci_movie, fname_new, correlation_image, peak_to_noise_ratio, dims, total_components, num_accepted, num_rejected
