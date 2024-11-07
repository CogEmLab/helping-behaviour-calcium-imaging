import logging
import psutil
import caiman as cm
from caiman.source_extraction.cnmf import params as params
import numpy as np
import yaml

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def general_setup():
    
    if 'cluster' in locals():  # 'locals' contains list of current local variables
        logging.info('Closing previous cluster')
        cm.stop_server(dview=cluster)


    logging.info("Setting up new cluster")
    _, cluster, n_processes = cm.cluster.setup_cluster(backend='multiprocessing', 
                                                    n_processes=psutil.cpu_count(), 
                                                    ignore_preexisting=False)
    logging.info(f"Successfully set up cluster with {n_processes} processes")

    return cluster, n_processes

def get_params_mot_corr(ci_path, config):
    """
    Get motion correction parameters for caiman.

    Parameters:
    - ci_movie (str or list): Path to the calcium imaging movie or list of paths to multiple movies.
    - frate (float): Movie frame rate. Default is 30.
    - decay_time (float): Length of a typical transient in seconds. Default is 0.3.
    - pw_rigid (bool): Flag for performing piecewise-rigid motion correction (otherwise just rigid). Default is False.
    - gSig_filt (tuple): Sigma for high pass spatial filter applied before motion correction, used in 1p data. Default is (3, 3).
    - max_shifts (tuple): Maximum allowed rigid shift. Default is (5, 5).
    - strides (tuple): Start a new patch for pw-rigid motion correction every x pixels. Default is (48, 48).
    - overlaps (tuple): Overlap between patches (size of patch = strides + overlaps). Default is (24, 24).
    - max_deviation_rigid (int): Maximum deviation allowed for patch with respect to rigid shifts. Default is 3.
    - border_nan (str): Replicate values along the boundaries. Default is 'copy'.

    Returns:
    - parameters (caiman.source_extraction.cnmf.params.CNMFParams): Motion correction parameters.
    """
    # dataset dependent parameters
    frate = 30                       # movie frame rate
    decay_time = 0.3                 # length of a typical transient in seconds

    # motion correction parameters
    motion_correct = True    # flag for performing motion correction
    pw_rigid = False         # flag for performing piecewise-rigid motion correction (otherwise just rigid)
    gSig_filt = (3, 3)       # sigma for high pass spatial filter applied before motion correction, used in 1p data
    max_shifts = (5, 5)      # maximum allowed rigid shift
    strides = (48, 48)       # start a new patch for pw-rigid motion correction every x pixels
    overlaps = (24, 24)      # overlap between patches (size of patch = strides + overlaps)
    max_deviation_rigid = 3  # maximum deviation allowed for patch with respect to rigid shifts
    border_nan = 'copy'      # replicate values along the boundaries

    mc_dict = {
        'fnames': ci_path,
        'fr': frate,
        'decay_time': decay_time,
        'pw_rigid': pw_rigid,
        'max_shifts': max_shifts,
        'gSig_filt': gSig_filt,
        'strides': strides,
        'overlaps': overlaps,
        'max_deviation_rigid': max_deviation_rigid,
        'border_nan': border_nan
    }

    parameters = params.CNMFParams(params_dict=mc_dict)
    # mc_dict = config['motion_correction']
    # mc_dict['fnames'] = ci_path
    
    # parameters = params.CNMFParams(params_dict=mc_dict)
    return parameters

def get_params_cnmfe(parameters=None, config=None):
    """
    Get CNMF-E parameters for caiman.

    Parameters:
    - parameters (caiman.source_extraction.cnmf.params.CNMFParams): Existing CNMFParams object. Default is None.
    - bord_px (int): Number of pixels to not consider in the borders. Default is 0.
    - p (int): Order of the autoregressive system. Default is 1.
    - K (int): Upper bound on number of components per patch. Default is None.
    - gSig (np.ndarray): Expected half-width of neurons in pixels. Default is np.array([3, 3]).
    - gSiz (int): Half-width of bounding box created around neurons during initialization. Default is None.
    - merge_thr (float): Merging threshold, max correlation allowed. Default is 0.7.
    - rf (int): Half-size of the patches in pixels. Default is 20.
    - stride_cnmf (int): Amount of overlap between the patches in pixels. Default is 10.
    - tsub (int): Downsampling factor in time for initialization. Default is 2.
    - ssub (int): Downsampling factor in space for initialization. Default is 1.
    - gnb (int): Number of background components (rank). Default is 0.
    - low_rank_background (bool): Flag to leave background of each patch intact. Default is None.
    - nb_patch (int): Number of background components (rank) per patch. Default is 0.
    - min_corr (float): Min peak value from correlation image. Default is 0.8.
    - min_pnr (int): Min peak to noise ratio from PNR image. Default is 10.
    - ssub_B (int): Additional downsampling factor in space for background. Default is 2.
    - ring_size_factor (float): Radius of ring is gSiz * ring_size_factor. Default is 1.4.
    - min_SNR (float): Minimum SNR for accepting new components. Default is 2.5.
    - SNR_lowest

    Returns:
    - parameters (caiman.source_extraction.cnmf.params.CNMFParams): CNMF-E parameters.
    """
    
    # parameters for source extraction and deconvolution
    p = 1               # order of the autoregressive system
    K = None            # upper bound on number of components per patch, in general None for CNMFE
    gSig = np.array([3, 3])  # expected half-width of neurons in pixels 
    gSiz = 2*gSig + 1     # half-width of bounding box created around neurons during initialization
    merge_thr = .7      # merging threshold, max correlation allowed
    rf = 20          # half-size of the patches in pixels. e.g., if rf=40, patches are 80x80
    stride_cnmf = 10    # amount of overlap between the patches in pixels 
    tsub = 2            # downsampling factor in time for initialization, increase if you have memory problems
    ssub = 1            # downsampling factor in space for initialization, increase if you have memory problems
    gnb = 0             # number of background components (rank) if positive, set to 0 for CNMFE
    low_rank_background = None  # None leaves background of each patch intact (use True if gnb>0)
    nb_patch = 0        # number of background components (rank) per patch (0 for CNMFE)
    min_corr = .8       # min peak value from correlation image
    min_pnr = 10        # min peak to noise ration from PNR image
    ssub_B = 2          # additional downsampling factor in space for background (increase to 2 if slow)
    ring_size_factor = 1.4  # radius of ring is gSiz*ring_size_factor
    min_SNR = 3.0  
    SNR_lowest = 2.0 
    rval_lowest = 0.3
    border_nan = 'copy'
    bord_px = 0 if border_nan == 'copy' else bord_px

    parameters.change_params(params_dict={'method_init': 'corr_pnr',  # use this for 1 photon
                                    'K': K,
                                    'gSig': gSig,
                                    'gSiz': gSiz,
                                    'merge_thr': merge_thr,
                                    'p': p,
                                    'tsub': tsub,
                                    'ssub': ssub,
                                    'rf': rf,
                                    'stride': stride_cnmf,
                                    'only_init': True,    # set it to True to run CNMF-E
                                    'nb': gnb,
                                    'nb_patch': nb_patch,
                                    'method_deconvolution': 'oasis',       # could use 'cvxpy' alternatively
                                    'low_rank_background': low_rank_background,
                                    'update_background_components': True,  # sometimes setting to False improve the results
                                    'min_corr': min_corr,
                                    'min_pnr': min_pnr,
                                    'normalize_init': False,               # just leave as is
                                    'center_psf': True,                    # True for 1p
                                    'ssub_B': ssub_B,
                                    'ring_size_factor': ring_size_factor,
                                    'del_duplicates': True,                # whether to remove duplicates from initialization
                                    'border_pix': bord_px});  

    # if parameters is None:
    #     parameters = params.CNMFParams()
    
    # cnmfe_dict = config['cnmfe']
    
    # for key in ['gSig']:
    #     if key in cnmfe_dict and isinstance(cnmfe_dict[key], list):
    #         cnmfe_dict[key] = np.array(cnmfe_dict[key])
    
    # parameters.change_params(params_dict=cnmfe_dict)
    return parameters
