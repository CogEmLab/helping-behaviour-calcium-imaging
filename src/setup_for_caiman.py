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
    - config file (yaml file): File with the parameters 

    Returns:
    - parameters (caiman.source_extraction.cnmf.params.CNMFParams): Motion correction parameters.
    """

    # Read parameters from config
    motion_correct = bool(config['motion_correction']['motion_correct'])          # flag for performing motion correction
    frate = int(config['motion_correction']['fr'])                                # movie frame rate
    decay_time = float(config['motion_correction']['decay_time'])                 # length of a typical transient in seconds
    pw_rigid = bool(config['motion_correction']['pw_rigid'])                      # flag for performing piecewise-rigid motion correction (otherwise just rigid)
    gSig_filt = tuple(config['motion_correction']['gSig_filt'])                   # sigma for high pass spatial filter applied before motion correction, used in 1p data
    max_shifts = tuple(config['motion_correction']['max_shifts'])                 # maximum allowed rigid shift
    strides = tuple(config['motion_correction']['strides'])                       # start a new patch for pw-rigid motion correction every x pixels
    overlaps = tuple(config['motion_correction']['overlaps'])                     # overlap between patches (size of patch = strides + overlaps)
    max_deviation_rigid = int(config['motion_correction']['max_deviation_rigid']) # maximum deviation allowed for patch with respect to rigid shifts
    border_nan = str(config['motion_correction']['border_nan'])                   # replicate values along the boundaries

    # Organize into a dictionary
    mc_dict = {
        'motion_correct': motion_correct,
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

    # Save to the CNMFParams
    parameters = params.CNMFParams(params_dict=mc_dict)
    return parameters

def get_params_cnmfe(parameters=None, config=None):
    """
    Get CNMF-E parameters for caiman.

    Parameters:
    - parameters (caiman.source_extraction.cnmf.params.CNMFParams): Existing CNMFParams object. Default is None.
    - config file (yaml file): File with the parameters

    Returns:
    - parameters (caiman.source_extraction.cnmf.params.CNMFParams): CNMF-E parameters.
    """
    
    # Parameters for source extraction and deconvolution
    p = int(config['motion_correction']['p'])                                         # order of the autoregressive system
    K = type(config['motion_correction']['K'])                                        # upper bound on number of components per patch, in general None for CNMFE
    gSig = np.array(config['motion_correction'][' gSig'])                              # expected half-width of neurons in pixels 
    gSiz = 2*gSig + 1                                                                 # half-width of bounding box created around neurons during initialization
    merge_thr = float(config['motion_correction']['merge_thr'])                       # merging threshold, max correlation allowed
    rf = int(config['motion_correction']['rf'])                                       # half-size of the patches in pixels. e.g., if rf=40, patches are 80x80
    stride_cnmf = int(config['motion_correction']['stride_cnmf'])                     # amount of overlap between the patches in pixels 
    tsub = int(config['motion_correction']['tsub'])                                   # downsampling factor in time for initialization, increase if you have memory problems
    ssub = int(config['motion_correction']['ssub'])                                   # downsampling factor in space for initialization, increase if you have memory problems
    gnb = int(config['motion_correction']['gnb'])                                     # number of background components (rank) if positive, set to 0 for CNMFE
    low_rank_background = type(config['motion_correction']['low_rank_background'])    # None leaves background of each patch intact (use True if gnb>0)
    nb_patch = int(config['motion_correction']['nb_patch'])                           # number of background components (rank) per patch (0 for CNMFE)
    min_corr = float(config['motion_correction']['min_corr'])                         # min peak value from correlation image
    min_pnr = int(config['motion_correction']['min_pnr'])                             # min peak to noise ration from PNR image
    ssub_B = int(config['motion_correction']['ssub_B'])                               # additional downsampling factor in space for background (increase to 2 if slow)
    ring_size_factor = float(config['motion_correction']['ring_size_factor'])         # radius of ring is gSiz*ring_size_factor
    min_SNR = float(config['motion_correction']['min_SNR'])                           # minimum SNR for accepting new components
    SNR_lowest = float(config['motion_correction']['SNR_lowest'])
    rval_lowest = float(config['motion_correction']['rval_lowest'])
    use_cuda = bool(config['motion_correction']['use_cuda'])                          # flag for gpu acceleration

    parameters.change_params(params_dict={'method_init': 'corr_pnr',       # use this for 1 photon
                                    'K': K,
                                    'gSig': gSig,
                                    'gSiz': gSiz,
                                    'merge_thr': merge_thr,
                                    'p': p,
                                    'tsub': tsub,
                                    'ssub': ssub,
                                    'rf': rf,
                                    'stride': stride_cnmf,
                                    'only_init': True,                     # set it to True to run CNMF-E
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
                                    'border_pix': 0,
                                    'min_SNR': min_SNR,
                                    'SNR_lowest': SNR_lowest,
                                    'rval_lowest': rval_lowest,
                                    'use_cuda': use_cuda});  

    return parameters
