import logging
import psutil
import caiman as cm
from caiman.source_extraction.cnmf import params as params
import numpy as np
import yaml

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
    motion_correct = bool(config['caiman']['motion_correction']['motion_correct'])          # flag for performing motion correction
    frate = int(config['caiman']['motion_correction']['fr'])                                # movie frame rate
    decay_time = float(config['caiman']['motion_correction']['decay_time'])                 # length of a typical transient in seconds
    pw_rigid = bool(config['caiman']['motion_correction']['pw_rigid'])                      # flag for performing piecewise-rigid motion correction (otherwise just rigid)
    gSig_filt = tuple(config['caiman']['motion_correction']['gSig_filt'])                   # sigma for high pass spatial filter applied before motion correction, used in 1p data
    max_shifts = tuple(config['caiman']['motion_correction']['max_shifts'])                 # maximum allowed rigid shift
    strides = tuple(config['caiman']['motion_correction']['strides'])                       # start a new patch for pw-rigid motion correction every x pixels
    overlaps = tuple(config['caiman']['motion_correction']['overlaps'])                     # overlap between patches (size of patch = strides + overlaps)
    max_deviation_rigid = int(config['caiman']['motion_correction']['max_deviation_rigid']) # maximum deviation allowed for patch with respect to rigid shifts
    border_nan = str(config['caiman']['motion_correction']['border_nan'])                   # replicate values along the boundaries

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
    p = int(config['caiman']['cnmfe']['p'])                                         # order of the autoregressive system
    K = None                                         # upper bound on number of components per patch, in general None for CNMFE
    gSig = np.array(config['caiman']['cnmfe']['gSig'])                              # expected half-width of neurons in pixels 
    gSiz = 2*gSig + 1                                                     # half-width of bounding box created around neurons during initialization
    merge_thr = float(config['caiman']['cnmfe']['merge_thr'])                       # merging threshold, max correlation allowed
    rf = int(config['caiman']['cnmfe']['rf'])                                       # half-size of the patches in pixels. e.g., if rf=40, patches are 80x80
    stride_cnmf = int(config['caiman']['cnmfe']['stride_cnmf'])                     # amount of overlap between the patches in pixels 
    tsub = int(config['caiman']['cnmfe']['tsub'])                                   # downsampling factor in time for initialization, increase if you have memory problems
    ssub = int(config['caiman']['cnmfe']['ssub'])                                   # downsampling factor in space for initialization, increase if you have memory problems
    gnb = int(config['caiman']['cnmfe']['gnb'])                                     # number of background components (rank) if positive, set to 0 for CNMFE
    low_rank_background = None                           # None leaves background of each patch intact (use True if gnb>0)
    nb_patch = int(config['caiman']['cnmfe']['nb_patch'])                           # number of background components (rank) per patch (0 for CNMFE)
    min_corr = float(config['caiman']['cnmfe']['min_corr'])                         # min peak value from correlation image
    min_pnr = int(config['caiman']['cnmfe']['min_pnr'])                             # min peak to noise ration from PNR image
    ssub_B = int(config['caiman']['cnmfe']['ssub_B'])                               # additional downsampling factor in space for background (increase to 2 if slow)
    ring_size_factor = float(config['caiman']['cnmfe']['ring_size_factor'])         # radius of ring is gSiz*ring_size_factor
    min_SNR = float(config['caiman']['cnmfe']['min_SNR'])                           # minimum SNR for accepting new components
    SNR_lowest = float(config['caiman']['cnmfe']['SNR_lowest'])
    rval_lowest = float(config['caiman']['cnmfe']['rval_lowest'])
    use_cuda = bool(config['caiman']['cnmfe']['use_cuda'])                          # flag for gpu acceleration
    method_init = str(config['caiman']['cnmfe']['method_init'])    
    only_init = bool(config['caiman']['cnmfe']['only_init'])  
    update_background_components = bool(config['caiman']['cnmfe']['update_background_components'])  
    center_psf = bool(config['caiman']['cnmfe']['center_psf'])  
    del_duplicates = bool(config['caiman']['cnmfe']['del_duplicates'])               # whether to remove duplicates from initialization

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
                                    'nb_patch': nb_patch,
                                    'method_deconvolution': 'oasis',       # could use 'cvxpy' alternatively
                                    'low_rank_background': low_rank_background,
                                    'min_corr': min_corr,
                                    'min_pnr': min_pnr,
                                    'nb': gnb,
                                    'normalize_init': False,               # just leave as is
                                    'ssub_B': ssub_B,
                                    'ring_size_factor': ring_size_factor,
                                    'border_pix': 0,
                                    'min_SNR': min_SNR,
                                    'SNR_lowest': SNR_lowest,
                                    'rval_lowest': rval_lowest,
                                    'use_cuda': use_cuda,
                                    'method_init': method_init,
                                    'only_init': only_init,
                                    'update_background_components': update_background_components,
                                    'center_psf': center_psf,
                                    'del_duplicates': del_duplicates});  

    return parameters
