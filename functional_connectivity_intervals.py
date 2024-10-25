import matplotlib
matplotlib.use('Agg')  # Set the backend to 'Agg' before importing pyplot
matplotlib.rcParams['axes.unicode_minus'] = False
import datetime
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
from src.plotting import *
from src.session_loader import Session
from src.functional_connectivity_utils import *
import networkx as nx
from src.utils import load_session_file
from collections import defaultdict

# Load the configuration
config = load_config('tech_docs/functional_connectivity_config.yaml')

# Access the configuration values
SESSION_FILE = load_session_file(config['session_file_dir'])
METADATA_FILE = config['metadata_file']
ANNOTATIONS_DIR = config['annotations_dir']

IMG_DIMS = tuple(config['img_dims'])
THRESHOLD = config['threshold']
SESSION_ORDER = config['session_order']
DAYS_OF_HELPING_ORDER = config['days_of_helping_order']

def process_session(SESSION_FILE, row, plot, axes):
    """Processes a session to calculate mean correlations and return a DataFrame."""

    # Create a session object
    session = Session(row)

    # Load the data
    session.get_neuron_labels()
    dff_signal = session.load_dff_signal()
    neuron_positions = session.neuron_positions
    
    # Load the metadata for liberation components
    __ = session.load_liberation_metadata('Liberation')
    __ = session.load_liberation_metadata('Random')

    # Get the IDs of the liberation cells; it will return None if metadata for this session is empty
    liberation_cells_ids = session.get_event_associated_cells_ids('Liberation')
    random_event_cells_ids = session.get_event_associated_cells_ids('Random')
    # Any other cells?

    # Get the timecode of the victim liberation
    victim_lib_timecode = parse_timecode(session.victim_lib_timecode)
    victim_insr_timecode = parse_timecode(session.victim_insr_timecode)
    ...

    timecodes = [victim_lib_timecode, victim_insr_timecode]
    interval_names = ['before_insr', 'before_lib', 'after_lib']

    correlations = {}
    for i, timecode in enumerate(timecodes):
        if timecode is None:
            continue

        border = int((timecode * session.calcium_fps) - 15 * session.calcium_fps)

        if i == 0:
            signal = dff_signal[:, :border]
        elif i == 1:
            signal = dff_signal[:, border + 15 * session.calcium_fps:border - 15 * session.calcium_fps]
        else:
            signal = dff_signal[:, border + 15 * session.calcium_fps:]

        # Compute correlations
        __, correlation_matrix, __, __ = compute_correlations(neuron_positions, signal, THRESHOLD, bin_size_seconds=60, sampling_rate=session.calcium_fps)

        correlations[interval_names[i]] = get_values(correlation_matrix)

    return correlations

def main(SESSION_FILE):

    # Filter the session file
    SESSION_FILE = SESSION_FILE[SESSION_FILE['Excluded'] != True]

    # Get random index for plotting 
    correlations = defaultdict(list)
    for _, row in tqdm(SESSION_FILE.iterrows()):
        # Process the session
        for interval, correlation_values in process_session(SESSION_FILE, row).items():
            correlations[interval].extend(correlation_values)

    # Plot the results
    fig, axes = plt.subplots(1, len(correlations), figsize=(3 * len(correlations), 3))
    for i, (interval, correlation_values) in enumerate(correlations.items()):
        axes[i].hist(correlation_values, bins=100, cumulative=True, histtype='step', label=interval)
        axes[i].set_xlabel('Correlation')
        axes[i].set_ylabel('Number of neurons')
        axes[i].legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main(SESSION_FILE=SESSION_FILE)
