import matplotlib
# matplotlib.use('Agg')  # Set the backend to 'Agg' before importing pyplot
# matplotlib.rcParams['axes.unicode_minus'] = False
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
from src.specificity import calculate_event_specificity
from math import isnan

# Load the configuration

# Access the configuration values
SESSION_FILE = load_session_file('data')

def process_session(row, plot):
    """Processes a session to calculate mean correlations and return a DataFrame."""

    # Create a session object
    session = Session(row)

    # Load the data
    pnr = session.load_pnr_image()
    session.get_neuron_labels()
    dff_signal = session.load_dff_signal()
    snr_values = session.load_snr_values()
    victim_lib_timecode = session.victim_lib_timecode * session.calcium_fps

    if isnan(victim_lib_timecode):
        return None
    
    event_window = tuple((int(victim_lib_timecode - 30 * session.calcium_fps), int(victim_lib_timecode + 30 * session.calcium_fps)))
    print(event_window)
    event_duration = 60
    

    p_values = []
    for signal, snr in zip(dff_signal, snr_values):
        print('snr', snr)
        p_value = calculate_event_specificity(signal, event_window, event_duration, session.calcium_fps, snr, plot=False)
        p_values.append(p_value)
    print(p_values)

    # # Load the metadata for liberation components
    # liberation_associated_metadata = session.load_liberation_metadata('Liberation')
    # # Get the IDs of the liberation cells; it will return None if metadata for this session is empty
    # liberation_cells_ids = session.get_event_associated_cells_ids('Liberation')

    # print(liberation_cells_ids)

def main(SESSION_FILE):

    # Filter the session file
    SESSION_FILE = SESSION_FILE[SESSION_FILE['Excluded'] != True]

    # Get random index for plotting 
    idx = np.random.randint(0, len(SESSION_FILE))
    for i, row in tqdm(SESSION_FILE.iterrows()):

        if i == idx:
            print('Plotting session with index:', i)
            plot=True
        else:
            plot=False

        # Process the session
        process_session(row, plot)
 
if __name__ == "__main__":
    main(SESSION_FILE=SESSION_FILE)
