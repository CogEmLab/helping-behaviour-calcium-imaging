# import matplotlib
# matplotlib.use('Agg')  # Set the backend to 'Agg' before importing pyplot
#TODO: try also measuring the entropy

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.session_loader import Session
from src.functional_connectivity_utils import *
from src.utils import shuffle_signal
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import quantities as pq
from elephant.statistics import mean_firing_rate, isi

# Load the configuration
# config = load_config(...)

SESSION_FILE = 'data/session_file_backup/SESSION_FILE_event_associated_cells_definition_2024-10-25_10-51-23.csv'

def process_session(row, results):
    """Processes a session to calculate mean correlations and return a DataFrame."""

    # Create a session object
    session = Session(row)

    # Load the data
    __ = session.load_caiman_model()
    dff = session.load_dff_signal()
    accepted_components = session.load_accepted_components()

    # Remove padding
    dff = dff[:, 1:]

    # Get the transient duration
    transient_duration = 0.4 * session.calcium_fps
    # Estimate noise level 
    noise_level = 1

    # Compute event rate for each neuron
    event_rates = []
    isis = []
    for neuron_data in dff:
        # Detect peaks
        peaks, _ = find_peaks(neuron_data, distance=3*transient_duration, width=transient_duration, height=noise_level*1.5)
        if len(peaks) == 0:
            event_rates.append(0)
            isis.append(0)
            continue

        event_train = peaks / session.calcium_fps
        rate = mean_firing_rate(event_train)
        mean_isi = np.mean(isi(event_train))

        event_rates.append(rate)
        isis.append(mean_isi)
    
    # # Append the information to each column accordingly
    # results['Session'].extend([session.session] * len(event_rates))
    # results['Session Duration'].extend([session.session_length] * len(event_rates))
    # results['Mouse'].extend([session.animal_id] * len(event_rates))
    # results['Day of Helping'].extend([session.day_of_helping] * len(event_rates))
    # results['Session Outcome'].extend([session.session_outcome] * len(event_rates))
    # results['Event Rate'].extend(event_rates)
    # results['ISI'].extend(isis)
    results['Session'].extend([session.session])
    results['Session Duration'].extend([session.session_length])
    results['Mouse'].extend([session.animal_id])
    results['Day of Helping'].extend([session.day_of_helping])
    results['Session Outcome'].extend([session.session_outcome])
    results['Event Rate'].extend([np.mean(event_rates)])
    results['ISI'].extend([np.mean(isis)])
    
    return results

def main():
    session_df = pd.read_csv(SESSION_FILE)
    session_df = session_df[~session_df['Session'].str.contains('Hab')]
    # session_df = session_df[session_df['Mouse'] == 5592]
 
    # Reindex the DataFrame
    session_df = session_df.reset_index(drop=True)

    results = defaultdict(list)

    for __, row in tqdm(session_df.iterrows()):

        results = process_session(row, results)

    # Save the results
    results = pd.DataFrame(results)
    print(results.head())
    results.to_csv('event_rate.csv', index=False)
    
if __name__ == "__main__":
    main()
