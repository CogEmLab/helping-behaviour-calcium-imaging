# import matplotlib
# matplotlib.use('Agg')  # Set the backend to 'Agg' before importing pyplot
#TODO: try also measuring the entropy

import pandas as pd
from tqdm import tqdm
from src.session_loader import Session
from src.functional_connectivity_utils import *
from collections import defaultdict
from src.utils import compute_event_rates

SESSION_FILE = 'data/SESSION_FILE_batch_cnmfe.csv'

def process_session(row, results):
    """Processes a session to calculate mean correlations and return a DataFrame."""

    # Create a session object
    session = Session(row)
    # Load the data
    __ = session.load_caiman_model()
    dff = session.load_dff_signal()
    snr_values = session.load_snr_values()

    # Remove padding
    dff = dff[:, 1:]

    # Get the transient duration
    transient_duration = 0.4 * session.calcium_fps
    
    # Compute the event rates
    event_rates, __ = compute_event_rates(dff, session.cohort, session.animal_id, session.session, snr_values, session.calcium_fps, distance=3*transient_duration, width=transient_duration)
    
    # Append the information to each column accordingly
    results['Event Rate'].extend(event_rates)
    results['Session'].extend([session.session] * len(event_rates))
    results['Day of Helping'].extend([session.day_of_helping] * len(event_rates))
    results['Cohort'].extend([session.cohort] * len(event_rates))
    results['Animal'].extend([session.animal_id] * len(event_rates))

    return results

def main():
    session_df = pd.read_csv(SESSION_FILE)
 
    # Reindex the DataFrame
    session_df = session_df.reset_index(drop=True)

    results = defaultdict(list)

    for __, row in tqdm(session_df.iterrows()):

        results = process_session(row, results)

    # Save the results
    results_df = pd.DataFrame(results)
    results_df.to_csv('results/event_rate.csv', index=False)

if __name__ == "__main__":
    main()
