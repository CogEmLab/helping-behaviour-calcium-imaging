import matplotlib
matplotlib.use('Agg')  # Set the backend to 'Agg' before importing pyplot
matplotlib.rcParams['axes.unicode_minus'] = False
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.plotting import *
from src.session_loader import Session
from src.functional_connectivity_utils import *
import networkx as nx
from src.utils import load_session_file
from src.specificity import calculate_event_specificity
from math import isnan

# Access the configuration values
SESSION_FILE = load_session_file('data')

def process_session(row):
    print(row)
    """Processes a session to calculate selectivity data."""
    session = Session(row)

    # Load the data
    session.load_caiman_model()
    dff_signal = session.load_dff_signal()
    snr_values = session.load_snr_values()
    liberation_cells_ids = session.get_event_associated_cells_ids('Liberation')
    victim_lib_timecode = session.victim_lib_timecode * session.calcium_fps
    number_of_liberation_cells = session.number_of_liberation_cells

    if isnan(number_of_liberation_cells) or isnan(victim_lib_timecode):
        return None

    event_window = tuple((int(victim_lib_timecode - 30 * session.calcium_fps), 
                          int(victim_lib_timecode + 30 * session.calcium_fps)))
    event_duration = 60

    p_values = []
    for signal, snr in zip(dff_signal, snr_values):
        p_value = calculate_event_specificity(signal, event_window, event_duration, session.calcium_fps, snr, plot=False)
        p_values.append(p_value)

    p_values = np.array(p_values)
    liberation_p_values = p_values[liberation_cells_ids] if liberation_cells_ids is not None else np.array([])

    # Plot the top 5 highest and lowest p-value signals
    sorted_indices = np.argsort(p_values)
    lowest_indices = sorted_indices[:5]
    highest_indices = sorted_indices[-5:]

    plt.figure(figsize=(12, 8))
    for idx, cell_idx in enumerate(lowest_indices):
        plt.plot(dff_signal[cell_idx] - 5 * idx, label=f'Lowest #{idx+1} (p={p_values[cell_idx]:.4f})')
    for idx, cell_idx in enumerate(highest_indices):
        plt.plot(dff_signal[cell_idx] - 5 * (idx + 5), label=f'Highest #{idx+1} (p={p_values[cell_idx]:.4f})')

    plt.title(f'Top 5 Lowest and Highest P-Value Signals (Session: {session.session})')
    plt.xlabel('Time')
    plt.ylabel('DFF Signal')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'figures/top_signals_{session.session}_{session.animal_id}.png', transparent=True)
    plt.close()

    return {
        'p_values': p_values,
        'liberation_p_values': liberation_p_values,
        'total_neurons': len(p_values),
        'selective_neurons': (p_values < 0.05).sum(),
        'selective_percentage': 100 * (p_values < 0.05).sum() / len(p_values),
        'liberation_cells': len(liberation_cells_ids) if liberation_cells_ids is not None else 0,
        'liberation_selective_percentage': 100 * (liberation_p_values < 0.05).sum() / len(liberation_p_values) if liberation_p_values.size > 0 else None
    }

def main(SESSION_FILE):
    SESSION_FILE = SESSION_FILE[SESSION_FILE['Excluded'] != True]

    all_p_values = []
    all_liberation_p_values = []
    results = []

    for _, row in tqdm(SESSION_FILE.iterrows()):
        session_result = process_session(row)
        if session_result:
            all_p_values.extend(session_result['p_values'])
            all_liberation_p_values.extend(session_result['liberation_p_values'])
            results.append(session_result)

    all_p_values = np.array(all_p_values)
    all_liberation_p_values = np.array(all_liberation_p_values)

    # Plot histogram of p-values
    plt.figure(figsize=(8, 6))
    plt.hist(all_p_values, bins=50, color='gray', alpha=0.7, label='All Neurons')
    if all_liberation_p_values.size > 0:
        plt.hist(all_liberation_p_values, bins=50, color='red', alpha=0.7, label='Liberation Cells')
    plt.axvline(0.05, color='blue', linestyle='--', label='p=0.05 Threshold')
    plt.xlabel('P-Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of P-Values Across All Sessions')
    plt.legend()
    plt.tight_layout()
    plt.savefig('figures/p_value_histogram_all_sessions.png', transparent=True)
    plt.close()

    # Save results to a CSV file
    results_df = pd.DataFrame(results)
    results_df.to_csv('selectivity_results.csv', index=False)

    # Print summary statistics
    print(results_df.describe())

if __name__ == "__main__":
    main(SESSION_FILE=SESSION_FILE)
