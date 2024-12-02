from caiman.source_extraction.cnmf.cnmf import load_CNMF
from caiman.base.rois import register_multisession
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from tifffile import imread
from tqdm import tqdm
from collections import defaultdict
from src.session_loader import Session, load_session_file
from src.plotting import plot_boxplots, plot_registration_validation
from src.utils import compute_distances

def preprocess_session_file(file_path):
    """Load and preprocess the session file."""
    session_file = load_session_file(file_path)
    session_file['Mouse'] = session_file['Mouse'].astype(str).apply(lambda x: x.zfill(4))
    return session_file

def load_animal_data(session_file, animal_id):
    """Load all data for a given animal."""
    return session_file[session_file['Mouse'] == animal_id]

def register_sessions(animal_data, dims):
    """Register CNMF sessions and extract neuron positions."""
    spatial = []
    templates = []
    positions = {}
    session_indexing = {}

    for i, session in enumerate(animal_data.itertuples()):
        session_obj = Session(session)
        positions[session.Session] = session_obj.extract_neuron_positions(session_obj.get_neuron_labels())
        session_indexing[session.Session] = i
        spatial.append(load_CNMF(session.CNMFE_Model_Path).estimates.A.toarray())
        templates.append(imread(session.Peak_to_Noise_Ratio_Path))  # Assuming "Peak_to_Noise_Ratio_Path" is a valid column

    # Register the sessions
    A_union, assignments, __ = register_multisession(A=spatial, templates=templates, dims=dims)
    return A_union, assignments, positions, session_indexing

def calculate_registration_stats(assignments):
    """Calculate the percentage of neurons registered across sessions."""
    data = defaultdict(list)
    N = assignments.shape[0]
    K = range(1, assignments.shape[1] + 1)

    for k in K:
        reg_matrix_filtered = assignments[np.sum(~np.isnan(assignments), axis=1) >= k]
        n = reg_matrix_filtered.shape[0]
        data['Percentage'].append(round(n / N * 100, 2))
        data['K'].append(k)

    return pd.DataFrame(data)

def save_results(output_dir, cohort, animal, assignments, A_union, session_indexing):
    """Save registration results to disk."""
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, f'{animal}_registered_ids.npy'), assignments)
    np.save(os.path.join(output_dir, f'{animal}_registered_spatial.npy'), A_union)
    np.save(os.path.join(output_dir, f'{animal}_session_indexing.npy'), session_indexing)

def process_animal(cohort, animal, animal_data, dims, output_dir):
    """Process a single animal: register sessions, calculate statistics, and plot results."""
    # Register sessions
    A_union, assignments, positions, session_indexing = register_sessions(animal_data, dims)

    # Registration validation
    data_df = calculate_registration_stats(assignments)
    plot_registration_validation(
        data=data_df, cohort=cohort, animal=animal, 
        output_path=os.path.join(output_dir, f'{animal}_registration_validation.png')
    )

    # Compute distances
    registered_distances, non_registered_distances = compute_distances(assignments, positions)

    # Plot boxplots for distances
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_boxplots(
        np.array(registered_distances),
        np.array(non_registered_distances),
        ax,
        title=f'Distances: Registered vs. Non-Registered Components (Cohort {cohort})',
        labels=('Registered', 'Non-Registered'),
        ylabel='Distance (pixels)'
    )
    plt.savefig(os.path.join(output_dir, f'{animal}_distance_boxplot.png'))
    plt.close()

    # Save registration results
    save_results(output_dir, cohort, animal, assignments, A_union, session_indexing)

def main(session_file_path, output_base_dir):
    """Main function to process all animals in all cohorts."""
    session_file = preprocess_session_file(session_file_path)
    cohorts = session_file['Cohort'].unique()
    dims = (400, 640)  # Assumes fixed imaging dimensions

    for cohort in cohorts:
        cohort_dir = os.path.join(output_base_dir, cohort)
        os.makedirs(cohort_dir, exist_ok=True)

        mice = session_file[session_file['Cohort'] == cohort]['Mouse'].unique()
        for animal in tqdm(mice, desc=f"Processing Cohort {cohort}"):
            animal_data = load_animal_data(session_file, animal)
            process_animal(cohort, animal, animal_data, dims, cohort_dir)

# Execute the script
if __name__ == "__main__":
    session_file_path = 'data/in-silico-session-file.csv'
    output_base_dir = 'processed-data'
    main(session_file_path, output_base_dir)
