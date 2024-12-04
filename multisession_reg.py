from caiman.source_extraction.cnmf.cnmf import load_CNMF
from caiman.base.rois import register_multisession
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from tifffile import imread
from tqdm import tqdm
from collections import defaultdict
from src.session_loader import Session
from src.plotting import plot_boxplots, plot_registration_validation
from src.utils import load_session_file
from src.registration_utils import *

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

    for i, session in enumerate(animal_data.iterrows()):
        session = session[1]
        session_obj = Session(session)
        positions[session_obj.session] = session_obj.extract_neuron_positions(session_obj.get_neuron_labels())
        session_indexing[session_obj.session] = i
        spatial.append(load_CNMF(session_obj.model_path).estimates.A.toarray())
        templates.append(imread(session_obj.path_to_pnr))  # Assuming "Peak_to_Noise_Ratio_Path" is a valid column

    # Register the sessions
    A_union, assignments, __ = register_multisession(A=spatial, templates=templates, dims=dims)
    return A_union, assignments, positions, session_indexing

def save_results(output_dir, cohort, animal, assignments, A_union, session_indexing):
    """Save registration results to disk."""
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, f'{animal}_registered_ids.npy'), assignments)
    np.save(os.path.join(output_dir, f'{animal}_registered_spatial.npy'), A_union)
    np.save(os.path.join(output_dir, f'{animal}_session_indexing.npy'), session_indexing)

def gather_registration_data(assignments, cohort, animal, registration_data):
    """Add registration stats for an animal to the collective cohort data."""
    data_df = calculate_registration_stats(assignments)
    data_df['Animal'] = animal  # Track which animal this data belongs to
    data_df['Cohort'] = cohort
    registration_data.append(data_df)

def gather_distance_data(registered_distances, non_registered_distances, cohort, animal, distance_data):
    """Add average distances for an animal to the collective cohort data."""
    avg_registered = np.mean(registered_distances)
    avg_non_registered = np.mean(non_registered_distances)
    distance_data.append({
        'Animal': animal,
        'Cohort': cohort,
        'Category': 'Registered',
        'Distance': avg_registered
    })
    distance_data.append({
        'Animal': animal,
        'Cohort': cohort,
        'Category': 'Non-Registered',
        'Distance': avg_non_registered
    })

def plot_combined_distances(distance_data,ax=None, output_path=None):
    """
    Plot combined boxplots for distances across all animals in a cohort.

    Parameters:
        distance_data (pd.DataFrame): A DataFrame containing 'Cohort', 'Animal', 'Distance', and 'Type'.
        ax (matplotlib.axes._subplots.AxesSubplot): The axis to plot on. If None, a new figure and axis are created.
        output_path (str): Path to save the generated plot.
    """
    if ax is None:
        plt.figure(figsize=(10, 7))
        ax = plt.gca()

    # Extract data by type
    registered_data = distance_data[distance_data['Type'] == 'Registered']
    non_registered_data = distance_data[distance_data['Type'] == 'Non-Registered']

    # Generate an array of animal-color codes
    registered_colors = [f'C{i}' for i in range(len(registered_data['Animal'].unique()))]
    non_registered_colors = [f'C{i}' for i in range(len(non_registered_data['Animal'].unique()))]

    # Plot boxplots with hue for each animal
    plot_boxplots(
        registered_data['Distance'].values,
        non_registered_data['Distance'].values,
        ax=ax,
        labels=("Registered", "Non-Registered"),
        ylabel="Distance (pixels)",
        title="Combined Distances: Registered vs Non-Registered Components",
        colors1=registered_colors,
        colors2=non_registered_colors
    )

    if not output_path is None:
        # Save the plot
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
    
    return ax

def process_cohort(session_file, cohort, dims, cohort_dir):
    """Process all animals in a cohort."""
    mice = session_file[session_file['Cohort'] == cohort]['Mouse'].unique()
    registration_data = []
    distance_data = []

    for animal in tqdm(mice, desc=f"Processing Cohort {cohort}"):
        animal_data = load_animal_data(session_file, animal)
        A_union, assignments, positions, session_indexing = register_sessions(animal_data, dims)

        # Gather validation and distance data
        gather_registration_data(assignments, cohort, animal, registration_data)
        registered_distances, non_registered_distances = compute_distances(assignments, session_indexing, positions)
        gather_distance_data(registered_distances, non_registered_distances, cohort, animal, distance_data)

        # Save registration results
        save_results(cohort_dir, cohort, animal, assignments, A_union, session_indexing)

    plot_combined_validation(registration_data, distance_data, cohort_dir, cohort)

def plot_combined_validation(registration_data, distance_data, output_dir, cohort):
    """Plot combined registration validation and average distances across all animals in a cohort."""
    # Combine and plot registration validation
    combined_registration_data = pd.concat(registration_data, ignore_index=True)
    # Create a grid for validation plot to fit 2 plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    plot_registration_validation(combined_registration_data, cohort, ax=ax1)
    
    # Combine and plot average distances
    combined_distance_data = pd.DataFrame(distance_data)
    plot_combined_distances(combined_distance_data, ax=ax2)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'combined_validation.png'))
    plt.close()

def main(session_file_path, output_base_dir):
    """Main function to process all cohorts."""
    session_file = preprocess_session_file(session_file_path)
    cohorts = session_file['Cohort'].unique()
    dims = (400, 640)  # Assumes fixed imaging dimensions

    for cohort in cohorts:
        cohort_dir = os.path.join(output_base_dir, cohort, 'registration')
        os.makedirs(cohort_dir, exist_ok=True)
        process_cohort(session_file, cohort, dims, cohort_dir)

# Execute the script
if __name__ == "__main__":
    session_file_path = 'data'
    output_base_dir = 'processed-data'
    main(session_file_path, output_base_dir)
