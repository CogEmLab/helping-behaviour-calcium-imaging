from caiman.source_extraction.cnmf.cnmf import load_CNMF
from caiman.base.rois import register_multisession
from caiman.utils import visualization
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from tifffile import imread
from tqdm import tqdm

session_file = pd.read_csv('data/updated_session_file_MiniscopeCW_all_Moises.csv')

session_file['Mouse'] = session_file['Mouse'].astype(str)
session_file['Mouse'] = session_file['Mouse'].apply(lambda x: x.zfill(4))

# Get all unique cohorts in the dataset
cohorts = session_file['Cohort'].unique()

for cohort in cohorts:
    print(cohort)

    # # Skip if "Oct_2021"
    # if cohort == 'Oct_2021' or cohort == 'Mar_2024' or cohort == 'Jan_2024':
    #     continue

    # Get all unique animals in the dataset
    mice = session_file[session_file['Cohort'] == cohort]['Mouse'].unique()

    for animal in tqdm(mice):
        animal_data = session_file[session_file['Mouse'] == animal]

        # Get dims
        dims = (400, 640)

        # Load the CNMF object for each session
        spatial = []
        templates = []
        session_indexing_list = {}
        for i in range(len(animal_data)):
            session = animal_data.iloc[i]
            # Get info on which session it is and store its correct index
            session_indexing_list[session['Session']] = i
            spatial.append(load_CNMF(session['CNMFE Model Path']).estimates.A.toarray())
            templates.append(imread(session['Correlation Image Path']))
        
        # Register the CNMF objects
        A_union, assignments, __ = register_multisession(A=spatial, templates=templates, dims=dims)

        np.save(os.path.join(f'processed-data/{cohort}', f'{animal}_registered_ids.npy'), assignments)
        np.save(os.path.join(f'processed-data/{cohort}', f'{animal}_registered_spatial.npy'), A_union)
        # Save indexing scheme
        np.save(os.path.join(f'processed-data/{cohort}', f'{animal}_session_indexing.npy'), session_indexing_list)