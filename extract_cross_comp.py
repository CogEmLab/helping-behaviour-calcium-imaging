from src.utils import *
from src.plotting import *

import pandas as pd
from glob import glob
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from caiman.source_extraction.cnmf.cnmf import load_CNMF
from tqdm import tqdm

# Set style for paper
plt.style.use('seaborn-v0_8-talk')
#print(plt.style.available)

# Create the 2x5 grid for the subplots and set the figure size
fig = plt.figure(figsize=(25, 10))
gs = GridSpec(2, 4, figure=fig)

# Add an axis for the heatmap example of the DF_F_Z_scored, it should be 2x2
ax1 = fig.add_subplot(gs[0:2, 0:2]) #& Axis for the example of the DF_F_Z_scored
ax2 = fig.add_subplot(gs[0, 2]) #& Axis for the example of the histogram of the DF_F_Z_scored
ax3 = fig.add_subplot(gs[0, 3]) #& Axis for the box plots of df/f values
ax4 = fig.add_subplot(gs[1, 2:4]) #& Axis for the % of components over days of helping

# Load the data
data = pd.read_csv('data/updated_session_file_MiniscopeCW_all_Moises.csv')

# Remove HAB1, HAB2, HAB3 sessions from data
data = data[data['Session'] != 'Hab1']
data = data[data['Session'] != 'Hab2']
data = data[data['Session'] != 'Hab3']

# Remove some control sessions from data
data = data[data['Session'] != 'Toy']
data = data[data['Session'] != 'Pool']
data = data[data['Session'] != 'Empty']

# Skip cohorts Oct_2021 and Mar_2024, May_2024
data = data[data['Cohort'] != 'Oct_2021']
data = data[data['Cohort'] != 'Mar_2024']
data = data[data['Cohort'] != 'May_2024']

# Convert Mouse colunm to str with 4 digits
data['Mouse'] = data['Mouse'].astype(str).str.zfill(4)

# Also add a column for the raw number of liberation cells
data['Num of liberation cells'] = np.nan
# Add the column for the % of liberation cells 
data[f'% of liberation cells'] = np.nan
# Add the column for normalized number of liberation cells
data['Normalized num of liberation cells'] = np.nan

# Create a folder to save the plots
if not os.path.exists('tech_plots/threshold_plots'):
    os.makedirs('tech_plots/threshold_plots')

WINDOW = 30 # seconds

# Collect metadata
COMPONENTS_ID = []
REG_IDS = []
COHORT = []
MICE_ID = []
SESSION = []
RESPONSE_SCORES = []
IDENTITY = []
DAY_OF_HELPING = []
SESSION = []
MODEL_PATH = []
SNR = []
SPTL_CORR = []

# Collect the data
# collected_data = []

# Collect sum of components entry
sum_of_components_selected = []
sum_of_components_not_selected = []

sessions = data['Session'].unique()
plot_example = False

# Get random from day data to plot an example
random_idx = np.random.choice(range(len(data)))

# For each row in the data
for i in tqdm(range(len(data))):

    # Set the flag whether to plot as an example of not
    if random_idx == i:
        plot_example = True

    # Get the session name
    session = data.iloc[i]['Session']
    # Get path to CNMFE file
    model_path = data.iloc[i]['CNMFE Model Path']
    # Get FPS from the data
    fps = data.iloc[i]['FPS (CI downsampled)']
    # Get the mouse id
    mouse_id = data.iloc[i]['Mouse']
    # Get the cohort name
    cohort = data.iloc[i]['Cohort']
    # Get the total number of components
    total_num_components = data.iloc[i]['Total Components']

    # Also get registration files
    path_to_reg = os.path.join(f'processed-data/{cohort}/registration', f'{mouse_id}_registered_ids.npy')
    path_to_reg_indexing = os.path.join(f'processed-data/{cohort}/registration', f'{mouse_id}_session_indexing.npy')

    reg_matrix = np.load(path_to_reg)
    reg_indexing = np.load(path_to_reg_indexing, allow_pickle=True).item()
    
    unique_reg_ids = []

    for j in range(reg_matrix.shape[1]):
        # Get iwhere non-nan values are
        ids = np.where(~np.isnan(reg_matrix[:, j]))[0]
        # Add only ids that are not already in any list
        unique_reg_ids.extend(list(set(ids) - set(unique_reg_ids)))
    
    len_matr = len(unique_reg_ids)
    
    assert reg_matrix.shape[0] == len_matr
    
    reg_vector = reg_matrix[:, reg_indexing[session]]

    # Define a vector with just indices of the registration ids
    reg_ids = np.arange(0, reg_vector.shape[0])
    
    if fps == 30:
        window_to_use = WINDOW
    elif fps == 20:
        window_to_use = 45

    # Get time when the victim crosses the helpers chamber
    time_victim_crossing = data.iloc[i]['TimeVictimBodyCrossing(s)'] # in seconds

    # If crossing time is not available, skip the row
    if np.isnan(time_victim_crossing):
        continue
    
    # Convert to time in frames
    time_victim_crossing = int(time_victim_crossing * fps)

    # Also convert window to frames
    window_to_use = window_to_use * fps

    # Load the model
    model = load_CNMF(model_path)
    model.estimates.detrend_df_f(detrend_only=True)
    DF_F = model.estimates.F_dff

    print('DF_F:', DF_F.shape)
    print('DO I DO THE Z-SCORING CORRECTLY?')

    # Normalize dF/F
    DF_F_Z_scored = (DF_F - DF_F.mean(axis=1)[:, None]) / DF_F.std(axis=1)[:, None]

    # Get SNR and spatial correlation values of the components
    snr = model.estimates.SNR_comp
    sptl_corr = model.estimates.r_values
    
    # Check that number of non-nan values in the registration matrix is the same as the number of components in the model
    assert model.estimates.A.shape[1] == np.sum(~np.isnan(reg_matrix[:, reg_indexing[session]]))
    
    # Get only window around TimeVictimCrossing
    start = int((time_victim_crossing - window_to_use))
    end = int((time_victim_crossing + window_to_use))
    DF_F_Z_scored = DF_F_Z_scored[:, start:end]
    num_of_comp_in_DF_F_Z_scored = DF_F_Z_scored.shape[0]

    # Check that number of components in the DF_F_Z_scored is the same as in the output file
    assert num_of_comp_in_DF_F_Z_scored == total_num_components

    # Check if the DF_F_Z_scored is empty
    if DF_F_Z_scored.size == 0:
        print('Empty DF_F_Z_scored, skipping')
        continue

    # Get for all components the sum of the values in the window
    sum_values = np.sum(DF_F_Z_scored, axis=1)

    #* Get only the components that are the most active in the window; ONE OF THE CENTRAL FUNCTIONS HERE
    DF_F_Z_scored_selected, selected_ids = get_most_active_components(DF_F_Z_scored, 'tech_plots/threshold_plots', plot_example=plot_example, ax=ax2)

    # Separate the sums into selected and not selected components
    sum_values_selected = sum_values[selected_ids]
    sum_values_not_selected = sum_values[~np.isin(np.arange(0, total_num_components), selected_ids)]

    # Take mean sum in each group
    mean_sum_selected = np.mean(sum_values_selected)
    mean_sum_not_selected = np.mean(sum_values_not_selected)

    # Add the mean sum to the list
    sum_of_components_selected.append(mean_sum_selected)
    sum_of_components_not_selected.append(mean_sum_not_selected)

    # Plot the heatmap of the DF_F_Z_scored_selected
    ax = plot_heatmap(DF_F_Z_scored_selected, None, window_to_use, fps, mouse_id, data.iloc[i]['DayofHelping'])

    # Save the heatmap of the DF_F_Z_scored_selected to the tech_plots/cohort/session folder, create folder if does not exist
    path_to_save = f'tech_plots/{cohort}/{session}/heatmap_{mouse_id}_{session}.png'

    if not os.path.exists(f'tech_plots/{cohort}/{session}'):
        os.makedirs(f'tech_plots/{cohort}/{session}')
    plt.savefig(path_to_save)
    plt.close()
    
    if plot_example:
        fig, ax1 = plot_heatmap(DF_F_Z_scored_selected, ax1, window_to_use, fps, mouse_id, data.iloc[i]['DayofHelping'])
        plot_example = False

    # Add a test to check if the DF_F_Z_scored is empty
    if DF_F_Z_scored_selected.size == 0:
        print('Empty DF_F_Z_scored_selected, skipping')
        continue
    
    # Take only snrs and spatial correlations of selected components 
    snr = snr[selected_ids]
    sptl_corr = sptl_corr[selected_ids]

    # Take only reg ids where reg[day+1] == selected_ids
    mask = np.isin(reg_vector, selected_ids)
    # Also print what is not present
    mask_not = ~np.isin(selected_ids, reg_vector)
    reg_ids = reg_ids[mask]

    # For each component in the DF_F_Z_scored compute pre-post response score 
    pre_response = np.sum(DF_F_Z_scored_selected[:, :window_to_use], axis=1)
    post_response = np.sum(DF_F_Z_scored_selected[:, window_to_use:], axis=1)
    total_response = np.sum(DF_F_Z_scored_selected, axis=1)

    # Remove ids where total response is 0
    mask = total_response != 0
    # selected_components_ids = selected_components_ids[mask]
    pre_response = pre_response[mask]
    post_response = post_response[mask]
    total_response = total_response[mask]
    DF_F_Z_scored_selected = DF_F_Z_scored_selected[mask]
    selected_ids = selected_ids[mask]
    reg_ids = reg_ids[mask]

    # Compute response score
    response_score = (post_response - pre_response) / total_response

    # Save info about day of crossing and average activity to the metadata matrix
    if DF_F_Z_scored_selected.shape[1] != window_to_use * 2:
        print('Skipping:', mouse_id, session, 'DF_F_Z_scored_selected:', DF_F_Z_scored_selected.shape)
        continue
                
    # Add metadata to the lists; components ID list, Session list (multiply by number of components) and Mouse ID (multiply by number of components)
    # Save selected ids and their reg ids
    COMPONENTS_ID.extend(selected_ids[mask].tolist())
    REG_IDS.extend(reg_ids[mask].tolist())
    COHORT.extend([cohort] * DF_F_Z_scored_selected.shape[0])
    MICE_ID.extend([mouse_id] * DF_F_Z_scored_selected.shape[0])
    RESPONSE_SCORES.extend(response_score.tolist())
    DAY_OF_HELPING.extend([data.iloc[i]['DayofHelping']] * DF_F_Z_scored_selected.shape[0])
    SESSION.extend([session] * DF_F_Z_scored_selected.shape[0])
    MODEL_PATH.extend([model_path] * DF_F_Z_scored_selected.shape[0])
    SNR.extend(snr[mask].tolist())
    SPTL_CORR.extend(sptl_corr[mask].tolist())

    # Print all lengths to check if they are the same
    print('Component ID:', len(COMPONENTS_ID), 'Reg ID:', len(REG_IDS), 'Cohort:', len(COHORT), 'Mouse ID:', len(MICE_ID), 'Response Score:', len(RESPONSE_SCORES), 'Day of Helping:', len(DAY_OF_HELPING), 'Session:', len(SESSION), 'Model Path:', len(MODEL_PATH), 'SNR:', len(SNR), 'Spatial Correlation:', len(SPTL_CORR))
    # Check that all arrays are the same length and print what violates the condition
    assert len(COMPONENTS_ID) == len(REG_IDS) == len(COHORT) == len(MICE_ID) == len(RESPONSE_SCORES) == len(DAY_OF_HELPING) == len(SESSION) == len(MODEL_PATH) == len(SNR) == len(SPTL_CORR)

    # Save the number and percentage of crossing cells
    data.loc[(data['Mouse'] == mouse_id) & (data['Session'] == session), 'Num of liberation cells'] = len(selected_ids)
    data.loc[(data['Mouse'] == mouse_id) & (data['Session'] == session), f'% of liberation cells'] = len(selected_ids) / total_num_components

# Save the metadata to the dataframe
metadata = pd.DataFrame({'Cohort': COHORT, 'Mouse': MICE_ID, 'Session': SESSION, 'Component ID': COMPONENTS_ID, 'Reg ID': REG_IDS, 'Response Score': RESPONSE_SCORES, 'Day of Helping': DAY_OF_HELPING, 'Model Path': MODEL_PATH, 'SNR': SNR, 'Spatial Correlation': SPTL_CORR})

# Separate the data into two groups based on the response score (positive and negative)
metadata['Group'] = 'Mixed'
metadata.loc[metadata['Response Score'] > 0.5, 'Group'] = 'Post-responsive'
metadata.loc[metadata['Response Score'] < -0.5, 'Group'] = 'Pre-responsive'

# Update index column with new correct indexes
metadata['Index'] = np.arange(0, len(metadata))

# Save the metadata to the csv file
metadata.to_csv('processed-data/metadata_cross_comp.csv', index=False)

data['Normalized liberation cells'] = data['Num of liberation cells'] / data['Ca Imaging Length(s)']

# For each unique mouse, sum the normalized components over all days
sum_normalized_components = data.groupby(['Mouse'])['Normalized liberation cells'].sum()
# Add the sum of normalized components to the data
updated_data = data.join(sum_normalized_components, on='Mouse', rsuffix='_sum')
# Compute K_id(Num Accepted components for animal i and day d) as k_id (normalized by Session Duration) /sum(k_il over all days) for each animal i and day d
updated_data['Normalized liberation cells'] = (updated_data['Normalized liberation cells'] / updated_data['Normalized liberation cells_sum']) * 100

# Remove the sum of normalized components
updated_data = updated_data.drop(columns=['Normalized liberation cells_sum'])

# Plot the boxplots of sum of DF/F values for selected and not selected components
plot_boxplots(sum_of_components_selected, sum_of_components_not_selected, ax3)

# Plot Normalized liberation cells over days of helping for each mouse
sns.barplot(x='DayofHelping', y='Normalized liberation cells', data=data, fill=False, capsize=0.1, err_kws={'linewidth': 1.5}, color='black', ax=ax4)

# Add the individual data points
sns.swarmplot(x='Session', y='Normalized components', data=data, hue='Mouse', size=6, palette='Set1', dodge=True, alpha=0.7, edgecolor='black', linewidth=0.5, ax=ax4)

# Set the labels and title
ax4.set_xlabel('Day of helping')
ax4.set_ylabel('Normalized number of liberation cells')
ax4.set_title('Normalized number of liberation cells over days of helping')
# Set ticks for all unique days of helping
ax4.set_xticks(data['DayofHelping'].unique())

# Turn off the top and right spines for all axes
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# Also save updated data
data.to_csv('processed-data/all_data_cnmfe_anna_normalized_cross_comp.csv', index=False)

# Save the figure 
plt.savefig('tech_plots/figure1.pdf')