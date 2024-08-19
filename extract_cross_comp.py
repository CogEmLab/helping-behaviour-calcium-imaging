import matplotlib
matplotlib.use('Agg')  # Set the backend to 'Agg' before importing pyplot

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from caiman.source_extraction.cnmf.cnmf import load_CNMF
from tqdm import tqdm
import os

# Import custom functions from utils and plotting modules
from src.utils import get_most_active_components
from src.plotting import plot_heatmap, plot_boxplots

# Constants
WINDOW = 30  # seconds
FIG_SIZE = (30, 15)
GRID_SPEC = (2, 4)

def setup_plot():
    plt.style.use('seaborn-v0_8-talk')
    fig = plt.figure(figsize=FIG_SIZE)
    gs = GridSpec(*GRID_SPEC, figure=fig)
    
    
    # Set larger font sizes
    plt.rcParams.update({
        'font.size': 14,          # General font size
        'axes.titlesize': 18,     # Font size for axes titles
        'axes.labelsize': 16,     # Font size for axes labels
        'xtick.labelsize': 14,    # Font size for x-axis tick labels
        'ytick.labelsize': 14,    # Font size for y-axis tick labels
        'legend.fontsize': 14,    # Font size for legend
        'figure.titlesize': 20    # Font size for figure title
    })

    axes = {
        'heatmap': fig.add_subplot(gs[0:2, 0:2]),
        'histogram': fig.add_subplot(gs[0, 2]),
        'boxplots': fig.add_subplot(gs[0, 3]),
        'components': fig.add_subplot(gs[1, 2:4])
    }
    
    return fig, axes

def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    
    # Filter out specific sessions and cohorts
    excluded_sessions = ['Hab1', 'Hab2', 'Hab3', 'Toy', 'Pool', 'Empty']
    excluded_cohorts = []
    
    data = data[~data['Session'].isin(excluded_sessions) & ~data['Cohort'].isin(excluded_cohorts)]
    
    # Convert Mouse column to str with 4 digits
    data['Mouse'] = data['Mouse'].astype(str).str.zfill(4)
    
    # Add new columns
    data['Num of liberation cells'] = np.nan
    data['% of liberation cells'] = np.nan
    
    return data

def process_session(row, fps, window, plot_example=False, axes=None):
    # Extract necessary data from the row
    session = row['Session']
    model_path = row['CNMFE Model Path']
    mouse_id = row['Mouse']
    cohort = row['Cohort']
    total_num_components = row['Total Components']
    time_victim_crossing = row['TimeVictimBodyCrossing(s)']

    # Skip if crossing time is not available
    if np.isnan(time_victim_crossing):
        return np.nan, np.nan, np.nan

    # Load CNMF model and process data
    model = load_CNMF(model_path)
    npy_filepath = model_path.replace('.hdf5', '_Dff_z_scored.npy').replace('cnmfe_model', 'npy_data')
    DF_F_Z_scored = np.load(npy_filepath)

    high_quality_components = model.estimates.idx_components

    # Get SNR and spatial correlation values
    snr = model.estimates.SNR_comp
    sptl_corr = model.estimates.r_values

    # Process time window
    time_victim_crossing = int(time_victim_crossing * fps)
    window_frames = window * fps
    start = int(time_victim_crossing - window_frames)
    end = int(time_victim_crossing + window_frames)
    
    # Ensure indices are within bounds
    if start < 0 or end > DF_F_Z_scored.shape[1]:
        return np.nan, np.nan, np.nan
    
    print(f"Processing window: {start} - {end} in {DF_F_Z_scored.shape[1]} frames")

    DF_F_Z_scored = DF_F_Z_scored[:, start:end]

    print(f'Current shape of DF_F_Z_scored: {DF_F_Z_scored.shape}')

    # Check if DF_F_Z_scored is empty or has insufficient data
    if DF_F_Z_scored.size == 0:
        return np.nan, np.nan, np.nan

    # Compute the sum of dff inside the window
    sum_dff = np.sum(DF_F_Z_scored, axis=1)

    # Get most active components
    DF_F_Z_scored_selected, selected_ids = get_most_active_components(
        DF_F_Z_scored, 'tech_plots/threshold_plots', cohort=cohort, session=session, animal=mouse_id, plot_example=plot_example, ax=axes['histogram']
    )

    print(f"Selected {len(selected_ids)} components")

    # Separate sum_dff into selected and not selected components
    sum_dff_selected = sum_dff[selected_ids]
    sum_dff_not_selected = sum_dff[~np.isin(np.arange(len(sum_dff)), selected_ids)]

    # Plot heatmaps for selected components
    filename = f'tech_plots/heatmaps/{cohort}/{session}/{mouse_id}_heatmap.png'
    # Create a directory for the cohort and session
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plot_heatmap(DF_F_Z_scored_selected, filename, ax=None, window_to_use=window, FPS=fps, animal=mouse_id, day_of_helping=row['DayofHelping'])

    # Plot again if plot_example is True on heatmap axis of the setup_plot
    if plot_example:
        filename = None
        plot_heatmap(DF_F_Z_scored_selected, filename, axes['heatmap'], window, fps, mouse_id, row['DayofHelping'])

    # Calculate response scores
    pre_response = np.sum(DF_F_Z_scored_selected[:, :window_frames], axis=1)
    post_response = np.sum(DF_F_Z_scored_selected[:, window_frames:], axis=1)
    total_response = np.sum(DF_F_Z_scored_selected, axis=1)

    # Remove components with zero total response
    mask = total_response != 0
    if np.any(mask):
        response_score = (post_response[mask] - pre_response[mask]) / total_response[mask]
    else:
        response_score = np.nan

    print(f"Mask shape: {mask.shape}, response score shape: {response_score.shape}")

    # Check again that objects exist and not empty before returning
    if selected_ids.size == 0:
        return np.nan, np.nan, np.nan

    # Compute number of liberation cells that are selected and also in high_quality_components
    mask_for_counting_selected_comps = np.isin(selected_ids, high_quality_components)
    selected_ids_counted = selected_ids[mask_for_counting_selected_comps]

    return ({
        'selected_ids': selected_ids[mask],
        'response_score': response_score,
        'snr': snr[selected_ids][mask] if selected_ids.size > 0 else np.nan,
        'sptl_corr': sptl_corr[selected_ids][mask] if selected_ids.size > 0 else np.nan,
        'num_liberation_cells': len(selected_ids_counted),
        'percent_liberation_cells': (len(selected_ids_counted) / len(high_quality_components)) * 100 if total_num_components > 0 else np.nan
    }, sum_dff_selected, sum_dff_not_selected)

def main():
    # Setup
    fig, axes = setup_plot()
    data = load_and_preprocess_data('data/session_file_MiniscopeCW_all_Moises_latest.csv')
    
    # Create output directories
    os.makedirs('tech_plots/threshold_plots', exist_ok=True)
    
    # Initialize lists for metadata and components
    metadata = {
        'Cohort': [], 'Mouse': [], 'Session': [], 'Component ID': [],
        'Reg ID': [], 'Response Score': [], 'DayofHelping': [],
        'Model Path': [], 'SNR': [], 'Spatial Correlation': []
    }

    sum_components = {'selected': [], 'not_selected': []}
    
    # Make a random choice for the example plot
    random_idx = np.random.choice(data.index)

    # Process each session
    for idx, row in tqdm(data.iterrows(), total=len(data)):
        fps = row['FPS (CI downsampled)']
        window = 45 if fps == 20 else WINDOW

        # Plot example if random index
        plot_example = int(idx) == int(random_idx)
        
        result, sum_dff_selected, sum_dff_not_selected = process_session(row, fps, window, plot_example=plot_example, axes=axes)
                
        if result is not np.nan:
            # Update metadata
            for key in ['Cohort', 'Mouse', 'Session', 'DayofHelping', 'CNMFE Model Path']:
                metadata[key.replace('CNMFE Model Path', 'Model Path')].extend([row[key]] * len(result['selected_ids']))
                        
            metadata['Component ID'].extend(result['selected_ids'])
            metadata['Reg ID'].extend(range(len(result['selected_ids'])))  # Placeholder, update if needed
            metadata['Response Score'].extend(result['response_score'])
            metadata['SNR'].extend(result['snr'])
            metadata['Spatial Correlation'].extend(result['sptl_corr'])
            
            # Update data with liberation cell information
            data.loc[data.index == row.name, 'Num of liberation cells'] = result['num_liberation_cells']
            data.loc[data.index == row.name, '% of liberation cells'] = result['percent_liberation_cells']
            
            # Update sum of components with mean values
            sum_components['selected'].append(np.mean(sum_dff_selected))
            sum_components['not_selected'].append(np.mean(sum_dff_not_selected))

    # Create and save metadata DataFrame
    metadata_df = pd.DataFrame(metadata)
    metadata_df['Group'] = pd.cut(metadata_df['Response Score'], 
                                  bins=[-np.inf, -0.5, 0.5, np.inf],
                                  labels=['Pre-responsive', 'Mixed', 'Post-responsive'])
    metadata_df['Index'] = range(len(metadata_df))
    metadata_df.to_csv('processed-data/metadata_cross_comp.csv', index=False)
    
    # Update data with normalized liberation cells
    data['Normalized liberation cells'] = data['Num of liberation cells'] / data['Ca Imaging Length(s)']
    sum_normalized = data.groupby('Mouse')['Normalized liberation cells'].sum()
    data = data.join(sum_normalized, on='Mouse', rsuffix='_sum')
    data['Normalized liberation cells'] = (data['Normalized liberation cells'] / data['Normalized liberation cells_sum']) * 100
    data = data.drop(columns=['Normalized liberation cells_sum'])
    
    # # Remove rows with NaN values
    # data = data.dropna(subset=['Num of liberation cells', '% of liberation cells', 'Normalized liberation cells'])

    # # Remove NaN values from sum_components
    # sum_components = {key: np.array(value) for key, value in sum_components.items()}
    # sum_components = {key: value[~np.isnan(value)] for key, value in sum_components.items()}

    plot_boxplots(sum_components['selected'], sum_components['not_selected'], axes['boxplots'])

    order = [0, 1, 2, 3, 4, 'RECNvlCtx', 'REMNvlCtx', 'REMHBTD']
    sns.barplot(x='DayofHelping', y='% of liberation cells', data=data, 
                fill=False, capsize=0.1, err_kws={'linewidth': 1.5}, color='black', ax=axes['components'], order=order)
    sns.swarmplot(x='DayofHelping', y='% of liberation cells', data=data, 
                  hue='Mouse', size=8, palette='Set1', dodge=True, alpha=0.7, 
                  edgecolor='black', linewidth=0.5, ax=axes['components'], order=order)
    
    axes['components'].set_xlabel('Day of helping')
    axes['components'].set_ylabel('Percentage of liberation cells')
    # Set xticks in the same order as the order
    axes['components'].set_xticklabels(order)

    # Turn off spines for top and right side
    axes['components'].spines['top'].set_visible(False)
    axes['components'].spines['right'].set_visible(False)

    # Set legend to outside the plot
    axes['components'].legend(loc='upper left', bbox_to_anchor=(1, 1), title='Mouse ID')

    # Save results
    data.to_csv('data/session_file_MiniscopeCW_all_Moises_latest_liberation_cells.csv', index=False)
    plt.savefig('tech_plots/overview_figure_liberation_cells_extraction.pdf')
    # Also save as PNG
    plt.savefig('tech_plots/overview_figure_liberation_cells_extraction.png')
    plt.close(fig)

if __name__ == "__main__":
    main()