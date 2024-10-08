import matplotlib
matplotlib.use('Agg')  # Set the backend to 'Agg' before importing pyplot

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from caiman.source_extraction.cnmf.cnmf import load_CNMF
from tifffile import imread
from scipy import ndimage
from scipy import stats

# Constants
FIG_SIZE = (25, 10)
GRID_SPEC = (2, 5)
IMG_DIMS = (400, 640)
SESSION_ORDER = ['Hab1', 'Hab2', 'Hab3', 'HBTD1', 'HBTD2', 'HBTD3', 'HBTD4', 'HBTD5', 'HBTD6', 'RECNvlCtx', 'Stranger', 'Toy', 'Empty', 'Pool', 'REMNvlCtx', 'REMHBTD']
DAYS_OF_HELPING_ORDER = [0, 1, 2, 3, 4, 'RECNvlCtx', 'REMNvlCtx', 'REMHBTD']

def setup_plot():
    """Set up the plot with a specific style and grid layout."""
    plt.style.use('seaborn-v0_8-talk')
    fig = plt.figure(figsize=FIG_SIZE)
    gs = GridSpec(*GRID_SPEC, figure=fig)
    
    axes = {
        'pnr_image': fig.add_subplot(gs[0, 0]),
        'labels_example': fig.add_subplot(gs[1, 0]),
        'comp_over_time': fig.add_subplot(gs[0, 1:3]),
        'comp_session_len': fig.add_subplot(gs[1, 1:2]),
        'normalized_comp': fig.add_subplot(gs[0, 3:]),
        'normalized_comp_over_days_of_helping': fig.add_subplot(gs[1, 2:4]),
        'norm_comp_latexncy': fig.add_subplot(gs[1, 4]),
    }
    
    return fig, axes

def load_and_preprocess_session_file(file_path):
    """Load and preprocess the session file."""
    data = pd.read_csv(file_path)
    
    # Filter out specific sessions and cohorts
    excluded_sessions = []
    excluded_cohorts = []
    
    data = data[~data['Session'].isin(excluded_sessions) & ~data['Cohort'].isin(excluded_cohorts)]
    
    # Convert Mouse column to str with 4 digits
    data['Mouse'] = data['Mouse'].astype(str).str.zfill(4)

    return data

def plot_pnr_image_and_labels(axes, pnr_image, A_labels):
    """Plot the correlation image and labels example."""
    pnr_image = pnr_image[:IMG_DIMS[0], :400]

    # Add title to the labels example
    A_labels = A_labels.astype(float)
    A_labels[A_labels == 0] = np.nan
    A_labels = A_labels[:IMG_DIMS[0], :400]
    
    axes['pnr_image'].imshow(pnr_image, cmap='gray')
    axes['labels_example'].imshow(A_labels, cmap='tab20c')
    
    axes['pnr_image'].axis('off')
    axes['labels_example'].axis('off')

def plot_components_over_time(ax, data):
    """Plot the number of accepted components over time."""
    sns.barplot(x='Session', y='Num Accepted', data=data, order=SESSION_ORDER,
                fill=False, capsize=0.1, err_kws={'linewidth': 1.5}, color='black', ax=ax)
    
    sns.swarmplot(x='Session', y='Num Accepted', data=data, hue='Mouse', size=8, 
                  palette='Set1', dodge=True, alpha=0.7, ax=ax, order=SESSION_ORDER, linewidth=1, legend=False)
    
    ax.set_xlabel('Session')
    ax.set_ylabel('# Accepted Components')
    ax.set_xticklabels(SESSION_ORDER, rotation=45)

def plot_normalized_components(ax, data):
    """Plot the normalized components over sessions."""

    # Remove habituation sessions from the data
    data = data[~data['Session'].str.contains('Hab')]
    SESSION_ORDER = ['HBTD1', 'HBTD2', 'HBTD3', 'HBTD4', 'HBTD5', 'HBTD6', 'RECNvlCtx', 'Stranger', 'Toy', 'Empty', 'Pool', 'REMNvlCtx', 'REMHBTD']

    sns.barplot(x='Session', y='Normalized components', data=data, fill=False, 
                capsize=0.1, err_kws={'linewidth': 1.5}, color='black', ax=ax, order=SESSION_ORDER)
    sns.swarmplot(x='Session', y='Normalized components', data=data, hue='Mouse', 
                  size=7, palette='Set1', dodge=True, alpha=0.7, ax=ax, order=SESSION_ORDER, linewidth=1)
    
    ax.set_xlabel('Session')
    ax.set_ylabel('Normalized Components')
    # Set legend outside
    ax.legend(title='Mouse ID', bbox_to_anchor=(1.05, 1))
    ax.set_xticklabels(SESSION_ORDER, rotation=45)

def plot_components_vs_session_length(ax, data):
    """Plot scatter plot of session length vs num accepted components with two regression lines."""
    # Scatter plot with two regression lines
    sns.scatterplot(x='Ca Imaging Length(s)', y='Num Accepted', data=data, hue='Cohort', palette='Set2', ax=ax, s=70, alpha=0.7, linewidth=0.5, edgecolor='k')

    # Subset data for Oct_2021 Cohort
    oct_2021_data = data[data['Cohort'] == 'Oct_2021']
    rest_of_data = data[data['Cohort'] != 'Oct_2021']

    # Remove nan values
    oct_2021_data = oct_2021_data.dropna(subset=['Ca Imaging Length(s)', 'Num Accepted'])
    rest_of_data = rest_of_data.dropna(subset=['Ca Imaging Length(s)', 'Num Accepted'])

    # Plot regression line for Oct_2021 Cohort
    sns.regplot(x='Ca Imaging Length(s)', y='Num Accepted', data=oct_2021_data, ax=ax, scatter=False, color='k', label='Oct_2021 Cohort')

    # Plot regression line for the rest of the data
    sns.regplot(x='Ca Imaging Length(s)', y='Num Accepted', data=rest_of_data, ax=ax, scatter=False, label='Other Cohorts', color='gray')
    
    ax.set_xlabel('Session Length (s)')
    ax.set_ylabel('# Accepted Components')
    ax.legend(title='Cohort')
    
def plot_normalized_components_over_days(ax, data):
    """Plot the normalized components over days of helping."""

    sns.barplot(x='DayofHelping', y='Normalized components', data=data, fill=False, 
                capsize=0.1, err_kws={'linewidth': 1.5}, color='black', ax=ax, order=DAYS_OF_HELPING_ORDER)
    sns.swarmplot(x='DayofHelping', y='Normalized components', data=data, hue='Mouse', 
                  size=7, palette='Set1', dodge=True, alpha=0.7, ax=ax, order=DAYS_OF_HELPING_ORDER, linewidth=1, legend=False)
    
    ax.set_xlabel('Days of Helping')
    ax.set_ylabel('Normalized Components')

def plot_normalized_components_vs_latency(ax, data):
    """Plot the normalized components over latency."""
    sns.scatterplot(x='LatencyMinConverted', y='Normalized components', data=data, ax=ax, s=30, color='black')

    # Remove nan values
    data = data.dropna(subset=['LatencyMinConverted', 'Normalized components'])
    
    # plot regression line
    sns.regplot(x='LatencyMinConverted', y='Normalized components', data=data, ax=ax, scatter=False, color='black')

    # Add correlation coefficient
    corr = stats.pearsonr(data['LatencyMinConverted'], data['Normalized components'])[0]
    ax.text(0.05, 0.95, f'Correlation: {corr:.2f}', transform=ax.transAxes)

    # Add axis labels
    ax.set_xlabel('Latency (min)')
    ax.set_ylabel('Normalized Components')

def smooth_and_shrink_label(label, smooth_sigma=1, shrink_pixels=1):
    # Smooth the label
    smoothed = ndimage.gaussian_filter(label.astype(float), sigma=smooth_sigma)
    
    # Threshold to recreate binary mask
    smoothed = (smoothed > 0.5).astype(int)
    
    # Erode to shrink by a few pixels
    if shrink_pixels > 0:
        smoothed = ndimage.binary_erosion(smoothed, iterations=shrink_pixels).astype(int)
    
    return smoothed * (label != 0)  # Preserve original label values

def main():
    # Setup
    fig, axes = setup_plot()
    session_file = load_and_preprocess_session_file('data/session_file_MiniscopeCW_all_Moises_latest.csv')

    # Make a random choice for the example plot
    random_idx = np.random.choice(session_file.index)
    mouse, session, cohort = session_file.loc[random_idx, ['Mouse', 'Session', 'Cohort']]
    print(f'Randomly selected mouse: {mouse}, session: {session}, cohort: {cohort}')

    # Get the correlation image path and the CNFE model path
    pnr_image_path = session_file.loc[random_idx, 'Peak-to-Noise Ratio Path']
    cnmfe_model_path = session_file.loc[random_idx, 'CNMFE Model Path']

    # Load the CNMF model and correlation image
    cnmfe_model = load_CNMF(cnmfe_model_path)
    accepted_components = cnmfe_model.estimates.idx_components
    pnr_image = imread(pnr_image_path)

    # Generate labels for components
    A_labels = np.zeros(pnr_image.shape, dtype=int)

    for i, comp in enumerate(accepted_components):
        label = (cnmfe_model.estimates.A[:, comp].toarray().reshape(pnr_image.shape, order='F') > 0).astype(int)
        smoothed_label = smooth_and_shrink_label(label, smooth_sigma=3, shrink_pixels=2)
        A_labels = np.maximum(A_labels, smoothed_label * (i + 1))
        
    # Plot correlation image and labels
    plot_pnr_image_and_labels(axes, pnr_image, A_labels)
    # Add titles to both image with cohort, session and mouse
    axes['pnr_image'].set_title(f'{cohort} - {session} - {mouse}')
    axes['labels_example'].set_title(f'{cohort} - {session} - {mouse}')

    # Plot various components
    plot_components_over_time(axes['comp_over_time'], session_file)
    plot_normalized_components(axes['normalized_comp'], session_file)
    plot_components_vs_session_length(axes['comp_session_len'], session_file)
    plot_normalized_components_over_days(axes['normalized_comp_over_days_of_helping'], session_file)
    plot_normalized_components_vs_latency(axes['norm_comp_latexncy'], session_file)

    # Turn off spines for top and right side for all axes
    for ax in axes.values():
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
                                                      
    # Finalize and save the figure
    plt.tight_layout()
    plt.savefig('tech_plots/components_overview.pdf', format='pdf')
    plt.savefig('tech_plots/components_overview.png', format='png')

if __name__ == "__main__":
    main()