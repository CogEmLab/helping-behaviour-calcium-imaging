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
from typing import Dict, Tuple
import plotly.graph_objects as go
import yaml

# Constants
FIG_SIZE = (25, 25)
GRID_SPEC = (4, 4)
IMG_DIMS = (400, 640)
SESSION_ORDER = ['HBTD1', 'HBTD2', 'HBTD3', 'HBTD4', 'HBTD5', 'HBTD6', 'RECNvlCtx', 'Stranger', 'REMNvlCtx', 'REMHBTD']
DAYS_OF_HELPING_ORDER = [0, 1, 2, 3, 4, 'RECNvlCtx', 'Stranger', 'REMNvlCtx', 'REMHBTD']
GROUPS_COLORS = {'Pre-responsive': '#5798A1', 'Post-responsive': '#AB9F9D', 'Mixed': '#E3CFCC'}

with open('tech_docs/identity_switch_table.yaml', 'r') as f:
    identity_switch_data= yaml.safe_load(f)


def setup_plot() -> Tuple[plt.Figure, Dict[str, plt.Axes]]:
    """Set up the plot with a specific style and grid layout."""
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
        'pnr_reg_example1': fig.add_subplot(gs[0, 2]),
        'pnr_reg_example2': fig.add_subplot(gs[0, 3]),
        'violin_plot_response_score': fig.add_subplot(gs[0, :2]),
        'pre-responsive_examples': fig.add_subplot(gs[1, :2]),
        'post-responsive_examples': fig.add_subplot(gs[2, :2]),
        'mixed_examples': fig.add_subplot(gs[3, :2]),
        'flow_chart': fig.add_subplot(gs[3, 2:]),
        'pie_charts_1': fig.add_subplot(gs[1, 2]),
        'pie_charts_2': fig.add_subplot(gs[1, 3]),
        'pie_charts_3': fig.add_subplot(gs[2, 2]),
        'pie_charts_4': fig.add_subplot(gs[2, 3])
    }

    
    return fig, axes

def create_sankey_diagram(metadata):
    
    # Filter for specific days
    DAYS_OF_HELPING_ORDER = ['0', '1', '2', '3', '4', '5+']
    metadata = metadata[metadata['DayofHelping'].isin(DAYS_OF_HELPING_ORDER)]

    # Treat 'DayofHelping' as a categorical variable with sorted categories
    metadata['DayofHelping'] = pd.Categorical(metadata['DayofHelping'], 
                                               categories=sorted(metadata['DayofHelping'].unique()), 
                                               ordered=True)

    # Initialize transitions dictionary
    days = metadata['DayofHelping'].cat.categories
    groups = metadata['Group'].unique()
    transitions = {day: {group: {next_group: 0 for next_group in groups} for group in groups} 
                   for day in days[:-1]}

    # Count transitions between groups across days
    for day in days[:-1]:
        next_day_data = metadata[metadata['DayofHelping'] == days[days.get_loc(day) + 1]]
        id_switches = next_day_data['Identity Switch Value'].to_numpy()
        next_day_data['Preserved Identity Value'] = next_day_data['Preserved Identity Value'].fillna('nan').astype(str)
        preserved_identities = next_day_data['Preserved Identity Value'].to_numpy()

        unique_switches, counts = np.unique(id_switches, return_counts=True)
        unique_preserved_identities, counts_preserved = np.unique(preserved_identities, return_counts=True)

        # Update transitions based on counts
        for i, switch in enumerate(unique_switches[~np.isnan(unique_switches)]):
            code = identity_switch_data['identity_switch'][switch]
            if switch == 0:
                for j, preserved in enumerate(unique_preserved_identities[unique_preserved_identities != 'nan']):
                    transitions[day][preserved][preserved] = counts_preserved[j]
            else:
                transitions[day][code[0]][code[1]] = counts[i]

    # Prepare data for the Sankey diagram
    source, target, value, label_source, label_target = [], [], [], [], []
    for i, day in enumerate(days[:-1]):
        for group in groups:
            source_index = i * len(groups) + list(groups).index(group)
            if f"{day}_{group}" not in label_source:
                label_source.append(f"{day}_{group}")
            for next_group in groups:
                target_index = (i + 1) * len(groups) + list(groups).index(next_group)
                
                if f"{days[i+1]}_{next_group}" not in label_target:
                    label_target.append(f"{days[i+1]}_{next_group}")
                
                if transitions[day][group][next_group] > 0:
                    source.append(source_index)
                    target.append(target_index)
                    value.append(transitions[day][group][next_group])

    # Concatenate source and target labels
    label = label_source + label_target[-len(groups):]
    # Assign unique colors to each group based on the group name; read from the GROUPS_COLORS dictionary
    group_colors = {group.split('_')[1]: GROUPS_COLORS[group.split('_')[1]] for group in label}

    # Create the Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        node=dict(pad=15, thickness=20, line=dict(color="black", width=0.5), label=label, color=[group_colors[group.split('_')[1]] for group in label]),
        link=dict(source=source, target=target, value=value)
    )])
    # Figure layout
    # fig.update_layout(title_text="Identity Transitions over Days of Helping", font_size=20)
    # Save as html
    fig.write_html('figures/sankey_diagram.html')
    # Save the Sankey diagram as png and upload back
    fig.write_image('figures/sankey_diagram.png', width=1800, height=1000)
    # Load the image back
    img = plt.imread('figures/sankey_diagram.png')
    return img

def load_and_preprocess_session_file(file_path: str) -> pd.DataFrame:
    """Load and preprocess the session file."""
    try:
        data = pd.read_csv(file_path)
        
        # Filter out specific sessions and cohorts
        excluded_sessions = []
        excluded_cohorts = []
        
        data = data[~data['Session'].isin(excluded_sessions) & ~data['Cohort'].isin(excluded_cohorts)]
        
        return data
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return pd.DataFrame()

def plot_violin_response_score(metadata: pd.DataFrame, ax: plt.Axes) -> None:
    """Plot violin plot of response score over sessions."""
    sns.violinplot(x='Session', y='Response Score', data=metadata, ax=ax, order=SESSION_ORDER, color='k', fill=False, linewidth=1)
    sns.stripplot(x='Session', y='Response Score', data=metadata, ax=ax, order=SESSION_ORDER, hue='Group', dodge=True, alpha=0.8, palette=GROUPS_COLORS, jitter=False, s=8, linewidth=1)   
    # Make legend bigger
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[1:], labels=labels[1:], title='Group', title_fontsize='large', fontsize='large', loc='upper left')
    ax.set_title('Response Score over Sessions')
    ax.set_xlabel('Session')
    ax.set_ylabel('Response Score')

def get_random_tracked_metadata(metadata: pd.DataFrame) -> Tuple[str, str, str]:
    """Get a random choice from the tracked subset of metadata."""
    tracked_metadata = metadata[metadata['Tracked']]
    random_idx = np.random.choice(tracked_metadata.index)
    cohort, mouse, session = tracked_metadata.loc[random_idx, ['Cohort', 'Mouse', 'Session']]
    return cohort, mouse, session

def plot_pnr_images(axes: Dict[str, plt.Axes], pnr_image1: np.ndarray, pnr_image2: np.ndarray, 
                    com1: Tuple[float, float], com2: Tuple[float, float], 
                    session1: str, session2: str, component_id1: int, component_id2: int, mouse: str) -> None:
    """Plot the PNR images with center of mass markers."""
    axes['pnr_reg_example1'].imshow(pnr_image1, cmap='gray', aspect='equal')
    axes['pnr_reg_example1'].scatter(com1[1], com1[0], s=100, c='r', marker='o')
    axes['pnr_reg_example1'].set_title(f'{session1}, Mouse: {mouse}')
    axes['pnr_reg_example1'].axis('off')
    
    axes['pnr_reg_example2'].imshow(pnr_image2, cmap='gray', aspect='equal')
    axes['pnr_reg_example2'].scatter(com2[1], com2[0], s=100, c='r', marker='o')
    axes['pnr_reg_example2'].set_title(f'{session2}, Mouse: {mouse}')
    axes['pnr_reg_example2'].axis('off')

def plot_cell_examples(axes: Dict[str, plt.Axes], dff_z_scored: np.ndarray, component_ids: np.ndarray, 
                       group: str, time_victim_crossing: float, fps: float) -> None:
    """Plot examples of cell responses."""
    signal_start = int(time_victim_crossing - (30*fps))
    signal_end = int(time_victim_crossing + (30*fps))
    
    window_start = int(time_victim_crossing - (15*fps))
    window_end = int(time_victim_crossing + (15*fps))

    for j, component_id in enumerate(component_ids):
        dff_z_scored_component = dff_z_scored[component_id] + 10 * j
        axes[f'{group.lower()}_examples'].plot(dff_z_scored_component[signal_start:signal_end], 
                                               label=f'Component ID: {component_id}', color='k')
    
        # Highlight the window of interest
        axes[f'{group.lower()}_examples'].axvspan(window_start - signal_start, window_end - signal_start, color='lightgray', alpha=0.2)

    axes[f'{group.lower()}_examples'].set_title(f'{group} Examples')
    axes[f'{group.lower()}_examples'].set_xlabel('Time (s)')
    axes[f'{group.lower()}_examples'].set_ylabel('Dff Z-Scored')
    axes[f'{group.lower()}_examples'].set_xticks([0, 30*fps, 60*fps])
    axes[f'{group.lower()}_examples'].set_xticklabels([-30, 0, 30])

def main():
    # Setup
    fig, axes = setup_plot()
    session_file = load_and_preprocess_session_file('data/session_file_MiniscopeCW_all_Moises_latest_liberation_cells.csv')
    metadata = pd.read_csv('processed-data/metadata_cross_comp_updated.csv')

    # Plot violin plot
    plot_violin_response_score(metadata, axes['violin_plot_response_score'])

    # Plot sankey diagram for Group over days of helping
    img = create_sankey_diagram(metadata)
    # Plot the Sankey diagram, make it bigger

    axes['flow_chart'].imshow(img, aspect='equal')
    axes['flow_chart'].axis('off')

    # Plot pie charts for Group distribution for different days of helping (0,1,2)
    for i, day in enumerate(DAYS_OF_HELPING_ORDER[:4]):
        metadata_day = metadata[metadata['DayofHelping'] == str(day)]

        group_counts = metadata_day['Group'].value_counts()
        axes[f'pie_charts_{i+1}'].pie(group_counts, labels=group_counts.index, autopct='%1.1f%%', startangle=90, colors=[GROUPS_COLORS[group] for group in group_counts.index])
        axes[f'pie_charts_{i+1}'].set_title(f'Identity Distribution on Day of Helping {day}')

    # Get random tracked metadata
    cohort, mouse, session = get_random_tracked_metadata(metadata)


    # Process mouse metadata
    reg_id = metadata[(metadata['Mouse'] == mouse) & (metadata['Cohort'] == cohort)]['Reg ID'].iloc[0]
    mouse_metadata = metadata[(metadata['Mouse'] == mouse) & (metadata['Cohort'] == cohort) & (metadata['Reg ID'] == reg_id)].sort_values('Day')
    
    session1, session2 = mouse_metadata.loc[mouse_metadata['Day'].isin([1, 2]), 'Session'].values
    component_id1, component_id2 = mouse_metadata.loc[mouse_metadata['Day'].isin([1, 2]), 'Component ID'].values

    # Get PNR image paths and load CNMF models
    session_file_mouse = session_file[(session_file['Mouse'] == mouse) & (session_file['Cohort'] == cohort)]
    pnr_image_path1 = session_file_mouse[session_file_mouse['Session'] == session1]['Peak-to-Noise Ratio Path'].values[0]
    pnr_image_path2 = session_file_mouse[session_file_mouse['Session'] == session2]['Peak-to-Noise Ratio Path'].values[0]
    cnmfe_model_path1 = session_file_mouse[session_file_mouse['Session'] == session1]['CNMFE Model Path'].values[0]
    cnmfe_model_path2 = session_file_mouse[session_file_mouse['Session'] == session2]['CNMFE Model Path'].values[0]

    cnmfe_model1 = load_CNMF(cnmfe_model_path1)
    cnmfe_model2 = load_CNMF(cnmfe_model_path2)

    pnr_image1 = imread(pnr_image_path1)
    pnr_image2 = imread(pnr_image_path2)

    # Calculate center of mass for labels
    label1 = (cnmfe_model1.estimates.A[:, component_id1].toarray().reshape(pnr_image1.shape, order='F') > 0).astype(int)
    label2 = (cnmfe_model2.estimates.A[:, component_id2].toarray().reshape(pnr_image2.shape, order='F') > 0).astype(int)
    com1 = ndimage.measurements.center_of_mass(label1)
    com2 = ndimage.measurements.center_of_mass(label2)

    # Plot PNR images
    plot_pnr_images(axes, pnr_image1, pnr_image2, com1, com2, session1, session2, component_id1, component_id2, mouse)

    # Get metadata for example
    metadata_example = metadata[(metadata['Cohort'] == cohort) & (metadata['Mouse'] == mouse) & (metadata['Session'] == session1)]
    session_file_example = session_file[(session_file['Cohort'] == cohort) & (session_file['Mouse'] == mouse) & (session_file['Session'] == session1)]
    fps = session_file_example['FPS (CI downsampled)'].values[0]
    time_victim_crossing = session_file_example['TimeVictimBodyCrossing(s)'].values[0] * fps

    # Get examples of different cell types
    cell_examples = {
        'Pre-responsive': metadata_example[metadata_example['Group'] == 'Pre-responsive'].sample(3)['Component ID'].values,
        'Post-responsive': metadata_example[metadata_example['Group'] == 'Post-responsive'].sample(3)['Component ID'].values,
        'Mixed': metadata_example[metadata_example['Group'] == 'Mixed'].sample(3)['Component ID'].values
    }

    # Load DFF z-scored data
    npy_path = cnmfe_model_path1.replace('.hdf5', '_Dff_z_scored.npy').replace('cnmfe_model', 'npy_data')
    dff_z_scored = np.load(npy_path)

    # Plot examples of different cell types
    for group, ids in cell_examples.items():
        plot_cell_examples(axes, dff_z_scored, ids, group, time_victim_crossing, fps)

    # Finalize plot
    for ax in axes.values():
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig('tech_plots/reg_data_overview.pdf', format='pdf')
    plt.savefig('tech_plots/reg_data_overview.png', format='png')

if __name__ == "__main__":
    main()