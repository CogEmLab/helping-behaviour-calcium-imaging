import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from src.plotting import plot_random_traces, plot_heatmap
from src.session_loader import Session
from matplotlib.gridspec import GridSpec
from typing import Dict, Tuple
import random

FIG_SIZE = (30, 15)
GRID_SPEC = (4, 6)

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
        'pnr_image': fig.add_subplot(gs[0, 0]),
        'labels': fig.add_subplot(gs[0, 1]),
        'traces': fig.add_subplot(gs[1:, 0:2]),
        'heatmap_helper': fig.add_subplot(gs[0:2, 2:4]),
        'collective_activity_helper': fig.add_subplot(gs[2, 2:4]),
        'annotations_helper': fig.add_subplot(gs[3, 2:4]),
        'heatmap_non_helper': fig.add_subplot(gs[0:2, 4:6]),
        'collective_activity_non_helper': fig.add_subplot(gs[2, 4:6]),
        'annotations_non_helper': fig.add_subplot(gs[3, 4:6]),
    }
    
    return fig, axes

def plot_session(session: Session) -> None:
    """Plots various aspects of a session."""

    # Load data
    pnr_image = session.load_pnr_image()
    labels = session.get_neuron_labels()
    accepted_comps = session.load_accepted_components()
    dff_padded = session.load_dff_signal()
    filtered_dff_padded = session.filter_dff_signal(dff_padded, accepted_comps)
    all_behaviour_timestamps = session.load_annotations()

    # Create directories if they do not exist
    fig_dir = os.path.join('figures', 'general', session.cohort, session.session)
    os.makedirs(fig_dir, exist_ok=True)

    # Plot PNR image
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(pnr_image[:, 100:-100], cmap='gray')
    ax.axis('off')
    filename = os.path.join(fig_dir, f'{session.animal_id}_{session.session_outcome}_pnr.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    # Plot labels
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    cmap = plt.cm.rainbow
    colors = cmap(np.linspace(0, 1, 80))
    colors[0] = (1, 1, 1, 1)  # Set first color to white
    np.random.shuffle(colors[1:])  # Shuffle the rest of the colors
    cmap = matplotlib.colors.ListedColormap(colors)
    ax.imshow(labels[:, 100:-100], cmap=cmap)
    ax.axis('off')
    filename = os.path.join(fig_dir, f'{session.animal_id}_{session.session_outcome}_labels.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    # Plot random traces
    plot_random_traces(
        filtered_dff_padded[:, 1:], num_traces=15,
        filename=os.path.join(fig_dir, f'{session.animal_id}_{session.session_outcome}_random_traces.png'), fps=session.calcium_fps)

    # Sort traces before plotting heatmap
    sorted_traces = filtered_dff_padded[np.argsort(np.mean(filtered_dff_padded[:, 1:], axis=1))]
    # Plot heatmap
    plot_heatmap(
        sorted_traces[:, 1:], filename=os.path.join(fig_dir, f'{session.animal_id}_{session.session_outcome}_heatmap_sorted.png'),
        FPS=session.calcium_fps, aligned=False, title=f'{session.animal_id}_{session.session_outcome}_generated_heatmap')

    # Plot collective behaviour
    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    ax.plot(np.mean(filtered_dff_padded[:, 1:], axis=0), color='k')
    ax.set_title(f'Mean Activity across {session.animal_id}_{session.session_outcome} Neurons')
    ax.set_ylabel('Mean Activity')
    ax.set_xlabel('Time (s)')

    # Set xticks to be in seconds
    tick_locs = np.linspace(0, filtered_dff_padded[:, 1:].shape[1], 10, endpoint=False).astype(int)
    ax.set_xticks(tick_locs)
    ax.set_xticklabels([f'{int(tick/session.calcium_fps):d}' for tick in tick_locs])

    # ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    filename = os.path.join(fig_dir, f'{session.animal_id}_{session.session_outcome}_collective_behaviour.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    # Plot annotations
    if all_behaviour_timestamps:
        fig, ax = plt.subplots(1, 1, figsize=(15, 5))
        unique_behaviors = list(all_behaviour_timestamps.keys())

        # Plot each behavior
        for i, behavior in enumerate(unique_behaviors):
            for start, end in all_behaviour_timestamps[behavior]:
                ax.fill_betweenx([i-0.5, i+0.5], start, end, color='black', alpha=0.5)
  
        # Set labels
        ax.set_yticks(np.arange(len(unique_behaviors)))
        ax.set_yticklabels(unique_behaviors)
        ax.set_ylabel('')
        # Set xticks to be in seconds
        tick_locs = np.linspace(0, filtered_dff_padded[:, 1:].shape[1], 10, endpoint=False).astype(int)
        ax.set_xticks(tick_locs)
        ax.set_xticklabels([f'{int(tick/session.calcium_fps):d}' for tick in tick_locs])
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # Save plot
        filename = os.path.join(fig_dir, f'{session.animal_id}_{session.session_outcome}_annotations.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        # Also save as .eps
        plt.savefig(os.path.join(fig_dir, f'{session.animal_id}_{session.session_outcome}_annotations.eps'), format='eps')
        plt.close()

def create_overview_plot(helper_session: Session, non_helper_session: Session, random_session: Session) -> None:
    """Creates an overview plot comparing a helper and non-helper session and random PNR/labels/traces."""
    
    # Load data for helper, non-helper, and random sessions
    helper_data = {
        'pnr_image': helper_session.load_pnr_image(),
        'labels': helper_session.get_neuron_labels(),
        'dff_padded': helper_session.filter_dff_signal(helper_session.load_dff_signal(), helper_session.load_accepted_components()),
        'annotations': helper_session.load_annotations()
    }
    
    non_helper_data = {
        'pnr_image': non_helper_session.load_pnr_image(),
        'labels': non_helper_session.get_neuron_labels(),
        'dff_padded': non_helper_session.filter_dff_signal(non_helper_session.load_dff_signal(), non_helper_session.load_accepted_components()),
        'annotations': non_helper_session.load_annotations()
    }
    
    random_data = {
        'pnr_image': random_session.load_pnr_image(),
        'labels': random_session.get_neuron_labels(),
        'dff_padded': random_session.filter_dff_signal(random_session.load_dff_signal(), random_session.load_accepted_components())
    }
    
    # Setup figure and axes
    fig, axes = setup_plot()

    # Random session plots (PNR, labels, and traces)
    axes['pnr_image'].imshow(random_data['pnr_image'][:, 100:-100], cmap='gray')
    axes['pnr_image'].set_xticks([])
    axes['pnr_image'].set_yticks([])
    # turn axes off completely
    axes['pnr_image'].axis('off')

    cmap = plt.cm.rainbow
    colors = cmap(np.linspace(0, 1, 80))
    colors[0] = (1, 1, 1, 1)  # Set first color to white
    np.random.shuffle(colors[1:])  # Shuffle the rest of the colors
    cmap = matplotlib.colors.ListedColormap(colors)
    axes['labels'].imshow(random_data['labels'][:, 100:-100], cmap=cmap)
    axes['labels'].set_xticks([])
    axes['labels'].set_yticks([])
    # turn axes off completely
    axes['labels'].axis('off')

    plot_random_traces(random_data['dff_padded'][:, 1:], num_traces=15, filename=None, ax=axes['traces'], fps=random_session.calcium_fps)
    
    # Helper session plots
    plot_heatmap(helper_data['dff_padded'][:, 1:], filename=None, ax=axes['heatmap_helper'], FPS=helper_session.calcium_fps, aligned=False, cbar=False, victim_crossing_line=False, title=f'Heatmap of {helper_session.animal_id} {helper_session.session_outcome}')
    axes['heatmap_helper'].set_xticks([])
    axes['collective_activity_helper'].plot(np.mean(helper_data['dff_padded'][:, 1:], axis=0), color='k')
    axes['collective_activity_helper'].set_xticks([])
    
    # Non-helper session plots
    plot_heatmap(non_helper_data['dff_padded'][:, 1:], filename=None, ax=axes['heatmap_non_helper'], FPS=non_helper_session.calcium_fps, aligned=False, cbar=False, victim_crossing_line=False, title=f'Heatmap of {non_helper_session.animal_id} {non_helper_session.session_outcome}')
    axes['heatmap_non_helper'].set_xticks([])
    axes['collective_activity_non_helper'].plot(np.mean(non_helper_data['dff_padded'][:, 1:], axis=0), color='k')
    axes['collective_activity_non_helper'].set_xticks([])
    # Plot annotations if available on the annotations_helper and annotations_non_helper axes
    if helper_data['annotations']:
        unique_behaviors = list(helper_data['annotations'].keys())

        # Plot each behavior
        for i, behavior in enumerate(unique_behaviors):
            color = matplotlib.cm.tab20(i/len(unique_behaviors))
            for j, (start, end) in enumerate(helper_data['annotations'][behavior]):
                axes['annotations_helper'].fill_betweenx([i-0.5, i+0.5], start, end, color=color, alpha=0.5, label=behavior if j == 0 else None)

        # Set xticks to be in seconds
        tick_locs = np.linspace(0, helper_data['dff_padded'][:, 1:].shape[1], 10, endpoint=False).astype(int)
        axes['annotations_helper'].set_xticks(tick_locs)
        axes['annotations_helper'].set_xticklabels([f'{int(tick/helper_session.calcium_fps)}s' for tick in tick_locs])
        # Remove y ticks
        axes['annotations_helper'].set_yticks([])
        # Add legend
        axes['annotations_helper'].legend(loc='upper right', title='Behavior')

    if non_helper_data['annotations']:
        unique_behaviors = list(non_helper_data['annotations'].keys())

        # Plot each behavior
        for i, behavior in enumerate(unique_behaviors):
            color = matplotlib.cm.tab20(i/len(unique_behaviors))
            for j, (start, end) in enumerate(non_helper_data['annotations'][behavior]):
                axes['annotations_non_helper'].fill_betweenx([i-0.5, i+0.5], start, end, color=color, alpha=0.5, label=behavior if j == 0 else None)
        # Set xticks and xticklabels
        tick_locs = np.linspace(0, non_helper_data['dff_padded'][:, 1:].shape[1], 10, endpoint=False).astype(int)
        axes['annotations_non_helper'].set_xticks(tick_locs)
        axes['annotations_non_helper'].set_xticklabels([f'{int(tick/non_helper_session.calcium_fps)}s' for tick in tick_locs])
        # Remove y ticks
        axes['annotations_non_helper'].set_yticks([])
        # Add legend
        axes['annotations_non_helper'].legend(loc='upper right', title='Behavior')
    # Save plot
    for ax in axes.values():
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # Save overview plot
    plt.savefig('overview_plot.png', dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    data = pd.read_csv('data/session_file_MiniscopeCW_all_Moises_latest_liberation_cells.csv')

    helper_sessions = data[data['Session Outcome'] == 'H']
    non_helper_sessions = data[data['Session Outcome'] == 'NH']

    # Randomly select one helper and one non-helper session
    random_helper_row = helper_sessions.sample().iloc[0]
    random_non_helper_row = non_helper_sessions.sample().iloc[0]
    random_row = data.sample().iloc[0]

    helper_session = Session(random_helper_row)
    non_helper_session = Session(random_non_helper_row)
    random_session = Session(random_row)

    # Plot and save the overview plot
    create_overview_plot(helper_session, non_helper_session, random_session)

    for index, row in data.iterrows():
        session = Session(row)
        plot_session(session)
        

