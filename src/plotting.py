import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy import stats


def plot_random_traces(DF_F_Z_scored, snr, output_path=None, ax=None):
    """
    Plot 10 random calcium traces from the detrended and z-scored data, colored by SNR.
    Handle potential infinite values in the data.
    
    :param DF_F_Z_scored: numpy array of detrended and z-scored calcium traces
    :param snr: signal-to-noise ratio values for each component
    :param output_path: path to save the plot, if None, the plot is not saved
    :param ax: matplotlib axis object, if None, a new plot is created
    """
    # Set bigger font size for better readability
    plt.rcParams.update({'font.size': 14})

    # # Remove any infinite values from SNR and corresponding traces
    # finite_mask = np.isfinite(snr)
    # DF_F_Z_scored = DF_F_Z_scored[finite_mask]
    # snr = snr[finite_mask]

    num_traces = min(10, DF_F_Z_scored.shape[0])
    if num_traces == 0:
        print("No finite traces available to plot.")
        return

    random_indices = np.random.choice(DF_F_Z_scored.shape[0], num_traces, replace=False)
    
    # Create a color map based on SNR values
    snr_min, snr_max = np.percentile(snr, [1, 99])  # Use percentiles to avoid extreme values
    norm = Normalize(vmin=snr_min, vmax=snr_max)
    cmap = plt.get_cmap('viridis')

    # Create a figure and axis if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(15, 10))

    for i, idx in enumerate(random_indices):
        trace = DF_F_Z_scored[idx - 1]
        # Remove any remaining infinite values in the trace
        trace[~np.isfinite(trace)] = np.nan
        color = cmap(norm(min(max(snr[idx], snr_min), snr_max)))
        ax.plot(trace + i*6, linewidth=1, color=color, label=f'SNR: {snr[idx]:.2f}')
    
    # Turn off spines on the top and right side of the plot
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_title(f'Random {num_traces} Detrended and Z-scored Calcium Traces')
    ax.set_xlabel('Time (frames)')
    ax.set_ylabel('Z-scored ΔF/F + offset')
    
    # Add colorbar
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', pad=0.2)
    cbar.set_label('SNR')
    
    # Add legend
    # ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    plt.tight_layout()

    # Save the figure only if output_path is provided and no axis is given
    if output_path is not None and ax is None:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()  # Close the figure to free up memory

def plot_histogram(values, threshold, filename, ax=None):
    """
    Plot a histogram of the sum of the components in the DF_F_Z_scored.

    Parameters:
        values (np.array): The values to plot the histogram of.
        threshold (float): The threshold to plot as a vertical line.
        filename (str): The path to the file to save the histogram.
        ax (matplotlib.axis): The axis to plot the histogram on.

    Returns:
        None or ax (matplotlib.axis): The axis with the histogram plotted on it.
    """
    
    if ax is None:
        save = True
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        save = False
        fig = ax.get_figure()

    # Plot the histogram of the sum of the components
    ax.hist(values, bins=20, alpha=0.8, label='Sum of the components', edgecolor='black', fill=False, histtype='step')

    # Plot the threshold
    ax.axvline(threshold, color='black', linestyle='dashed', linewidth=2, label='Threshold')

    # Also add the line to indicate the mean
    ax.axvline(np.mean(values), linewidth=2, label='Mean', color='black')

    # Add annotation of 1 sigma above the mean, make it bigger and locate it higher
    ax.annotate(
        f'$\\mu + \\sigma$', 
        xy=(np.mean(values) + np.std(values)*1.5, ax.get_ylim()[1] * 0.5), 
        xytext=(np.mean(values) + np.std(values)*1.5, ax.get_ylim()[1] * 0.6), 
        fontsize=12
    )

    # Add band specifying components selected 
    ax.axvspan(threshold, np.max(values), alpha=0.5, color='gray', label='Selected components')

    # Modify legend to be smaller and placed inside the plot (top right corner)
    ax.legend(loc='best', fontsize='small')

    # Remove y-axis label
    ax.set_ylabel('')

    # Add labels and title
    ax.set_xlabel('Sum of $\\Delta F/F$ values in the window')

    # Remove the top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if save:
        fig.savefig(filename)
        plt.close(fig)

    return ax

    
def plot_heatmap(DF_F_Z_scored, filename, ax, window_to_use, FPS, animal, day_of_helping):
    '''
    Function to plot the heatmap of the DF_F_Z_scored of selected components matrix.

    Parameters:
        DF_F_Z_scored (np.array): The matrix of the selected components.
        ax (matplotlib.axis): The axis to plot the heatmap on.
        window_to_use (int): The window to use for the heatmap.
        FPS (int): The frames per second of the calcium recording.
        animal (str): The animal ID.
        day_of_helping (str): The day of helping.
    
    Returns:
        ax (matplotlib.axis): The axis with the heatmap plotted on it.
    '''

    if ax is None:
        save = True
        fig, ax = plt.subplots(figsize=(10, 8)) # Create a figure and axis if not provided, size 10x8 (width x height)
    else:
        save = False
        fig = ax.get_figure()

    # Plot the heatmap of the DF_F_Z_scored and add a colorbar
    # print('DF_F_Z_scored:', DF_F_Z_scored.shape)
    im = ax.imshow(DF_F_Z_scored, aspect='auto', cmap='viridis', vmin=-5, vmax=5)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('$\\Delta F/F (Z-scored)$', rotation=270, labelpad=15)

    # Span the center of the plot for the victim crossing time
    ax.axvline(window_to_use * FPS, color='black', linestyle='dashed', linewidth=2, label='Victim crossing')

    # Add ticks labels for the x-axis from -30 sec to 30 sec
    ax.set_xticks([0, window_to_use * FPS, 2 * window_to_use * FPS])
    ax.set_xticklabels([-window_to_use, 0, window_to_use])

    # Remove ticks on y-axis
    ax.set_yticks([])
    ax.set_yticklabels([])

    # Add labels and title
    ax.set_title(f'Heatmap of selected components, animal {animal}, day of helping {day_of_helping}')
    ax.set_xlabel('Time (s) centered around victim crossing')  # Add x-label

    # Turn off the top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if save:
        fig.savefig(filename)
        plt.close(fig)

    return ax


import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def plot_boxplots(data1, data2, ax, title='', labels=('Group 1', 'Group 2'), ylabel='Y-axis Label'):
    '''
    Function to plot boxplots for two sets of data.
    Performs statistical tests if conditions are met and annotates the plot accordingly.

    Parameters:
        data1 (np.array): Data for the first group.
        data2 (np.array): Data for the second group.
        ax (matplotlib.axes.Axes): The axis to plot on.
        title (str): Title of the plot.
        labels (tuple): Labels for the boxplot groups.
        ylabel (str): Label for the Y-axis.

    Returns:
        ax (matplotlib.axes.Axes): The axis with the boxplots plotted on it.    
    '''

    # Plot the boxplots without connecting lines
    bp = ax.boxplot([data1, data2], labels=labels, showmeans=False)
    
    # Customize boxplot appearance
    for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp[element], color='black')
    
    # Add scatter plot of the data points (small and black)
    for i, data in enumerate([data1, data2], start=1):
        ax.scatter(np.random.normal(i, 0.04, size=len(data)), data, 
                   alpha=0.7, s=3, color='black', zorder=3)
    
    # Add labels and title
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # Test for normality and equal variance
    _, normality_p1 = stats.shapiro(data1)
    _, normality_p2 = stats.shapiro(data2)
    _, equal_var_p = stats.levene(data1, data2)

    # Determine if conditions for parametric test are met
    normality_met = normality_p1 > 0.05 and normality_p2 > 0.05
    equal_var_met = equal_var_p > 0.05

    # Perform statistical test
    if normality_met and equal_var_met:
        _, p_value = stats.ttest_ind(data1, data2)
    else:
        _, p_value = stats.mannwhitneyu(data1, data2)

    # Add significance annotation if p-value is significant
    if p_value < 0.05:
        y_max = max(np.max(data1), np.max(data2))
        y_min = min(np.min(data1), np.min(data2))
        y_range = y_max - y_min
        line_height = y_max + 0.1 * y_range
        
        if p_value < 0.001:
            stars = '***'
        elif p_value < 0.01:
            stars = '**'
        else:
            stars = '*'
        
        ax.plot([1, 1, 2, 2], [line_height, line_height*1.05, line_height*1.05, line_height], lw=1.5, c='black')
        ax.text(1.5, line_height*1.05, stars, ha='center', va='bottom', color='black')

    # Turn off the top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    return ax

