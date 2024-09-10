import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy import stats
import seaborn as sns

def plot_matrix(matrix, ax=None, title='', xlabel='', ylabel='', cbar_label='', cmap='viridis', vmin=None, vmax=None, aspect='auto', xticks=None, xticklabels=None, yticks=None, yticklabels=None, save_path=None, cbar=True):
    """
    General function to plot a matrix as a heatmap.

    Parameters
    ----------
    matrix : numpy array
        The matrix to be plotted.
    ax : matplotlib axis, optional
        The axis to plot on. If None, a new figure and axis are created.
    title : str
        The title of the plot.
    xlabel : str
        The label for the x-axis.
    ylabel : str
        The label for the y-axis.
    cbar_label : str
        The label for the colorbar.
    cmap : str
        The colormap to use for the heatmap.
    vmin, vmax : float, optional
        Minimum and maximum values for the heatmap color scaling.
    aspect : str
        Aspect ratio of the plot.
    xticks : list, optional
        X-axis tick locations.
    xticklabels : list, optional
        Labels for the x-axis ticks.
    yticks : list, optional
        Y-axis tick locations.
    yticklabels : list, optional
        Labels for the y-axis ticks.
    save_path : str, optional
        If provided, saves the figure to the specified path.
    cbar : bool, optional
        If True, plots a colorbar.
    
    Returns
    -------
    ax : matplotlib axis
        The axis with the heatmap plotted on it.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        fig = ax.get_figure()

    # Plot the heatmap
    im = ax.imshow(matrix, aspect=aspect, cmap=cmap, vmin=vmin, vmax=vmax)
    
    # Add colorbar
    if cbar:
        cbar = plt.colorbar(im, ax=ax)
        if cbar_label:
            cbar.set_label(cbar_label, rotation=270, labelpad=15)

    # Set title and labels
    ax.set_title(title, fontsize=16)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)

    # Set x and y ticks
    if xticks is not None:
        ax.set_xticks(xticks)
    if xticklabels is not None:
        ax.set_xticklabels(xticklabels)
    if yticks is not None:
        ax.set_yticks(yticks)
    if yticklabels is not None:
        ax.set_yticklabels(yticklabels)

    # Turn off the top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if save_path:
        fig.savefig(save_path)
        if save_path.endswith('.png'):
            eps_path = save_path.replace('.png', '.eps')
            fig.savefig(eps_path, format='eps')
        plt.close(fig)
    
    return ax

def plot_correlated_traces(ax, traces, component_pairs, distances=None, annotations=None, color_palette='husl', n_to_plot=8, vertical_shift=10, line_alpha=0.7, text_offset=5):
    """
    Plot paired traces with optional annotations and distances.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis to plot on.
    traces : numpy.ndarray
        2D array where each row is a trace to be plotted; 1st row are real indecies in the original array.
    component_pairs : tuple
        Tuple of two lists/arrays, each containing indices of components to be paired for plotting.
    distances : numpy.ndarray, optional
        2D distance matrix between components. Defaults to None.
    annotations : dict, optional
        Dictionary with annotation names as keys and list of tuples (start, end) as values. Defaults to None.
    color_palette : str, optional
        Name of the Seaborn color palette to use for annotations. Defaults to 'husl'.
    n_to_plot : int, optional
        Number of component pairs to plot. Defaults to 8.
    vertical_shift : int, optional
        Vertical shift to apply between successive traces. Defaults to 10.
    line_alpha : float, optional
        Alpha transparency for the trace lines. Defaults to 0.7.

    Returns
    -------
    None
    """
    n_to_plot = min(n_to_plot, len(component_pairs))

    # Generate a color palette based on the number of annotations
    colors = None
    if annotations is not None:
        colors = sns.color_palette(color_palette, len(annotations))

    # Plot annotations (if provided)
    if annotations is not None and colors is not None:
        for j, (annotation, timestamps) in enumerate(annotations.items()):
            for start, end in timestamps:
                ax.axvspan(start, end, label=annotation, color=colors[j], alpha=0.3)

    # Plot the paired traces
    for i, (idx1, idx2) in enumerate(component_pairs[:n_to_plot]):
        ax.plot(traces[idx1, 1:] + vertical_shift * i, color='k', alpha=line_alpha)
        ax.plot(traces[idx2, 1:] + vertical_shift * i, color='tab:blue', alpha=line_alpha)

        # Add distance if provided
        if distances is not None:
            distance = distances[idx1, idx2]
            ax.text(10, vertical_shift * i + text_offset, f'Distance between cell {int(idx1)} and cell {int(idx2)}: {distance:.2f}', fontsize=12)

def plot_random_traces(DF_F_Z_scored, num_traces=10, fps=30, snr=None, ax=None, filename=None):
    """
    Plot random calcium traces from the detrended and z-scored data.
    Optionally color traces by SNR.
    
    :param DF_F_Z_scored: numpy array of detrended and z-scored calcium traces
    :param num_traces: number of traces to plot, defaults to 10
    :param fps: frames per second, defaults to 20
    :param snr: optional, signal-to-noise ratio values for each component
    :param ax: optional, matplotlib axis object; if None, a new plot is created
    :param filename: optional, path to save the image to; if given, the image is saved in png and eps formats
    :return: matplotlib axis with the plotted traces
    """
    # Set bigger font size for better readability
    plt.rcParams.update({'font.size': 14})

    # Limit the number of traces to plot
    num_traces = min(num_traces, DF_F_Z_scored.shape[0])
    if num_traces == 0:
        print("No finite traces available to plot.")
        return ax

    # Select random indices to plot
    random_indices = np.random.choice(DF_F_Z_scored.shape[0], num_traces, replace=False)
    
    # Handle SNR-based coloring if SNR is provided
    if snr is not None:
        snr_min, snr_max = np.percentile(snr, [1, 99])  # Use percentiles to avoid extreme values
        norm = Normalize(vmin=snr_min, vmax=snr_max)
        cmap = plt.get_cmap('viridis')

    # Create a figure and axis if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(15, 10))

    for i, idx in enumerate(random_indices):
        trace = DF_F_Z_scored[idx]
        trace[~np.isfinite(trace)] = np.nan  # Remove any infinite values
        color = cmap(norm(min(max(snr[idx], snr_min), snr_max))) if snr is not None else 'k'
        ax.plot(trace + i*6, linewidth=1, color=color)

    # Turn off spines on the top and right side of the plot
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_title(f'Random {num_traces} Detrended and Z-scored Calcium Traces')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Z-scored ΔF/F + offset')
    
    # Add colorbar if SNR is provided
    if snr is not None:
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', pad=0.2)
        cbar.set_label('SNR')

    # Set xticks to be in seconds
    tick_locs = np.linspace(0, DF_F_Z_scored.shape[1], 10, endpoint=False).astype(int)
    ax.set_xticks(tick_locs)
    ax.set_xticklabels([f'{int(tick/fps):d}' for tick in tick_locs])

    # Remove y ticks
    ax.set_yticks([])

    plt.tight_layout()

    if filename is not None:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.savefig(str(filename).replace('.png', '.eps'), dpi=300, bbox_inches='tight')
    
    return ax


def plot_histogram(values, threshold, filename, ax=None, suffix=None):
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
        fig.savefig(f"{filename}{suffix}.png", dpi=300)
        filename = filename.replace('.png', '.eps')
        fig.savefig(f"{filename}{suffix}.eps", dpi=300)
        plt.close(fig)

    return ax

def plot_heatmap(matrix, title=None, xlabel='Time (s)', aligned=True, filename=None, ax=None, window_to_use=None, FPS=None, animal=None, day_of_helping=None, cbar=True, victim_crossing_line=True):
    """
    Plot the heatmap of a matrix with custom settings.

    Parameters
    ----------
    matrix : np.ndarray
        The matrix to be plotted.
    title : str, optional
        The title of the plot. If None, use a default title.
    xlabel : str, optional
        The label for the x-axis. If None, use a default label.
    aligned : bool, optional
        If True, center the time axis around 0. If False, don't do it.
    filename : str, optional
        The filename to save the plot to. If None, don't save the plot.
    ax : matplotlib.axes.Axes, optional
        The axis to plot on. If None, create a new figure and axis.
    window_to_use : int, optional
        The window to use for the x-axis. If None, use half of the matrix size.
    FPS : int, optional
        The frames per second. If None, use 30.
    animal : int, optional
        The animal ID. If None, don't add it to the title.
    day_of_helping : int, optional
        The day of helping. If None, don't add it to the title.
    cbar : bool, optional
        If True, include the colorbar. If False, don't include it.
    victim_crossing_line : bool, optional
        If True, add a vertical line for victim crossing. If False, don't add it.
    """
    if ax is None:
        fig, ax = plt.subplots()

    if window_to_use is None:
        window_to_use = int(np.floor(matrix.shape[1] / 2 / FPS))

    if title is None:
        title = f'Heatmap of selected components, animal {animal}, day of helping {day_of_helping}'

    if aligned:
        xticks = [0, window_to_use * FPS, 2 * window_to_use * FPS]
        xticklabels = [-window_to_use, 0, window_to_use]
    else:
        xticks = np.linspace(0, matrix.shape[1], 4)
        xticklabels = [int(tick / FPS) for tick in xticks]

    ax = plot_matrix(
        matrix=matrix,
        ax=ax,
        title=title,
        xlabel=xlabel,
        cbar_label='$\\Delta F/F (Z-scored)$' if cbar else None,
        vmin=-5,
        vmax=5,
        xticks=xticks,
        xticklabels=xticklabels,
        save_path=filename,
        cbar=cbar
    )

    if victim_crossing_line:
        ax.axvline(window_to_use * FPS, color='black', linestyle='dashed', linewidth=2, label='Victim crossing')
    return ax


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

def plot_cross_correlation_matrix(axes, correlation_matrix, title='Cross-correlation matrix', save_path=None):
    """
    Plot a sorted cross-correlation matrix of the dF/F signals of the components.
    """
    # Sort the correlation matrix
    sorted_correlation_matrix = correlation_matrix[:, np.argsort(-correlation_matrix[0])]
    sorted_correlation_matrix = sorted_correlation_matrix[np.argsort(-sorted_correlation_matrix[:, 0]), :]

    ax = plot_matrix(
        matrix=sorted_correlation_matrix,
        ax=axes,
        title=title,
        cbar_label='Correlation value',
        aspect='equal', save_path=save_path, vmin=-1, vmax=1
    )
    return ax

def plot_distance_matrix(axes, distances, title='Distance matrix', save_path=None):
    """
    Plot a distance matrix using the generalized plot_matrix function.

    Parameters
    ----------
    axes : matplotlib axes object
        The axes to plot on.
    distances : numpy array
        The distance matrix to plot.
    title : str, optional
        The title of the plot. Default is 'Distance matrix'.
    """
    plot_matrix(
        matrix=distances,
        ax=axes,
        title=title,
        aspect='equal',
        cbar_label='Distance', save_path=save_path
    )
