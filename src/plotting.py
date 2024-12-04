import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy import stats
import seaborn as sns
import os
from scipy.optimize import curve_fit
from scipy.stats import ks_2samp
from scipy.stats import false_discovery_control
import pandas as pd

def plot_registration_validation(data, cohort, output_path):
    """
    Plot combined percentage of registered components vs. the number of sessions for all animals in a cohort.

    Parameters:
    ----------
    data : DataFrame
        Data containing 'Percentage', 'K', and 'Animal' columns.
    cohort : str
        Cohort identifier for the title and labels.
    output_path : str
        Path to save the generated plot.
    """
    plt.figure(figsize=(10, 8))

    # Barplot for averages
    ax = sns.barplot(
        data=data, x='K', y='Percentage', 
        ci=None, estimator=np.mean, color='k', alpha=0.6
    )
    
    # Swarmplot for individual points
    sns.swarmplot(
        data=data, x='K', y='Percentage', hue='Animal', 
        dodge=True, palette='Set2', s=4, alpha=0.7
    )

    plt.title(f'% of Registered Components vs. Number of Sessions\nCohort {cohort}')
    plt.ylabel('Percentage (%)')
    plt.xlabel('Number of Sessions (K)')
    plt.legend(title='Animal', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Remove unnecessary spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    # Save and close the plot
    plt.savefig(output_path)
    plt.close()

def plot_ecdf_with_ks_test(first_dist_values, second_dist_values, optional_third_dist_values=None, ax=None, 
                           significance_color='r', second_dist_color='grey', third_dist_color='b',
                           p_value_threshold=0.05, filename=None, legend=False, 
                           true_label='True', shuffled_label='Shuffled', third_label='Random Event Cells'):
    """
    Plot ECDF for true and shuffled data, perform the Kolmogorov-Smirnov (KS) test, and 
    display the results with a significance star if the p-value is below the threshold. 
    Optionally save the plot to a file.

    Parameters
    ----------
    first_dist_values : list of float
        Array of values for the first distribution.
    second_dist_values : list of float
        Array of values for the second distribution.
    optional_third_dist_values : list of float, optional
        Array of values for the third distribution.
    ax : matplotlib axis object, optional
        The axis to plot on. If None, create a new figure and axis.
    significance_color : str, optional
        Color of the significance star (default is 'r' - red).
    second_dist_color : str, optional
        Color for the ECDF plot of the shuffled data (default is 'grey').
    third_dist_color : str, optional
        Color for the ECDF plot of the third data (default is 'b' - blue).
    p_value_threshold : float, optional
        Threshold for statistical significance (default is 0.05).
    filename : str, optional
        If provided, the plot will be saved as both PNG and EPS with the given filename.
    legend : bool, optional
        If True, add a legend to the plot.
    true_label : str, optional
        Label for the true distribution.
    shuffled_label : str, optional
        Label for the shuffled distribution.
    third_label : str, optional
        Label for the third distribution.
    """
    
    # Remove None from the arrays
    first_dist_values = [x for x in first_dist_values if x is not None]
    second_dist_values = [x for x in second_dist_values if x is not None]
    if optional_third_dist_values is not None:
        optional_third_dist_values = [x for x in optional_third_dist_values if x is not None]

    # Convert inputs to lists of floats
    first_dist_values = [float(x) for x in first_dist_values]
    second_dist_values = [float(x) for x in second_dist_values]
    if optional_third_dist_values is not None:
        optional_third_dist_values = [float(x) for x in optional_third_dist_values]
    
    # Create a new figure and axis if one is not provided
    if ax is None:
        __, ax = plt.subplots(figsize=(8, 6))
    
    # Select equal number of samples for all distributions, but randomize which samples to select
    if optional_third_dist_values is not None:
        n_samples = min(len(first_dist_values), len(second_dist_values), len(optional_third_dist_values))
        first_sample_idx = np.random.RandomState(0).permutation(len(first_dist_values))[:n_samples]
        second_sample_idx = np.random.RandomState(0).permutation(len(second_dist_values))[:n_samples]
        third_sample_idx = np.random.RandomState(0).permutation(len(optional_third_dist_values))[:n_samples]
        first_dist_values = [first_dist_values[i] for i in first_sample_idx]
        second_dist_values = [second_dist_values[i] for i in second_sample_idx]
        optional_third_dist_values = [optional_third_dist_values[i] for i in third_sample_idx]
    else:
        n_samples = min(len(first_dist_values), len(second_dist_values))
        first_sample_idx = np.random.RandomState(0).permutation(len(first_dist_values))[:n_samples]
        second_sample_idx = np.random.RandomState(0).permutation(len(second_dist_values))[:n_samples]
        first_dist_values = [first_dist_values[i] for i in first_sample_idx]
        second_dist_values = [second_dist_values[i] for i in second_sample_idx]
    
    # Plot ECDF for True and Shuffled values
    sns.ecdfplot(first_dist_values, ax=ax, color='k', alpha=0.9, label=true_label)
    sns.ecdfplot(second_dist_values, ax=ax, color=second_dist_color, alpha=0.9, label=shuffled_label)

    # Plot ECDF for third values if provided
    if optional_third_dist_values is not None:
        sns.ecdfplot(optional_third_dist_values, ax=ax, color=third_dist_color, alpha=0.9, label=third_label)

    # Add legend if requested
    if legend:
        ax.legend()

    # Set axis limits and aesthetics
    ax.set_xlim(-0.5, 0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Perform the Kolmogorov-Smirnov test with Bonferroni correction
    if optional_third_dist_values is None:
        ks_result = ks_2samp(second_dist_values, first_dist_values)
        p_value_corrected = ks_result[1]
        # Compute the effect size (D statistic)
        D_statistic = np.abs(np.mean(first_dist_values) - np.mean(second_dist_values)) / np.sqrt(np.var(first_dist_values) + np.var(second_dist_values) + 1e-10)
        # Add p-value and effect size to the plot
        ax.text(0.65, 0.85, 'p-value: {:.4f}\nD-statistic: {:.4f}'.format(p_value_corrected, D_statistic), transform=ax.transAxes, ha='center', va='center')
        # Save the results to a file
        with open(filename.replace('.png', '.txt'), 'w') as f:
            f.write('KS statistic: {:.4f}\np-value: {:.4f}\np-value corrected: {:.4f}\nD-statistic: {:.4f}\n'.format(
                ks_result[0], ks_result[1], p_value_corrected, D_statistic))
    else:
        # Perform three tests and Bonferroni correct the p-values
        comparisons = [(second_dist_values, first_dist_values), (second_dist_values, optional_third_dist_values), (first_dist_values, optional_third_dist_values)]
        labels = [(shuffled_label, true_label), (shuffled_label, third_label), (true_label, third_label)]
        test_results = [ks_2samp(*comparison) for comparison in comparisons]
        p_values = [test_result[1] for test_result in test_results]

        # Correct the p-values using the Bonferroni correction
        p_values_corrected = false_discovery_control(p_values, method='bh')
        
        # Compute the effect size (D statistic) for each comparison
        D_statistics = []
        for comparison in comparisons:
            D_statistic = np.abs(np.mean(comparison[0]) - np.mean(comparison[1])) / np.sqrt(np.var(comparison[0]) + np.var(comparison[1]) + 1e-10)
            D_statistics.append(D_statistic)

        # Save the results to a file
        with open(filename.replace('.png', '.txt'), 'w') as f:
            for i, test_result in enumerate(test_results):
                f.write('Test {}: comparing {} and {} distributions\nKS statistic: {:.4f}\np-value: {:.4f}\np-value corrected: {:.4f}\nD-statistic: {:.4f}\n'.format(
                    i+1, labels[i][0], labels[i][1], test_result[0], test_result[1], p_values_corrected[i], D_statistics[i]))
                
    # If a filename is provided, save the plot
    if filename is not None:
        # Create the directory if it doesn't exist
        directory = os.path.dirname(filename)
        if not os.path.exists(directory):
            os.makedirs(directory)
        # Save the plot to both PNG and EPS formats
        plt.savefig(filename + '.png', dpi=300)
        plt.savefig(filename.replace('.png', '.eps'))
        plt.close(plt.gcf())  # Close the figure after saving
    else:
        return ax

def exp_func(x, a, b):
            return a * np.exp(b * x)

def plot_scatter(
    x, 
    y, 
    ax=None, 
    title='', 
    xlabel='', 
    ylabel='', 
    alpha=0.5, 
    s=1, 
    c='k',  # Color for scatter points
    significance_color='r',  # Color for significance star only
    xlim=None, 
    ylim=None, 
    fit_type=None,   # 'linear' or 'exponential', or None for no fit
    perform_test=False, # If True, perform statistical test (Pearson correlation)
    p_value_threshold=0.05,  # Threshold for significance
    filename=None # If present, save to both png and eps
):
    """
    Generalized function to plot a scatter plot (e.g., correlation vs distance),
    with optional linear or exponential fit and statistical test.

    Parameters
    ----------
    x : array-like
        X values for the scatter plot.
    y : array-like
        Y values for the scatter plot.
    ax : matplotlib axis object, optional
        The axis to plot on. If None, create a new figure and axis.
    title : str, optional
        The title of the plot. Default is an empty string.
    xlabel : str, optional
        The label for the x-axis. Default is an empty string.
    ylabel : str, optional
        The label for the y-axis. Default is an empty string.
    alpha : float, optional
        The transparency level of the points. Default is 0.5.
    s : float, optional
        The size of the points. Default is 1.
    c : color, optional
        The color of the points. Default is 'k' (black).
    significance_color : color, optional
        The color of the significance star only. Default is 'r' (red).
    xlim : tuple, optional
        Limits for the x-axis. Default is None.
    ylim : tuple, optional
        Limits for the y-axis. Default is None.
    fit_type : str, optional
        Type of fit to apply: 'linear', 'exponential', or None for no fit.
    perform_test : bool, optional
        If True, perform a statistical test (Pearson correlation) and display results.
    p_value_threshold : float, optional
        Threshold for statistical significance (default is 0.05).
    filename : str, optional
        If present, save to both png and eps with the given filename.

    Returns
    -------
    ax : matplotlib axis
        The axis with the scatter plot and optional fit.
    """
    
    # Convert x and y to 1D numpy arrays if they are not
    x = np.array(x).flatten()
    y = np.array(y).flatten()

    if ax is None:
        # Create a small figure if ax is not provided
        fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot the scatter plot
    ax.scatter(x, y, alpha=alpha, s=s, c=c)

    # Set title and labels
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Set axis limits if provided
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    # Set y ticks to be -1, -0.5, 0, 0.5, 1
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])

    # Turn off the top and right spines for clean aesthetics
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # If a fit type is specified, perform and plot the fit
    if fit_type == 'linear':
        # Perform linear fit
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        fit_line = slope * np.array(x) + intercept
        ax.plot(x, fit_line, color='red', label=f'Linear fit (r={r_value:.2f}, p={p_value:.2e})')
        ax.legend()

    if fit_type == 'exponential':
        # TODO: fix the fitting. it doesn't work.
        # Perform exponential fit
        x_nonan = x[np.isfinite(x) & np.isfinite(y)]
        y_nonan = y[np.isfinite(x) & np.isfinite(y)]
        popt, pcov = curve_fit(exp_func, x_nonan, y_nonan)
        fit_line = exp_func(x_nonan, *popt)
        ax.plot(x_nonan, fit_line, color='red', label=f'Exponential fit (a={popt[0]:.2f}, b={popt[1]:.2f})')
        ax.legend()

    # Perform Pearson correlation as a statistical test if required
    if perform_test:
        x_nonan = x[np.isfinite(x) & np.isfinite(y)]
        y_nonan = y[np.isfinite(x) & np.isfinite(y)]
        corr_coef, p_value = stats.pearsonr(x_nonan, y_nonan)

        # Prepare the results string, adding a star if p-value is significant
        significance_star = ''
        if float(p_value) < p_value_threshold:
            significance_star = ' *'  # Add a star for significant p-values

        # Display the test results
        ax.text(
            0.05, 0.9, 
            f"r={corr_coef:.2f}, p={p_value:.2e}", 
            transform=ax.transAxes, 
            color='black'  # Keep main text black
        )

        # Display the star next to the text, if applicable
        if significance_star:
            ax.text(
                0.45, 0.9,  # Positioning the star to appear next to the p-value text
                significance_star, 
                transform=ax.transAxes, 
                color=significance_color,  # Use the specified significance color for the star
                fontsize=14  # You can adjust this size to better match the plot
            )

    if filename is not None:
        # Create the directory if it doesn't exist
        directory = os.path.dirname(filename)
        if not os.path.exists(directory):
            os.makedirs(directory)
        # Save the plot to both png and eps
        plt.savefig(filename + '.png', dpi=300)
        plt.savefig(filename + '.eps')
        plt.close(plt.gcf()) # make sure to close the figure afterwards
    else:
        return ax

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
    im = ax.imshow(matrix, aspect=aspect, cmap=cmap, interpolation=None, rasterized=True, vmin=vmin, vmax=vmax)
    
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
        # Create directory if it doesn't exist yet
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        fig.savefig(save_path, bbox_inches='tight')
        if save_path.endswith('.png'):
            eps_path = save_path.replace('.png', '.eps')
            fig.savefig(eps_path, format='eps', bbox_inches='tight')
        plt.close(fig)
    
    return ax

def plot_cross_correlation_matrix(axes, correlation_matrix, title='Cross-correlation matrix', cbar_label='Correlation value', aspect='equal', save_path=None, vmin=-1, vmax=1, cmap='bwr', cbar=True, xticks=None, yticks=None):
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
        cbar_label=cbar_label,
        aspect=aspect,
        save_path=save_path,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        cbar=cbar,
        xticks=xticks,
        yticks=yticks)
    
    return ax

def plot_distance_matrix(axes, distances, title='Distance matrix', aspect='equal', cbar_label='Distance', save_path=None, vmin=None, vmax=None, cmap='viridis', cbar=True, xticks=None, yticks=None):
    """
    Plot a distance matrix using the generalized plot_matrix function.

    """
    ax = plot_matrix(
        matrix=distances,
        ax=axes,
        title=title,
        aspect=aspect,
        cbar_label=cbar_label,
        save_path=save_path,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        cbar=cbar,
        xticks=xticks,
        yticks=yticks
    )

    return ax

def plot_correlated_traces(ax=None, traces=None, component_pairs=None, distances=None, correlation_matrix=None, annotations=None, victim_lib_time=None, liberation_cell_ids=None, title=None, color_palette='husl', n_to_plot=6, vertical_shift=10, line_alpha=0.9, line_width=0.5, text_offset=5, filename=None, fps=30):
    """
    Plot paired traces with optional annotations and distances.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis to plot on. If None, a new figure and axis are created.
    traces : numpy.ndarray
        2D array where each row is a trace to be plotted; 1st row are real indecies in the original array.
    component_pairs : tuple
        Tuple of two lists/arrays, each containing indices of components to be paired for plotting.
    distances : numpy.ndarray, optional
        2D distance matrix between components. Defaults to None.
    correlation_matrix : numpy.ndarray, optional
        2D correlation matrix between components. Defaults to None.
    annotations : dict, optional
        Dictionary with annotation names as keys and list of tuples (start, end) as values. Defaults to None.
    victim_lib_time : float, optional
        Timestamp of the victim liberation. If provided, will be plotted as a dashed vertical line. Defaults to None.
    liberation_cell_ids : list, optional
        List of ids of liberation cells. If a cell id is same as one of these, change its color to tab:red. Defaults to None.
    title : str, optional
        Title of the plot. Defaults to None.
    color_palette : str, optional
        Name of the Seaborn color palette to use for annotations. Defaults to 'husl'.
    n_to_plot : int, optional
        Number of component pairs to plot. Defaults to 8.
    vertical_shift : int, optional
        Vertical shift to apply between successive traces. Defaults to 10.
    line_alpha : float, optional
        Alpha transparency for the trace lines. Defaults to 0.7.
    line_width : float, optional
        Line width for the trace lines. Defaults to 0.5.
    text_offset : int, optional
        Offset for text labels. Defaults to 5.
    filename : str, optional
        If provided, save the figure in png and eps formats and close the figure. Defaults to None.
    fps : int, optional
        Frames per second. Defaults to 30.

    Returns
    -------
    None
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(20, 10))

    n_to_plot = min(n_to_plot, len(component_pairs))

    # Generate a color palette based on the number of annotations
    colors = None
    if annotations is not None:
        colors = sns.color_palette(color_palette, len(annotations))

    # Plot annotations (if provided)
    if annotations is not None and colors is not None:
        for j, (annotation, timestamps) in enumerate(annotations.items()):
            for start, end in timestamps:
                ax.axvspan(start / fps, end / fps, label=annotation, color=colors[j], alpha=0.3)

    # Plot victim liberation timestamp (if provided)
    if victim_lib_time is not None:
        ax.axvline(victim_lib_time, color='k', linestyle='dashed', linewidth=2)

    # Plot the paired traces sorted by correlation value
    correlation_values = [correlation_matrix[idx1, idx2] for idx1, idx2 in component_pairs[:n_to_plot]]
    sorted_indices = np.argsort(correlation_values)[::-1]
    for i, idx in enumerate(sorted_indices):
        idx1, idx2 = component_pairs[idx]
        color1 = 'k' if liberation_cell_ids is None or idx1 not in liberation_cell_ids else 'tab:red'
        color2 = '#2AC5F4' if liberation_cell_ids is None or idx2 not in liberation_cell_ids else 'tab:red'
        ax.plot(traces[idx1, 1:] + vertical_shift * i, color=color1, alpha=line_alpha, linewidth=line_width)
        ax.plot(traces[idx2, 1:] + vertical_shift * i, color=color2, alpha=line_alpha, linewidth=line_width)

        # # Add distance and/or correlation if provided
        # if distances is not None and correlation_matrix is not None:
        #     distance = distances[idx1, idx2]
        #     correlation = correlation_matrix[idx1, idx2]
        #     ax.text(10, vertical_shift * i + text_offset, f'Distance between cell {int(idx1)} and cell {int(idx2)}: {distance:.2f}, Correlation: {correlation:.2f}', fontsize=12)
    
    if len(correlation_values) > 0:
        # Add an arrow showing an increase of PCC
        arrow_head_width = 10
        arrow_head_length = 5
        arrow_start_x = -10
        arrow_start_y = 0
        arrow_dx = 0 - 5
        arrow_dy = vertical_shift * n_to_plot

        # Fix the arrow
        # ax.arrow(arrow_start_x, arrow_start_y, arrow_dx, arrow_dy, head_width=arrow_head_width, head_length=arrow_head_length, fc='k', ec='k', lw=1)

        # Add text labels
        ax.text(arrow_start_x - 15, arrow_start_y, f'{correlation_values[0]:.2f}', ha='right', va='center', fontsize=12)
        ax.text(arrow_start_x - 25, arrow_start_y + arrow_dy/2, 'Increasing PCC', ha='center', va='center', fontsize=14, rotation=90)
        ax.text(arrow_start_x - 15, arrow_start_y + arrow_dy, f'{correlation_values[-1]:.2f}', ha='right', va='center', fontsize=12)

    # Set xticks to be in seconds
    tick_locs = np.linspace(-20, traces.shape[1], 10, endpoint=False).astype(int)
    ax.set_xticks(tick_locs)
    ax.set_xticklabels([f'{int(tick/fps):d}' for tick in tick_locs])

    ax.set_yticks([])
    ax.set_xlabel('Time (s)')

    # Turn off the top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    if title is not None:
        ax.set_title(title)

    if filename is not None:
        dir_name = os.path.dirname(filename)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)

        plt.savefig(f'{filename}', dpi=300, bbox_inches='tight')
        plt.savefig(f'{filename}'.replace('.png', '.eps'), format='eps', bbox_inches='tight')
        plt.close()

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
        cbar = plt.colorbar(sm, ax=ax)
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


def plot_histogram(values, threshold, filename, ax=None, suffix=None, legend=True):
    """
    Plot a histogram of the sum of the components in the DF_F_Z_scored.

    Parameters:
        values (np.array): The values to plot the histogram of.
        threshold (float): The threshold to plot as a vertical line.
        filename (str): The path to the file to save the histogram.
        ax (matplotlib.axis): The axis to plot the histogram on.
        legend (bool): Whether to show the legend.

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

    if legend:
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
        xticks = [0, int(window_to_use * FPS / 2), int(window_to_use * FPS)]
        xticklabels = [-int(window_to_use / 2), 0, int(window_to_use / 2)]
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
        ax.axvline(int((window_to_use * FPS) / 2), color='black', linestyle='dashed', linewidth=2, label='Victim crossing')
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

def plot_values_over_days(data, x_column, y_column, ax=None, order=None, title=None, hue_column=None, figsize=(10, 6), filename=None):
    """
    Plot values as a bar plot over days with optional swarm plot overlay and save the figure.
    
    Parameters:
    - data: pandas DataFrame containing the data
    - x_column: str, name of the column to use for x-axis (e.g., 'Session', 'DayOfHelping')
    - y_column: str, name of the column to plot on y-axis
    - ax: matplotlib axis object to plot on (optional)
    - order: list, optional, specifying the order of x-axis categories
    - title: str, optional, title for the plot
    - hue_column: str, optional, name of the column to use for color-coding points in the swarm plot
    - figsize: tuple, optional, figure size if creating a new figure
    - filename: str, optional, if provided, the figure will be saved with this name (without extension)
    
    Returns:
    - fig: matplotlib figure object
    - ax: matplotlib axis object
    """
    # Create a new figure if ax is not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Bar plot
    sns.barplot(x=x_column, y=y_column, data=data, ax=ax, fill=False, 
                capsize=0.1, err_kws={'linewidth': 1.5}, color='black', order=order)
    
    # Swarm plot (if hue_column is provided)
    if hue_column:
        sns.swarmplot(x=x_column, y=y_column, data=data, hue=hue_column, size=8,
                      palette='Set1', dodge=True, alpha=0.7, ax=ax, order=order, linewidth=1)
        ax.legend(title=hue_column, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Set title
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'{y_column} over {x_column}')
    
    # Adjust x-axis labels if needed
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Adjust layout to prevent cutting off labels
    fig.tight_layout()
    
    # Save the figure if filename is provided
    if filename:
        # Save as PNG
        fig.savefig(f"{filename}.png", dpi=300, bbox_inches='tight')
        print(f"Figure saved as {filename}.png")
        
        # Save as EPS
        fig.savefig(f"{filename}.eps", format='eps', bbox_inches='tight')
        print(f"Figure saved as {filename}.eps")
    

    return fig, ax
