

def plot_si_histogram(df_info_scores, S_info_scores, output_dir, title_prefix=''):
    """
    Plot histogram of spatial information (SI) values for ΔF/F and inferred spikes.

    Args:
        df_info_scores (ndarray or list): SI values for ΔF/F
        S_info_scores (ndarray or list): SI values for inferred spikes
        output_dir (str): folder to save plot
        title_prefix (str): text to prepend to figure title and filename
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(6, 4))
    bins = np.linspace(0, max(np.nanmax(df_info_scores), np.nanmax(S_info_scores)), 30)
    
    plt.hist(df_info_scores, bins=bins, alpha=0.6, label='ΔF/F', color='tab:blue')
    plt.hist(S_info_scores, bins=bins, alpha=0.6, label='Spikes', color='tab:orange')
    
    plt.xlabel('Spatial Information (bits/event)')
    plt.ylabel('Number of neurons')
    plt.title(f'{title_prefix} SI Histogram')
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, f"{title_prefix}_SI_histogram.png"), dpi=300)
    plt.close()

def plot_tuning_maps(signal_data, x_data, y_data, place_cells, scores, arena_size, n_bins_x, n_bins_y, output_dir, title_prefix=''):
    """
    Plot tuning maps for a set of place cells and save them.

    Args:
        signal_data (ndarray): shape (n_neurons, n_timepoints)
        x_data, y_data (ndarray): position coordinates
        place_cells (list[int]): indices of neurons classified as place cells
        arena_size (tuple): (width, height) in cm
        n_bins_x, n_bins_y (int): number of bins in X and Y
        output_dir (str): folder to save plot
        title_prefix (str): text to prepend to figure title and filename
    """
    import matplotlib.pyplot as plt
    from scipy.stats import binned_statistic_2d
    import numpy as np
    import os

    if len(place_cells) == 0:
        print(f"No place cells to plot for {title_prefix}.")
        return

    os.makedirs(output_dir, exist_ok=True)
    n_cols = 5
    n_rows = int(np.ceil(len(place_cells) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    axes = axes.flatten()

    for i, n in enumerate(place_cells):
        tm, _, _, _ = binned_statistic_2d(
            x_data, y_data, signal_data[n, :],
            statistic='mean', bins=[n_bins_x, n_bins_y],
            range=[[0, arena_size[0]], [0, arena_size[1]]]
        )
        im = axes[i].imshow(tm.T, origin='lower', cmap='plasma')
        axes[i].set_title(f'Neuron {n} - Info: {scores[n]:.2f} ')
        axes[i].axis('off')

    for j in range(i+1, len(axes)):
        axes[j].axis('off')

    # fig.suptitle(f"{title_prefix} Tuning Maps")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{title_prefix}_placecell_maps.png"), dpi=300)
    plt.close()