import numpy as np
from collections import defaultdict
from scipy.spatial.distance import pdist
from typing import Dict, List, Tuple
import pandas as pd

def compute_distances(assignments: np.ndarray, 
                       indexing: Dict[str, int], 
                       positions: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute distances between registered and non-registered neurons.

    Parameters
    ----------
    assignments : np.ndarray
        Array of assigned neuron IDs across sessions.
    indexing : Dict[str, int]
        Dictionary mapping session keys to column indices in assignments.
    positions : Dict[str, np.ndarray]
        Dictionary mapping session keys to arrays of neuron positions.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Two numpy arrays containing equal number of pairwise distances: 
        registered_distances and non_registered_distances.
    """
    registered_distances: List[float] = []
    non_registered_distances: List[float] = []

    # Calculate distances for registered neurons
    for i in range(assignments.shape[0]):
        neuron_positions = [positions[k][int(assignments[i, j]), :] for k, j in indexing.items() if not np.isnan(assignments[i, j])]
        if len(neuron_positions) > 1:
            dists = pdist(neuron_positions)
            registered_distances.extend(dists)

    # Calculate distances for all neurons within each session
    for s in positions.values():
        dists = pdist(s)
        non_registered_distances.extend(dists)

    # Ensure both lists have the same number of samples
    min_length = min(len(registered_distances), len(non_registered_distances))
    return np.array(registered_distances[:min_length]), np.array(non_registered_distances[:min_length])

def calculate_registration_stats(assignments):
    """Calculate the percentage of neurons registered across sessions."""
    data = defaultdict(list)
    N = assignments.shape[0]
    K = range(1, assignments.shape[1] + 1)

    for k in K:
        reg_matrix_filtered = assignments[np.sum(~np.isnan(assignments), axis=1) >= k]
        n = reg_matrix_filtered.shape[0]
        data['Percentage'].append(round(n / N * 100, 2))
        data['K'].append(k)

    return pd.DataFrame(data)