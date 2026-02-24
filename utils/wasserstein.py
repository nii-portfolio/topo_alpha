import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import List, Tuple


def sliced_wasserstein_distance(dgm1: np.ndarray,
                                 dgm2: np.ndarray,
                                 n_projections: int = 10) -> float:
    """
    Approximate Wasserstein distance using random projections.
    Faster than exact computation for high-dimensional diagrams.
    """
    if len(dgm1) == 0 and len(dgm2) == 0:
        return 0.0
    
    # Generate random projection directions
    thetas = np.random.uniform(0, 2 * np.pi, n_projections)
    
    distances = []
    for theta in thetas:
        # Project diagrams onto line at angle theta
        proj1 = dgm1[:, 0] * np.cos(theta) + dgm1[:, 1] * np.sin(theta)
        proj2 = dgm2[:, 0] * np.cos(theta) + dgm2[:, 1] * np.sin(theta)
        
        # 1D Wasserstein (sort and match)
        proj1_sorted = np.sort(proj1)
        proj2_sorted = np.sort(proj2)
        
        # Pad to same length
        max_len = max(len(proj1), len(proj2))
        p1 = np.pad(proj1_sorted, (0, max_len - len(proj1)), mode='edge')
        p2 = np.pad(proj2_sorted, (0, max_len - len(proj2)), mode='edge')
        
        dist = np.mean(np.abs(p1 - p2))
        distances.append(dist)
    
    return float(np.mean(distances))


def persistence_weighted_gaussian_kernel(dgm1: np.ndarray,
                                          dgm2: np.ndarray,
                                          sigma: float = 0.5) -> float:
    """
    Compute kernel similarity between persistence diagrams.
    """
    if len(dgm1) == 0 or len(dgm2) == 0:
        return 0.0
    
    # Compute pairwise Gaussian kernel
    K = 0.0
    for p1 in dgm1:
        for p2 in dgm2:
            dist_sq = np.sum((p1 - p2) ** 2)
            K += np.exp(-dist_sq / (2 * sigma ** 2))
    
    # Normalize
    K /= (len(dgm1) * len(dgm2))
    
    return float(K)