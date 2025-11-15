"""
Krum aggregation algorithm for Byzantine-resilient federated learning.
Selects updates with minimum sum of distances to neighbors.

Usage:
    from server.krum import krum_aggregate
"""
import numpy as np
from typing import List


def krum_aggregate(weights_list: List[List[np.ndarray]], num_malicious: int = 1) -> List[np.ndarray]:
    """
    Krum aggregation: select the update closest to the cluster of honest clients.
    
    Args:
        weights_list: List of model weights from clients (each is list of np arrays)
        num_malicious: Number of potentially malicious clients to exclude
        
    Returns:
        Selected weights (one of the input weight sets)
    """
    n = len(weights_list)
    
    if n <= 2 * num_malicious + 2:
        print(f"Warning: Krum needs n > 2f+2 (n={n}, f={num_malicious}). Using first update.")
        return weights_list[0]
    
    # Flatten each client's weights into a single vector
    flattened = []
    for weights in weights_list:
        flat = np.concatenate([w.flatten() for w in weights])
        flattened.append(flat)
    
    # Compute pairwise distances
    n = len(flattened)
    distances = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(flattened[i] - flattened[j])
            distances[i, j] = dist
            distances[j, i] = dist
    
    # For each client, compute score = sum of distances to n-f-2 closest neighbors
    scores = []
    m = n - num_malicious - 2  # Number of neighbors to consider
    
    for i in range(n):
        # Get distances to all other clients
        dists_i = distances[i].copy()
        dists_i[i] = np.inf  # Exclude self
        
        # Select m closest neighbors
        closest_indices = np.argsort(dists_i)[:m]
        score = np.sum(dists_i[closest_indices])
        scores.append(score)
    
    # Select client with minimum score (most central)
    selected_idx = np.argmin(scores)
    
    print(f"Krum selected client {selected_idx} (score: {scores[selected_idx]:.3f})")
    
    return weights_list[selected_idx]


def multi_krum(weights_list: List[List[np.ndarray]], num_malicious: int = 1, 
               num_select: int = 3) -> List[np.ndarray]:
    """
    Multi-Krum: select top-k updates and average them.
    
    Args:
        weights_list: List of model weights from clients
        num_malicious: Number of potentially malicious clients
        num_select: Number of updates to select and average
        
    Returns:
        Averaged weights from selected clients
    """
    n = len(weights_list)
    
    if n <= 2 * num_malicious + 2:
        # Fallback to simple average
        return average_weights(weights_list)
    
    # Flatten weights
    flattened = []
    for weights in weights_list:
        flat = np.concatenate([w.flatten() for w in weights])
        flattened.append(flat)
    
    # Compute pairwise distances
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(flattened[i] - flattened[j])
            distances[i, j] = dist
            distances[j, i] = dist
    
    # Compute scores
    scores = []
    m = n - num_malicious - 2
    
    for i in range(n):
        dists_i = distances[i].copy()
        dists_i[i] = np.inf
        closest_indices = np.argsort(dists_i)[:m]
        score = np.sum(dists_i[closest_indices])
        scores.append(score)
    
    # Select top-k with lowest scores
    selected_indices = np.argsort(scores)[:num_select]
    selected_weights = [weights_list[i] for i in selected_indices]
    
    # Average selected weights
    return average_weights(selected_weights)


def average_weights(weights_list: List[List[np.ndarray]]) -> List[np.ndarray]:
    """
    Simple average of weight lists.
    
    Args:
        weights_list: List of weight lists
        
    Returns:
        Averaged weights
    """
    num_clients = len(weights_list)
    averaged = []
    
    for layer_idx in range(len(weights_list[0])):
        layer_sum = np.sum([w[layer_idx] for w in weights_list], axis=0)
        averaged.append(layer_sum / num_clients)
    
    return averaged
