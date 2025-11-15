"""
Utility functions for server operations.
"""
import numpy as np
from typing import List, Dict, Any


def compute_weighted_average(results: List[tuple[int, Dict[str, Any]]]) -> Dict[str, float]:
    """
    Compute weighted average of metrics from multiple clients.
    
    Args:
        results: List of (num_examples, metrics_dict) tuples
        
    Returns:
        Dictionary of averaged metrics
    """
    if not results:
        return {}
    
    total_examples = sum(num_examples for num_examples, _ in results)
    
    # Collect all metric keys
    all_keys = set()
    for _, metrics in results:
        all_keys.update(metrics.keys())
    
    # Compute weighted average for each metric
    averaged = {}
    for key in all_keys:
        weighted_sum = sum(
            num_examples * metrics.get(key, 0)
            for num_examples, metrics in results
        )
        averaged[key] = weighted_sum / total_examples if total_examples > 0 else 0
    
    return averaged


def parameters_to_weights(parameters) -> List[np.ndarray]:
    """Convert Flower Parameters to list of NumPy arrays."""
    from flwr.common import parameters_to_ndarrays
    return parameters_to_ndarrays(parameters)


def weights_to_parameters(weights: List[np.ndarray]):
    """Convert list of NumPy arrays to Flower Parameters."""
    from flwr.common import ndarrays_to_parameters
    return ndarrays_to_parameters(weights)
