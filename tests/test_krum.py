"""
Unit tests for Krum aggregation algorithm.
Tests Byzantine-resilient aggregation by detecting outlier updates.
"""
import pytest
import torch
from collections import OrderedDict
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from server.krum import krum_aggregate, multi_krum, average_weights


def create_model_weights(values):
    """Helper to create mock model weights."""
    weights = OrderedDict()
    weights["fc1.weight"] = torch.tensor(values, dtype=torch.float32)
    return weights


def test_krum_rejects_outlier():
    """Test that Krum rejects outlier (attack) update."""
    # 4 honest clients with similar weights
    honest_updates = [
        create_model_weights([1.0, 1.0, 1.0]),
        create_model_weights([1.1, 0.9, 1.0]),
        create_model_weights([0.9, 1.1, 1.0]),
        create_model_weights([1.0, 1.0, 1.1]),
    ]
    
    # 1 malicious client with very different weights (attack)
    malicious_update = create_model_weights([100.0, 100.0, 100.0])
    
    all_updates = honest_updates + [malicious_update]
    
    # Krum should select one of the honest updates
    selected = krum_aggregate(all_updates, num_malicious=1)
    
    # Check that selected weights are close to honest range (not 100)
    selected_values = selected["fc1.weight"].numpy()
    assert all(0.5 < v < 2.0 for v in selected_values), \
        f"Krum selected outlier! Values: {selected_values}"
    
    print("✓ Krum correctly rejected outlier update")


def test_multi_krum():
    """Test Multi-Krum aggregation."""
    updates = [
        create_model_weights([1.0, 1.0, 1.0]),
        create_model_weights([1.1, 0.9, 1.0]),
        create_model_weights([0.9, 1.1, 1.0]),
        create_model_weights([100.0, 100.0, 100.0]),  # outlier
    ]
    
    # Multi-Krum with k=2 should average 2 best updates
    aggregated = multi_krum(updates, num_malicious=1, k=2)
    
    # Should be close to 1.0 (average of honest)
    values = aggregated["fc1.weight"].numpy()
    assert all(0.8 < v < 1.2 for v in values), \
        f"Multi-Krum failed! Values: {values}"
    
    print("✓ Multi-Krum correctly aggregated honest updates")


def test_average_weights():
    """Test simple averaging."""
    weights_list = [
        create_model_weights([1.0, 2.0, 3.0]),
        create_model_weights([3.0, 2.0, 1.0]),
    ]
    
    avg = average_weights(weights_list)
    expected = torch.tensor([2.0, 2.0, 2.0])
    
    assert torch.allclose(avg["fc1.weight"], expected), \
        f"Averaging failed! Got {avg['fc1.weight']}, expected {expected}"
    
    print("✓ Average aggregation correct")


if __name__ == "__main__":
    test_krum_rejects_outlier()
    test_multi_krum()
    test_average_weights()
    print("\n✅ All Krum tests passed!")
