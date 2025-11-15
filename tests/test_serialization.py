"""
Unit tests for model serialization and communication cost computation.
Tests that model serialization works correctly for CER comm_kb calculation.
"""
import pytest
import torch
import torch.nn as nn
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from clients.model import BinaryClassifier
from evaluate.compute_cer import compute_comm_cost


def test_model_serialization():
    """Test that model can be serialized to bytes."""
    model = BinaryClassifier(input_dim=20)
    
    # Serialize to bytes
    import io
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    size_bytes = buffer.tell()
    
    assert size_bytes > 0, "Model serialization failed"
    print(f"✓ Model serialized: {size_bytes} bytes")


def test_compute_comm_cost():
    """Test communication cost computation."""
    model = BinaryClassifier(input_dim=20)
    
    comm_kb = compute_comm_cost(model)
    
    assert comm_kb > 0, f"Comm cost should be positive, got {comm_kb}"
    assert comm_kb < 1000, f"Comm cost too high: {comm_kb} KB"
    
    print(f"✓ Comm cost computed: {comm_kb:.2f} KB")


def test_state_dict_roundtrip():
    """Test that state_dict can be saved and loaded."""
    model1 = BinaryClassifier(input_dim=20)
    model2 = BinaryClassifier(input_dim=20)
    
    # Set some weights
    with torch.no_grad():
        for param in model1.parameters():
            param.fill_(1.0)
    
    # Save and load
    state = model1.state_dict()
    model2.load_state_dict(state)
    
    # Check equality
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        assert torch.allclose(p1, p2), "State dict roundtrip failed"
    
    print("✓ State dict roundtrip successful")


def test_comm_cost_consistency():
    """Test that comm cost is consistent across calls."""
    model = BinaryClassifier(input_dim=20)
    
    cost1 = compute_comm_cost(model)
    cost2 = compute_comm_cost(model)
    
    assert cost1 == cost2, f"Comm cost inconsistent: {cost1} != {cost2}"
    print(f"✓ Comm cost consistent: {cost1:.2f} KB")


if __name__ == "__main__":
    test_model_serialization()
    test_compute_comm_cost()
    test_state_dict_roundtrip()
    test_comm_cost_consistency()
    print("\n✅ All serialization tests passed!")
    
