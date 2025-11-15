"""
Unit tests for differential privacy implementation.
Tests Opacus PrivacyEngine integration and epsilon tracking.
"""
import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from clients.dp_wrapper import make_private, get_epsilon


class SimpleModel(nn.Module):
    """Simple model for testing."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)
    
    def forward(self, x):
        return self.fc(x)


def test_dp_wrapper():
    """Test that DP wrapper successfully privatizes model."""
    model = SimpleModel()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    # Create dummy data
    X = torch.randn(20, 10)
    y = torch.randint(0, 2, (20,))
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=5)
    
    # Apply DP
    dp_config = {
        "target_epsilon": 2.0,
        "target_delta": 1e-5,
        "max_grad_norm": 1.0,
        "epochs": 1
    }
    
    model_dp, optimizer_dp, dataloader_dp, privacy_engine = make_private(
        model, optimizer, dataloader, dp_config
    )
    
    assert privacy_engine is not None, "PrivacyEngine not returned"
    print("✓ DP wrapper successfully privatized model")


def test_epsilon_tracking():
    """Test epsilon tracking after training."""
    model = SimpleModel()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    X = torch.randn(20, 10)
    y = torch.randint(0, 2, (20,))
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=5)
    
    dp_config = {
        "target_epsilon": 2.0,
        "target_delta": 1e-5,
        "max_grad_norm": 1.0,
        "epochs": 1
    }
    
    model_dp, optimizer_dp, dataloader_dp, privacy_engine = make_private(
        model, optimizer, dataloader, dp_config
    )
    
    # Simulate one epoch of training
    criterion = nn.CrossEntropyLoss()
    model_dp.train()
    for batch_X, batch_y in dataloader_dp:
        optimizer_dp.zero_grad()
        outputs = model_dp(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer_dp.step()
    
    # Check epsilon
    epsilon = get_epsilon(privacy_engine, dp_config["target_delta"])
    
    assert epsilon > 0, f"Epsilon should be positive, got {epsilon}"
    assert epsilon <= dp_config["target_epsilon"] * 2, \
        f"Epsilon too high: {epsilon} > {dp_config['target_epsilon'] * 2}"
    
    print(f"✓ Epsilon tracked: {epsilon:.4f}")


if __name__ == "__main__":
    test_dp_wrapper()
    test_epsilon_tracking()
    print("\n✅ All DP tests passed!")
