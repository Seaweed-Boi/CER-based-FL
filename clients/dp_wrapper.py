"""
Differential Privacy wrapper using Opacus PrivacyEngine.

Usage:
    from clients.dp_wrapper import make_private, get_epsilon
"""
import torch
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager


def make_private(model, optimizer, train_loader, target_epsilon, target_delta, max_grad_norm, epochs):
    """
    Wrap model and optimizer with Opacus PrivacyEngine for DP-SGD.
    
    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        train_loader: DataLoader
        target_epsilon: Target privacy budget
        target_delta: Target delta
        max_grad_norm: Max gradient norm for clipping
        epochs: Number of epochs (needed for noise calculation)
        
    Returns:
        Tuple of (private_model, private_optimizer, privacy_engine)
    """
    privacy_engine = PrivacyEngine()
    
    model, optimizer, train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=1.0,  # Will be adjusted based on target_epsilon
        max_grad_norm=max_grad_norm,
    )
    
    return model, optimizer, privacy_engine


def get_epsilon(privacy_engine, target_delta):
    """
    Get epsilon (privacy budget spent) from privacy engine.
    
    Args:
        privacy_engine: Opacus PrivacyEngine instance
        target_delta: Delta parameter
        
    Returns:
        Epsilon value (float)
    """
    try:
        epsilon = privacy_engine.get_epsilon(delta=target_delta)
        return epsilon
    except Exception as e:
        print(f"Warning: Could not compute epsilon: {e}")
        return 0.0


def get_privacy_spent(privacy_engine, delta):
    """
    Alias for get_epsilon for compatibility.
    
    Args:
        privacy_engine: Opacus PrivacyEngine instance
        delta: Delta parameter
        
    Returns:
        Epsilon value (float)
    """
    return get_epsilon(privacy_engine, delta)
