"""
Attack implementations for testing FL robustness.

Usage:
    from clients.attacks import apply_label_flip, apply_gradient_scaling
"""
import torch
import numpy as np


def apply_label_flip(labels, flip_ratio=0.3):
    """
    Flip labels for attack simulation.
    
    Args:
        labels: Original labels tensor
        flip_ratio: Fraction of labels to flip
        
    Returns:
        Flipped labels tensor
    """
    labels = labels.clone()
    n = len(labels)
    n_flip = int(n * flip_ratio)
    
    if n_flip > 0:
        flip_indices = np.random.choice(n, n_flip, replace=False)
        labels[flip_indices] = 1 - labels[flip_indices]  # Binary flip
    
    return labels


def apply_gradient_scaling(model, scale=10.0):
    """
    Scale model gradients by a factor (Byzantine attack).
    
    Args:
        model: PyTorch model with computed gradients
        scale: Scaling factor
    """
    with torch.no_grad():
        for param in model.parameters():
            if param.grad is not None:
                param.grad.mul_(scale)


def apply_gradient_noise(model, noise_scale=1.0):
    """
    Add large noise to gradients (attack simulation).
    
    Args:
        model: PyTorch model
        noise_scale: Scale of noise to add
    """
    with torch.no_grad():
        for param in model.parameters():
            if param.grad is not None:
                noise = torch.randn_like(param.grad) * noise_scale
                param.grad.add_(noise)
