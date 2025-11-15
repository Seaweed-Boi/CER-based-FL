"""
Compute Client Error Rate (CER) metric and resilience for FL.

Resilience = max(0, 1 - (F2_clean - F2_attack) / F2_clean)
CER = (F2 * Resilience) / (comm_kb * epsilon)

For DP-off: CER = (F2 * Resilience) / (comm_kb * 1e-6)

Usage:
    from evaluate.compute_cer import compute_resilience, compute_cer
"""
import numpy as np


def compute_resilience(f2_clean: float, f2_attack: float) -> float:
    """
    Compute resilience metric comparing clean vs attack performance.
    
    Resilience = max(0, 1 - (F2_clean - F2_attack) / F2_clean)
    
    Higher resilience means the system maintained performance despite attacks.
    
    Args:
        f2_clean: F2 score without attack
        f2_attack: F2 score with attack
        
    Returns:
        Resilience value in [0, 1]
    """
    if f2_clean <= 0:
        return 0.0
    
    resilience = 1 - (f2_clean - f2_attack) / f2_clean
    resilience = max(0.0, min(1.0, resilience))  # Clamp to [0, 1]
    
    return resilience


def compute_cer(f2_score: float, resilience: float, comm_kb: float, epsilon: float = None) -> float:
    """
    Compute Client Error Rate (CER) metric.
    
    CER balances performance (F2), robustness (resilience), communication cost,
    and privacy budget.
    
    When epsilon is None (DP off), uses a small constant (1e-6) in place of epsilon
    to allow comparison across configs.
    
    Args:
        f2_score: F2 score
        resilience: Resilience metric [0, 1]
        comm_kb: Communication cost in KB
        epsilon: Privacy budget (or None if DP off)
        
    Returns:
        CER score (higher is better)
    """
    if comm_kb <= 0:
        return 0.0
    
    # Handle DP-off case
    if epsilon is None:
        # When DP is off, use a very small constant to avoid division issues
        # This makes DP-off configs comparable while penalizing them less for "infinite privacy"
        epsilon_effective = 1e-6
    else:
        epsilon_effective = max(epsilon, 1e-10)  # Avoid division by zero
    
    cer = (f2_score * resilience) / (comm_kb * epsilon_effective)
    
    return cer


def compute_comm_cost(model_state_dict) -> float:
    """
    Compute communication cost in KB from model state dict.
    
    Args:
        model_state_dict: PyTorch state_dict
        
    Returns:
        Communication cost in KB
    """
    import torch
    import io
    
    # Serialize state dict
    buffer = io.BytesIO()
    torch.save(model_state_dict, buffer)
    size_bytes = buffer.tell()
    size_kb = size_bytes / 1024.0
    
    return size_kb
