"""
Binary classification MLP model with Focal Loss for imbalanced fraud detection.

Usage:
    from clients.model import BinaryClassifier, FocalLoss
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryClassifier(nn.Module):
    """Small MLP for binary classification."""
    
    def __init__(self, input_dim=20, hidden_dims=[64, 32]):
        """
        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer dimensions
        """
        super(BinaryClassifier, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Logits of shape (batch_size, 1)
        """
        return self.model(x)


class FocalLoss(nn.Module):
    """
    Focal Loss for imbalanced binary classification.
    
    Focal Loss = -alpha * (1 - p_t)^gamma * log(p_t)
    where p_t is the probability of the correct class.
    """
    
    def __init__(self, alpha=0.25, gamma=2.0):
        """
        Args:
            alpha: Weighting factor for positive class
            gamma: Focusing parameter (higher = more focus on hard examples)
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits, targets):
        """
        Args:
            logits: Model outputs of shape (batch_size, 1)
            targets: Ground truth labels of shape (batch_size,)
            
        Returns:
            Scalar loss value
        """
        # Convert logits to probabilities
        probs = torch.sigmoid(logits.squeeze())
        targets = targets.float()
        
        # Compute p_t
        p_t = probs * targets + (1 - probs) * (1 - targets)
        
        # Compute focal loss
        focal_weight = (1 - p_t) ** self.gamma
        
        # Binary cross-entropy
        bce = F.binary_cross_entropy(probs, targets, reduction='none')
        
        # Apply alpha weighting
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        loss = alpha_t * focal_weight * bce
        
        return loss.mean()


def create_model(input_dim=20):
    """
    Factory function to create a binary classifier.
    
    Args:
        input_dim: Number of input features
        
    Returns:
        BinaryClassifier instance
    """
    return BinaryClassifier(input_dim=input_dim)
