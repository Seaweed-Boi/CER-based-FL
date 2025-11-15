"""
Evaluation metrics for fraud detection - F2 score focus.

Usage:
    from evaluate.metrics import compute_f2_score, compute_metrics
"""
import numpy as np
from sklearn.metrics import fbeta_score, precision_recall_fscore_support, confusion_matrix


def compute_f2_score(y_true, y_pred):
    """
    Compute F2 score (emphasizes recall over precision).
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        F2 score (float)
    """
    return fbeta_score(y_true, y_pred, beta=2, average='binary', zero_division=0)


def compute_metrics(y_true, y_pred):
    """
    Compute comprehensive metrics for binary classification.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary of metrics
    """
    f2 = compute_f2_score(y_true, y_pred)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )
    
    accuracy = np.mean(np.array(y_true) == np.array(y_pred))
    
    return {
        "f2": float(f2),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "accuracy": float(accuracy),
    }


def compute_confusion_matrix(y_true, y_pred):
    """
    Compute confusion matrix for binary classification.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        2x2 confusion matrix
    """
    return confusion_matrix(y_true, y_pred, labels=[0, 1])
