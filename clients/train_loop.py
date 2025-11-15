"""
Local training loop for FL clients with DP and attack support.

Usage:
    from clients.train_loop import train, evaluate
"""
import torch
import torch.nn as nn
import torch.optim as optim
from clients.model import FocalLoss
from clients.attacks import apply_label_flip, apply_gradient_scaling


def train(model, train_loader, epochs, device, lr=0.01, 
          use_dp=False, dp_config=None, is_malicious=False, attack_config=None):
    """
    Train model locally on client data.
    
    Args:
        model: PyTorch model
        train_loader: DataLoader
        epochs: Number of epochs
        device: Device to train on
        lr: Learning rate
        use_dp: Whether to use differential privacy
        dp_config: DP configuration dict (if use_dp=True)
        is_malicious: Whether this client is malicious
        attack_config: Attack configuration dict (if is_malicious=True)
        
    Returns:
        Tuple of (trained_model, privacy_engine or None)
    """
    model.to(device)
    model.train()
    
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    
    privacy_engine = None
    
    # Apply DP if requested
    if use_dp and dp_config:
        from clients.dp_wrapper import make_private
        model, optimizer, privacy_engine = make_private(
            model, optimizer, train_loader,
            target_epsilon=dp_config.get("target_epsilon", 2.0),
            target_delta=dp_config.get("target_delta", 1e-5),
            max_grad_norm=dp_config.get("max_grad_norm", 1.0),
            epochs=epochs
        )
    
    for epoch in range(epochs):
        running_loss = 0.0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            
            # Apply label flip attack if malicious
            if is_malicious and attack_config:
                flip_ratio = attack_config.get("label_flip_ratio", 0.3)
                labels = apply_label_flip(labels, flip_ratio)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Apply gradient scaling attack if malicious (after backward)
            if is_malicious and attack_config:
                scale = attack_config.get("gradient_scale", 10.0)
                apply_gradient_scaling(model, scale)
            
            optimizer.step()
            running_loss += loss.item()
        
        avg_loss = running_loss / len(train_loader)
        if epoch % 2 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    return model, privacy_engine


def evaluate(model, test_loader, device):
    """
    Evaluate model on test data.
    
    Args:
        model: PyTorch model
        test_loader: DataLoader for test data
        device: Device to evaluate on
        
    Returns:
        Tuple of (y_true, y_pred) as numpy arrays
    """
    model.to(device)
    model.eval()
    
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            outputs = model(features)
            probs = torch.sigmoid(outputs.squeeze())
            preds = (probs >= 0.5).long()
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    
    return all_labels, all_preds
