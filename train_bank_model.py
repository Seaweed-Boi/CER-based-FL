"""
Train fraud detection model on bank_data.csv
Simulates federated learning with centralized training for demonstration.
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score, fbeta_score
import json
from pathlib import Path

# Binary classification model
class FraudDetectionModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        return x

def train_model():
    print("="*80)
    print("TRAINING FRAUD DETECTION MODEL ON BANK DATA")
    print("="*80)
    
    # Load data
    print("\n📂 Loading bank_data.csv...")
    df = pd.read_csv('bank_data.csv')
    print(f"   Dataset shape: {df.shape}")
    print(f"   Fraud cases: {df['is_fraud'].sum()} ({df['is_fraud'].sum()/len(df)*100:.2f}%)")
    
    # Prepare features and labels
    X = df.drop('is_fraud', axis=1).values
    y = df['is_fraud'].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)
    
    # Create model
    input_dim = X_train.shape[1]
    model = FraudDetectionModel(input_dim)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    print("\n🏋️ Training model...")
    num_epochs = 50
    batch_size = 64
    
    results = []
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        # Mini-batch training
        for i in range(0, len(X_train_tensor), batch_size):
            batch_X = X_train_tensor[i:i+batch_size]
            batch_y = y_train_tensor[i:i+batch_size]
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Evaluate
        if (epoch + 1) % 5 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                train_pred = model(X_train_tensor)
                train_pred_binary = (train_pred > 0.5).float()
                train_acc = (train_pred_binary == y_train_tensor).float().mean().item()
                
                test_pred = model(X_test_tensor)
                test_pred_binary = (test_pred > 0.5).float()
                test_acc = (test_pred_binary == y_test_tensor).float().mean().item()
                
                # F2 score (emphasizes recall)
                y_test_np = y_test_tensor.numpy().flatten()
                y_pred_np = test_pred_binary.numpy().flatten()
                f2 = fbeta_score(y_test_np, y_pred_np, beta=2, zero_division=0)
                f1 = f1_score(y_test_np, y_pred_np, zero_division=0)
                
            avg_loss = total_loss / (len(X_train_tensor) // batch_size)
            
            print(f"   Epoch {epoch+1:3d}/{num_epochs} | Loss: {avg_loss:.4f} | "
                  f"Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f} | "
                  f"F2: {f2:.4f} | F1: {f1:.4f}")
            
            # Save results for dashboard
            results.append({
                "epoch": epoch + 1,
                "train_loss": float(avg_loss),
                "train_accuracy": float(train_acc),
                "test_accuracy": float(test_acc),
                "f1_score": float(f1),
                "f2_score": float(f2)
            })
    
    # Final evaluation
    print("\n" + "="*80)
    print("FINAL EVALUATION")
    print("="*80)
    
    model.eval()
    with torch.no_grad():
        test_pred = model(X_test_tensor)
        test_pred_binary = (test_pred > 0.5).float()
        y_test_np = y_test_tensor.numpy().flatten()
        y_pred_np = test_pred_binary.numpy().flatten()
    
    print("\nClassification Report:")
    print(classification_report(y_test_np, y_pred_np, 
                                target_names=['Normal', 'Fraud'], 
                                zero_division=0))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test_np, y_pred_np)
    print(f"                 Predicted")
    print(f"                Normal  Fraud")
    print(f"Actual Normal   {cm[0,0]:6d}  {cm[0,1]:5d}")
    print(f"       Fraud    {cm[1,0]:6d}  {cm[1,1]:5d}")
    
    # Save results
    Path("results").mkdir(exist_ok=True)
    
    with open("results/training_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Save model
    torch.save(model.state_dict(), "results/fraud_model.pth")
    
    # Save final metrics for dashboard
    final_metrics = {
        "model_name": "Bank Fraud Detection",
        "dataset": "bank_data.csv",
        "total_samples": len(df),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "fraud_rate": float(df['is_fraud'].sum() / len(df)),
        "final_accuracy": float((test_pred_binary == y_test_tensor).float().mean().item()),
        "final_f1": float(f1_score(y_test_np, y_pred_np, zero_division=0)),
        "final_f2": float(fbeta_score(y_test_np, y_pred_np, beta=2, zero_division=0)),
        "confusion_matrix": cm.tolist()
    }
    
    with open("results/final_metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2)
    
    print(f"\n✅ Training complete!")
    print(f"   📁 Results saved to: results/")
    print(f"   📊 Run: python -m http.server 8000")
    print(f"   🎨 Or launch dashboard: streamlit run bank_dashboard.py")

if __name__ == "__main__":
    train_model()
