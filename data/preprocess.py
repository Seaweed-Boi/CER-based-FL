"""
Preprocessing utilities for tabular fraud detection data.

Usage:
    from data.preprocess import load_client_data
"""
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torch


class FraudDataset(Dataset):
    """PyTorch Dataset for fraud detection."""
    
    def __init__(self, df):
        """
        Args:
            df: DataFrame with feature columns and 'label' column
        """
        self.features = df.drop(columns=["label"]).values.astype(np.float32)
        self.labels = df["label"].values.astype(np.int64)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return torch.tensor(self.features[idx]), torch.tensor(self.labels[idx])


def load_client_data(client_id, batch_size=32):
    """
    Load training data for a specific client from shard file.
    
    Args:
        client_id: Client ID (0, 1, 2, ...)
        batch_size: Batch size for DataLoader
        
    Returns:
        DataLoader for client's training data
    """
    shard_path = Path(f"data/shards/client_{client_id}.csv")
    
    if not shard_path.exists():
        raise FileNotFoundError(
            f"Shard file not found: {shard_path}\n"
            "Run 'python data/split_shards.py --num-clients N' first"
        )
    
    df = pd.read_csv(shard_path)
    dataset = FraudDataset(df)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader


def load_test_data(batch_size=32):
    """
    Load centralized test data.
    
    Args:
        batch_size: Batch size for DataLoader
        
    Returns:
        DataLoader for test data
    """
    test_path = Path("data/raw/test.csv")
    
    if not test_path.exists():
        raise FileNotFoundError(
            f"Test file not found: {test_path}\n"
            "Run 'python data/generate_synthetic.py' first"
        )
    
    df = pd.read_csv(test_path)
    dataset = FraudDataset(df)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    return dataloader
