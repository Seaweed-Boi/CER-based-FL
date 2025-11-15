"""
Split dataset into non-IID shards for federated clients.
Implements varying fraud rates per client to simulate real-world heterogeneity.

Usage:
    python data/split_shards.py --num-clients 3
"""
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import pickle


def split_non_iid_by_fraud_rate(df, num_clients, random_state=42):
    """
    Split dataset into non-IID shards with varying fraud rates per client.
    
    Args:
        df: DataFrame with features and label column
        num_clients: Number of clients to split data among
        random_state: Random seed
        
    Returns:
        List of DataFrames, one per client
    """
    np.random.seed(random_state)
    
    # Separate fraud and non-fraud
    fraud_df = df[df["label"] == 1].copy()
    non_fraud_df = df[df["label"] == 0].copy()
    
    # Shuffle
    fraud_df = fraud_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    non_fraud_df = non_fraud_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # Vary fraud rates: give different clients different fraud proportions
    # E.g., client 0 gets more fraud, client 1 less, etc.
    fraud_ratios = np.random.dirichlet(np.ones(num_clients) * 2, size=1)[0]
    fraud_ratios = fraud_ratios / fraud_ratios.sum()
    
    # Split fraud samples according to ratios
    fraud_splits = []
    start = 0
    for i in range(num_clients):
        if i < num_clients - 1:
            end = start + int(len(fraud_df) * fraud_ratios[i])
        else:
            end = len(fraud_df)
        fraud_splits.append(fraud_df.iloc[start:end])
        start = end
    
    # Split non-fraud samples more evenly
    non_fraud_per_client = len(non_fraud_df) // num_clients
    non_fraud_splits = []
    for i in range(num_clients):
        start = i * non_fraud_per_client
        if i < num_clients - 1:
            end = start + non_fraud_per_client
        else:
            end = len(non_fraud_df)
        non_fraud_splits.append(non_fraud_df.iloc[start:end])
    
    # Combine fraud and non-fraud for each client
    client_dfs = []
    for i in range(num_clients):
        client_df = pd.concat([fraud_splits[i], non_fraud_splits[i]], axis=0)
        client_df = client_df.sample(frac=1, random_state=random_state + i).reset_index(drop=True)
        client_dfs.append(client_df)
        
        fraud_rate = client_df["label"].mean()
        print(f"Client {i}: {len(client_df)} samples, fraud rate: {fraud_rate:.3f}")
    
    return client_dfs


def create_shards(num_clients, random_state=42):
    """
    Create and save data shards for clients.
    
    Args:
        num_clients: Number of clients
        random_state: Random seed
    """
    # Load train data
    train_path = Path("data/raw/train.csv")
    if not train_path.exists():
        print("ERROR: train.csv not found. Run 'python data/generate_synthetic.py' first.")
        return
    
    train_df = pd.read_csv(train_path)
    print(f"Loaded training data: {len(train_df)} samples")
    
    # Split into non-IID shards
    print(f"\nSplitting into {num_clients} non-IID shards...")
    client_dfs = split_non_iid_by_fraud_rate(train_df, num_clients, random_state)
    
    # Save shards
    shard_dir = Path("data/shards")
    shard_dir.mkdir(parents=True, exist_ok=True)
    
    for client_id, client_df in enumerate(client_dfs):
        shard_path = shard_dir / f"client_{client_id}.csv"
        client_df.to_csv(shard_path, index=False)
    
    print(f"\nShards saved to {shard_dir}/")
    print(f"  Created {num_clients} client shard files")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split data into client shards")
    parser.add_argument("--num-clients", type=int, default=3, help="Number of clients")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    create_shards(args.num_clients, args.random_state)
