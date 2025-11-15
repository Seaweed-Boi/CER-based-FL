"""
Load custom dataset for federated learning.
Loads data from CSV file and creates train/test splits.

Usage:
    python data/generate_synthetic.py --dataset path/to/your/dataset.csv --target column_name
    
    Or for synthetic data:
    python data/generate_synthetic.py --synthetic
"""
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from pathlib import Path
import yaml
import argparse


def load_custom_data(dataset_path, target_column, test_size=0.2, random_state=42):
    """
    Load custom dataset from CSV file.
    
    Args:
        dataset_path: Path to CSV file
        target_column: Name of target/label column
        test_size: Fraction of data for test set
        random_state: Random seed for reproducibility
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    print(f"Loading dataset from {dataset_path}...")
    
    # Load CSV
    df = pd.read_csv(dataset_path)
    print(f"  Total samples: {len(df)}")
    print(f"  Total columns: {len(df.columns)}")
    
    # Separate features and target
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset. Available columns: {df.columns.tolist()}")
    
    y = df[target_column].values
    X_df = df.drop(columns=[target_column])
    
    # Select only numeric columns
    numeric_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) == 0:
        raise ValueError("No numeric columns found in dataset. Please ensure your dataset has numeric features.")
    
    X = X_df[numeric_cols].values
    
    print(f"  Features: {X.shape[1]} (numeric columns: {numeric_cols})")
    print(f"  Positive class rate: {y.mean():.3f}")
    
    # Handle missing values (simple imputation)
    if np.any(np.isnan(X)):
        print("  Warning: Dataset contains missing values. Filling with column means.")
        col_means = np.nanmean(X, axis=0)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_means, inds[1])
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if len(np.unique(y)) > 1 else None
    )
    
    return X_train, X_test, y_train, y_test


def generate_synthetic_data(config_path="config.yaml"):
    """
    Generate synthetic fraud-like binary classification dataset.
    
    Args:
        config_path: Path to config file with data parameters
    """
    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    data_config = config["data"]
    
    print("Generating synthetic dataset...")
    print(f"  Samples: {data_config['n_samples']}")
    print(f"  Features: {data_config['n_features']}")
    print(f"  Fraud weight: {data_config['fraud_weight']}")
    
    # Generate imbalanced classification dataset
    X, y = make_classification(
        n_samples=data_config["n_samples"],
        n_features=data_config["n_features"],
        n_informative=data_config["n_informative"],
        n_redundant=data_config["n_redundant"],
        n_classes=2,
        weights=[1 - data_config["fraud_weight"], data_config["fraud_weight"]],
        flip_y=0.01,  # Small label noise
        random_state=data_config["random_state"],
    )
    
    # Split into train and centralized test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=data_config["test_size"],
        random_state=data_config["random_state"],
        stratify=y
    )
    
    print(f"  Train set: {len(X_train)} samples")
    print(f"  Test set: {len(X_test)} samples")
    print(f"  Fraud rate (train): {y_train.mean():.3f}")
    print(f"  Fraud rate (test): {y_test.mean():.3f}")
    
    return X_train, X_test, y_train, y_test


def save_data(X_train, X_test, y_train, y_test):
    """Save train/test data to CSV files."""
    # Save to CSV
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    train_df = pd.DataFrame(X_train, columns=[f"feature_{i}" for i in range(X_train.shape[1])])
    train_df["label"] = y_train
    train_df.to_csv(data_dir / "train.csv", index=False)
    
    test_df = pd.DataFrame(X_test, columns=[f"feature_{i}" for i in range(X_test.shape[1])])
    test_df["label"] = y_test
    test_df.to_csv(data_dir / "test.csv", index=False)
    
    print(f"\nDataset saved to {data_dir}/")
    print("  - train.csv")
    print("  - test.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate or load dataset for FL")
    parser.add_argument("--dataset", type=str, help="Path to custom dataset CSV file")
    parser.add_argument("--target", type=str, default="label", help="Name of target column")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic data instead")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set fraction (default: 0.2)")
    args = parser.parse_args()
    
    if args.synthetic or not args.dataset:
        # Generate synthetic data
        X_train, X_test, y_train, y_test = generate_synthetic_data()
    else:
        # Load custom dataset
        X_train, X_test, y_train, y_test = load_custom_data(
            args.dataset, 
            args.target,
            test_size=args.test_size
        )
    
    # Save to standard location
    save_data(X_train, X_test, y_train, y_test)
