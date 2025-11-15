"""
Plotting utilities for CER-based FL visualizations.

Usage:
    from evaluate.plot import plot_cer_vs_rounds, plot_config_comparison
"""
import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path
import pandas as pd


def load_logs(log_file="logs/rounds.jsonl"):
    """
    Load logs from JSONL file into DataFrame.
    
    Args:
        log_file: Path to log file
        
    Returns:
        DataFrame with log entries
    """
    logs = []
    log_path = Path(log_file)
    
    if not log_path.exists():
        print(f"Warning: Log file not found: {log_file}")
        return pd.DataFrame()
    
    with open(log_path, "r") as f:
        for line in f:
            try:
                logs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    
    return pd.DataFrame(logs)


def plot_cer_vs_rounds(log_file="logs/rounds.jsonl", save_path="plots/cer_vs_rounds.png"):
    """
    Plot CER vs rounds for each configuration.
    
    Args:
        log_file: Path to log file
        save_path: Path to save plot
    """
    df = load_logs(log_file)
    
    if df.empty:
        print("No data to plot")
        return
    
    # Create plot directory
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    
    for config in df["config"].unique():
        config_df = df[df["config"] == config]
        plt.plot(config_df["round"], config_df["CER"], marker="o", label=config, linewidth=2)
    
    plt.xlabel("Round", fontsize=12)
    plt.ylabel("CER Score", fontsize=12)
    plt.title("CER vs Rounds by Configuration", fontsize=14, fontweight="bold")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"CER vs rounds plot saved to {save_path}")


def plot_f2_vs_rounds(log_file="logs/rounds.jsonl", save_path="plots/f2_vs_rounds.png"):
    """
    Plot F2 score vs rounds for each configuration.
    
    Args:
        log_file: Path to log file
        save_path: Path to save plot
    """
    df = load_logs(log_file)
    
    if df.empty:
        print("No data to plot")
        return
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    
    for config in df["config"].unique():
        config_df = df[df["config"] == config]
        plt.plot(config_df["round"], config_df["F2"], marker="o", label=config, linewidth=2)
    
    plt.xlabel("Round", fontsize=12)
    plt.ylabel("F2 Score", fontsize=12)
    plt.title("F2 Score vs Rounds by Configuration", fontsize=14, fontweight="bold")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"F2 vs rounds plot saved to {save_path}")


def plot_config_comparison(log_file="logs/rounds.jsonl", save_path="plots/config_comparison.png"):
    """
    Bar chart comparing final CER for each configuration.
    
    Args:
        log_file: Path to log file
        save_path: Path to save plot
    """
    df = load_logs(log_file)
    
    if df.empty:
        print("No data to plot")
        return
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Get final CER for each config
    final_cers = {}
    for config in df["config"].unique():
        config_df = df[df["config"] == config]
        if not config_df.empty:
            final_cers[config] = config_df.iloc[-1]["CER"]
    
    configs = list(final_cers.keys())
    cers = list(final_cers.values())
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(configs, cers, edgecolor="black", alpha=0.7, width=0.6)
    
    # Color bars
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    plt.ylabel("Final CER Score", fontsize=12)
    plt.title("Configuration Comparison (Final CER)", fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"Configuration comparison plot saved to {save_path}")


def plot_all(log_file="logs/rounds.jsonl"):
    """
    Generate all plots.
    
    Args:
        log_file: Path to log file
    """
    print("Generating plots...")
    plot_cer_vs_rounds(log_file)
    plot_f2_vs_rounds(log_file)
    plot_config_comparison(log_file)
    print("All plots generated successfully!")


if __name__ == "__main__":
    plot_all()
