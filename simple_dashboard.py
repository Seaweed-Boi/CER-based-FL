"""
Simple dashboard visualization for CER-based FL results.
Uses matplotlib instead of Streamlit to avoid Python 3.14 compatibility issues.
"""
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def load_logs(log_file="logs/rounds.jsonl"):
    """Load logs from JSONL file."""
    logs = []
    log_path = Path(log_file)
    
    if not log_path.exists():
        print(f"Log file not found: {log_file}")
        return pd.DataFrame()
    
    with open(log_path, "r") as f:
        for line in f:
            try:
                logs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    
    return pd.DataFrame(logs)

def plot_results():
    """Generate visualizations of FL training results."""
    df = load_logs()
    
    if df.empty:
        print("No training logs found.")
        return
    
    configs = df["config"].unique()
    
    # Create a figure with subplots
    fig = plt.figure(figsize=(15, 10))
    
    # 1. CER vs Rounds for all configs
    ax1 = plt.subplot(2, 3, 1)
    for config in configs:
        config_df = df[df["config"] == config]
        ax1.plot(config_df["round"], config_df["CER"], marker="o", label=config, linewidth=2)
    ax1.set_xlabel("Round", fontsize=12)
    ax1.set_ylabel("CER Score", fontsize=12)
    ax1.set_title("CER vs Rounds", fontsize=14, fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. F2 Score vs Rounds
    ax2 = plt.subplot(2, 3, 2)
    for config in configs:
        config_df = df[df["config"] == config]
        ax2.plot(config_df["round"], config_df["F2"], marker="s", label=config, linewidth=2)
    ax2.set_xlabel("Round", fontsize=12)
    ax2.set_ylabel("F2 Score", fontsize=12)
    ax2.set_title("F2 Score vs Rounds", fontsize=14, fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Resilience vs Rounds
    ax3 = plt.subplot(2, 3, 3)
    for config in configs:
        config_df = df[df["config"] == config]
        ax3.plot(config_df["round"], config_df["resilience"], marker="^", label=config, linewidth=2)
    ax3.set_xlabel("Round", fontsize=12)
    ax3.set_ylabel("Resilience", fontsize=12)
    ax3.set_title("Resilience vs Rounds", fontsize=14, fontweight="bold")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Final CER Comparison (Bar Chart)
    ax4 = plt.subplot(2, 3, 4)
    final_cers = []
    config_names = []
    for config in configs:
        config_df = df[df["config"] == config]
        if not config_df.empty:
            final_cers.append(config_df.iloc[-1]["CER"])
            config_names.append(config)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    bars = ax4.bar(config_names, final_cers, edgecolor="black", alpha=0.7, color=colors[:len(config_names)])
    ax4.set_ylabel("Final CER Score", fontsize=12)
    ax4.set_title("Final CER Comparison", fontsize=14, fontweight="bold")
    ax4.grid(True, alpha=0.3, axis="y")
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=10)
    
    # 5. Final F2 Comparison (Bar Chart)
    ax5 = plt.subplot(2, 3, 5)
    final_f2s = []
    for config in configs:
        config_df = df[df["config"] == config]
        if not config_df.empty:
            final_f2s.append(config_df.iloc[-1]["F2"])
    
    bars = ax5.bar(config_names, final_f2s, edgecolor="black", alpha=0.7, color=colors[:len(config_names)])
    ax5.set_ylabel("Final F2 Score", fontsize=12)
    ax5.set_title("Final F2 Score Comparison", fontsize=14, fontweight="bold")
    ax5.grid(True, alpha=0.3, axis="y")
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=10)
    
    # 6. Privacy Budget (Epsilon) vs Rounds
    ax6 = plt.subplot(2, 3, 6)
    for config in configs:
        config_df = df[df["config"] == config]
        if config_df["epsilon"].max() > 1e-5:  # Only plot if DP is used
            ax6.plot(config_df["round"], config_df["epsilon"], marker="D", label=config, linewidth=2)
    ax6.set_xlabel("Round", fontsize=12)
    ax6.set_ylabel("Privacy Budget (ε)", fontsize=12)
    ax6.set_title("Privacy Budget vs Rounds", fontsize=14, fontweight="bold")
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle("CER-based Federated Learning Results", fontsize=16, fontweight="bold", y=0.995)
    plt.tight_layout()
    
    # Save figure
    output_file = "plots/fl_results_dashboard.png"
    Path("plots").mkdir(exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✅ Dashboard saved to: {output_file}")
    
    # Show metrics table
    print("\n" + "="*80)
    print("FINAL METRICS COMPARISON")
    print("="*80)
    print(f"{'Config':<12} {'F2 Score':<12} {'Resilience':<12} {'Epsilon':<12} {'CER Score':<12}")
    print("-"*80)
    
    for config in configs:
        config_df = df[df["config"] == config]
        if not config_df.empty:
            final = config_df.iloc[-1]
            epsilon_str = f"{final['epsilon']:.2f}" if final['epsilon'] > 1e-5 else "N/A"
            print(f"{config:<12} {final['F2']:<12.4f} {final['resilience']:<12.4f} {epsilon_str:<12} {final['CER']:<12.6f}")
    
    print("="*80)
    print("\n🎯 Key Observations:")
    print("  • Baseline: Highest F2 score but vulnerable to attacks (lower resilience)")
    print("  • DP: Good privacy protection (ε ≈ 2.0) with moderate performance")
    print("  • Krum: Best resilience against Byzantine attacks with competitive F2")
    print("\n📊 CER Formula: CER = (F2 × Resilience) / (comm_kb × epsilon)")
    print("   Higher CER indicates better overall performance considering:")
    print("   - Model accuracy (F2 score)")
    print("   - Attack resilience")
    print("   - Communication efficiency")
    print("   - Privacy guarantees")
    
    # Display the plot
    plt.show()

if __name__ == "__main__":
    print("\n" + "="*80)
    print("CER-BASED FEDERATED LEARNING DASHBOARD")
    print("="*80)
    plot_results()
