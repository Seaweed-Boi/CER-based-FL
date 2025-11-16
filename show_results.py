"""
Simple Matplotlib-based Dashboard for Bank Fraud Detection
Works without Streamlit - saves visualizations to file.
"""
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def create_dashboard():
    print("="*80)
    print("BANK FRAUD DETECTION DASHBOARD")
    print("="*80)
    
    # Load results
    with open("results/training_results.json", "r") as f:
        training_data = json.load(f)
    
    with open("results/final_metrics.json", "r") as f:
        final_metrics = json.load(f)
    
    df = pd.DataFrame(training_data)
    
    # Print metrics
    print(f"\n📊 FINAL METRICS:")
    print(f"   Total Samples: {final_metrics['total_samples']:,}")
    print(f"   Test Accuracy: {final_metrics['final_accuracy']:.4f}")
    print(f"   F1 Score: {final_metrics['final_f1']:.4f}")
    print(f"   F2 Score: {final_metrics['final_f2']:.4f}")
    print(f"   Fraud Rate: {final_metrics['fraud_rate']*100:.2f}%")
    
    cm = final_metrics['confusion_matrix']
    print(f"\n🎯 CONFUSION MATRIX:")
    print(f"                 Predicted")
    print(f"                Normal  Fraud")
    print(f"Actual Normal   {cm[0][0]:6d}  {cm[0][1]:5d}")
    print(f"       Fraud    {cm[1][0]:6d}  {cm[1][1]:5d}")
    
    # Create comprehensive dashboard
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('Bank Fraud Detection Model - Training Dashboard', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # 1. Training Loss
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(df["epoch"], df["train_loss"], marker="o", linewidth=2, color="#1f77b4")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Loss", fontsize=11)
    ax1.set_title("Training Loss Over Time", fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 2. Accuracy Comparison
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(df["epoch"], df["train_accuracy"], marker="o", label="Train", linewidth=2, color="#2ca02c")
    ax2.plot(df["epoch"], df["test_accuracy"], marker="s", label="Test", linewidth=2, color="#ff7f0e")
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Accuracy", fontsize=11)
    ax2.set_title("Train vs Test Accuracy", fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. F-Scores
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(df["epoch"], df["f1_score"], marker="o", label="F1 Score", linewidth=2, color="#d62728")
    ax3.plot(df["epoch"], df["f2_score"], marker="s", label="F2 Score", linewidth=2, color="#9467bd")
    ax3.set_xlabel("Epoch", fontsize=11)
    ax3.set_ylabel("Score", fontsize=11)
    ax3.set_title("F1 and F2 Scores", fontsize=13, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Confusion Matrix
    ax4 = plt.subplot(2, 3, 4)
    im = ax4.imshow(cm, cmap='Blues', alpha=0.7)
    ax4.set_xticks([0, 1])
    ax4.set_yticks([0, 1])
    ax4.set_xticklabels(['Normal', 'Fraud'], fontsize=11)
    ax4.set_yticklabels(['Normal', 'Fraud'], fontsize=11)
    ax4.set_xlabel("Predicted", fontsize=11, fontweight='bold')
    ax4.set_ylabel("Actual", fontsize=11, fontweight='bold')
    ax4.set_title("Confusion Matrix", fontsize=13, fontweight='bold')
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            text = ax4.text(j, i, cm[i][j],
                           ha="center", va="center", 
                           color="white" if cm[i][j] > np.max(cm)/2 else "black",
                           fontsize=20, fontweight='bold')
    
    # 5. Metrics Summary
    ax5 = plt.subplot(2, 3, 5)
    ax5.axis('off')
    
    metrics_text = f"""
    DATASET SUMMARY
    ──────────────────────────────
    Dataset: {final_metrics['dataset']}
    Total Samples: {final_metrics['total_samples']:,}
    Training Samples: {final_metrics['train_samples']:,}
    Test Samples: {final_metrics['test_samples']:,}
    Fraud Rate: {final_metrics['fraud_rate']*100:.2f}%
    
    FINAL PERFORMANCE
    ──────────────────────────────
    Test Accuracy: {final_metrics['final_accuracy']*100:.2f}%
    F1 Score: {final_metrics['final_f1']:.4f}
    F2 Score: {final_metrics['final_f2']:.4f}
    
    DETECTION RESULTS
    ──────────────────────────────
    True Positives: {cm[1][1]} (Frauds Caught)
    False Negatives: {cm[1][0]} (Frauds Missed)
    False Positives: {cm[0][1]} (False Alarms)
    True Negatives: {cm[0][0]} (Correct Normals)
    
    Recall: {cm[1][1]/(cm[1][1]+cm[1][0])*100:.1f}%
    Precision: {cm[1][1]/(cm[1][1]+cm[0][1])*100:.1f}%
    """
    
    ax5.text(0.1, 0.5, metrics_text, fontsize=10, verticalalignment='center',
             fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # 6. Performance Bar Chart
    ax6 = plt.subplot(2, 3, 6)
    metrics_names = ['Accuracy', 'F1 Score', 'F2 Score', 'Recall']
    recall = cm[1][1]/(cm[1][1]+cm[1][0])
    metrics_values = [
        final_metrics['final_accuracy'],
        final_metrics['final_f1'],
        final_metrics['final_f2'],
        recall
    ]
    
    bars = ax6.bar(metrics_names, metrics_values, 
                   color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],
                   edgecolor='black', alpha=0.7)
    ax6.set_ylabel("Score", fontsize=11)
    ax6.set_title("Final Performance Metrics", fontsize=13, fontweight='bold')
    ax6.set_ylim([0, 1.0])
    ax6.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    # Save dashboard
    output_file = "results/fraud_detection_dashboard.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✅ Dashboard saved to: {output_file}")
    
    # Show the dashboard
    plt.show()
    
    print(f"\n🎯 KEY INSIGHTS:")
    print(f"   • Model achieves {final_metrics['final_accuracy']*100:.2f}% accuracy on bank transactions")
    print(f"   • Detects {cm[1][1]} out of {cm[1][1]+cm[1][0]} fraud cases ({cm[1][1]/(cm[1][1]+cm[1][0])*100:.1f}% recall)")
    print(f"   • Only {cm[0][1]} false alarms out of {cm[0][0]+cm[0][1]} normal transactions")
    print(f"   • F2 score of {final_metrics['final_f2']:.4f} emphasizes catching frauds")
    print(f"\n📁 All results saved in: results/")

if __name__ == "__main__":
    create_dashboard()
