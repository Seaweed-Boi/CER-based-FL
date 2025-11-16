"""
Streamlit Dashboard for Bank Fraud Detection Model
Works without protobuf dependencies - pure visualization.
"""
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Check if results exist
if not Path("results/training_results.json").exists():
    print("❌ No training results found!")
    print("   Run: python train_bank_model.py")
    sys.exit(1)

try:
    import streamlit as st
    
    # Page config
    st.set_page_config(
        page_title="Bank Fraud Detection Dashboard",
        page_icon="🏦",
        layout="wide"
    )
    
    # Title
    st.title("🏦 Bank Fraud Detection Model Dashboard")
    st.markdown("Real-time monitoring of fraud detection model training")
    
    # Load results
    with open("results/training_results.json", "r") as f:
        training_data = json.load(f)
    
    with open("results/final_metrics.json", "r") as f:
        final_metrics = json.load(f)
    
    df = pd.DataFrame(training_data)
    
    # Metrics overview
    st.header("📊 Model Performance Overview")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Samples", f"{final_metrics['total_samples']:,}")
    col2.metric("Test Accuracy", f"{final_metrics['final_accuracy']:.4f}")
    col3.metric("F1 Score", f"{final_metrics['final_f1']:.4f}")
    col4.metric("F2 Score", f"{final_metrics['final_f2']:.4f}")
    col5.metric("Fraud Rate", f"{final_metrics['fraud_rate']*100:.2f}%")
    
    # Training curves
    st.header("📈 Training Progress")
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("Training Loss")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(df["epoch"], df["train_loss"], marker="o", linewidth=2, color="#1f77b4")
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Loss", fontsize=12)
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    with col_right:
        st.subheader("Accuracy Comparison")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(df["epoch"], df["train_accuracy"], marker="o", label="Train", linewidth=2)
        ax.plot(df["epoch"], df["test_accuracy"], marker="s", label="Test", linewidth=2)
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Accuracy", fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    # F-scores
    st.header("🎯 Performance Metrics")
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("F-Scores Over Time")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(df["epoch"], df["f1_score"], marker="o", label="F1 Score", linewidth=2)
        ax.plot(df["epoch"], df["f2_score"], marker="s", label="F2 Score", linewidth=2)
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Score", fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    with col_right:
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots(figsize=(6, 5))
        cm = final_metrics["confusion_matrix"]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Normal', 'Fraud'],
                    yticklabels=['Normal', 'Fraud'])
        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_ylabel("Actual", fontsize=12)
        st.pyplot(fig)
    
    # Dataset info
    st.header("📋 Dataset Information")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Training Samples", f"{final_metrics['train_samples']:,}")
    col2.metric("Test Samples", f"{final_metrics['test_samples']:,}")
    col3.metric("Model", final_metrics['model_name'])
    
    # Raw data
    with st.expander("📊 View Training Data"):
        st.dataframe(df, use_container_width=True)
    
    # Key findings
    st.header("🔍 Key Findings")
    
    st.markdown(f"""
    - **Model Type**: Neural Network (4 layers: 64→32→16→1)
    - **Dataset**: {final_metrics['dataset']} with {final_metrics['total_samples']:,} transactions
    - **Fraud Detection**: Achieved {final_metrics['final_accuracy']*100:.2f}% accuracy
    - **F2 Score**: {final_metrics['final_f2']:.4f} (emphasizes recall for fraud detection)
    - **True Positives**: {cm[1][1]} frauds correctly detected
    - **False Negatives**: {cm[1][0]} frauds missed
    - **False Positives**: {cm[0][1]} normal transactions flagged
    """)
    
    # Refresh button
    if st.sidebar.button("🔄 Refresh Data"):
        st.rerun()

except ImportError as e:
    print("="*80)
    print("BANK FRAUD DETECTION - TRAINING RESULTS")
    print("="*80)
    
    # Load and display results without Streamlit
    with open("results/training_results.json", "r") as f:
        training_data = json.load(f)
    
    with open("results/final_metrics.json", "r") as f:
        final_metrics = json.load(f)
    
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
    
    print(f"\n✅ Results available in: results/")
    print(f"   To view dashboard with Streamlit:")
    print(f"   1. Fix Python environment (use Python 3.11 or 3.12)")
    print(f"   2. Run: streamlit run bank_dashboard.py")
