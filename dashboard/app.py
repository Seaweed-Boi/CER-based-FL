"""
Streamlit dashboard for CER-based FL monitoring.

Usage:
    streamlit run dashboard/app.py
"""
import streamlit as st
import pandas as pd
import json
import matplotlib.pyplot as plt
from pathlib import Path


def load_logs(log_file="logs/rounds.jsonl"):
    """Load logs from JSONL file."""
    logs = []
    log_path = Path(log_file)
    
    if not log_path.exists():
        st.error(f"Log file not found: {log_file}")
        return pd.DataFrame()
    
    with open(log_path, "r") as f:
        for line in f:
            try:
                logs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    
    return pd.DataFrame(logs)


def main():
    st.set_page_config(page_title="CER-based FL Dashboard", layout="wide")
    st.title("🔐 CER-based Federated Learning Dashboard")
    st.markdown("Real-time monitoring of FL configurations with CER metrics")
    
    # Load logs
    df = load_logs()
    
    if df.empty:
        st.warning("No training logs found. Run the FL server first.")
        return
    
    # Sidebar controls
    st.sidebar.header("Configuration")
    configs = df["config"].unique().tolist()
    selected_config = st.sidebar.selectbox("Select Configuration", configs)
    show_attack = st.sidebar.checkbox("Show Attack Impact", value=True)
    
    # Filter data
    config_df = df[df["config"] == selected_config]
    
    # Metrics overview
    st.header(f"📊 Configuration: {selected_config}")
    
    if not config_df.empty:
        latest = config_df.iloc[-1]
        
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("F2 Score", f"{latest['F2']:.4f}")
        col2.metric("Resilience", f"{latest['resilience']:.4f}")
        col3.metric("Epsilon", f"{latest['epsilon']:.2f}" if latest['epsilon'] else "N/A")
        col4.metric("Comm (KB)", f"{latest['comm_kb']:.2f}")
        col5.metric("CER Score", f"{latest['CER']:.6f}")
    
    # Plots
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("CER vs Rounds")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(config_df["round"], config_df["CER"], marker="o", linewidth=2, color="#1f77b4")
        ax.set_xlabel("Round")
        ax.set_ylabel("CER Score")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    with col_right:
        st.subheader("F2 Score vs Rounds")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(config_df["round"], config_df["F2"], marker="o", linewidth=2, color="#ff7f0e")
        ax.set_xlabel("Round")
        ax.set_ylabel("F2 Score")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    # Configuration comparison
    st.header("⚖️ Configuration Comparison")
    
    comparison_data = []
    for config in configs:
        c_df = df[df["config"] == config]
        if not c_df.empty:
            final = c_df.iloc[-1]
            comparison_data.append({
                "Config": config,
                "Final F2": f"{final['F2']:.4f}",
                "Resilience": f"{final['resilience']:.4f}",
                "Epsilon": f"{final['epsilon']:.2f}" if final['epsilon'] else "N/A",
                "Comm (KB)": f"{final['comm_kb']:.2f}",
                "Final CER": f"{final['CER']:.6f}"
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True)
    
    # Bar chart comparison
    st.subheader("Final CER Comparison")
    final_cers = {}
    for config in configs:
        c_df = df[df["config"] == config]
        if not c_df.empty:
            final_cers[config] = c_df.iloc[-1]["CER"]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(final_cers.keys(), final_cers.values(), edgecolor="black", alpha=0.7)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    ax.set_ylabel("Final CER Score")
    ax.grid(True, alpha=0.3, axis="y")
    st.pyplot(fig)
    
    # Raw logs
    with st.expander("📋 View Raw Logs"):
        st.dataframe(config_df, use_container_width=True)
    
    # Refresh button
    if st.sidebar.button("🔄 Refresh Data"):
        st.rerun()


if __name__ == "__main__":
    main()
