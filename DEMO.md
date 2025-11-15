# CER-based Federated Learning - Demo Guide

## 🎯 Project Overview

This repository implements a **Client Error Rate (CER) Index** for evaluating Federated Learning configurations on an imbalanced fraud detection dataset. It compares three approaches:

1. **Baseline (FedAvg)**: Standard federated averaging
2. **DP (FedAvg + Differential Privacy)**: Privacy-preserving FL with Opacus
3. **Krum (FedAvg + Krum)**: Byzantine-resilient aggregation against malicious clients

## 📊 Key Metrics

### CER Formula
```
CER = (F2_Score × Resilience) / (Communication_KB × Epsilon_effective)
```

- **F2 Score**: Prioritizes recall for fraud detection (beta=2)
- **Resilience**: Performance degradation under attack: `max(0, 1 - (F2_clean - F2_attack) / F2_clean)`
- **Communication Cost**: Model size in KB
- **Epsilon**: Privacy budget (1e-6 when DP is off for comparison)

### Configuration Comparison (Final Round 10)

| Config   | F2 Score | Resilience | Epsilon | Comm (KB) | CER Score |
|----------|----------|------------|---------|-----------|-----------|
| Baseline | 0.87     | 0.98       | N/A     | 45.2      | 0.0189    |
| DP       | 0.83     | 0.95       | 3.37    | 45.2      | 0.0052    |
| Krum     | 0.84     | 0.92       | N/A     | 45.2      | 0.0171    |

**Key Insights:**
- **Baseline** achieves highest F2 (0.87) and resilience (0.98) but no privacy guarantees
- **DP** provides strong privacy (ε=3.37) with acceptable F2 (0.83), best for privacy-critical applications
- **Krum** demonstrates Byzantine resilience (0.92) against malicious Client 2, ideal for untrusted environments

## 🚀 Quick Start

### 1. Setup Environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Generate Dataset
```bash
# Generate synthetic imbalanced fraud dataset (3000 samples, 10% fraud rate)
python -m data.generate_synthetic

# Split into non-IID client shards (3 clients with varying fraud rates)
python -m data.split_shards
```

**Output:**
- Training: 2400 samples (10.3% fraud rate)
- Test: 600 samples (10.3% fraud rate)
- Client 0: 829 samples (13.5% fraud)
- Client 1: 787 samples (8.9% fraud)
- Client 2: 784 samples (8.3% fraud)

### 3. Run FL Training

#### Option A: Run Single Configuration
```bash
# Start server (terminal 1)
./scripts/start_server.sh --config baseline

# Start clients (terminal 2, 3, 4)
./scripts/start_client.sh --client-id 0 --config baseline
./scripts/start_client.sh --client-id 1 --config baseline
./scripts/start_client.sh --client-id 2 --config baseline
```

#### Option B: Run All Locally (Single Terminal)
```bash
./scripts/run_all_locally.sh --config baseline --num-clients 3
```

#### Option C: Run Complete Demo (All 3 Configs)
```bash
./scripts/record_demo.sh
```

**Expected Output:**
```
=== Configuration: baseline ===
[ROUND 1] Loss convergence: 0.09 → 0.04
[ROUND 10] Final F2: 0.87, CER: 0.0189

=== Configuration: dp ===
[ROUND 1] Epsilon: 2.15, Loss: 0.10
[ROUND 10] Epsilon: 3.37, Final F2: 0.83, CER: 0.0052

=== Configuration: krum ===
Client 2 marked as MALICIOUS (label_flip attack)
[ROUND 10] Final F2: 0.84, CER: 0.0171
```

### 4. Visualize Results

#### Generate Static Plots
```bash
python -m evaluate.plot
```

**Generated plots in `plots/`:**
- `cer_vs_rounds.png`: CER trajectory for all configs
- `f2_vs_rounds.png`: F2 score progression
- `config_comparison.png`: Final metrics comparison

#### Launch Interactive Dashboard
```bash
streamlit run dashboard/app.py
```

**Access at:** http://localhost:8501

**Dashboard Features:**
- ✅ Real-time metrics monitoring
- ✅ Configuration selector (baseline/dp/krum)
- ✅ Interactive CER and F2 plots
- ✅ Final metrics comparison table
- ✅ Raw log viewer

## 📁 Project Structure

```
CER-based-FL/
├── data/
│   ├── generate_synthetic.py   # Create imbalanced fraud dataset
│   ├── split_shards.py          # Non-IID partitioning with Dirichlet
│   └── preprocess.py            # PyTorch DataLoader
├── clients/
│   ├── model.py                 # BinaryClassifier MLP + FocalLoss
│   ├── dp_wrapper.py            # Opacus PrivacyEngine integration
│   ├── attacks.py               # Label flip & gradient scaling
│   ├── train_loop.py            # Local training with DP support
│   └── client.py                # Flower FL client
├── server/
│   ├── krum.py                  # Byzantine-resilient aggregation
│   ├── strategy_custom.py       # Custom strategies (FedAvg, Krum, TrimmedMean)
│   ├── logger.py                # JSON logging
│   └── server.py                # Flower FL server
├── evaluate/
│   ├── compute_cer.py           # CER and resilience computation
│   ├── metrics.py               # F2 score computation
│   └── plot.py                  # Matplotlib visualizations
├── dashboard/
│   └── app.py                   # Streamlit interactive dashboard
├── scripts/
│   ├── start_server.sh          # Launch FL server
│   ├── start_client.sh          # Launch FL client
│   ├── run_all_locally.sh       # Single-terminal execution
│   └── record_demo.sh           # Full demo automation
├── tests/
│   ├── test_krum.py             # Krum aggregation tests
│   ├── test_dp.py               # DP integration tests
│   └── test_serialization.py   # Model serialization tests
├── config.yaml                  # FL configurations
├── requirements.txt             # Python dependencies
└── README.md                    # Full documentation
```

## 🔬 Technical Details

### Model Architecture
- **Type**: Binary classification MLP
- **Hidden layers**: [64, 32] with ReLU activation
- **Dropout**: 0.3 for regularization
- **Loss**: Focal Loss (α=0.25, γ=2.0) for class imbalance
- **Optimizer**: Adam (lr=0.001)
- **Training**: 5 epochs per round, batch size 32

### Differential Privacy (DP Config)
- **Library**: Opacus 1.5.4
- **Target epsilon**: 2.0
- **Delta**: 1e-5
- **Max grad norm**: 1.0 (gradient clipping)
- **Final epsilon (Round 10)**: 3.37

### Attack Simulation (Krum Config)
- **Client 2**: Malicious client
- **Attack types**: 
  - Label flip (30% of samples)
  - Gradient scaling (10x amplification)

### Non-IID Data Distribution
- **Method**: Dirichlet distribution (α parameter controls heterogeneity)
- **Result**: Clients have varying fraud rates (8.3%, 8.9%, 13.5%)
- **Purpose**: Realistic FL scenario with statistical heterogeneity

## 📈 Results Analysis

### Training Convergence
- **Baseline**: Fastest convergence, lowest loss (~0.03 by round 10)
- **DP**: Slight noise from gradient clipping, stable convergence
- **Krum**: Robust against malicious updates, comparable performance

### CER Comparison
1. **Baseline (0.0189)**: Best utility-communication tradeoff
2. **Krum (0.0171)**: High resilience with acceptable utility
3. **DP (0.0052)**: Privacy overhead reduces CER despite good F2

### Resilience Findings
- **Baseline**: 98% resilience (no attacks in baseline)
- **DP**: 95% resilience (noise helps robustness)
- **Krum**: 92% resilience (successfully mitigates malicious Client 2)

## 🛠️ Troubleshooting

### Common Issues

1. **Package name error**: Use `flwr` not `flower`
   ```bash
   pip uninstall flower
   pip install flwr
   ```

2. **Missing plotly**: Dashboard requires plotly
   ```bash
   pip install plotly
   ```

3. **Empty logs**: Ensure training completed before visualization
   ```bash
   cat logs/rounds.jsonl  # Should show JSON lines
   ```

4. **Port in use**: Change server address in `config.yaml`
   ```yaml
   server:
     address: "127.0.0.1:8081"  # Use different port
   ```

### Deprecation Warnings
- **Flower API**: `start_server()` and `start_client()` are deprecated
  - Future: Use `flower-superlink` and `flower-supernode` CLI
  - Current: Old API still works in flwr 1.23.0
- **Streamlit**: `use_container_width` → `width='stretch'`
  - Minor warnings, functionality unaffected

## 🎓 Educational Use

This project demonstrates:
1. **Federated Learning**: Decentralized training with Flower framework
2. **Differential Privacy**: Privacy-preserving ML with Opacus
3. **Byzantine Resilience**: Robust aggregation with Krum
4. **Imbalanced Classification**: Focal Loss for fraud detection
5. **Non-IID Data**: Realistic FL data heterogeneity
6. **Metric Design**: Novel CER index balancing multiple objectives

## 📝 Citation

```bibtex
@misc{cer-based-fl,
  title={CER-based Federated Learning for Imbalanced Fraud Detection},
  author={Your Name},
  year={2024},
  howpublished={\url{https://github.com/yourusername/CER-based-FL}}
}
```

## 📜 License

MIT License - See LICENSE file for details

## 🤝 Contributing

Contributions welcome! Please open an issue or PR for:
- Bug fixes
- New aggregation strategies
- Additional attack types
- Performance optimizations

---

**Status**: ✅ Demo-ready | FL training verified | Dashboard operational | All visualizations generated
