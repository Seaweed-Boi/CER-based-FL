# CER-based Federated Learning

A hackathon-ready implementation of **Communication-Efficient and Robust (CER) Federated Learning** for fraud detection with three FL configurations: FedAvg baseline, FedAvg+DP, and FedAvg+Krum.

## 📋 Overview

This project implements a novel **CER Index** metric that balances:
- **Performance**: F2 score (emphasizes recall for fraud detection)
- **Robustness**: Resilience against malicious clients  
- **Communication Efficiency**: Model size in KB
- **Privacy**: Differential privacy budget (epsilon)

**CER Formula**: `CER = (F2 * Resilience) / (comm_kb * epsilon)`

## 🎯 Features

- ✅ **Three FL Configurations**:
  - `baseline`: FedAvg aggregation without DP
  - `dp`: FedAvg with Differential Privacy (Opacus)
  - `krum`: Byzantine-resilient Krum aggregation
  
- ✅ **Synthetic Fraud Dataset**: Imbalanced binary classification with 10% fraud rate
- ✅ **Non-IID Data Split**: Varying fraud rates across clients using Dirichlet
- ✅ **Attack Simulation**: Label-flip attack (30% ratio) on malicious clients
- ✅ **Privacy Protection**: Opacus DP with ε=2.0, δ=1e-5
- ✅ **Focal Loss**: Handles class imbalance (α=0.25, γ=2.0)
- ✅ **Real-time Dashboard**: Streamlit visualization with CER tracking
- ✅ **Automated Testing**: Krum, DP, serialization tests

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repo-url>
cd CER-based-FL

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation

```bash
# Generate synthetic fraud dataset
python -m data.generate_synthetic

# Split into non-IID client shards
python -m data.split_shards
```

### 3. Run FL Training

**Option A: Quick Demo (All Configurations)**

```bash
chmod +x scripts/*.sh
./scripts/record_demo.sh
```

This runs all three configurations sequentially and generates plots.

**Option B: Single Configuration**

```bash
# Terminal 1: Start server
./scripts/start_server.sh --config baseline

# Terminal 2-4: Start clients
./scripts/start_client.sh --client-id 0 --config baseline
./scripts/start_client.sh --client-id 1 --config baseline
./scripts/start_client.sh --client-id 2 --config baseline
```

**Option C: Local Simulation**

```bash
# Run server + 3 clients locally
./scripts/run_all_locally.sh --config dp --num-clients 3
```

### 4. Visualize Results

**Generate Plots**:
```bash
python -m evaluate.plot
```

**Launch Dashboard**:
```bash
streamlit run dashboard/app.py
```

## 📊 Project Structure

```
CER-based-FL/
├── config.yaml              # FL configurations (baseline, dp, krum)
├── requirements.txt         # Dependencies
│
├── data/
│   ├── generate_synthetic.py  # Create fraud dataset
│   ├── split_shards.py         # Non-IID client splitting
│   └── preprocess.py           # PyTorch Dataset
│
├── clients/
│   ├── model.py                # BinaryClassifier + FocalLoss
│   ├── dp_wrapper.py           # Opacus PrivacyEngine
│   ├── attacks.py              # Label-flip & gradient scaling
│   ├── train_loop.py           # Local training with DP/attacks
│   └── client.py               # Flower FL client
│
├── server/
│   ├── krum.py                 # Byzantine-resilient aggregation
│   ├── strategy_custom.py      # Custom FL strategies
│   ├── logger.py               # JSON logging
│   └── server.py               # Flower FL server
│
├── evaluate/
│   ├── compute_cer.py          # CER and resilience computation
│   ├── metrics.py              # F2 score calculation
│   └── plot.py                 # Visualization utilities
│
├── dashboard/
│   └── app.py                  # Streamlit dashboard
│
├── scripts/
│   ├── start_server.sh         # Start FL server
│   ├── start_client.sh         # Start FL client
│   ├── run_all_locally.sh      # Run server + clients
│   └── record_demo.sh          # Full demo run
│
└── tests/
    ├── test_krum.py            # Krum aggregation tests
    ├── test_dp.py              # DP wrapper tests
    └── test_serialization.py   # Comm cost tests
```

## 🔧 Configuration

Edit `config.yaml` to customize:

```yaml
active_config: "baseline"  # Choose: baseline, dp, krum

configs:
  baseline:
    agg_method: "fedavg"
    use_dp: false
    attack_clients: [2]  # Client 2 is malicious
    
  dp:
    agg_method: "fedavg"
    use_dp: true
    dp_params:
      target_epsilon: 2.0
      target_delta: 1.0e-05
      max_grad_norm: 1.0
      
  krum:
    agg_method: "krum"
    use_dp: false
    attack_clients: [2]
    num_malicious: 1
```

## 🧪 Testing

Run all tests:
```bash
pytest tests/
```

Individual test suites:
```bash
python tests/test_krum.py          # Outlier rejection
python tests/test_dp.py            # Epsilon tracking
python tests/test_serialization.py # Comm cost
```

## 📈 Metrics

### CER Index
```
CER = (F2 * Resilience) / (comm_kb * epsilon_effective)
```

Where:
- **F2 Score**: `fbeta_score(beta=2)` - emphasizes recall
- **Resilience**: `max(0, 1 - (F2_clean - F2_attack) / F2_clean)`
- **comm_kb**: Model size in KB
- **epsilon_effective**: DP budget (1e-6 if DP off)

### JSON Log Schema
```json
{
  "round": 3,
  "config": "dp",
  "F2": 0.8234,
  "epsilon": 1.85,
  "comm_kb": 5.12,
  "resilience": 0.91,
  "CER": 0.1621
}
```

## 📊 Expected Results

After running `record_demo.sh`, expect:

- **Baseline**: Highest F2 but low resilience (vulnerable to attacks)
- **DP**: Lower F2 but improved privacy (ε ≈ 2.0)
- **Krum**: Good resilience against Byzantine attacks, moderate F2

Check `plots/config_comparison.png` for final CER comparison!

## 🐛 Troubleshooting

**Import errors**:
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

**Opacus errors**:
```bash
pip install --upgrade opacus
```

**Streamlit dashboard empty**:
- Ensure `logs/rounds.jsonl` exists
- Run training first: `./scripts/record_demo.sh`

## 📚 References

- **Flower**: Federated Learning Framework
- **Opacus**: PyTorch Differential Privacy
- **Krum**: Byzantine-resilient aggregation (Blanchard et al., 2017)
- **Focal Loss**: Handles class imbalance (Lin et al., 2017)

## 📝 License

MIT License - See LICENSE file for details

---

**Happy Hacking! 🚀**