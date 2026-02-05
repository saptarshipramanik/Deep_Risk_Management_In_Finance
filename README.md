# Deep Hedging: Production-Ready Implementation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-ready implementation of Deep Hedging using deep reinforcement learning for optimal derivative hedging under transaction costs and risk constraints.

## 📚 Reference

This implementation is based on the paper:
> Buehler, H., Gonon, L., Teichmann, J., & Wood, B. (2019). Deep hedging. Quantitative Finance, 19(8), 1271-1291.

## 🎯 Features

- **Multiple Policy Architectures**: Feedforward and LSTM networks
- **Market Models**: GBM and Heston stochastic volatility
- **Risk Measures**: Entropic risk, CVaR, mean-variance
- **Transaction Costs**: Integrated proportional costs
- **Benchmarks**: Delta hedging and static strategies
- **Production-Ready**: Modular, documented, tested

## 📦 Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## 🚀 Quick Start

```python
from deep_hedging import RecurrentPolicy, Trainer, TrainingConfig

config = TrainingConfig(num_epochs=2000, batch_size=2048)
policy = RecurrentPolicy(state_dim=2, action_dim=1)
trainer = Trainer(policy, config)
history = trainer.train()
```

## 📊 Project Structure

```
deep_hedging/
├── models/       # Neural network policies
├── market/       # Market simulation
├── training/     # Training infrastructure
├── evaluation/   # Metrics and benchmarks
└── utils/        # Visualization
```

## 📓 Notebooks

- `Deep_Hedging_Demo.ipynb` - Comprehensive demonstration
- `Deep_Risk_Management_DDP.ipynb` - Original implementation

## 📖 Documentation

See `docs/technical_report.md` for complete documentation.

## 📄 License

MIT License
