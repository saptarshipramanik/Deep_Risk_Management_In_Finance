# Deep Hedging - Complete Usage Guide

## 📋 Table of Contents
1. [Installation](#installation)
2. [Running the Code - Step by Step](#running-the-code)
3. [Example Workflows](#example-workflows)
4. [GitHub Setup](#github-setup)
5. [Troubleshooting](#troubleshooting)

---

## 1. Installation

### Step 1: Navigate to Project Directory
```bash
cd D:\Studies\DDP
```

### Step 2: Install Dependencies
```bash
# Install all required packages
pip install -r requirements.txt

# Install the deep_hedging package in development mode
pip install -e .
```

**What this does**: Installs PyTorch, NumPy, Matplotlib, and other dependencies, then makes the `deep_hedging` package importable.

---

## 2. Running the Code - Step by Step

### Quick Start (Recommended Order)

#### **Option A: Using Python Scripts**

**Step 1: Create a simple training script**

Create `train_basic.py`:
```python
from deep_hedging import RecurrentPolicy, Trainer, TrainingConfig

# Configure training
config = TrainingConfig(
    num_epochs=1000,        # Start with fewer epochs for testing
    batch_size=1024,
    learning_rate=1e-4,
    policy_type="recurrent",
    loss_type="entropic",
    lambda_risk=1.0,
    transaction_cost=0.001
)

# Initialize policy
policy = RecurrentPolicy(
    state_dim=2,
    action_dim=1,
    hidden_dim=64,
    num_layers=2
)

# Train
trainer = Trainer(policy, config)
history = trainer.train()

# Save model
trainer.save_checkpoint("trained_model.pt")

print("\nTraining complete! Model saved to checkpoints/trained_model.pt")
```

**Run it**:
```bash
python train_basic.py
```

**Step 2: Evaluate the trained model**

Create `evaluate.py`:
```python
from deep_hedging import RecurrentPolicy, Trainer, TrainingConfig
from deep_hedging.evaluation import RiskMetrics, DeltaHedging
from deep_hedging.utils import plot_pnl_comparison
import numpy as np

# Load configuration and policy
config = TrainingConfig()
policy = RecurrentPolicy(state_dim=2, action_dim=1)
trainer = Trainer(policy, config)

# Load trained model
trainer.load_checkpoint("trained_model.pt")

# Evaluate
print("Evaluating deep hedging...")
metrics_deep, pnl_deep, paths, deltas_deep = trainer.evaluate(num_paths=10000)

# Compare with delta hedging
print("\nComputing delta hedging benchmark...")
delta_hedge = DeltaHedging(sigma=0.2)
pnl_delta = delta_hedge.compute_pnl(paths, trainer.option, transaction_cost=0.001)

# Print metrics
print("\n" + "="*60)
RiskMetrics.print_metrics(
    RiskMetrics.compute_all(pnl_deep),
    "Deep Hedging Performance"
)

print("\n" + "="*60)
RiskMetrics.print_metrics(
    RiskMetrics.compute_all(pnl_delta),
    "Delta Hedging Performance"
)

# Visualize
plot_pnl_comparison(pnl_deep, pnl_delta)
```

**Run it**:
```bash
python evaluate.py
```

#### **Option B: Using Jupyter Notebook** (Interactive)

**Step 1: Start Jupyter**
```bash
jupyter notebook
```

**Step 2: Create a new notebook or use existing**

Create `Deep_Hedging_Demo.ipynb` with these cells:

**Cell 1: Imports**
```python
from deep_hedging import RecurrentPolicy, FeedForwardPolicy, Trainer, TrainingConfig
from deep_hedging.evaluation import RiskMetrics, DeltaHedging
from deep_hedging.utils import plot_training_history, plot_pnl_comparison
import torch
import numpy as np
import matplotlib.pyplot as plt
```

**Cell 2: Configure and Train**
```python
# Configuration
config = TrainingConfig(
    num_epochs=1000,
    batch_size=2048,
    learning_rate=1e-4,
    loss_type="entropic",
    lambda_risk=1.0,
    transaction_cost=0.001
)

# Initialize policy
policy = RecurrentPolicy(state_dim=2, action_dim=1, hidden_dim=64)

# Train
trainer = Trainer(policy, config)
history = trainer.train()
```

**Cell 3: Visualize Training**
```python
plot_training_history(history)
```

**Cell 4: Evaluate**
```python
metrics, pnl, paths, deltas = trainer.evaluate(num_paths=10000)
RiskMetrics.print_metrics(metrics, "Deep Hedging")
```

**Cell 5: Compare with Delta Hedging**
```python
delta_hedge = DeltaHedging(sigma=0.2)
pnl_delta = delta_hedge.compute_pnl(paths, trainer.option, transaction_cost=0.001)

plot_pnl_comparison(pnl, pnl_delta)
```

---

## 3. Example Workflows

### Workflow 1: Compare Different Risk Measures

Create `compare_risk_measures.py`:
```python
from deep_hedging import RecurrentPolicy, Trainer, TrainingConfig
from deep_hedging.evaluation import RiskMetrics
import pandas as pd

results = []

for loss_type in ["entropic", "cvar", "variance"]:
    print(f"\n{'='*60}")
    print(f"Training with {loss_type} loss...")
    print('='*60)
    
    config = TrainingConfig(
        num_epochs=1000,
        batch_size=2048,
        loss_type=loss_type,
        transaction_cost=0.001
    )
    
    policy = RecurrentPolicy(state_dim=2, action_dim=1)
    trainer = Trainer(policy, config)
    history = trainer.train()
    
    # Evaluate
    metrics, pnl, _, _ = trainer.evaluate(num_paths=10000)
    
    results.append({
        'Loss Type': loss_type,
        'Mean P&L': metrics['mean'],
        'Std P&L': metrics['std'],
        '5% CVaR': metrics['cvar_5'],
        'Sharpe': metrics['sharpe']
    })
    
    # Save model
    trainer.save_checkpoint(f"model_{loss_type}.pt")

# Display results
df = pd.DataFrame(results)
print("\n" + "="*60)
print("COMPARISON RESULTS")
print("="*60)
print(df.to_string(index=False))
```

### Workflow 2: Test Different Transaction Costs

Create `test_transaction_costs.py`:
```python
from deep_hedging import RecurrentPolicy, Trainer, TrainingConfig
import matplotlib.pyplot as plt
import numpy as np

transaction_costs = [0.0, 0.0005, 0.001, 0.002, 0.005]
mean_pnls = []
std_pnls = []

for tc in transaction_costs:
    print(f"\nTraining with transaction cost: {tc*100:.2f}%")
    
    config = TrainingConfig(
        num_epochs=1000,
        batch_size=2048,
        transaction_cost=tc
    )
    
    policy = RecurrentPolicy(state_dim=2, action_dim=1)
    trainer = Trainer(policy, config)
    history = trainer.train()
    
    metrics, _, _, _ = trainer.evaluate(num_paths=10000)
    mean_pnls.append(metrics['mean'])
    std_pnls.append(metrics['std'])

# Plot results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot([tc*100 for tc in transaction_costs], mean_pnls, marker='o')
ax1.set_xlabel('Transaction Cost (%)')
ax1.set_ylabel('Mean P&L')
ax1.set_title('Impact of Transaction Costs on Mean P&L')
ax1.grid(True, alpha=0.3)

ax2.plot([tc*100 for tc in transaction_costs], std_pnls, marker='o')
ax2.set_xlabel('Transaction Cost (%)')
ax2.set_ylabel('Std P&L')
ax2.set_title('Impact of Transaction Costs on P&L Volatility')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('transaction_cost_analysis.png', dpi=300)
plt.show()
```

### Workflow 3: Feedforward vs LSTM Comparison

Create `compare_architectures.py`:
```python
from deep_hedging import RecurrentPolicy, FeedForwardPolicy, Trainer, TrainingConfig
from deep_hedging.evaluation import RiskMetrics

# Train Feedforward
print("Training Feedforward Policy...")
config = TrainingConfig(num_epochs=1000, batch_size=2048)
ff_policy = FeedForwardPolicy(state_dim=2, action_dim=1, hidden_dims=[64, 64])
ff_trainer = Trainer(ff_policy, config)
ff_history = ff_trainer.train()
ff_metrics, ff_pnl, _, _ = ff_trainer.evaluate(num_paths=10000)

# Train LSTM
print("\nTraining LSTM Policy...")
lstm_policy = RecurrentPolicy(state_dim=2, action_dim=1, hidden_dim=64)
lstm_trainer = Trainer(lstm_policy, config)
lstm_history = lstm_trainer.train()
lstm_metrics, lstm_pnl, _, _ = lstm_trainer.evaluate(num_paths=10000)

# Compare
print("\n" + "="*60)
print("ARCHITECTURE COMPARISON")
print("="*60)
print(f"\nFeedforward Policy:")
print(f"  Mean P&L: {ff_metrics['mean']:.4f}")
print(f"  Std P&L: {ff_metrics['std']:.4f}")
print(f"  5% CVaR: {ff_metrics['cvar_5']:.4f}")
print(f"  Sharpe: {ff_metrics['sharpe']:.4f}")

print(f"\nLSTM Policy:")
print(f"  Mean P&L: {lstm_metrics['mean']:.4f}")
print(f"  Std P&L: {lstm_metrics['std']:.4f}")
print(f"  5% CVaR: {lstm_metrics['cvar_5']:.4f}")
print(f"  Sharpe: {lstm_metrics['sharpe']:.4f}")
```

---

## 4. Execution Order Summary

### For First-Time Users:

1. **Install** (once):
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

2. **Test basic training** (5-10 minutes):
   ```bash
   python train_basic.py
   ```

3. **Evaluate results**:
   ```bash
   python evaluate.py
   ```

4. **Run experiments** (optional):
   ```bash
   python compare_risk_measures.py
   python test_transaction_costs.py
   python compare_architectures.py
   ```

### For Jupyter Users:

1. **Start Jupyter**:
   ```bash
   jupyter notebook
   ```

2. **Open/Create notebook** and run cells sequentially

3. **Experiment interactively**

---

## 5. Expected Output

### Training Output:
```
Training RecurrentPolicy for 1000 epochs
Batch size: 2048, Learning rate: 0.0001
Loss function: entropic
----------------------------------------------------------------------
Training: 100%|████████████| 1000/1000 [05:23<00:00, 3.09it/s, loss=1.8234, mean_pnl=-1.23, std_pnl=2.56]

Training completed!
```

### Evaluation Output:
```
Risk Metrics
==================================================

Central Tendency:
  mean        :    -1.2345
  median      :    -0.8912

Dispersion:
  std         :     2.5678
  var         :     6.5936

Risk Measures:
  cvar_1      :     5.6789
  cvar_5      :     4.1234
  cvar_10     :     3.4567

Performance:
  sharpe      :    -0.4807
  sortino     :    -0.3456
```

---

## 6. Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'deep_hedging'"
**Solution**: Run `pip install -e .` from the DDP directory

### Issue: "CUDA out of memory"
**Solution**: Reduce batch size in config:
```python
config = TrainingConfig(batch_size=512)  # Instead of 2048
```

### Issue: Training is slow
**Solution**: 
- Reduce num_epochs for testing
- Use GPU if available (PyTorch will auto-detect)
- Reduce batch_size if memory is an issue

### Issue: "No module named 'torch'"
**Solution**: Install PyTorch:
```bash
pip install torch
```

---

## 7. Next Steps

After running the basic examples:

1. **Modify configurations** to experiment with different parameters
2. **Add custom market models** by extending `BaseSimulator`
3. **Implement new risk measures** in `loss_functions.py`
4. **Create custom visualizations** using the utilities
5. **Integrate with real market data**

---

## 8. File Locations

- **Trained models**: `checkpoints/`
- **Plots**: Current directory (or specify path)
- **Logs**: Console output (can redirect to file)
- **Source code**: `deep_hedging/`

---

**Happy Hedging! 🚀**
