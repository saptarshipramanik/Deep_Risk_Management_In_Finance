# How to Check Your Saved Model

## Quick Methods

### Method 1: Use the Model Checker Script (Easiest)

I've created a script that does everything for you:

```powershell
python check_model.py
```

This will:
- ✅ List all saved models
- ✅ Show model details (size, configuration, training history)
- ✅ Let you evaluate the model
- ✅ Compare multiple models

### Method 2: Check Manually

**See if the model file exists:**
```powershell
dir checkpoints
```

You should see: `quick_start_model.pt`

**Check file size:**
```powershell
Get-Item checkpoints\quick_start_model.pt | Select-Object Name, Length
```

### Method 3: Load and Use in Python

**Simple check:**
```python
from pathlib import Path

model_path = Path("checkpoints/quick_start_model.pt")
if model_path.exists():
    print(f"✓ Model found!")
    print(f"  Size: {model_path.stat().st_size / 1024:.2f} KB")
else:
    print("✗ Model not found")
```

**Load and inspect:**
```python
import torch

# Load checkpoint
checkpoint = torch.load("checkpoints/quick_start_model.pt", map_location='cpu')

# See what's inside
print("Checkpoint keys:", checkpoint.keys())

# View configuration
print("\nConfiguration:")
for key, value in checkpoint['config'].items():
    print(f"  {key}: {value}")

# View training history
print(f"\nFinal training loss: {checkpoint['history']['loss'][-1]:.4f}")
print(f"Final mean P&L: {checkpoint['history']['mean_pnl'][-1]:.4f}")
```

**Load and use for predictions:**
```python
from deep_hedging import RecurrentPolicy, Trainer, TrainingConfig

# Create trainer
config = TrainingConfig()
policy = RecurrentPolicy(state_dim=2, action_dim=1)
trainer = Trainer(policy, config)

# Load saved model
trainer.load_checkpoint("quick_start_model.pt")

# Evaluate on new data
metrics, pnl, paths, deltas = trainer.evaluate(num_paths=10000)

print("Model performance:")
print(f"  Mean P&L: {metrics['mean']:.4f}")
print(f"  Std P&L: {metrics['std']:.4f}")
print(f"  5% CVaR: {metrics['cvar_5']:.4f}")
```

## What's Inside a Saved Model?

Each `.pt` file contains:

1. **`policy_state_dict`**: Neural network weights and biases
2. **`optimizer_state_dict`**: Optimizer state (for resuming training)
3. **`config`**: All training parameters used
4. **`history`**: Training metrics over time
   - Loss values
   - Mean P&L
   - Standard deviation
   - CVaR

## Common Tasks

### Task 1: Resume Training

```python
from deep_hedging import RecurrentPolicy, Trainer, TrainingConfig

# Load existing model
config = TrainingConfig(num_epochs=1000)  # Train for 1000 more epochs
policy = RecurrentPolicy(state_dim=2, action_dim=1)
trainer = Trainer(policy, config)
trainer.load_checkpoint("quick_start_model.pt")

# Continue training
history = trainer.train()
trainer.save_checkpoint("quick_start_model_extended.pt")
```

### Task 2: Use Model for Hedging

```python
import torch
from deep_hedging import RecurrentPolicy, Trainer, TrainingConfig

# Load model
config = TrainingConfig()
policy = RecurrentPolicy(state_dim=2, action_dim=1)
trainer = Trainer(policy, config)
trainer.load_checkpoint("quick_start_model.pt")

# Get hedging decision for current state
# Example: Stock price = 105, time remaining = 0.5 (50% of maturity)
state = torch.tensor([[105/100, 0.5]])  # Normalized price, time to maturity
policy.eval()
with torch.no_grad():
    delta, _ = policy(state, None)
    print(f"Recommended hedge ratio (delta): {delta.item():.4f}")
```

### Task 3: Compare Models

```python
import torch

models = ["quick_start_model.pt", "model_entropic.pt", "model_cvar.pt"]

for model_name in models:
    checkpoint = torch.load(f"checkpoints/{model_name}", map_location='cpu')
    history = checkpoint['history']
    
    print(f"\n{model_name}:")
    print(f"  Final Loss: {history['loss'][-1]:.4f}")
    print(f"  Mean P&L: {history['mean_pnl'][-1]:.4f}")
    print(f"  Std P&L: {history['std_pnl'][-1]:.4f}")
```

## Troubleshooting

### "FileNotFoundError: checkpoints/quick_start_model.pt"

**Cause**: Model hasn't been saved yet or training didn't complete.

**Solution**: Run training first:
```powershell
python quick_start.py
```

### "RuntimeError: Error loading state_dict"

**Cause**: Model architecture doesn't match saved weights.

**Solution**: Make sure you create the same policy architecture:
```python
# Must match the architecture used during training
policy = RecurrentPolicy(state_dim=2, action_dim=1, hidden_dim=64, num_layers=2)
```

## Quick Reference

| Task | Command |
|------|---------|
| List models | `dir checkpoints` |
| Check model | `python check_model.py` |
| Load model | `trainer.load_checkpoint("model.pt")` |
| Evaluate model | `trainer.evaluate(num_paths=10000)` |
| Resume training | Load, then `trainer.train()` |

---

**Pro Tip**: Always keep your best models! You can compare different configurations and choose the one with the best performance for your specific risk preferences.
