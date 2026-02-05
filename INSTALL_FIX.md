# Installation Fix for Windows Long Path Error

## The Problem

You encountered this error:
```
OSError: [Errno 2] No such file or directory: 'C:\\Users\\Saptarshi Pramanik\\AppData\\Local\\Programs\\Python\\Python311\\share\\jupyter\\...'
HINT: This error might have occurred since this system does not have Windows Long Path support enabled.
```

This happens because Windows has a 260-character path limit, and Jupyter's installation exceeds this.

## Solution: Install Without Jupyter (Recommended)

I've updated `requirements.txt` to remove Jupyter and other optional packages. The core deep hedging functionality doesn't need them.

### Step 1: Install Core Dependencies

```powershell
cd D:\Studies\DDP
pip install -r requirements.txt
```

This will install only:
- PyTorch (deep learning)
- NumPy (numerical computing)
- SciPy (scientific computing)
- Matplotlib (plotting)
- tqdm (progress bars)
- pandas (data analysis)

### Step 2: Install the Package

```powershell
pip install -e .
```

### Step 3: Test Installation

```powershell
python -c "from deep_hedging import RecurrentPolicy; print('Success!')"
```

If you see "Success!", you're ready to go!

## If You Need Jupyter (Optional)

### Option 1: Enable Windows Long Paths

1. Press `Win + R`, type `regedit`, press Enter
2. Navigate to: `HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem`
3. Find `LongPathsEnabled`, double-click, set value to `1`
4. Restart your computer
5. Then install Jupyter:
   ```powershell
   pip install notebook
   ```

### Option 2: Use Jupyter Without Installing

Use Google Colab (free, online):
1. Go to https://colab.research.google.com/
2. Upload your code
3. Install dependencies in a cell:
   ```python
   !pip install torch numpy scipy matplotlib tqdm pandas
   ```

### Option 3: Use VS Code with Python Extension

VS Code has an interactive Python mode that works like Jupyter:
1. Install VS Code: https://code.visualstudio.com/
2. Install Python extension
3. Create `.py` files and run cells with `# %%`

## Running the Code Without Jupyter

You can run everything using Python scripts:

### Quick Start:
```powershell
python quick_start.py
```

### Examples:
```powershell
python examples\compare_with_delta.py
python examples\compare_risk_measures.py
```

### Interactive Python:
```powershell
python
```
Then type:
```python
from deep_hedging import RecurrentPolicy, Trainer, TrainingConfig
config = TrainingConfig(num_epochs=500)
policy = RecurrentPolicy(state_dim=2, action_dim=1)
trainer = Trainer(policy, config)
history = trainer.train()
```

## Verification

After installation, verify everything works:

```powershell
# Test 1: Import check
python -c "from deep_hedging import RecurrentPolicy, Trainer, TrainingConfig; print('Imports OK')"

# Test 2: Quick training
python quick_start.py
```

## Summary

✅ **Core installation works without Jupyter**
✅ **All deep hedging functionality is available**
✅ **Use Python scripts instead of notebooks**
✅ **Jupyter is optional, not required**

You can do everything the framework offers using regular Python scripts!
