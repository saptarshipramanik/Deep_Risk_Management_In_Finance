"""
How to Check and Use Saved Models

This script demonstrates how to:
1. Check if a model exists
2. Load a saved model
3. Inspect model details
4. Use the model for predictions
5. Compare different saved models
"""

import torch
from pathlib import Path
from deep_hedging import RecurrentPolicy, Trainer, TrainingConfig
from deep_hedging.evaluation import RiskMetrics
import os

def check_saved_models():
    """List all saved models in the checkpoints directory."""
    checkpoint_dir = Path("checkpoints")
    
    if not checkpoint_dir.exists():
        print("No checkpoints directory found.")
        return []
    
    models = list(checkpoint_dir.glob("*.pt"))
    
    if not models:
        print("No saved models found in checkpoints/")
        return []
    
    print("=" * 70)
    print("SAVED MODELS")
    print("=" * 70)
    
    for i, model_path in enumerate(models, 1):
        file_size = model_path.stat().st_size / 1024  # KB
        print(f"\n{i}. {model_path.name}")
        print(f"   Size: {file_size:.2f} KB")
        print(f"   Path: {model_path}")
    
    return models

def load_and_inspect_model(model_path="checkpoints/quick_start_model.pt"):
    """Load a saved model and display its details."""
    
    if not Path(model_path).exists():
        print(f"Model not found: {model_path}")
        return None
    
    print("\n" + "=" * 70)
    print(f"LOADING MODEL: {model_path}")
    print("=" * 70)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Display checkpoint contents
    print("\nCheckpoint Contents:")
    print(f"  - Keys: {list(checkpoint.keys())}")
    
    if 'config' in checkpoint:
        print("\nTraining Configuration:")
        config_dict = checkpoint['config']
        for key, value in config_dict.items():
            print(f"  {key:20s}: {value}")
    
    if 'history' in checkpoint:
        history = checkpoint['history']
        print("\nTraining History:")
        print(f"  Total epochs: {len(history['loss'])}")
        print(f"  Final loss: {history['loss'][-1]:.4f}")
        print(f"  Final mean P&L: {history['mean_pnl'][-1]:.4f}")
        print(f"  Final std P&L: {history['std_pnl'][-1]:.4f}")
    
    # Count model parameters
    if 'policy_state_dict' in checkpoint:
        state_dict = checkpoint['policy_state_dict']
        total_params = sum(p.numel() for p in state_dict.values())
        print(f"\nModel Parameters: {total_params:,}")
    
    return checkpoint

def use_loaded_model(model_path="checkpoints/quick_start_model.pt"):
    """Load a model and use it for evaluation."""
    
    print("\n" + "=" * 70)
    print("USING LOADED MODEL FOR EVALUATION")
    print("=" * 70)
    
    # Create policy and trainer
    config = TrainingConfig()
    policy = RecurrentPolicy(state_dim=2, action_dim=1)
    trainer = Trainer(policy, config)
    
    # Load the saved model
    trainer.load_checkpoint(Path(model_path).name)
    
    print("\nModel loaded successfully!")
    print("Running evaluation on 10,000 paths...")
    
    # Evaluate
    metrics, pnl, paths, deltas = trainer.evaluate(num_paths=10000)
    
    # Display results
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    RiskMetrics.print_metrics(metrics, "Loaded Model Performance")
    
    return metrics, pnl, paths, deltas

def compare_models():
    """Compare multiple saved models."""
    
    checkpoint_dir = Path("checkpoints")
    models = list(checkpoint_dir.glob("*.pt"))
    
    if len(models) < 2:
        print("Need at least 2 models to compare.")
        return
    
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    
    results = []
    
    for model_path in models:
        checkpoint = torch.load(model_path, map_location='cpu')
        
        if 'history' in checkpoint:
            history = checkpoint['history']
            results.append({
                'Model': model_path.name,
                'Final Loss': f"{history['loss'][-1]:.4f}",
                'Mean P&L': f"{history['mean_pnl'][-1]:.4f}",
                'Std P&L': f"{history['std_pnl'][-1]:.4f}",
                'CVaR': f"{history['cvar'][-1]:.4f}"
            })
    
    # Display comparison
    if results:
        import pandas as pd
        df = pd.DataFrame(results)
        print("\n" + df.to_string(index=False))

def main():
    """Main function to demonstrate all model checking methods."""
    
    print("=" * 70)
    print("MODEL CHECKER - Deep Hedging")
    print("=" * 70)
    
    # 1. List all saved models
    models = check_saved_models()
    
    if not models:
        print("\nNo models found. Train a model first:")
        print("  python quick_start.py")
        return
    
    # 2. Load and inspect the first model
    model_path = models[0]
    checkpoint = load_and_inspect_model(str(model_path))
    
    # 3. Ask user if they want to evaluate
    print("\n" + "=" * 70)
    print("OPTIONS:")
    print("=" * 70)
    print("1. Evaluate this model (recommended)")
    print("2. Compare all models")
    print("3. Exit")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        use_loaded_model(str(model_path))
    elif choice == "2":
        compare_models()
    else:
        print("Exiting.")

if __name__ == "__main__":
    main()
