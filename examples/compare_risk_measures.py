"""
Example: Experiment with different risk measures

This script trains models with different risk measures and compares
their performance.
"""

from deep_hedging import RecurrentPolicy, Trainer, TrainingConfig
from deep_hedging.evaluation import RiskMetrics
import pandas as pd

def train_with_loss(loss_type, config_params):
    """Train a model with specified loss function."""
    config = TrainingConfig(
        loss_type=loss_type,
        **config_params
    )
    
    policy = RecurrentPolicy(state_dim=2, action_dim=1, hidden_dim=64)
    trainer = Trainer(policy, config)
    history = trainer.train()
    
    metrics, pnl, _, _ = trainer.evaluate(num_paths=10000)
    
    # Save model
    trainer.save_checkpoint(f"model_{loss_type}.pt")
    
    return metrics, pnl

def main():
    print("="*70)
    print("Risk Measure Comparison Experiment")
    print("="*70)
    
    # Common configuration
    config_params = {
        'num_epochs': 1000,
        'batch_size': 2048,
        'learning_rate': 1e-4,
        'transaction_cost': 0.001,
        'log_interval': 200
    }
    
    results = []
    
    # Test different risk measures
    risk_measures = ["entropic", "cvar", "variance"]
    
    for i, loss_type in enumerate(risk_measures, 1):
        print(f"\n[{i}/{len(risk_measures)}] Training with {loss_type.upper()} loss...")
        print("-"*70)
        
        metrics, pnl = train_with_loss(loss_type, config_params)
        
        results.append({
            'Loss Type': loss_type.capitalize(),
            'Mean P&L': f"{metrics['mean']:.4f}",
            'Std P&L': f"{metrics['std']:.4f}",
            '5% CVaR': f"{metrics['cvar_5']:.4f}",
            'Sharpe': f"{metrics['sharpe']:.4f}",
            'Sortino': f"{metrics['sortino']:.4f}"
        })
    
    # Display results
    print("\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70)
    
    df = pd.DataFrame(results)
    print("\n" + df.to_string(index=False))
    
    print("\n" + "="*70)
    print("Models saved to checkpoints/")
    print("="*70)
    print("\nKey Insights:")
    print("  - Entropic: Balances mean and variance")
    print("  - CVaR: Focuses on tail risk")
    print("  - Variance: Minimizes P&L volatility")

if __name__ == "__main__":
    main()
