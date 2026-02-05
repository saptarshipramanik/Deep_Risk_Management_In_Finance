"""
Example: Compare Deep Hedging with Delta Hedging

This script trains a deep hedging model and compares its performance
with classical Black-Scholes delta hedging.
"""

from deep_hedging import RecurrentPolicy, Trainer, TrainingConfig
from deep_hedging.evaluation import RiskMetrics, DeltaHedging
from deep_hedging.utils import plot_pnl_comparison, plot_delta_comparison
import numpy as np

def main():
    print("="*70)
    print("Deep Hedging vs Delta Hedging Comparison")
    print("="*70)
    
    # Configuration
    config = TrainingConfig(
        num_epochs=1000,
        batch_size=2048,
        learning_rate=1e-4,
        loss_type="entropic",
        lambda_risk=1.0,
        transaction_cost=0.001,
        sigma=0.2,
        log_interval=200
    )
    
    # Train deep hedging
    print("\n[1/3] Training Deep Hedging model...")
    print("-"*70)
    policy = RecurrentPolicy(state_dim=2, action_dim=1, hidden_dim=64)
    trainer = Trainer(policy, config)
    history = trainer.train()
    
    # Evaluate deep hedging
    print("\n[2/3] Evaluating Deep Hedging...")
    metrics_deep, pnl_deep, paths, deltas_deep = trainer.evaluate(num_paths=10000)
    
    # Evaluate delta hedging
    print("\n[3/3] Evaluating Delta Hedging...")
    delta_hedge = DeltaHedging(sigma=config.sigma)
    pnl_delta = delta_hedge.compute_pnl(
        paths, 
        trainer.option, 
        transaction_cost=config.transaction_cost
    )
    metrics_delta = RiskMetrics.compute_all(pnl_delta)
    
    # Print comparison
    print("\n" + "="*70)
    print("PERFORMANCE COMPARISON")
    print("="*70)
    
    print("\nDeep Hedging:")
    print(f"  Mean P&L:    {metrics_deep['mean']:10.4f}")
    print(f"  Std P&L:     {metrics_deep['std']:10.4f}")
    print(f"  5% CVaR:     {metrics_deep['cvar_5']:10.4f}")
    print(f"  Sharpe:      {metrics_deep['sharpe']:10.4f}")
    
    print("\nDelta Hedging:")
    print(f"  Mean P&L:    {metrics_delta['mean']:10.4f}")
    print(f"  Std P&L:     {metrics_delta['std']:10.4f}")
    print(f"  5% CVaR:     {metrics_delta['cvar_5']:10.4f}")
    print(f"  Sharpe:      {metrics_delta['sharpe']:10.4f}")
    
    print("\nImprovement:")
    print(f"  Mean P&L:    {((metrics_deep['mean'] - metrics_delta['mean'])):+10.4f}")
    print(f"  Std P&L:     {((metrics_deep['std'] - metrics_delta['std'])):+10.4f}")
    print(f"  5% CVaR:     {((metrics_deep['cvar_5'] - metrics_delta['cvar_5'])):+10.4f}")
    
    # Visualizations
    print("\n" + "="*70)
    print("Generating visualizations...")
    print("="*70)
    
    # P&L comparison
    plot_pnl_comparison(pnl_deep, pnl_delta)
    
    # Delta comparison for sample path
    t = np.linspace(0, config.T, config.N + 1)
    deltas_bs = delta_hedge.hedge(paths, t, trainer.option)
    plot_delta_comparison(paths, deltas_deep, deltas_bs, path_idx=0)
    
    print("\nComparison complete!")

if __name__ == "__main__":
    main()
