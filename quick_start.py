"""
Quick Start Example: Train a basic Deep Hedging model

This script demonstrates the simplest way to train and evaluate
a deep hedging model.
"""

from deep_hedging import RecurrentPolicy, Trainer, TrainingConfig
from deep_hedging.evaluation import RiskMetrics

def main():
    print("="*70)
    print("Deep Hedging - Quick Start Example")
    print("="*70)
    
    # Step 1: Configure training
    print("\n[1/4] Configuring training...")
    config = TrainingConfig(
        num_epochs=500,          # Reduced for quick demo
        batch_size=1024,
        learning_rate=1e-4,
        policy_type="recurrent",
        loss_type="entropic",
        lambda_risk=1.0,
        transaction_cost=0.001,  # 0.1% transaction cost
        log_interval=100
    )
    print(f"  - Epochs: {config.num_epochs}")
    print(f"  - Batch size: {config.batch_size}")
    print(f"  - Loss type: {config.loss_type}")
    print(f"  - Transaction cost: {config.transaction_cost*100:.2f}%")
    
    # Step 2: Initialize policy
    print("\n[2/4] Initializing LSTM policy...")
    policy = RecurrentPolicy(
        state_dim=2,      # (normalized_price, time_to_maturity)
        action_dim=1,     # delta
        hidden_dim=64,
        num_layers=2
    )
    print(f"  - Architecture: LSTM")
    print(f"  - Hidden dim: 64")
    print(f"  - Num layers: 2")
    print(f"  - Parameters: {sum(p.numel() for p in policy.parameters()):,}")
    
    # Step 3: Train
    print("\n[3/4] Training model...")
    print("-"*70)
    trainer = Trainer(policy, config)
    history = trainer.train()
    
    # Save model
    trainer.save_checkpoint("quick_start_model.pt")
    print(f"\nModel saved to: checkpoints/quick_start_model.pt")
    
    # Step 4: Evaluate
    print("\n[4/4] Evaluating model...")
    print("-"*70)
    metrics, pnl, paths, deltas = trainer.evaluate(num_paths=10000)
    
    # Print results
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    RiskMetrics.print_metrics(metrics, "Deep Hedging Performance")
    
    print("\n" + "="*70)
    print("Quick Start Complete!")
    print("="*70)
    print("\nNext steps:")
    print("  1. Check 'checkpoints/quick_start_model.pt' for saved model")
    print("  2. Run 'python examples/compare_with_delta.py' for benchmarking")
    print("  3. See USAGE_GUIDE.md for more examples")

if __name__ == "__main__":
    main()
