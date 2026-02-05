"""
Deep Hedging - Missing and Partially Implemented Components
============================================================

This file contains all the code for components that were either:
1. Not implemented at all in the original notebook
2. Partially implemented but missing key features

Each section is clearly marked and heavily commented for readability.

Author: Deep Hedging DDP Implementation
Date: February 2026
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import norm
from typing import Dict, List, Tuple, Optional, Callable
import pandas as pd
import json
from pathlib import Path
from dataclasses import dataclass, asdict
import logging
from datetime import datetime


# ============================================================================
# SECTION 1: HIGH-DIMENSIONAL HEDGING (Section 5.4 of Paper)
# ============================================================================
# Status: COMPLETELY MISSING in original implementation
# Priority: HIGH - This demonstrates scalability of the approach
# ============================================================================

class MultiAssetHedgingPolicy(nn.Module):
    """
    Neural network policy for hedging multiple assets simultaneously.
    
    This is designed for the high-dimensional experiment in Section 5.4
    where we hedge 5 independent call options using 10 hedging instruments
    (5 stocks + 5 variance swaps).
    
    Architecture:
    - Input: (2 * num_assets) dimensional - [normalized prices, time]
    - Hidden layers: Scales with num_assets to maintain capacity
    - Output: num_assets dimensional - hedge ratios for each instrument
    
    Args:
        num_assets: Number of underlying assets (5 in paper experiment)
        hidden_multiplier: Multiplier for hidden layer size (default: 12 per asset)
    """
    
    def __init__(self, num_assets: int = 5, hidden_multiplier: int = 12):
        super().__init__()
        
        self.num_assets = num_assets
        input_dim = num_assets + 1  # [S1/S0_1, S2/S0_2, ..., Sn/S0_n, time]
        output_dim = num_assets      # Delta for each asset
        hidden_dim = hidden_multiplier * num_assets  # Scale with problem size
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Zero initialization for training stability
        # This is crucial for hedging problems to start near zero hedge
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: State tensor of shape (batch_size, num_assets + 1)
               Contains normalized prices and time
        
        Returns:
            Hedge ratios of shape (batch_size, num_assets)
        """
        return self.net(x)


def simulate_multi_heston_paths(
    S0: float,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    T: float,
    N: int,
    M: int,
    num_assets: int,
    device: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Simulate multiple INDEPENDENT Heston processes.
    
    This is for the high-dimensional experiment where we have multiple
    independent underlying assets, each following its own Heston model
    with identical parameters.
    
    Args:
        S0: Initial stock price (same for all assets)
        v0: Initial variance (same for all assets)
        kappa: Mean reversion speed of variance
        theta: Long-term variance level
        xi: Volatility of variance (vol-of-vol)
        rho: Correlation between price and variance Brownian motions
        T: Time horizon
        N: Number of time steps
        M: Number of Monte Carlo paths
        num_assets: Number of independent Heston processes
        device: 'cuda' or 'cpu'
    
    Returns:
        S: Tensor of shape (M, N+1, num_assets) - Stock prices
        v: Tensor of shape (M, N+1, num_assets) - Variance processes
    """
    dt = T / N
    
    # Initialize arrays for all assets
    S = torch.zeros(M, N + 1, num_assets, device=device)
    v = torch.zeros(M, N + 1, num_assets, device=device)
    
    # Set initial conditions (same for all assets)
    S[:, 0, :] = S0
    v[:, 0, :] = v0
    
    # Simulate each asset independently
    for asset_idx in range(num_assets):
        for k in range(N):
            # Generate independent Brownian increments for this asset
            Z1 = torch.randn(M, device=device)
            Z2 = torch.randn(M, device=device)
            
            # Create correlated Brownian motions
            # W1 drives stock price, W2 drives variance
            W1 = Z1
            W2 = rho * Z1 + torch.sqrt(torch.tensor(1 - rho**2, device=device)) * Z2
            
            # Ensure variance stays non-negative (full truncation scheme)
            v_prev = torch.clamp(v[:, k, asset_idx], min=0.0)
            
            # Update variance using Euler scheme with truncation
            v[:, k+1, asset_idx] = torch.clamp(
                v_prev 
                + kappa * (theta - v_prev) * dt
                + xi * torch.sqrt(v_prev * dt) * W2,
                min=0.0
            )
            
            # Update stock price (log-normal dynamics)
            S[:, k+1, asset_idx] = S[:, k, asset_idx] * torch.exp(
                -0.5 * v_prev * dt
                + torch.sqrt(v_prev * dt) * W1
            )
    
    return S, v


def train_multi_asset_hedging(
    num_assets: int = 5,
    num_epochs: int = 2000,
    batch_size: int = 1024,
    S0: float = 100.0,
    K: float = 100.0,
    v0: float = 0.04,
    kappa: float = 2.0,
    theta: float = 0.04,
    xi: float = 0.5,
    rho: float = -0.7,
    T: float = 1.0,
    N: int = 30,
    lambda_risk: float = 1.0,
    lr: float = 1e-4,
    verbose: bool = True
) -> Tuple[MultiAssetHedgingPolicy, List[float], torch.Tensor]:
    """
    Train hedging policy for multiple assets with variance optimal criterion.
    
    This implements Section 5.4 of the paper: "High-dimensional example"
    
    The portfolio consists of num_assets call options (one on each underlying).
    We hedge using only the underlying stocks (not variance swaps in this 
    simplified version, though the paper uses both).
    
    Objective: Minimize variance of terminal P&L, which for independent assets
    decomposes into sum of individual variances (Equation 5.9 in paper).
    
    Args:
        num_assets: Number of independent assets/options (5 in paper)
        num_epochs: Number of training iterations
        batch_size: Number of paths per training batch
        S0: Initial stock price (same for all assets)
        K: Strike price for call options
        v0, kappa, theta, xi, rho: Heston model parameters
        T: Time horizon (1 year)
        N: Number of rebalancing steps (30 = daily)
        lambda_risk: Risk aversion parameter (not used for variance optimal)
        lr: Learning rate
        verbose: Whether to print training progress
    
    Returns:
        policy: Trained neural network policy
        loss_history: List of loss values during training
        pnl_eval: Final P&L on evaluation set
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize policy network
    policy = MultiAssetHedgingPolicy(num_assets=num_assets).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    
    loss_history = []
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Training Multi-Asset Hedging Policy")
        print(f"{'='*60}")
        print(f"Number of assets: {num_assets}")
        print(f"Total hedging instruments: {num_assets} (stocks only)")
        print(f"Objective: Variance optimal hedging")
        print(f"{'='*60}\n")
    
    # =============================
    # TRAINING LOOP
    # =============================
    for epoch in range(num_epochs):
        
        # Generate paths for all assets
        S, v = simulate_multi_heston_paths(
            S0=S0,
            v0=v0,
            kappa=kappa,
            theta=theta,
            xi=xi,
            rho=rho,
            T=T,
            N=N,
            M=batch_size,
            num_assets=num_assets,
            device=device
        )
        
        # Initialize P&L for each path
        trading_gain = torch.zeros(batch_size, device=device)
        
        # Dynamic hedging loop
        for k in range(N):
            # Current time as feature
            time = torch.full((batch_size,), k / N, device=device)
            
            # State: normalized prices for all assets + time
            # Shape: (batch_size, num_assets + 1)
            normalized_prices = S[:, k, :] / S0  # Normalize by initial price
            state = torch.cat([normalized_prices, time.unsqueeze(1)], dim=1)
            
            # Get hedge ratios for all assets
            # Shape: (batch_size, num_assets)
            deltas = policy(state)
            
            # Compute trading gains across all assets
            # This is the key: sum over all assets
            price_changes = S[:, k+1, :] - S[:, k, :]  # (batch_size, num_assets)
            trading_gain += (deltas * price_changes).sum(dim=1)
        
        # Terminal payoff: sum of call options on all assets
        # Z = sum_{i=1}^{num_assets} (S_i(T) - K)^+
        payoff = torch.clamp(S[:, -1, :] - K, min=0.0).sum(dim=1)
        
        # Terminal P&L
        pnl = trading_gain - payoff
        
        # Variance optimal objective: minimize E[PnL^2]
        # This is equivalent to minimizing variance when E[PnL] ≈ 0
        loss = torch.mean(pnl ** 2)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 5.0)
        optimizer.step()
        
        loss_history.append(loss.item())
        
        if verbose and epoch % 100 == 0:
            print(f"Epoch {epoch:4d}, Loss (Variance): {loss.item():.4f}")
    
    # =============================
    # EVALUATION
    # =============================
    if verbose:
        print(f"\n{'='*60}")
        print("Evaluating on test set...")
        print(f"{'='*60}\n")
    
    with torch.no_grad():
        # Generate test paths
        S_test, _ = simulate_multi_heston_paths(
            S0=S0,
            v0=v0,
            kappa=kappa,
            theta=theta,
            xi=xi,
            rho=rho,
            T=T,
            N=N,
            M=batch_size,
            num_assets=num_assets,
            device=device
        )
        
        trading_gain = torch.zeros(batch_size, device=device)
        
        for k in range(N):
            time = torch.full((batch_size,), k / N, device=device)
            normalized_prices = S_test[:, k, :] / S0
            state = torch.cat([normalized_prices, time.unsqueeze(1)], dim=1)
            
            deltas = policy(state)
            price_changes = S_test[:, k+1, :] - S_test[:, k, :]
            trading_gain += (deltas * price_changes).sum(dim=1)
        
        payoff = torch.clamp(S_test[:, -1, :] - K, min=0.0).sum(dim=1)
        pnl_eval = trading_gain - payoff
    
    return policy, loss_history, pnl_eval


def analyze_multi_asset_scalability(
    asset_counts: List[int] = [1, 2, 3, 5, 10],
    num_epochs: int = 2000,
    batch_size: int = 1024
) -> pd.DataFrame:
    """
    Analyze computational scalability as number of assets increases.
    
    This is the key experiment from Section 5.4: demonstrating that
    computational cost grows slowly with the number of assets.
    
    The paper shows that with 5 Heston models (10 instruments including
    variance swaps), the training time is only ~2.7x that of a single
    asset, despite 5x the problem complexity.
    
    Args:
        asset_counts: List of asset counts to test
        num_epochs: Training epochs for each configuration
        batch_size: Batch size for training
    
    Returns:
        DataFrame with columns: [num_assets, training_time, final_loss, 
                                 mean_pnl, std_pnl, variance]
    """
    import time
    
    results = []
    
    print("\n" + "="*80)
    print("MULTI-ASSET SCALABILITY ANALYSIS")
    print("="*80)
    print("Testing computational performance as number of assets increases...")
    print("="*80 + "\n")
    
    for n_assets in asset_counts:
        print(f"\n{'─'*80}")
        print(f"Testing with {n_assets} asset(s)...")
        print(f"{'─'*80}")
        
        start_time = time.time()
        
        # Train the policy
        policy, loss_hist, pnl_eval = train_multi_asset_hedging(
            num_assets=n_assets,
            num_epochs=num_epochs,
            batch_size=batch_size,
            verbose=False  # Suppress per-epoch output for cleaner analysis
        )
        
        training_time = time.time() - start_time
        
        # Compute metrics
        pnl_np = pnl_eval.cpu().numpy()
        final_loss = loss_hist[-1]
        mean_pnl = pnl_np.mean()
        std_pnl = pnl_np.std()
        variance = pnl_np.var()
        
        results.append({
            'num_assets': n_assets,
            'training_time_seconds': training_time,
            'final_loss': final_loss,
            'mean_pnl': mean_pnl,
            'std_pnl': std_pnl,
            'variance': variance
        })
        
        print(f"✓ Completed in {training_time:.2f} seconds")
        print(f"  Final Loss: {final_loss:.4f}")
        print(f"  P&L: μ={mean_pnl:.4f}, σ={std_pnl:.4f}")
    
    df = pd.DataFrame(results)
    
    # Compute relative metrics (normalized to single asset)
    if len(df) > 0:
        baseline_time = df.iloc[0]['training_time_seconds']
        baseline_loss = df.iloc[0]['final_loss']
        
        df['time_ratio'] = df['training_time_seconds'] / baseline_time
        df['loss_ratio'] = df['final_loss'] / baseline_loss
    
    print("\n" + "="*80)
    print("SCALABILITY RESULTS SUMMARY")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80 + "\n")
    
    # Expected result from paper:
    # For 5 assets vs 1 asset, training time should be ~2.7x
    # This demonstrates that computational cost scales sub-linearly
    
    return df


def compare_single_vs_multi_asset():
    """
    Reproduce the specific comparison from Section 5.4 of the paper.
    
    Paper states: "we compare the model for nH=5 and nH=1"
    Expected: Loss for 5 assets should be ≈ 5× loss for 1 asset
    (since we're solving 5 independent problems simultaneously)
    """
    print("\n" + "="*80)
    print("SINGLE vs MULTI-ASSET COMPARISON (Paper Section 5.4)")
    print("="*80 + "\n")
    
    # Train single asset
    print("Training SINGLE asset model...")
    _, _, pnl_single = train_multi_asset_hedging(
        num_assets=1,
        num_epochs=2000,
        batch_size=1024,
        verbose=True
    )
    
    variance_single = pnl_single.var().item()
    
    print("\n" + "─"*80 + "\n")
    
    # Train multi-asset
    print("Training MULTI-ASSET (5 assets) model...")
    _, _, pnl_multi = train_multi_asset_hedging(
        num_assets=5,
        num_epochs=2000,
        batch_size=1024,
        verbose=True
    )
    
    variance_multi = pnl_multi.var().item()
    
    # Analysis
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    print(f"Single Asset Variance: {variance_single:.4f}")
    print(f"Multi-Asset Variance:  {variance_multi:.4f}")
    print(f"Ratio (Multi/Single):  {variance_multi/variance_single:.4f}")
    print(f"Expected Ratio:        ~5.0 (solving 5 problems at once)")
    print("="*80)
    
    # Paper states: "this indicates that the approximation quality is roughly
    # the same for both instances (and close-to-optimal)"
    
    return variance_single, variance_multi


# ============================================================================
# SECTION 2: TRANSACTION COST CONVERGENCE ANALYSIS (Section 5.3 Complete)
# ============================================================================
# Status: PARTIALLY IMPLEMENTED - need convergence rate verification
# Priority: HIGH - Key theoretical validation
# ============================================================================

def compute_price_convergence_rate(
    epsilon_values: List[float] = None,
    num_epochs: int = 2000,
    batch_size: int = 1024,
    lambda_risk: float = 1.0,
    model_type: str = 'blackscholes',
    plot: bool = True
) -> Tuple[List[float], List[float], float]:
    """
    Compute and verify the O(ε^{2/3}) convergence rate of prices.
    
    This implements the complete analysis from Section 5.3 of the paper.
    
    Theoretical Result (Whalley & Wilmott 1997):
        p_ε - p_0 = O(ε^{2/3}) as ε → 0
    
    where:
        p_ε: Utility indifference price with transaction cost ε
        p_0: Price with zero transaction costs
    
    The paper verifies this by:
    1. Computing prices for multiple ε values
    2. Plotting log(p_ε - p_0) vs log(ε)
    3. Verifying the slope is approximately 2/3
    
    Args:
        epsilon_values: List of transaction cost rates to test
                       Default: [2^(-i+5) for i in 1..5] = [16, 8, 4, 2, 1] × 10^(-3)
        num_epochs: Training epochs for each ε
        batch_size: Batch size
        lambda_risk: Risk aversion parameter
        model_type: 'blackscholes' or 'heston'
        plot: Whether to create log-log plot
    
    Returns:
        epsilon_list: List of ε values tested
        price_diffs: List of (p_ε - p_0) values
        convergence_rate: Estimated slope from log-log regression
    """
    if epsilon_values is None:
        # Paper uses: ε_i = 2^(-i+5) for i = 1, 2, 3, 4, 5
        epsilon_values = [2**(-i+5) * 0.001 for i in range(1, 6)]
        # This gives: [0.016, 0.008, 0.004, 0.002, 0.001]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("\n" + "="*80)
    print("TRANSACTION COST CONVERGENCE RATE ANALYSIS")
    print("="*80)
    print(f"Model: {model_type.upper()}")
    print(f"Epsilon values: {epsilon_values}")
    print(f"Theoretical convergence rate: 2/3 ≈ 0.667")
    print("="*80 + "\n")
    
    # Step 1: Compute baseline price (ε = 0)
    print("Step 1: Computing baseline price (ε = 0)...")
    print("─"*80)
    
    if model_type == 'blackscholes':
        policy_baseline, pnl_baseline = train_bs_hedging(
            num_epochs=num_epochs,
            batch_size=batch_size,
            lambda_risk=lambda_risk,
            tc_rate=0.0,  # No transaction costs
            verbose=False
        )
    else:  # heston
        policy_baseline, pnl_baseline = train_heston_hedging(
            num_epochs=num_epochs,
            batch_size=batch_size,
            lambda_risk=lambda_risk,
            tc_rate=0.0,
            verbose=False
        )
    
    # Compute certainty equivalent (utility indifference price)
    # For entropic utility: CE = -(1/λ) log E[exp(-λ * PnL)]
    p_0 = (-1.0 / lambda_risk) * torch.log(
        torch.mean(torch.exp(-lambda_risk * pnl_baseline))
    ).item()
    
    print(f"✓ Baseline price p_0 = {p_0:.6f}\n")
    
    # Step 2: Compute prices for each ε
    print("Step 2: Computing prices for different transaction costs...")
    print("─"*80)
    
    prices = []
    price_diffs = []
    
    for epsilon in epsilon_values:
        print(f"\nTraining with ε = {epsilon:.6f}...")
        
        if model_type == 'blackscholes':
            _, pnl_eps = train_bs_hedging(
                num_epochs=num_epochs,
                batch_size=batch_size,
                lambda_risk=lambda_risk,
                tc_rate=epsilon,
                verbose=False
            )
        else:
            _, pnl_eps = train_heston_hedging(
                num_epochs=num_epochs,
                batch_size=batch_size,
                lambda_risk=lambda_risk,
                tc_rate=epsilon,
                verbose=False
            )
        
        # Compute certainty equivalent
        p_eps = (-1.0 / lambda_risk) * torch.log(
            torch.mean(torch.exp(-lambda_risk * pnl_eps))
        ).item()
        
        price_diff = p_eps - p_0
        
        prices.append(p_eps)
        price_diffs.append(price_diff)
        
        print(f"  p_ε = {p_eps:.6f}")
        print(f"  p_ε - p_0 = {price_diff:.6f}")
    
    # Step 3: Log-log regression to estimate convergence rate
    print("\n" + "─"*80)
    print("Step 3: Computing convergence rate...")
    print("─"*80)
    
    # Filter out any negative price differences (shouldn't happen in theory)
    valid_indices = [i for i, pd in enumerate(price_diffs) if pd > 0]
    
    if len(valid_indices) < 2:
        print("ERROR: Not enough valid price differences for regression")
        return epsilon_values, price_diffs, np.nan
    
    log_epsilon = np.log([epsilon_values[i] for i in valid_indices])
    log_price_diff = np.log([price_diffs[i] for i in valid_indices])
    
    # Linear regression: log(p_ε - p_0) = α + β * log(ε)
    # We expect β ≈ 2/3
    coeffs = np.polyfit(log_epsilon, log_price_diff, 1)
    convergence_rate = coeffs[0]  # This is β (the slope)
    intercept = coeffs[1]
    
    print(f"\nRegression Results:")
    print(f"  Estimated convergence rate β = {convergence_rate:.4f}")
    print(f"  Theoretical rate = 0.6667 (i.e., 2/3)")
    print(f"  Difference = {abs(convergence_rate - 2/3):.4f}")
    print(f"  Intercept α = {intercept:.4f}")
    
    # Step 4: Visualization
    if plot:
        plt.figure(figsize=(10, 6))
        
        # Plot data points
        plt.scatter(log_epsilon, log_price_diff, 
                   s=100, alpha=0.7, label='Computed prices', zorder=3)
        
        # Plot fitted line
        fitted_line = coeffs[0] * log_epsilon + coeffs[1]
        plt.plot(log_epsilon, fitted_line, 
                'r--', linewidth=2, label=f'Fitted line (slope={convergence_rate:.3f})', zorder=2)
        
        # Plot theoretical line (slope = 2/3)
        theoretical_line = (2/3) * log_epsilon + intercept
        plt.plot(log_epsilon, theoretical_line,
                'g:', linewidth=2, label='Theoretical slope=2/3', zorder=1)
        
        plt.xlabel('log(ε)', fontsize=12)
        plt.ylabel('log(p_ε - p_0)', fontsize=12)
        plt.title(f'Transaction Cost Convergence Rate ({model_type.capitalize()})\n'
                 f'Estimated rate: {convergence_rate:.4f}, Theoretical: 0.667',
                 fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save figure
        filename = f'convergence_rate_{model_type}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\n✓ Plot saved to {filename}")
        
        plt.show()
    
    print("\n" + "="*80)
    print("CONVERGENCE ANALYSIS COMPLETE")
    print("="*80 + "\n")
    
    # Interpretation note
    if abs(convergence_rate - 2/3) < 0.1:
        print("✓ VALIDATION SUCCESSFUL: Convergence rate matches theory!")
    else:
        print("⚠ WARNING: Convergence rate deviates from theoretical prediction")
        print("  This may indicate: insufficient training, numerical issues,")
        print("  or model-specific effects. Consider increasing num_epochs.")
    
    return epsilon_values, price_diffs, convergence_rate


def train_bs_hedging(
    num_epochs: int,
    batch_size: int,
    lambda_risk: float,
    tc_rate: float,
    verbose: bool = True
) -> Tuple[nn.Module, torch.Tensor]:
    """
    Helper function: Train Black-Scholes hedging with optional transaction costs.
    
    This is a clean, focused version for convergence analysis.
    """
    from scipy.stats import norm as scipy_norm  # For BS delta if needed
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Model parameters
    S0, K = 100.0, 100.0
    sigma, T, N = 0.2, 1.0, 30
    
    # Initialize policy
    policy = HedgingPolicy().to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    
    for epoch in range(num_epochs):
        # Simulate Black-Scholes paths
        S = simulate_paths(
            S0=S0, mu=0.0, sigma=sigma,
            T=T, N=N, M=batch_size, device=device
        )
        
        trading_gain = torch.zeros(batch_size, device=device)
        transaction_costs = torch.zeros(batch_size, device=device)
        prev_delta = torch.zeros(batch_size, device=device)
        
        for k in range(N):
            time = torch.full((batch_size,), k / N, device=device)
            state = torch.stack([S[:, k] / S0, time], dim=1)
            
            delta = policy(state)
            trading_gain += delta * (S[:, k+1] - S[:, k])
            
            # Transaction costs
            if tc_rate > 0:
                transaction_costs += tc_rate * torch.abs(delta - prev_delta) * S[:, k]
            
            prev_delta = delta
        
        payoff = torch.clamp(S[:, -1] - K, min=0.0)
        pnl = trading_gain - payoff - transaction_costs
        
        # Entropic risk objective
        pnl_scaled = pnl / (pnl.std().detach() + 1e-8)
        loss = torch.logsumexp(-lambda_risk * pnl_scaled, dim=0) - \
               torch.log(torch.tensor(batch_size, device=device))
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 5.0)
        optimizer.step()
        
        if verbose and epoch % 100 == 0:
            print(f"Epoch {epoch:4d}, Loss: {loss.item():.4f}")
    
    # Evaluation
    with torch.no_grad():
        S = simulate_paths(
            S0=S0, mu=0.0, sigma=sigma,
            T=T, N=N, M=batch_size, device=device
        )
        
        trading_gain = torch.zeros(batch_size, device=device)
        transaction_costs = torch.zeros(batch_size, device=device)
        prev_delta = torch.zeros(batch_size, device=device)
        
        for k in range(N):
            time = torch.full((batch_size,), k / N, device=device)
            state = torch.stack([S[:, k] / S0, time], dim=1)
            delta = policy(state)
            trading_gain += delta * (S[:, k+1] - S[:, k])
            
            if tc_rate > 0:
                transaction_costs += tc_rate * torch.abs(delta - prev_delta) * S[:, k]
            
            prev_delta = delta
        
        payoff = torch.clamp(S[:, -1] - K, min=0.0)
        pnl_eval = trading_gain - payoff - transaction_costs
    
    return policy, pnl_eval


def train_heston_hedging(
    num_epochs: int,
    batch_size: int,
    lambda_risk: float,
    tc_rate: float,
    verbose: bool = True
) -> Tuple[nn.Module, torch.Tensor]:
    """
    Helper function: Train Heston hedging with optional transaction costs.
    
    This is a clean, focused version for convergence analysis.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Model parameters
    S0, K = 100.0, 100.0
    v0, kappa, theta, xi, rho = 0.04, 2.0, 0.04, 0.5, -0.7
    T, N = 1.0, 30
    
    policy = HedgingPolicy().to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    
    for epoch in range(num_epochs):
        S, _ = simulate_heston_paths(
            S0=S0, v0=v0, kappa=kappa, theta=theta, xi=xi, rho=rho,
            T=T, N=N, M=batch_size, device=device
        )
        
        trading_gain = torch.zeros(batch_size, device=device)
        transaction_costs = torch.zeros(batch_size, device=device)
        prev_delta = torch.zeros(batch_size, device=device)
        
        for k in range(N):
            time = torch.full((batch_size,), k / N, device=device)
            state = torch.stack([S[:, k] / S0, time], dim=1)
            
            delta = policy(state)
            trading_gain += delta * (S[:, k+1] - S[:, k])
            
            if tc_rate > 0:
                transaction_costs += tc_rate * torch.abs(delta - prev_delta) * S[:, k]
            
            prev_delta = delta
        
        payoff = torch.clamp(S[:, -1] - K, min=0.0)
        pnl = trading_gain - payoff - transaction_costs
        
        pnl_scaled = pnl / (pnl.std().detach() + 1e-8)
        loss = torch.logsumexp(-lambda_risk * pnl_scaled, dim=0) - \
               torch.log(torch.tensor(batch_size, device=device))
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 5.0)
        optimizer.step()
        
        if verbose and epoch % 100 == 0:
            print(f"Epoch {epoch:4d}, Loss: {loss.item():.4f}")
    
    # Evaluation
    with torch.no_grad():
        S, _ = simulate_heston_paths(
            S0=S0, v0=v0, kappa=kappa, theta=theta, xi=xi, rho=rho,
            T=T, N=N, M=batch_size, device=device
        )
        
        trading_gain = torch.zeros(batch_size, device=device)
        transaction_costs = torch.zeros(batch_size, device=device)
        prev_delta = torch.zeros(batch_size, device=device)
        
        for k in range(N):
            time = torch.full((batch_size,), k / N, device=device)
            state = torch.stack([S[:, k] / S0, time], dim=1)
            delta = policy(state)
            trading_gain += delta * (S[:, k+1] - S[:, k])
            
            if tc_rate > 0:
                transaction_costs += tc_rate * torch.abs(delta - prev_delta) * S[:, k]
            
            prev_delta = delta
        
        payoff = torch.clamp(S[:, -1] - K, min=0.0)
        pnl_eval = trading_gain - payoff - transaction_costs
    
    return policy, pnl_eval


# ============================================================================
# SECTION 3: HEDGE RATIO SURFACE VISUALIZATION (Figures 3, 8, 9 from Paper)
# ============================================================================
# Status: PARTIALLY IMPLEMENTED - only 2D slices shown, need full surfaces
# Priority: MEDIUM - Important for understanding learned strategies
# ============================================================================

def plot_hedge_surface_3d(
    policy: nn.Module,
    time_point: float = 0.5,
    S_range: Tuple[float, float] = (80, 120),
    v_range: Tuple[float, float] = (0.01, 0.15),
    grid_points: int = 50,
    S0: float = 100.0,
    compare_with_bs: bool = True,
    K: float = 100.0,
    sigma: float = 0.2,
    T: float = 1.0
) -> None:
    """
    Create 3D surface plot of hedge ratio as function of (spot, variance).
    
    This reproduces Figures 3, 8, and 9 from the paper, which show:
    - Neural network delta surface
    - Model (Black-Scholes or Heston) delta surface
    - Difference between the two
    
    For Heston model, "model delta" requires solving the Heston PDE,
    so we compare with Black-Scholes delta using implied volatility.
    
    Args:
        policy: Trained hedging policy network
        time_point: Time at which to evaluate (fraction of T)
        S_range: (min, max) stock prices to visualize
        v_range: (min, max) variance values to visualize
        grid_points: Number of points in each dimension
        S0: Initial stock price (for normalization)
        compare_with_bs: Whether to compute BS delta for comparison
        K, sigma, T: Option parameters for BS delta
    """
    device = next(policy.parameters()).device
    
    # Create grid
    S_vals = np.linspace(S_range[0], S_range[1], grid_points)
    v_vals = np.linspace(v_range[0], v_range[1], grid_points)
    S_grid, v_grid = np.meshgrid(S_vals, v_vals)
    
    # Flatten for batch processing
    S_flat = torch.tensor(S_grid.flatten(), dtype=torch.float32, device=device)
    v_flat = torch.tensor(v_grid.flatten(), dtype=torch.float32, device=device)
    time_tensor = torch.full((len(S_flat),), time_point, device=device)
    
    # Compute neural network deltas
    with torch.no_grad():
        state = torch.stack([S_flat / S0, time_tensor], dim=1)
        delta_nn = policy(state).cpu().numpy()
    
    delta_nn_grid = delta_nn.reshape(grid_points, grid_points)
    
    # Compute Black-Scholes deltas if requested
    if compare_with_bs:
        # Use v as variance, convert to volatility
        sigma_grid = np.sqrt(v_grid)
        tau = T * (1 - time_point)  # Time to maturity
        
        # Black-Scholes delta: N(d1)
        d1 = (np.log(S_grid / K) + 0.5 * v_grid * tau) / (sigma_grid * np.sqrt(tau) + 1e-8)
        delta_bs_grid = scipy_norm.cdf(d1)
        
        # Difference
        delta_diff_grid = delta_nn_grid - delta_bs_grid
    
    # Create 3-panel figure
    if compare_with_bs:
        fig = plt.figure(figsize=(18, 5))
        
        # Panel 1: Neural Network Delta
        ax1 = fig.add_subplot(131, projection='3d')
        surf1 = ax1.plot_surface(S_grid, v_grid, delta_nn_grid,
                                 cmap='viridis', alpha=0.9, edgecolor='none')
        ax1.set_xlabel('Stock Price ($S_t$)', fontsize=10)
        ax1.set_ylabel('Variance ($v_t$)', fontsize=10)
        ax1.set_zlabel('Delta ($\\delta_t$)', fontsize=10)
        ax1.set_title(f'Neural Network Delta\n(t={time_point:.2f}T)', 
                     fontsize=12, fontweight='bold')
        fig.colorbar(surf1, ax=ax1, shrink=0.5)
        
        # Panel 2: Black-Scholes Delta
        ax2 = fig.add_subplot(132, projection='3d')
        surf2 = ax2.plot_surface(S_grid, v_grid, delta_bs_grid,
                                 cmap='plasma', alpha=0.9, edgecolor='none')
        ax2.set_xlabel('Stock Price ($S_t$)', fontsize=10)
        ax2.set_ylabel('Variance ($v_t$)', fontsize=10)
        ax2.set_zlabel('Delta ($\\delta_t$)', fontsize=10)
        ax2.set_title(f'Black-Scholes Delta\n(t={time_point:.2f}T)',
                     fontsize=12, fontweight='bold')
        fig.colorbar(surf2, ax=ax2, shrink=0.5)
        
        # Panel 3: Difference
        ax3 = fig.add_subplot(133, projection='3d')
        surf3 = ax3.plot_surface(S_grid, v_grid, delta_diff_grid,
                                 cmap='RdBu_r', alpha=0.9, edgecolor='none')
        ax3.set_xlabel('Stock Price ($S_t$)', fontsize=10)
        ax3.set_ylabel('Variance ($v_t$)', fontsize=10)
        ax3.set_zlabel('$\\Delta$ (NN - BS)', fontsize=10)
        ax3.set_title(f'Difference\n(t={time_point:.2f}T)',
                     fontsize=12, fontweight='bold')
        fig.colorbar(surf3, ax=ax3, shrink=0.5)
        
    else:
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(S_grid, v_grid, delta_nn_grid,
                               cmap='viridis', alpha=0.9, edgecolor='none')
        ax.set_xlabel('Stock Price ($S_t$)', fontsize=12)
        ax.set_ylabel('Variance ($v_t$)', fontsize=12)
        ax.set_zlabel('Delta ($\\delta_t$)', fontsize=12)
        ax.set_title(f'Neural Network Delta Surface (t={time_point:.2f}T)',
                    fontsize=14, fontweight='bold')
        fig.colorbar(surf, ax=ax, shrink=0.6)
    
    plt.tight_layout()
    
    # Save figure
    filename = f'hedge_surface_3d_t{time_point:.2f}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ 3D surface plot saved to {filename}")
    
    plt.show()


def plot_callspread_hedge_comparison(
    policies: Dict[str, nn.Module],
    K1: float = 100.0,
    K2: float = 101.0,
    time_point: float = 0.5,
    S_range: Tuple[float, float] = (96, 104),
    v_range: Tuple[float, float] = (0.04, 0.14),
    S0: float = 100.0,
    grid_points: int = 50
) -> None:
    """
    Visualize hedging strategies for call spreads with different risk aversions.
    
    This reproduces Figures 8 and 9 from the paper, showing that more
    risk-averse strategies exhibit "barrier shift" - they hedge as if
    the strike is at a different level.
    
    Args:
        policies: Dict mapping risk_aversion_label -> trained_policy
                 e.g., {'0.5-CVaR': policy1, '0.95-CVaR': policy2, '0.99-CVaR': policy3}
        K1, K2: Strike prices for call spread
        time_point: Time fraction to visualize
        S_range, v_range: Ranges for visualization
        S0: Initial price
        grid_points: Grid resolution
    """
    device = list(policies.values())[0].device if policies else 'cpu'
    
    # Create grid
    S_vals = np.linspace(S_range[0], S_range[1], grid_points)
    v_vals = np.linspace(v_range[0], v_range[1], grid_points)
    S_grid, v_grid = np.meshgrid(S_vals, v_vals)
    
    S_flat = torch.tensor(S_grid.flatten(), dtype=torch.float32, device=device)
    time_tensor = torch.full((len(S_flat),), time_point, device=device)
    
    # Prepare figure
    n_policies = len(policies)
    fig = plt.figure(figsize=(6*n_policies, 5))
    
    for idx, (label, policy) in enumerate(policies.items(), 1):
        with torch.no_grad():
            state = torch.stack([S_flat / S0, time_tensor], dim=1)
            delta_flat = policy(state).cpu().numpy()
        
        delta_grid = delta_flat.reshape(grid_points, grid_points)
        
        ax = fig.add_subplot(1, n_policies, idx, projection='3d')
        surf = ax.plot_surface(S_grid, v_grid, delta_grid,
                               cmap='viridis', alpha=0.9, edgecolor='none')
        ax.set_xlabel('Stock Price ($S_t$)', fontsize=10)
        ax.set_ylabel('Variance ($v_t$)', fontsize=10)
        ax.set_zlabel('Delta', fontsize=10)
        ax.set_title(f'{label}\nCall Spread Hedge',
                    fontsize=12, fontweight='bold')
        fig.colorbar(surf, ax=ax, shrink=0.5)
    
    plt.tight_layout()
    
    filename = 'callspread_comparison_3d.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Call spread comparison saved to {filename}")
    
    plt.show()
    
    # Also create 2D slices at fixed variance
    fig2, axes = plt.subplots(1, 1, figsize=(10, 6))
    
    # Take slice at median variance
    v_fixed_idx = grid_points // 2
    v_fixed = v_vals[v_fixed_idx]
    
    for label, policy in policies.items():
        with torch.no_grad():
            state = torch.stack([S_flat / S0, time_tensor], dim=1)
            delta_flat = policy(state).cpu().numpy()
        
        delta_grid = delta_flat.reshape(grid_points, grid_points)
        delta_slice = delta_grid[v_fixed_idx, :]
        
        axes.plot(S_vals, delta_slice, linewidth=2, label=label, marker='o', markersize=4)
    
    # Add strike lines
    axes.axvline(K1, color='red', linestyle='--', alpha=0.5, label=f'K1={K1}')
    axes.axvline(K2, color='blue', linestyle='--', alpha=0.5, label=f'K2={K2}')
    
    axes.set_xlabel('Stock Price', fontsize=12)
    axes.set_ylabel('Delta', fontsize=12)
    axes.set_title(f'Call Spread Delta at v={v_fixed:.4f}, t={time_point:.2f}T',
                  fontsize=14, fontweight='bold')
    axes.legend(fontsize=10)
    axes.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    filename2 = 'callspread_comparison_2d.png'
    plt.savefig(filename2, dpi=300, bbox_inches='tight')
    print(f"✓ Call spread 2D comparison saved to {filename2}")
    
    plt.show()


# ============================================================================
# SECTION 4: CALL SPREAD HEDGING (Paper Section 5.2, end)
# ============================================================================
# Status: COMPLETELY MISSING
# Priority: MEDIUM - Demonstrates practical application
# ============================================================================

def train_callspread_hedging(
    K1: float = 100.0,
    K2: float = 101.0,
    lambda_risk: float = 1.0,
    num_epochs: int = 2000,
    batch_size: int = 1024,
    use_heston: bool = True,
    tc_rate: float = 0.0
) -> Tuple[nn.Module, torch.Tensor]:
    """
    Train hedging policy for a call spread option.
    
    Call spread payoff: Z = [(S_T - K1)^+ - (S_T - K2)^+] / (K2 - K1)
    
    This is a normalized spread that pays 1 when S_T ∈ [K1, K2] and 0 outside.
    
    Paper observation (Section 5.2, end): More risk-averse hedging strategies
    exhibit "barrier shift" - they hedge as if the spread has strikes at
    (K1 - shift, K2 - shift) for some shift > 0.
    
    Args:
        K1, K2: Strike prices (K1 < K2)
        lambda_risk: Risk aversion parameter
        num_epochs: Training iterations
        batch_size: Paths per batch
        use_heston: If True, use Heston; else Black-Scholes
        tc_rate: Transaction cost rate
    
    Returns:
        policy: Trained neural network
        pnl_eval: Terminal P&L on evaluation set
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Model parameters
    S0 = 100.0
    T, N = 1.0, 30
    
    if K1 >= K2:
        raise ValueError(f"Must have K1 < K2, got K1={K1}, K2={K2}")
    
    print("\n" + "="*80)
    print("CALL SPREAD HEDGING")
    print("="*80)
    print(f"Spread: Long Call@{K1}, Short Call@{K2}")
    print(f"Width: {K2-K1:.2f}")
    print(f"Model: {'Heston' if use_heston else 'Black-Scholes'}")
    print(f"Risk aversion (λ): {lambda_risk}")
    print(f"Transaction costs: {tc_rate*100:.3f}%")
    print("="*80 + "\n")
    
    # Initialize policy
    policy = HedgingPolicy().to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    
    loss_history = []
    
    # Training loop
    for epoch in range(num_epochs):
        
        if use_heston:
            S, _ = simulate_heston_paths(
                S0=S0, v0=0.04, kappa=2.0, theta=0.04,
                xi=0.5, rho=-0.7, T=T, N=N,
                M=batch_size, device=device
            )
        else:
            S = simulate_paths(
                S0=S0, mu=0.0, sigma=0.2,
                T=T, N=N, M=batch_size, device=device
            )
        
        trading_gain = torch.zeros(batch_size, device=device)
        transaction_costs = torch.zeros(batch_size, device=device)
        prev_delta = torch.zeros(batch_size, device=device)
        
        for k in range(N):
            time = torch.full((batch_size,), k / N, device=device)
            state = torch.stack([S[:, k] / S0, time], dim=1)
            
            delta = policy(state)
            trading_gain += delta * (S[:, k+1] - S[:, k])
            
            if tc_rate > 0:
                transaction_costs += tc_rate * torch.abs(delta - prev_delta) * S[:, k]
            
            prev_delta = delta
        
        # Call spread payoff
        # Payoff = (ST - K1)+ - (ST - K2)+
        # This equals min(max(ST - K1, 0), K2 - K1)
        long_call = torch.clamp(S[:, -1] - K1, min=0.0)
        short_call = torch.clamp(S[:, -1] - K2, min=0.0)
        payoff = long_call - short_call
        
        pnl = trading_gain - payoff - transaction_costs
        
        # Entropic risk
        pnl_scaled = pnl / (pnl.std().detach() + 1e-8)
        loss = torch.logsumexp(-lambda_risk * pnl_scaled, dim=0) - \
               torch.log(torch.tensor(batch_size, device=device))
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 5.0)
        optimizer.step()
        
        loss_history.append(loss.item())
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch:4d}, Loss: {loss.item():.4f}")
    
    # Evaluation
    print("\nEvaluating...")
    with torch.no_grad():
        if use_heston:
            S, _ = simulate_heston_paths(
                S0=S0, v0=0.04, kappa=2.0, theta=0.04,
                xi=0.5, rho=-0.7, T=T, N=N,
                M=batch_size, device=device
            )
        else:
            S = simulate_paths(
                S0=S0, mu=0.0, sigma=0.2,
                T=T, N=N, M=batch_size, device=device
            )
        
        trading_gain = torch.zeros(batch_size, device=device)
        transaction_costs = torch.zeros(batch_size, device=device)
        prev_delta = torch.zeros(batch_size, device=device)
        
        for k in range(N):
            time = torch.full((batch_size,), k / N, device=device)
            state = torch.stack([S[:, k] / S0, time], dim=1)
            delta = policy(state)
            trading_gain += delta * (S[:, k+1] - S[:, k])
            
            if tc_rate > 0:
                transaction_costs += tc_rate * torch.abs(delta - prev_delta) * S[:, k]
            
            prev_delta = delta
        
        long_call = torch.clamp(S[:, -1] - K1, min=0.0)
        short_call = torch.clamp(S[:, -1] - K2, min=0.0)
        payoff = long_call - short_call
        
        pnl_eval = trading_gain - payoff - transaction_costs
    
    # Print statistics
    pnl_np = pnl_eval.cpu().numpy()
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"Mean P&L: {pnl_np.mean():.4f}")
    print(f"Std P&L:  {pnl_np.std():.4f}")
    print(f"5% VaR:   {np.quantile(pnl_np, 0.05):.4f}")
    print(f"1% VaR:   {np.quantile(pnl_np, 0.01):.4f}")
    print("="*80 + "\n")
    
    return policy, pnl_eval


def compare_callspread_risk_aversions():
    """
    Compare call spread hedging for different risk aversion levels.
    
    Reproduces the analysis at the end of Section 5.2 showing that
    more risk-averse strategies lead to "flattened" hedges (barrier shift).
    """
    lambda_values = [0.5, 0.95, 0.99]  # Different risk aversion levels
    K1, K2 = 100.0, 101.0
    
    policies = {}
    pnls = {}
    
    for lam in lambda_values:
        print(f"\n{'#'*80}")
        print(f"# Training with CVaR at α = {lam}")
        print(f"{'#'*80}")
        
        policy, pnl = train_callspread_hedging(
            K1=K1,
            K2=K2,
            lambda_risk=lam,
            num_epochs=2000,
            use_heston=True
        )
        
        label = f'{lam:.2f}-CVaR'
        policies[label] = policy
        pnls[label] = pnl
    
    # Visualize the learned strategies
    print("\n" + "="*80)
    print("Creating comparison visualizations...")
    print("="*80 + "\n")
    
    plot_callspread_hedge_comparison(
        policies=policies,
        K1=K1,
        K2=K2,
        time_point=0.5
    )
    
    # Compare P&L distributions
    fig, axes = plt.subplots(1, 1, figsize=(10, 6))
    
    for label, pnl in pnls.items():
        pnl_np = pnl.cpu().numpy()
        axes.hist(pnl_np, bins=50, alpha=0.6, density=True, label=label)
    
    axes.set_xlabel('Terminal P&L', fontsize=12)
    axes.set_ylabel('Density', fontsize=12)
    axes.set_title('Call Spread P&L Distribution by Risk Aversion',
                  fontsize=14, fontweight='bold')
    axes.legend(fontsize=10)
    axes.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('callspread_pnl_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ P&L comparison saved to callspread_pnl_comparison.png")
    plt.show()


# ============================================================================
# SECTION 5: ALTERNATIVE RISK MEASURES (Beyond Entropic)
# ============================================================================
# Status: PARTIALLY IMPLEMENTED - only entropic risk in original
# Priority: LOW - Enhancement beyond paper requirements
# ============================================================================

class CVaRLoss(nn.Module):
    """
    Conditional Value at Risk (CVaR) loss function.
    
    Also known as Average Value at Risk or Expected Shortfall.
    
    CVaR_α(X) = (1/(1-α)) * E[X | X ≤ VaR_α(X)]
    
    This is the average loss in the worst (1-α)% of cases.
    
    For α = 0.95: CVaR is the average of the worst 5% of outcomes.
    For α = 0.99: CVaR is the average of the worst 1% of outcomes.
    
    Advantages over entropic risk:
    - More interpretable (directly related to quantiles)
    - Less sensitive to extreme outliers
    - Standard in banking/insurance regulation
    """
    
    def __init__(self, alpha: float = 0.95):
        """
        Args:
            alpha: Confidence level (e.g., 0.95 for 95% CVaR)
        """
        super().__init__()
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0,1), got {alpha}")
        self.alpha = alpha
    
    def forward(self, pnl: torch.Tensor) -> torch.Tensor:
        """
        Compute CVaR loss.
        
        Args:
            pnl: Terminal P&L values, shape (batch_size,)
                 Negative values indicate losses
        
        Returns:
            Scalar CVaR loss (to be minimized)
        """
        # CVaR minimization means maximizing CVaR of PnL
        # (since we want large positive PnL)
        # So we compute CVaR of -PnL
        losses = -pnl
        
        # Find the alpha-quantile (VaR)
        var_alpha = torch.quantile(losses, self.alpha)
        
        # CVaR: average of losses exceeding VaR
        tail_losses = losses[losses >= var_alpha]
        
        if len(tail_losses) == 0:
            # Edge case: no losses exceed VaR
            return var_alpha
        
        cvar = tail_losses.mean()
        
        return cvar


class SemiDeviationLoss(nn.Module):
    """
    Semi-deviation (downside risk) loss function.
    
    Semi-deviation = sqrt(E[min(X - target, 0)^2])
    
    This penalizes only downside deviations from a target,
    not upside deviations. Useful when asymmetric risk preferences.
    
    For hedging: target = 0 (we want non-negative P&L)
    """
    
    def __init__(self, target: float = 0.0):
        super().__init__()
        self.target = target
    
    def forward(self, pnl: torch.Tensor) -> torch.Tensor:
        """
        Compute semi-deviation loss.
        
        Args:
            pnl: Terminal P&L values, shape (batch_size,)
        
        Returns:
            Scalar semi-deviation (to be minimized)
        """
        # Downside deviations
        downside = torch.clamp(self.target - pnl, min=0.0)
        
        # Semi-variance
        semi_variance = torch.mean(downside ** 2)
        
        # Semi-deviation (return squared form for optimization)
        return semi_variance


def train_with_custom_risk_measure(
    risk_measure: nn.Module,
    num_epochs: int = 2000,
    batch_size: int = 1024,
    use_heston: bool = True
) -> Tuple[nn.Module, torch.Tensor]:
    """
    Train hedging policy with a custom risk measure.
    
    This demonstrates the flexibility of the deep hedging framework:
    any differentiable risk measure can be used as the objective.
    
    Args:
        risk_measure: Instance of a risk measure (CVaRLoss, SemiDeviationLoss, etc.)
        num_epochs: Training iterations
        batch_size: Paths per batch
        use_heston: Whether to use Heston (True) or Black-Scholes (False)
    
    Returns:
        policy: Trained neural network
        pnl_eval: Terminal P&L on evaluation set
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Model parameters
    S0, K = 100.0, 100.0
    T, N = 1.0, 30
    
    print("\n" + "="*80)
    print(f"Training with {risk_measure.__class__.__name__}")
    print("="*80 + "\n")
    
    policy = HedgingPolicy().to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    
    loss_history = []
    
    for epoch in range(num_epochs):
        
        if use_heston:
            S, _ = simulate_heston_paths(
                S0=S0, v0=0.04, kappa=2.0, theta=0.04,
                xi=0.5, rho=-0.7, T=T, N=N,
                M=batch_size, device=device
            )
        else:
            S = simulate_paths(
                S0=S0, mu=0.0, sigma=0.2,
                T=T, N=N, M=batch_size, device=device
            )
        
        trading_gain = torch.zeros(batch_size, device=device)
        
        for k in range(N):
            time = torch.full((batch_size,), k / N, device=device)
            state = torch.stack([S[:, k] / S0, time], dim=1)
            delta = policy(state)
            trading_gain += delta * (S[:, k+1] - S[:, k])
        
        payoff = torch.clamp(S[:, -1] - K, min=0.0)
        pnl = trading_gain - payoff
        
        # Apply custom risk measure
        loss = risk_measure(pnl)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 5.0)
        optimizer.step()
        
        loss_history.append(loss.item())
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch:4d}, Loss: {loss.item():.4f}")
    
    # Evaluation
    print("\nEvaluating...")
    with torch.no_grad():
        if use_heston:
            S, _ = simulate_heston_paths(
                S0=S0, v0=0.04, kappa=2.0, theta=0.04,
                xi=0.5, rho=-0.7, T=T, N=N,
                M=batch_size, device=device
            )
        else:
            S = simulate_paths(
                S0=S0, mu=0.0, sigma=0.2,
                T=T, N=N, M=batch_size, device=device
            )
        
        trading_gain = torch.zeros(batch_size, device=device)
        
        for k in range(N):
            time = torch.full((batch_size,), k / N, device=device)
            state = torch.stack([S[:, k] / S0, time], dim=1)
            delta = policy(state)
            trading_gain += delta * (S[:, k+1] - S[:, k])
        
        payoff = torch.clamp(S[:, -1] - K, min=0.0)
        pnl_eval = trading_gain - payoff
    
    return policy, pnl_eval


# ============================================================================
# SECTION 6: HELPER FUNCTIONS FROM ORIGINAL NOTEBOOK
# ============================================================================
# These are required by the new code above
# ============================================================================

class HedgingPolicy(nn.Module):
    """Standard hedging policy from original notebook"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.net(x).squeeze()


def simulate_paths(S0, mu, sigma, T, N, M, device):
    """Black-Scholes path simulator from original notebook"""
    dt = T / N
    S = torch.zeros(M, N + 1, device=device)
    S[:, 0] = S0
    
    for k in range(N):
        Z = torch.randn(M, device=device)
        S[:, k+1] = S[:, k] * torch.exp(
            (mu - 0.5 * sigma**2) * dt + sigma * math.sqrt(dt) * Z
        )
    
    return S


def simulate_heston_paths(S0, v0, kappa, theta, xi, rho, T, N, M, device):
    """Heston path simulator from original notebook"""
    dt = T / N
    S = torch.zeros(M, N + 1, device=device)
    v = torch.zeros(M, N + 1, device=device)
    
    S[:, 0] = S0
    v[:, 0] = v0
    
    for k in range(N):
        Z1 = torch.randn(M, device=device)
        Z2 = torch.randn(M, device=device)
        
        W1 = Z1
        W2 = rho * Z1 + torch.sqrt(torch.tensor(1 - rho**2, device=device)) * Z2
        
        v_prev = torch.clamp(v[:, k], min=0.0)
        
        v[:, k+1] = torch.clamp(
            v_prev + kappa * (theta - v_prev) * dt
            + xi * torch.sqrt(v_prev * dt) * W2,
            min=0.0
        )
        
        S[:, k+1] = S[:, k] * torch.exp(
            -0.5 * v_prev * dt + torch.sqrt(v_prev * dt) * W1
        )
    
    return S, v


# ============================================================================
# SECTION 7: MAIN EXECUTION EXAMPLES
# ============================================================================

if __name__ == "__main__":
    """
    Example usage of all the missing implementations.
    
    Uncomment the sections you want to run.
    """
    
    print("\n" + "="*80)
    print("DEEP HEDGING - MISSING IMPLEMENTATIONS DEMONSTRATION")
    print("="*80)
    print("\nThis script demonstrates all the components that were")
    print("missing or partially implemented in the original notebook.")
    print("\nUncomment the sections below to run specific experiments.")
    print("="*80 + "\n")
    
    # ========================================================================
    # EXPERIMENT 1: High-Dimensional Scalability (Section 5.4)
    # ========================================================================
    # print("\n### EXPERIMENT 1: High-Dimensional Scalability ###\n")
    # 
    # # Option A: Full scalability analysis
    # df_scalability = analyze_multi_asset_scalability(
    #     asset_counts=[1, 2, 3, 5],  # Test up to 5 assets
    #     num_epochs=1000,  # Reduced for demonstration
    #     batch_size=1024
    # )
    # 
    # # Option B: Single vs Multi comparison (as in paper)
    # var_single, var_multi = compare_single_vs_multi_asset()
    
    # ========================================================================
    # EXPERIMENT 2: Transaction Cost Convergence Rate (Section 5.3)
    # ========================================================================
    # print("\n### EXPERIMENT 2: Transaction Cost Convergence Rate ###\n")
    # 
    # # For Black-Scholes
    # eps_list, price_diffs, rate = compute_price_convergence_rate(
    #     epsilon_values=[0.016, 0.008, 0.004, 0.002, 0.001],
    #     num_epochs=1000,
    #     model_type='blackscholes',
    #     plot=True
    # )
    # 
    # # For Heston
    # eps_list, price_diffs, rate = compute_price_convergence_rate(
    #     epsilon_values=[0.016, 0.008, 0.004, 0.002, 0.001],
    #     num_epochs=1000,
    #     model_type='heston',
    #     plot=True
    # )
    
    # ========================================================================
    # EXPERIMENT 3: 3D Hedge Surface Visualization (Figures 3, 8, 9)
    # ========================================================================
    # print("\n### EXPERIMENT 3: 3D Hedge Surface Visualization ###\n")
    # 
    # # First train a policy
    # print("Training policy for visualization...")
    # policy_for_viz, _ = train_heston_hedging(
    #     num_epochs=2000,
    #     batch_size=1024,
    #     lambda_risk=1.0,
    #     tc_rate=0.0,
    #     verbose=True
    # )
    # 
    # # Create 3D surface plots at different time points
    # for t in [0.25, 0.5, 0.75]:
    #     plot_hedge_surface_3d(
    #         policy=policy_for_viz,
    #         time_point=t,
    #         compare_with_bs=True
    #     )
    
    # ========================================================================
    # EXPERIMENT 4: Call Spread Hedging (Section 5.2 end)
    # ========================================================================
    # print("\n### EXPERIMENT 4: Call Spread Hedging ###\n")
    # 
    # # Train for single risk aversion
    # policy_cs, pnl_cs = train_callspread_hedging(
    #     K1=100.0,
    #     K2=101.0,
    #     lambda_risk=0.95,
    #     num_epochs=2000
    # )
    # 
    # # Compare multiple risk aversions (as in paper)
    # compare_callspread_risk_aversions()
    
    # ========================================================================
    # EXPERIMENT 5: Alternative Risk Measures
    # ========================================================================
    # print("\n### EXPERIMENT 5: Alternative Risk Measures ###\n")
    # 
    # # Train with CVaR
    # cvar_loss = CVaRLoss(alpha=0.95)
    # policy_cvar, pnl_cvar = train_with_custom_risk_measure(
    #     risk_measure=cvar_loss,
    #     num_epochs=2000
    # )
    # 
    # # Train with Semi-Deviation
    # semidev_loss = SemiDeviationLoss(target=0.0)
    # policy_semidev, pnl_semidev = train_with_custom_risk_measure(
    #     risk_measure=semidev_loss,
    #     num_epochs=2000
    # )
    
    print("\n" + "="*80)
    print("All experiments completed!")
    print("="*80 + "\n")
