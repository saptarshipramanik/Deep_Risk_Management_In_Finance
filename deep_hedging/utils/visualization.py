"""Visualization utilities for deep hedging."""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional


def plot_training_history(
    history: Dict[str, List[float]],
    figsize: tuple = (15, 10)
):
    """
    Plot training history.
    
    Args:
        history: Dictionary with 'loss', 'mean_pnl', 'std_pnl', 'cvar'
        figsize: Figure size
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Loss
    axes[0, 0].plot(history['loss'])
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Mean P&L
    axes[0, 1].plot(history['mean_pnl'])
    axes[0, 1].set_title('Mean P&L')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Mean P&L')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    # Std P&L
    axes[1, 0].plot(history['std_pnl'])
    axes[1, 0].set_title('P&L Standard Deviation')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Std P&L')
    axes[1, 0].grid(True, alpha=0.3)
    
    # CVaR
    axes[1, 1].plot(history['cvar'])
    axes[1, 1].set_title('5% CVaR')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('CVaR')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_pnl_comparison(
    pnl_deep: np.ndarray,
    pnl_delta: np.ndarray,
    bins: int = 50,
    figsize: tuple = (12, 6)
):
    """
    Compare P&L distributions.
    
    Args:
        pnl_deep: P&L from deep hedging
        pnl_delta: P&L from delta hedging
        bins: Number of histogram bins
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Histogram comparison
    axes[0].hist(pnl_deep, bins=bins, alpha=0.6, label='Deep Hedging', density=True)
    axes[0].hist(pnl_delta, bins=bins, alpha=0.6, label='Delta Hedging', density=True)
    axes[0].set_xlabel('P&L')
    axes[0].set_ylabel('Density')
    axes[0].set_title('P&L Distribution Comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Q-Q plot
    quantiles = np.linspace(0, 1, 100)
    q_deep = np.quantile(pnl_deep, quantiles)
    q_delta = np.quantile(pnl_delta, quantiles)
    
    axes[1].scatter(q_delta, q_deep, alpha=0.5, s=20)
    axes[1].plot([q_delta.min(), q_delta.max()], 
                 [q_delta.min(), q_delta.max()], 
                 'r--', label='y=x')
    axes[1].set_xlabel('Delta Hedging P&L Quantiles')
    axes[1].set_ylabel('Deep Hedging P&L Quantiles')
    axes[1].set_title('Q-Q Plot')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_delta_comparison(
    S: np.ndarray,
    delta_deep: np.ndarray,
    delta_bs: np.ndarray,
    path_idx: int = 0,
    figsize: tuple = (12, 5)
):
    """
    Compare hedging deltas along a price path.
    
    Args:
        S: Stock price path, shape (num_paths, num_steps)
        delta_deep: Deep hedging deltas, shape (num_paths, num_steps-1)
        delta_bs: Black-Scholes deltas, shape (num_paths, num_steps-1)
        path_idx: Index of path to plot
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    time_steps = np.arange(len(delta_deep[path_idx]))
    
    # Delta comparison
    axes[0].plot(time_steps, delta_deep[path_idx], label='Deep Hedging', marker='o', markersize=3)
    axes[0].plot(time_steps, delta_bs[path_idx], label='Black-Scholes', marker='s', markersize=3)
    axes[0].set_xlabel('Time Step')
    axes[0].set_ylabel('Delta')
    axes[0].set_title('Hedging Delta Comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Price path
    axes[1].plot(S[path_idx], label='Stock Price')
    axes[1].set_xlabel('Time Step')
    axes[1].set_ylabel('Price')
    axes[1].set_title('Stock Price Path')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_risk_comparison(
    metrics_deep: Dict[str, float],
    metrics_delta: Dict[str, float],
    figsize: tuple = (10, 6)
):
    """
    Compare risk metrics between strategies.
    
    Args:
        metrics_deep: Metrics from deep hedging
        metrics_delta: Metrics from delta hedging
        figsize: Figure size
    """
    # Select key metrics to compare
    keys = ['mean', 'std', 'cvar_1', 'cvar_5', 'sharpe']
    labels = ['Mean', 'Std Dev', '1% CVaR', '5% CVaR', 'Sharpe']
    
    deep_values = [metrics_deep.get(k, 0) for k in keys]
    delta_values = [metrics_delta.get(k, 0) for k in keys]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x - width/2, deep_values, width, label='Deep Hedging', alpha=0.8)
    ax.bar(x + width/2, delta_values, width, label='Delta Hedging', alpha=0.8)
    
    ax.set_xlabel('Metric')
    ax.set_ylabel('Value')
    ax.set_title('Risk Metrics Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
