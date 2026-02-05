"""Risk metrics for evaluating hedging performance."""

import numpy as np
from typing import Dict


class RiskMetrics:
    """
    Compute comprehensive risk metrics for P&L distributions.
    
    Example:
        >>> pnl = np.random.randn(10000)
        >>> metrics = RiskMetrics.compute_all(pnl)
        >>> print(f"Mean: {metrics['mean']:.2f}, CVaR 5%: {metrics['cvar_5']:.2f}")
    """
    
    @staticmethod
    def compute_all(pnl: np.ndarray) -> Dict[str, float]:
        """
        Compute all risk metrics.
        
        Args:
            pnl: P&L array
            
        Returns:
            Dictionary of metrics
        """
        return {
            # Central tendency
            'mean': np.mean(pnl),
            'median': np.median(pnl),
            
            # Dispersion
            'std': np.std(pnl),
            'var': np.var(pnl),
            
            # Extremes
            'min': np.min(pnl),
            'max': np.max(pnl),
            
            # Quantiles
            'q01': np.quantile(pnl, 0.01),
            'q05': np.quantile(pnl, 0.05),
            'q10': np.quantile(pnl, 0.10),
            'q25': np.quantile(pnl, 0.25),
            'q75': np.quantile(pnl, 0.75),
            'q90': np.quantile(pnl, 0.90),
            'q95': np.quantile(pnl, 0.95),
            'q99': np.quantile(pnl, 0.99),
            
            # Risk measures
            'cvar_1': RiskMetrics.cvar(pnl, 0.01),
            'cvar_5': RiskMetrics.cvar(pnl, 0.05),
            'cvar_10': RiskMetrics.cvar(pnl, 0.10),
            
            # Performance ratios
            'sharpe': RiskMetrics.sharpe_ratio(pnl),
            'sortino': RiskMetrics.sortino_ratio(pnl),
        }
    
    @staticmethod
    def cvar(pnl: np.ndarray, alpha: float) -> float:
        """
        Conditional Value at Risk (Expected Shortfall).
        
        Args:
            pnl: P&L array
            alpha: Confidence level
            
        Returns:
            CVaR value (negative of expected loss in worst α% cases)
        """
        var = np.quantile(pnl, alpha)
        return -np.mean(pnl[pnl <= var])
    
    @staticmethod
    def sharpe_ratio(pnl: np.ndarray) -> float:
        """
        Sharpe ratio (mean / std).
        
        Args:
            pnl: P&L array
            
        Returns:
            Sharpe ratio
        """
        if np.std(pnl) == 0:
            return 0.0
        return np.mean(pnl) / np.std(pnl)
    
    @staticmethod
    def sortino_ratio(pnl: np.ndarray) -> float:
        """
        Sortino ratio (mean / downside deviation).
        
        Args:
            pnl: P&L array
            
        Returns:
            Sortino ratio
        """
        downside = pnl[pnl < 0]
        if len(downside) == 0 or np.std(downside) == 0:
            return 0.0
        return np.mean(pnl) / np.std(downside)
    
    @staticmethod
    def print_metrics(metrics: Dict[str, float], title: str = "Risk Metrics"):
        """
        Print metrics in a formatted table.
        
        Args:
            metrics: Dictionary of metrics
            title: Title for the table
        """
        print(f"\n{title}")
        print("=" * 50)
        
        # Group metrics
        groups = {
            "Central Tendency": ['mean', 'median'],
            "Dispersion": ['std', 'var'],
            "Extremes": ['min', 'max'],
            "Quantiles": ['q01', 'q05', 'q10', 'q25', 'q75', 'q90', 'q95', 'q99'],
            "Risk Measures": ['cvar_1', 'cvar_5', 'cvar_10'],
            "Performance": ['sharpe', 'sortino']
        }
        
        for group_name, keys in groups.items():
            print(f"\n{group_name}:")
            for key in keys:
                if key in metrics:
                    print(f"  {key:12s}: {metrics[key]:10.4f}")
