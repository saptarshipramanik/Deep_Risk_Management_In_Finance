"""Benchmark hedging strategies for comparison."""

import numpy as np
import torch
from abc import ABC, abstractmethod
from typing import Tuple

from ..market import EuropeanOption


class HedgingBenchmark(ABC):
    """Base class for benchmark hedging strategies."""
    
    @abstractmethod
    def hedge(
        self,
        S: np.ndarray,
        t: np.ndarray,
        option: EuropeanOption
    ) -> np.ndarray:
        """
        Compute hedging positions.
        
        Args:
            S: Stock prices, shape (num_paths, num_steps)
            t: Time points, shape (num_steps,)
            option: Option to hedge
            
        Returns:
            Hedging positions, shape (num_paths, num_steps-1)
        """
        pass


class DeltaHedging(HedgingBenchmark):
    """
    Black-Scholes delta hedging strategy.
    
    Uses the analytical Black-Scholes delta as the hedging ratio.
    
    Example:
        >>> from deep_hedging.market import EuropeanOption
        >>> option = EuropeanOption(K=100, T=1.0)
        >>> delta_hedge = DeltaHedging(sigma=0.2)
        >>> 
        >>> S = np.random.lognormal(size=(1000, 31)) * 100
        >>> t = np.linspace(0, 1, 31)
        >>> deltas = delta_hedge.hedge(S, t, option)
    """
    
    def __init__(self, sigma: float, r: float = 0.0):
        """
        Initialize delta hedging.
        
        Args:
            sigma: Volatility parameter
            r: Risk-free rate
        """
        self.sigma = sigma
        self.r = r
    
    def hedge(
        self,
        S: np.ndarray,
        t: np.ndarray,
        option: EuropeanOption
    ) -> np.ndarray:
        """
        Compute Black-Scholes delta hedges.
        
        Args:
            S: Stock prices, shape (num_paths, num_steps)
            t: Time points, shape (num_steps,)
            option: Option to hedge
            
        Returns:
            Delta hedges, shape (num_paths, num_steps-1)
        """
        num_paths, num_steps = S.shape
        deltas = np.zeros((num_paths, num_steps - 1))
        
        for i in range(num_steps - 1):
            for j in range(num_paths):
                deltas[j, i] = option.bs_delta(
                    S=S[j, i],
                    t=t[i],
                    sigma=self.sigma,
                    r=self.r
                )
        
        return deltas
    
    def compute_pnl(
        self,
        S: np.ndarray,
        option: EuropeanOption,
        transaction_cost: float = 0.0
    ) -> np.ndarray:
        """
        Compute P&L from delta hedging.
        
        Args:
            S: Stock prices, shape (num_paths, num_steps)
            option: Option to hedge
            transaction_cost: Proportional transaction cost
            
        Returns:
            P&L for each path, shape (num_paths,)
        """
        num_paths, num_steps = S.shape
        t = np.linspace(0, option.T, num_steps)
        
        # Get delta hedges
        deltas = self.hedge(S, t, option)
        
        # Trading gains
        price_changes = S[:, 1:] - S[:, :-1]
        trading_gains = (deltas * price_changes).sum(axis=1)
        
        # Transaction costs
        if transaction_cost > 0:
            delta_changes = np.abs(np.diff(deltas, axis=1, prepend=0))
            costs = (transaction_cost * delta_changes * S[:, :-1]).sum(axis=1)
        else:
            costs = 0
        
        # Option payoff
        payoffs = option.payoff(torch.tensor(S[:, -1])).numpy()
        
        # P&L = trading gains - costs - payoff
        pnl = trading_gains - costs - payoffs
        
        return pnl


class StaticHedging(HedgingBenchmark):
    """
    Static hedging strategy (buy and hold).
    
    Computes initial delta and holds it constant.
    """
    
    def __init__(self, sigma: float, r: float = 0.0):
        self.sigma = sigma
        self.r = r
    
    def hedge(
        self,
        S: np.ndarray,
        t: np.ndarray,
        option: EuropeanOption
    ) -> np.ndarray:
        """Compute static hedge (constant delta)."""
        num_paths, num_steps = S.shape
        
        # Compute initial delta
        initial_deltas = np.array([
            option.bs_delta(S[i, 0], t[0], self.sigma, self.r)
            for i in range(num_paths)
        ])
        
        # Repeat for all time steps
        deltas = np.tile(initial_deltas[:, np.newaxis], (1, num_steps - 1))
        
        return deltas
