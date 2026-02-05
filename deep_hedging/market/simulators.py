"""Market simulators for generating price paths."""

from abc import ABC, abstractmethod
from typing import Tuple
import torch
import numpy as np


class BaseSimulator(ABC):
    """Base class for market simulators."""
    
    def __init__(self, device: str = "cpu"):
        self.device = device
    
    @abstractmethod
    def simulate(
        self,
        S0: float,
        T: float,
        N: int,
        M: int,
        **kwargs
    ) -> torch.Tensor:
        """
        Simulate price paths.
        
        Args:
            S0: Initial price
            T: Time to maturity
            N: Number of time steps
            M: Number of paths
            **kwargs: Additional parameters
            
        Returns:
            Tensor of shape (M, N+1) containing price paths
        """
        pass


class GBMSimulator(BaseSimulator):
    """
    Geometric Brownian Motion simulator.
    
    Simulates asset prices following:
        dS_t = μ S_t dt + σ S_t dW_t
    
    Example:
        >>> sim = GBMSimulator()
        >>> paths = sim.simulate(S0=100, T=1.0, N=252, M=10000, mu=0.05, sigma=0.2)
        >>> paths.shape
        torch.Size([10000, 253])
    """
    
    def simulate(
        self,
        S0: float,
        T: float,
        N: int,
        M: int,
        mu: float = 0.0,
        sigma: float = 0.2,
        seed: int = None
    ) -> torch.Tensor:
        """
        Simulate GBM paths.
        
        Args:
            S0: Initial stock price
            T: Time to maturity (in years)
            N: Number of time steps
            M: Number of simulation paths
            mu: Drift parameter
            sigma: Volatility parameter
            seed: Random seed for reproducibility
            
        Returns:
            Tensor of shape (M, N+1) with simulated paths
        """
        if seed is not None:
            torch.manual_seed(seed)
        
        dt = T / N
        S = torch.zeros(M, N + 1, device=self.device)
        S[:, 0] = S0
        
        for k in range(N):
            Z = torch.randn(M, device=self.device)
            S[:, k+1] = S[:, k] * torch.exp(
                (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
            )
        
        return S


class HestonSimulator(BaseSimulator):
    """
    Heston stochastic volatility model simulator.
    
    Simulates asset prices and variance following:
        dS_t = μ S_t dt + √v_t S_t dW_t^S
        dv_t = κ(θ - v_t)dt + ξ√v_t dW_t^v
        
    where dW_t^S and dW_t^v have correlation ρ.
    
    Example:
        >>> sim = HestonSimulator()
        >>> paths, variance = sim.simulate(
        ...     S0=100, v0=0.04, T=1.0, N=252, M=10000,
        ...     kappa=2.0, theta=0.04, xi=0.3, rho=-0.7
        ... )
    """
    
    def simulate(
        self,
        S0: float,
        v0: float,
        T: float,
        N: int,
        M: int,
        mu: float = 0.0,
        kappa: float = 2.0,
        theta: float = 0.04,
        xi: float = 0.3,
        rho: float = -0.7,
        seed: int = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Simulate Heston model paths.
        
        Args:
            S0: Initial stock price
            v0: Initial variance
            T: Time to maturity
            N: Number of time steps
            M: Number of paths
            mu: Drift
            kappa: Mean reversion speed
            theta: Long-term variance
            xi: Volatility of volatility
            rho: Correlation between price and variance
            seed: Random seed
            
        Returns:
            Tuple of (price_paths, variance_paths), each of shape (M, N+1)
        """
        if seed is not None:
            torch.manual_seed(seed)
        
        dt = T / N
        
        S = torch.zeros(M, N + 1, device=self.device)
        v = torch.zeros(M, N + 1, device=self.device)
        
        S[:, 0] = S0
        v[:, 0] = v0
        
        for k in range(N):
            # Generate correlated random variables
            Z1 = torch.randn(M, device=self.device)
            Z2 = torch.randn(M, device=self.device)
            
            W_S = Z1
            W_v = rho * Z1 + np.sqrt(1 - rho**2) * Z2
            
            # Update variance (with Feller condition handling)
            v_next = v[:, k] + kappa * (theta - v[:, k]) * dt + \
                     xi * torch.sqrt(torch.clamp(v[:, k], min=0)) * np.sqrt(dt) * W_v
            v[:, k+1] = torch.clamp(v_next, min=0)  # Ensure non-negative variance
            
            # Update price
            S[:, k+1] = S[:, k] * torch.exp(
                (mu - 0.5 * v[:, k]) * dt + 
                torch.sqrt(torch.clamp(v[:, k], min=0)) * np.sqrt(dt) * W_S
            )
        
        return S, v
