"""Financial instruments for hedging."""

import torch
import numpy as np
from scipy.stats import norm


class EuropeanOption:
    """
    European option pricing and Greeks.
    
    Supports both call and put options with Black-Scholes pricing.
    """
    
    def __init__(self, K: float, T: float, option_type: str = "call"):
        """
        Initialize European option.
        
        Args:
            K: Strike price
            T: Time to maturity
            option_type: 'call' or 'put'
        """
        self.K = K
        self.T = T
        self.option_type = option_type.lower()
        
        if self.option_type not in ['call', 'put']:
            raise ValueError("option_type must be 'call' or 'put'")
    
    def payoff(self, S: torch.Tensor) -> torch.Tensor:
        """
        Calculate option payoff at maturity.
        
        Args:
            S: Stock price at maturity
            
        Returns:
            Option payoff
        """
        if self.option_type == 'call':
            return torch.clamp(S - self.K, min=0.0)
        else:
            return torch.clamp(self.K - S, min=0.0)
    
    def bs_price(
        self,
        S: float,
        t: float,
        sigma: float,
        r: float = 0.0
    ) -> float:
        """
        Black-Scholes option price.
        
        Args:
            S: Current stock price
            t: Current time
            sigma: Volatility
            r: Risk-free rate
            
        Returns:
            Option price
        """
        tau = self.T - t
        if tau <= 0:
            return float(self.payoff(torch.tensor(S)))
        
        d1 = (np.log(S / self.K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
        d2 = d1 - sigma * np.sqrt(tau)
        
        if self.option_type == 'call':
            price = S * norm.cdf(d1) - self.K * np.exp(-r * tau) * norm.cdf(d2)
        else:
            price = self.K * np.exp(-r * tau) * norm.cdf(-d2) - S * norm.cdf(-d1)
        
        return price
    
    def bs_delta(
        self,
        S: float,
        t: float,
        sigma: float,
        r: float = 0.0
    ) -> float:
        """
        Black-Scholes delta (hedge ratio).
        
        Args:
            S: Current stock price
            t: Current time
            sigma: Volatility
            r: Risk-free rate
            
        Returns:
            Delta (∂V/∂S)
        """
        tau = self.T - t
        if tau <= 0:
            if self.option_type == 'call':
                return 1.0 if S > self.K else 0.0
            else:
                return -1.0 if S < self.K else 0.0
        
        d1 = (np.log(S / self.K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
        
        if self.option_type == 'call':
            return norm.cdf(d1)
        else:
            return norm.cdf(d1) - 1.0
    
    def bs_gamma(
        self,
        S: float,
        t: float,
        sigma: float,
        r: float = 0.0
    ) -> float:
        """
        Black-Scholes gamma (second derivative).
        
        Args:
            S: Current stock price
            t: Current time
            sigma: Volatility
            r: Risk-free rate
            
        Returns:
            Gamma (∂²V/∂S²)
        """
        tau = self.T - t
        if tau <= 0:
            return 0.0
        
        d1 = (np.log(S / self.K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
        
        return norm.pdf(d1) / (S * sigma * np.sqrt(tau))
    
    def bs_vega(
        self,
        S: float,
        t: float,
        sigma: float,
        r: float = 0.0
    ) -> float:
        """
        Black-Scholes vega (sensitivity to volatility).
        
        Args:
            S: Current stock price
            t: Current time
            sigma: Volatility
            r: Risk-free rate
            
        Returns:
            Vega (∂V/∂σ)
        """
        tau = self.T - t
        if tau <= 0:
            return 0.0
        
        d1 = (np.log(S / self.K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
        
        return S * norm.pdf(d1) * np.sqrt(tau)
