"""Risk-based loss functions for deep hedging."""

import torch
import torch.nn as nn


class EntropicRisk(nn.Module):
    """
    Entropic risk measure for deep hedging.
    
    The entropic risk measure is defined as:
        ρ_λ(X) = (1/λ) * log(E[exp(-λ * X)])
    
    where λ > 0 is the risk aversion parameter. Higher λ means more risk averse.
    
    This is a coherent risk measure that penalizes both mean and variance of P&L.
    
    Example:
        >>> loss_fn = EntropicRisk(lambda_risk=1.0)
        >>> pnl = torch.randn(1000)  # P&L distribution
        >>> risk = loss_fn(pnl)
    """
    
    def __init__(self, lambda_risk: float = 1.0, normalize: bool = True):
        """
        Initialize entropic risk measure.
        
        Args:
            lambda_risk: Risk aversion parameter (λ > 0)
            normalize: Whether to normalize P&L by its std before computing risk
        """
        super().__init__()
        if lambda_risk <= 0:
            raise ValueError("lambda_risk must be positive")
        
        self.lambda_risk = lambda_risk
        self.normalize = normalize
    
    def forward(self, pnl: torch.Tensor) -> torch.Tensor:
        """
        Compute entropic risk.
        
        Args:
            pnl: Profit and loss tensor of shape (batch_size,)
            
        Returns:
            Scalar risk value
        """
        if self.normalize:
            # Normalize by standard deviation for numerical stability
            pnl_std = pnl.std().detach()
            if pnl_std > 0:
                pnl = pnl / pnl_std
        
        # Compute entropic risk: (1/λ) * log(E[exp(-λ * P&L)])
        # Use logsumexp for numerical stability
        risk = (torch.logsumexp(-self.lambda_risk * pnl, dim=0) - 
                torch.log(torch.tensor(pnl.size(0), dtype=pnl.dtype, device=pnl.device)))
        
        return risk / self.lambda_risk


class CVaRLoss(nn.Module):
    """
    Conditional Value at Risk (CVaR) loss.
    
    CVaR is the expected loss in the worst α% of cases.
    Also known as Expected Shortfall (ES) or Average Value at Risk (AVaR).
    
    CVaR_α(X) = E[X | X ≤ VaR_α(X)]
    
    Example:
        >>> loss_fn = CVaRLoss(alpha=0.05)  # 5% CVaR
        >>> pnl = torch.randn(1000)
        >>> cvar = loss_fn(pnl)
    """
    
    def __init__(self, alpha: float = 0.05):
        """
        Initialize CVaR loss.
        
        Args:
            alpha: Confidence level (e.g., 0.05 for 95% CVaR)
        """
        super().__init__()
        if not 0 < alpha < 1:
            raise ValueError("alpha must be between 0 and 1")
        
        self.alpha = alpha
    
    def forward(self, pnl: torch.Tensor) -> torch.Tensor:
        """
        Compute CVaR (minimize negative P&L in worst cases).
        
        Args:
            pnl: Profit and loss tensor of shape (batch_size,)
            
        Returns:
            CVaR value (negative for loss)
        """
        # Sort P&L in ascending order (worst to best)
        sorted_pnl, _ = torch.sort(pnl)
        
        # Number of samples in the tail
        n_tail = max(1, int(self.alpha * pnl.size(0)))
        
        # CVaR is the mean of the worst α% cases
        cvar = sorted_pnl[:n_tail].mean()
        
        return -cvar  # Negative because we want to minimize losses


class VariancePenalty(nn.Module):
    """
    Mean-variance objective for hedging.
    
    Combines mean P&L with variance penalty:
        Loss = -E[P&L] + λ * Var[P&L]
    
    Example:
        >>> loss_fn = VariancePenalty(lambda_var=0.5)
        >>> pnl = torch.randn(1000)
        >>> loss = loss_fn(pnl)
    """
    
    def __init__(self, lambda_var: float = 0.5):
        """
        Initialize variance penalty.
        
        Args:
            lambda_var: Weight for variance penalty (λ ≥ 0)
        """
        super().__init__()
        if lambda_var < 0:
            raise ValueError("lambda_var must be non-negative")
        
        self.lambda_var = lambda_var
    
    def forward(self, pnl: torch.Tensor) -> torch.Tensor:
        """
        Compute mean-variance loss.
        
        Args:
            pnl: Profit and loss tensor of shape (batch_size,)
            
        Returns:
            Mean-variance loss
        """
        mean_pnl = pnl.mean()
        var_pnl = pnl.var()
        
        # Minimize: -mean + λ * variance
        loss = -mean_pnl + self.lambda_var * var_pnl
        
        return loss


class TransactionCostPenalty(nn.Module):
    """
    Transaction cost penalty for hedging.
    
    Penalizes changes in hedging position:
        Cost = c * |Δ_t - Δ_{t-1}| * S_t
    
    where c is the proportional transaction cost.
    """
    
    def __init__(self, cost_rate: float = 0.001):
        """
        Initialize transaction cost penalty.
        
        Args:
            cost_rate: Proportional transaction cost (e.g., 0.001 = 0.1%)
        """
        super().__init__()
        if cost_rate < 0:
            raise ValueError("cost_rate must be non-negative")
        
        self.cost_rate = cost_rate
    
    def compute_costs(
        self,
        deltas: torch.Tensor,
        prices: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute transaction costs over time.
        
        Args:
            deltas: Hedging positions over time, shape (batch, time_steps)
            prices: Asset prices over time, shape (batch, time_steps)
            
        Returns:
            Total transaction costs, shape (batch,)
        """
        # Compute changes in position
        delta_changes = torch.abs(deltas[:, 1:] - deltas[:, :-1])
        
        # Transaction costs = cost_rate * |change| * price
        costs = self.cost_rate * delta_changes * prices[:, :-1]
        
        # Sum over time
        total_costs = costs.sum(dim=1)
        
        return total_costs
