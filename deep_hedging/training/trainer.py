"""Trainer for deep hedging models."""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import numpy as np
from tqdm import tqdm

from ..models import BasePolicy
from ..market import GBMSimulator, HestonSimulator, EuropeanOption
from .loss_functions import EntropicRisk, CVaRLoss, VariancePenalty, TransactionCostPenalty
from .config import TrainingConfig


class Trainer:
    """
    Trainer for deep hedging policies.
    
    Handles the complete training loop including:
    - Path simulation
    - Hedging strategy execution
    - P&L calculation
    - Risk-based loss computation
    - Optimization
    
    Example:
        >>> from deep_hedging import RecurrentPolicy, Trainer, TrainingConfig
        >>> 
        >>> config = TrainingConfig(num_epochs=1000, batch_size=2048)
        >>> policy = RecurrentPolicy(state_dim=3, action_dim=1)
        >>> trainer = Trainer(policy, config)
        >>> 
        >>> history = trainer.train()
        >>> trainer.save_checkpoint("model.pt")
    """
    
    def __init__(
        self,
        policy: BasePolicy,
        config: TrainingConfig
    ):
        """
        Initialize trainer.
        
        Args:
            policy: Hedging policy network
            config: Training configuration
        """
        self.policy = policy
        self.config = config
        self.device = config.device
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.policy.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Initialize market simulator
        self.simulator = GBMSimulator(device=self.device)
        
        # Initialize option
        self.option = EuropeanOption(K=config.K, T=config.T, option_type="call")
        
        # Initialize loss function
        self.loss_fn = self._get_loss_function()
        
        # Initialize transaction cost penalty if needed
        if config.transaction_cost > 0:
            self.tc_penalty = TransactionCostPenalty(config.transaction_cost)
        else:
            self.tc_penalty = None
        
        # Training history
        self.history = {
            'loss': [],
            'mean_pnl': [],
            'std_pnl': [],
            'cvar': []
        }
    
    def _get_loss_function(self) -> nn.Module:
        """Get loss function based on config."""
        if self.config.loss_type == "entropic":
            return EntropicRisk(lambda_risk=self.config.lambda_risk)
        elif self.config.loss_type == "cvar":
            return CVaRLoss(alpha=self.config.alpha_cvar)
        elif self.config.loss_type == "variance":
            return VariancePenalty(lambda_var=self.config.lambda_var)
        else:
            raise ValueError(f"Unknown loss type: {self.config.loss_type}")
    
    def simulate_paths(self) -> torch.Tensor:
        """Simulate price paths."""
        return self.simulator.simulate(
            S0=self.config.S0,
            T=self.config.T,
            N=self.config.N,
            M=self.config.batch_size,
            mu=self.config.mu,
            sigma=self.config.sigma
        )
    
    def compute_pnl(
        self,
        S: torch.Tensor,
        deltas: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute P&L from hedging strategy.
        
        Args:
            S: Price paths of shape (batch, N+1)
            deltas: Hedging positions of shape (batch, N)
            
        Returns:
            P&L for each path, shape (batch,)
        """
        # Trading gains from hedging
        trading_gain = (deltas * (S[:, 1:] - S[:, :-1])).sum(dim=1)
        
        # Option payoff at maturity
        payoff = self.option.payoff(S[:, -1])
        
        # P&L = trading gains - option payoff
        pnl = trading_gain - payoff
        
        # Subtract transaction costs if applicable
        if self.tc_penalty is not None:
            tc = self.tc_penalty.compute_costs(deltas, S[:, :-1])
            pnl = pnl - tc
        
        return pnl
    
    def train_step(self) -> Tuple[float, Dict[str, float]]:
        """
        Execute one training step.
        
        Returns:
            loss: Loss value
            metrics: Dictionary of metrics
        """
        self.policy.train()
        self.optimizer.zero_grad()
        
        # Simulate paths
        S = self.simulate_paths()
        
        # Execute hedging strategy
        deltas = []
        hidden = self.policy.reset_hidden(self.config.batch_size)
        
        for t in range(self.config.N):
            # Prepare state: (normalized_price, time_to_maturity)
            time_to_maturity = (self.config.N - t) / self.config.N
            state = torch.stack([
                S[:, t] / self.config.S0,
                torch.full((self.config.batch_size,), time_to_maturity, device=self.device)
            ], dim=1)
            
            # Get action from policy
            delta, hidden = self.policy(state, hidden)
            deltas.append(delta.squeeze())
        
        deltas = torch.stack(deltas, dim=1)
        
        # Compute P&L
        pnl = self.compute_pnl(S, deltas)
        
        # Compute loss
        loss = self.loss_fn(pnl)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if self.config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(),
                self.config.grad_clip
            )
        
        self.optimizer.step()
        
        # Compute metrics
        with torch.no_grad():
            metrics = {
                'mean_pnl': pnl.mean().item(),
                'std_pnl': pnl.std().item(),
                'cvar_5': -torch.quantile(pnl, 0.05).item()
            }
        
        return loss.item(), metrics
    
    def train(self) -> Dict[str, List[float]]:
        """
        Train the policy.
        
        Returns:
            Training history dictionary
        """
        print(f"Training {self.policy.__class__.__name__} for {self.config.num_epochs} epochs")
        print(f"Batch size: {self.config.batch_size}, Learning rate: {self.config.learning_rate}")
        print(f"Loss function: {self.config.loss_type}")
        print("-" * 70)
        
        pbar = tqdm(range(self.config.num_epochs), desc="Training")
        
        for epoch in pbar:
            loss, metrics = self.train_step()
            
            # Record history
            self.history['loss'].append(loss)
            self.history['mean_pnl'].append(metrics['mean_pnl'])
            self.history['std_pnl'].append(metrics['std_pnl'])
            self.history['cvar'].append(metrics['cvar_5'])
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss:.4f}",
                'mean_pnl': f"{metrics['mean_pnl']:.2f}",
                'std_pnl': f"{metrics['std_pnl']:.2f}"
            })
            
            # Logging
            if (epoch + 1) % self.config.log_interval == 0:
                print(f"\nEpoch {epoch+1}/{self.config.num_epochs}")
                print(f"  Loss: {loss:.4f}")
                print(f"  Mean P&L: {metrics['mean_pnl']:.4f}")
                print(f"  Std P&L: {metrics['std_pnl']:.4f}")
                print(f"  5% CVaR: {metrics['cvar_5']:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % self.config.save_interval == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch+1}.pt")
        
        print("\nTraining completed!")
        return self.history
    
    def evaluate(self, num_paths: int = 10000) -> Dict[str, float]:
        """
        Evaluate the trained policy.
        
        Args:
            num_paths: Number of paths for evaluation
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.policy.eval()
        
        with torch.no_grad():
            # Simulate paths
            old_batch_size = self.config.batch_size
            self.config.batch_size = num_paths
            S = self.simulate_paths()
            self.config.batch_size = old_batch_size
            
            # Execute hedging strategy
            deltas = []
            hidden = self.policy.reset_hidden(num_paths)
            
            for t in range(self.config.N):
                time_to_maturity = (self.config.N - t) / self.config.N
                state = torch.stack([
                    S[:, t] / self.config.S0,
                    torch.full((num_paths,), time_to_maturity, device=self.device)
                ], dim=1)
                
                delta, hidden = self.policy(state, hidden)
                deltas.append(delta.squeeze())
            
            deltas = torch.stack(deltas, dim=1)
            
            # Compute P&L
            pnl = self.compute_pnl(S, deltas)
            
            # Compute comprehensive metrics using RiskMetrics
            pnl_np = pnl.cpu().numpy()
            
            # Import RiskMetrics here to avoid circular import
            from ..evaluation import RiskMetrics
            metrics = RiskMetrics.compute_all(pnl_np)
        
        return metrics, pnl_np, S.cpu().numpy(), deltas.cpu().numpy()

    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        Path(self.config.save_dir).mkdir(exist_ok=True)
        filepath = Path(self.config.save_dir) / filename
        
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.to_dict(),
            'history': self.history
        }, filepath)
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        filepath = Path(self.config.save_dir) / filename
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
