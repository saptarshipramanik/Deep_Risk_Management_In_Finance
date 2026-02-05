"""Base class for hedging policies."""

from abc import ABC, abstractmethod
from typing import Tuple, Optional
import torch
import torch.nn as nn


class BasePolicy(nn.Module, ABC):
    """
    Abstract base class for hedging policies.
    
    A hedging policy maps the current state (market information, time, portfolio)
    to a hedging action (position in hedging instruments).
    
    Attributes:
        state_dim (int): Dimension of the state space
        action_dim (int): Dimension of the action space
        hidden_dim (int): Dimension of hidden layers
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        device: str = "cpu"
    ):
        """
        Initialize the policy.
        
        Args:
            state_dim: Dimension of state vector
            action_dim: Dimension of action vector (number of hedging instruments)
            hidden_dim: Size of hidden layers
            device: Device to run the model on ('cpu' or 'cuda')
        """
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.device = device
        
    @abstractmethod
    def forward(
        self,
        state: torch.Tensor,
        hidden: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through the policy network.
        
        Args:
            state: Current state tensor of shape (batch_size, state_dim)
            hidden: Hidden state for recurrent policies (optional)
            
        Returns:
            action: Hedging action of shape (batch_size, action_dim)
            new_hidden: Updated hidden state (None for feedforward policies)
        """
        pass
    
    def reset_hidden(self, batch_size: int) -> Optional[torch.Tensor]:
        """
        Reset hidden state for recurrent policies.
        
        Args:
            batch_size: Size of the batch
            
        Returns:
            Initial hidden state or None for feedforward policies
        """
        return None
    
    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def save(self, path: str):
        """Save model weights."""
        torch.save(self.state_dict(), path)
    
    def load(self, path: str):
        """Load model weights."""
        self.load_state_dict(torch.load(path, map_location=self.device))
