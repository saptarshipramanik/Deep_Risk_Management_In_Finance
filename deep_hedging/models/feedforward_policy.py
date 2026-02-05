"""Feedforward neural network policy for hedging."""

from typing import Tuple, Optional, List
import torch
import torch.nn as nn
from .base_policy import BasePolicy


class FeedForwardPolicy(BasePolicy):
    """
    Feedforward neural network policy for hedging.
    
    This is a simple multi-layer perceptron that maps states to actions.
    Suitable for Markovian environments where the current state contains
    all necessary information.
    
    Example:
        >>> policy = FeedForwardPolicy(state_dim=3, action_dim=1, hidden_dims=[64, 64])
        >>> state = torch.randn(32, 3)  # batch_size=32
        >>> action, _ = policy(state)
        >>> action.shape
        torch.Size([32, 1])
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [64, 64],
        activation: str = "relu",
        dropout: float = 0.0,
        batch_norm: bool = False,
        device: str = "cpu"
    ):
        """
        Initialize feedforward policy.
        
        Args:
            state_dim: Dimension of state vector
            action_dim: Dimension of action vector
            hidden_dims: List of hidden layer sizes
            activation: Activation function ('relu', 'tanh', 'elu')
            dropout: Dropout probability (0 = no dropout)
            batch_norm: Whether to use batch normalization
            device: Device to run on
        """
        super().__init__(state_dim, action_dim, hidden_dims[0] if hidden_dims else 64, device)
        
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.batch_norm = batch_norm
        
        # Select activation function
        activation_map = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "elu": nn.ELU(),
            "leaky_relu": nn.LeakyReLU()
        }
        self.activation = activation_map.get(activation.lower(), nn.ReLU())
        
        # Build network layers
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            layers.append(self.activation)
            
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            input_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(input_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
        
        self.to(device)
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        state: torch.Tensor,
        hidden: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through the network.
        
        Args:
            state: State tensor of shape (batch_size, state_dim)
            hidden: Not used for feedforward policy
            
        Returns:
            action: Action tensor of shape (batch_size, action_dim)
            None: No hidden state for feedforward policy
        """
        action = self.network(state)
        return action, None
    
    def __repr__(self) -> str:
        return (
            f"FeedForwardPolicy(\n"
            f"  state_dim={self.state_dim},\n"
            f"  action_dim={self.action_dim},\n"
            f"  hidden_dims={self.hidden_dims},\n"
            f"  num_parameters={self.get_num_parameters()}\n"
            f")"
        )
