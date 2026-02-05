"""Recurrent neural network policy for hedging."""

from typing import Tuple, Optional
import torch
import torch.nn as nn
from .base_policy import BasePolicy


class RecurrentPolicy(BasePolicy):
    """
    Recurrent neural network policy for hedging using LSTM.
    
    This policy uses LSTM layers to capture temporal dependencies in the
    hedging problem. It maintains a hidden state across time steps, allowing
    it to learn from the history of states and actions.
    
    This is the architecture used in the original Deep Hedging paper.
    
    Example:
        >>> policy = RecurrentPolicy(state_dim=3, action_dim=1, hidden_dim=64)
        >>> batch_size, seq_len = 32, 30
        >>> states = torch.randn(batch_size, seq_len, 3)
        >>> 
        >>> hidden = policy.reset_hidden(batch_size)
        >>> actions = []
        >>> for t in range(seq_len):
        >>>     action, hidden = policy(states[:, t, :], hidden)
        >>>     actions.append(action)
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.0,
        bidirectional: bool = False,
        device: str = "cpu"
    ):
        """
        Initialize recurrent policy.
        
        Args:
            state_dim: Dimension of state vector
            action_dim: Dimension of action vector
            hidden_dim: Size of LSTM hidden state
            num_layers: Number of LSTM layers
            dropout: Dropout probability between LSTM layers
            bidirectional: Whether to use bidirectional LSTM
            device: Device to run on
        """
        super().__init__(state_dim, action_dim, hidden_dim, device)
        
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=state_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Output layer
        lstm_output_dim = hidden_dim * self.num_directions
        self.fc = nn.Linear(lstm_output_dim, action_dim)
        
        # Initialize weights
        self._initialize_weights()
        
        self.to(device)
    
    def _initialize_weights(self):
        """Initialize LSTM and linear layer weights."""
        # Initialize LSTM weights
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.zeros_(param.data)
        
        # Initialize output layer
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
    
    def forward(
        self,
        state: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the recurrent network.
        
        Args:
            state: State tensor of shape (batch_size, state_dim)
            hidden: Tuple of (h_0, c_0) hidden states, each of shape
                   (num_layers * num_directions, batch_size, hidden_dim)
                   If None, initializes to zeros
            
        Returns:
            action: Action tensor of shape (batch_size, action_dim)
            new_hidden: Updated hidden state tuple (h_n, c_n)
        """
        # Add sequence dimension if needed
        if state.dim() == 2:
            state = state.unsqueeze(1)  # (batch, 1, state_dim)
        
        # Initialize hidden state if not provided
        if hidden is None:
            hidden = self.reset_hidden(state.size(0))
        
        # LSTM forward pass
        lstm_out, new_hidden = self.lstm(state, hidden)
        
        # Take the last output (or only output if seq_len=1)
        lstm_out = lstm_out[:, -1, :]  # (batch, hidden_dim * num_directions)
        
        # Output layer
        action = self.fc(lstm_out)
        
        return action, new_hidden
    
    def reset_hidden(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize hidden state for LSTM.
        
        Args:
            batch_size: Size of the batch
            
        Returns:
            Tuple of (h_0, c_0) initialized to zeros
        """
        h_0 = torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_dim,
            device=self.device
        )
        c_0 = torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_dim,
            device=self.device
        )
        return (h_0, c_0)
    
    def __repr__(self) -> str:
        return (
            f"RecurrentPolicy(\n"
            f"  state_dim={self.state_dim},\n"
            f"  action_dim={self.action_dim},\n"
            f"  hidden_dim={self.hidden_dim},\n"
            f"  num_layers={self.num_layers},\n"
            f"  bidirectional={self.bidirectional},\n"
            f"  num_parameters={self.get_num_parameters()}\n"
            f")"
        )
