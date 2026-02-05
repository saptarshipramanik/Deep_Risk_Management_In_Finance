"""Training configuration dataclass."""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class TrainingConfig:
    """
    Configuration for deep hedging training.
    
    Example:
        >>> config = TrainingConfig(
        ...     num_epochs=1000,
        ...     batch_size=2048,
        ...     learning_rate=1e-4
        ... )
    """
    
    # Training hyperparameters
    num_epochs: int = 2000
    batch_size: int = 1024
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    grad_clip: float = 5.0
    
    # Market parameters
    S0: float = 100.0
    K: float = 100.0
    T: float = 1.0
    N: int = 30  # Number of hedging steps
    sigma: float = 0.2
    mu: float = 0.0
    
    # Risk parameters
    lambda_risk: float = 1.0  # Risk aversion for entropic risk
    alpha_cvar: float = 0.05  # Confidence level for CVaR
    lambda_var: float = 0.5   # Variance penalty weight
    
    # Transaction costs
    transaction_cost: float = 0.0  # Proportional cost
    
    # Model parameters
    hidden_dims: list = field(default_factory=lambda: [64, 64])
    policy_type: str = "feedforward"  # 'feedforward' or 'recurrent'
    
    # Loss function
    loss_type: str = "entropic"  # 'entropic', 'cvar', 'variance'
    
    # Device
    device: str = "cpu"
    
    # Logging
    log_interval: int = 100
    save_interval: int = 500
    
    # Paths
    save_dir: str = "checkpoints"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_')
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """Create config from dictionary."""
        return cls(**config_dict)
