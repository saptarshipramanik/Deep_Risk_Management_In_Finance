"""
Deep Hedging - Production-Ready Modular Architecture
====================================================

This file contains the modular, production-ready components that were
missing from the original notebook implementation.

These components enable:
1. Configuration management
2. Model persistence and loading
3. Logging infrastructure
4. Experiment tracking
5. Clean separation of concerns
6. Integration-ready interfaces

Author: Deep Hedging DDP Implementation
Date: February 2026
"""

import torch
import torch.nn as nn
import numpy as np
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
from abc import ABC, abstractmethod
import pickle


# ============================================================================
# CONFIGURATION MANAGEMENT
# ============================================================================

@dataclass
class ModelConfig:
    """
    Configuration for hedging policy neural network.
    
    Attributes:
        input_dim: Dimension of state space (e.g., 2 for [S/S0, time])
        output_dim: Dimension of action space (number of hedging instruments)
        hidden_layers: List of hidden layer dimensions
        activation: Activation function name ('relu', 'tanh', 'elu')
        dropout_rate: Dropout probability (0.0 = no dropout)
        batch_norm: Whether to use batch normalization
        zero_init: Whether to use zero initialization (recommended for hedging)
    """
    input_dim: int = 2
    output_dim: int = 1
    hidden_layers: List[int] = field(default_factory=lambda: [32, 32])
    activation: str = 'relu'
    dropout_rate: float = 0.0
    batch_norm: bool = False
    zero_init: bool = True


@dataclass
class MarketConfig:
    """
    Configuration for market dynamics simulator.
    
    Attributes:
        model_type: 'blackscholes' or 'heston'
        S0: Initial stock price
        T: Time horizon (years)
        N: Number of time steps
        
        # Black-Scholes parameters
        mu: Drift (typically 0 for risk-neutral)
        sigma: Volatility
        
        # Heston parameters
        v0: Initial variance
        kappa: Mean reversion speed
        theta: Long-term variance
        xi: Volatility of variance
        rho: Correlation between price and variance
    """
    model_type: str = 'heston'
    S0: float = 100.0
    T: float = 1.0
    N: int = 30
    
    # Black-Scholes
    mu: float = 0.0
    sigma: float = 0.2
    
    # Heston
    v0: float = 0.04
    kappa: float = 2.0
    theta: float = 0.04
    xi: float = 0.5
    rho: float = -0.7


@dataclass
class DerivativeConfig:
    """
    Configuration for derivative contract.
    
    Attributes:
        contract_type: 'call', 'put', 'callspread', 'digital', etc.
        strike: Strike price (K)
        strike2: Second strike (for spreads)
        maturity: Time to maturity (if different from market T)
    """
    contract_type: str = 'call'
    strike: float = 100.0
    strike2: Optional[float] = None
    maturity: Optional[float] = None


@dataclass
class RiskMeasureConfig:
    """
    Configuration for risk measure / objective function.
    
    Attributes:
        measure_type: 'entropic', 'cvar', 'variance', 'semi_deviation'
        lambda_risk: Risk aversion parameter (for entropic)
        alpha: Confidence level (for CVaR, e.g., 0.95)
        target: Target level (for semi-deviation)
    """
    measure_type: str = 'entropic'
    lambda_risk: float = 1.0
    alpha: float = 0.95
    target: float = 0.0


@dataclass
class TransactionCostConfig:
    """
    Configuration for transaction costs.
    
    Attributes:
        cost_type: 'proportional', 'fixed', 'none'
        rate: Proportional cost rate (ε)
        fixed_cost: Fixed cost per trade
    """
    cost_type: str = 'proportional'
    rate: float = 0.001
    fixed_cost: float = 0.0


@dataclass
class TrainingConfig:
    """
    Configuration for training procedure.
    
    Attributes:
        num_epochs: Number of training iterations
        batch_size: Number of paths per batch
        learning_rate: Initial learning rate
        optimizer: 'adam', 'sgd', 'rmsprop'
        grad_clip_norm: Gradient clipping threshold
        scheduler: Learning rate scheduler config
        early_stopping: Early stopping config
    """
    num_epochs: int = 2000
    batch_size: int = 1024
    learning_rate: float = 1e-4
    optimizer: str = 'adam'
    grad_clip_norm: float = 5.0
    scheduler: Optional[Dict[str, Any]] = None
    early_stopping: Optional[Dict[str, Any]] = None


@dataclass
class ExperimentConfig:
    """
    Complete experiment configuration combining all sub-configs.
    
    This is the main configuration object that fully specifies an experiment.
    """
    name: str = "deep_hedging_experiment"
    description: str = ""
    seed: int = 42
    device: str = "cuda"  # or "cpu"
    
    model: ModelConfig = field(default_factory=ModelConfig)
    market: MarketConfig = field(default_factory=MarketConfig)
    derivative: DerivativeConfig = field(default_factory=DerivativeConfig)
    risk_measure: RiskMeasureConfig = field(default_factory=RiskMeasureConfig)
    transaction_cost: TransactionCostConfig = field(default_factory=TransactionCostConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    # Output configuration
    output_dir: str = "./outputs"
    save_checkpoints: bool = True
    checkpoint_frequency: int = 500
    
    def save_yaml(self, filepath: str) -> None:
        """Save configuration to YAML file"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            yaml.dump(asdict(self), f, default_flow_style=False)
    
    @classmethod
    def load_yaml(cls, filepath: str) -> 'ExperimentConfig':
        """Load configuration from YAML file"""
        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Reconstruct nested dataclasses
        return cls(
            name=config_dict.get('name', "experiment"),
            description=config_dict.get('description', ""),
            seed=config_dict.get('seed', 42),
            device=config_dict.get('device', "cuda"),
            model=ModelConfig(**config_dict.get('model', {})),
            market=MarketConfig(**config_dict.get('market', {})),
            derivative=DerivativeConfig(**config_dict.get('derivative', {})),
            risk_measure=RiskMeasureConfig(**config_dict.get('risk_measure', {})),
            transaction_cost=TransactionCostConfig(**config_dict.get('transaction_cost', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            output_dir=config_dict.get('output_dir', "./outputs"),
            save_checkpoints=config_dict.get('save_checkpoints', True),
            checkpoint_frequency=config_dict.get('checkpoint_frequency', 500)
        )
    
    def save_json(self, filepath: str) -> None:
        """Save configuration to JSON file"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load_json(cls, filepath: str) -> 'ExperimentConfig':
        """Load configuration from JSON file"""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        return cls(
            name=config_dict.get('name', "experiment"),
            description=config_dict.get('description', ""),
            seed=config_dict.get('seed', 42),
            device=config_dict.get('device', "cuda"),
            model=ModelConfig(**config_dict.get('model', {})),
            market=MarketConfig(**config_dict.get('market', {})),
            derivative=DerivativeConfig(**config_dict.get('derivative', {})),
            risk_measure=RiskMeasureConfig(**config_dict.get('risk_measure', {})),
            transaction_cost=TransactionCostConfig(**config_dict.get('transaction_cost', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            output_dir=config_dict.get('output_dir', "./outputs"),
            save_checkpoints=config_dict.get('save_checkpoints', True),
            checkpoint_frequency=config_dict.get('checkpoint_frequency', 500)
        )


# ============================================================================
# LOGGING INFRASTRUCTURE
# ============================================================================

def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """
    Setup logger with both file and console handlers.
    
    Args:
        name: Logger name
        log_file: Path to log file (if None, only console logging)
        level: Logging level
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
    
    # Format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


class MetricsTracker:
    """
    Track and log training/evaluation metrics.
    
    This class maintains a history of metrics and provides utilities
    for saving, loading, and analyzing them.
    """
    
    def __init__(self):
        self.metrics: Dict[str, List[float]] = {}
        self.epoch_count = 0
    
    def log(self, **kwargs) -> None:
        """
        Log metrics for current epoch.
        
        Example:
            tracker.log(loss=0.123, pnl_mean=-1.5, pnl_std=2.3)
        """
        for key, value in kwargs.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(float(value))
        
        self.epoch_count += 1
    
    def get_metric(self, name: str) -> List[float]:
        """Get history of a specific metric"""
        return self.metrics.get(name, [])
    
    def get_latest(self, name: str) -> Optional[float]:
        """Get most recent value of a metric"""
        history = self.get_metric(name)
        return history[-1] if history else None
    
    def save(self, filepath: str) -> None:
        """Save metrics to JSON file"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump({
                'metrics': self.metrics,
                'epoch_count': self.epoch_count
            }, f, indent=2)
    
    def load(self, filepath: str) -> None:
        """Load metrics from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        self.metrics = data['metrics']
        self.epoch_count = data['epoch_count']
    
    def summary(self) -> Dict[str, Dict[str, float]]:
        """
        Compute summary statistics for all metrics.
        
        Returns:
            Dictionary mapping metric_name -> {mean, std, min, max, final}
        """
        summary = {}
        for name, values in self.metrics.items():
            if values:
                summary[name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'final': values[-1]
                }
        return summary


# ============================================================================
# MODEL PERSISTENCE
# ============================================================================

class ModelCheckpoint:
    """
    Handle saving and loading of model checkpoints.
    
    A checkpoint includes:
    - Model state dict
    - Optimizer state dict
    - Training configuration
    - Metrics history
    - Random state (for reproducibility)
    """
    
    def __init__(self, checkpoint_dir: str = "./checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: ExperimentConfig,
        metrics: MetricsTracker,
        epoch: int,
        additional_state: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save a checkpoint.
        
        Args:
            model: Neural network model
            optimizer: Optimizer
            config: Experiment configuration
            metrics: Metrics tracker
            epoch: Current epoch number
            additional_state: Any additional state to save
        
        Returns:
            Path to saved checkpoint
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': asdict(config),
            'metrics': metrics.metrics,
            'torch_rng_state': torch.get_rng_state(),
            'numpy_rng_state': np.random.get_state(),
        }
        
        if additional_state:
            checkpoint['additional_state'] = additional_state
        
        # Generate filename with timestamp and epoch
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"checkpoint_epoch{epoch:04d}_{timestamp}.pt"
        filepath = self.checkpoint_dir / filename
        
        torch.save(checkpoint, filepath)
        
        # Also save a "latest" checkpoint
        latest_path = self.checkpoint_dir / "checkpoint_latest.pt"
        torch.save(checkpoint, latest_path)
        
        return str(filepath)
    
    def load(
        self,
        filepath: str,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        restore_rng: bool = True
    ) -> Tuple[ExperimentConfig, MetricsTracker, int]:
        """
        Load a checkpoint.
        
        Args:
            filepath: Path to checkpoint file
            model: Model to load state into
            optimizer: Optimizer to load state into (optional)
            restore_rng: Whether to restore random state
        
        Returns:
            (config, metrics, epoch)
        """
        checkpoint = torch.load(filepath, map_location='cpu')
        
        # Restore model
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Restore optimizer if provided
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Restore RNG state if requested
        if restore_rng:
            if 'torch_rng_state' in checkpoint:
                torch.set_rng_state(checkpoint['torch_rng_state'])
            if 'numpy_rng_state' in checkpoint:
                np.random.set_state(checkpoint['numpy_rng_state'])
        
        # Reconstruct config
        config_dict = checkpoint['config']
        config = ExperimentConfig(
            name=config_dict.get('name', "experiment"),
            description=config_dict.get('description', ""),
            seed=config_dict.get('seed', 42),
            device=config_dict.get('device', "cuda"),
            model=ModelConfig(**config_dict.get('model', {})),
            market=MarketConfig(**config_dict.get('market', {})),
            derivative=DerivativeConfig(**config_dict.get('derivative', {})),
            risk_measure=RiskMeasureConfig(**config_dict.get('risk_measure', {})),
            transaction_cost=TransactionCostConfig(**config_dict.get('transaction_cost', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            output_dir=config_dict.get('output_dir', "./outputs"),
            save_checkpoints=config_dict.get('save_checkpoints', True),
            checkpoint_frequency=config_dict.get('checkpoint_frequency', 500)
        )
        
        # Reconstruct metrics
        metrics = MetricsTracker()
        metrics.metrics = checkpoint['metrics']
        metrics.epoch_count = checkpoint['epoch']
        
        epoch = checkpoint['epoch']
        
        return config, metrics, epoch
    
    def load_latest(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None
    ) -> Tuple[ExperimentConfig, MetricsTracker, int]:
        """Load the most recent checkpoint"""
        latest_path = self.checkpoint_dir / "checkpoint_latest.pt"
        if not latest_path.exists():
            raise FileNotFoundError(f"No checkpoint found at {latest_path}")
        return self.load(str(latest_path), model, optimizer)


class ModelSerializer:
    """
    Serialize trained models for deployment.
    
    This class handles exporting models in various formats:
    - PyTorch .pt format
    - ONNX format (for cross-platform deployment)
    - TorchScript (for production C++ inference)
    """
    
    @staticmethod
    def save_pytorch(
        model: nn.Module,
        filepath: str,
        config: Optional[ExperimentConfig] = None
    ) -> None:
        """
        Save model in PyTorch format.
        
        Args:
            model: Trained model
            filepath: Output file path
            config: Configuration (saved as metadata if provided)
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        save_dict = {
            'model_state_dict': model.state_dict(),
            'model_class': model.__class__.__name__,
        }
        
        if config:
            save_dict['config'] = asdict(config)
        
        torch.save(save_dict, filepath)
    
    @staticmethod
    def load_pytorch(
        filepath: str,
        model_class: type
    ) -> Tuple[nn.Module, Optional[ExperimentConfig]]:
        """
        Load model from PyTorch format.
        
        Args:
            filepath: Path to saved model
            model_class: Class of the model (e.g., HedgingPolicy)
        
        Returns:
            (model, config) tuple
        """
        checkpoint = torch.load(filepath, map_location='cpu')
        
        # Instantiate model (assumes default constructor)
        model = model_class()
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Reconstruct config if available
        config = None
        if 'config' in checkpoint:
            config_dict = checkpoint['config']
            config = ExperimentConfig(
                name=config_dict.get('name', "experiment"),
                description=config_dict.get('description', ""),
                seed=config_dict.get('seed', 42),
                device=config_dict.get('device', "cuda"),
                model=ModelConfig(**config_dict.get('model', {})),
                market=MarketConfig(**config_dict.get('market', {})),
                derivative=DerivativeConfig(**config_dict.get('derivative', {})),
                risk_measure=RiskMeasureConfig(**config_dict.get('risk_measure', {})),
                transaction_cost=TransactionCostConfig(**config_dict.get('transaction_cost', {})),
                training=TrainingConfig(**config_dict.get('training', {})),
                output_dir=config_dict.get('output_dir', "./outputs"),
                save_checkpoints=config_dict.get('save_checkpoints', True),
                checkpoint_frequency=config_dict.get('checkpoint_frequency', 500)
            )
        
        return model, config
    
    @staticmethod
    def export_onnx(
        model: nn.Module,
        filepath: str,
        input_shape: Tuple[int, ...] = (1, 2),
        device: str = 'cpu'
    ) -> None:
        """
        Export model to ONNX format for cross-platform deployment.
        
        Args:
            model: Trained model
            filepath: Output file path (.onnx)
            input_shape: Shape of dummy input for tracing
            device: Device for export
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        model.to(device)
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(input_shape, device=device)
        
        # Export
        torch.onnx.export(
            model,
            dummy_input,
            filepath,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['state'],
            output_names=['action'],
            dynamic_axes={
                'state': {0: 'batch_size'},
                'action': {0: 'batch_size'}
            }
        )
    
    @staticmethod
    def export_torchscript(
        model: nn.Module,
        filepath: str,
        input_shape: Tuple[int, ...] = (1, 2),
        device: str = 'cpu'
    ) -> None:
        """
        Export model to TorchScript for C++ deployment.
        
        Args:
            model: Trained model
            filepath: Output file path (.pt)
            input_shape: Shape of example input for tracing
            device: Device for export
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        model.to(device)
        model.eval()
        
        # Create example input
        example_input = torch.randn(input_shape, device=device)
        
        # Trace the model
        traced_model = torch.jit.trace(model, example_input)
        
        # Save
        traced_model.save(filepath)


# ============================================================================
# ABSTRACT BASE CLASSES FOR CLEAN ARCHITECTURE
# ============================================================================

class MarketSimulator(ABC):
    """
    Abstract base class for market simulators.
    
    This defines the interface that all market simulators must implement,
    enabling easy swapping of different market dynamics.
    """
    
    @abstractmethod
    def simulate(
        self,
        num_paths: int,
        device: str = 'cpu'
    ) -> torch.Tensor:
        """
        Simulate market paths.
        
        Args:
            num_paths: Number of Monte Carlo paths
            device: 'cuda' or 'cpu'
        
        Returns:
            Tensor of shape (num_paths, num_timesteps + 1, num_assets)
        """
        pass
    
    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Return simulator configuration as dictionary"""
        pass


class RiskMeasure(ABC):
    """
    Abstract base class for risk measures.
    
    This enables easy implementation of different risk measures
    (entropic, CVaR, variance, etc.) with a common interface.
    """
    
    @abstractmethod
    def compute(self, pnl: torch.Tensor) -> torch.Tensor:
        """
        Compute risk measure of P&L distribution.
        
        Args:
            pnl: Terminal P&L values, shape (batch_size,)
        
        Returns:
            Scalar risk value (to be minimized)
        """
        pass
    
    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Return risk measure configuration"""
        pass


class DerivativePayoff(ABC):
    """
    Abstract base class for derivative payoffs.
    
    This enables implementation of various derivative types
    with a unified interface.
    """
    
    @abstractmethod
    def compute_payoff(self, price_paths: torch.Tensor) -> torch.Tensor:
        """
        Compute derivative payoff given price paths.
        
        Args:
            price_paths: Tensor of shape (batch_size, num_timesteps + 1, num_assets)
        
        Returns:
            Payoff values, shape (batch_size,)
        """
        pass
    
    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Return derivative configuration"""
        pass


# ============================================================================
# CONCRETE IMPLEMENTATIONS OF ABSTRACT CLASSES
# ============================================================================

class BlackScholesSimulator(MarketSimulator):
    """Black-Scholes market simulator"""
    
    def __init__(self, config: MarketConfig):
        self.config = config
        self.S0 = config.S0
        self.mu = config.mu
        self.sigma = config.sigma
        self.T = config.T
        self.N = config.N
        self.dt = self.T / self.N
    
    def simulate(self, num_paths: int, device: str = 'cpu') -> torch.Tensor:
        S = torch.zeros(num_paths, self.N + 1, device=device)
        S[:, 0] = self.S0
        
        for k in range(self.N):
            Z = torch.randn(num_paths, device=device)
            S[:, k+1] = S[:, k] * torch.exp(
                (self.mu - 0.5 * self.sigma**2) * self.dt 
                + self.sigma * np.sqrt(self.dt) * Z
            )
        
        return S.unsqueeze(-1)  # Shape: (num_paths, N+1, 1)
    
    def get_config(self) -> Dict[str, Any]:
        return asdict(self.config)


class HestonSimulator(MarketSimulator):
    """Heston stochastic volatility simulator"""
    
    def __init__(self, config: MarketConfig):
        self.config = config
        self.S0 = config.S0
        self.v0 = config.v0
        self.kappa = config.kappa
        self.theta = config.theta
        self.xi = config.xi
        self.rho = config.rho
        self.T = config.T
        self.N = config.N
        self.dt = self.T / self.N
    
    def simulate(self, num_paths: int, device: str = 'cpu') -> torch.Tensor:
        S = torch.zeros(num_paths, self.N + 1, device=device)
        v = torch.zeros(num_paths, self.N + 1, device=device)
        
        S[:, 0] = self.S0
        v[:, 0] = self.v0
        
        for k in range(self.N):
            Z1 = torch.randn(num_paths, device=device)
            Z2 = torch.randn(num_paths, device=device)
            
            W1 = Z1
            W2 = self.rho * Z1 + np.sqrt(1 - self.rho**2) * Z2
            
            v_prev = torch.clamp(v[:, k], min=0.0)
            
            v[:, k+1] = torch.clamp(
                v_prev 
                + self.kappa * (self.theta - v_prev) * self.dt
                + self.xi * torch.sqrt(v_prev * self.dt) * W2,
                min=0.0
            )
            
            S[:, k+1] = S[:, k] * torch.exp(
                -0.5 * v_prev * self.dt 
                + torch.sqrt(v_prev * self.dt) * W1
            )
        
        return S.unsqueeze(-1)  # Shape: (num_paths, N+1, 1)
    
    def get_config(self) -> Dict[str, Any]:
        return asdict(self.config)


class EntropicRisk(RiskMeasure):
    """Entropic risk measure (exponential utility)"""
    
    def __init__(self, lambda_risk: float = 1.0):
        self.lambda_risk = lambda_risk
    
    def compute(self, pnl: torch.Tensor) -> torch.Tensor:
        # Numerical stability: scale P&L
        pnl_scaled = pnl / (pnl.std().detach() + 1e-8)
        
        # Entropic risk: (1/λ) log E[exp(-λ * PnL)]
        # Computed stably using log-sum-exp
        batch_size = pnl.shape[0]
        loss = (
            torch.logsumexp(-self.lambda_risk * pnl_scaled, dim=0)
            - torch.log(torch.tensor(float(batch_size), device=pnl.device))
        )
        return loss
    
    def get_config(self) -> Dict[str, Any]:
        return {'measure_type': 'entropic', 'lambda_risk': self.lambda_risk}


class CallOption(DerivativePayoff):
    """European call option"""
    
    def __init__(self, strike: float):
        self.strike = strike
    
    def compute_payoff(self, price_paths: torch.Tensor) -> torch.Tensor:
        # Terminal price (assuming single asset)
        S_T = price_paths[:, -1, 0]
        return torch.clamp(S_T - self.strike, min=0.0)
    
    def get_config(self) -> Dict[str, Any]:
        return {'contract_type': 'call', 'strike': self.strike}


class CallSpreadOption(DerivativePayoff):
    """Call spread: long K1, short K2"""
    
    def __init__(self, strike1: float, strike2: float):
        if strike1 >= strike2:
            raise ValueError("Must have strike1 < strike2")
        self.strike1 = strike1
        self.strike2 = strike2
    
    def compute_payoff(self, price_paths: torch.Tensor) -> torch.Tensor:
        S_T = price_paths[:, -1, 0]
        long_call = torch.clamp(S_T - self.strike1, min=0.0)
        short_call = torch.clamp(S_T - self.strike2, min=0.0)
        return long_call - short_call
    
    def get_config(self) -> Dict[str, Any]:
        return {
            'contract_type': 'callspread',
            'strike1': self.strike1,
            'strike2': self.strike2
        }


# ============================================================================
# EXAMPLE: CREATING DEFAULT CONFIGURATION FILES
# ============================================================================

def create_default_configs(output_dir: str = "./configs") -> None:
    """
    Create example configuration files for common experiments.
    
    This creates YAML files for:
    1. Black-Scholes hedging without transaction costs
    2. Heston hedging with transaction costs
    3. Multi-asset hedging (high-dimensional)
    4. Call spread hedging with different risk aversions
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Config 1: Black-Scholes baseline
    config_bs = ExperimentConfig(
        name="bs_baseline",
        description="Black-Scholes hedging without transaction costs",
        market=MarketConfig(model_type='blackscholes', sigma=0.2),
        derivative=DerivativeConfig(contract_type='call', strike=100.0),
        risk_measure=RiskMeasureConfig(measure_type='entropic', lambda_risk=1.0),
        transaction_cost=TransactionCostConfig(cost_type='none'),
        training=TrainingConfig(num_epochs=2000, batch_size=1024)
    )
    config_bs.save_yaml(str(output_path / "bs_baseline.yaml"))
    
    # Config 2: Heston with transaction costs
    config_heston_tc = ExperimentConfig(
        name="heston_with_tc",
        description="Heston hedging with proportional transaction costs",
        market=MarketConfig(model_type='heston'),
        derivative=DerivativeConfig(contract_type='call', strike=100.0),
        risk_measure=RiskMeasureConfig(measure_type='entropic', lambda_risk=1.0),
        transaction_cost=TransactionCostConfig(cost_type='proportional', rate=0.001),
        training=TrainingConfig(num_epochs=2000, batch_size=1024)
    )
    config_heston_tc.save_yaml(str(output_path / "heston_with_tc.yaml"))
    
    # Config 3: Multi-asset (high-dimensional)
    config_multi = ExperimentConfig(
        name="multi_asset_5dim",
        description="Multi-asset hedging with 5 independent Heston processes",
        model=ModelConfig(input_dim=6, output_dim=5),  # 5 assets + time
        market=MarketConfig(model_type='heston'),
        derivative=DerivativeConfig(contract_type='call', strike=100.0),
        risk_measure=RiskMeasureConfig(measure_type='variance'),
        transaction_cost=TransactionCostConfig(cost_type='none'),
        training=TrainingConfig(num_epochs=2000, batch_size=1024)
    )
    config_multi.save_yaml(str(output_path / "multi_asset_5dim.yaml"))
    
    # Config 4: Call spread with high risk aversion
    config_callspread = ExperimentConfig(
        name="callspread_riskaverse",
        description="Call spread hedging with high risk aversion",
        market=MarketConfig(model_type='heston'),
        derivative=DerivativeConfig(
            contract_type='callspread',
            strike=100.0,
            strike2=101.0
        ),
        risk_measure=RiskMeasureConfig(measure_type='entropic', lambda_risk=2.0),
        transaction_cost=TransactionCostConfig(cost_type='proportional', rate=0.001),
        training=TrainingConfig(num_epochs=2000, batch_size=1024)
    )
    config_callspread.save_yaml(str(output_path / "callspread_riskaverse.yaml"))
    
    print(f"✓ Created 4 example configuration files in {output_dir}/")
    print("  - bs_baseline.yaml")
    print("  - heston_with_tc.yaml")
    print("  - multi_asset_5dim.yaml")
    print("  - callspread_riskaverse.yaml")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    """
    Demonstrate usage of modular architecture components.
    """
    
    print("\n" + "="*80)
    print("DEEP HEDGING - MODULAR ARCHITECTURE DEMONSTRATION")
    print("="*80 + "\n")
    
    # 1. Create and save a configuration
    print("1. Creating experiment configuration...")
    config = ExperimentConfig(
        name="demo_experiment",
        description="Demonstration of modular architecture",
        seed=42
    )
    config.save_yaml("./demo_config.yaml")
    print("   ✓ Saved to demo_config.yaml\n")
    
    # 2. Setup logging
    print("2. Setting up logging...")
    logger = setup_logger(
        "deep_hedging",
        log_file="./logs/experiment.log"
    )
    logger.info("Logging system initialized")
    print("   ✓ Logger configured\n")
    
    # 3. Create metrics tracker
    print("3. Creating metrics tracker...")
    metrics = MetricsTracker()
    metrics.log(loss=1.5, pnl_mean=-2.0, pnl_std=3.0)
    metrics.log(loss=1.2, pnl_mean=-1.5, pnl_std=2.8)
    metrics.save("./metrics.json")
    print("   ✓ Metrics saved to metrics.json\n")
    
    # 4. Demonstrate abstract classes
    print("4. Demonstrating market simulators...")
    bs_sim = BlackScholesSimulator(config.market)
    paths = bs_sim.simulate(num_paths=100, device='cpu')
    print(f"   ✓ Simulated {paths.shape[0]} paths with {paths.shape[1]} timesteps\n")
    
    # 5. Create default configuration files
    print("5. Creating default configuration templates...")
    create_default_configs("./configs")
    print()
    
    print("="*80)
    print("Demonstration complete!")
    print("="*80 + "\n")
