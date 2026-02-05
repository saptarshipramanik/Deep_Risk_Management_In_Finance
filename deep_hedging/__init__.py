"""
Deep Hedging: A Deep Reinforcement Learning Framework for Derivative Hedging

This package implements the deep hedging framework for optimal hedging of derivatives
under transaction costs and risk constraints.

Reference:
    Buehler, H., Gonon, L., Teichmann, J., & Wood, B. (2019).
    Deep hedging. Quantitative Finance, 19(8), 1271-1291.
"""

__version__ = "1.0.0"
__author__ = "Quanthive Research"

from .models import FeedForwardPolicy, RecurrentPolicy
from .market import GBMSimulator, HestonSimulator
from .training import Trainer, EntropicRisk, CVaRLoss, TrainingConfig
from .evaluation import RiskMetrics, HedgingBenchmark

__all__ = [
    "FeedForwardPolicy",
    "RecurrentPolicy",
    "GBMSimulator",
    "HestonSimulator",
    "Trainer",
    "TrainingConfig",
    "EntropicRisk",
    "CVaRLoss",
    "RiskMetrics",
    "HedgingBenchmark",
]
