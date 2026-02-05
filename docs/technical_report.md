# Deep Hedging: Technical Report

**Author**: Quanthive Research Team  
**Date**: February 2026  
**Version**: 1.0

---

## Executive Summary

This report documents the implementation of Deep Hedging, a deep reinforcement learning framework for optimal derivative hedging under transaction costs and risk constraints. The implementation reproduces key results from the seminal paper by Buehler et al. (2019) and provides a production-ready, modular codebase suitable for integration into quantitative trading systems.

**Key Findings**:
- Deep hedging outperforms classical delta hedging under transaction costs
- LSTM-based policies capture temporal dependencies effectively
- Entropic risk measure provides robust risk-adjusted hedging
- Framework is extensible to multiple assets and complex derivatives

---

## 1. Introduction

### 1.1 Problem Statement

Classical derivative hedging relies on continuous rebalancing assumptions that break down in practice due to:
1. **Transaction costs**: Proportional costs make continuous hedging infeasible
2. **Discrete hedging**: Real-world hedging occurs at discrete time intervals
3. **Model risk**: Black-Scholes assumptions (constant volatility, no jumps) are violated
4. **Risk preferences**: Traders have varying risk aversion levels

### 1.2 Deep Hedging Solution

Deep hedging formulates derivative hedging as a stochastic optimal control problem solved using deep reinforcement learning.

---

## 2. Architecture & Implementation

### 2.1 Neural Network Policies

**Feedforward Policy**:
- Input → Dense(64) → ReLU → Dense(64) → ReLU → Output
- Simple, fast training
- Markovian assumption

**Recurrent Policy (LSTM)**:
- Input → LSTM(64, num_layers=2) → Dense(action_dim)
- Captures temporal dependencies
- Better for incomplete information

### 2.2 Market Simulation

**Geometric Brownian Motion**: Standard GBM simulation

**Heston Stochastic Volatility**: Two-factor model with stochastic variance

### 2.3 Training Procedure

1. Initialize policy network
2. For each epoch:
   - Simulate price paths
   - Execute hedging strategy
   - Compute P&L with transaction costs
   - Calculate risk-based loss
   - Backpropagate and update
3. Evaluate on out-of-sample paths

---

## 3. Experimental Setup

### 3.1 Base Configuration

- Initial price: 100
- Strike: 100 (ATM)
- Maturity: 1 year
- Hedging steps: 30
- Volatility: 20%

### 3.2 Training Configuration

- Epochs: 2000
- Batch size: 2048
- Learning rate: 1e-4
- Hidden dim: 64
- LSTM layers: 2

---

## 4. Results & Analysis

### 4.1 Deep Hedging vs Delta Hedging

**With Transaction Costs (0.1%)**:
- Deep hedging: Lower mean loss, lower variance
- Better tail risk (CVaR)
- Learns to trade less frequently

### 4.2 Impact of Risk Aversion

Higher risk aversion → lower variance, lower trading frequency

### 4.3 Stochastic Volatility

Deep hedging adapts better to changing volatility than constant-volatility delta hedging.

---

## 5. Comparison with Classical Approaches

**Delta Hedging**:
- Pros: Analytical, fast, interpretable
- Cons: Ignores costs, model-dependent

**Deep Hedging**:
- Pros: Learns from data, handles costs, risk-aware
- Cons: Requires training, less interpretable

---

## 6. Conclusion

This implementation demonstrates that deep hedging provides a robust framework for derivative hedging under realistic conditions. The framework is production-ready and extensible.

---

## References

1. Buehler, H., Gonon, L., Teichmann, J., & Wood, B. (2019). Deep hedging. Quantitative Finance, 19(8), 1271-1291.

---

**End of Technical Report**
