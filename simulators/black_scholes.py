import torch
import math

def simulate_paths(S0, mu, sigma, T, N, M, device):
    dt = T / N
    S = torch.zeros(M, N + 1, device=device)
    S[:, 0] = S0

    for k in range(N):
        Z = torch.randn(M, device=device)
        S[:, k+1] = S[:, k] * torch.exp(
            (mu - 0.5 * sigma**2) * dt + sigma * math.sqrt(dt) * Z
        )

    return S