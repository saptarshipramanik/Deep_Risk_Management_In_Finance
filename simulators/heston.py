import torch

def simulate_heston_paths(S0, v0, kappa, theta, xi, rho, T, N, M, device):
    dt = T / N

    S = torch.zeros(M, N + 1, device=device)
    v = torch.zeros(M, N + 1, device=device)

    S[:, 0] = S0
    v[:, 0] = v0

    for k in range(N):
        Z1 = torch.randn(M, device=device)
        Z2 = torch.randn(M, device=device)

        # Correlated Brownian motions
        W1 = Z1
        W2 = rho * Z1 + torch.sqrt(torch.tensor(1 - rho**2, device=device)) * Z2

        v_prev = torch.clamp(v[:, k], min=0.0)

        # Variance process (full truncation)
        v[:, k+1] = torch.clamp(
            v_prev
            + kappa * (theta - v_prev) * dt
            + xi * torch.sqrt(v_prev * dt) * W2,
            min=0.0
        )

        # Stock price process
        S[:, k+1] = S[:, k] * torch.exp(
            -0.5 * v_prev * dt
            + torch.sqrt(v_prev * dt) * W1
        )

    return S, v