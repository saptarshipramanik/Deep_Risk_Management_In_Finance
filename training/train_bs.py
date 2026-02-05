from DDP.models.policy import HedgingPolicy
from DDP.models.risk import entropic_loss
from DDP.simulators.black_scholes import simulate_paths
import torch

def train(
    num_epochs=2000,
    batch_size=1024,
    S0=100.0,
    K=100.0,
    sigma=0.2,
    T=1.0,
    N=30,
    lambda_risk=1.0,
    lr=1e-4
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    policy = HedgingPolicy().to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    loss_history = []

    # =====================
    # TRAINING LOOP
    # =====================
    for epoch in range(num_epochs):

        S = simulate_paths(
            S0=S0,
            mu=0.0,
            sigma=sigma,
            T=T,
            N=N,
            M=batch_size,
            device=device
        )

        trading_gain = torch.zeros(batch_size, device=device)

        for k in range(N):
            time = torch.full((batch_size,), k / N, device=device)
            state = torch.stack([S[:, k] / S0, time], dim=1)

            delta = policy(state)
            trading_gain += delta * (S[:, k+1] - S[:, k])

        payoff = torch.clamp(S[:, -1] - K, min=0.0)
        pnl = trading_gain - payoff

        # ---- scaled P&L ----
        pnl_scaled = pnl / pnl.std().detach()

        # ---- stable entropic loss ----
        loss = (
            torch.logsumexp(-lambda_risk * pnl_scaled, dim=0)
            - torch.log(torch.tensor(batch_size, device=device))
        )

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(policy.parameters(), 5.0)

        optimizer.step()

        loss_history.append(loss.item())

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss {loss.item():.4f}")

    return policy, loss_history