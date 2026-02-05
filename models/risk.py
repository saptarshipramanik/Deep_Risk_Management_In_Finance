import torch

def entropic_loss(pnl_scaled, lambda_risk):
    return (
        torch.logsumexp(-lambda_risk * pnl_scaled, dim=0)
        - torch.log(torch.tensor(pnl_scaled.shape[0], device=pnl_scaled.device))
    )

def certainty_equivalent(pnl, lambda_risk):
    return (-1 / lambda_risk) * torch.log(torch.mean(torch.exp(-lambda_risk * pnl)))