from __future__ import annotations

import torch


def _validate_gamma(gamma: float) -> None:
    if gamma != 1.0:
        raise ValueError("energy SSP targets require gamma=1")


def scalar_ssp_target(
    cost: torch.Tensor,
    next_value: torch.Tensor,
    charger_hit: torch.Tensor,
    *,
    gamma: float = 1.0,
) -> torch.Tensor:
    _validate_gamma(gamma)
    return cost + (~charger_hit.bool()).to(cost.dtype) * next_value


def quantile_ssp_target(
    cost: torch.Tensor,
    next_quantiles: torch.Tensor,
    charger_hit: torch.Tensor,
    *,
    gamma: float = 1.0,
) -> torch.Tensor:
    _validate_gamma(gamma)
    while cost.ndim < next_quantiles.ndim:
        cost = cost.unsqueeze(-1)
    mask = (~charger_hit.bool()).to(next_quantiles.dtype)
    while mask.ndim < next_quantiles.ndim:
        mask = mask.unsqueeze(-1)
    return cost + mask * next_quantiles


def quantile_huber_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    quantile_levels: torch.Tensor,
    *,
    kappa: float = 1.0,
) -> torch.Tensor:
    if kappa <= 0.0:
        raise ValueError("kappa must be positive")
    if predictions.ndim != 2 or targets.ndim != 2:
        raise ValueError("predictions and targets must be [batch, quantiles]")
    if quantile_levels.ndim != 1 or quantile_levels.shape[0] != predictions.shape[1]:
        raise ValueError("quantile levels must align with predicted quantiles")
    residual = targets.unsqueeze(1) - predictions.unsqueeze(2)
    absolute = residual.abs()
    huber = torch.where(absolute <= kappa, 0.5 * residual.square(), kappa * (absolute - 0.5 * kappa))
    levels = quantile_levels.view(1, -1, 1)
    weights = (levels - (residual.detach() < 0.0).to(predictions.dtype)).abs()
    return (weights * huber / kappa).mean()
