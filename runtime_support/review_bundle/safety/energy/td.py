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
    target_weights: torch.Tensor | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    if kappa <= 0.0:
        raise ValueError("kappa must be positive")
    if predictions.ndim != 2 or targets.ndim != 2:
        raise ValueError("predictions and targets must be [batch, quantiles]")
    if quantile_levels.ndim != 1 or quantile_levels.shape[0] != predictions.shape[1]:
        raise ValueError("quantile levels must align with predicted quantiles")
    if targets.shape[0] != predictions.shape[0]:
        raise ValueError("predictions and targets must share the batch dimension")
    if target_weights is not None:
        if target_weights.ndim != 1 or target_weights.shape[0] != targets.shape[1]:
            raise ValueError("target weights must align with target atoms")
        if torch.any(target_weights < 0.0) or not torch.isclose(
            target_weights.sum(), torch.ones((), device=target_weights.device, dtype=target_weights.dtype)
        ):
            raise ValueError("target weights must be nonnegative and sum to one")
    if reduction not in {"mean", "none"}:
        raise ValueError("reduction must be mean or none")
    residual = targets.unsqueeze(1) - predictions.unsqueeze(2)
    absolute = residual.abs()
    huber = torch.where(absolute <= kappa, 0.5 * residual.square(), kappa * (absolute - 0.5 * kappa))
    levels = quantile_levels.view(1, -1, 1)
    weights = (levels - (residual.detach() < 0.0).to(predictions.dtype)).abs()
    losses = weights * huber / kappa
    if target_weights is None:
        per_sample = losses.mean(dim=(1, 2))
    else:
        atom_weights = target_weights.to(device=losses.device, dtype=losses.dtype).view(1, 1, -1)
        per_sample = (losses * atom_weights).sum(dim=2).mean(dim=1)
    return per_sample.mean() if reduction == "mean" else per_sample


def quantile_atom_weights(quantile_levels: torch.Tensor) -> torch.Tensor:
    if quantile_levels.ndim != 1 or quantile_levels.numel() == 0:
        raise ValueError("quantile levels must be a nonempty vector")
    if torch.any(quantile_levels <= 0.0) or torch.any(quantile_levels >= 1.0):
        raise ValueError("quantile levels must lie in (0, 1)")
    if quantile_levels.numel() > 1 and torch.any(quantile_levels[1:] <= quantile_levels[:-1]):
        raise ValueError("quantile levels must be strictly increasing")
    boundaries = torch.cat(
        (
            torch.zeros(1, device=quantile_levels.device, dtype=quantile_levels.dtype),
            0.5 * (quantile_levels[:-1] + quantile_levels[1:]),
            torch.ones(1, device=quantile_levels.device, dtype=quantile_levels.dtype),
        )
    )
    return boundaries[1:] - boundaries[:-1]
