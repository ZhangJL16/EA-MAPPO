from __future__ import annotations

import torch


def _validate_action_batch(value: torch.Tensor, name: str) -> None:
    if value.ndim != 2 or value.shape[1] != 3:
        raise ValueError(f"{name} must have shape [N, 3]")


def local_projected_action(
    actor_action: torch.Tensor,
    anchor_nominal_action: torch.Tensor,
    anchor_executed_action: torch.Tensor,
    projection_jacobian: torch.Tensor,
) -> torch.Tensor:
    _validate_action_batch(actor_action, "actor_action")
    _validate_action_batch(anchor_nominal_action, "anchor_nominal_action")
    _validate_action_batch(anchor_executed_action, "anchor_executed_action")
    if projection_jacobian.shape != (actor_action.shape[0], 3, 3):
        raise ValueError("projection_jacobian must have shape [N, 3, 3]")
    offset = anchor_executed_action - torch.bmm(
        projection_jacobian,
        anchor_nominal_action.unsqueeze(-1),
    ).squeeze(-1)
    return offset.detach() + torch.bmm(
        projection_jacobian,
        actor_action.unsqueeze(-1),
    ).squeeze(-1)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if values.ndim != 1 or mask.shape != values.shape:
        raise ValueError("values and mask must be aligned vectors")
    weights = mask.to(dtype=values.dtype)
    denominator = torch.clamp(weights.sum(), min=1.0)
    return (values * weights).sum() / denominator


def shield_consistency_loss(
    actor_action: torch.Tensor,
    anchor_nominal_action: torch.Tensor,
    anchor_executed_action: torch.Tensor,
    projection_jacobian: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    trust_region_radius: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    projected = local_projected_action(
        actor_action,
        anchor_nominal_action,
        anchor_executed_action,
        projection_jacobian,
    )
    mask = valid_mask.to(dtype=torch.bool)
    if trust_region_radius is not None:
        if trust_region_radius <= 0.0:
            raise ValueError("trust_region_radius must be positive")
        delta = torch.linalg.vector_norm(actor_action - anchor_nominal_action, dim=1)
        mask = mask & (delta <= trust_region_radius)
    per_sample = torch.sum(torch.square(actor_action - projected), dim=1)
    return _masked_mean(per_sample, mask), projected, mask


def masked_energy_bridge_loss(
    predicted_energy: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    energy_scale: float,
) -> torch.Tensor:
    if predicted_energy.ndim != 1:
        raise ValueError("predicted_energy must be a vector")
    if energy_scale <= 0.0:
        raise ValueError("energy_scale must be positive")
    return _masked_mean(predicted_energy / energy_scale, valid_mask.to(dtype=torch.bool))
