from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional


FEATURE_NAMES = (
    "position_x",
    "position_y",
    "position_z",
    "velocity_x",
    "velocity_y",
    "velocity_z",
    "task_direction_x",
    "task_direction_y",
    "task_direction_z",
    "task_distance",
    "charger_direction_x",
    "charger_direction_y",
    "charger_direction_z",
    "charger_distance",
    "return_leg",
    "remaining_horizon",
    "frozen_action_x",
    "frozen_action_y",
    "frozen_action_z",
    "proposed_action_x",
    "proposed_action_y",
    "proposed_action_z",
    "previous_action_x",
    "previous_action_y",
    "previous_action_z",
    "previous_energy",
    "previous_filter_intervention",
    "intervention_gain",
    "intervention_sigma",
    "intervention_rho",
)

ACTION_SLICE = slice(19, 22)
CURRENT_ACTION_CONDITIONING = np.asarray([16, 17, 18, 19, 20, 21], dtype=np.int64)
GEOMETRY_FEATURES = np.asarray(
    [3, 4, 5, 9, 13, 14, 15, 25, 26, 27, 28, 29], dtype=np.int64
)


def _unit_direction(delta: np.ndarray) -> tuple[np.ndarray, float]:
    vector = np.asarray(delta, dtype=np.float32)
    distance = float(np.linalg.norm(vector))
    if distance <= np.finfo(np.float32).eps:
        return np.zeros(3, dtype=np.float32), 0.0
    return (vector / distance).astype(np.float32), distance


def decode_active_goal_state(
    compact_observation: np.ndarray,
    active_goal: np.ndarray,
    *,
    d_max: float,
    horizontal_v_max: float,
    vertical_v_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Invert the stored seven-dimensional SAC goal prefix.

    The prefix stores normalized velocity, goal direction, and
    log1p(distance)/log1p(d_max).  The result is the physical state immediately
    before the recorded action.
    """

    compact = np.asarray(compact_observation, dtype=np.float64)
    goal = np.asarray(active_goal, dtype=np.float64)
    if compact.shape != (7,) or goal.shape != (3,):
        raise ValueError("compact observation and active goal have invalid shapes")
    if min(d_max, horizontal_v_max, vertical_v_max) <= 0.0:
        raise ValueError("physical normalizers must be positive")
    if not np.all(np.isfinite(compact)) or not np.all(np.isfinite(goal)):
        raise ValueError("state reconstruction requires finite inputs")
    direction = compact[3:6]
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm > 1e-8:
        direction = direction / direction_norm
    distance = float(np.expm1(np.clip(compact[6], 0.0, 1.0) * np.log1p(d_max)))
    position = goal - direction * distance
    velocity = compact[:3] * np.asarray(
        [horizontal_v_max, horizontal_v_max, vertical_v_max], dtype=np.float64
    )
    return position.astype(np.float32), velocity.astype(np.float32)


def build_pre_action_context(
    *,
    position: np.ndarray,
    velocity: np.ndarray,
    task_goal: np.ndarray,
    charger_goal: np.ndarray,
    world_extent: np.ndarray,
    d_max: float,
    horizontal_v_max: float,
    vertical_v_max: float,
    leg: int,
    remaining_horizon_fraction: float,
    frozen_action: np.ndarray,
    proposed_action: np.ndarray,
    previous_action: np.ndarray,
    previous_energy_fraction: float,
    previous_filter_intervention: float,
    intervention_descriptor: np.ndarray,
) -> np.ndarray:
    """Build a strictly pre-action feature vector.

    Current-step realized energy, filter result, collision flags, suffix costs,
    and terminal outcomes are deliberately absent.  They are labels or audit
    values and would leak the answer at deployment time.
    """

    position = np.asarray(position, dtype=np.float32)
    velocity = np.asarray(velocity, dtype=np.float32)
    task_goal = np.asarray(task_goal, dtype=np.float32)
    charger_goal = np.asarray(charger_goal, dtype=np.float32)
    extent = np.asarray(world_extent, dtype=np.float32)
    frozen_action = np.asarray(frozen_action, dtype=np.float32)
    proposed_action = np.asarray(proposed_action, dtype=np.float32)
    previous_action = np.asarray(previous_action, dtype=np.float32)
    descriptor = np.asarray(intervention_descriptor, dtype=np.float32)
    for name, value in {
        "position": position,
        "velocity": velocity,
        "task_goal": task_goal,
        "charger_goal": charger_goal,
        "world_extent": extent,
        "frozen_action": frozen_action,
        "proposed_action": proposed_action,
        "previous_action": previous_action,
        "intervention_descriptor": descriptor,
    }.items():
        if value.shape != (3,) or not np.all(np.isfinite(value)):
            raise ValueError(f"{name} must be a finite three-vector")
    if min(d_max, horizontal_v_max, vertical_v_max) <= 0.0 or np.any(extent <= 0.0):
        raise ValueError("normalizers must be positive")
    task_direction, task_distance = _unit_direction(task_goal - position)
    charger_direction, charger_distance = _unit_direction(charger_goal - position)
    velocity_feature = velocity / np.asarray(
        [horizontal_v_max, horizontal_v_max, vertical_v_max], dtype=np.float32
    )
    context = np.concatenate(
        [
            np.clip(position / extent, 0.0, 1.0),
            np.clip(velocity_feature, -1.0, 1.0),
            task_direction,
            np.asarray([np.clip(task_distance / d_max, 0.0, 1.0)], dtype=np.float32),
            charger_direction,
            np.asarray([np.clip(charger_distance / d_max, 0.0, 1.0)], dtype=np.float32),
            np.asarray([float(bool(leg))], dtype=np.float32),
            np.asarray([np.clip(remaining_horizon_fraction, 0.0, 1.0)], dtype=np.float32),
            np.clip(frozen_action, -1.0, 1.0),
            np.clip(proposed_action, -1.0, 1.0),
            np.clip(previous_action, -1.0, 1.0),
            np.asarray([max(0.0, previous_energy_fraction)], dtype=np.float32),
            np.asarray([max(0.0, previous_filter_intervention)], dtype=np.float32),
            descriptor,
        ]
    ).astype(np.float32)
    if context.shape != (len(FEATURE_NAMES),) or not np.all(np.isfinite(context)):
        raise RuntimeError("constructed rechargeability context is invalid")
    return context


def grouped_nested_split(
    scene_indices: np.ndarray,
    *,
    outer_fold: int,
    outer_folds: int = 3,
    validation_modulus: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scenes = np.asarray(scene_indices, dtype=np.int64)
    if scenes.ndim != 1 or np.any(scenes < 0):
        raise ValueError("scene indices must be a nonnegative vector")
    if outer_folds < 3 or not 0 <= outer_fold < outer_folds:
        raise ValueError("invalid outer-fold request")
    if validation_modulus < 3:
        raise ValueError("validation modulus must be at least three")
    # Scene identifiers are provenance keys, not necessarily dense integers.
    # Split on their stable ordinal rank so bucket-stratified IDs such as
    # 0, 30, 60, ... cannot accidentally collapse a fold to an empty set.
    unique_scenes = np.unique(scenes)
    scene_rank = np.searchsorted(unique_scenes, scenes)
    test = scene_rank % outer_folds == outer_fold
    validation = (~test) & (
        ((scene_rank // outer_folds) + outer_fold) % validation_modulus == 0
    )
    train = ~(test | validation)
    return train, validation, test


class MonotoneBudgetCritic(nn.Module):
    """CDF critic whose rechargeability is non-decreasing in available budget."""

    def __init__(
        self,
        context_dim: int,
        *,
        hidden_dim: int = 96,
        budget_knots: int = 17,
        maximum_budget: float = 0.70,
    ) -> None:
        super().__init__()
        if context_dim <= 0 or hidden_dim <= 0 or budget_knots < 3:
            raise ValueError("invalid critic dimensions")
        if maximum_budget <= 0.0:
            raise ValueError("maximum budget must be positive")
        self.context_dim = int(context_dim)
        self.budget_knots = int(budget_knots)
        self.maximum_budget = float(maximum_budget)
        self.network = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, budget_knots),
        )

    def knot_logits(self, context: torch.Tensor) -> torch.Tensor:
        raw = self.network(context)
        base = raw[..., :1]
        increments = functional.softplus(raw[..., 1:]) / float(self.budget_knots - 1)
        return torch.cat([base, base + torch.cumsum(increments, dim=-1)], dim=-1)

    def forward(self, context: torch.Tensor, budget: torch.Tensor) -> torch.Tensor:
        if context.shape[-1] != self.context_dim:
            raise ValueError("context dimension does not match critic")
        values = budget.reshape(-1)
        if context.shape[0] != values.shape[0]:
            raise ValueError("context and budget batch sizes differ")
        logits = self.knot_logits(context)
        scaled = torch.clamp(values / self.maximum_budget, 0.0, 1.0)
        scaled = scaled * float(self.budget_knots - 1)
        lower = torch.floor(scaled).long()
        upper = torch.clamp(lower + 1, max=self.budget_knots - 1)
        fraction = scaled - lower.to(scaled.dtype)
        rows = torch.arange(context.shape[0], device=context.device)
        interpolated = logits[rows, lower] * (1.0 - fraction) + logits[rows, upper] * fraction
        probability = torch.sigmoid(interpolated)
        return torch.where(values < 0.0, torch.zeros_like(probability), probability)


class ResidualMonotoneBudgetCritic(nn.Module):
    """Zero-initialized residual over a frozen monotone geometry critic."""

    def __init__(
        self,
        geometry_critic: MonotoneBudgetCritic,
        residual_dim: int,
        *,
        hidden_dim: int = 96,
    ) -> None:
        super().__init__()
        if residual_dim <= 0 or hidden_dim <= 0:
            raise ValueError("residual critic dimensions must be positive")
        self.geometry_critic = geometry_critic
        for parameter in self.geometry_critic.parameters():
            parameter.requires_grad_(False)
        self.geometry_critic.eval()
        self.residual_dim = int(residual_dim)
        self.budget_knots = geometry_critic.budget_knots
        self.maximum_budget = geometry_critic.maximum_budget
        self.residual_network = nn.Sequential(
            nn.Linear(residual_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.budget_knots),
        )
        nn.init.zeros_(self.residual_network[-1].weight)
        nn.init.zeros_(self.residual_network[-1].bias)

    def train(self, mode: bool = True):
        super().train(mode)
        self.geometry_critic.eval()
        return self

    def knot_logits(
        self,
        geometry_context: torch.Tensor,
        residual_context: torch.Tensor,
    ) -> torch.Tensor:
        if residual_context.shape[-1] != self.residual_dim:
            raise ValueError("residual context dimension does not match critic")
        with torch.no_grad():
            geometry_logits = self.geometry_critic.knot_logits(geometry_context)
        residual = self.residual_network(residual_context)
        geometry_increments = torch.diff(geometry_logits, dim=-1)
        scaled = torch.clamp(
            geometry_increments * float(self.budget_knots - 1), min=1e-7
        )
        # Stable inverse softplus.  log(expm1(x)) overflows around x=88 in
        # float32, which made valid large geometry increments become inf and
        # then NaN during interpolation (inf * 0).
        inverse_softplus = scaled + torch.log(-torch.expm1(-scaled))
        increments = functional.softplus(
            inverse_softplus + residual[..., 1:]
        ) / float(self.budget_knots - 1)
        base = geometry_logits[..., :1] + residual[..., :1]
        return torch.cat([base, base + torch.cumsum(increments, dim=-1)], dim=-1)

    def forward(
        self,
        geometry_context: torch.Tensor,
        residual_context: torch.Tensor,
        budget: torch.Tensor,
    ) -> torch.Tensor:
        values = budget.reshape(-1)
        if geometry_context.shape[0] != values.shape[0] or residual_context.shape[0] != values.shape[0]:
            raise ValueError("context and budget batch sizes differ")
        logits = self.knot_logits(geometry_context, residual_context)
        scaled = torch.clamp(values / self.maximum_budget, 0.0, 1.0)
        scaled = scaled * float(self.budget_knots - 1)
        lower = torch.floor(scaled).long()
        upper = torch.clamp(lower + 1, max=self.budget_knots - 1)
        fraction = scaled - lower.to(scaled.dtype)
        rows = torch.arange(values.shape[0], device=values.device)
        interpolated = logits[rows, lower] * (1.0 - fraction) + logits[rows, upper] * fraction
        probability = torch.sigmoid(interpolated)
        return torch.where(values < 0.0, torch.zeros_like(probability), probability)


class ConservativeHazardResidualCritic(nn.Module):
    """Learn nonnegative hazard evidence below a frozen geometry probability."""

    def __init__(
        self,
        geometry_critic: MonotoneBudgetCritic,
        hazard_dim: int,
        *,
        hidden_dim: int = 96,
        geometry_temperature: float = 1.0,
        initial_hazard: float = 1e-4,
    ) -> None:
        super().__init__()
        if hazard_dim <= 0 or hidden_dim <= 0:
            raise ValueError("hazard critic dimensions must be positive")
        if geometry_temperature <= 0.0 or initial_hazard <= 0.0:
            raise ValueError("temperatures and initial hazard must be positive")
        self.geometry_critic = geometry_critic
        for parameter in self.geometry_critic.parameters():
            parameter.requires_grad_(False)
        self.geometry_critic.eval()
        self.hazard_dim = int(hazard_dim)
        self.geometry_temperature = float(geometry_temperature)
        self.budget_knots = geometry_critic.budget_knots
        self.maximum_budget = geometry_critic.maximum_budget
        self.hazard_network = nn.Sequential(
            nn.Linear(hazard_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.hazard_network[-1].weight)
        inverse = np.log(np.expm1(float(initial_hazard)))
        nn.init.constant_(self.hazard_network[-1].bias, inverse)

    def train(self, mode: bool = True):
        super().train(mode)
        self.geometry_critic.eval()
        return self

    def hazard(self, hazard_context: torch.Tensor) -> torch.Tensor:
        if hazard_context.shape[-1] != self.hazard_dim:
            raise ValueError("hazard context dimension does not match critic")
        return functional.softplus(self.hazard_network(hazard_context)).reshape(-1)

    def logits(
        self,
        geometry_context: torch.Tensor,
        hazard_context: torch.Tensor,
        budget: torch.Tensor,
    ) -> torch.Tensor:
        values = budget.reshape(-1)
        if geometry_context.shape[0] != values.shape[0] or hazard_context.shape[0] != values.shape[0]:
            raise ValueError("context and budget batch sizes differ")
        with torch.no_grad():
            # Match the registered geometry calibration path exactly.  This
            # clamp is also used by standard_probabilities before temperature
            # scaling, including when float32 sigmoid saturates to 0 or 1.
            geometry_probability = self.geometry_critic(
                geometry_context, values
            ).clamp(1e-6, 1.0 - 1e-6)
            geometry_logits = (
                torch.logit(geometry_probability) / self.geometry_temperature
            )
        return geometry_logits - self.hazard(hazard_context)

    def forward(
        self,
        geometry_context: torch.Tensor,
        hazard_context: torch.Tensor,
        budget: torch.Tensor,
    ) -> torch.Tensor:
        values = budget.reshape(-1)
        probability = torch.sigmoid(
            self.logits(geometry_context, hazard_context, values)
        )
        return torch.where(values < 0.0, torch.zeros_like(probability), probability)


def bellman_target(
    *,
    next_probability: torch.Tensor,
    budget: torch.Tensor,
    realized_energy: torch.Tensor,
    terminal_success: torch.Tensor,
    terminal_failure: torch.Tensor,
) -> torch.Tensor:
    """One-step target for safe, timely recharge under an executed-energy budget."""

    shapes = {
        tuple(tensor.shape)
        for tensor in (
            next_probability,
            budget,
            realized_energy,
            terminal_success,
            terminal_failure,
        )
    }
    if len(shapes) != 1:
        raise ValueError("Bellman target tensors must have matching shapes")
    affordable = budget >= realized_energy
    continuation = torch.where(affordable, next_probability, torch.zeros_like(next_probability))
    success_value = affordable.to(next_probability.dtype)
    target = torch.where(terminal_success.bool(), success_value, continuation)
    target = torch.where(terminal_failure.bool(), torch.zeros_like(target), target)
    return target


@dataclass(frozen=True)
class ProbabilityMetrics:
    brier: float
    ece: float
    dangerous_false_safe_rate: float
    prevalence: float


def probability_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    false_safe_threshold: float = 0.90,
    bins: int = 10,
) -> ProbabilityMetrics:
    labels = np.asarray(labels, dtype=np.float64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if labels.shape != probabilities.shape or labels.ndim != 1 or labels.size == 0:
        raise ValueError("labels and probabilities must be nonempty matching vectors")
    probabilities = np.clip(probabilities, 0.0, 1.0)
    brier = float(np.mean((probabilities - labels) ** 2))
    ece = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for index in range(bins):
        mask = (probabilities >= edges[index]) & (
            probabilities <= edges[index + 1]
            if index == bins - 1
            else probabilities < edges[index + 1]
        )
        if np.any(mask):
            ece += float(np.mean(mask)) * abs(
                float(np.mean(probabilities[mask]) - np.mean(labels[mask]))
            )
    predicted_safe = probabilities >= false_safe_threshold
    dangerous = predicted_safe & (labels < 0.5)
    return ProbabilityMetrics(
        brier=brier,
        ece=float(ece),
        dangerous_false_safe_rate=float(np.mean(dangerous)),
        prevalence=float(np.mean(labels)),
    )
