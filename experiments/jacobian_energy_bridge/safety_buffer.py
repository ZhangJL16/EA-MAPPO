from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch


SAFETY_CONTEXT_DIM = 11


def _finite_scalar(value: object, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    result = float(value)
    return result if np.isfinite(result) else float(default)


def projection_context(
    geometry: Mapping[str, object],
    *,
    intervention_norm: float,
    emergency: bool,
    maximum_barrier_constraints: int = 16,
    slack_scale: float = 5.0,
) -> np.ndarray:
    if maximum_barrier_constraints <= 0 or slack_scale <= 0.0:
        raise ValueError("normalization scales must be positive")
    singular_values = np.asarray(
        geometry.get("singular_values", [0.0, 0.0, 0.0]),
        dtype=np.float64,
    )
    if singular_values.shape != (3,):
        raise ValueError("projection singular values must have shape (3,)")
    nominal_slack = _finite_scalar(geometry.get("minimum_nominal_slack"))
    executed_slack = _finite_scalar(geometry.get("minimum_executed_slack"))
    values = np.asarray(
        [
            np.clip(_finite_scalar(geometry.get("action_authority")), 0.0, 1.0),
            *np.clip(singular_values, 0.0, 2.0),
            np.clip(_finite_scalar(geometry.get("rank")) / 3.0, 0.0, 1.0),
            np.clip(
                _finite_scalar(geometry.get("active_barrier_constraints"))
                / maximum_barrier_constraints,
                0.0,
                1.0,
            ),
            np.clip(float(intervention_norm) / 2.0, 0.0, 2.0),
            float(bool(geometry.get("nominal_safe", False))),
            np.tanh(nominal_slack / slack_scale),
            np.tanh(executed_slack / slack_scale),
            float(bool(emergency)),
        ],
        dtype=np.float32,
    )
    if values.shape != (SAFETY_CONTEXT_DIM,) or not np.all(np.isfinite(values)):
        raise ValueError("invalid compact projection context")
    return values


@dataclass(frozen=True)
class SafetyBridgeBatch:
    observations: torch.Tensor
    compact_energy_states: torch.Tensor
    nominal_actions: torch.Tensor
    executed_actions: torch.Tensor
    jacobians: torch.Tensor
    safety_contexts: torch.Tensor
    valid_masks: torch.Tensor
    sample_ages: torch.Tensor


class SafetyBridgeReplay:
    def __init__(
        self,
        *,
        capacity: int,
        observation_dim: int,
        seed: int,
        observation_dtype: np.dtype = np.float16,
        maximum_barrier_constraints: int = 16,
        slack_scale: float = 5.0,
    ) -> None:
        if capacity <= 0 or observation_dim <= 0:
            raise ValueError("capacity and observation_dim must be positive")
        self.capacity = int(capacity)
        self.observation_dim = int(observation_dim)
        self.maximum_barrier_constraints = int(maximum_barrier_constraints)
        self.slack_scale = float(slack_scale)
        self.rng = np.random.default_rng(seed)
        self.observations = np.empty(
            (capacity, observation_dim),
            dtype=observation_dtype,
        )
        self.compact_energy_states = np.empty((capacity, 7), dtype=np.float32)
        self.nominal_actions = np.empty((capacity, 3), dtype=np.float32)
        self.executed_actions = np.empty((capacity, 3), dtype=np.float32)
        self.jacobians = np.empty((capacity, 3, 3), dtype=np.float32)
        self.safety_contexts = np.empty(
            (capacity, SAFETY_CONTEXT_DIM),
            dtype=np.float32,
        )
        self.valid_masks = np.empty(capacity, dtype=np.bool_)
        self.insertion_ids = np.empty(capacity, dtype=np.int64)
        self.position = 0
        self.size = 0
        self.total_added = 0

    def __len__(self) -> int:
        return self.size

    def add_from_info(self, info: Mapping[str, object]) -> None:
        observation = np.asarray(info["anchor_sac_observation"], dtype=np.float32)
        compact_state = np.asarray(
            info["anchor_compact_energy_state"],
            dtype=np.float32,
        )
        nominal = np.asarray(info["nominal_action"], dtype=np.float32)
        executed = np.asarray(info["anchor_executed_action"], dtype=np.float32)
        geometry = info["projection_geometry"]
        if not isinstance(geometry, Mapping):
            raise TypeError("projection_geometry must be a mapping")
        jacobian = np.asarray(geometry["jacobian_total"], dtype=np.float32)
        if observation.shape != (self.observation_dim,):
            raise ValueError("anchor observation dimension mismatch")
        if compact_state.shape != (7,) or nominal.shape != (3,) or executed.shape != (3,):
            raise ValueError("invalid safety bridge transition shape")
        if jacobian.shape != (3, 3):
            raise ValueError("projection Jacobian must have shape (3, 3)")
        emergency = bool(info.get("hocbf_emergency_brake", False))
        fallback = bool(info.get("hocbf_fallback_used", False))
        valid = bool(
            geometry.get("valid", False)
            and geometry.get("active_set_stable", False)
            and geometry.get("coordinate_map_stable", False)
            and not emergency
            and not fallback
        )
        context = projection_context(
            geometry,
            intervention_norm=float(info.get("hocbf_intervention_norm", 0.0)),
            emergency=emergency,
            maximum_barrier_constraints=self.maximum_barrier_constraints,
            slack_scale=self.slack_scale,
        )
        index = self.position
        self.observations[index] = observation
        self.compact_energy_states[index] = compact_state
        self.nominal_actions[index] = nominal
        self.executed_actions[index] = executed
        self.jacobians[index] = jacobian
        self.safety_contexts[index] = context
        self.valid_masks[index] = valid
        self.insertion_ids[index] = self.total_added
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self.total_added += 1

    def sample(self, batch_size: int, device: torch.device | str) -> SafetyBridgeBatch:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.size == 0:
            raise RuntimeError("cannot sample an empty safety bridge replay")
        indices = self.rng.integers(0, self.size, size=batch_size)

        def tensor(values: np.ndarray, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
            return torch.as_tensor(values[indices], dtype=dtype, device=device)

        sample_ages = (self.total_added - 1) - self.insertion_ids[indices]
        if np.any(sample_ages < 0):
            raise RuntimeError("safety bridge replay produced a negative sample age")

        return SafetyBridgeBatch(
            observations=tensor(self.observations),
            compact_energy_states=tensor(self.compact_energy_states),
            nominal_actions=tensor(self.nominal_actions),
            executed_actions=tensor(self.executed_actions),
            jacobians=tensor(self.jacobians),
            safety_contexts=tensor(self.safety_contexts),
            valid_masks=tensor(self.valid_masks, dtype=torch.bool),
            sample_ages=torch.as_tensor(sample_ages, dtype=torch.int64, device=device),
        )

    def metadata(self) -> dict[str, object]:
        valid_count = int(np.sum(self.valid_masks[: self.size]))
        current_ages = (
            np.empty(0, dtype=np.int64)
            if self.size == 0
            else (self.total_added - 1) - self.insertion_ids[: self.size]
        )
        return {
            "capacity": self.capacity,
            "observation_dim": self.observation_dim,
            "size": self.size,
            "total_added": self.total_added,
            "valid_count": valid_count,
            "valid_fraction": float(valid_count / max(self.size, 1)),
            "sample_age_min": None if self.size == 0 else int(np.min(current_ages)),
            "sample_age_p50": (
                None if self.size == 0 else float(np.quantile(current_ages, 0.50))
            ),
            "sample_age_p90": (
                None if self.size == 0 else float(np.quantile(current_ages, 0.90))
            ),
            "sample_age_p95": (
                None if self.size == 0 else float(np.quantile(current_ages, 0.95))
            ),
            "sample_age_max": None if self.size == 0 else int(np.max(current_ages)),
            "observation_storage_dtype": str(self.observations.dtype),
            "safety_context_dim": SAFETY_CONTEXT_DIM,
            "maximum_barrier_constraints": self.maximum_barrier_constraints,
            "slack_scale": self.slack_scale,
        }
