from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch
from torch import Tensor, nn


def _slice_width(value: slice) -> int:
    if value.start is None or value.stop is None or value.step not in (None, 1):
        raise ValueError("feature slices must have explicit contiguous bounds")
    return int(value.stop - value.start)


@dataclass(frozen=True, slots=True)
class DualFieldFeatureSpec:
    """Hard whitelist for task-independent proposal-field inputs."""

    position: slice
    velocity: slice
    station_delta: slice
    lidar_distances: slice
    lidar_valid: slice

    @classmethod
    def from_observation_layout(cls, layout: Mapping[str, slice]) -> "DualFieldFeatureSpec":
        required = ("position", "velocity", "station_delta", "lidar_distances", "lidar_valid")
        missing = [name for name in required if name not in layout]
        if missing:
            raise ValueError(f"observation layout lacks dual-field whitelist entries: {missing}")
        spec = cls(*(layout[name] for name in required))
        if tuple(_slice_width(value) for value in (spec.position, spec.velocity, spec.station_delta)) != (3, 3, 3):
            raise ValueError("position, velocity, and station_delta must each have width three")
        if _slice_width(spec.lidar_distances) != 32 or _slice_width(spec.lidar_valid) != 32:
            raise ValueError("the first dual-field fixture requires 32 lidar beams")
        return spec

    @property
    def collision_dimension(self) -> int:
        return 76

    @property
    def recovery_energy_dimension(self) -> int:
        return 9

    def physical_features(self, observation: np.ndarray) -> np.ndarray:
        observation = np.asarray(observation, dtype=np.float32)
        if observation.ndim != 1:
            raise ValueError("one observation vector is required")
        values = np.concatenate((
            observation[self.position],
            observation[self.velocity],
            observation[self.station_delta],
            observation[self.lidar_distances],
            observation[self.lidar_valid],
        )).astype(np.float32, copy=False)
        if values.shape != (73,) or not np.all(np.isfinite(values)):
            raise ValueError("collision physical features must be finite with shape (73,)")
        return values

    def collision_features(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        action_maximum: np.ndarray,
    ) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32)
        maximum = np.asarray(action_maximum, dtype=np.float32)
        if action.shape != (3,) or maximum.shape != (3,) or np.any(maximum <= 0.0):
            raise ValueError("action and positive action maximum must have shape (3,)")
        result = np.concatenate((self.physical_features(observation), action / maximum))
        if result.shape != (self.collision_dimension,) or not np.all(np.isfinite(result)):
            raise ValueError("collision field features are invalid")
        return result.astype(np.float32, copy=False)

    def recovery_energy_features(self, observation: np.ndarray) -> np.ndarray:
        observation = np.asarray(observation, dtype=np.float32)
        result = np.concatenate((
            observation[self.position],
            observation[self.velocity],
            observation[self.station_delta],
        ))
        if result.shape != (self.recovery_energy_dimension,) or not np.all(np.isfinite(result)):
            raise ValueError("recovery-energy features must be finite with shape (9,)")
        return result.astype(np.float32, copy=False)


class _MeanScaleField(nn.Module):
    def __init__(self, input_dimension: int, hidden_dimension: int = 64) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dimension, hidden_dimension),
            nn.ReLU(),
            nn.Linear(hidden_dimension, hidden_dimension),
            nn.ReLU(),
            nn.Linear(hidden_dimension, 2),
        )

    def raw(self, features: Tensor) -> tuple[Tensor, Tensor]:
        output = self.network(features)
        return output[..., 0], nn.functional.softplus(output[..., 1]) + 1e-6


class CollisionProposalField(_MeanScaleField):
    def __init__(self, hidden_dimension: int = 64) -> None:
        super().__init__(76, hidden_dimension)

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        return self.raw(features)


class RecoveryEnergyProposalField(_MeanScaleField):
    def __init__(self, hidden_dimension: int = 64) -> None:
        super().__init__(9, hidden_dimension)

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        raw_mean, scale = self.raw(features)
        return nn.functional.softplus(raw_mean), scale


class SeparateDualFields(nn.Module):
    def __init__(self, hidden_dimension: int = 64) -> None:
        super().__init__()
        self.collision = CollisionProposalField(hidden_dimension)
        self.recovery_energy = RecoveryEnergyProposalField(hidden_dimension)

    def calibrated_collision_lower(self, features: Tensor, quantile: float) -> Tensor:
        mean, scale = self.collision(features)
        return mean - float(quantile) * scale

    def calibrated_energy_upper(self, features: Tensor, quantile: float) -> Tensor:
        mean, scale = self.recovery_energy(features)
        return mean + float(quantile) * scale
