"""V2 sensor-only resource context; legacy observations/physics stay unchanged.

The recovery navigation plant disables finite-battery dynamics. Here battery is
an explicitly labelled telemetry ledger, NOT its constant ``agent.energy``.
This observation experiment does not implement an energy-safe mission policy.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn
from torch.nn import functional as F

from experiments.directional_navigation.standard_baselines import ContinuousRecovery
from experiments.jacobian_energy_bridge.features import StructuredLidarFeatureExtractor

VERSION = "deployable_distance_resource_v2"
RAYS = 1024
DIM = 1039
FIELDS = {
    "velocity": [0, 3],
    "active_goal_direction": [3, 6],
    "active_goal_log_distance": [6, 7],
    "lidar_normalized_distance": [7, 1031],
    "remaining_episode_fraction": [1031, 1032],
    "charger_direction": [1032, 1035],
    "charger_log_distance": [1035, 1036],
    "remaining_energy_fraction": [1036, 1037],
    "return_command": [1037, 1038],
    "previous_policy_step_contact": [1038, 1039],
}


def observation_space() -> gym.spaces.Box:
    low = np.zeros(DIM, dtype=np.float32)
    low[:6] = -1.0
    low[1032:1035] = -1.0
    return gym.spaces.Box(low, np.ones(DIM, dtype=np.float32), dtype=np.float32)


def pack_observation(
    base_observation: np.ndarray,
    *,
    charger_displacement: np.ndarray,
    distance_scale: float,
    remaining_episode_fraction: float,
    remaining_energy_fraction: float,
    return_command: bool,
    previous_contact: bool,
) -> np.ndarray:
    """Only observable inputs; no layout, radius, clearance, or ray hit flags.

base_observation is the legacy 2055-D pre-wrapper observation. Retain all
1024 distances, including max-range sentinels; ignore the redundant flags.
The first seven entries already describe the *active* goal in this frame.
"""
    base = np.asarray(base_observation, dtype=np.float32)
    delta = np.asarray(charger_displacement, dtype=np.float64)
    if base.shape != (2055,) or delta.shape != (3,):
        raise ValueError("unexpected sensor/charger shape")
    if not np.isfinite(base[:1031]).all() or not np.isfinite(delta).all():
        raise ValueError("nonfinite deployable input")
    if not np.isfinite(distance_scale) or distance_scale <= 0:
        raise ValueError("positive finite distance scale required")
    for value in (remaining_episode_fraction, remaining_energy_fraction):
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("resource fractions must be finite and in [0,1]")
    distance = float(np.linalg.norm(delta))
    direction = delta / distance if distance > 1e-8 else np.zeros(3)
    result = np.concatenate((
        base[:1031], [remaining_episode_fraction], direction,
        [min(1.0, np.log1p(distance) / np.log1p(distance_scale)),
         remaining_energy_fraction, float(return_command), float(previous_contact)],
    )).astype(np.float32)
    if not observation_space().contains(result):
        raise ValueError("observation violates v2 bounds")
    return result


class DeployableRecovery(ContinuousRecovery):
    """Observation-only navigation variant with inherited full-state snapshots.

Budget units match TelemetryCostModel.realized_cost. The ledger is reset per
navigation task and clamped at zero; reaching zero does NOT alter dynamics,
rewards, or termination. Such runs cannot establish energy sustainability.
Return commands are explicit reset options, never inferred by a hidden rule.
"""

    def __init__(self, *, battery_capacity: float, initial_soc: float = 1.0,
                 seed_start: int = 993600001, seed_stride: int = 1,
                 horizon: int = 4000, obstacles: int = 24):
        if not np.isfinite(battery_capacity) or battery_capacity <= 0:
            raise ValueError("explicit finite positive battery capacity required")
        if not np.isfinite(initial_soc) or not 0.0 <= initial_soc <= 1.0:
            raise ValueError("initial SOC must lie in [0,1]")
        self.battery_capacity = float(battery_capacity)
        self.initial_soc = float(initial_soc)
        self.episode_initial_soc = self.initial_soc
        self.return_command = False
        super().__init__(seed_start=seed_start, seed_stride=seed_stride,
                         horizon=horizon, obstacles=obstacles)
        self.observation_space = observation_space()
        self.parked_observation = np.zeros(DIM, np.float32)

    @property
    def telemetry_remaining_energy(self) -> float:
        return max(0.0, self.battery_capacity * self.episode_initial_soc - self.energy)

    def _observation(self, obs: np.ndarray) -> np.ndarray:
        return pack_observation(
            obs,
            charger_displacement=self.base.charger_position - self.base.agent.pos,
            distance_scale=self.base.d_max,
            remaining_episode_fraction=max(0.0, 1.0 - self.steps / self.horizon),
            remaining_energy_fraction=self.telemetry_remaining_energy / self.battery_capacity,
            return_command=self.return_command,
            # At decision t+1, collided contains contact from the just executed
            # step t. prev_collided is one step too old for this observation.
            previous_contact=bool(self.base.agent.collided),
        )

    def reset(self, *, seed=None, options=None):
        settings = dict(options or {})
        mode = settings.pop("mission_mode", "task")
        if mode not in {"task", "return"}:
            raise ValueError("mission_mode must be task or return")
        soc = float(settings.pop("initial_soc", self.initial_soc))
        if not np.isfinite(soc) or not 0.0 <= soc <= 1.0:
            raise ValueError("initial SOC must lie in [0,1]")
        self.episode_initial_soc, self.return_command = soc, mode == "return"
        if self.return_command:
            if "task_point" in settings and not np.array_equal(
                np.asarray(settings["task_point"], dtype=np.float32), self.base.charger_position
            ):
                raise ValueError("return command conflicts with supplied active goal")
            settings["task_point"] = self.base.charger_position
        obs, info = super().reset(seed=seed, options=settings)
        return obs, {**info, "observation_version": VERSION,
                     "battery_source": "initial_budget_minus_realized_telemetry"}


class DeployableLidarExtractor(BaseFeaturesExtractor):
    """Same two-convolution ordered readout, but a single range channel.

Resource context is appended explicitly; no extra attention/recurrent/critic
module. Historical checkpoint shapes are deliberately incompatible.
"""

    def __init__(self, observation_space: gym.spaces.Box):
        if observation_space.shape != (DIM,):
            raise ValueError("v2 extractor requires the 1039-D observation contract")
        super().__init__(observation_space, features_dim=32 + 64 * 2 * 16 + 8)
        self.goal_encoder = nn.Sequential(nn.Linear(7, 32), nn.ReLU(),
                                          nn.Linear(32, 32), nn.ReLU())
        self.lidar_convolutions = nn.ModuleList([
            nn.Conv2d(1, 16, kernel_size=3), nn.Conv2d(16, 32, kernel_size=3),
        ])

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.ndim != 2 or observations.shape[1] != DIM:
            raise ValueError("unexpected v2 observation tensor")
        features = observations[:, 7:1031].reshape(-1, 1, 8, 128)
        for convolution in self.lidar_convolutions:
            features = StructuredLidarFeatureExtractor._circular_azimuth_bounded_elevation_pad(features)
            features = F.relu(convolution(features))
        return torch.cat((
            self.goal_encoder(observations[:, :7]),
            F.adaptive_avg_pool2d(features, (2, 16)).flatten(1),
            F.adaptive_max_pool2d(features, (2, 16)).flatten(1),
            observations[:, 1031:],
        ), dim=1)


def policy_kwargs() -> dict:
    return {"features_extractor_class": DeployableLidarExtractor,
            "share_features_extractor": False,
            "net_arch": {"pi": [256, 256], "qf": [256, 256]}}
