from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np

from .environment import NavigationEnv


@dataclass(frozen=True)
class OperationalEnergyConfig:
    capacity: float
    reserve: float = 0.0
    cost_multiplier: float = 1.0
    enforce_exhaustion: bool = False

    def __post_init__(self) -> None:
        if not np.isfinite(self.capacity) or self.capacity <= 0.0:
            raise ValueError("operational energy capacity must be finite and positive")
        if not np.isfinite(self.reserve) or not 0.0 <= self.reserve < self.capacity:
            raise ValueError("reserve must lie in [0, capacity)")
        if not np.isfinite(self.cost_multiplier) or self.cost_multiplier <= 0.0:
            raise ValueError("cost multiplier must be finite and positive")


class OperationalEnergyWrapper(gym.Wrapper):
    """Tracks finite research energy without changing the SAC observation.

    The wrapped ``NavigationEnv`` retains its 1000-unit compatibility state.
    Operational energy is separate telemetry used only by energy experiments.
    """

    def __init__(self, environment: NavigationEnv, config: OperationalEnergyConfig) -> None:
        super().__init__(environment)
        self.config = config
        self.operational_energy = config.capacity
        self.operational_energy_used = 0.0

    @property
    def navigation_env(self) -> NavigationEnv:
        return self.env

    @property
    def operational_soc(self) -> float:
        return self.operational_energy / self.config.capacity

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        reset_options = {} if options is None else dict(options)
        initial_energy = float(reset_options.pop("operational_initial_energy", self.config.capacity))
        if not np.isfinite(initial_energy) or not 0.0 < initial_energy <= self.config.capacity:
            raise ValueError("operational_initial_energy must lie in (0, capacity]")
        observation, info = self.env.reset(seed=seed, options=reset_options)
        self.operational_energy = initial_energy
        self.operational_energy_used = 0.0
        return observation, self._augment_info(info, step_cost=0.0, exhausted=False)

    def step(self, action: np.ndarray):
        observation, reward, terminated, truncated, info = self.env.step(action)
        step_cost = float(info["energy_usage"]) * self.config.cost_multiplier
        self.operational_energy_used += step_cost
        self.operational_energy = max(0.0, self.operational_energy - step_cost)
        exhausted = self.operational_energy <= 0.0
        if self.config.enforce_exhaustion and exhausted:
            terminated = True
        return observation, reward, terminated, truncated, self._augment_info(info, step_cost, exhausted)

    def _augment_info(
        self,
        info: dict[str, Any],
        step_cost: float,
        exhausted: bool,
    ) -> dict[str, Any]:
        return dict(info) | {
            "operational_energy_capacity": self.config.capacity,
            "operational_energy_remaining": self.operational_energy,
            "operational_energy_used": self.operational_energy_used,
            "operational_energy_step_cost": step_cost,
            "operational_soc": self.operational_soc,
            "operational_reserve": self.config.reserve,
            "operational_energy_exhausted": exhausted,
            "operational_exhaustion_enforced": self.config.enforce_exhaustion,
            "navigation_energy_is_compatibility_only": True,
        }
