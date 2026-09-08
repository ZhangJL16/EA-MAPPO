"""Physically explicit post-controller execution error for return rollouts."""

from __future__ import annotations

from typing import cast

import gymnasium as gym
import numpy as np

from experiments.directional_navigation.anchor_return import AnchorReturnRecovery
from experiments.directional_navigation.recovery import LockedRecoveryPlant


class StochasticExecutionPlant(LockedRecoveryPlant):
    """Apply a bounded AR(1) acceleration-command error after policy/filter."""

    def __init__(
        self,
        *,
        execution_error_sigma: float,
        execution_error_rho: float,
        execution_error_clip_sigma: float,
        **kwargs,
    ) -> None:
        if not 0.0 < execution_error_sigma < 1.0:
            raise ValueError("execution_error_sigma must lie in (0, 1)")
        if not 0.0 <= execution_error_rho < 1.0:
            raise ValueError("execution_error_rho must lie in [0, 1)")
        if execution_error_clip_sigma <= 0.0:
            raise ValueError("execution_error_clip_sigma must be positive")
        self.execution_error_sigma = float(execution_error_sigma)
        self.execution_error_rho = float(execution_error_rho)
        self.execution_error_clip_sigma = float(execution_error_clip_sigma)
        self._execution_rng = np.random.default_rng(0)
        self._execution_error = np.zeros(3, dtype=np.float32)
        self._execution_error_step = -1
        self.last_requested_execution_error = np.zeros(3, dtype=np.float32)
        self.last_realized_execution_error = np.zeros(3, dtype=np.float32)
        super().__init__(**kwargs)

    def configure_execution_disturbance(self, seed: int) -> None:
        self._execution_rng = np.random.default_rng(int(seed))
        stationary = self._execution_rng.normal(0.0, self.execution_error_sigma, size=3)
        limit = self.execution_error_clip_sigma * self.execution_error_sigma
        self._execution_error = np.clip(stationary, -limit, limit).astype(np.float32)
        self._execution_error_step = -1
        self.last_requested_execution_error[:] = 0.0
        self.last_realized_execution_error[:] = 0.0

    def _advance_execution_error(self) -> None:
        innovation_scale = self.execution_error_sigma * np.sqrt(
            1.0 - self.execution_error_rho**2
        )
        innovation = self._execution_rng.normal(0.0, innovation_scale, size=3)
        value = self.execution_error_rho * self._execution_error + innovation
        limit = self.execution_error_clip_sigma * self.execution_error_sigma
        self._execution_error = np.clip(value, -limit, limit).astype(np.float32)
        self._execution_error_step = int(self.current_step)

    def _safety_filtered_action(self, nominal_action: np.ndarray):
        filtered, diagnostics = super()._safety_filtered_action(nominal_action)
        if self._execution_error_step != int(self.current_step):
            self._advance_execution_error()
        executed = np.clip(filtered + self._execution_error, -1.0, 1.0).astype(np.float32)
        self.last_requested_execution_error = self._execution_error.copy()
        self.last_realized_execution_error = (executed - filtered).astype(np.float32)
        diagnostics = dict(diagnostics)
        diagnostics["post_filter_execution_error"] = self.last_realized_execution_error.copy()
        return executed, diagnostics


class StochasticAnchorReturnRecovery(AnchorReturnRecovery):
    """Anchor return wrapper with a reproducible physical execution-error law."""

    def __init__(
        self,
        *,
        horizon: int = 4000,
        obstacles: int = 24,
        hocbf: bool = False,
        execution_error_sigma: float = 0.04,
        execution_error_rho: float = 0.95,
        execution_error_clip_sigma: float = 3.0,
    ) -> None:
        if horizon <= 0:
            raise ValueError("positive horizon required")
        gym.Wrapper.__init__(
            self,
            StochasticExecutionPlant(
                execution_error_sigma=execution_error_sigma,
                execution_error_rho=execution_error_rho,
                execution_error_clip_sigma=execution_error_clip_sigma,
                lidar_enabled=True,
                lidar_horizontal_sectors=128,
                lidar_vertical_sectors=8,
                num_obstacles=obstacles,
                cbf_enabled=hocbf,
                projection_geometry_enabled=False,
                phase1_episode_max_policy_steps=horizon,
                max_steps_per_task=horizon,
            ),
        )
        self.base = cast(StochasticExecutionPlant, self.env)
        self.horizon, self.reward_scale = horizon, 0.01
        self.seed_start, self.seed_stride = 193600001, 1
        self.next_episode_index = 0
        self.finished = True
        space = self.base.observation_space
        assert isinstance(space, gym.spaces.Box)
        self.observation_space = gym.spaces.Box(
            np.append(space.low, 0.0).astype(np.float32),
            np.append(space.high, 1.0).astype(np.float32),
            dtype=np.float32,
        )
        shape = self.observation_space.shape
        assert shape is not None
        self.parked_observation = np.zeros(shape, dtype=np.float32)
        self.parked = self.park_after_terminal = False
        self.collision_count = 0
        self.hocbf_step_interventions = 0
        self.execution_error_squared_norm_sum = 0.0
        self.execution_error_max_norm = 0.0

    def reset_from_anchor(
        self,
        anchor: dict[str, object],
        world_seed: int,
        disturbance_seed: int,
    ):
        observation, info = super().reset_from_anchor(anchor, world_seed)
        self.base.configure_execution_disturbance(disturbance_seed)
        self.execution_error_squared_norm_sum = 0.0
        self.execution_error_max_norm = 0.0
        info["disturbance_seed"] = int(disturbance_seed)
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        norm = float(np.linalg.norm(self.base.last_realized_execution_error))
        self.execution_error_squared_norm_sum += norm * norm
        self.execution_error_max_norm = max(self.execution_error_max_norm, norm)
        if "navigation_episode" in info:
            steps = int(info["navigation_episode"]["steps"])
            info["navigation_episode"]["execution_error_rms_norm"] = float(
                np.sqrt(self.execution_error_squared_norm_sum / max(steps, 1))
            )
            info["navigation_episode"]["execution_error_max_norm"] = (
                self.execution_error_max_norm
            )
        return observation, reward, terminated, truncated, info
