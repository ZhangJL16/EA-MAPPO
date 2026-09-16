"""Branch-only physical resampling for the conditional-history experiment."""
from __future__ import annotations

from typing import cast

import gymnasium as gym
import numpy as np

from envs.UAVEnergyDeliverySAC import SACTrainingPhase
from experiments.directional_navigation.battery_sortie import CAPACITY
from experiments.directional_navigation.correction_supervision import TeacherConfig
from experiments.directional_navigation.dvoi_h_branch import DVOIHBranch
from experiments.directional_navigation.recovery import LockedRecoveryPlant
from review_bundle.safety.switching.commitment import SortieMode


class BranchOnlyExecutionPlant(LockedRecoveryPlant):
    """Bounded AR(1) post-filter execution error, disabled until explicitly armed."""

    def __init__(
        self,
        *,
        execution_error_sigma: float = 0.04,
        execution_error_rho: float = 0.95,
        execution_error_clip_sigma: float = 3.0,
        **kwargs,
    ) -> None:
        if not 0.0 < execution_error_sigma < 1.0:
            raise ValueError("execution_error_sigma must lie in (0,1)")
        if not 0.0 <= execution_error_rho < 1.0:
            raise ValueError("execution_error_rho must lie in [0,1)")
        if execution_error_clip_sigma <= 0:
            raise ValueError("execution_error_clip_sigma must be positive")
        self.execution_error_sigma = float(execution_error_sigma)
        self.execution_error_rho = float(execution_error_rho)
        self.execution_error_clip_sigma = float(execution_error_clip_sigma)
        self.execution_disturbance_enabled = False
        self._execution_rng = np.random.default_rng(0)
        self._execution_error = np.zeros(3, dtype=np.float32)
        self._execution_error_step = -1
        self.last_requested_execution_error = np.zeros(3, dtype=np.float32)
        self.last_realized_execution_error = np.zeros(3, dtype=np.float32)
        self.execution_error_squared_norm_sum = 0.0
        self.execution_error_sample_count = 0
        self.execution_error_max_norm = 0.0
        super().__init__(**kwargs)

    def disable_execution_disturbance(self) -> None:
        self.execution_disturbance_enabled = False
        self._execution_error[:] = 0.0
        self._execution_error_step = -1
        self.last_requested_execution_error[:] = 0.0
        self.last_realized_execution_error[:] = 0.0
        self.execution_error_squared_norm_sum = 0.0
        self.execution_error_sample_count = 0
        self.execution_error_max_norm = 0.0

    def configure_execution_disturbance(self, seed: int) -> None:
        self._execution_rng = np.random.default_rng(int(seed))
        limit = self.execution_error_clip_sigma * self.execution_error_sigma
        stationary = self._execution_rng.normal(0.0, self.execution_error_sigma, size=3)
        self._execution_error = np.clip(stationary, -limit, limit).astype(np.float32)
        self._execution_error_step = -1
        self.execution_disturbance_enabled = True
        self.last_requested_execution_error[:] = 0.0
        self.last_realized_execution_error[:] = 0.0
        self.execution_error_squared_norm_sum = 0.0
        self.execution_error_sample_count = 0
        self.execution_error_max_norm = 0.0

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
        if not self.execution_disturbance_enabled:
            self.last_requested_execution_error[:] = 0.0
            self.last_realized_execution_error[:] = 0.0
            return filtered, diagnostics
        if self._execution_error_step != int(self.current_step):
            self._advance_execution_error()
        executed = np.clip(filtered + self._execution_error, -1.0, 1.0).astype(np.float32)
        self.last_requested_execution_error = self._execution_error.copy()
        self.last_realized_execution_error = (executed - filtered).astype(np.float32)
        norm = float(np.linalg.norm(self.last_realized_execution_error))
        self.execution_error_squared_norm_sum += norm * norm
        self.execution_error_sample_count += 1
        self.execution_error_max_norm = max(self.execution_error_max_norm, norm)
        diagnostics = dict(diagnostics)
        diagnostics["post_filter_execution_error"] = self.last_realized_execution_error.copy()
        return executed, diagnostics


class ConditionalModelAccessBranch(DVOIHBranch):
    """DVOI R/C semantics with nominal history and branch-only resampling."""

    MAX_PREBRANCH_PREFIX_STEPS = 768

    def __init__(self, *, obstacles: int = 48, guard: int = 12_000) -> None:
        gym.Wrapper.__init__(
            self,
            BranchOnlyExecutionPlant(
                execution_error_sigma=0.04,
                execution_error_rho=0.95,
                execution_error_clip_sigma=3.0,
                lidar_enabled=True,
                lidar_horizontal_sectors=128,
                lidar_vertical_sectors=8,
                num_obstacles=obstacles,
                cbf_enabled=True,
                projection_geometry_enabled=False,
                phase1_episode_max_policy_steps=guard + self.MAX_PREBRANCH_PREFIX_STEPS + 1,
                max_steps_per_task=guard + self.MAX_PREBRANCH_PREFIX_STEPS + 1,
            ),
        )
        self.base = cast(BranchOnlyExecutionPlant, self.env)
        self.horizon = guard + self.MAX_PREBRANCH_PREFIX_STEPS + 1
        self.reward_scale = 0.01
        self.seed_start, self.seed_stride = 193600001, 1
        self.next_episode_index = 0
        self.finished = True
        self.parked = self.park_after_terminal = False
        space = self.base.observation_space
        assert isinstance(space, gym.spaces.Box)
        self.observation_space = gym.spaces.Box(
            np.append(space.low, 0.0).astype(np.float32),
            np.append(space.high, 1.0).astype(np.float32),
            dtype=np.float32,
        )
        shape = self.observation_space.shape
        assert shape is not None
        self.parked_observation = np.zeros(shape, np.float32)

        self.teacher_config = TeacherConfig()
        for key, value in self.teacher_config.__dict__.items():
            if getattr(self.base, key) != value:
                raise ValueError(f"teacher/environment contract mismatch: {key}")
        if self.base.hocbf_top_k is not None:
            raise ValueError("all LiDAR constraints must remain enabled")
        self.base.safety_filter = self.teacher_config.make_filter()
        self.base.set_phase(SACTrainingPhase.ENERGY_MANAGED)
        self.base.configure_calibrated_battery(
            CAPACITY, reserve_fraction=0.0, source="historical_synthetic_capacity"
        )
        self.base.reset_at_charger = True
        self.base.mission_switching_enabled = False
        self.base.energy_learning_enabled = False
        administrative_limit = self.MAX_PREBRANCH_PREFIX_STEPS + guard + 1
        self.base.max_steps_per_task = administrative_limit
        self.base.phase2_episode_limit = administrative_limit

        original = self.observation_space
        self.observation_space = gym.spaces.Dict(
            {
                "nav": original,
                "return": original,
                "battery": gym.spaces.Box(0.0, np.inf, (1,), np.float32),
                "at_home": gym.spaces.Box(0.0, 1.0, (1,), np.float32),
            }
        )
        self.action_space = gym.spaces.Box(-1.0, 1.0, (3,), np.float32)
        self.world_seed = 0
        self.total_steps = 0
        self.return_steps = 0
        self.collision_count = 0
        self.branch_anchor_tasks = 0
        self.branch_anchor_contacts = 0
        self.branch_anchor_energy = 0.0
        self.branch_anchor_step = 0
        self.branch_mode = "HISTORY"
        self.guard = int(guard)

    def reset_world(self, world_seed: int, *, soc: float = 1.0):
        self.base.disable_execution_disturbance()
        return super().reset_world(world_seed, soc=soc)

    def arm_resample(self, seed: int) -> None:
        self.base.configure_execution_disturbance(seed)

    def disturbance_summary(self, steps: int) -> dict[str, float]:
        del steps  # A policy step may invoke the safety filter more than once.
        return {
            "execution_error_rms_norm": float(
                np.sqrt(
                    self.base.execution_error_squared_norm_sum
                    / max(self.base.execution_error_sample_count, 1)
                )
            ),
            "execution_error_max_norm": float(self.base.execution_error_max_norm),
        }
