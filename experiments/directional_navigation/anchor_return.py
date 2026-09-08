"""Return-to-charger evaluation from preserved anchors under locked recovery."""

from __future__ import annotations

import hashlib
import json
from typing import Any, cast

import gymnasium as gym
import numpy as np

from experiments.directional_navigation.recovery import LockedRecoveryPlant, RecoveryCohort
from experiments.forked_action_safety.core import sanitize_snapshot_position


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def obstacle_layout_sha256(value: object) -> str:
    payload = json.dumps(_json_value(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class AnchorReturnRecovery(RecoveryCohort):
    """RecoveryCohort with exact anchor reset and an optional standard HOCBF."""

    def __init__(self, *, horizon: int = 4000, obstacles: int = 24, hocbf: bool = False):
        if horizon <= 0:
            raise ValueError("positive horizon required")
        gym.Wrapper.__init__(
            self,
            LockedRecoveryPlant(
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
        self.base = cast(LockedRecoveryPlant, self.env)
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

    def reset_from_anchor(self, anchor: dict[str, object], world_seed: int):
        """Restore physical state, then begin a fresh return leg to the charger."""

        observation, _ = self.reset(
            seed=int(world_seed) + int(anchor["scene_index"]),
            options={
                "start_position": np.asarray(anchor["validation_start_position"], dtype=np.float32),
                "start_velocity": np.zeros(3, dtype=np.float32),
                "task_point": np.asarray(anchor["active_goal"], dtype=np.float32),
                "static_obstacles": list(anchor["obstacle_layout"]),
            },
        )
        del observation
        if obstacle_layout_sha256(self.base.static_obstacle_layout()) != str(
            anchor["obstacle_layout_sha256"]
        ):
            raise RuntimeError("restored obstacle layout does not match the anchor")
        world_extent = np.asarray(
            [self.base.length, self.base.width, self.base.height], dtype=np.float32
        )
        position = self.base._validate_position(
            sanitize_snapshot_position(
                np.asarray(anchor["position"], dtype=np.float32),
                world_extent=world_extent,
                safe_radius=self.base.safe_radius,
            ),
            "anchor_position",
            check_obstacles=False,
        )
        velocity = self.base._validate_velocity(
            np.asarray(anchor["velocity"], dtype=np.float32)
        )
        charger = self.base._validate_task_point(
            np.asarray(anchor["charger_goal"], dtype=np.float32), position
        )
        self.base.agent.pos = position.copy()
        self.base.agent.prev_pos = position.copy()
        self.base.agent.last_pos = position.copy()
        self.base.agent.spawn_pos = position.copy()
        self.base.agent.vel = velocity.copy()
        self.base.agent.prev_collided = False
        self.base.agent.collided = False
        self.base.current_task_point = charger.copy()
        self.base.agent.goal = charger.copy()
        self.base.agent.reached = False
        self.base.current_step = 0
        self.base.steps_in_current_task = 0
        self.base.simulation_time = 0.0
        self.base.agent_paths = [[position.copy()]]
        self.base._start_goal_trajectory(charger)
        self.base._current_goal_initial_distance = float(np.linalg.norm(charger - position))
        self.base._current_goal_path_length = 0.0
        self.base._update_lidar()
        self.steps, self.raw_return, self.energy = 0, 0.0, 0.0
        self.initial_distance = self.base._current_goal_initial_distance
        self.finished = False
        self.parked = self.park_after_terminal = False
        self.collision_count = 0
        self.hocbf_step_interventions = 0
        return self._observation(self.base._active_goal_sac_observation()), {
            "anchor_id": str(anchor["anchor_id"]),
            "episode_seed": self.episode_seed,
        }

    def step(self, action):
        before = int(self.base.safety_interventions)
        observation, reward, terminated, truncated, info = super().step(action)
        intervened = int(self.base.safety_interventions) > before
        self.hocbf_step_interventions += int(intervened)
        info["hocbf_intervened"] = intervened
        if "navigation_episode" in info:
            info["navigation_episode"]["hocbf_step_interventions"] = (
                self.hocbf_step_interventions
            )
        return observation, reward, terminated, truncated, info
