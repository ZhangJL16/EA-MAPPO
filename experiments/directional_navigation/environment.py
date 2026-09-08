"""Finite-task wrapper: retain physical repair, terminate on first contact."""

from __future__ import annotations

from typing import cast

import gymnasium as gym
import numpy as np

from envs.UAVEnergyDeliverySAC import UAVEnergyDeliverySACEnv


class FirstContactNavigation(gym.Wrapper):
    def __init__(
        self,
        *,
        horizon: int = 4000,
        obstacles: int = 24,
        seed_start: int = 193600001,
        seed_stride: int = 1,
        reward_scale: float = 0.01,
    ) -> None:
        if (
            horizon <= 0
            or seed_stride <= 0
            or not np.isfinite(reward_scale)
            or reward_scale <= 0
        ):
            raise ValueError("invalid finite-task contract")
        super().__init__(
            UAVEnergyDeliverySACEnv(
                lidar_enabled=True,
                lidar_horizontal_sectors=128,
                lidar_vertical_sectors=8,
                num_obstacles=obstacles,
                cbf_enabled=False,
                projection_geometry_enabled=False,
                phase1_episode_max_policy_steps=horizon,
                max_steps_per_task=horizon,
            )
        )
        self.base = cast(UAVEnergyDeliverySACEnv, self.env)
        self.horizon, self.reward_scale = horizon, reward_scale
        self.seed_start, self.seed_stride = seed_start, seed_stride
        self.next_episode_index = 0
        self.finished = True
        base_space = self.base.observation_space
        assert isinstance(base_space, gym.spaces.Box)
        self.observation_space = gym.spaces.Box(
            np.append(base_space.low, 0.0).astype(np.float32),
            np.append(base_space.high, 1.0).astype(np.float32),
            dtype=np.float32,
        )

    def _observation(self, obs: np.ndarray) -> np.ndarray:
        return np.append(obs, max(0.0, 1.0 - self.steps / self.horizon)).astype(
            np.float32
        )

    def reset(self, *, seed=None, options=None):
        if seed is None:
            seed = self.seed_start + self.seed_stride * self.next_episode_index
            self.next_episode_index += 1
        obs, _ = self.env.reset(seed=seed, options=options)
        self.episode_seed = int(seed)
        self.steps, self.raw_return, self.energy = 0, 0.0, 0.0
        self.initial_distance = float(
            np.linalg.norm(self.base.active_goal - self.base.agent.pos)
        )
        self.finished = False
        return self._observation(obs), {"episode_seed": self.episode_seed}

    def step(self, action):
        if self.finished:
            raise RuntimeError("reset required after a finite task terminates")
        obs, _, base_terminated, base_truncated, info = self.env.step(action)
        self.steps += 1
        contact = bool(info["obstacle_collision"] or info["boundary_contact"])
        goal = bool(info["is_success"] and not contact)
        deadline = bool(self.steps >= self.horizon or base_truncated)
        if base_terminated and not (contact or goal):
            raise RuntimeError("unexpected termination outside navigation contract")
        components = dict(info["reward_components"])
        if contact:
            # No credit for displacement/goal produced by collision repair.
            for key in (
                "progress_reward_component",
                "velocity_reward_component",
                "task_completion_reward_component",
            ):
                components[key] = 0.0
        raw_reward = float(sum(components.values()))
        self.raw_return += raw_reward
        self.energy += float(info["realized_energy_cost"])
        self.finished = bool(contact or goal or deadline)
        light = {
            "cost": float(contact),
            "is_success": goal,
            "physics_substeps": info["physics_substeps"],
            "obstacle_collision": bool(info["obstacle_collision"]),
            "boundary_contact": bool(info["boundary_contact"]),
            "raw_reward": raw_reward,
        }
        if self.finished:
            outcome = "contact" if contact else "safe_goal" if goal else "timeout"
            remaining = float(
                np.linalg.norm(self.base.active_goal - self.base.agent.pos)
            )
            light["navigation_episode"] = {
                "seed": self.episode_seed,
                "outcome": outcome,
                "steps": self.steps,
                "raw_return": self.raw_return,
                "energy": self.energy,
                "start_distance": self.initial_distance,
                "remaining_distance": remaining,
                "progress_fraction": 1.0 - remaining / max(self.initial_distance, 1e-8),
                "success_path_ratio": (
                    self.base._current_goal_path_length
                    / max(self.initial_distance, 1e-8)
                    if goal
                    else None
                ),
            }
        # A finite task deadline is a true terminal, NOT an artificial rollout cut.
        return (
            self._observation(obs),
            raw_reward * self.reward_scale,
            self.finished,
            False,
            light,
        )

    def episode_cursor(self) -> int:
        return self.next_episode_index

    def restore_episode_cursor(self, value: int) -> None:
        if value < 0:
            raise ValueError("negative episode cursor")
        self.next_episode_index = int(value)
