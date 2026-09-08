"""User-locked contact rewards and nonterminal collision recovery.

Historical FirstContactNavigation is preserved, not used for new rollouts.
"""

from __future__ import annotations

from typing import cast

import gymnasium as gym
import numpy as np

from envs.UAVEnergyDeliverySAC import UAVEnergyDeliverySACEnv
from experiments.directional_navigation.environment import FirstContactNavigation


class LockedRecoveryPlant(UAVEnergyDeliverySACEnv):
    """Keep legacy local position repair; zero full velocity at repair instant."""

    def __setattr__(self, name, value):
        locked = {
            "boundary_penalty": 1.2,
            "obstacle_collision_penalty": 1.2,
            "repeat_collision_scale": 0.35,
        }
        if name in locked and value != locked[name]:
            raise ValueError(f"user-locked collision setting: {name}")
        super().__setattr__(name, value)

    def _apply_boundary_constraints(self, agent):
        result = super()._apply_boundary_constraints(agent)
        if result[0]:
            agent.vel[:] = 0.0
        return result

    def _resolve_obstacle_collisions(self, agent):
        result = super()._resolve_obstacle_collisions(agent)
        if result[0]:
            agent.vel[:] = 0.0
        return result

    def _reward_components(self, **kwargs):
        components = super()._reward_components(**kwargs)
        scale = 0.35 if self.agent.prev_collided else 1.0
        for event, key in (
            ("boundary_contact", "boundary_penalty_component"),
            ("obstacle_collision", "obstacle_penalty_component"),
        ):
            expected = -1.2 * scale * bool(kwargs[event])
            if components[key] != expected:
                raise RuntimeError("user-locked collision reward changed")
        return components


class RecoveryCohort(FirstContactNavigation):
    """Reuse observation/cursor utilities, never the first-contact step method."""

    def __init__(self, *, horizon: int = 4000, obstacles: int = 24):
        if horizon <= 0:
            raise ValueError("positive horizon required")
        gym.Wrapper.__init__(
            self,
            LockedRecoveryPlant(
                lidar_enabled=True,
                lidar_horizontal_sectors=128,
                lidar_vertical_sectors=8,
                num_obstacles=obstacles,
                cbf_enabled=False,
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
        self.parked = self.park_after_terminal = False
        shape = self.observation_space.shape
        assert shape is not None
        self.parked_observation = np.zeros(shape, np.float32)
        self.collision_count = 0

    def park(self):
        self.parked = self.park_after_terminal = True

    def reset(self, *, seed=None, options=None):
        if seed is None and self.park_after_terminal:
            self.parked = True
            return self.parked_observation.copy(), {}
        self.parked = self.park_after_terminal = False
        self.collision_count = 0
        return super().reset(seed=seed, options=options)

    def step(self, action):
        if self.parked:
            return self.parked_observation.copy(), 0.0, False, False, {"parked": True}
        if self.finished:
            raise RuntimeError("reset required after finite task")
        obs, _, terminated, truncated, info = self.env.step(action)
        self.steps += 1
        contact = bool(info["obstacle_collision"] or info["boundary_contact"])
        goal = bool(info["is_success"])
        if terminated and not goal:
            raise RuntimeError("unexpected termination in navigation-only contract")
        self.collision_count += int(contact)
        components = dict(info["reward_components"])
        if contact:
            for key in (
                "progress_reward_component",
                "velocity_reward_component",
                "task_completion_reward_component",
            ):
                components[key] = 0.0
        raw = float(sum(components.values()))
        self.raw_return += raw
        self.energy += float(info["realized_energy_cost"])
        self.finished = bool(goal or truncated or self.steps >= self.horizon)
        light = {
            "cost": float(contact),
            "collision_count": self.collision_count,
            "is_success": goal,
            "raw_reward": raw,
            "physics_substeps": info["physics_substeps"],
        }
        if self.finished:
            self.park_after_terminal = True
            remaining = float(
                np.linalg.norm(self.base.active_goal - self.base.agent.pos)
            )
            # Legacy outcome is only a collector compatibility field: contact
            # now means any contact by TASK END, never immediate termination.
            light["navigation_episode"] = {
                "seed": self.episode_seed,
                "outcome": "contact"
                if self.collision_count
                else "safe_goal"
                if goal
                else "timeout",
                "goal_reached": goal,
                "safe_goal": goal and self.collision_count == 0,
                "termination": "goal" if goal else "deadline",
                "collision_count": self.collision_count,
                "steps": self.steps,
                "raw_return": self.raw_return,
                "energy": self.energy,
                "start_distance": self.initial_distance,
                "remaining_distance": remaining,
                "success_path_ratio": self.base._current_goal_path_length
                / max(self.initial_distance, 1e-8)
                if goal
                else None,
            }
        return (
            self._observation(obs),
            raw * self.reward_scale,
            self.finished,
            False,
            light,
        )
