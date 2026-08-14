from __future__ import annotations

from dataclasses import replace
from typing import Any

import gymnasium as gym
import numpy as np

from .environment import NavigationEnv, NavigationRewardConfig
from .obstacles import StaticWorld
from .state import NavigationState, as_vec3


D_NEAR = 2.0


class RelativeGoalNavigationEnv(NavigationEnv):
    """LiDAR-free 7D goal-following environment with fixed physical semantics."""

    def __init__(
        self,
        *,
        world_size_xy: float = 4.0,
        world_height: float = 2.0,
        max_episode_steps: int = 800,
        goal_radius: float = 0.20,
        sampling_margin: float = 0.30,
        near_probability: float = 0.50,
        initial_velocity_fraction: float = 0.10,
        reward_config: NavigationRewardConfig | None = None,
    ) -> None:
        if world_size_xy < 4.0:
            raise ValueError("relative-goal evaluation worlds must be at least 4x4 m")
        if world_height != 2.0:
            raise ValueError("world height is fixed at 2 m")
        if not 0.0 <= near_probability <= 1.0:
            raise ValueError("near_probability must lie in [0, 1]")
        if not 0.0 <= initial_velocity_fraction <= 0.25:
            raise ValueError("initial_velocity_fraction must lie in [0, 0.25]")
        physical_reward = reward_config or NavigationRewardConfig(
            distance_potential_scale=1.0,
            gamma=0.99,
            velocity_toward_goal_weight=0.1,
            time_cost=0.01,
            task_completion_reward=10.0,
            collision_penalty=1.2,
            energy_cost_weight=0.0,
            backup_intervention_cost=0.0,
        )
        if physical_reward.energy_cost_weight != 0.0:
            raise ValueError("relative-goal navigation cannot include an energy reward")
        super().__init__(
            "random_persistent_open.json",
            max_episode_steps=max_episode_steps,
            navigation_energy_capacity=1000.0,
            goal_radius=goal_radius,
            minimum_goal_separation=0.30,
            sampling_margin=sampling_margin,
            reward_config=physical_reward,
        )
        self.world_size_xy = float(world_size_xy)
        self.world_height = float(world_height)
        self.near_probability = float(near_probability)
        self.initial_velocity_fraction = float(initial_velocity_fraction)
        world_size = np.array([self.world_size_xy, self.world_size_xy, self.world_height], dtype=np.float64)
        self.config = replace(self.config, world_size=world_size)
        self.world = StaticWorld(world_size)
        self.scenario = replace(self.scenario, world_size=world_size, world=self.world)
        self.sampled_distance_group = ""
        self.sampled_initial_distance = 0.0
        self.consecutive_negative_progress_steps = 0

        self.observation_layout = {
            "normalized_velocity": slice(0, 3),
            "relative_goal_direction": slice(3, 6),
            "relative_goal_distance": slice(6, 7),
        }
        self.observation_space = gym.spaces.Box(
            np.array([-1.0] * 6 + [0.0], dtype=np.float32),
            np.ones(7, dtype=np.float32),
            dtype=np.float32,
        )

    @property
    def observation_definition(self) -> dict[str, Any]:
        return {
            "contract": "RELATIVE_GOAL_NAVIGATION_7D_V1",
            "dimension": 7,
            "fields": list(self.observation_fields),
            "normalized_velocity": "velocity / fixed componentwise v_max",
            "relative_goal_direction": "(goal-position) / max(norm(goal-position), eps)",
            "relative_goal_distance": "clip(norm(goal-position) / 2m, 0, 1)",
            "D_NEAR_m": D_NEAR,
            "excluded": [
                "absolute_position",
                "absolute_goal_position",
                "station_position",
                "state_of_charge",
                "lidar_distances",
                "lidar_valid",
            ],
        }

    def distance_potential(self, distance: float) -> float:
        value = float(distance)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError("distance must be finite and nonnegative")
        return -value

    def distance_potential_shaping(self, distance_before: float, distance_after: float) -> float:
        return self.reward_config.distance_potential_scale * (float(distance_before) - float(distance_after))

    def _observation(self) -> np.ndarray:
        delta_goal = self.goal - self.state.position
        distance = float(np.linalg.norm(delta_goal))
        direction = delta_goal / max(distance, 1e-12)
        observation = np.concatenate(
            (
                self.state.velocity / self.config.v_max,
                direction,
                np.array([np.clip(distance / D_NEAR, 0.0, 1.0)]),
            )
        )
        if observation.shape != (7,) or not np.all(np.isfinite(observation)):
            raise FloatingPointError("invalid relative-goal navigation observation")
        return np.clip(observation, self.observation_space.low, self.observation_space.high).astype(np.float32)

    def effective_distance_intervals(self) -> dict[str, tuple[float, float]]:
        span = self.config.world_size - 2.0 * self.sampling_margin
        z_span = min(float(span[2]), 0.40)
        maximum = 0.98 * float(np.linalg.norm([span[0], span[1], z_span]))
        intervals = {
            "NEAR": (0.30, min(D_NEAR, maximum)),
            "FAR": (D_NEAR, maximum),
        }
        return {name: interval for name, interval in intervals.items() if interval[1] > interval[0]}

    def _validated_interval(self, requested: tuple[float, float] | None) -> tuple[str, float, float]:
        intervals = self.effective_distance_intervals()
        if requested is not None:
            lower, upper = (float(value) for value in requested)
            effective_max = max(interval[1] for interval in intervals.values())
            clipped = (max(0.30, lower), min(upper, effective_max))
            if clipped[1] <= clipped[0]:
                raise ValueError("requested distance interval is infeasible in this physical world")
            label = "NEAR" if clipped[1] <= D_NEAR else "FAR" if clipped[0] >= D_NEAR else "MIXED"
            return label, clipped[0], clipped[1]
        near_available = "NEAR" in intervals
        far_available = "FAR" in intervals
        if near_available and far_available:
            label = "NEAR" if self.np_random.random() < self.near_probability else "FAR"
        elif near_available:
            label = "NEAR"
        elif far_available:
            label = "FAR"
        else:
            raise RuntimeError("no valid near/far sampling interval")
        lower, upper = intervals[label]
        if upper <= lower:
            raise RuntimeError("sampler contract produced an empty interval")
        return label, lower, upper

    def _sample_start_goal(
        self,
        requested_interval: tuple[float, float] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, str, float]:
        label, lower, upper = self._validated_interval(requested_interval)
        for _ in range(100_000):
            distance = float(self.np_random.uniform(lower, upper))
            delta_z_limit = min(0.40, 0.25 * distance)
            delta_z = float(self.np_random.uniform(-delta_z_limit, delta_z_limit))
            xy_distance = float(np.sqrt(max(distance * distance - delta_z * delta_z, 0.0)))
            angle = float(self.np_random.uniform(0.0, 2.0 * np.pi))
            displacement = np.array([xy_distance * np.cos(angle), xy_distance * np.sin(angle), delta_z])
            low = self.sampling_margin + np.maximum(0.0, -displacement)
            high = self.config.world_size - self.sampling_margin - np.maximum(0.0, displacement)
            if np.any(high <= low):
                continue
            start = self.np_random.uniform(low, high)
            goal = start + displacement
            if self._is_legal_position(start) and self._is_legal_position(goal):
                realized = float(np.linalg.norm(goal - start))
                if not lower <= realized <= upper + 1e-10:
                    raise AssertionError("sampled distance escaped its validated interval")
                return start, goal, label, realized
        raise RuntimeError(f"failed to construct a legal {label} start/goal pair")

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        gym.Env.reset(self, seed=seed)
        reset_options = {} if options is None else dict(options)
        forbidden = {"station_position", "randomize_station", "operational_initial_energy"} & set(reset_options)
        if forbidden:
            raise ValueError(f"relative-goal navigation has no station/energy reset semantics: {sorted(forbidden)}")
        requested_start = reset_options.pop("start_position", None)
        requested_goal = reset_options.pop("goal_position", None)
        requested_interval = reset_options.pop("distance_interval", None)
        if (requested_start is None) != (requested_goal is None):
            raise ValueError("start_position and goal_position must be supplied together")
        if requested_start is None:
            start, goal, distance_group, initial_distance = self._sample_start_goal(requested_interval)
        else:
            if requested_interval is not None:
                raise ValueError("distance_interval cannot be combined with explicit start/goal")
            start = self._validate_reset_position(requested_start, "start_position")
            goal = self._validate_reset_position(requested_goal, "goal_position")
            initial_distance = float(np.linalg.norm(goal - start))
            if initial_distance < 0.30:
                raise ValueError("explicit start/goal distance must be at least 0.30 m")
            distance_group = "NEAR" if initial_distance < D_NEAR else "FAR"
        if "start_velocity" in reset_options:
            velocity = np.clip(
                as_vec3(np.asarray(reset_options.pop("start_velocity"), dtype=np.float64), "start_velocity"),
                -self.config.v_max,
                self.config.v_max,
            )
        else:
            velocity = self.np_random.uniform(
                -self.initial_velocity_fraction * self.config.v_max,
                self.initial_velocity_fraction * self.config.v_max,
            )
        if reset_options:
            raise ValueError(f"unsupported relative-goal reset options: {sorted(reset_options)}")

        self.state = NavigationState(start, velocity, self.navigation_energy_capacity, 0.0)
        self.goal = goal
        self.station_position = goal.copy()
        self.scenario = replace(self.scenario, station_position=goal, task_goal=goal)
        self.last_lidar = self.lidar_model.measure(self.state, self.world, self.np_random)
        self.episode_step = 0
        self.tasks_completed = 0
        self.collision_count = 0
        self.boundary_collision_count = 0
        self.obstacle_collision_count = 0
        self.velocity_saturation_count = 0
        self.cumulative_energy_usage = 0.0
        self.minimum_goal_distance = initial_distance
        self.goal_distance_sum = initial_distance
        self.goal_distance_samples = 1
        self.current_consecutive_boundary_contacts = 0
        self.maximum_consecutive_boundary_contacts = 0
        self.boundary_lock_event_count = 0
        self._episode_index += 1
        self._next_goal_index = 0
        self._episode_truncated = False
        self._start_goal_attempt()
        self.sampled_distance_group = distance_group
        self.sampled_initial_distance = initial_distance
        self.consecutive_negative_progress_steps = 0
        return self._observation(), {
            "sampled_start": start.copy(),
            "sampled_goal": goal.copy(),
            "world_size": self.config.world_size.copy(),
            "distance_group": distance_group,
            "initial_goal_distance": initial_distance,
            "observation_contract": "RELATIVE_GOAL_NAVIGATION_7D_V1",
            "energy_semantics": "NOT_OBSERVED_NOT_REWARDED_NOT_TERMINATING",
            "lidar_observed_by_policy": False,
        }

    def step(self, action: np.ndarray):
        position_before = self.state.position.copy()
        velocity_before = self.state.velocity.copy()
        goal_before = self.goal.copy()
        distance_before = float(np.linalg.norm(goal_before - position_before))
        direction_before = (goal_before - position_before) / max(distance_before, 1e-12)
        physical_action = self.normalized_to_physical_action(action)
        observation, reward, _, truncated, info = super().step(action)
        progress = float(info["goal_progress"])
        negative_progress = progress < -1e-9
        self.consecutive_negative_progress_steps = self.consecutive_negative_progress_steps + 1 if negative_progress else 0
        action_norm = float(np.linalg.norm(physical_action))
        action_alignment = 0.0 if action_norm <= 1e-12 else float(np.dot(physical_action, direction_before) / action_norm)
        velocity_toward_goal_before = float(np.dot(velocity_before, direction_before))
        overshoot = bool(distance_before < D_NEAR and velocity_toward_goal_before > 0.0 and negative_progress)
        terminated = bool(info["task_completed_now"])
        info = dict(info) | {
            "world_size": self.config.world_size.copy(),
            "distance_group": self.sampled_distance_group,
            "initial_goal_distance": self.sampled_initial_distance,
            "far_state": distance_before >= D_NEAR,
            "action_goal_alignment": action_alignment,
            "negative_progress_step": negative_progress,
            "consecutive_negative_progress_steps": self.consecutive_negative_progress_steps,
            "overshoot_event": overshoot,
            "observation_contract": "RELATIVE_GOAL_NAVIGATION_7D_V1",
            "lidar_observed_by_policy": False,
        }
        return observation, reward, terminated, False if terminated else truncated, info

