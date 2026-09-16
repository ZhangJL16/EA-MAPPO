from __future__ import annotations

from dataclasses import replace
from typing import Any

import gymnasium as gym
import numpy as np

from .environment import NavigationEnv, NavigationRewardConfig
from .obstacles import StaticWorld
from .state import NavigationState, as_vec3


GOAL_DISTANCE_BINS: tuple[tuple[str, float, float], ...] = (
    ("0.5-2", 0.5, 2.0),
    ("2-4", 2.0, 4.0),
    ("4-8", 4.0, 8.0),
    ("8-12", 8.0, 12.0),
    (">12", 12.0, np.inf),
)


class ScaleInvariantNavigationEnv(NavigationEnv):
    """Obstacle-free navigation with a fixed-physics, map-independent observation.

    This class deliberately does not alter ``NavigationEnv``. It reuses its
    physical transition implementation while replacing the episode world
    sampler, observation contract, and map-invariant progress reward.
    """

    goal_distance_scale = 6.0

    def __init__(
        self,
        *,
        world_size_min: float = 4.0,
        world_size_max: float = 16.0,
        world_height: float = 2.0,
        max_episode_steps: int = 800,
        goal_radius: float = 0.20,
        sampling_margin: float = 0.30,
        initial_velocity_fraction: float = 0.10,
        reward_config: NavigationRewardConfig | None = None,
    ) -> None:
        if not 0.0 < world_size_min <= world_size_max:
            raise ValueError("world-size range must be positive and ordered")
        if world_size_min < 4.0 or world_size_max > 16.0:
            raise ValueError("the scale-invariant training range is restricted to 4-16 m")
        if world_height != 2.0:
            raise ValueError("world height is fixed at 2 m")
        if not 0.0 <= initial_velocity_fraction <= 0.25:
            raise ValueError("initial velocity fraction must lie in [0, 0.25]")
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
            raise ValueError("scale-invariant navigation reward cannot include energy authority")
        super().__init__(
            "random_persistent_open.json",
            max_episode_steps=max_episode_steps,
            navigation_energy_capacity=1000.0,
            goal_radius=goal_radius,
            minimum_goal_separation=0.50,
            sampling_margin=sampling_margin,
            reward_config=physical_reward,
        )
        self.world_size_min = float(world_size_min)
        self.world_size_max = float(world_size_max)
        self.world_height = float(world_height)
        self.initial_velocity_fraction = float(initial_velocity_fraction)
        self._physical_config = self.config
        self.sampled_distance_bin = ""
        self.sampled_initial_distance = 0.0

        cursor = 0
        self.observation_layout = {}
        for name, length in (
            ("normalized_velocity", 3),
            ("relative_goal_direction", 3),
            ("relative_goal_distance", 1),
            ("lidar_distances", self.config.num_lasers),
            ("lidar_valid", self.config.num_lasers),
        ):
            self.observation_layout[name] = slice(cursor, cursor + length)
            cursor += length
        observation_low = np.concatenate(
            (
                -np.ones(3),
                -np.ones(3),
                np.zeros(1 + 2 * self.config.num_lasers),
            )
        )
        observation_high = np.ones(cursor)
        self.observation_space = gym.spaces.Box(
            observation_low.astype(np.float32),
            observation_high.astype(np.float32),
            dtype=np.float32,
        )

    @property
    def observation_definition(self) -> dict[str, Any]:
        return {
            "dimension": int(self.observation_space.shape[0]),
            "fields": list(self.observation_fields),
            "normalized_velocity": "velocity / fixed componentwise v_max",
            "relative_goal_direction": "(goal-position) / max(norm(goal-position), eps)",
            "relative_goal_distance": "clip(norm(goal-position) / 6m, 0, 1)",
            "goal_distance_scale_m": self.goal_distance_scale,
            "lidar_distances": "distance / fixed 6m lidar_range",
            "excluded": ["absolute_position", "absolute_goal", "station_position", "state_of_charge"],
        }

    def distance_potential(self, distance: float) -> float:
        value = float(distance)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError("distance must be finite and nonnegative")
        return -value

    def distance_potential_shaping(self, distance_before: float, distance_after: float) -> float:
        return self.reward_config.distance_potential_scale * (float(distance_before) - float(distance_after))

    def _observation(self) -> np.ndarray:
        if self.last_lidar is None:
            raise RuntimeError("LiDAR packet unavailable")
        delta_goal = self.goal - self.state.position
        distance = float(np.linalg.norm(delta_goal))
        direction = delta_goal / max(distance, 1e-12)
        observation = np.concatenate(
            (
                self.state.velocity / self.config.v_max,
                direction,
                np.array([np.clip(distance / self.goal_distance_scale, 0.0, 1.0)]),
                self.last_lidar.distances / self.config.lidar_range,
                self.last_lidar.valid.astype(np.float64),
            )
        )
        if not np.all(np.isfinite(observation)):
            raise FloatingPointError("nonfinite scale-invariant navigation observation")
        return np.clip(observation, self.observation_space.low, self.observation_space.high).astype(np.float32)

    def _sample_world_width(self, requested: float | None) -> float:
        if requested is not None:
            width = float(requested)
            if not self.world_size_min <= width <= self.world_size_max:
                raise ValueError("requested world width lies outside the configured range")
            return width
        return float(self.np_random.uniform(self.world_size_min, self.world_size_max))

    def _distance_bin(self, label: str | None = None) -> tuple[str, float, float]:
        maximum = float(np.linalg.norm(self.config.world_size - 2.0 * self.sampling_margin))
        feasible = [item for item in GOAL_DISTANCE_BINS if item[1] < maximum - 1e-6]
        if label is not None:
            matches = [item for item in feasible if item[0] == label]
            if not matches:
                raise ValueError(f"distance bin {label!r} is infeasible in this world")
            return matches[0]
        return feasible[int(self.np_random.integers(0, len(feasible)))]

    def _sample_start_goal(self, distance_bin: str | None) -> tuple[np.ndarray, np.ndarray, str, float]:
        label, lower, upper = self._distance_bin(distance_bin)
        maximum = float(np.linalg.norm(self.config.world_size - 2.0 * self.sampling_margin))
        effective_upper = min(upper, maximum * 0.98)
        if effective_upper <= lower:
            raise RuntimeError(f"empty physical distance interval for bin {label}")
        low = np.full(3, self.sampling_margin)
        high = self.config.world_size - self.sampling_margin
        for _ in range(100_000):
            start = self.np_random.uniform(low, high)
            distance = float(self.np_random.uniform(lower, effective_upper))
            delta_z_limit = min(0.40, 0.25 * distance)
            delta_z = float(self.np_random.uniform(-delta_z_limit, delta_z_limit))
            xy_distance = float(np.sqrt(max(distance * distance - delta_z * delta_z, 0.0)))
            angle = float(self.np_random.uniform(0.0, 2.0 * np.pi))
            goal = start + np.array([xy_distance * np.cos(angle), xy_distance * np.sin(angle), delta_z])
            if np.any(goal < low) or np.any(goal > high):
                continue
            if self._is_legal_position(start) and self._is_legal_position(goal):
                realized = float(np.linalg.norm(goal - start))
                return start, goal, label, realized
        raise RuntimeError(f"failed to sample start/goal for physical distance bin {label}")

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
            raise ValueError(f"scale-invariant navigation has no station/energy reset semantics: {sorted(forbidden)}")
        width = self._sample_world_width(reset_options.pop("world_size_xy", None))
        world_size = np.array([width, width, self.world_height], dtype=np.float64)
        self.config = replace(self._physical_config, world_size=world_size)
        self.world = StaticWorld(world_size)

        requested_start = reset_options.pop("start_position", None)
        requested_goal = reset_options.pop("goal_position", None)
        requested_bin = reset_options.pop("distance_bin", None)
        if (requested_start is None) != (requested_goal is None):
            raise ValueError("start_position and goal_position must be supplied together")
        if requested_start is None:
            start, goal, selected_bin, initial_distance = self._sample_start_goal(requested_bin)
        else:
            if requested_bin is not None:
                raise ValueError("distance_bin cannot be combined with explicit start/goal")
            start = self._validate_reset_position(requested_start, "start_position")
            goal = self._validate_reset_position(requested_goal, "goal_position")
            initial_distance = float(np.linalg.norm(goal - start))
            if initial_distance < self.minimum_goal_separation:
                raise ValueError("explicit start/goal separation is too small")
            selected_bin = "EXPLICIT"
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
            raise ValueError(f"unsupported scale-invariant reset options: {sorted(reset_options)}")

        self.state = NavigationState(start, velocity, self.navigation_energy_capacity, 0.0)
        self.goal = goal
        self.station_position = goal.copy()
        self.scenario = replace(
            self.scenario,
            world_size=world_size,
            station_position=goal,
            task_goal=goal,
            world=self.world,
        )
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
        self.sampled_distance_bin = selected_bin
        self.sampled_initial_distance = initial_distance
        return self._observation(), {
            "sampled_start": start.copy(),
            "sampled_goal": goal.copy(),
            "world_size": world_size.copy(),
            "distance_bin": selected_bin,
            "initial_goal_distance": initial_distance,
            "observation_layout": dict(self.observation_layout),
            "observation_contract": "SCALE_INVARIANT_NAVIGATION_71D_V1",
            "energy_semantics": "NOT_OBSERVED_NOT_REWARDED_NOT_TERMINATING",
        }

    def step(self, action: np.ndarray):
        observation, reward, _, truncated, info = super().step(action)
        terminated = bool(info["task_completed_now"])
        info = dict(info) | {
            "world_size": self.config.world_size.copy(),
            "distance_bin": self.sampled_distance_bin,
            "initial_goal_distance": self.sampled_initial_distance,
            "observation_contract": "SCALE_INVARIANT_NAVIGATION_71D_V1",
        }
        return observation, reward, terminated, False if terminated else truncated, info

