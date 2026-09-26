"""Gymnasium interface for raw training and shielded evaluation.

One step is one policy decision.  A voluntary or shield-forced return is
advanced until docking or failure because no policy decisions are available
during that committed return.  Charging remains one time-advancing event.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from .action_adapter import ACTION_COUNT, FEATURE_COUNT, ActionAdapter
from .environment import DualConstraintEnv
from .raw_training_env import RawTrainingEnv
from .synthetic import SyntheticEnergy


class DualConstraintGym(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        map_ids: tuple[int, ...],
        capacity: float,
        full_charge_seconds: float,
        *,
        shielded: bool,
        horizon_s: float = 600.0,
        energy_model: SyntheticEnergy | None = None,
        map_seed: int = 0,
    ) -> None:
        super().__init__()
        if not map_ids or len(set(map_ids)) != len(map_ids):
            raise ValueError("map IDs must be nonempty and distinct")
        self.map_ids = tuple(map_ids)
        self.capacity = capacity
        self.full_charge_seconds = full_charge_seconds
        self.shielded = shielded
        self.horizon_s = horizon_s
        self.energy_model = energy_model or SyntheticEnergy()
        self.map_rng = np.random.default_rng(map_seed)
        self.action_space = gym.spaces.Discrete(ACTION_COUNT)
        self.observation_space = gym.spaces.Box(
            low=-5.0, high=5.0, shape=(FEATURE_COUNT,), dtype=np.float32
        )
        self.env: DualConstraintEnv | RawTrainingEnv | None = None
        self.adapter: ActionAdapter | None = None
        self.current_map_id: int | None = None
        self.policy_decisions = 0

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self.map_rng = np.random.default_rng(seed)
        map_id = (
            int(options["map_id"])
            if options is not None and "map_id" in options
            else int(self.map_rng.choice(self.map_ids))
        )
        if map_id not in self.map_ids:
            raise ValueError("requested map is outside this environment split")
        env_type = DualConstraintEnv if self.shielded else RawTrainingEnv
        self.env = env_type(
            map_id, self.capacity, self.full_charge_seconds,
            horizon_s=self.horizon_s, energy_model=self.energy_model,
        )
        self.adapter = ActionAdapter(self.env.case)
        self.current_map_id = map_id
        self.policy_decisions = 0
        return self.adapter.features(self.env.observe()), {
            "map_id": map_id, "shielded": self.shielded
        }

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self.env is None or self.adapter is None:
            raise RuntimeError("reset before stepping")
        if self.env.done:
            raise RuntimeError("episode complete; reset before another step")
        action_id = int(action)
        if not self.action_space.contains(action_id):
            raise ValueError("invalid discrete action")
        before = self.env.observe()
        before_goal = before["target_xy"]
        before_time = self.env.time_s
        before_distance = self.adapter.route_direction(
            before["position_xy"], before_goal
        )[1]
        first = self.adapter.decode(action_id, before)
        observation, base_reward, done, info = self.env.step(first)
        plant_trace = [info]
        total_reward = base_reward
        subdecisions = 1
        takeover = int(info.get("event") == "return_takeover")
        # A return request commits to station; shield takeovers are mandatory.
        committed = before["mode"] == "flight" and action_id == 9
        while not done and (
            self.env.mode == "return" or
            (committed and self.env.mode == "flight")
        ):
            if self.env.mode == "return":
                command = None
            else:
                command = self.adapter.decode(9, self.env.observe())
            observation, reward, done, info = self.env.step(command)
            plant_trace.append(info)
            total_reward += reward
            subdecisions += 1
            takeover += int(info.get("event") == "return_takeover")
            if subdecisions > 4000:
                raise RuntimeError("committed return exceeded execution budget")
        elapsed = self.env.time_s - before_time
        # Small route-potential shaping helps learning without changing the
        # evaluation completion-count metric.  Do not bridge two different goals.
        if before_goal == observation["target_xy"] and not done:
            after_distance = self.adapter.route_direction(
                observation["position_xy"], before_goal
            )[1]
            total_reward += 0.1 * (before_distance - after_distance) / self.env.case.config.side_m
        total_reward -= 0.001 * elapsed
        total_reward -= 0.2 * takeover
        if info.get("failure_reason") is not None:
            total_reward -= 2.0
        self.policy_decisions += 1
        info = {
            **info,
            "plant_trace": plant_trace,
            "map_id": self.current_map_id,
            "policy_action_id": action_id,
            "policy_decisions": self.policy_decisions,
            "plant_decisions": subdecisions,
            "simulated_elapsed_s": elapsed,
            "time_s": self.env.time_s,
            "energy": self.env.energy,
            "safety_cost": self.env.state.safety_cost,
            "return_takeovers": self.env.return_takeovers,
            "energy_filter_events": self.env.energy_filter_events,
            "charge_events": self.env.charge_events,
            "completed_targets": self.env.completed_targets,
        }
        truncated = done and self.env.failure_reason is None
        terminated = done and self.env.failure_reason is not None
        return self.adapter.features(observation), float(total_reward), terminated, truncated, info
