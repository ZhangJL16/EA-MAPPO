"""Complete, fixed-policy task cohorts; finished slots do not spawn extra tasks."""

from __future__ import annotations

import numpy as np

from experiments.directional_navigation.environment import FirstContactNavigation


class CohortNavigation(FirstContactNavigation):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.parked = False
        self.park_after_terminal = False
        shape = self.observation_space.shape
        assert shape is not None
        self.parked_observation = np.zeros(shape, np.float32)

    def park(self) -> None:
        self.parked = True
        self.park_after_terminal = True

    def reset(self, *, seed=None, options=None):
        if seed is None and self.park_after_terminal:
            self.parked = True
            return self.parked_observation.copy(), {}
        self.parked = False
        self.park_after_terminal = False
        return super().reset(seed=seed, options=options)

    def step(self, action):
        if self.parked:
            return (
                self.parked_observation.copy(),
                0.0,
                False,
                False,
                {"parked": True},
            )
        position = self.base.agent.pos.copy()
        velocity = np.asarray(self.base.agent.vel).copy()
        obs, reward, done, truncated, info = super().step(action)
        if done:
            self.park_after_terminal = True
            info["navigation_episode"].update(
                obstacle_collision=info["obstacle_collision"],
                boundary_contact=info["boundary_contact"],
                position_before=position.tolist(),
                velocity_before=velocity.tolist(),
                nominal_action=np.asarray(action).tolist(),
                position_after=self.base.agent.pos.tolist(),
            )
        return obs, reward, done, truncated, info


def complete_task_cost(rows: list[dict], expected_seeds: list[int]) -> float:
    if len(rows) != len(expected_seeds) or sorted(r["seed"] for r in rows) != sorted(
        expected_seeds
    ):
        raise ValueError(
            "dual update requires exactly the registered complete task cohort"
        )
    if len(set(expected_seeds)) != len(expected_seeds):
        raise ValueError("duplicate task seeds")
    if any(r["outcome"] not in {"safe_goal", "contact", "timeout"} for r in rows):
        raise ValueError("incomplete/unknown task outcome")
    return float(np.mean([r["outcome"] == "contact" for r in rows]))


def projected_multiplier(value: float, cost: float, limit: float, rate: float) -> float:
    if not np.isfinite([value, cost, limit, rate]).all() or value < 0 or rate < 0:
        raise ValueError("invalid multiplier update")
    if not 0 <= cost <= 1 or not 0 <= limit <= 1:
        raise ValueError("cost and limit must be task-event probabilities")
    return max(0.0, value + rate * (cost - limit))


def terminal_gae(
    rewards: np.ndarray, values: np.ndarray, lam: float = 0.95
) -> tuple[np.ndarray, np.ndarray]:
    """Undiscounted GAE for ONE complete task, zero terminal continuation."""
    if rewards.ndim != 1 or rewards.shape != values.shape or not 0 <= lam <= 1:
        raise ValueError("invalid complete-task GAE inputs")
    if not np.isfinite(rewards).all() or not np.isfinite(values).all():
        raise ValueError("nonfinite complete-task targets")
    advantages = np.zeros_like(rewards)
    advantage, next_value = 0.0, 0.0
    for i in range(len(rewards) - 1, -1, -1):
        advantage = float(rewards[i]) + next_value - float(values[i]) + lam * advantage
        advantages[i] = advantage
        next_value = float(values[i])
    return advantages, advantages + values
