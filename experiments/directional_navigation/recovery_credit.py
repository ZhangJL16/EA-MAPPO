"""Long-horizon actor credit test with unchanged reward-critic targets."""

from __future__ import annotations

import numpy as np

from experiments.directional_navigation.lagrangian import TwoValuePolicy
from experiments.directional_navigation.recovery_ppo import update_reward_only


def actor_reward_trace(
    data: dict[str, np.ndarray], rows: list[dict], seeds: list[int], trace: float
) -> dict[str, np.ndarray]:
    """Set only actor reward advantage; trace=.95 preserves control bitwise."""
    if trace not in (0.95, 1.0):
        raise ValueError("registered trace must be .95 or 1")
    if len(set(seeds)) != len(seeds) or sorted(r["seed"] for r in rows) != sorted(
        seeds
    ):
        raise ValueError("exact unique complete cohort required")
    if trace == 0.95:
        return data
    by_seed = {r["seed"]: r for r in rows}
    values = data["rr"] - data["ar"]
    offset, advantages = 0, []
    for seed in seeds:
        row = by_seed[seed]
        n = int(row["steps"])
        segment = slice(offset, offset + n)
        a = data["ar"][segment].astype(np.float64)
        v = values[segment].astype(np.float64)
        if n <= 0 or len(a) != n:
            raise ValueError("episode lengths do not cover batch")
        next_a, next_v = np.r_[a[1:], 0.0], np.r_[v[1:], 0.0]
        rewards = a + v - next_v - 0.95 * next_a
        returns = np.cumsum(rewards[::-1])[::-1]
        if not np.isclose(returns[0], row["raw_return"] * 0.01, atol=2e-4, rtol=0):
            raise ValueError("reconstructed scaled reward does not match episode")
        advantages.append((returns - v).astype(np.float32))
        offset += n
    if offset != len(values):
        raise ValueError("unregistered transitions in reward batch")
    modified = dict(data)
    modified["ar"] = np.concatenate(advantages)
    return modified


def update_actor_reward_trace(
    policy: TwoValuePolicy,
    data: dict[str, np.ndarray],
    rows: list[dict],
    seeds: list[int],
    trace: float,
    *,
    epochs: int = 10,
) -> dict:
    modified = actor_reward_trace(data, rows, seeds, trace)
    metrics = update_reward_only(policy, modified, epochs=epochs)
    metrics.update(
        reward_actor_trace=trace,
        reward_actor_advantage_std=float(np.std(modified["ar"])),
        reward_gae95_advantage_std=float(np.std(data["ar"])),
        reward_critic_targets_unchanged=modified["rr"] is data["rr"],
    )
    return metrics
