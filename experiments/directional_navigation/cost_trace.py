"""GAE actor cost advantage with unchanged complete-task MC critic targets."""

from __future__ import annotations

import numpy as np

from experiments.directional_navigation.cohort import terminal_gae
from experiments.directional_navigation.lagrangian import TwoValuePolicy, update_policy


def actor_cost_trace(
    data: dict[str, np.ndarray], episodes: list[dict], seeds: list[int], trace: float
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    if not 0 <= trace <= 1 or not np.isfinite(trace):
        raise ValueError("trace must be in [0,1]")
    if len(set(seeds)) != len(seeds) or sorted(r["seed"] for r in episodes) != sorted(
        seeds
    ):
        raise ValueError("exact complete task cohort required")
    by_seed = {r["seed"]: r for r in episodes}
    vc = data["rc"] - data["ac"]
    offset, parts = 0, []
    for seed in seeds:
        row = by_seed[seed]
        n = int(row["steps"])
        if n <= 0 or row["outcome"] not in {"safe_goal", "contact", "timeout"}:
            raise ValueError("invalid complete episode")
        segment = slice(offset, offset + n)
        failure = float(row["outcome"] == "contact")
        if len(data["rc"][segment]) != n or not np.allclose(
            data["rc"][segment], failure, atol=1e-5
        ):
            raise ValueError("MC targets/trajectory order do not match episodes")
        costs = np.zeros(n, dtype=np.float32)
        costs[-1] = failure
        parts.append(terminal_gae(costs, vc[segment], trace)[0])
        offset += n
    if offset != len(vc):
        raise ValueError("episode lengths do not cover batch")
    # Preserve MC control bitwise; GAE only replaces the actor's ac field.
    result = dict(data)
    if trace != 1.0:
        result["ac"] = np.concatenate(parts)
    return result, vc


def update_with_cost_trace(
    policy: TwoValuePolicy,
    data: dict[str, np.ndarray],
    episodes: list[dict],
    seeds: list[int],
    multiplier: float,
    trace: float,
    *,
    epochs: int = 10,
) -> dict:
    modified, vc = actor_cost_trace(data, episodes, seeds, trace)
    metrics = update_policy(policy, modified, multiplier, epochs=epochs)
    # The legacy diagnostic rc-ac assumes MC actor advantages. Override its
    # three value-range fields using the ACTUAL frozen critic for GAE.
    metrics.update(
        cost_value_min_before=float(vc.min()),
        cost_value_max_before=float(vc.max()),
        cost_value_outside_unit_interval_before=float(np.mean((vc < 0) | (vc > 1))),
        cost_actor_trace=trace,
        cost_actor_advantage_std=float(np.std(modified["ac"])),
        cost_mc_advantage_std=float(np.std(data["ac"])),
        cost_mc_target_mean=float(np.mean(data["rc"])),
        cost_mc_value_mse_before=float(np.mean((vc - data["rc"]) ** 2)),
    )
    return metrics
