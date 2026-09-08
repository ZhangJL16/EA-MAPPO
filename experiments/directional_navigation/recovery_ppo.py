"""Reward-only PPO baseline; audit repeated contact counts without a cost loss."""

from __future__ import annotations

import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experiments.directional_navigation.lagrangian import (
    TwoValuePolicy,
    clipped_objective,
    collect_cohort,
)


def validate_count_targets(data: dict, rows: list[dict], seeds: list[int]) -> None:
    if len(set(seeds)) != len(seeds) or sorted(r["seed"] for r in rows) != sorted(
        seeds
    ):
        raise ValueError("exact unique complete recovery cohort required")
    by_seed = {r["seed"]: r for r in rows}
    offset = 0
    for seed in seeds:
        row = by_seed[seed]
        n = row["steps"]
        values = data["rc"][offset : offset + n]
        if n <= 0 or len(values) != n or not np.isfinite(values).all():
            raise ValueError("invalid complete cost trajectory")
        increments = values - np.r_[values[1:], 0.0]
        # rc = float32(advantage + baseline); allow two float32 ULPs at
        # the finite horizon, not a change to integer contact semantics.
        tolerance = max(1e-5, 2 * float(np.spacing(np.float32(n))))
        if not 0 <= row["collision_count"] <= n:
            raise ValueError("invalid episode collision count")
        if not np.isclose(values[0], row["collision_count"], atol=tolerance, rtol=0):
            raise ValueError("MC count does not match episode collision count")
        if not np.all(
            np.isclose(increments, 0.0, atol=tolerance, rtol=0)
            | np.isclose(increments, 1.0, atol=tolerance, rtol=0)
        ):
            raise ValueError("MC differences must be unified step contacts")
        offset += n
    if offset != len(data["rc"]):
        raise ValueError("unregistered transitions in batch")


def collect_recovery(policy, vec, seeds):
    data, rows, metrics = collect_cohort(policy, vec, seeds)
    validate_count_targets(data, rows, seeds)
    # This probability summary is not the new count objective or a dual input.
    metrics.pop("task_contact_rate")
    metrics.update(
        total_collision_count=sum(r["collision_count"] for r in rows),
        mean_collision_count=float(np.mean([r["collision_count"] for r in rows])),
        goal_reached=sum(r["goal_reached"] for r in rows),
        safe_goal=sum(r["safe_goal"] for r in rows),
    )
    for row in rows:
        row.pop("outcome")
    return data, rows, metrics


def update_reward_only(
    policy: TwoValuePolicy, data: dict, *, epochs=10, batch_size=256
) -> dict:
    """No cost tensor enters the loss, including the old binary cost head."""
    if len(data["ar"]) < 2 or epochs <= 0 or batch_size <= 0:
        raise ValueError("PPO needs at least two transitions")
    tensors = {
        k: torch.as_tensor(data[k], device=policy.device)
        for k in ("obs", "actions", "old_logprob", "ar", "rr")
    }
    if not all(torch.isfinite(v).all() for v in tensors.values()):
        raise FloatingPointError("nonfinite reward-PPO batch")
    before = {k: v.detach().clone() for k, v in policy.cost_net.state_dict().items()}
    policy.set_training_mode(True)
    started = time.perf_counter()
    logs, stopped, steps = [], False, 0
    for _ in range(epochs):
        order = np.random.permutation(len(data["ar"]))
        for indices in np.array_split(
            order, max(1, int(np.ceil(len(order) / batch_size)))
        ):
            b = {k: v[indices] for k, v in tensors.items()}
            _, vr, _, logp, _ = policy.evaluate_both(b["obs"], b["actions"])
            log_ratio = logp - b["old_logprob"]
            kl = ((log_ratio.exp() - 1.0) - log_ratio).mean()
            pi_loss = clipped_objective(
                logp, b["old_logprob"], b["ar"], torch.zeros_like(b["ar"]), 0.0
            )
            value_loss = F.mse_loss(vr, b["rr"])
            loss = pi_loss + 0.5 * value_loss
            if not torch.isfinite(loss) or not torch.isfinite(kl):
                raise FloatingPointError("nonfinite PPO objective")
            logs.append([pi_loss.item(), value_loss.item(), kl.item()])
            if kl.item() > 0.03:
                stopped = True
                break
            policy.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if any(p.grad is not None for p in policy.cost_net.parameters()):
                raise RuntimeError("unused cost head received gradient")
            norm = nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            if not torch.isfinite(norm):
                raise FloatingPointError("nonfinite PPO gradient")
            policy.optimizer.step()
            steps += 1
        if stopped:
            break
    if any(
        not torch.equal(v, before[k]) for k, v in policy.cost_net.state_dict().items()
    ):
        raise RuntimeError("unused cost head changed")
    values = np.mean(logs, axis=0)
    return {
        "policy_loss": float(values[0]),
        "reward_value_loss": float(values[1]),
        "approximate_kl": float(values[2]),
        "gradient_steps": steps,
        "early_kl_stop": stopped,
        "cost_head_unchanged": True,
        "action_std": float(policy.log_std.detach().exp().mean()),
        "update_seconds": time.perf_counter() - started,
    }
