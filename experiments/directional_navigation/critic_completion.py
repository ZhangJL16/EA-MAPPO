"""Finish the registered value-update budget after PPO's policy KL stop.

The original joint update runs verbatim. Only its unused value-update budget
is completed, using its existing fixed lambda-return targets and Adam state.
This is a value-training ablation, not the complete PPG algorithm.
"""

from __future__ import annotations

import time

import numpy as np
import torch
from torch.nn import functional as F

from experiments.directional_navigation.lagrangian import TwoValuePolicy
from experiments.directional_navigation.recovery_ppo import update_reward_only


def value_parameters(policy: TwoValuePolicy) -> list[torch.nn.Parameter]:
    if policy.share_features_extractor:
        raise ValueError("critic-only completion requires disjoint extractors")
    modules = (
        policy.vf_features_extractor,
        policy.mlp_extractor.value_net,
        policy.value_net,
    )
    values = list({id(p): p for m in modules for p in m.parameters()}.values())
    actor_modules = (
        policy.pi_features_extractor,
        policy.mlp_extractor.policy_net,
        policy.action_net,
    )
    actor_ids = {id(p) for m in actor_modules for p in m.parameters()} | {
        id(policy.log_std)
    }
    if actor_ids & {id(p) for p in values}:
        raise ValueError("actor and critic parameters overlap")
    return values


def update_with_critic_completion(
    policy: TwoValuePolicy,
    data: dict,
    *,
    complete: bool,
    epochs: int = 10,
    batch_size: int = 256,
) -> dict:
    # Control branch and all policy updates use the unmodified legacy routine.
    metrics = update_reward_only(policy, data, epochs=epochs, batch_size=batch_size)
    per_epoch = int(np.ceil(len(data["ar"]) / batch_size))
    planned = epochs * per_epoch
    joint_steps = metrics["gradient_steps"]
    extra = planned - joint_steps if complete else 0
    params = value_parameters(policy)
    value_ids = {id(p) for p in params}
    protected = [
        (p, p.detach().clone()) for p in policy.parameters() if id(p) not in value_ids
    ]
    observations = torch.as_tensor(data["obs"], device=policy.device)
    targets = torch.as_tensor(data["rr"], device=policy.device)

    def mse() -> float:
        with torch.no_grad():
            total = sum(
                F.mse_loss(
                    policy.predict_values(observations[i : i + 1024]).flatten(),
                    targets[i : i + 1024],
                    reduction="sum",
                ).item()
                for i in range(0, len(targets), 1024)
            )
        return total / len(targets)

    before = mse()
    started, done = time.perf_counter(), 0
    # Do not consume the actor's NumPy or Torch random stream in this phase.
    # Each value pass gets a deterministic independent shuffle.
    rng = np.random.default_rng(918237)
    while done < extra:
        for indices in np.array_split(rng.permutation(len(targets)), per_epoch):
            if done == extra:
                break
            policy.optimizer.zero_grad(set_to_none=True)
            loss = 0.5 * F.mse_loss(
                policy.predict_values(observations[indices]).flatten(), targets[indices]
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite critic completion loss")
            loss.backward()
            if any(p.grad is not None for p, _ in protected):
                raise RuntimeError("critic phase reached a protected parameter")
            norm = torch.nn.utils.clip_grad_norm_(params, 0.5)
            if not torch.isfinite(norm):
                raise FloatingPointError("nonfinite critic completion gradient")
            policy.optimizer.step()
            done += 1
    if any(not torch.equal(p, old) for p, old in protected):
        raise RuntimeError("critic phase changed policy or unused cost head")
    after = mse()
    if not np.isfinite([before, after]).all():
        raise FloatingPointError("nonfinite value diagnostics")
    metrics.update(
        complete_critic_budget=complete,
        policy_gradient_steps=joint_steps,
        critic_gradient_steps=joint_steps + done,
        critic_extra_steps=done,
        critic_planned_steps=planned,
        critic_pass_equivalent=(joint_steps + done) / per_epoch,
        value_target_mse_after_joint=before,
        value_target_mse_after_completion=after,
        actor_unchanged_during_completion=True,
        critic_completion_seconds=time.perf_counter() - started,
    )
    return metrics
