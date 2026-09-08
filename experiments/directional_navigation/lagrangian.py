"""Finite-task PPO-Lagrangian: standard clipped surrogate, not a safety proof.

Reference: OmniSafe PPOLag's (A_r - lambda A_c)/(1+lambda), and projected
Lagrange ascent. PPO clipping/approximate KL follow the installed SB3 PPO.
Adaptations: gamma=1, complete task cohorts, MC first-contact cost targets.
"""

from __future__ import annotations

import time
from typing import cast

import numpy as np
import torch
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.vec_env import VecEnv
from torch import nn
from torch.nn import functional as F

from experiments.directional_navigation.cohort import complete_task_cost, terminal_gae


class TwoValuePolicy(ActorCriticPolicy):
    def _build(self, lr_schedule: Schedule) -> None:
        super()._build(lr_schedule)
        self.cost_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        nn.init.zeros_(self.cost_net.weight)
        nn.init.zeros_(self.cost_net.bias)
        optimizer_kwargs = dict(self.optimizer_kwargs)
        optimizer_kwargs["lr"] = lr_schedule(1)
        self.optimizer = self.optimizer_class(self.parameters(), **optimizer_kwargs)

    def evaluate_both(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        features = self.extract_features(obs)
        if not isinstance(features, tuple):
            raise TypeError("actor and critic extractors must be separate")
        pi = self.mlp_extractor.forward_actor(features[0])
        vf = self.mlp_extractor.forward_critic(features[1])
        distribution = self._get_action_dist_from_latent(pi)
        if actions is None:
            actions = distribution.get_actions(deterministic=deterministic)
        entropy = distribution.entropy()
        if entropy is None:
            raise ValueError("this implementation requires Gaussian entropy")
        return (
            actions,
            self.value_net(vf).flatten(),
            self.cost_net(vf).flatten(),
            distribution.log_prob(actions),
            entropy,
        )


def clipped_objective(
    log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    reward_adv: torch.Tensor,
    cost_adv: torch.Tensor,
    multiplier: float,
    *,
    normalize: bool = True,
) -> torch.Tensor:
    advantage = (reward_adv - multiplier * cost_adv) / (1.0 + multiplier)
    if normalize:
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
    ratio = (log_prob - old_log_prob).exp()
    return -torch.minimum(advantage * ratio, advantage * ratio.clamp(0.8, 1.2)).mean()


def collect_cohort(
    policy: ActorCriticPolicy,
    vec: VecEnv,
    seeds: list[int],
    *,
    deterministic: bool = False,
    training: bool = True,
) -> tuple[dict, list[dict], dict]:
    if not 0 < len(seeds) <= vec.num_envs or len(set(seeds)) != len(seeds):
        raise ValueError("one unique registered task per active worker required")
    observations = np.zeros(
        (vec.num_envs, *cast(tuple, vec.observation_space.shape)), np.float32
    )
    for i, seed in enumerate(seeds):
        observations[i] = vec.env_method("reset", seed=seed, indices=[i])[0][0]
    for i in range(len(seeds), vec.num_envs):
        vec.env_method("park", indices=[i])
    active = np.arange(vec.num_envs) < len(seeds)
    trajectories: list[list[tuple]] = [[] for _ in seeds]
    episodes: list[dict] = []
    physical_steps, worker_slots = 0, 0
    started = time.perf_counter()
    policy.set_training_mode(False)
    while active.any():
        predictions = None
        with torch.no_grad():
            obs_tensor = torch.as_tensor(observations, device=policy.device)
            if training:
                action, reward_v, cost_v, logprob, _ = cast(
                    TwoValuePolicy, policy
                ).evaluate_both(obs_tensor)
                predictions = [v.cpu().numpy() for v in (reward_v, cost_v, logprob)]
            else:
                action, _, _ = policy(obs_tensor, deterministic=deterministic)
            actions = action.cpu().numpy()
        next_obs, rewards, dones, infos = vec.step(np.clip(actions, -1.0, 1.0))
        for i in np.flatnonzero(active):
            physical_steps += 1
            if training:
                assert predictions is not None
                rewards_v, costs_v, log_probs = predictions
                trajectories[i].append(
                    (
                        observations[i].copy(),
                        actions[i].copy(),
                        rewards[i],
                        float(infos[i]["cost"]),
                        rewards_v[i],
                        costs_v[i],
                        log_probs[i],
                    )
                )
            if dones[i]:
                row = dict(infos[i]["navigation_episode"])
                if row["seed"] != seeds[i]:
                    raise AssertionError("cohort task seed mismatch")
                episodes.append(row)
                active[i] = False
        worker_slots += vec.num_envs
        observations = cast(np.ndarray, next_obs)
    risk = complete_task_cost(episodes, seeds)
    data: dict[str, np.ndarray] = {}
    if training:
        fields: dict[str, list[np.ndarray]] = {
            k: [] for k in ("obs", "actions", "old_logprob", "ar", "ac", "rr", "rc")
        }
        for trajectory in trajectories:
            obs, actions, rewards, costs, vr, vc, logp = map(
                np.asarray, zip(*trajectory)
            )
            ar, rr = terminal_gae(rewards, vr, 0.95)
            # Complete finite tasks permit exact MC cost-to-go targets. No
            # timeouts are excluded and late contacts are not discounted away.
            ac, rc = terminal_gae(costs, vc, 1.0)
            for key, value in zip(fields, (obs, actions, logp, ar, ac, rr, rc)):
                fields[key].append(value.astype(np.float32))
        data = {key: np.concatenate(value) for key, value in fields.items()}
    return (
        data,
        episodes,
        {
            "task_contact_rate": risk,
            "physical_steps": physical_steps,
            "worker_slots": worker_slots,
            "parked_slots": worker_slots - physical_steps,
            "collection_seconds": time.perf_counter() - started,
        },
    )


def update_policy(
    policy: TwoValuePolicy,
    data: dict[str, np.ndarray],
    multiplier: float,
    *,
    epochs: int = 10,
    batch_size: int = 256,
) -> dict:
    if len(data["ar"]) < 2:
        raise ValueError("PPO requires at least two transitions")
    tensors = {k: torch.as_tensor(v, device=policy.device) for k, v in data.items()}
    if not all(torch.isfinite(v).all().item() for v in tensors.values()):
        raise FloatingPointError("nonfinite training data")
    policy.set_training_mode(True)
    started = time.perf_counter()
    logs, stopped = [], False
    gradient_steps = 0
    for _ in range(epochs):
        order = np.random.permutation(len(data["ar"]))
        for indices in np.array_split(
            order, max(1, int(np.ceil(len(order) / batch_size)))
        ):
            batch = {k: v[indices] for k, v in tensors.items()}
            _, vr, vc, logp, _entropy = policy.evaluate_both(
                batch["obs"], batch["actions"]
            )
            log_ratio = logp - batch["old_logprob"]
            kl = ((log_ratio.exp() - 1.0) - log_ratio).mean()
            pi_loss = clipped_objective(
                logp, batch["old_logprob"], batch["ar"], batch["ac"], multiplier
            )
            reward_loss, cost_loss = (
                F.mse_loss(vr, batch["rr"]),
                F.mse_loss(vc, batch["rc"]),
            )
            loss = pi_loss + 0.5 * (reward_loss + cost_loss)
            if not torch.isfinite(loss) or not torch.isfinite(kl):
                raise FloatingPointError("nonfinite objective")
            logs.append(
                [pi_loss.item(), reward_loss.item(), cost_loss.item(), kl.item()]
            )
            if kl.item() > 0.03:  # Same 1.5 * target_KL as SB3 PPO.
                stopped = True
                break
            policy.optimizer.zero_grad()
            loss.backward()
            norm = nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            if not torch.isfinite(norm):
                raise FloatingPointError("nonfinite gradient")
            policy.optimizer.step()
            gradient_steps += 1
        if stopped:
            break
    means = np.mean(logs, axis=0)
    cost_values_before = data["rc"] - data["ac"]
    return {
        "policy_loss": float(means[0]),
        "reward_value_loss": float(means[1]),
        "cost_value_loss": float(means[2]),
        "approximate_kl": float(means[3]),
        "cost_value_min_before": float(cost_values_before.min()),
        "cost_value_max_before": float(cost_values_before.max()),
        "cost_value_outside_unit_interval_before": float(
            np.mean((cost_values_before < 0.0) | (cost_values_before > 1.0))
        ),
        "gradient_steps": gradient_steps,
        "early_kl_stop": stopped,
        "action_std": float(policy.log_std.detach().exp().mean().item()),
        "update_seconds": time.perf_counter() - started,
    }
