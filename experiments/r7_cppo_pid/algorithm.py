from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .model import R7ActorCritic


@dataclass(frozen=True)
class PIDLagrangianConfig:
    kp: float = 0.1
    ki: float = 0.01
    kd: float = 0.01
    derivative_delay: int = 10
    proportional_ema_alpha: float = 0.95
    derivative_ema_alpha: float = 0.95
    cost_limit: float = 300.0
    penalty_max: float = 100.0
    initial_integral: float = 0.001
    sum_normalization: bool = True

    def __post_init__(self) -> None:
        if min(
            self.kp,
            self.ki,
            self.kd,
            self.proportional_ema_alpha,
            self.derivative_ema_alpha,
            self.cost_limit,
            self.penalty_max,
            self.initial_integral,
        ) < 0.0:
            raise ValueError("PID-Lagrangian parameters must be nonnegative")
        if self.derivative_delay <= 0:
            raise ValueError("derivative_delay must be positive")
        if not 0.0 <= self.proportional_ema_alpha < 1.0:
            raise ValueError("proportional EMA alpha must lie in [0, 1)")
        if not 0.0 <= self.derivative_ema_alpha < 1.0:
            raise ValueError("derivative EMA alpha must lie in [0, 1)")
        if self.penalty_max <= 0.0:
            raise ValueError("penalty_max must be positive")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class PIDLagrangian:
    """Filtered PID multiplier used by ICML-2020 CPPO-PID.

    The update follows the public OmniSafe reference semantics, with an
    integral projection at zero, EMA-filtered proportional/derivative signals,
    delayed positive derivative response, and optional sum normalization.
    """

    def __init__(self, config: PIDLagrangianConfig) -> None:
        self.config = config
        self.integral = float(config.initial_integral)
        self.proportional_error = 0.0
        self.filtered_cost = 0.0
        self.cost_history: deque[float] = deque(maxlen=config.derivative_delay)
        self.cost_history.append(0.0)
        self.penalty = 0.0
        self.updates = 0

    def update(self, episode_cost_mean: float) -> float:
        if not np.isfinite(episode_cost_mean) or episode_cost_mean < 0.0:
            raise ValueError("episode cost mean must be finite and nonnegative")
        error = float(episode_cost_mean - self.config.cost_limit)
        self.integral = max(0.0, self.integral + self.config.ki * error)
        alpha_p = self.config.proportional_ema_alpha
        self.proportional_error = (
            alpha_p * self.proportional_error + (1.0 - alpha_p) * error
        )
        alpha_d = self.config.derivative_ema_alpha
        self.filtered_cost = (
            alpha_d * self.filtered_cost + (1.0 - alpha_d) * episode_cost_mean
        )
        derivative = max(0.0, self.filtered_cost - self.cost_history[0])
        raw_penalty = (
            self.config.kp * self.proportional_error
            + self.integral
            + self.config.kd * derivative
        )
        self.penalty = max(0.0, raw_penalty)
        if self.config.sum_normalization:
            # Stooke et al.'s scale-invariant sum normalization is implemented
            # in the actor advantage as division by (1 + penalty).
            pass
        else:
            self.penalty = min(self.penalty, self.config.penalty_max)
        self.cost_history.append(self.filtered_cost)
        self.updates += 1
        return self.penalty

    def state_dict(self) -> dict[str, object]:
        return {
            "config": self.config.as_dict(),
            "integral": self.integral,
            "proportional_error": self.proportional_error,
            "filtered_cost": self.filtered_cost,
            "cost_history": list(self.cost_history),
            "penalty": self.penalty,
            "updates": self.updates,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        """Restore a PID controller without silently changing its contract."""
        if state.get("config") != self.config.as_dict():
            raise ValueError("PID-Lagrangian checkpoint configuration mismatch")
        history = [float(value) for value in state["cost_history"]]
        if not history or len(history) > self.config.derivative_delay:
            raise ValueError("invalid PID-Lagrangian cost history")
        scalar_fields = (
            "integral",
            "proportional_error",
            "filtered_cost",
            "penalty",
        )
        scalars = {key: float(state[key]) for key in scalar_fields}
        if not all(np.isfinite(value) for value in (*scalars.values(), *history)):
            raise ValueError("PID-Lagrangian checkpoint contains non-finite state")
        if scalars["integral"] < 0.0 or scalars["penalty"] < 0.0:
            raise ValueError("PID-Lagrangian checkpoint contains negative state")
        self.integral = scalars["integral"]
        self.proportional_error = scalars["proportional_error"]
        self.filtered_cost = scalars["filtered_cost"]
        self.penalty = scalars["penalty"]
        self.cost_history.clear()
        self.cost_history.extend(history)
        self.updates = int(state["updates"])
        if self.updates < 0:
            raise ValueError("PID-Lagrangian update count must be nonnegative")


@dataclass(frozen=True)
class CPPOPIDConfig:
    gamma: float = 0.99
    cost_gamma: float = 0.99
    gae_lambda: float = 0.95
    cost_gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    entropy_coefficient: float = 0.0
    actor_max_grad_norm: float = 40.0
    critic_max_grad_norm: float = 40.0
    update_epochs: int = 10
    minibatch_size: int = 256
    target_kl: float = 0.02

    def __post_init__(self) -> None:
        probabilities = (
            self.gamma,
            self.cost_gamma,
            self.gae_lambda,
            self.cost_gae_lambda,
        )
        if any(not 0.0 < value <= 1.0 for value in probabilities):
            raise ValueError("discount and GAE factors must lie in (0, 1]")
        if min(
            self.clip_ratio,
            self.actor_max_grad_norm,
            self.critic_max_grad_norm,
            self.update_epochs,
            self.minibatch_size,
        ) <= 0:
            raise ValueError("CPPO-PID schedule parameters must be positive")
        if min(self.entropy_coefficient, self.target_kl) < 0.0:
            raise ValueError("entropy coefficient and target KL must be nonnegative")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class RolloutBatch:
    observations: torch.Tensor
    latents: torch.Tensor
    old_log_probs: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor
    costs: torch.Tensor
    dones: torch.Tensor
    last_values: torch.Tensor

    def validate(self) -> None:
        time_steps, environments = self.rewards.shape
        leading = (time_steps, environments)
        if any(
            tuple(tensor.shape) != leading
            for tensor in (self.old_log_probs, self.costs, self.dones)
        ):
            raise ValueError("rollout scalar tensors have inconsistent shapes")
        if self.observations.shape[:2] != leading or self.latents.shape[:2] != leading:
            raise ValueError("rollout observation/latent shapes are inconsistent")
        if self.values.shape != (time_steps, environments, 2):
            raise ValueError("rollout must have reward and scalar-cost values")
        if self.last_values.shape != (environments, 2):
            raise ValueError("last_values must have shape (environments, 2)")
        tensors = (
            self.observations,
            self.latents,
            self.old_log_probs,
            self.values,
            self.rewards,
            self.costs,
            self.last_values,
        )
        if not all(bool(torch.isfinite(tensor).all()) for tensor in tensors):
            raise ValueError("rollout contains non-finite values")


def compute_gae(
    signal: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    last_values: torch.Tensor,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if signal.shape != values.shape or signal.shape != dones.shape:
        raise ValueError("signal, values, and dones must have identical shapes")
    if last_values.shape != (signal.shape[1],):
        raise ValueError("last_values must have one value per environment")
    advantages = torch.zeros_like(signal)
    running = torch.zeros_like(last_values)
    next_values = last_values
    for index in range(signal.shape[0] - 1, -1, -1):
        nonterminal = 1.0 - dones[index]
        delta = signal[index] + gamma * next_values * nonterminal - values[index]
        running = delta + gamma * gae_lambda * nonterminal * running
        advantages[index] = running
        next_values = values[index]
    return advantages, advantages + values


def _standardize(values: torch.Tensor) -> torch.Tensor:
    return (values - values.mean()) / values.std(unbiased=False).clamp_min(1e-6)


def cppo_pid_update(
    model: R7ActorCritic,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    rollout: RolloutBatch,
    pid: PIDLagrangian,
    config: CPPOPIDConfig,
    *,
    device: torch.device,
    rng: np.random.Generator,
    episode_cost_mean: float | None,
) -> dict[str, float]:
    rollout.validate()
    reward_advantage, reward_return = compute_gae(
        rollout.rewards,
        rollout.values[..., 0],
        rollout.dones,
        rollout.last_values[:, 0],
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
    )
    cost_advantage, cost_return = compute_gae(
        rollout.costs,
        rollout.values[..., 1],
        rollout.dones,
        rollout.last_values[:, 1],
        gamma=config.cost_gamma,
        gae_lambda=config.cost_gae_lambda,
    )
    if episode_cost_mean is not None:
        pid.update(episode_cost_mean)
    penalty = float(pid.penalty)
    combined_advantage = (
        _standardize(reward_advantage) - penalty * _standardize(cost_advantage)
    ) / (1.0 + penalty)

    flat_observations = rollout.observations.reshape(-1, rollout.observations.shape[-1])
    flat_latents = rollout.latents.reshape(-1, rollout.latents.shape[-1])
    flat_old_log_probs = rollout.old_log_probs.reshape(-1)
    flat_advantages = combined_advantage.reshape(-1)
    flat_reward_returns = reward_return.reshape(-1)
    flat_cost_returns = cost_return.reshape(-1)
    sample_count = flat_advantages.shape[0]
    actor_rows: list[dict[str, float]] = []
    critic_rows: list[dict[str, float]] = []
    actor_updates = 0
    actor_early_stop = False
    actor_parameters = list(model.actor_parameters())
    critic_parameters = list(model.critic_parameters())

    for _epoch in range(config.update_epochs):
        order = rng.permutation(sample_count)
        for start in range(0, sample_count, config.minibatch_size):
            indices = torch.as_tensor(
                order[start : start + config.minibatch_size],
                dtype=torch.long,
                device=device,
            )
            observations = flat_observations.index_select(0, indices.cpu()).to(device)
            latents = flat_latents.index_select(0, indices.cpu()).to(device)
            old_log_probs = flat_old_log_probs.index_select(0, indices.cpu()).to(device)
            advantages = flat_advantages.index_select(0, indices.cpu()).to(device)
            log_probs, entropies = model.evaluate_latents(observations, latents)
            log_ratio = log_probs - old_log_probs
            ratio = torch.exp(log_ratio)
            unclipped = ratio * advantages
            clipped = ratio.clamp(
                1.0 - config.clip_ratio,
                1.0 + config.clip_ratio,
            ) * advantages
            policy_loss = -torch.minimum(unclipped, clipped).mean()
            entropy = entropies.mean()
            actor_loss = policy_loss - config.entropy_coefficient * entropy
            if not bool(torch.isfinite(actor_loss)):
                raise FloatingPointError("R7 CPPO-PID produced a non-finite actor loss")
            actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            actor_gradient_norm = nn.utils.clip_grad_norm_(
                actor_parameters,
                config.actor_max_grad_norm,
            )
            actor_optimizer.step()
            approximate_kl = float(((ratio - 1.0) - log_ratio).mean().detach().cpu())
            actor_rows.append(
                {
                    "actor_loss": float(actor_loss.detach().cpu()),
                    "policy_loss": float(policy_loss.detach().cpu()),
                    "entropy": float(entropy.detach().cpu()),
                    "actor_gradient_norm": float(actor_gradient_norm.detach().cpu()),
                    "approximate_kl": approximate_kl,
                    "clip_fraction": float(
                        (torch.abs(ratio - 1.0) > config.clip_ratio)
                        .float()
                        .mean()
                        .detach()
                        .cpu()
                    ),
                }
            )
            actor_updates += 1
            if config.target_kl > 0.0 and approximate_kl > config.target_kl:
                actor_early_stop = True
                break
        if actor_early_stop:
            break

    # Critic fitting is intentionally independent from actor KL stopping.
    for _epoch in range(config.update_epochs):
        order = rng.permutation(sample_count)
        for start in range(0, sample_count, config.minibatch_size):
            indices_cpu = torch.as_tensor(
                order[start : start + config.minibatch_size], dtype=torch.long
            )
            observations = flat_observations.index_select(0, indices_cpu).to(device)
            reward_targets = flat_reward_returns.index_select(0, indices_cpu).to(device)
            cost_targets = flat_cost_returns.index_select(0, indices_cpu).to(device)
            predicted = model.values(observations)
            reward_value_loss = F.smooth_l1_loss(predicted[:, 0], reward_targets)
            cost_value_loss = F.smooth_l1_loss(predicted[:, 1], cost_targets)
            critic_loss = 0.5 * (reward_value_loss + cost_value_loss)
            if not bool(torch.isfinite(critic_loss)):
                raise FloatingPointError("R7 CPPO-PID produced a non-finite critic loss")
            critic_optimizer.zero_grad(set_to_none=True)
            critic_loss.backward()
            critic_gradient_norm = nn.utils.clip_grad_norm_(
                critic_parameters,
                config.critic_max_grad_norm,
            )
            critic_optimizer.step()
            critic_rows.append(
                {
                    "critic_loss": float(critic_loss.detach().cpu()),
                    "reward_value_loss": float(reward_value_loss.detach().cpu()),
                    "cost_value_loss": float(cost_value_loss.detach().cpu()),
                    "critic_gradient_norm": float(critic_gradient_norm.detach().cpu()),
                }
            )

    if not actor_rows or not critic_rows:
        raise RuntimeError("R7 CPPO-PID update produced no optimizer minibatches")
    metrics = {
        key: float(np.mean([row[key] for row in actor_rows]))
        for key in actor_rows[0]
    }
    metrics.update(
        {
            key: float(np.mean([row[key] for row in critic_rows]))
            for key in critic_rows[0]
        }
    )
    metrics.update(
        {
            "actor_minibatch_updates": float(actor_updates),
            "critic_minibatch_updates": float(len(critic_rows)),
            "actor_early_stop_kl": float(actor_early_stop),
            "lagrange_multiplier": penalty,
            "episode_cost_mean_for_pid": (
                float("nan") if episode_cost_mean is None else float(episode_cost_mean)
            ),
            "mean_step_cost": float(rollout.costs.mean()),
            "reward_advantage_mean": float(reward_advantage.mean()),
            "cost_advantage_mean": float(cost_advantage.mean()),
            "combined_advantage_std": float(combined_advantage.std(unbiased=False)),
        }
    )
    return metrics
