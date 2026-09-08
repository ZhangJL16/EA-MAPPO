from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .model import RecurrentSetActorCritic


@dataclass(frozen=True)
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.20
    entropy_coefficient: float = 0.002
    value_coefficient: float = 0.5
    distillation_coefficient: float = 0.20
    max_grad_norm: float = 0.5
    update_epochs: int = 4
    sequence_length: int = 64
    minibatch_sequences: int = 4
    target_kl: float = 0.03

    def __post_init__(self) -> None:
        if not 0.0 < self.gamma <= 1.0 or not 0.0 < self.gae_lambda <= 1.0:
            raise ValueError("gamma and gae_lambda must lie in (0, 1]")
        if self.clip_ratio <= 0.0 or self.max_grad_norm <= 0.0:
            raise ValueError("clip_ratio and max_grad_norm must be positive")
        if min(self.update_epochs, self.sequence_length, self.minibatch_sequences) <= 0:
            raise ValueError("PPO update schedule must be positive")
        if min(
            self.entropy_coefficient,
            self.value_coefficient,
            self.distillation_coefficient,
            self.target_kl,
        ) < 0.0:
            raise ValueError("PPO coefficients must be nonnegative")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class ConstraintState:
    collision_multiplier: float = 0.0
    intervention_multiplier: float = 0.0
    collision_budget: float = 0.0
    intervention_budget: float = 0.0
    dual_learning_rate: float = 0.05
    maximum_multiplier: float = 20.0

    def __post_init__(self) -> None:
        if min(
            self.collision_multiplier,
            self.intervention_multiplier,
            self.collision_budget,
            self.intervention_budget,
            self.dual_learning_rate,
            self.maximum_multiplier,
        ) < 0.0:
            raise ValueError("constraint state values must be nonnegative")
        if self.maximum_multiplier <= 0.0:
            raise ValueError("maximum_multiplier must be positive")

    def update(self, collision_mean: float, intervention_mean: float) -> None:
        self.collision_multiplier = float(
            np.clip(
                self.collision_multiplier
                + self.dual_learning_rate * (collision_mean - self.collision_budget),
                0.0,
                self.maximum_multiplier,
            )
        )
        self.intervention_multiplier = float(
            np.clip(
                self.intervention_multiplier
                + self.dual_learning_rate
                * (intervention_mean - self.intervention_budget),
                0.0,
                self.maximum_multiplier,
            )
        )

    def as_dict(self) -> dict[str, float]:
        return {
            key: float(value)
            for key, value in asdict(self).items()
        }


@dataclass(frozen=True)
class RolloutBatch:
    observations: torch.Tensor
    feedback: torch.Tensor
    hiddens: torch.Tensor
    episode_starts: torch.Tensor
    actions: torch.Tensor
    executed_actions: torch.Tensor
    old_log_probs: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor
    collision_costs: torch.Tensor
    intervention_costs: torch.Tensor
    dones: torch.Tensor
    last_values: torch.Tensor

    def validate(self) -> None:
        time_steps, environments = self.rewards.shape
        leading = (time_steps, environments)
        scalar_fields = (
            self.episode_starts,
            self.old_log_probs,
            self.collision_costs,
            self.intervention_costs,
            self.dones,
        )
        if any(tuple(field.shape) != leading for field in scalar_fields):
            raise ValueError("rollout scalar tensors have inconsistent leading shapes")
        if self.observations.shape[:2] != leading or self.feedback.shape[:2] != leading:
            raise ValueError("rollout input tensors have inconsistent leading shapes")
        if self.hiddens.shape[:2] != leading:
            raise ValueError("rollout hidden tensor has inconsistent leading shapes")
        if self.actions.shape[:2] != leading or self.executed_actions.shape != self.actions.shape:
            raise ValueError("rollout action tensors are inconsistent")
        if self.values.shape != (time_steps, environments, 3):
            raise ValueError("rollout values must have three reward/cost heads")
        if self.last_values.shape != (environments, 3):
            raise ValueError("last_values must have shape (environments, 3)")
        tensors = (
            self.observations,
            self.feedback,
            self.hiddens,
            self.actions,
            self.executed_actions,
            self.old_log_probs,
            self.values,
            self.rewards,
            self.collision_costs,
            self.intervention_costs,
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


def _sequence_minibatches(
    time_steps: int,
    environments: int,
    sequence_length: int,
    minibatch_sequences: int,
    rng: np.random.Generator,
) -> list[list[tuple[int, int]]]:
    if time_steps % sequence_length != 0:
        raise ValueError("rollout steps must be divisible by sequence_length")
    sequences = [
        (start, environment)
        for start in range(0, time_steps, sequence_length)
        for environment in range(environments)
    ]
    rng.shuffle(sequences)
    return [
        sequences[index : index + minibatch_sequences]
        for index in range(0, len(sequences), minibatch_sequences)
    ]


def _stack_sequences(
    tensor: torch.Tensor,
    selected: list[tuple[int, int]],
    length: int,
) -> torch.Tensor:
    return torch.stack(
        [tensor[start : start + length, environment] for start, environment in selected],
        dim=1,
    )


def ppo_update(
    model: RecurrentSetActorCritic,
    optimizer: torch.optim.Optimizer,
    rollout: RolloutBatch,
    constraints: ConstraintState,
    config: PPOConfig,
    *,
    device: torch.device,
    rng: np.random.Generator,
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
    collision_advantage, collision_return = compute_gae(
        rollout.collision_costs,
        rollout.values[..., 1],
        rollout.dones,
        rollout.last_values[:, 1],
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
    )
    intervention_advantage, intervention_return = compute_gae(
        rollout.intervention_costs,
        rollout.values[..., 2],
        rollout.dones,
        rollout.last_values[:, 2],
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
    )
    collision_mean = float(rollout.collision_costs.mean())
    intervention_mean = float(rollout.intervention_costs.mean())
    constraints.update(collision_mean, intervention_mean)
    policy_advantage = (
        reward_advantage
        - constraints.collision_multiplier * collision_advantage
        - constraints.intervention_multiplier * intervention_advantage
    )
    policy_advantage = (
        policy_advantage - policy_advantage.mean()
    ) / policy_advantage.std(unbiased=False).clamp_min(1e-6)
    returns = torch.stack(
        (reward_return, collision_return, intervention_return),
        dim=-1,
    )
    time_steps, environments = rollout.rewards.shape
    metric_rows: list[dict[str, float]] = []
    early_stop = False
    for _epoch in range(config.update_epochs):
        minibatches = _sequence_minibatches(
            time_steps,
            environments,
            config.sequence_length,
            config.minibatch_sequences,
            rng,
        )
        for selected in minibatches:
            observations = _stack_sequences(
                rollout.observations, selected, config.sequence_length
            ).to(device)
            feedback = _stack_sequences(
                rollout.feedback, selected, config.sequence_length
            ).to(device)
            episode_starts = _stack_sequences(
                rollout.episode_starts, selected, config.sequence_length
            ).to(device)
            actions = _stack_sequences(
                rollout.actions, selected, config.sequence_length
            ).to(device)
            executed_actions = _stack_sequences(
                rollout.executed_actions, selected, config.sequence_length
            ).to(device)
            old_log_probs = _stack_sequences(
                rollout.old_log_probs, selected, config.sequence_length
            ).to(device)
            advantages = _stack_sequences(
                policy_advantage, selected, config.sequence_length
            ).to(device)
            targets = _stack_sequences(
                returns, selected, config.sequence_length
            ).to(device)
            intervention_weights = _stack_sequences(
                rollout.intervention_costs, selected, config.sequence_length
            ).to(device)
            initial_hidden = torch.stack(
                [rollout.hiddens[start, environment] for start, environment in selected]
            ).detach().to(device)
            log_probs, entropies, predicted_values, mean_actions = model.evaluate_sequence(
                observations,
                feedback,
                initial_hidden,
                episode_starts,
                actions,
            )
            log_ratio = log_probs - old_log_probs
            ratio = torch.exp(log_ratio)
            unclipped = ratio * advantages
            clipped = ratio.clamp(
                1.0 - config.clip_ratio,
                1.0 + config.clip_ratio,
            ) * advantages
            policy_loss = -torch.minimum(unclipped, clipped).mean()
            # Navigation rewards change scale sharply when an episode first
            # reaches its terminal bonus. A per-head Huber objective prevents
            # those early critic residuals from monopolizing the shared
            # encoder's globally clipped gradient.
            value_loss = sum(
                F.smooth_l1_loss(predicted_values[..., head], targets[..., head])
                for head in range(3)
            ) / 3.0
            entropy = entropies.mean()
            per_step_distillation = torch.square(
                mean_actions - executed_actions
            ).mean(dim=-1)
            weight_sum = intervention_weights.sum()
            distillation_loss = torch.where(
                weight_sum > 1e-8,
                (per_step_distillation * intervention_weights).sum()
                / weight_sum.clamp_min(1e-8),
                torch.zeros((), device=device),
            )
            loss = (
                policy_loss
                + config.value_coefficient * value_loss
                - config.entropy_coefficient * entropy
                + config.distillation_coefficient * distillation_loss
            )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("R6 PPO produced a non-finite loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = nn.utils.clip_grad_norm_(
                model.parameters(), config.max_grad_norm
            )
            if not bool(torch.isfinite(gradient_norm)):
                raise FloatingPointError("R6 PPO produced a non-finite gradient norm")
            optimizer.step()
            approximate_kl = float(((ratio - 1.0) - log_ratio).mean().detach().cpu())
            metric_rows.append(
                {
                    "loss": float(loss.detach().cpu()),
                    "policy_loss": float(policy_loss.detach().cpu()),
                    "value_loss": float(value_loss.detach().cpu()),
                    "entropy": float(entropy.detach().cpu()),
                    "distillation_loss": float(distillation_loss.detach().cpu()),
                    "gradient_norm": float(gradient_norm.detach().cpu()),
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
            if config.target_kl > 0.0 and approximate_kl > config.target_kl:
                early_stop = True
                break
        if early_stop:
            break
    if not metric_rows:
        raise RuntimeError("R6 PPO update produced no minibatches")
    metrics = {
        key: float(np.mean([row[key] for row in metric_rows]))
        for key in metric_rows[0]
    }
    metrics.update(
        {
            "minibatch_updates": float(len(metric_rows)),
            "early_stop_kl": float(early_stop),
            "mean_collision_cost": collision_mean,
            "mean_intervention_cost": intervention_mean,
            "collision_multiplier": constraints.collision_multiplier,
            "intervention_multiplier": constraints.intervention_multiplier,
            "reward_advantage_mean": float(reward_advantage.mean()),
            "policy_advantage_std": float(policy_advantage.std(unbiased=False)),
        }
    )
    return metrics
