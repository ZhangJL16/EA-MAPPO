"""SAC with one fresh projection-distance auxiliary loss, no extra networks."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.utils import polyak_update
from torch.nn import functional as F

from .correction_supervision import (
    OBSERVATION_DIM, FreshHOCBFTeacher, TeacherConfig, correction_loss, physical_actions,
)


class CorrectionSAC(SAC):
    """SB3 SAC2.8 update equations + an explicitly sparse actor auxiliary loss.

The first k examples of the uniformly sampled SAC minibatch are a uniform
subbatch. Reuse their CURRENT reparameterized actions: no extra stochastic
draws, historical action targets, simulator rollouts or Jacobian caches.
"""

    def __init__(self, *args: Any, correction_weight: float = 0.1,
                 correction_batch_size: int = 8, correction_interval: int = 8,
                 teacher_parameters: dict | None = None, **kwargs: Any):
        if not np.isfinite(correction_weight) or correction_weight < 0:
            raise ValueError("nonnegative finite correction weight required")
        if correction_batch_size <= 0 or correction_interval <= 0:
            raise ValueError("positive correction batch/interval required")
        self.correction_weight = float(correction_weight)
        self.correction_batch_size = int(correction_batch_size)
        self.correction_interval = int(correction_interval)
        self.teacher_parameters = asdict(TeacherConfig(**(teacher_parameters or {})))
        self.correction_totals = {
            "updates": 0, "nonzero_updates": 0, "queries": 0, "valid": 0,
            "corrected": 0, "solver_invalid": 0, "emergency_region": 0,
            "constraint_invalid": 0, "teacher_seconds": 0.0, "loss_sum": 0.0,
        }
        self.teacher: FreshHOCBFTeacher | None = None
        super().__init__(*args, **kwargs)

    def _setup_model(self) -> None:
        super()._setup_model()
        if self.observation_space.shape != (OBSERVATION_DIM,):
            raise ValueError("correction SAC retains the old2056 observation contract")
        if self.action_space.shape != (3,) or not (
            np.all(self.action_space.low == -1) and np.all(self.action_space.high == 1)
        ):
            raise ValueError("normalized three-component SAC actions required")
        if self._vec_normalize_env is not None:
            raise ValueError("teacher requires raw unnormalized sensor observations")
        self.teacher = FreshHOCBFTeacher(TeacherConfig(**self.teacher_parameters))

    def _excluded_save_params(self) -> list[str]:
        return super()._excluded_save_params() + ["teacher"]

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # Keep SAC optimizer/target/entropy ordering unchanged. With lambda=0
        # this is bit-identical to SB3 SAC on the same CPU RNG/replay fixture.
        self.policy.set_training_mode(True)
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]
        self._update_learning_rate(optimizers)
        ent_coef_losses, ent_coefs, actor_losses, critic_losses = [], [], [], []
        losses = []
        for gradient_step in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma
            if self.use_sde:
                self.actor.reset_noise()
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)
            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                ent_coef = torch.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor
            ent_coefs.append(ent_coef.item())
            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()
            with torch.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                next_q = torch.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q, _ = torch.min(next_q, dim=1, keepdim=True)
                next_q = next_q - ent_coef * next_log_prob.reshape(-1, 1)
                targets = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q
            current_q = self.critic(replay_data.observations, replay_data.actions)
            critic_loss_value = 0.5 * sum(F.mse_loss(q, targets) for q in current_q)
            critic_losses.append(critic_loss_value.item())
            self.critic.optimizer.zero_grad()
            critic_loss_value.backward()
            self.critic.optimizer.step()
            q_pi = torch.cat(self.critic(replay_data.observations, actions_pi), dim=1)
            min_q, _ = torch.min(q_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_q).mean()
            update = self._n_updates + gradient_step
            if self.correction_weight > 0 and update % self.correction_interval == 0:
                count = min(batch_size, self.correction_batch_size)
                assert self.teacher is not None
                physical = physical_actions(actions_pi[:count], self.teacher.config)
                teacher_actions, valid, metrics = self.teacher.batch(
                    replay_data.observations[:count], physical)
                auxiliary = correction_loss(physical, teacher_actions, valid)
                if not torch.isfinite(auxiliary):
                    raise FloatingPointError("nonfinite correction loss")
                actor_loss = actor_loss + self.correction_weight * auxiliary
                value = float(auxiliary.detach())
                losses.append(value)
                self.correction_totals["updates"] += 1
                self.correction_totals["nonzero_updates"] += int(value > 1e-12)
                self.correction_totals["loss_sum"] += value
                for key, val in metrics.items():
                    self.correction_totals[key] += val
            actor_losses.append(actor_loss.item())
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)
        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if ent_coef_losses:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))
        self.logger.record("correction/loss", float(np.mean(losses)) if losses else 0.0)
        for key, value in self.correction_totals.items():
            self.logger.record(f"correction/total_{key}", value)


def warm_start(model: CorrectionSAC, source: SAC) -> None:
    """Copy learned function and entropy, NOT old-interface replay/optimizers."""
    if model.observation_space != source.observation_space or model.action_space != source.action_space:
        raise ValueError("warm-start spaces differ")
    if model.num_timesteps or model.replay_buffer.size():
        raise ValueError("warm start requires an unused destination")
    model.policy.load_state_dict(source.policy.state_dict(), strict=True)
    if model.log_ent_coef is None or source.log_ent_coef is None:
        raise ValueError("automatic SAC entropy coefficient required")
    with torch.no_grad():
        model.log_ent_coef.copy_(source.log_ent_coef)
    model.source_transitions = int(source.num_timesteps)
