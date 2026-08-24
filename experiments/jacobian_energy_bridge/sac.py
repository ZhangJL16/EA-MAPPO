from __future__ import annotations

from typing import Any

import numpy as np
import torch as th
from stable_baselines3 import SAC
from stable_baselines3.common.utils import polyak_update
from torch.nn import functional as F

from .energy_model import ActionConditionedEnergyCritic
from .losses import masked_energy_bridge_loss, shield_consistency_loss
from .safety_buffer import SafetyBridgeReplay


class JacobianBridgeSAC(SAC):
    def __init__(
        self,
        *args: Any,
        shield_loss_weight: float = 0.0,
        energy_loss_weight: float = 0.0,
        bridge_batch_size: int = 256,
        bridge_learning_starts: int = 10_000,
        bridge_trust_region: float = 0.35,
        bridge_gradient_clip: float = 10.0,
        **kwargs: Any,
    ) -> None:
        if min(
            shield_loss_weight,
            energy_loss_weight,
            bridge_learning_starts,
            bridge_gradient_clip,
        ) < 0.0:
            raise ValueError("bridge loss configuration must be nonnegative")
        if bridge_batch_size <= 0 or bridge_trust_region <= 0.0:
            raise ValueError("bridge batch size and trust region must be positive")
        self.shield_loss_weight = float(shield_loss_weight)
        self.energy_loss_weight = float(energy_loss_weight)
        self.bridge_batch_size = int(bridge_batch_size)
        self.bridge_learning_starts = int(bridge_learning_starts)
        self.bridge_trust_region = float(bridge_trust_region)
        self.bridge_gradient_clip = float(bridge_gradient_clip)
        self.bridge_replay: SafetyBridgeReplay | None = None
        self.bridge_energy_critic: ActionConditionedEnergyCritic | None = None
        self.bridge_energy_scale = 1.0
        self.bridge_energy_phase_start = 0
        self.bridge_energy_warmup = 0
        self.bridge_energy_ramp = 1
        self.last_bridge_metrics: dict[str, float] = {}
        self._bridge_training_totals = {
            "actor_gradient_steps": 0,
            "bridge_gradient_steps": 0,
            "shield_loss_sum": 0.0,
            "shield_nonzero_steps": 0,
            "energy_loss_sum": 0.0,
            "energy_nonzero_steps": 0,
            "valid_fraction_sum": 0.0,
            "energy_weight_sum": 0.0,
        }
        super().__init__(*args, **kwargs)

    def _excluded_save_params(self) -> list[str]:
        return super()._excluded_save_params() + ["bridge_replay", "bridge_energy_critic"]

    def set_bridge_replay(self, replay: SafetyBridgeReplay | None) -> None:
        self.bridge_replay = replay

    def set_energy_bridge(
        self,
        critic: ActionConditionedEnergyCritic | None,
        *,
        energy_scale: float,
        phase_start_transition: int,
        warmup_transitions: int,
        ramp_transitions: int,
    ) -> None:
        if energy_scale <= 0.0 or warmup_transitions < 0 or ramp_transitions <= 0:
            raise ValueError("invalid energy bridge schedule")
        self.bridge_energy_critic = critic
        self.bridge_energy_scale = float(energy_scale)
        self.bridge_energy_phase_start = int(phase_start_transition)
        self.bridge_energy_warmup = int(warmup_transitions)
        self.bridge_energy_ramp = int(ramp_transitions)
        if critic is not None:
            critic.to(self.device)
            critic.freeze()

    def _energy_weight_now(self) -> float:
        elapsed = self.num_timesteps - self.bridge_energy_phase_start - self.bridge_energy_warmup
        ramp = float(np.clip(elapsed / self.bridge_energy_ramp, 0.0, 1.0))
        return self.energy_loss_weight * ramp

    def bridge_training_metrics(self) -> dict[str, float | int]:
        return self.summarize_bridge_training(self._bridge_training_totals)

    def bridge_training_counters(self) -> dict[str, float | int]:
        return dict(self._bridge_training_totals)

    @staticmethod
    def summarize_bridge_training(
        totals: dict[str, float | int],
    ) -> dict[str, float | int]:
        bridge_steps = int(totals["bridge_gradient_steps"])
        actor_steps = int(totals["actor_gradient_steps"])
        return {
            "actor_gradient_steps": actor_steps,
            "bridge_gradient_steps": bridge_steps,
            "bridge_gradient_step_fraction": bridge_steps / max(actor_steps, 1),
            "mean_shield_loss": float(totals["shield_loss_sum"]) / max(bridge_steps, 1),
            "shield_nonzero_steps": int(totals["shield_nonzero_steps"]),
            "mean_energy_loss": float(totals["energy_loss_sum"]) / max(
                int(totals["energy_nonzero_steps"]),
                1,
            ),
            "energy_nonzero_steps": int(totals["energy_nonzero_steps"]),
            "mean_valid_fraction": float(totals["valid_fraction_sum"]) / max(
                bridge_steps,
                1,
            ),
            "mean_energy_weight_when_bridge_ready": float(
                totals["energy_weight_sum"]
            ) / max(bridge_steps, 1),
        }

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        self.policy.set_training_mode(True)
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]
        self._update_learning_rate(optimizers)

        ent_coef_losses: list[float] = []
        ent_coefs: list[float] = []
        actor_losses: list[float] = []
        critic_losses: list[float] = []
        shield_losses: list[float] = []
        energy_losses: list[float] = []
        bridge_valid_fractions: list[float] = []
        energy_weight = (
            self._energy_weight_now()
            if self.bridge_energy_critic is not None
            else 0.0
        )

        for gradient_step in range(gradient_steps):
            replay_data = self.replay_buffer.sample(
                batch_size,
                env=self._vec_normalize_env,
            )
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma
            if self.use_sde:
                self.actor.reset_noise()
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)
            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                ent_coef = th.exp(self.log_ent_coef.detach())
                assert isinstance(self.target_entropy, float)
                ent_coef_loss = -(
                    self.log_ent_coef * (log_prob + self.target_entropy).detach()
                ).mean()
                ent_coef_losses.append(float(ent_coef_loss.item()))
            else:
                ent_coef = self.ent_coef_tensor
            ent_coefs.append(float(ent_coef.item()))
            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with th.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(
                    replay_data.next_observations
                )
                next_q_values = th.cat(
                    self.critic_target(replay_data.next_observations, next_actions),
                    dim=1,
                )
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                target_q_values = replay_data.rewards + (
                    1 - replay_data.dones
                ) * discounts * next_q_values
            current_q_values = self.critic(replay_data.observations, replay_data.actions)
            critic_loss = 0.5 * sum(
                F.mse_loss(current_q, target_q_values) for current_q in current_q_values
            )
            critic_losses.append(float(critic_loss.item()))
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            q_values_pi = th.cat(
                self.critic(replay_data.observations, actions_pi),
                dim=1,
            )
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()

            bridge_ready = bool(
                self.bridge_replay is not None
                and len(self.bridge_replay) >= max(
                    self.bridge_batch_size,
                    self.bridge_learning_starts,
                )
            )
            if bridge_ready:
                assert self.bridge_replay is not None
                bridge = self.bridge_replay.sample(self.bridge_batch_size, self.device)
                bridge_actions = self.actor(bridge.observations, deterministic=True)
                shield_loss, projected_actions, valid_mask = shield_consistency_loss(
                    bridge_actions,
                    bridge.nominal_actions,
                    bridge.executed_actions,
                    bridge.jacobians,
                    bridge.valid_masks,
                    trust_region_radius=self.bridge_trust_region,
                )
                actor_loss = actor_loss + self.shield_loss_weight * shield_loss
                shield_losses.append(float(shield_loss.detach().cpu()))
                bridge_valid_fractions.append(float(valid_mask.float().mean().detach().cpu()))
                if self.bridge_energy_critic is not None and energy_weight > 0.0:
                    predicted_energy = self.bridge_energy_critic.energy(
                        bridge.compact_energy_states,
                        bridge.safety_contexts,
                        projected_actions,
                    )
                    energy_loss = masked_energy_bridge_loss(
                        predicted_energy,
                        valid_mask,
                        energy_scale=self.bridge_energy_scale,
                    )
                    actor_loss = actor_loss + energy_weight * energy_loss
                    energy_losses.append(float(energy_loss.detach().cpu()))
            actor_losses.append(float(actor_loss.detach().cpu()))
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            if self.bridge_gradient_clip > 0.0:
                th.nn.utils.clip_grad_norm_(
                    self.actor.parameters(),
                    self.bridge_gradient_clip,
                )
            self.actor.optimizer.step()

            if gradient_step % self.target_update_interval == 0:
                polyak_update(
                    self.critic.parameters(),
                    self.critic_target.parameters(),
                    self.tau,
                )
                polyak_update(
                    self.batch_norm_stats,
                    self.batch_norm_stats_target,
                    1.0,
                )

        self._n_updates += gradient_steps
        self._bridge_training_totals["actor_gradient_steps"] += int(gradient_steps)
        self._bridge_training_totals["bridge_gradient_steps"] += len(shield_losses)
        self._bridge_training_totals["shield_loss_sum"] += float(np.sum(shield_losses))
        self._bridge_training_totals["shield_nonzero_steps"] += int(
            np.sum(np.asarray(shield_losses) > 1e-12)
        )
        self._bridge_training_totals["energy_loss_sum"] += float(np.sum(energy_losses))
        self._bridge_training_totals["energy_nonzero_steps"] += int(
            np.sum(np.asarray(energy_losses) > 1e-12)
        )
        self._bridge_training_totals["valid_fraction_sum"] += float(
            np.sum(bridge_valid_fractions)
        )
        self._bridge_training_totals["energy_weight_sum"] += float(
            energy_weight * len(shield_losses)
        )
        self.last_bridge_metrics = {
            "shield_loss": float(np.mean(shield_losses)) if shield_losses else 0.0,
            "energy_loss": float(np.mean(energy_losses)) if energy_losses else 0.0,
            "valid_fraction": (
                float(np.mean(bridge_valid_fractions)) if bridge_valid_fractions else 0.0
            ),
            "energy_weight": float(energy_weight),
        }
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        self.logger.record("jseb/shield_loss", self.last_bridge_metrics["shield_loss"])
        self.logger.record("jseb/energy_loss", self.last_bridge_metrics["energy_loss"])
        self.logger.record("jseb/valid_fraction", self.last_bridge_metrics["valid_fraction"])
        self.logger.record("jseb/energy_weight", self.last_bridge_metrics["energy_weight"])
        if ent_coef_losses:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))
