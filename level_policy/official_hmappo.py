import glob
import os

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal

from level_policy.mappo import (
    CriticNetwork,
    DiscreteActorNetwork,
    GaussianActorNetwork,
    HybridHighActorNetwork,
    MAPPO,
)


class OfficialHMAPPO(MAPPO):
    """Shared-parameter hierarchical MAPPO.

    This is the hierarchical counterpart of the official MAPPO style: one
    shared low-level actor/critic and one shared high-level actor/critic are
    trained with all agents' rollout samples flattened into a single batch.
    """

    def __init__(self, args):
        self.args = args
        self.n_agents = args.n_agents
        self.low_n_actions = args.n_actions
        self.low_obs_shape = args.obs_shape
        self.low_state_shape = args.state_shape
        self.high_n_actions = int(getattr(args, "high_level_n_actions", 0))
        self.high_mode_n_actions = int(getattr(args, "high_level_mode_n_actions", 0))
        self.high_continuous_dim = max(0, self.high_n_actions - 1)
        self.use_hybrid_high_policy = (
            self.high_mode_n_actions > 1 and self.high_n_actions >= 2
        )
        self.high_obs_shape = int(getattr(args, "high_level_obs_shape", 0))
        self.high_state_shape = int(getattr(args, "high_level_state_shape", 0))
        self.high_actor_obs_shape = self.high_obs_shape
        self.last_train_metrics = {}
        self.high_level_enabled = (
            self.high_n_actions > 0
            and self.high_obs_shape > 0
            and self.high_state_shape > 0
        )

        self.device = torch.device(
            f"cuda:{getattr(self.args, 'gpu_id', 0)}"
            if self.args.cuda and torch.cuda.is_available()
            else "cpu"
        )

        actor_hidden_dim = getattr(args, "actor_hidden_dim", 128)
        critic_hidden_dim = getattr(args, "critic_hidden_dim", 128)
        high_actor_hidden_dim = getattr(args, "high_actor_hidden_dim", actor_hidden_dim)
        high_critic_hidden_dim = getattr(args, "high_critic_hidden_dim", critic_hidden_dim)
        high_lr_actor = getattr(args, "high_lr_actor", args.lr_actor)
        high_lr_critic = getattr(args, "high_lr_critic", args.lr_critic)

        self.low_actor = DiscreteActorNetwork(
            self.low_obs_shape,
            self.low_n_actions,
            actor_hidden_dim,
        ).to(self.device)
        self.low_critic = CriticNetwork(self.low_state_shape, critic_hidden_dim).to(
            self.device
        )
        if self.high_level_enabled:
            if self.use_hybrid_high_policy:
                self.high_actor = HybridHighActorNetwork(
                    self.high_actor_obs_shape,
                    self.high_mode_n_actions,
                    self.high_continuous_dim,
                    high_actor_hidden_dim,
                ).to(self.device)
            else:
                self.high_actor = GaussianActorNetwork(
                    self.high_actor_obs_shape,
                    self.high_n_actions,
                    high_actor_hidden_dim,
                ).to(self.device)
            self.high_critic = CriticNetwork(
                self.high_state_shape,
                high_critic_hidden_dim,
            ).to(self.device)
        else:
            self.high_actor = None
            self.high_critic = None

        self.low_actor_optimizer = torch.optim.Adam(
            self.low_actor.parameters(),
            lr=args.lr_actor,
        )
        self.low_critic_optimizer = torch.optim.Adam(
            self.low_critic.parameters(),
            lr=args.lr_critic,
        )
        self.high_actor_optimizer = (
            torch.optim.Adam(self.high_actor.parameters(), lr=high_lr_actor)
            if self.high_level_enabled
            else None
        )
        self.high_critic_optimizer = (
            torch.optim.Adam(self.high_critic.parameters(), lr=high_lr_critic)
            if self.high_level_enabled
            else None
        )
        self.model_dir = args.model_dir + "/" + args.alg + "/" + args.map
        self.eval_hidden = None

        if self.args.load_model:
            self._load_models()
        pretrained_low_dir = getattr(self.args, "hmappo_pretrained_low_model_dir", "")
        if pretrained_low_dir:
            self._load_pretrained_low_models(pretrained_low_dir)

    @torch.no_grad()
    def choose_action(self, observation, agent_idx, avail_actions, evaluate=False):
        del agent_idx
        obs = torch.tensor(
            observation,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        avail = torch.tensor(
            avail_actions,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        if torch.sum(avail) <= 0:
            return 0
        dist = self._masked_categorical(self.low_actor(obs), avail)
        if evaluate:
            action = torch.argmax(dist.logits, dim=-1)
        else:
            action = dist.sample()
        return int(action.item())

    @torch.no_grad()
    def choose_high_level_action(
        self,
        observation,
        agent_idx,
        avail_actions=None,
        evaluate=False,
    ):
        del agent_idx
        if not self.high_level_enabled:
            return np.zeros((0,), dtype=np.float32)
        obs = torch.tensor(
            observation,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        mode_avail = None
        if avail_actions is not None:
            mode_avail = torch.tensor(
                avail_actions,
                dtype=torch.float32,
                device=self.device,
            ).view(1, -1)

        if self.use_hybrid_high_policy:
            mode_dist, continuous_dist = self._hybrid_high_dist(
                self.high_actor,
                obs,
                agent_idx=None,
                mode_avail=mode_avail,
            )
            if evaluate:
                mode = torch.argmax(mode_dist.logits, dim=-1)
                continuous = continuous_dist.mean
            else:
                mode = mode_dist.sample()
                continuous = continuous_dist.sample()
            continuous = torch.clamp(continuous, -1.0, 1.0)
            action = torch.cat([mode.float().unsqueeze(-1), continuous], dim=-1)
            return action.squeeze(0).detach().cpu().numpy().astype("float32")

        dist = self._gaussian_dist(self.high_actor, obs)
        if evaluate:
            action = dist.mean
        else:
            action = dist.sample()
        action = torch.clamp(action, -1.0, 1.0)
        return action.squeeze(0).detach().cpu().numpy().astype("float32")

    def _compute_multiagent_advantages(
        self,
        rewards,
        values,
        next_values,
        terminated,
        mask,
        durations=None,
    ):
        advantages = torch.zeros_like(rewards)
        gae = torch.zeros(rewards.size(0), rewards.size(2), device=self.device)
        terminated = terminated.unsqueeze(-1)
        if durations is None:
            gamma_k = torch.full_like(rewards, float(self.args.gamma))
            lambda_k = torch.full_like(rewards, float(self.args.gamma * self.args.gae_lambda))
        else:
            durations = torch.clamp(durations, min=1.0)
            if durations.dim() == 2:
                durations = durations.unsqueeze(-1).expand_as(rewards)
            gamma_k = torch.pow(
                torch.full_like(durations, float(self.args.gamma)),
                durations,
            )
            lambda_k = torch.pow(
                torch.full_like(durations, float(self.args.gamma * self.args.gae_lambda)),
                durations,
            )
        for t in reversed(range(rewards.size(1))):
            not_done = 1.0 - terminated[:, t]
            delta = rewards[:, t] + gamma_k[:, t] * next_values[:, t] * not_done - values[:, t]
            gae = delta + lambda_k[:, t] * not_done * gae
            gae = gae * mask[:, t]
            advantages[:, t] = gae
        returns = advantages + values
        valid_advantages = advantages[mask > 0]
        if valid_advantages.numel() > 0:
            advantages = (advantages - valid_advantages.mean()) / (
                valid_advantages.std(unbiased=False) + 1e-8
            )
        return advantages, returns

    def _expand_states_for_agents(self, states, state_dim):
        return (
            states.unsqueeze(2)
            .expand(-1, -1, self.n_agents, -1)
            .reshape(-1, state_dim)
        )

    def learn(self, batch, max_episode_len, train_step, epsilon):
        del max_episode_len, epsilon
        self.last_train_metrics = {}
        batch = self._prepare_batch(batch)
        if not bool(getattr(self.args, "hmappo_freeze_low_level", False)):
            self._learn_shared_low_level(batch)
        if (
            not self.high_level_enabled
            or "high_o" not in batch
            or bool(getattr(self.args, "hmappo_freeze_high_level", False))
        ):
            return

        high_action_key = "high_u"
        if self.use_hybrid_high_policy:
            self._learn_shared_hybrid_high_level(batch, high_action_key)
        else:
            self._learn_shared_continuous_high_level(batch, high_action_key)

    def _learn_shared_low_level(self, batch):
        states = batch["s"]
        next_states = batch["s_next"]
        obs = batch["o"]
        actions = batch["u"].squeeze(-1)
        avail_actions = batch["avail_u"]
        terminated = batch["terminated"].squeeze(-1)
        mask = 1.0 - batch["padded"].squeeze(-1)
        active_mask = batch.get("agent_active_mask", None)
        if active_mask is None:
            active_mask = torch.ones(
                (*mask.shape, self.n_agents, 1),
                dtype=mask.dtype,
                device=self.device,
            )
        active_mask = active_mask.squeeze(-1)
        guard_applied = batch.get("guard_applied", None)
        if guard_applied is not None:
            guard_applied = guard_applied.squeeze(-1)
        rewards = batch["r"]
        if rewards.size(-1) == 1:
            rewards = rewards.expand(-1, -1, self.n_agents)

        episode_num = states.size(0)
        time_len = states.size(1)
        agent_mask = mask.unsqueeze(-1) * active_mask
        flat_obs = obs.reshape(episode_num * time_len * self.n_agents, -1)
        flat_actions = actions.reshape(-1)
        flat_avail = avail_actions.reshape(-1, self.low_n_actions)
        flat_states = self._expand_states_for_agents(states, self.low_state_shape)
        flat_next_states = self._expand_states_for_agents(next_states, self.low_state_shape)

        with torch.no_grad():
            values = self.low_critic(flat_states).reshape(
                episode_num,
                time_len,
                self.n_agents,
            )
            next_values = self.low_critic(flat_next_states).reshape(
                episode_num,
                time_len,
                self.n_agents,
            )
            advantages, returns = self._compute_multiagent_advantages(
                rewards,
                values,
                next_values,
                terminated,
                agent_mask,
            )
            old_dist = self._masked_categorical(self.low_actor(flat_obs), flat_avail)
            old_log_probs = old_dist.log_prob(flat_actions)

        flat_agent_mask = agent_mask.reshape(-1) > 0
        if guard_applied is not None:
            actor_mask = agent_mask * (1.0 - guard_applied)
        else:
            actor_mask = agent_mask
        flat_actor_mask = actor_mask.reshape(-1) > 0
        self._ppo_update_discrete(
            actor=self.low_actor,
            critic=self.low_critic,
            actor_optimizer=self.low_actor_optimizer,
            critic_optimizer=self.low_critic_optimizer,
            flat_obs=flat_obs,
            flat_actions=flat_actions,
            flat_avail=flat_avail,
            flat_states=flat_states,
            advantages=advantages.reshape(-1),
            returns=returns.reshape(-1),
            old_log_probs=old_log_probs,
            flat_agent_mask=flat_agent_mask,
            flat_actor_mask=flat_actor_mask,
            metric_prefix="official_hmappo_low",
        )

    def _ppo_update_discrete(
        self,
        actor,
        critic,
        actor_optimizer,
        critic_optimizer,
        flat_obs,
        flat_actions,
        flat_avail,
        flat_states,
        advantages,
        returns,
        old_log_probs,
        flat_agent_mask,
        flat_actor_mask,
        metric_prefix,
    ):
        valid_states = flat_states[flat_agent_mask]
        valid_returns = returns[flat_agent_mask]
        valid_actor_obs = flat_obs[flat_actor_mask]
        valid_actions = flat_actions[flat_actor_mask]
        valid_avail = flat_avail[flat_actor_mask]
        valid_advantages = advantages[flat_actor_mask]
        valid_old_log_probs = old_log_probs[flat_actor_mask]
        if valid_states.size(0) == 0:
            return

        batch_size = min(getattr(self.args, "batch_size", 64), valid_states.size(0))
        last_actor_loss = None
        last_critic_loss = None
        last_entropy = None
        for _ in range(self.args.ppo_epoch):
            if valid_actor_obs.size(0) > 0:
                actor_batch_size = min(batch_size, valid_actor_obs.size(0))
                permutation = torch.randperm(valid_actor_obs.size(0), device=self.device)
                for start in range(0, valid_actor_obs.size(0), actor_batch_size):
                    indices = permutation[start : start + actor_batch_size]
                    dist = self._masked_categorical(
                        actor(valid_actor_obs[indices]),
                        valid_avail[indices],
                    )
                    new_log_probs = dist.log_prob(valid_actions[indices])
                    ratio = torch.exp(new_log_probs - valid_old_log_probs[indices])
                    clipped_ratio = torch.clamp(
                        ratio,
                        1.0 - self.args.clip_param,
                        1.0 + self.args.clip_param,
                    )
                    surrogate_1 = ratio * valid_advantages[indices]
                    surrogate_2 = clipped_ratio * valid_advantages[indices]
                    entropy = dist.entropy()
                    actor_loss = -torch.min(surrogate_1, surrogate_2)
                    actor_loss -= self.args.entropy_coef * entropy
                    actor_loss = actor_loss.mean()
                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(actor.parameters(), self.args.grad_norm_clip)
                    actor_optimizer.step()
                    last_actor_loss = actor_loss.detach()
                    last_entropy = entropy.mean().detach()

            critic_permutation = torch.randperm(valid_states.size(0), device=self.device)
            for start in range(0, valid_states.size(0), batch_size):
                indices = critic_permutation[start : start + batch_size]
                critic_values = critic(valid_states[indices]).squeeze(-1)
                critic_loss = (critic_values - valid_returns[indices]).pow(2).mean()
                critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), self.args.grad_norm_clip)
                critic_optimizer.step()
                last_critic_loss = critic_loss.detach()

        if last_actor_loss is not None:
            self.last_train_metrics[f"{metric_prefix}_actor_loss"] = float(
                last_actor_loss.item()
            )
        if last_critic_loss is not None:
            self.last_train_metrics[f"{metric_prefix}_critic_loss"] = float(
                last_critic_loss.item()
            )
        if last_entropy is not None:
            self.last_train_metrics[f"{metric_prefix}_entropy"] = float(
                last_entropy.item()
            )

    def _learn_shared_continuous_high_level(self, batch, action_key):
        states = batch["high_s"]
        next_states = batch["high_s_next"]
        obs = batch["high_o"]
        actions = batch[action_key]
        terminated = batch["high_terminated"].squeeze(-1)
        mask = 1.0 - batch["high_padded"].squeeze(-1)
        active_mask = batch.get("high_agent_active_mask", None)
        if active_mask is None:
            active_mask = torch.ones(
                (*mask.shape, self.n_agents, 1),
                dtype=mask.dtype,
                device=self.device,
            )
        active_mask = active_mask.squeeze(-1)
        intervention_mask = batch.get("high_intervention_mask", None)
        if intervention_mask is None:
            intervention_mask = torch.zeros(
                (*mask.shape, self.n_agents, 1),
                dtype=mask.dtype,
                device=self.device,
            )
        intervention_mask = intervention_mask.squeeze(-1).clamp(0.0, 1.0)
        durations = batch.get("high_duration", None)
        if durations is None:
            durations = torch.ones((*mask.shape, 1), dtype=mask.dtype, device=self.device)
        durations = durations.squeeze(-1)
        rewards = batch["high_r"]
        if rewards.size(-1) == 1:
            rewards = rewards.expand(-1, -1, self.n_agents)

        episode_num = states.size(0)
        time_len = states.size(1)
        agent_mask = mask.unsqueeze(-1) * active_mask
        actor_mask = agent_mask * (1.0 - intervention_mask)
        flat_obs = obs.reshape(episode_num * time_len * self.n_agents, -1)
        flat_actor_obs = flat_obs
        flat_actions = actions.reshape(-1, self.high_n_actions)
        flat_states = self._expand_states_for_agents(states, self.high_state_shape)
        flat_next_states = self._expand_states_for_agents(next_states, self.high_state_shape)

        with torch.no_grad():
            values = self.high_critic(flat_states).reshape(
                episode_num,
                time_len,
                self.n_agents,
            )
            next_values = self.high_critic(flat_next_states).reshape(
                episode_num,
                time_len,
                self.n_agents,
            )
            advantages, returns = self._compute_multiagent_advantages(
                rewards,
                values,
                next_values,
                terminated,
                agent_mask,
                durations=durations,
            )
            old_dist = self._gaussian_dist(self.high_actor, flat_obs)
            old_log_probs = old_dist.log_prob(flat_actions).sum(dim=-1)

        self._ppo_update_gaussian(
            actor=self.high_actor,
            critic=self.high_critic,
            actor_optimizer=self.high_actor_optimizer,
            critic_optimizer=self.high_critic_optimizer,
            flat_obs=flat_obs,
            flat_actions=flat_actions,
            flat_states=flat_states,
            advantages=advantages.reshape(-1),
            returns=returns.reshape(-1),
            old_log_probs=old_log_probs,
            flat_agent_mask=agent_mask.reshape(-1) > 0,
            flat_actor_mask=actor_mask.reshape(-1) > 0,
            metric_prefix="official_hmappo_high",
        )

    def _ppo_update_gaussian(
        self,
        actor,
        critic,
        actor_optimizer,
        critic_optimizer,
        flat_obs,
        flat_actions,
        flat_states,
        advantages,
        returns,
        old_log_probs,
        flat_agent_mask,
        flat_actor_mask,
        metric_prefix,
    ):
        valid_states = flat_states[flat_agent_mask]
        valid_returns = returns[flat_agent_mask]
        valid_actor_obs = flat_obs[flat_actor_mask]
        valid_actions = flat_actions[flat_actor_mask]
        valid_advantages = advantages[flat_actor_mask]
        valid_old_log_probs = old_log_probs[flat_actor_mask]
        if valid_states.size(0) == 0:
            return
        batch_size = min(getattr(self.args, "batch_size", 64), valid_states.size(0))
        last_actor_loss = None
        last_critic_loss = None
        last_entropy = None
        for _ in range(self.args.ppo_epoch):
            if valid_actor_obs.size(0) > 0:
                actor_batch_size = min(batch_size, valid_actor_obs.size(0))
                permutation = torch.randperm(valid_actor_obs.size(0), device=self.device)
                for start in range(0, valid_actor_obs.size(0), actor_batch_size):
                    indices = permutation[start : start + actor_batch_size]
                    dist = self._gaussian_dist(actor, valid_actor_obs[indices])
                    new_log_probs = dist.log_prob(valid_actions[indices]).sum(dim=-1)
                    ratio = torch.exp(new_log_probs - valid_old_log_probs[indices])
                    clipped_ratio = torch.clamp(
                        ratio,
                        1.0 - self.args.clip_param,
                        1.0 + self.args.clip_param,
                    )
                    surrogate_1 = ratio * valid_advantages[indices]
                    surrogate_2 = clipped_ratio * valid_advantages[indices]
                    entropy = dist.entropy().sum(dim=-1)
                    actor_loss = -torch.min(surrogate_1, surrogate_2)
                    actor_loss -= self.args.entropy_coef * entropy
                    actor_loss = actor_loss.mean()
                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(actor.parameters(), self.args.grad_norm_clip)
                    actor_optimizer.step()
                    last_actor_loss = actor_loss.detach()
                    last_entropy = entropy.mean().detach()

            permutation = torch.randperm(valid_states.size(0), device=self.device)
            for start in range(0, valid_states.size(0), batch_size):
                indices = permutation[start : start + batch_size]
                critic_values = critic(valid_states[indices]).squeeze(-1)
                critic_loss = (critic_values - valid_returns[indices]).pow(2).mean()
                critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), self.args.grad_norm_clip)
                critic_optimizer.step()
                last_critic_loss = critic_loss.detach()

        if last_actor_loss is not None:
            self.last_train_metrics[f"{metric_prefix}_actor_loss"] = float(last_actor_loss.item())
        if last_critic_loss is not None:
            self.last_train_metrics[f"{metric_prefix}_critic_loss"] = float(last_critic_loss.item())
        if last_entropy is not None:
            self.last_train_metrics[f"{metric_prefix}_entropy"] = float(last_entropy.item())

    def _learn_shared_hybrid_high_level(self, batch, action_key):
        states = batch["high_s"]
        next_states = batch["high_s_next"]
        obs = batch["high_o"]
        actions = batch[action_key]
        avail_actions = batch.get("high_avail_u", None)
        terminated = batch["high_terminated"].squeeze(-1)
        mask = 1.0 - batch["high_padded"].squeeze(-1)
        active_mask = batch.get("high_agent_active_mask", None)
        if active_mask is None:
            active_mask = torch.ones(
                (*mask.shape, self.n_agents, 1),
                dtype=mask.dtype,
                device=self.device,
            )
        active_mask = active_mask.squeeze(-1)
        mode_train_mask = batch.get("high_mode_train_mask", None)
        if mode_train_mask is None:
            mode_train_mask = torch.ones(
                (*mask.shape, self.n_agents, 1),
                dtype=mask.dtype,
                device=self.device,
            )
        mode_train_mask = mode_train_mask.squeeze(-1)
        intervention_mask = batch.get("high_intervention_mask", None)
        if intervention_mask is None:
            intervention_mask = torch.zeros(
                (*mask.shape, self.n_agents, 1),
                dtype=mask.dtype,
                device=self.device,
            )
        intervention_mask = intervention_mask.squeeze(-1).clamp(0.0, 1.0)
        durations = batch.get("high_duration", None)
        if durations is None:
            durations = torch.ones((*mask.shape, 1), dtype=mask.dtype, device=self.device)
        durations = durations.squeeze(-1)
        rewards = batch["high_r"]
        if rewards.size(-1) == 1:
            rewards = rewards.expand(-1, -1, self.n_agents)

        episode_num = states.size(0)
        time_len = states.size(1)
        agent_mask = mask.unsqueeze(-1) * active_mask
        actor_mask = agent_mask * (1.0 - intervention_mask)
        flat_obs = obs.reshape(episode_num * time_len * self.n_agents, -1)
        flat_actor_obs = flat_obs
        flat_actions = actions.reshape(-1, self.high_n_actions)
        flat_modes = torch.clamp(
            torch.round(flat_actions[:, 0]).long(),
            0,
            self.high_mode_n_actions - 1,
        )
        flat_continuous = flat_actions[:, 1 : 1 + self.high_continuous_dim]
        if avail_actions is not None:
            flat_avail = avail_actions.reshape(-1, avail_actions.size(-1))
        else:
            flat_avail = torch.ones(
                (episode_num * time_len * self.n_agents, self.high_mode_n_actions),
                dtype=flat_obs.dtype,
                device=self.device,
            )
        flat_states = self._expand_states_for_agents(states, self.high_state_shape)
        flat_next_states = self._expand_states_for_agents(next_states, self.high_state_shape)

        with torch.no_grad():
            values = self.high_critic(flat_states).reshape(
                episode_num,
                time_len,
                self.n_agents,
            )
            next_values = self.high_critic(flat_next_states).reshape(
                episode_num,
                time_len,
                self.n_agents,
            )
            advantages, returns = self._compute_multiagent_advantages(
                rewards,
                values,
                next_values,
                terminated,
                agent_mask,
                durations=durations,
            )
            old_mode_dist, old_cont_dist = self._hybrid_high_dist(
                self.high_actor,
                flat_actor_obs,
                agent_idx=None,
                mode_avail=flat_avail,
            )
            old_mode_log_probs = old_mode_dist.log_prob(flat_modes)
            old_cont_log_probs = old_cont_dist.log_prob(flat_continuous).sum(dim=-1)
            flat_mode_mask = (
                actor_mask * mode_train_mask
            ).reshape(-1).clamp(0.0, 1.0)
            old_log_probs = old_cont_log_probs + old_mode_log_probs * flat_mode_mask

        self._ppo_update_hybrid(
            flat_actor_obs=flat_actor_obs,
            flat_modes=flat_modes,
            flat_continuous=flat_continuous,
            flat_avail=flat_avail,
            flat_states=flat_states,
            advantages=advantages.reshape(-1),
            returns=returns.reshape(-1),
            old_log_probs=old_log_probs,
            flat_mode_mask=flat_mode_mask,
            flat_agent_mask=agent_mask.reshape(-1) > 0,
            flat_actor_mask=actor_mask.reshape(-1) > 0,
        )

    def _ppo_update_hybrid(
        self,
        flat_actor_obs,
        flat_modes,
        flat_continuous,
        flat_avail,
        flat_states,
        advantages,
        returns,
        old_log_probs,
        flat_mode_mask,
        flat_agent_mask,
        flat_actor_mask,
    ):
        valid_states = flat_states[flat_agent_mask]
        valid_returns = returns[flat_agent_mask]
        valid_actor_obs = flat_actor_obs[flat_actor_mask]
        valid_modes = flat_modes[flat_actor_mask]
        valid_continuous = flat_continuous[flat_actor_mask]
        valid_avail = flat_avail[flat_actor_mask]
        valid_advantages = advantages[flat_actor_mask]
        valid_old_log_probs = old_log_probs[flat_actor_mask]
        valid_mode_mask = flat_mode_mask[flat_actor_mask]
        if valid_states.size(0) == 0:
            return
        batch_size = min(getattr(self.args, "batch_size", 64), valid_states.size(0))
        last_actor_loss = None
        last_critic_loss = None
        last_entropy = None
        for _ in range(self.args.ppo_epoch):
            if valid_actor_obs.size(0) > 0:
                actor_batch_size = min(batch_size, valid_actor_obs.size(0))
                permutation = torch.randperm(valid_actor_obs.size(0), device=self.device)
                for start in range(0, valid_actor_obs.size(0), actor_batch_size):
                    indices = permutation[start : start + actor_batch_size]
                    mode_dist, cont_dist, _, _ = self._hybrid_high_dist(
                        self.high_actor,
                        valid_actor_obs[indices],
                        agent_idx=None,
                        return_aux=True,
                        mode_avail=valid_avail[indices],
                    )
                    new_mode_log_probs = mode_dist.log_prob(valid_modes[indices])
                    new_cont_log_probs = cont_dist.log_prob(
                        valid_continuous[indices]
                    ).sum(dim=-1)
                    mode_mask = valid_mode_mask[indices]
                    new_log_probs = new_cont_log_probs + new_mode_log_probs * mode_mask
                    ratio = torch.exp(
                        torch.clamp(new_log_probs - valid_old_log_probs[indices], -20.0, 20.0)
                    )
                    clipped_ratio = torch.clamp(
                        ratio,
                        1.0 - self.args.clip_param,
                        1.0 + self.args.clip_param,
                    )
                    surrogate_1 = ratio * valid_advantages[indices]
                    surrogate_2 = clipped_ratio * valid_advantages[indices]
                    entropy = cont_dist.entropy().sum(dim=-1) + mode_dist.entropy() * mode_mask
                    actor_loss = -torch.min(surrogate_1, surrogate_2)
                    actor_loss -= self.args.entropy_coef * entropy
                    actor_loss = actor_loss.mean()
                    self.high_actor_optimizer.zero_grad()
                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.high_actor.parameters(), self.args.grad_norm_clip)
                    self.high_actor_optimizer.step()
                    last_actor_loss = actor_loss.detach()
                    last_entropy = entropy.mean().detach()

            permutation = torch.randperm(valid_states.size(0), device=self.device)
            for start in range(0, valid_states.size(0), batch_size):
                indices = permutation[start : start + batch_size]
                critic_values = self.high_critic(valid_states[indices]).squeeze(-1)
                critic_loss = (critic_values - valid_returns[indices]).pow(2).mean()
                self.high_critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.high_critic.parameters(), self.args.grad_norm_clip)
                self.high_critic_optimizer.step()
                last_critic_loss = critic_loss.detach()

        if last_actor_loss is not None:
            self.last_train_metrics["official_hmappo_high_actor_loss"] = float(last_actor_loss.item())
        if last_critic_loss is not None:
            self.last_train_metrics["official_hmappo_high_critic_loss"] = float(last_critic_loss.item())
        if last_entropy is not None:
            self.last_train_metrics["official_hmappo_high_entropy"] = float(last_entropy.item())

    def _shared_filename(self, level, role, prefix="", model_dir=None):
        model_dir = self.model_dir if model_dir is None else model_dir
        return os.path.join(model_dir, f"{prefix}{level}_shared_{role}_ppo")

    def _agent_filename(self, level, role, agent_idx, prefix="", model_dir=None):
        model_dir = self.model_dir if model_dir is None else model_dir
        return os.path.join(
            model_dir,
            f"{prefix}{level}_agent_{agent_idx}_{role}_discrete_ppo",
        )

    def _latest_shared_path(self, level, role, model_dir=None):
        direct = self._shared_filename(level, role, model_dir=model_dir)
        if os.path.exists(direct):
            return direct
        candidates = glob.glob(
            self._shared_filename(level, role, prefix="*_", model_dir=model_dir)
        )
        if not candidates:
            return None
        return max(candidates, key=self._checkpoint_sort_key)

    def _latest_agent_path(self, level, role, agent_idx, model_dir=None):
        direct = self._agent_filename(level, role, agent_idx, model_dir=model_dir)
        if os.path.exists(direct):
            return direct
        candidates = glob.glob(
            self._agent_filename(level, role, agent_idx, prefix="*_", model_dir=model_dir)
        )
        if not candidates:
            return None
        return max(candidates, key=self._checkpoint_sort_key)

    @staticmethod
    def _checkpoint_sort_key(path):
        prefix = os.path.basename(path).split("_", 1)[0]
        return int(prefix) if prefix.isdigit() else -1

    def _resolve_checkpoint_dir(self, checkpoint_dir):
        checkpoint_dir = os.path.normpath(str(checkpoint_dir))
        candidates = [
            checkpoint_dir,
            os.path.join(checkpoint_dir, self.args.alg, self.args.map),
            os.path.join(checkpoint_dir, "official_hmappo", self.args.map),
            os.path.join(checkpoint_dir, "hmappo", self.args.map),
        ]
        for candidate in candidates:
            if (
                self._latest_shared_path("low", "actor", model_dir=candidate) is not None
                or self._latest_agent_path("low", "actor", 0, model_dir=candidate) is not None
            ):
                return candidate
        raise Exception(f"No pretrained low-level model found under {checkpoint_dir}!")

    def _load_compatible(self, module, state, label):
        current = module.state_dict()
        compatible = {
            key: value
            for key, value in state.items()
            if key in current and current[key].shape == value.shape
        }
        skipped = sorted(set(state.keys()) - set(compatible.keys()))
        current.update(compatible)
        module.load_state_dict(current)
        if skipped:
            print(
                f"Skipped {len(skipped)} incompatible tensors while loading {label}: "
                + ", ".join(skipped[:6])
                + ("..." if len(skipped) > 6 else "")
            )

    def _average_agent_states(self, paths):
        states = [torch.load(path, map_location=self.device) for path in paths]
        averaged = {}
        for key in states[0].keys():
            tensors = [state[key] for state in states if key in state]
            if len(tensors) != len(states) or not torch.is_tensor(tensors[0]):
                averaged[key] = states[0][key]
                continue
            if not all(t.shape == tensors[0].shape for t in tensors):
                averaged[key] = tensors[0]
                continue
            if torch.is_floating_point(tensors[0]):
                averaged[key] = torch.stack([t.float() for t in tensors], dim=0).mean(dim=0).to(tensors[0].dtype)
            else:
                averaged[key] = tensors[0]
        return averaged

    def _load_level_shared(self, level, actor, critic, model_dir=None):
        actor_path = self._latest_shared_path(level, "actor", model_dir=model_dir)
        critic_path = self._latest_shared_path(level, "critic", model_dir=model_dir)
        if actor_path is not None and critic_path is not None:
            self._load_compatible(
                actor,
                torch.load(actor_path, map_location=self.device),
                f"{level} shared actor",
            )
            self._load_compatible(
                critic,
                torch.load(critic_path, map_location=self.device),
                f"{level} shared critic",
            )
            return

        actor_paths = [
            self._latest_agent_path(level, "actor", agent_idx, model_dir=model_dir)
            for agent_idx in range(self.n_agents)
        ]
        critic_paths = [
            self._latest_agent_path(level, "critic", agent_idx, model_dir=model_dir)
            for agent_idx in range(self.n_agents)
        ]
        if any(path is None for path in actor_paths) or any(path is None for path in critic_paths):
            raise Exception(f"No {level} shared/per-agent model found!")
        self._load_compatible(
            actor,
            self._average_agent_states(actor_paths),
            f"{level} averaged actor",
        )
        self._load_compatible(
            critic,
            self._average_agent_states(critic_paths),
            f"{level} averaged critic",
        )

    def _load_pretrained_low_models(self, checkpoint_dir):
        checkpoint_dir = self._resolve_checkpoint_dir(checkpoint_dir)
        self._load_level_shared(
            "low",
            self.low_actor,
            self.low_critic,
            model_dir=checkpoint_dir,
        )

    def _load_models(self):
        self._load_level_shared("low", self.low_actor, self.low_critic)
        if self.high_level_enabled:
            self._load_level_shared("high", self.high_actor, self.high_critic)

    def save_model(self, train_step):
        num = str(train_step // max(self.args.save_cycle, 1))
        os.makedirs(self.model_dir, exist_ok=True)
        for prefix in (f"{num}_", ""):
            torch.save(
                self.low_actor.state_dict(),
                self._shared_filename("low", "actor", prefix=prefix),
            )
            torch.save(
                self.low_critic.state_dict(),
                self._shared_filename("low", "critic", prefix=prefix),
            )
            if self.high_level_enabled:
                torch.save(
                    self.high_actor.state_dict(),
                    self._shared_filename("high", "actor", prefix=prefix),
                )
                torch.save(
                    self.high_critic.state_dict(),
                    self._shared_filename("high", "critic", prefix=prefix),
                )
