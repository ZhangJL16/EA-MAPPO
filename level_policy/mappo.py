import glob
import os
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

POLICY_SCOPE = "level_policy"


class DiscreteActorNetwork(nn.Module):
    def __init__(self, input_dims, n_actions, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dims, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.logits = nn.Linear(hidden_dim, n_actions)

    def forward(self, state):
        x = torch.tanh(self.fc1(state))
        x = torch.tanh(self.fc2(x))
        return self.logits(x)


class GaussianActorNetwork(nn.Module):
    def __init__(self, input_dims, action_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dims, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state):
        x = torch.tanh(self.fc1(state))
        x = torch.tanh(self.fc2(x))
        mean = torch.tanh(self.mean(x))
        log_std = torch.clamp(self.log_std, -5.0, 2.0)
        std = torch.exp(log_std).expand_as(mean)
        return mean, std


class HybridHighActorNetwork(nn.Module):
    def __init__(
        self,
        input_dims,
        n_modes,
        continuous_dim,
        hidden_dim=128,
        consequence_dim=5,
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_dims, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mode_logits = nn.Linear(hidden_dim, n_modes)
        self.mean = nn.Linear(hidden_dim, continuous_dim)
        self.log_std = nn.Parameter(torch.zeros(continuous_dim))
        self.energy_head = nn.Sequential(
            nn.Linear(hidden_dim + 1 + continuous_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.energy_consequence = nn.Linear(hidden_dim, consequence_dim)
        self.energy_feas_logit = nn.Linear(hidden_dim, 1)

    def encode(self, state):
        x = torch.tanh(self.fc1(state))
        return torch.tanh(self.fc2(x))

    def forward(self, state):
        h = self.encode(state)
        mode_logits = self.mode_logits(h)
        mean = torch.tanh(self.mean(h))
        log_std = torch.clamp(self.log_std, -5.0, 2.0)
        std = torch.exp(log_std).expand_as(mean)
        return mode_logits, mean, std, h

    def energy_predict(self, h, high_action):
        e = self.energy_head(torch.cat([h, high_action], dim=-1))
        consequence = self.energy_consequence(e)
        feas_logit = self.energy_feas_logit(e)
        feas_prob = torch.sigmoid(feas_logit)
        return feas_prob, consequence, feas_logit


class CriticNetwork(nn.Module):
    def __init__(self, input_dims, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dims, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        x = torch.tanh(self.fc1(state))
        x = torch.tanh(self.fc2(x))
        return self.v(x)


class ActionEnergyCriticNetwork(nn.Module):
    def __init__(self, obs_dims, action_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(obs_dims + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q = nn.Linear(hidden_dim, 1)

    def forward(self, observation, high_action):
        x = torch.cat([observation, high_action], dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return F.softplus(self.q(x)).squeeze(-1)


class MAPPO:
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
        self.high_energy_consequence_shape = int(
            getattr(args, "high_energy_consequence_shape", 5)
        )
        self.last_train_metrics = {}
        if self.high_n_actions <= 0 or self.high_obs_shape <= 0 or self.high_state_shape <= 0:
            raise ValueError("Level MAPPO requires high-level env_info fields.")

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

        self.low_actors = nn.ModuleList(
            [
                DiscreteActorNetwork(
                    self.low_obs_shape, self.low_n_actions, actor_hidden_dim
                )
                for _ in range(self.n_agents)
            ]
        ).to(self.device)
        self.low_critics = nn.ModuleList(
            [
                CriticNetwork(self.low_state_shape, critic_hidden_dim)
                for _ in range(self.n_agents)
            ]
        ).to(self.device)
        self.high_actors = nn.ModuleList(
            [
                (
                    HybridHighActorNetwork(
                        self.high_obs_shape,
                        self.high_mode_n_actions,
                        self.high_continuous_dim,
                        high_actor_hidden_dim,
                        self.high_energy_consequence_shape,
                    )
                    if self.use_hybrid_high_policy
                    else GaussianActorNetwork(
                        self.high_obs_shape,
                        self.high_n_actions,
                        high_actor_hidden_dim,
                    )
                )
                for _ in range(self.n_agents)
            ]
        ).to(self.device)
        self.high_critics = nn.ModuleList(
            [
                CriticNetwork(self.high_state_shape, high_critic_hidden_dim)
                for _ in range(self.n_agents)
            ]
        ).to(self.device)
        self.use_td_energy_critic = (
            bool(getattr(args, "hrl_td_energy_critic_enabled", False))
            and self.use_hybrid_high_policy
        )
        self.td_energy_ensemble = max(
            1, int(getattr(args, "hrl_td_energy_critic_ensemble", 3))
        )
        if self.use_td_energy_critic:
            self.high_energy_critics = nn.ModuleList(
                [
                    nn.ModuleList(
                        [
                            ActionEnergyCriticNetwork(
                                self.high_obs_shape,
                                self.high_n_actions,
                                high_critic_hidden_dim,
                            )
                            for _ in range(self.td_energy_ensemble)
                        ]
                    )
                    for _ in range(self.n_agents)
                ]
            ).to(self.device)
            self.high_energy_target_critics = copy.deepcopy(
                self.high_energy_critics
            ).to(self.device)
            for target_critics in self.high_energy_target_critics:
                for critic in target_critics:
                    critic.requires_grad_(False)
            td_energy_lr = float(
                getattr(args, "hrl_td_energy_critic_lr", high_lr_critic)
            )
            self.high_energy_critic_optimizers = [
                [
                    torch.optim.Adam(critic.parameters(), lr=td_energy_lr)
                    for critic in self.high_energy_critics[agent_idx]
                ]
                for agent_idx in range(self.n_agents)
            ]
            self.td_energy_lagrange = torch.zeros(
                self.n_agents, dtype=torch.float32, device=self.device
            )
        else:
            self.high_energy_critics = None
            self.high_energy_target_critics = None
            self.high_energy_critic_optimizers = None
            self.td_energy_lagrange = torch.zeros(
                self.n_agents, dtype=torch.float32, device=self.device
            )

        self.low_actor_optimizers = [
            torch.optim.Adam(actor.parameters(), lr=args.lr_actor)
            for actor in self.low_actors
        ]
        self.low_critic_optimizers = [
            torch.optim.Adam(critic.parameters(), lr=args.lr_critic)
            for critic in self.low_critics
        ]
        self.high_actor_optimizers = [
            torch.optim.Adam(actor.parameters(), lr=high_lr_actor)
            for actor in self.high_actors
        ]
        self.high_critic_optimizers = [
            torch.optim.Adam(critic.parameters(), lr=high_lr_critic)
            for critic in self.high_critics
        ]

        self.model_dir = args.model_dir + "/" + args.alg + "/" + args.map
        self.eval_hidden = None

        if self.args.load_model:
            self._load_models()
        pretrained_low_dir = getattr(self.args, "hmappo_pretrained_low_model_dir", "")
        if pretrained_low_dir:
            self._load_pretrained_low_models(pretrained_low_dir)

    def init_hidden(self, episode_num):
        self.eval_hidden = torch.zeros(
            (episode_num, self.n_agents, 1), device=self.device
        )

    def _masked_categorical(self, logits, avail_actions):
        masked_logits = logits.masked_fill(avail_actions <= 0, -1e10)
        return Categorical(logits=masked_logits)

    def _gaussian_dist(self, network, observation):
        mean, std = network(observation)
        return Normal(mean, std)

    def _ecf_candidate_mode_q(self, network, h, mean):
        q_modes = []
        for mode_id in range(self.high_mode_n_actions):
            mode = torch.full(
                (mean.size(0), 1),
                float(mode_id),
                dtype=mean.dtype,
                device=mean.device,
            )
            candidate_action = torch.cat([mode, mean], dim=-1)
            q_feas, _, _ = network.energy_predict(h, candidate_action)
            q_modes.append(q_feas)
        return torch.cat(q_modes, dim=-1)

    def _ecf_adjust_mode_logits(self, network, h, mean, mode_logits):
        if (
            not bool(getattr(self.args, "hrl_ecf_enabled", True))
            or float(getattr(self.args, "hrl_ecf_logit_bias_coef", 0.0)) <= 0.0
            or not hasattr(network, "energy_predict")
            or self.high_mode_n_actions <= 0
        ):
            return mode_logits

        q_modes = self._ecf_candidate_mode_q(network, h, mean).detach()
        alpha = float(getattr(self.args, "hrl_ecf_logit_bias_coef", 0.0))
        return mode_logits + alpha * torch.log(q_modes.clamp_min(1e-6))

    def _td_energy_predict(self, agent_idx, observation, high_action, target=False):
        if not self.use_td_energy_critic:
            zeros = torch.zeros(observation.size(0), device=observation.device)
            return zeros, zeros, zeros.unsqueeze(0)
        critics = (
            self.high_energy_target_critics[agent_idx]
            if target
            else self.high_energy_critics[agent_idx]
        )
        preds = torch.stack(
            [critic(observation, high_action) for critic in critics],
            dim=0,
        )
        mean = preds.mean(dim=0)
        std = preds.std(dim=0, unbiased=False) if preds.size(0) > 1 else torch.zeros_like(mean)
        return mean, std, preds

    def _td_energy_candidate_costs(self, agent_idx, observation, mean, target=False):
        safe_costs = []
        mu_costs = []
        std_costs = []
        kappa = float(getattr(self.args, "hrl_td_energy_kappa", 1.0))
        for mode_id in range(self.high_mode_n_actions):
            mode = torch.full(
                (mean.size(0), 1),
                float(mode_id),
                dtype=mean.dtype,
                device=mean.device,
            )
            candidate_action = torch.cat([mode, mean], dim=-1)
            mu, std, _ = self._td_energy_predict(
                agent_idx, observation, candidate_action, target=target
            )
            mu_costs.append(mu)
            std_costs.append(std)
            safe_costs.append(mu + kappa * std)
        return (
            torch.stack(safe_costs, dim=-1),
            torch.stack(mu_costs, dim=-1),
            torch.stack(std_costs, dim=-1),
        )

    def _td_energy_adjust_mode_logits(self, agent_idx, observation, mean, mode_logits):
        if (
            not self.use_td_energy_critic
            or agent_idx is None
            or self.high_mode_n_actions <= 0
        ):
            return mode_logits

        safe_costs, _, _ = self._td_energy_candidate_costs(
            agent_idx, observation, mean, target=False
        )
        adjusted = mode_logits
        bias_coef = float(getattr(self.args, "hrl_td_energy_logit_bias_coef", 0.0))
        if bias_coef > 0.0:
            adjusted = adjusted - bias_coef * safe_costs.detach()

        if bool(getattr(self.args, "hrl_td_energy_shield_enabled", False)):
            energy_idx = int(getattr(self.args, "hrl_td_energy_obs_energy_index", 4))
            if 0 <= energy_idx < observation.size(-1):
                battery = observation[:, energy_idx].clamp(0.0, 1.0)
            else:
                battery = torch.ones(observation.size(0), device=observation.device)
            budget = (
                battery - float(getattr(self.args, "hrl_td_energy_safe_ratio", 0.12))
            ).clamp_min(0.0)
            unsafe = safe_costs.detach() > budget.unsqueeze(-1)
            if self.high_mode_n_actions > 0:
                unsafe[:, 0] = False
            all_unsafe = unsafe.all(dim=-1)
            if torch.any(all_unsafe):
                unsafe[all_unsafe, 0] = False
            adjusted = adjusted.masked_fill(unsafe, -1e10)
        return adjusted

    def _soft_update_td_energy_targets(self, agent_idx):
        if not self.use_td_energy_critic:
            return
        tau = float(getattr(self.args, "hrl_td_energy_target_tau", 0.02))
        tau = max(0.0, min(1.0, tau))
        with torch.no_grad():
            for critic, target_critic in zip(
                self.high_energy_critics[agent_idx],
                self.high_energy_target_critics[agent_idx],
            ):
                for param, target_param in zip(
                    critic.parameters(), target_critic.parameters()
                ):
                    target_param.data.mul_(1.0 - tau).add_(tau * param.data)

    def _hybrid_high_dist(self, network, observation, agent_idx=None, return_aux=False):
        mode_logits, mean, std, h = network(observation)
        adjusted_logits = self._ecf_adjust_mode_logits(network, h, mean, mode_logits)
        adjusted_logits = self._td_energy_adjust_mode_logits(
            agent_idx, observation, mean, adjusted_logits
        )
        mode_dist = Categorical(logits=adjusted_logits)
        cont_dist = Normal(mean, std)
        if return_aux:
            return mode_dist, cont_dist, h, mode_logits
        return mode_dist, cont_dist

    @torch.no_grad()
    def _choose_from_network(self, network, observation, avail_actions, evaluate=False):
        obs = torch.tensor(
            observation, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        avail = torch.tensor(
            avail_actions, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        if torch.sum(avail) <= 0:
            return 0
        logits = network(obs)
        dist = self._masked_categorical(logits, avail)
        if evaluate:
            action = torch.argmax(dist.logits, dim=-1)
        else:
            action = dist.sample()
        return int(action.item())

    @torch.no_grad()
    def choose_action(self, observation, agent_idx, avail_actions, evaluate=False):
        return self._choose_from_network(
            self.low_actors[agent_idx], observation, avail_actions, evaluate
        )

    @torch.no_grad()
    def choose_high_level_action(
        self, observation, agent_idx, avail_actions=None, evaluate=False
    ):
        del avail_actions
        obs = torch.tensor(
            observation, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        if self.use_hybrid_high_policy:
            mode_dist, continuous_dist = self._hybrid_high_dist(
                self.high_actors[agent_idx], obs, agent_idx=agent_idx
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

        dist = self._gaussian_dist(self.high_actors[agent_idx], obs)
        if evaluate:
            action = dist.mean
        else:
            action = dist.sample()
        action = torch.clamp(action, -1.0, 1.0)
        return action.squeeze(0).detach().cpu().numpy().astype("float32")

    def _prepare_batch(self, batch):
        tensor_batch = {}
        for key, value in batch.items():
            if key == "u":
                tensor_batch[key] = torch.tensor(
                    value, dtype=torch.long, device=self.device
                )
            else:
                tensor_batch[key] = torch.tensor(
                    value, dtype=torch.float32, device=self.device
                )
        return tensor_batch

    def _compute_advantages(self, rewards, values, next_values, terminated, mask):
        advantages = torch.zeros_like(rewards)
        gae = torch.zeros(rewards.size(0), device=self.device)
        for t in reversed(range(rewards.size(1))):
            not_done = 1 - terminated[:, t]
            delta = rewards[:, t] + self.args.gamma * next_values[:, t] * not_done - values[:, t]
            gae = delta + self.args.gamma * self.args.gae_lambda * not_done * gae
            gae = gae * mask[:, t]
            advantages[:, t] = gae
        returns = advantages + values

        valid_advantages = advantages[mask > 0]
        if valid_advantages.numel() > 0:
            advantages = (advantages - valid_advantages.mean()) / (
                valid_advantages.std(unbiased=False) + 1e-8
            )
        return advantages, returns

    def learn(self, batch, max_episode_len, train_step, epsilon):
        del max_episode_len, train_step, epsilon
        self.last_train_metrics = {}
        batch = self._prepare_batch(batch)
        freeze_low = bool(getattr(self.args, "hmappo_freeze_low_level", False))
        freeze_high = bool(getattr(self.args, "hmappo_freeze_high_level", False))
        if not freeze_low:
            self._learn_level(
                batch=batch,
                actors=self.low_actors,
                critics=self.low_critics,
                actor_optimizers=self.low_actor_optimizers,
                critic_optimizers=self.low_critic_optimizers,
                obs_key="o",
                state_key="s",
                next_state_key="s_next",
                action_key="u",
                avail_key="avail_u",
                reward_key="r",
                padded_key="padded",
                terminated_key="terminated",
                active_key="agent_active_mask",
                action_dim=self.low_n_actions,
                obs_dim=self.low_obs_shape,
                state_dim=self.low_state_shape,
                guard_key="guard_applied",
            )
        if (
            not freeze_low
            and "hindsight_o" in batch
            and getattr(self.args, "hrl_hindsight_aux_coef", 0.0) > 0.0
        ):
            self._learn_hindsight_aux(batch)

        if "high_o" not in batch or freeze_high:
            return

        learn_high = (
            self._learn_hybrid_high_level
            if self.use_hybrid_high_policy
            else self._learn_continuous_level
        )
        learn_high(
            batch=batch,
            actors=self.high_actors,
            critics=self.high_critics,
            actor_optimizers=self.high_actor_optimizers,
            critic_optimizers=self.high_critic_optimizers,
            obs_key="high_o",
            next_obs_key="high_o_next",
            state_key="high_s",
            next_state_key="high_s_next",
            action_key="high_u",
            reward_key="high_r",
            padded_key="high_padded",
            terminated_key="high_terminated",
            active_key="high_agent_active_mask",
            action_dim=self.high_n_actions,
            obs_dim=self.high_obs_shape,
            state_dim=self.high_state_shape,
            energy_margin_key="high_energy_margin",
            energy_order_mask_key="high_energy_order_mask",
            energy_consequence_key="high_energy_consequence",
            mode_mask_key="high_mode_train_mask",
        )

    def _learn_hindsight_aux(self, batch):
        coef = float(getattr(self.args, "hrl_hindsight_aux_coef", 0.0))
        if coef <= 0.0:
            return

        obs = batch["hindsight_o"]
        actions = batch["u"].squeeze(-1)
        avail_actions = batch["avail_u"]
        mask = 1 - batch["padded"].squeeze(-1)
        active_mask = batch.get("agent_active_mask", None)
        if active_mask is None:
            active_mask = torch.ones(
                (*mask.shape, self.n_agents, 1),
                dtype=mask.dtype,
                device=self.device,
            )
        active_mask = active_mask.squeeze(-1)
        hindsight_mask = batch.get("hindsight_mask", None)
        if hindsight_mask is None:
            hindsight_mask = torch.ones(
                (*mask.shape, self.n_agents, 1),
                dtype=mask.dtype,
                device=self.device,
            )
        hindsight_mask = hindsight_mask.squeeze(-1)

        episode_num = obs.size(0)
        time_len = obs.size(1)
        for agent_idx in range(self.n_agents):
            actor_states = obs[:, :, agent_idx, :].reshape(
                episode_num * time_len, -1
            )
            agent_actions = actions[:, :, agent_idx].reshape(-1)
            agent_avail = avail_actions[:, :, agent_idx, :].reshape(
                -1, self.low_n_actions
            )
            agent_mask = mask * active_mask[:, :, agent_idx] * hindsight_mask[:, :, agent_idx]
            flat_mask = agent_mask.reshape(-1) > 0
            if torch.sum(flat_mask) <= 0:
                continue

            valid_states = actor_states[flat_mask]
            valid_actions = agent_actions[flat_mask]
            valid_avail = agent_avail[flat_mask]
            batch_size = min(getattr(self.args, "batch_size", 64), valid_states.size(0))
            permutation = torch.randperm(valid_states.size(0), device=self.device)
            for start in range(0, valid_states.size(0), batch_size):
                indices = permutation[start : start + batch_size]
                dist = self._masked_categorical(
                    self.low_actors[agent_idx](valid_states[indices]),
                    valid_avail[indices],
                )
                aux_loss = -dist.log_prob(valid_actions[indices]).mean() * coef
                self.low_actor_optimizers[agent_idx].zero_grad()
                aux_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.low_actors[agent_idx].parameters(),
                    self.args.grad_norm_clip,
                )
                self.low_actor_optimizers[agent_idx].step()

    def _learn_hybrid_high_level(
        self,
        batch,
        actors,
        critics,
        actor_optimizers,
        critic_optimizers,
        obs_key,
        next_obs_key,
        state_key,
        next_state_key,
        action_key,
        reward_key,
        padded_key,
        terminated_key,
        active_key,
        action_dim,
        obs_dim,
        state_dim,
        energy_margin_key=None,
        energy_order_mask_key=None,
        energy_consequence_key=None,
        mode_mask_key=None,
    ):
        del action_dim
        states = batch[state_key]
        next_states = batch[next_state_key]
        obs = batch[obs_key]
        next_obs = batch.get(next_obs_key, None)
        actions = batch[action_key]
        terminated = batch[terminated_key].squeeze(-1)
        mask = 1 - batch[padded_key].squeeze(-1)
        active_mask = batch.get(active_key, None)
        if active_mask is None:
            active_mask = torch.ones(
                (*mask.shape, self.n_agents, 1),
                dtype=mask.dtype,
                device=self.device,
            )
        active_mask = active_mask.squeeze(-1)
        mode_train_mask = (
            batch.get(mode_mask_key, None) if mode_mask_key is not None else None
        )
        if mode_train_mask is None:
            mode_train_mask = torch.ones(
                (*mask.shape, self.n_agents, 1),
                dtype=mask.dtype,
                device=self.device,
            )
        mode_train_mask = mode_train_mask.squeeze(-1)

        rewards = batch[reward_key]
        if rewards.size(-1) == 1:
            rewards = rewards.expand(-1, -1, self.n_agents)
        energy_margins = (
            batch.get(energy_margin_key, None) if energy_margin_key is not None else None
        )
        energy_order_masks = (
            batch.get(energy_order_mask_key, None)
            if energy_order_mask_key is not None
            else None
        )
        energy_loss_coef = float(
            getattr(self.args, "hrl_energy_margin_loss_coef", 0.0)
        )
        use_energy_loss = energy_loss_coef > 0.0 and energy_margins is not None
        energy_consequences = (
            batch.get(energy_consequence_key, None)
            if energy_consequence_key is not None
            else None
        )
        ecf_enabled = (
            bool(getattr(self.args, "hrl_ecf_enabled", True))
            and energy_consequences is not None
        )
        ecf_consequence_coef = float(
            getattr(self.args, "hrl_ecf_consequence_loss_coef", 0.0)
        )
        ecf_feas_coef = float(getattr(self.args, "hrl_ecf_feas_loss_coef", 0.0))
        ecf_policy_coef = float(getattr(self.args, "hrl_ecf_policy_loss_coef", 0.0))
        ecf_charge_need_coef = float(
            getattr(self.args, "hrl_ecf_charge_need_loss_coef", 0.0)
        )
        ecf_charge_need_margin = float(
            getattr(self.args, "hrl_ecf_charge_need_margin", 0.05)
        )
        use_energy_aux = (
            (use_energy_loss or ecf_charge_need_coef > 0.0)
            and energy_margins is not None
        )
        use_ecf_loss = ecf_enabled and (
            ecf_consequence_coef > 0.0
            or ecf_feas_coef > 0.0
            or ecf_policy_coef > 0.0
            or ecf_charge_need_coef > 0.0
        )
        td_energy_enabled = (
            self.use_td_energy_critic and energy_consequences is not None
        )
        td_energy_gamma = float(getattr(self.args, "hrl_td_energy_gamma", 0.95))
        td_energy_discount = td_energy_gamma ** max(
            1, int(getattr(self.args, "hmappo_meta_period", 5))
        )
        td_energy_lagrange_enabled = bool(
            getattr(self.args, "hrl_td_energy_lagrange_enabled", False)
        )
        td_energy_actor_coef = float(
            getattr(self.args, "hrl_td_energy_lagrange_actor_coef", 1.0)
        )
        ecf_metric_sums = {
            "ecf_con_loss": 0.0,
            "ecf_feas_loss": 0.0,
            "ecf_policy_loss": 0.0,
            "ecf_charge_need_loss": 0.0,
            "ecf_q_charge": 0.0,
            "ecf_q_order": 0.0,
            "ecf_feas_target": 0.0,
            "ecf_charge_need_target": 0.0,
            "td_energy_loss": 0.0,
            "td_energy_cost": 0.0,
            "td_energy_mu_charge": 0.0,
            "td_energy_mu_order": 0.0,
            "td_energy_std_order": 0.0,
            "td_energy_lambda": 0.0,
        }
        ecf_metric_count = 0
        td_energy_metric_count = 0

        episode_num = states.size(0)
        time_len = states.size(1)
        flat_states = states.reshape(episode_num * time_len, -1)
        if next_obs is None:
            next_obs = obs

        for agent_idx in range(self.n_agents):
            actor_states = obs[:, :, agent_idx, :].reshape(
                episode_num * time_len, obs_dim
            )
            next_actor_states = next_obs[:, :, agent_idx, :].reshape(
                episode_num * time_len, obs_dim
            )
            actor_states = torch.nan_to_num(actor_states, nan=0.0, posinf=1.0, neginf=-1.0)
            next_actor_states = torch.nan_to_num(
                next_actor_states, nan=0.0, posinf=1.0, neginf=-1.0
            )
            agent_actions = actions[:, :, agent_idx, :].reshape(-1, self.high_n_actions)
            agent_actions = torch.nan_to_num(
                agent_actions, nan=0.0, posinf=1.0, neginf=-1.0
            )
            mode_actions = torch.clamp(
                torch.round(agent_actions[:, 0]).long(),
                0,
                self.high_mode_n_actions - 1,
            )
            continuous_actions = agent_actions[:, 1 : 1 + self.high_continuous_dim]
            agent_rewards = rewards[:, :, agent_idx]
            agent_mask = mask * active_mask[:, :, agent_idx]
            agent_mode_mask = agent_mask * mode_train_mask[:, :, agent_idx]
            flat_agent_mask = agent_mask.reshape(-1) > 0
            flat_mode_mask = agent_mode_mask.reshape(-1).clamp(0.0, 1.0)

            with torch.no_grad():
                values = critics[agent_idx](states.reshape(-1, state_dim)).reshape(
                    episode_num, time_len
                )
                next_values = critics[agent_idx](
                    next_states.reshape(-1, state_dim)
                ).reshape(episode_num, time_len)
                advantages, returns = self._compute_advantages(
                    agent_rewards, values, next_values, terminated, agent_mask
                )
                old_mode_dist, old_cont_dist = self._hybrid_high_dist(
                    actors[agent_idx], actor_states, agent_idx=agent_idx
                )
                old_mode_log_probs = old_mode_dist.log_prob(mode_actions)
                old_cont_log_probs = old_cont_dist.log_prob(
                    continuous_actions
                ).sum(dim=-1)
                old_log_probs = old_cont_log_probs + old_mode_log_probs * flat_mode_mask

            valid_states = flat_states[flat_agent_mask]
            valid_returns = returns.reshape(-1)[flat_agent_mask]
            valid_actor_states_pg = actor_states[flat_agent_mask]
            valid_next_actor_states_pg = next_actor_states[flat_agent_mask]
            valid_modes_pg = mode_actions[flat_agent_mask]
            valid_continuous_pg = continuous_actions[flat_agent_mask]
            valid_advantages_pg = advantages.reshape(-1)[flat_agent_mask]
            valid_old_log_probs_pg = old_log_probs[flat_agent_mask]
            valid_mode_mask_pg = flat_mode_mask[flat_agent_mask]
            valid_terminated_pg = terminated.reshape(-1)[flat_agent_mask].float()
            if use_energy_aux:
                agent_energy_margins = energy_margins[:, :, agent_idx, 0].reshape(-1)
                valid_energy_margins_pg = agent_energy_margins[flat_agent_mask]
                if energy_order_masks is not None:
                    agent_energy_order_masks = energy_order_masks[
                        :, :, agent_idx, 0
                    ].reshape(-1)
                    valid_energy_order_masks_pg = agent_energy_order_masks[
                        flat_agent_mask
                    ]
                else:
                    valid_energy_order_masks_pg = torch.ones_like(
                        valid_energy_margins_pg
                    )
            else:
                valid_energy_margins_pg = None
                valid_energy_order_masks_pg = None
            if use_ecf_loss or td_energy_enabled:
                consequence_dim = energy_consequences.size(-1)
                agent_energy_consequences = energy_consequences[
                    :, :, agent_idx, :
                ].reshape(-1, consequence_dim)
                valid_energy_consequences_pg = agent_energy_consequences[
                    flat_agent_mask
                ]
            else:
                valid_energy_consequences_pg = None

            if valid_states.size(0) == 0:
                continue

            batch_size = min(getattr(self.args, "batch_size", 64), valid_states.size(0))
            valid_td_energy_cost_pg = None
            if (
                td_energy_enabled
                and valid_energy_consequences_pg is not None
                and valid_energy_consequences_pg.size(-1) >= 2
            ):
                valid_td_energy_cost_pg = torch.relu(
                    -valid_energy_consequences_pg[:, 1]
                ).detach()
                if td_energy_lagrange_enabled and valid_td_energy_cost_pg.numel() > 0:
                    with torch.no_grad():
                        budget = float(getattr(self.args, "hrl_td_energy_budget", 0.025))
                        dual_lr = float(
                            getattr(self.args, "hrl_td_energy_lagrange_lr", 0.01)
                        )
                        self.td_energy_lagrange[agent_idx] = torch.clamp(
                            self.td_energy_lagrange[agent_idx]
                            + dual_lr * (valid_td_energy_cost_pg.mean() - budget),
                            min=0.0,
                            max=100.0,
                        )

            for _ in range(self.args.ppo_epoch):
                if td_energy_enabled and valid_td_energy_cost_pg is not None:
                    energy_permutation = torch.randperm(
                        valid_actor_states_pg.size(0), device=self.device
                    )
                    for start in range(0, valid_actor_states_pg.size(0), batch_size):
                        indices = energy_permutation[start : start + batch_size]
                        obs_batch = valid_actor_states_pg[indices]
                        action_batch = torch.cat(
                            [
                                valid_modes_pg[indices].float().unsqueeze(-1),
                                valid_continuous_pg[indices],
                            ],
                            dim=-1,
                        )
                        cost_batch = valid_td_energy_cost_pg[indices]
                        done_batch = valid_terminated_pg[indices]
                        next_obs_batch = valid_next_actor_states_pg[indices]
                        with torch.no_grad():
                            next_mode_logits, next_mean, _, _ = actors[agent_idx](
                                next_obs_batch
                            )
                            next_mode_probs = torch.softmax(next_mode_logits, dim=-1)
                            _, next_mu_costs, _ = self._td_energy_candidate_costs(
                                agent_idx,
                                next_obs_batch,
                                next_mean,
                                target=True,
                            )
                            expected_next_cost = (
                                next_mode_probs * next_mu_costs
                            ).sum(dim=-1)
                            td_target = cost_batch + (
                                1.0 - done_batch
                            ) * td_energy_discount * expected_next_cost

                        for critic_idx, energy_critic in enumerate(
                            self.high_energy_critics[agent_idx]
                        ):
                            pred = energy_critic(obs_batch, action_batch)
                            td_loss = F.mse_loss(pred, td_target)
                            if not torch.isfinite(td_loss):
                                continue
                            opt = self.high_energy_critic_optimizers[agent_idx][
                                critic_idx
                            ]
                            opt.zero_grad()
                            td_loss.backward()
                            torch.nn.utils.clip_grad_norm_(
                                energy_critic.parameters(), self.args.grad_norm_clip
                            )
                            opt.step()
                            with torch.no_grad():
                                ecf_metric_sums["td_energy_loss"] += float(
                                    td_loss.detach().item()
                                )
                                ecf_metric_sums["td_energy_cost"] += float(
                                    cost_batch.mean().item()
                                )
                                td_energy_metric_count += 1

                if valid_actor_states_pg.size(0) > 0:
                    actor_batch_size = min(batch_size, valid_actor_states_pg.size(0))
                    actor_permutation = torch.randperm(
                        valid_actor_states_pg.size(0), device=self.device
                    )
                    for start in range(0, valid_actor_states_pg.size(0), actor_batch_size):
                        indices = actor_permutation[start : start + actor_batch_size]
                        mode_dist, cont_dist, h, _ = self._hybrid_high_dist(
                            actors[agent_idx],
                            valid_actor_states_pg[indices],
                            agent_idx=agent_idx,
                            return_aux=True,
                        )
                        new_mode_log_probs = mode_dist.log_prob(valid_modes_pg[indices])
                        new_cont_log_probs = cont_dist.log_prob(
                            valid_continuous_pg[indices]
                        ).sum(dim=-1)
                        mode_mask = valid_mode_mask_pg[indices]
                        new_log_probs = new_cont_log_probs + new_mode_log_probs * mode_mask
                        log_ratio = torch.clamp(
                            new_log_probs - valid_old_log_probs_pg[indices],
                            -20.0,
                            20.0,
                        )
                        ratio = torch.exp(log_ratio)
                        clipped_ratio = torch.clamp(
                            ratio,
                            1.0 - self.args.clip_param,
                            1.0 + self.args.clip_param,
                        )
                        surrogate_1 = ratio * valid_advantages_pg[indices]
                        surrogate_2 = clipped_ratio * valid_advantages_pg[indices]
                        entropy = (
                            cont_dist.entropy().sum(dim=-1)
                            + mode_dist.entropy() * mode_mask
                        )
                        actor_loss = -torch.min(surrogate_1, surrogate_2)
                        actor_loss -= self.args.entropy_coef * entropy
                        actor_loss = actor_loss.mean()

                        if valid_energy_margins_pg is not None:
                            margin = valid_energy_margins_pg[indices]
                            order_mask = valid_energy_order_masks_pg[indices]
                            p_charge = mode_dist.probs[:, 0].clamp(1e-6, 1.0 - 1e-6)
                            if use_energy_loss:
                                beta = float(
                                    getattr(
                                        self.args,
                                        "hrl_energy_margin_charge_beta",
                                        0.5,
                                    )
                                )
                                p_order = 1.0 - p_charge
                                unsafe_order_weight = torch.relu(-margin)
                                feasible_order_weight = torch.relu(margin)
                                energy_terms = (
                                    unsafe_order_weight * p_order
                                    + beta * feasible_order_weight * p_charge
                                )
                                energy_terms = energy_terms * order_mask * mode_mask
                                denom = (order_mask * mode_mask).sum().clamp_min(1.0)
                                energy_loss = energy_terms.sum() / denom
                                actor_loss = actor_loss + energy_loss_coef * energy_loss

                            if ecf_charge_need_coef > 0.0:
                                charge_need_target = (
                                    margin < ecf_charge_need_margin
                                ).float()
                                charge_need_loss_terms = F.binary_cross_entropy(
                                    p_charge,
                                    charge_need_target,
                                    reduction="none",
                                )
                                charge_need_weight = order_mask.clamp(0.0, 1.0)
                                charge_need_denom = charge_need_weight.sum().clamp_min(1.0)
                                loss_charge_need = (
                                    charge_need_loss_terms * charge_need_weight
                                ).sum() / charge_need_denom
                                actor_loss = (
                                    actor_loss
                                    + ecf_charge_need_coef * loss_charge_need
                                )
                                with torch.no_grad():
                                    ecf_metric_sums[
                                        "ecf_charge_need_loss"
                                    ] += float(loss_charge_need.detach().item())
                                    if charge_need_weight.sum() > 0:
                                        ecf_metric_sums[
                                            "ecf_charge_need_target"
                                        ] += float(
                                            (
                                                charge_need_target
                                                * charge_need_weight
                                            ).sum().item()
                                            / charge_need_weight.sum().item()
                                        )

                        if use_ecf_loss and valid_energy_consequences_pg is not None:
                            mode_feature = valid_modes_pg[indices].float().unsqueeze(-1)
                            high_action = torch.cat(
                                [mode_feature, valid_continuous_pg[indices]],
                                dim=-1,
                            )
                            q_feas, c_pred, feas_logit = actors[
                                agent_idx
                            ].energy_predict(h, high_action)
                            c_target = valid_energy_consequences_pg[indices]
                            if c_target.size(-1) >= 5:
                                loss_con = F.mse_loss(
                                    c_pred[:, :4],
                                    c_target[:, :4],
                                )
                                y_target = c_target[:, 4:5].clamp(0.0, 1.0)
                            else:
                                loss_con = F.mse_loss(c_pred, c_target)
                                y_target = (
                                    c_target[:, :1] > 0.0
                                ).float()
                            loss_feas = F.binary_cross_entropy_with_logits(
                                feas_logit,
                                y_target,
                            )
                            q_modes_detached = None
                            loss_ecf_policy = torch.tensor(
                                0.0, dtype=actor_loss.dtype, device=actor_loss.device
                            )
                            if ecf_policy_coef > 0.0:
                                q_modes_detached = self._ecf_candidate_mode_q(
                                    actors[agent_idx],
                                    h,
                                    cont_dist.mean,
                                ).detach()
                                mode_probs = mode_dist.probs
                                mode_policy_terms = -(
                                    mode_probs
                                    * torch.log(q_modes_detached.clamp_min(1e-6))
                                ).sum(dim=-1)
                                loss_ecf_policy = mode_policy_terms.mean()
                            actor_loss = (
                                actor_loss
                                + ecf_consequence_coef * loss_con
                                + ecf_feas_coef * loss_feas
                                + ecf_policy_coef * loss_ecf_policy
                            )
                            with torch.no_grad():
                                if q_modes_detached is None:
                                    q_modes_detached = self._ecf_candidate_mode_q(
                                        actors[agent_idx],
                                        h,
                                        cont_dist.mean,
                                    ).detach()
                                ecf_metric_sums["ecf_con_loss"] += float(loss_con.detach().item())
                                ecf_metric_sums["ecf_feas_loss"] += float(loss_feas.detach().item())
                                ecf_metric_sums["ecf_policy_loss"] += float(
                                    loss_ecf_policy.detach().item()
                                )
                                ecf_metric_sums["ecf_q_charge"] += float(
                                    q_modes_detached[:, 0].mean().item()
                                )
                                if q_modes_detached.size(-1) > 1:
                                    ecf_metric_sums["ecf_q_order"] += float(
                                        q_modes_detached[:, 1].mean().item()
                                    )
                                ecf_metric_sums["ecf_feas_target"] += float(
                                    y_target.mean().item()
                                )
                                ecf_metric_count += 1

                        if td_energy_enabled and td_energy_lagrange_enabled:
                            with torch.no_grad():
                                safe_costs, mu_costs, std_costs = (
                                    self._td_energy_candidate_costs(
                                        agent_idx,
                                        valid_actor_states_pg[indices],
                                        cont_dist.mean,
                                        target=False,
                                    )
                                )
                            expected_safe_cost = (
                                mode_dist.probs * safe_costs
                            ).sum(dim=-1)
                            lagrange = self.td_energy_lagrange[agent_idx].detach()
                            actor_loss = actor_loss + (
                                td_energy_actor_coef
                                * lagrange
                                * expected_safe_cost.mean()
                            )
                            with torch.no_grad():
                                ecf_metric_sums["td_energy_mu_charge"] += float(
                                    mu_costs[:, 0].mean().item()
                                )
                                if mu_costs.size(-1) > 1:
                                    ecf_metric_sums["td_energy_mu_order"] += float(
                                        mu_costs[:, 1].mean().item()
                                    )
                                    ecf_metric_sums["td_energy_std_order"] += float(
                                        std_costs[:, 1].mean().item()
                                    )
                                ecf_metric_sums["td_energy_lambda"] += float(
                                    lagrange.item()
                                )
                                td_energy_metric_count += 1

                        if not torch.isfinite(actor_loss):
                            continue
                        actor_optimizers[agent_idx].zero_grad()
                        actor_loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            actors[agent_idx].parameters(), self.args.grad_norm_clip
                        )
                        has_bad_grad = any(
                            param.grad is not None
                            and not torch.isfinite(param.grad).all()
                            for param in actors[agent_idx].parameters()
                        )
                        if has_bad_grad:
                            actor_optimizers[agent_idx].zero_grad()
                            continue
                        actor_optimizers[agent_idx].step()

                critic_permutation = torch.randperm(valid_states.size(0), device=self.device)
                for start in range(0, valid_states.size(0), batch_size):
                    indices = critic_permutation[start : start + batch_size]
                    critic_values = critics[agent_idx](valid_states[indices]).squeeze(-1)
                    critic_loss = (critic_values - valid_returns[indices]).pow(2).mean()
                    if not torch.isfinite(critic_loss):
                        continue
                    critic_optimizers[agent_idx].zero_grad()
                    critic_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        critics[agent_idx].parameters(), self.args.grad_norm_clip
                    )
                    has_bad_grad = any(
                        param.grad is not None
                        and not torch.isfinite(param.grad).all()
                        for param in critics[agent_idx].parameters()
                    )
                    if has_bad_grad:
                        critic_optimizers[agent_idx].zero_grad()
                        continue
                    critic_optimizers[agent_idx].step()

            self._soft_update_td_energy_targets(agent_idx)

        metric_updates = {}
        if ecf_metric_count > 0:
            for key in (
                "ecf_con_loss",
                "ecf_feas_loss",
                "ecf_policy_loss",
                "ecf_charge_need_loss",
                "ecf_q_charge",
                "ecf_q_order",
                "ecf_feas_target",
                "ecf_charge_need_target",
            ):
                metric_updates[key] = ecf_metric_sums[key] / float(ecf_metric_count)
        if td_energy_metric_count > 0:
            for key in (
                "td_energy_loss",
                "td_energy_cost",
                "td_energy_mu_charge",
                "td_energy_mu_order",
                "td_energy_std_order",
                "td_energy_lambda",
            ):
                metric_updates[key] = ecf_metric_sums[key] / float(
                    td_energy_metric_count
                )
        if metric_updates:
            self.last_train_metrics.update(metric_updates)

    def _learn_continuous_level(
        self,
        batch,
        actors,
        critics,
        actor_optimizers,
        critic_optimizers,
        obs_key,
        state_key,
        next_state_key,
        action_key,
        reward_key,
        padded_key,
        terminated_key,
        active_key,
        action_dim,
        obs_dim,
        state_dim,
        energy_margin_key=None,
        energy_order_mask_key=None,
        mode_mask_key=None,
    ):
        del mode_mask_key
        states = batch[state_key]
        next_states = batch[next_state_key]
        obs = batch[obs_key]
        actions = batch[action_key]
        terminated = batch[terminated_key].squeeze(-1)
        mask = 1 - batch[padded_key].squeeze(-1)
        active_mask = batch.get(active_key, None)
        if active_mask is None:
            active_mask = torch.ones(
                (*mask.shape, self.n_agents, 1),
                dtype=mask.dtype,
                device=self.device,
            )
        active_mask = active_mask.squeeze(-1)

        rewards = batch[reward_key]
        if rewards.size(-1) == 1:
            rewards = rewards.expand(-1, -1, self.n_agents)
        energy_margins = (
            batch.get(energy_margin_key, None) if energy_margin_key is not None else None
        )
        energy_order_masks = (
            batch.get(energy_order_mask_key, None)
            if energy_order_mask_key is not None
            else None
        )
        energy_loss_coef = float(
            getattr(self.args, "hrl_energy_margin_loss_coef", 0.0)
        )
        use_energy_loss = (
            energy_loss_coef > 0.0
            and action_dim in (1, 2)
            and energy_margins is not None
        )

        episode_num = states.size(0)
        time_len = states.size(1)
        flat_states = states.reshape(episode_num * time_len, -1)

        for agent_idx in range(self.n_agents):
            actor_states = obs[:, :, agent_idx, :].reshape(
                episode_num * time_len, obs_dim
            )
            agent_actions = actions[:, :, agent_idx, :].reshape(-1, action_dim)
            agent_rewards = rewards[:, :, agent_idx]
            agent_mask = mask * active_mask[:, :, agent_idx]
            flat_agent_mask = agent_mask.reshape(-1) > 0

            with torch.no_grad():
                values = critics[agent_idx](states.reshape(-1, state_dim)).reshape(
                    episode_num, time_len
                )
                next_values = critics[agent_idx](
                    next_states.reshape(-1, state_dim)
                ).reshape(episode_num, time_len)
                advantages, returns = self._compute_advantages(
                    agent_rewards, values, next_values, terminated, agent_mask
                )
                old_dist = self._gaussian_dist(actors[agent_idx], actor_states)
                old_log_probs = old_dist.log_prob(agent_actions).sum(dim=-1)

            valid_states = flat_states[flat_agent_mask]
            valid_returns = returns.reshape(-1)[flat_agent_mask]
            valid_actor_states_pg = actor_states[flat_agent_mask]
            valid_actions_pg = agent_actions[flat_agent_mask]
            valid_advantages_pg = advantages.reshape(-1)[flat_agent_mask]
            valid_old_log_probs_pg = old_log_probs[flat_agent_mask]
            if use_energy_loss:
                agent_energy_margins = energy_margins[:, :, agent_idx, 0].reshape(-1)
                valid_energy_margins_pg = agent_energy_margins[flat_agent_mask]
                if energy_order_masks is not None:
                    agent_energy_order_masks = energy_order_masks[
                        :, :, agent_idx, 0
                    ].reshape(-1)
                    valid_energy_order_masks_pg = agent_energy_order_masks[
                        flat_agent_mask
                    ]
                else:
                    valid_energy_order_masks_pg = torch.ones_like(
                        valid_energy_margins_pg
                    )
            else:
                valid_energy_margins_pg = None
                valid_energy_order_masks_pg = None

            if valid_states.size(0) == 0:
                continue

            batch_size = min(getattr(self.args, "batch_size", 64), valid_states.size(0))

            for _ in range(self.args.ppo_epoch):
                if valid_actor_states_pg.size(0) > 0:
                    actor_batch_size = min(batch_size, valid_actor_states_pg.size(0))
                    actor_permutation = torch.randperm(
                        valid_actor_states_pg.size(0), device=self.device
                    )
                    for start in range(0, valid_actor_states_pg.size(0), actor_batch_size):
                        indices = actor_permutation[start : start + actor_batch_size]
                        dist = self._gaussian_dist(
                            actors[agent_idx], valid_actor_states_pg[indices]
                        )
                        new_log_probs = dist.log_prob(valid_actions_pg[indices]).sum(dim=-1)
                        ratio = torch.exp(new_log_probs - valid_old_log_probs_pg[indices])
                        clipped_ratio = torch.clamp(
                            ratio,
                            1.0 - self.args.clip_param,
                            1.0 + self.args.clip_param,
                        )
                        surrogate_1 = ratio * valid_advantages_pg[indices]
                        surrogate_2 = clipped_ratio * valid_advantages_pg[indices]
                        entropy = dist.entropy().sum(dim=-1)
                        actor_loss = -torch.min(surrogate_1, surrogate_2)
                        actor_loss -= self.args.entropy_coef * entropy
                        actor_loss = actor_loss.mean()
                        if valid_energy_margins_pg is not None:
                            margin = valid_energy_margins_pg[indices]
                            order_mask = valid_energy_order_masks_pg[indices]
                            beta = float(
                                getattr(
                                    self.args,
                                    "hrl_energy_margin_charge_beta",
                                    0.5,
                                )
                            )
                            charge_fraction = float(
                                getattr(self.args, "hrl_charge_mode_fraction", 0.5)
                            )
                            charge_fraction = max(0.01, min(0.99, charge_fraction))
                            charge_threshold = -1.0 + 2.0 * charge_fraction
                            if action_dim >= 2:
                                mode_dist = Normal(dist.loc[:, 0], dist.scale[:, 0])
                                p_charge = mode_dist.cdf(
                                    torch.full_like(mode_dist.loc, charge_threshold)
                                )
                            else:
                                p_charge = dist.cdf(
                                    torch.full_like(dist.loc, charge_threshold)
                                ).squeeze(-1)
                            p_order = 1.0 - p_charge
                            unsafe_order_weight = torch.relu(-margin)
                            feasible_order_weight = torch.relu(margin)
                            energy_terms = (
                                unsafe_order_weight * p_order
                                + beta * feasible_order_weight * p_charge
                            )
                            energy_terms = energy_terms * order_mask
                            denom = order_mask.sum().clamp_min(1.0)
                            energy_loss = energy_terms.sum() / denom
                            actor_loss = actor_loss + energy_loss_coef * energy_loss

                        actor_optimizers[agent_idx].zero_grad()
                        actor_loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            actors[agent_idx].parameters(), self.args.grad_norm_clip
                        )
                        actor_optimizers[agent_idx].step()

                critic_permutation = torch.randperm(valid_states.size(0), device=self.device)
                for start in range(0, valid_states.size(0), batch_size):
                    indices = critic_permutation[start : start + batch_size]
                    critic_values = critics[agent_idx](valid_states[indices]).squeeze(-1)
                    critic_loss = (critic_values - valid_returns[indices]).pow(2).mean()
                    critic_optimizers[agent_idx].zero_grad()
                    critic_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        critics[agent_idx].parameters(), self.args.grad_norm_clip
                    )
                    critic_optimizers[agent_idx].step()

    def _learn_level(
        self,
        batch,
        actors,
        critics,
        actor_optimizers,
        critic_optimizers,
        obs_key,
        state_key,
        next_state_key,
        action_key,
        avail_key,
        reward_key,
        padded_key,
        terminated_key,
        active_key,
        action_dim,
        obs_dim,
        state_dim,
        guard_key=None,
    ):
        states = batch[state_key]
        next_states = batch[next_state_key]
        obs = batch[obs_key]
        actions = batch[action_key].squeeze(-1)
        avail_actions = batch[avail_key]
        terminated = batch[terminated_key].squeeze(-1)
        mask = 1 - batch[padded_key].squeeze(-1)
        active_mask = batch.get(active_key, None)
        if active_mask is None:
            active_mask = torch.ones(
                (*mask.shape, self.n_agents, 1),
                dtype=mask.dtype,
                device=self.device,
            )
        active_mask = active_mask.squeeze(-1)
        guard_applied = batch.get(guard_key, None) if guard_key is not None else None
        if guard_applied is not None:
            guard_applied = guard_applied.squeeze(-1)

        rewards = batch[reward_key]
        if rewards.size(-1) == 1:
            rewards = rewards.expand(-1, -1, self.n_agents)

        episode_num = states.size(0)
        time_len = states.size(1)
        flat_states = states.reshape(episode_num * time_len, -1)

        for agent_idx in range(self.n_agents):
            actor_states = obs[:, :, agent_idx, :].reshape(
                episode_num * time_len, -1
            )
            agent_actions = actions[:, :, agent_idx].reshape(-1)
            agent_avail = avail_actions[:, :, agent_idx, :].reshape(-1, action_dim)
            agent_rewards = rewards[:, :, agent_idx]
            agent_mask = mask * active_mask[:, :, agent_idx]
            flat_agent_mask = agent_mask.reshape(-1) > 0

            with torch.no_grad():
                values = critics[agent_idx](states.reshape(-1, state_dim)).reshape(
                    episode_num, time_len
                )
                next_values = critics[agent_idx](
                    next_states.reshape(-1, state_dim)
                ).reshape(episode_num, time_len)
                advantages, returns = self._compute_advantages(
                    agent_rewards, values, next_values, terminated, agent_mask
                )
                old_dist = self._masked_categorical(
                    actors[agent_idx](actor_states), agent_avail
                )
                old_log_probs = old_dist.log_prob(agent_actions)

            valid_states = flat_states[flat_agent_mask]
            valid_returns = returns.reshape(-1)[flat_agent_mask]
            if guard_applied is not None:
                actor_mask = agent_mask * (1 - guard_applied[:, :, agent_idx])
                flat_actor_mask = actor_mask.reshape(-1) > 0
            else:
                flat_actor_mask = flat_agent_mask

            valid_actor_states_pg = actor_states[flat_actor_mask]
            valid_actions_pg = agent_actions[flat_actor_mask]
            valid_avail_pg = agent_avail[flat_actor_mask]
            valid_advantages_pg = advantages.reshape(-1)[flat_actor_mask]
            valid_old_log_probs_pg = old_log_probs[flat_actor_mask]

            if valid_states.size(0) == 0:
                continue

            batch_size = min(getattr(self.args, "batch_size", 64), valid_states.size(0))

            for _ in range(self.args.ppo_epoch):
                if valid_actor_states_pg.size(0) > 0:
                    actor_batch_size = min(batch_size, valid_actor_states_pg.size(0))
                    actor_permutation = torch.randperm(
                        valid_actor_states_pg.size(0), device=self.device
                    )
                    for start in range(0, valid_actor_states_pg.size(0), actor_batch_size):
                        indices = actor_permutation[start : start + actor_batch_size]
                        dist = self._masked_categorical(
                            actors[agent_idx](valid_actor_states_pg[indices]),
                            valid_avail_pg[indices],
                        )
                        new_log_probs = dist.log_prob(valid_actions_pg[indices])
                        ratio = torch.exp(new_log_probs - valid_old_log_probs_pg[indices])
                        clipped_ratio = torch.clamp(
                            ratio,
                            1.0 - self.args.clip_param,
                            1.0 + self.args.clip_param,
                        )
                        surrogate_1 = ratio * valid_advantages_pg[indices]
                        surrogate_2 = clipped_ratio * valid_advantages_pg[indices]
                        entropy = dist.entropy()
                        actor_loss = -torch.min(surrogate_1, surrogate_2)
                        actor_loss -= self.args.entropy_coef * entropy
                        actor_loss = actor_loss.mean()

                        actor_optimizers[agent_idx].zero_grad()
                        actor_loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            actors[agent_idx].parameters(), self.args.grad_norm_clip
                        )
                        actor_optimizers[agent_idx].step()

                critic_permutation = torch.randperm(valid_states.size(0), device=self.device)
                for start in range(0, valid_states.size(0), batch_size):
                    indices = critic_permutation[start : start + batch_size]
                    critic_values = critics[agent_idx](valid_states[indices]).squeeze(-1)
                    critic_loss = (critic_values - valid_returns[indices]).pow(2).mean()
                    critic_optimizers[agent_idx].zero_grad()
                    critic_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        critics[agent_idx].parameters(), self.args.grad_norm_clip
                    )
                    critic_optimizers[agent_idx].step()

    def _model_filename(self, level, role, agent_idx, prefix="", model_dir=None):
        model_dir = self.model_dir if model_dir is None else model_dir
        return os.path.join(
            model_dir,
            f"{prefix}{level}_agent_{agent_idx}_{role}_discrete_ppo",
        )

    def _latest_paths(self, level, role, agent_idx, model_dir=None):
        direct = self._model_filename(level, role, agent_idx, model_dir=model_dir)
        if os.path.exists(direct):
            return direct
        candidates = glob.glob(
            self._model_filename(level, role, agent_idx, prefix="*_", model_dir=model_dir)
        )
        if not candidates:
            return None

        def _sort_key(path):
            prefix = os.path.basename(path).split("_", 1)[0]
            return int(prefix) if prefix.isdigit() else -1

        return max(candidates, key=_sort_key)

    def _resolve_checkpoint_dir(self, checkpoint_dir):
        checkpoint_dir = os.path.normpath(str(checkpoint_dir))
        candidates = [
            checkpoint_dir,
            os.path.join(checkpoint_dir, self.args.alg, self.args.map),
            os.path.join(checkpoint_dir, "hmappo", self.args.map),
        ]
        for candidate in candidates:
            if self._latest_paths("low", "actor", 0, model_dir=candidate) is not None:
                return candidate
        raise Exception(f"No pretrained low-level model found under {checkpoint_dir}!")

    def _load_level_models(self, level, actors, critics, model_dir=None):
        for agent_idx in range(self.n_agents):
            actor_path = self._latest_paths(level, "actor", agent_idx, model_dir=model_dir)
            critic_path = self._latest_paths(level, "critic", agent_idx, model_dir=model_dir)
            if actor_path is None or critic_path is None:
                raise Exception(f"No {level} model for agent {agent_idx}!")
            actor_state = torch.load(actor_path, map_location=self.device)
            actors[agent_idx].load_state_dict(
                actor_state,
                strict=(level != "high"),
            )
            critics[agent_idx].load_state_dict(
                torch.load(critic_path, map_location=self.device)
            )

    def _load_pretrained_low_models(self, checkpoint_dir):
        checkpoint_dir = self._resolve_checkpoint_dir(checkpoint_dir)
        self._load_level_models(
            "low",
            self.low_actors,
            self.low_critics,
            model_dir=checkpoint_dir,
        )

    def _load_models(self):
        for level, actors, critics in (
            ("low", self.low_actors, self.low_critics),
            ("high", self.high_actors, self.high_critics),
        ):
            self._load_level_models(level, actors, critics)
        if self.use_td_energy_critic:
            for agent_idx in range(self.n_agents):
                for critic_idx, critic in enumerate(self.high_energy_critics[agent_idx]):
                    path = self._latest_paths(
                        "high_energy",
                        f"critic{critic_idx}",
                        agent_idx,
                    )
                    if path is None:
                        continue
                    state = torch.load(path, map_location=self.device)
                    critic.load_state_dict(state)
                    self.high_energy_target_critics[agent_idx][
                        critic_idx
                    ].load_state_dict(state)

    def save_model(self, train_step):
        num = str(train_step // max(self.args.save_cycle, 1))
        os.makedirs(self.model_dir, exist_ok=True)
        for level, actors, critics in (
            ("low", self.low_actors, self.low_critics),
            ("high", self.high_actors, self.high_critics),
        ):
            for agent_idx in range(self.n_agents):
                for prefix in (f"{num}_", ""):
                    torch.save(
                        actors[agent_idx].state_dict(),
                        self._model_filename(level, "actor", agent_idx, prefix=prefix),
                    )
                    torch.save(
                        critics[agent_idx].state_dict(),
                        self._model_filename(level, "critic", agent_idx, prefix=prefix),
                    )
        if self.use_td_energy_critic:
            for agent_idx in range(self.n_agents):
                for critic_idx, critic in enumerate(self.high_energy_critics[agent_idx]):
                    for prefix in (f"{num}_", ""):
                        torch.save(
                            critic.state_dict(),
                            self._model_filename(
                                "high_energy",
                                f"critic{critic_idx}",
                                agent_idx,
                                prefix=prefix,
                            ),
                        )
