import glob
import os

import torch
import torch.nn as nn
from torch.distributions import Categorical

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


class MAPPO:
    def __init__(self, args):
        self.args = args
        self.n_agents = args.n_agents
        self.low_n_actions = args.n_actions
        self.low_obs_shape = args.obs_shape
        self.low_state_shape = args.state_shape
        self.high_n_actions = int(getattr(args, "high_level_n_actions", 0))
        self.high_obs_shape = int(getattr(args, "high_level_obs_shape", 0))
        self.high_state_shape = int(getattr(args, "high_level_state_shape", 0))
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
                DiscreteActorNetwork(
                    self.high_obs_shape, self.high_n_actions, high_actor_hidden_dim
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

        self.low_actor_optimizers = [
            torch.optim.Adam(actor.parameters(), lr=args.lr_actor)
            for actor in self.low_actors
        ]
        self.low_critic_optimizers = [
            torch.optim.Adam(critic.parameters(), lr=args.lr_critic)
            for critic in self.low_critics
        ]
        high_lr_actor = getattr(args, "high_lr_actor", args.lr_actor)
        high_lr_critic = getattr(args, "high_lr_critic", args.lr_critic)
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

    def init_hidden(self, episode_num):
        self.eval_hidden = torch.zeros(
            (episode_num, self.n_agents, 1), device=self.device
        )

    def _masked_categorical(self, logits, avail_actions):
        masked_logits = logits.masked_fill(avail_actions <= 0, -1e10)
        return Categorical(logits=masked_logits)

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
        self, observation, agent_idx, avail_actions, evaluate=False
    ):
        return self._choose_from_network(
            self.high_actors[agent_idx], observation, avail_actions, evaluate
        )

    def _prepare_batch(self, batch):
        tensor_batch = {}
        for key, value in batch.items():
            if key in ("u", "high_u"):
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
        batch = self._prepare_batch(batch)
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

        if "high_o" not in batch:
            return

        self._learn_level(
            batch=batch,
            actors=self.high_actors,
            critics=self.high_critics,
            actor_optimizers=self.high_actor_optimizers,
            critic_optimizers=self.high_critic_optimizers,
            obs_key="high_o",
            state_key="high_s",
            next_state_key="high_s_next",
            action_key="high_u",
            avail_key="high_avail_u",
            reward_key="high_r",
            padded_key="high_padded",
            terminated_key="high_terminated",
            active_key="high_agent_active_mask",
            action_dim=self.high_n_actions,
            obs_dim=self.high_obs_shape,
            state_dim=self.high_state_shape,
        )

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

    def _model_filename(self, level, role, agent_idx, prefix=""):
        return os.path.join(
            self.model_dir,
            f"{prefix}{level}_agent_{agent_idx}_{role}_discrete_ppo",
        )

    def _latest_paths(self, level, role, agent_idx):
        direct = self._model_filename(level, role, agent_idx)
        if os.path.exists(direct):
            return direct
        candidates = glob.glob(self._model_filename(level, role, agent_idx, prefix="*_"))
        if not candidates:
            return None

        def _sort_key(path):
            prefix = os.path.basename(path).split("_", 1)[0]
            return int(prefix) if prefix.isdigit() else -1

        return max(candidates, key=_sort_key)

    def _load_models(self):
        for level, actors, critics in (
            ("low", self.low_actors, self.low_critics),
            ("high", self.high_actors, self.high_critics),
        ):
            for agent_idx in range(self.n_agents):
                actor_path = self._latest_paths(level, "actor", agent_idx)
                critic_path = self._latest_paths(level, "critic", agent_idx)
                if actor_path is None or critic_path is None:
                    raise Exception(f"No {level} model for agent {agent_idx}!")
                actors[agent_idx].load_state_dict(
                    torch.load(actor_path, map_location=self.device)
                )
                critics[agent_idx].load_state_dict(
                    torch.load(critic_path, map_location=self.device)
                )

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
