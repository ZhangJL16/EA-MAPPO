import numpy as np


class MPE2SimpleSpeakerListenerWrapper:
    def __init__(
        self,
        max_cycles=100,
        continuous_actions=False,
    ):
        env_module = None
        import_error = None
        for module_name in (
            "mpe2.simple_speaker_listener_v4",
            "pettingzoo.mpe.simple_speaker_listener_v4",
        ):
            try:
                env_module = __import__(module_name, fromlist=["parallel_env"])
                break
            except ImportError as exc:
                import_error = exc
        if env_module is None:
            raise ImportError(
                "simple_speaker_listener_v4 requires either `mpe2` or `pettingzoo[mpe]`."
            ) from import_error

        self.parallel_env = env_module.parallel_env(
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            render_mode=None,
        )
        self.max_cycles = max_cycles
        self.controlled_agents = ["speaker_0", "listener_0"]
        self.n_agents = len(self.controlled_agents)

        self.agent_action_dims = {
            agent: int(self.parallel_env.action_space(agent).n)
            for agent in self.controlled_agents
        }
        self.n_actions = max(self.agent_action_dims.values())

        self.agent_obs_dims = {
            agent: int(np.prod(self.parallel_env.observation_space(agent).shape))
            for agent in self.controlled_agents
        }
        self.obs_shape = max(self.agent_obs_dims.values())

        self._obs = {}
        self._infos = {}
        self._step_count = 0
        self._episode_reward = 0.0
        self._battle_won = False
        self._last_reward = 0.0

    def _pad_obs(self, agent_name, obs):
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        if obs.shape[0] == self.obs_shape:
            return obs
        padded = np.zeros(self.obs_shape, dtype=np.float32)
        padded[: obs.shape[0]] = obs
        return padded

    def reset(self):
        self._step_count = 0
        self._episode_reward = 0.0
        self._battle_won = False
        self._last_reward = 0.0

        reset_result = self.parallel_env.reset()
        if isinstance(reset_result, tuple) and len(reset_result) == 2:
            self._obs, self._infos = reset_result
        else:
            self._obs = reset_result
            self._infos = {agent: {} for agent in self.controlled_agents}
        return self.get_obs()

    def step(self, actions):
        action_dict = {}
        for agent_name, action in zip(self.controlled_agents, actions):
            max_action = self.agent_action_dims[agent_name]
            action = int(action)
            if action < 0 or action >= max_action:
                action = 0
            action_dict[agent_name] = action

        step_result = self.parallel_env.step(action_dict)
        if len(step_result) == 5:
            obs, rewards, terminations, truncations, infos = step_result
            done_dict = {
                agent: bool(terminations.get(agent, False) or truncations.get(agent, False))
                for agent in self.controlled_agents
            }
        else:
            obs, rewards, done_dict, infos = step_result

        self._obs = obs
        self._infos = infos
        self._step_count += 1

        cooperative_rewards = [float(rewards.get(agent, 0.0)) for agent in self.controlled_agents]
        mean_reward = float(np.mean(cooperative_rewards))
        self._episode_reward += mean_reward
        self._last_reward = mean_reward

        if mean_reward > -0.05:
            self._battle_won = True

        terminated = all(done_dict.get(agent, False) for agent in self.controlled_agents)
        info = {
            "battle_won": self._battle_won,
            "warning_signal": np.zeros((self.n_agents, 1), dtype=np.float32),
        }
        return mean_reward, terminated, info

    def get_obs(self):
        return np.stack(
            [self._pad_obs(agent, self._obs[agent]) for agent in self.controlled_agents],
            axis=0,
        )

    def get_obs_agent(self, agent_id):
        agent_name = self.controlled_agents[agent_id]
        return self._pad_obs(agent_name, self._obs[agent_name])

    def get_state(self):
        return np.concatenate(
            [self._pad_obs(agent, self._obs[agent]) for agent in self.controlled_agents],
            axis=0,
        )

    def get_avail_agent_actions(self, agent_id):
        agent_name = self.controlled_agents[agent_id]
        avail = np.zeros(self.n_actions, dtype=np.float32)
        avail[: self.agent_action_dims[agent_name]] = 1.0
        return avail

    def get_env_info(self):
        self.reset()
        return {
            "n_actions": self.n_actions,
            "n_agents": self.n_agents,
            "state_shape": int(self.get_state().shape[0]),
            "obs_shape": int(self.obs_shape),
            "episode_limit": self.max_cycles,
            "msg_shape": 0,
        }

    def summary(self):
        return {
            "step": float(self._step_count),
            "agent_health": float(self.n_agents),
            "enemy_health": max(0.0, -self._last_reward),
            "agent_alive": float(self.n_agents),
        }

    def close(self):
        if hasattr(self.parallel_env, "close"):
            self.parallel_env.close()

