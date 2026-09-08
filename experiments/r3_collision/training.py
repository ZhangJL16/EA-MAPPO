"""SAC-derived reach-avoid learning; no bridge, extra critic, or soft-Q target."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch as th
from torch.nn import functional as F
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.utils import polyak_update

from envs.UAVEnergyDeliverySAC import UAVEnergyDeliverySACEnv
from .core import correction_distance, shaped_transition, survival_weight


class ReachAvoidEnv(UAVEnergyDeliverySACEnv):
    def __init__(self, *, kappa: float, discount: float, **kwargs: Any):
        super().__init__(**kwargs)
        self.kappa, self.discount = kappa, discount
        self.corrections: list[float] = []

    def potential(self) -> float:
        return -float(np.clip(np.linalg.norm(self.active_goal-self.agent.pos)/self.d_max, 0, 1))

    def reset(self, *, seed=None, options=None):
        obs, _ = super().reset(seed=seed, options=options)
        self.episode_integral = 0.0
        self.episode_counts = dict(intervention=0, emergency=0, fallback=0)
        self.start_distance = float(np.linalg.norm(self.active_goal-self.agent.pos))
        return obs, {}

    def _safety_filtered_action(self, nominal_action):
        executed, diag = super()._safety_filtered_action(nominal_action)
        self.corrections.append(correction_distance(
            self._normalized_action_to_acceleration(nominal_action),
            self._normalized_action_to_acceleration(executed),
            self.horizontal_a_max, self.vertical_a_max,
        ))
        return executed, diag

    def step(self, action):
        before = self.potential()
        self.corrections = []
        obs, _, terminated, truncated, info = super().step(action)
        if len(self.corrections) != info["physics_substeps"]:
            raise AssertionError("missing physical correction substeps")
        integral = float(np.square(self.corrections).sum()*self.physics_dt)
        contact = bool(info["obstacle_collision"] or info["boundary_contact"])
        goal = bool(info["is_success"] and not contact)
        if terminated and not (goal or contact):
            raise RuntimeError(f"unexpected navigation termination: {info['end_reason']}")
        after = 0.0 if (goal or contact) else self.potential()
        reward, beta = shaped_transition(goal=goal, contact=contact, correction_integral=integral,
                                        kappa=self.kappa, gamma=self.discount,
                                        potential=before, next_potential=after)
        q = survival_weight(contact=contact, correction_integral=integral, kappa=self.kappa)
        self.episode_integral += integral
        for key, source in [('intervention','hocbf_intervened'), ('emergency','hocbf_emergency_brake'),
                            ('fallback','hocbf_fallback_used')]:
            self.episode_counts[key] += int(info[source])
        # Minimal IPC payload; old bridge anchors and constraint matrices are not sent.
        light = dict(ra_beta=beta, ra_q=q, ra_integral=integral, ra_phi=before,
                     ra_goal=goal, ra_contact=contact, is_success=goal)
        if terminated or truncated or contact:
            light['ra_episode'] = dict(safe_goal=goal, contact=contact, steps=self.current_step,
                end_reason='first_contact' if contact else info['end_reason'],
                path_ratio=self._current_goal_path_length/self.start_distance,
                correction_integral=self.episode_integral, **self.episode_counts)
        return obs, reward, bool(goal or contact), bool(truncated and not (goal or contact)), light


class ReachAvoidReplay(ReplayBuffer):
    """Store beta explicitly; never encode fractional survival in boolean dones.

    MC returns are used ONLY while the actor is frozen. A timeout has an unknown
    infinite-horizon continuation and therefore does not receive an MC label.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.optimize_memory_usage:
            raise ValueError('explicit terminal observations require separate next_obs storage')
        shape = (self.buffer_size, self.n_envs)
        self.betas = np.zeros(shape, np.float32)
        self.phis = np.zeros(shape, np.float32)
        self.survival_weights = np.zeros(shape, np.float32)
        self.correction_integrals = np.zeros(shape, np.float32)
        self.mc = np.full(shape, np.nan, np.float32)
        self.stamps = np.full(shape, -1, np.int64)
        self.sequence = 0
        self.pending: list[list[tuple[int, int]]] = [[] for _ in range(self.n_envs)]
        self.mc_episodes = 0

    def add(self, obs, next_obs, action, reward, done, infos):
        pos = self.pos
        for e, info in enumerate(infos):
            beta = float(info['ra_beta'])
            if not np.isfinite(beta) or not 0 <= beta < 1:
                raise ValueError('invalid reach-avoid continuation')
            if done[e] and not info.get('TimeLimit.truncated', False) and beta != 0:
                raise ValueError('physical terminal must have zero continuation')
            self.betas[pos,e], self.phis[pos,e] = beta, info['ra_phi']
            self.survival_weights[pos,e] = info['ra_q']
            self.correction_integrals[pos,e] = info['ra_integral']
            self.mc[pos,e] = np.nan
            self.stamps[pos,e] = self.sequence
            self.pending[e].append((pos,self.sequence))
        super().add(obs,next_obs,action,reward,done,infos)
        for e, terminal in enumerate(done):
            if terminal:
                if not infos[e].get('TimeLimit.truncated',False):
                    value = 0.0
                    for row, stamp in reversed(self.pending[e]):
                        if self.stamps[row,e] != stamp:
                            break
                        value = float(self.rewards[row,e])+float(self.betas[row,e])*value
                        self.mc[row,e] = value
                    self.mc_episodes += 1
                self.pending[e] = []
        self.sequence += 1

    def _get_samples(self, batch_inds, env=None):
        if env is not None:
            raise ValueError('VecNormalize would change the declared reach-avoid reward scale')
        ei = np.random.randint(self.n_envs, size=len(batch_inds))
        data = (self.observations[batch_inds,ei], self.actions[batch_inds,ei],
                self.next_observations[batch_inds,ei],
                (self.dones[batch_inds,ei]*(1-self.timeouts[batch_inds,ei])).reshape(-1,1),
                self.rewards[batch_inds,ei].reshape(-1,1), self.betas[batch_inds,ei].reshape(-1,1))
        return ReplayBufferSamples(*map(self.to_torch,data))

    def sample_mc(self, batch_size):
        eligible = np.flatnonzero(np.isfinite(self.mc))
        if not len(eligible):
            return None
        rows, ei = np.unravel_index(np.random.choice(eligible,batch_size),self.mc.shape)
        return tuple(self.to_torch(x) for x in (self.observations[rows,ei],self.actions[rows,ei],
                                                self.mc[rows,ei,None]))

    def discard_unfinished_episodes(self):
        # Restarted simulator episodes cannot complete old pending MC trajectories.
        self.pending = [[] for _ in range(self.n_envs)]


class ReachAvoidSAC(SAC):
    def __init__(self, *args, warmup_transitions=10000, training_budget=50000,
                 actor_lr=3e-5, actor_entropy=1e-3, **kwargs):
        self.warmup_transitions = int(warmup_transitions)
        self.training_budget = int(training_budget)
        self.actor_lr = float(actor_lr)
        self.actor_entropy = float(actor_entropy)
        self.actor_updates = 0
        self.mc_updates = 0
        self.last_ra_metrics: dict[str,float] = {}
        super().__init__(*args, **kwargs)

    def _sample_action(self, learning_starts, action_noise=None, n_envs=1):
        # Even collection warmup uses pretrained R3, never uniform random actions.
        return super()._sample_action(0, action_noise, n_envs)

    def train(self, gradient_steps: int, batch_size: int = 256):
        self.policy.set_training_mode(True)
        self._update_learning_rate([self.critic.optimizer])
        for group in self.actor.optimizer.param_groups:
            group['lr'] = self.actor_lr
        warmup = self.num_timesteps <= self.warmup_transitions
        fraction = np.clip((self.num_timesteps-self.warmup_transitions)
                           / max((self.training_budget-self.warmup_transitions)/2,1),0,1)
        alpha = self.actor_entropy*(1-fraction)
        if not warmup and self.mc_updates == 0:
            raise RuntimeError('no complete-trajectory critic initialization; refusing actor update')
        for _ in range(gradient_steps):
            if warmup:
                mc_batch = self.replay_buffer.sample_mc(batch_size)
                if mc_batch is None:
                    return
                obs, actions, target = mc_batch
            else:
                data = self.replay_buffer.sample(batch_size)
                obs, actions = data.observations, data.actions
                with th.no_grad():
                    next_actions, _ = self.actor.action_log_prob(data.next_observations)
                    nxt = th.cat(self.critic_target(data.next_observations,next_actions),1).min(1,keepdim=True).values
                    # beta already includes gamma, artificial survival and goal/contact.
                    target = data.rewards + data.discounts*nxt
            values = self.critic(obs,actions)
            loss = .5*sum(F.mse_loss(value,target) for value in values)
            if not th.isfinite(loss):
                raise FloatingPointError('nonfinite critic loss')
            self.critic.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(self.critic.parameters(),10,error_if_nonfinite=True)
            self.critic.optimizer.step()
            actor_loss = th.zeros((),device=self.device)
            if not warmup:
                # Do not accumulate critic gradients through the actor update.
                self.critic.requires_grad_(False)
                try:
                    sampled, logp = self.actor.action_log_prob(obs)
                    q = th.cat(self.critic(obs,sampled),1).min(1,keepdim=True).values
                    actor_loss = (alpha*logp.reshape(-1,1)-q).mean()
                    if not th.isfinite(actor_loss):
                        raise FloatingPointError('nonfinite actor loss')
                    self.actor.optimizer.zero_grad()
                    actor_loss.backward()
                    th.nn.utils.clip_grad_norm_(self.actor.parameters(),10,error_if_nonfinite=True)
                    self.actor.optimizer.step()
                finally:
                    self.critic.requires_grad_(True)
                self.actor_updates += 1
            else:
                self.mc_updates += 1
            polyak_update(self.critic.parameters(),self.critic_target.parameters(),self.tau)
            polyak_update(self.batch_norm_stats,self.batch_norm_stats_target,1.)
            self._n_updates += 1
            self.last_ra_metrics = dict(critic_loss=float(loss.detach()),actor_loss=float(actor_loss.detach()),
                target_mean=float(target.mean()),target_min=float(target.min()),target_max=float(target.max()),
                entropy_coefficient=float(alpha),actor_updates=self.actor_updates,mc_updates=self.mc_updates,
                mc_complete_episodes=self.replay_buffer.mc_episodes)
        for key,value in self.last_ra_metrics.items():
            self.logger.record('reach_avoid/'+key,value)
