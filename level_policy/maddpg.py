import random

import numpy as np
import pandas as pd
import torch
import os
from policy.actor_critic import Actor, Critic
import torch.nn as nn
from torch.distributions import Categorical

POLICY_SCOPE = "level_policy"


class DiscreteManagerActor(nn.Module):
    def __init__(self, input_dim, n_actions, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.logits = nn.Linear(hidden_dim, n_actions)

    def forward(self, obs):
        x = torch.relu(self.fc1(obs))
        x = torch.relu(self.fc2(x))
        return self.logits(x)


class DiscreteManagerCritic(nn.Module):
    def __init__(self, state_dim, joint_action_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + joint_action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q = nn.Linear(hidden_dim, 1)

    def forward(self, state, joint_action_onehot):
        x = torch.cat([state, joint_action_onehot], dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.q(x)


class MADDPG:
    def __init__(self, args, agent_id):  # 因为不同的agent的obs、act维度可能不一样，所以神经网络不同,需要agent_id来区分
        self.args = args
        self.agent_id = agent_id
        self.train_step = 0
        self.device = torch.device(
            f"cuda:{getattr(self.args, 'gpu_id', 0)}"
            if self.args.cuda and torch.cuda.is_available()
            else "cpu"
        )

        # create the network
        self.actor_network = Actor(args, agent_id)
        self.critic_network = Critic(args)

        # build up the target network
        self.actor_target_network = Actor(args, agent_id)
        self.critic_target_network = Critic(args)

        # load the weights into the target networks
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())

        # create the optimizer
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.args.lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=self.args.lr_critic)

        self.high_n_actions = int(getattr(args, "high_level_n_actions", 0))
        self.high_obs_shape = int(getattr(args, "high_level_obs_shape", 0))
        self.high_state_shape = int(getattr(args, "high_level_state_shape", 0))
        self.has_level_manager = (
            self.high_n_actions > 0
            and self.high_obs_shape > 0
            and self.high_state_shape > 0
        )
        if self.has_level_manager:
            hidden_dim = int(getattr(args, "high_actor_hidden_dim", 128))
            critic_hidden_dim = int(getattr(args, "high_critic_hidden_dim", hidden_dim))
            self.high_actor_network = DiscreteManagerActor(
                self.high_obs_shape,
                self.high_n_actions,
                hidden_dim,
            ).to(self.device)
            self.high_actor_target_network = DiscreteManagerActor(
                self.high_obs_shape,
                self.high_n_actions,
                hidden_dim,
            ).to(self.device)
            self.high_critic_network = DiscreteManagerCritic(
                self.high_state_shape,
                self.args.n_agents * self.high_n_actions,
                critic_hidden_dim,
            ).to(self.device)
            self.high_critic_target_network = DiscreteManagerCritic(
                self.high_state_shape,
                self.args.n_agents * self.high_n_actions,
                critic_hidden_dim,
            ).to(self.device)
            self.high_actor_target_network.load_state_dict(
                self.high_actor_network.state_dict()
            )
            self.high_critic_target_network.load_state_dict(
                self.high_critic_network.state_dict()
            )
            high_lr_actor = getattr(args, "high_lr_actor", args.lr_actor)
            high_lr_critic = getattr(args, "high_lr_critic", args.lr_critic)
            self.high_actor_optim = torch.optim.Adam(
                self.high_actor_network.parameters(), lr=high_lr_actor
            )
            self.high_critic_optim = torch.optim.Adam(
                self.high_critic_network.parameters(), lr=high_lr_critic
            )

        # create the dict for store the model
        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)
        # path to save the model
        self.model_path = self.args.save_dir + '/' + self.args.scenario_name
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        self.model_path = self.model_path + '/' + 'agent_%d' % agent_id
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        # path to load trained actor and critic models
        self.load_path = self.args.load_dir + '/' + self.args.scenario_name
        self.load_path = self.load_path + '/' + 'agent_%d' % agent_id

        # 加载模型
        if os.path.exists(self.load_path + '/3999_actor_params.pkl'):
            self.actor_network.load_state_dict(torch.load(self.load_path + '/3999_actor_params.pkl'))
            self.critic_network.load_state_dict(torch.load(self.load_path + '/3999_critic_params.pkl'))
            print('Agent {} successfully loaded actor_network: {}'.format(self.agent_id,
                                                                          self.load_path + '/3999_actor_params.pkl'))
            print('Agent {} successfully loaded critic_network: {}'.format(self.agent_id,
                                                                           self.load_path + '/3999_critic_params.pkl'))
        if self.has_level_manager and os.path.exists(self.load_path + '/3999_high_actor_params.pkl'):
            self.high_actor_network.load_state_dict(
                torch.load(
                    self.load_path + '/3999_high_actor_params.pkl',
                    map_location=self.device,
                )
            )
            self.high_critic_network.load_state_dict(
                torch.load(
                    self.load_path + '/3999_high_critic_params.pkl',
                    map_location=self.device,
                )
            )
            self.high_actor_target_network.load_state_dict(
                self.high_actor_network.state_dict()
            )
            self.high_critic_target_network.load_state_dict(
                self.high_critic_network.state_dict()
            )
            print('Agent {} successfully loaded high_actor_network: {}'.format(
                self.agent_id,
                self.load_path + '/3999_high_actor_params.pkl',
            ))
            print('Agent {} successfully loaded high_critic_network: {}'.format(
                self.agent_id,
                self.load_path + '/3999_high_critic_params.pkl',
            ))

    # soft update
    def _soft_update_target_network(self):
        for target_param, param in zip(self.actor_target_network.parameters(), self.actor_network.parameters()):
            target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)

        for target_param, param in zip(self.critic_target_network.parameters(), self.critic_network.parameters()):
            target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)

    def _soft_update_high_level_target_network(self):
        if not self.has_level_manager:
            return
        for target_param, param in zip(
            self.high_actor_target_network.parameters(),
            self.high_actor_network.parameters(),
        ):
            target_param.data.copy_(
                (1 - self.args.tau) * target_param.data + self.args.tau * param.data
            )
        for target_param, param in zip(
            self.high_critic_target_network.parameters(),
            self.high_critic_network.parameters(),
        ):
            target_param.data.copy_(
                (1 - self.args.tau) * target_param.data + self.args.tau * param.data
            )

    @torch.no_grad()
    def choose_high_level_action(self, observation, agent_idx, avail_actions, evaluate=False):
        del agent_idx
        if not self.has_level_manager:
            raise AttributeError("Current MADDPG policy was not initialized with high-level spaces.")

        avail_actions = np.asarray(avail_actions, dtype=np.float32).reshape(-1)
        if avail_actions.size != self.high_n_actions:
            raise ValueError(
                "high-level avail_actions size {} does not match high_n_actions {}".format(
                    avail_actions.size,
                    self.high_n_actions,
                )
            )
        valid_actions = np.nonzero(avail_actions > 0.0)[0]
        if valid_actions.size == 0:
            return 0

        obs = torch.tensor(
            np.asarray(observation, dtype=np.float32).reshape(1, -1),
            dtype=torch.float32,
            device=self.device,
        )
        avail = torch.tensor(
            avail_actions.reshape(1, -1),
            dtype=torch.float32,
            device=self.device,
        )
        logits = self.high_actor_network(obs)
        logits = logits.masked_fill(avail <= 0.0, -1e10)
        if evaluate:
            return int(torch.argmax(logits, dim=-1).item())
        return int(Categorical(logits=logits).sample().item())

    # update the network
    def train(self, transitions, other_agents):
        for key in transitions.keys():
            transitions[key] = torch.tensor(transitions[key], dtype=torch.float32)
        r = transitions['r_%d' % self.agent_id]  # 训练时只需要自己的reward
        o, u, o_next = [], [], []  # 用来装每个agent经验中的各项
        for agent_id in range(self.args.n_agents):
            o.append(transitions['o_%d' % agent_id])
            u.append(transitions['u_%d' % agent_id])
            o_next.append(transitions['o_next_%d' % agent_id])

        # calculate the target Q value function
        u_next = []
        with torch.no_grad():
            # 得到下一个状态对应的动作
            index = 0
            for agent_id in range(self.args.n_agents):
                if agent_id == self.agent_id:
                    u_next.append(self.actor_target_network(o_next[agent_id]))
                else:
                    # 因为传入的other_agents要比总数少一个，可能中间某个agent是当前agent，不能遍历去选择动作
                    u_next.append(other_agents[index].policy.actor_target_network(o_next[agent_id]))
                    index += 1
            q_next = self.critic_target_network(o_next, u_next).detach()

            target_q = (r.unsqueeze(1) + self.args.gamma * q_next).detach()

        # the q loss
        q_value = self.critic_network(o, u)
        critic_loss = (target_q - q_value).pow(2).mean()

        # the actor loss
        # 重新选择联合动作中当前agent的动作，其他agent的动作不变
        u[self.agent_id] = self.actor_network(o[self.agent_id])
        actor_loss = - self.critic_network(o, u).mean()
        # if self.agent_id == 0:
        #     print('critic_loss is {}, actor_loss is {}'.format(critic_loss, actor_loss))
        # update the network
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        self._soft_update_target_network()
        if self.train_step > 0 and self.train_step % self.args.save_rate == 0:
            self.save_model(self.train_step)
        self.train_step += 1

    def save_model(self, train_step):
        num = str(train_step // self.args.save_rate)
        model_path = os.path.join(self.args.save_dir, self.args.scenario_name)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, 'agent_%d' % self.agent_id)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.actor_network.state_dict(), model_path + '/' + num + '_actor_params.pkl')
        torch.save(self.critic_network.state_dict(),  model_path + '/' + num + '_critic_params.pkl')
        if self.has_level_manager:
            torch.save(
                self.high_actor_network.state_dict(),
                model_path + '/' + num + '_high_actor_params.pkl',
            )
            torch.save(
                self.high_critic_network.state_dict(),
                model_path + '/' + num + '_high_critic_params.pkl',
            )

    def trainit(self, transitions):
        o_fix_overall = []
        u_fix_overall = []

        o_result_list = []

        for j in range(self.args.batch_size):
            o_fix = []
            u_fix = []
            o_result_list.append([transitions['o_%d' % self.agent_id][j]])
            for i in range(self.args.batch_size):
                o_fix.append(transitions['o_%d' % self.agent_id][j])
                u_fix.append(transitions['u_%d' % self.agent_id][j])
            o_fix = np.array(o_fix)
            u_fix = np.array(u_fix)
            o_fix_overall.append(torch.tensor(o_fix, dtype=torch.float32))
            u_fix_overall.append(torch.tensor(u_fix, dtype=torch.float32))

        transitions_copy = {}
        for key in transitions.keys():
            transitions_copy[key] = torch.tensor(transitions[key], dtype=torch.float32)

        o_1_fix_with_other_overall = []
        u_1_fix_with_other_overall = []
        for i in range(self.args.batch_size):
            o = []
            u = []
            o.append(o_fix_overall[i])
            u.append(u_fix_overall[i])
            for j in range(self.args.n_agents):
                if (j == self.agent_id): continue
                o.append(transitions_copy['o_%d' % j])
                u.append(transitions_copy['u_%d' % j])
            o_1_fix_with_other_overall.append(o)
            u_1_fix_with_other_overall.append(u)

        q_value_overall = []
        for i in range(self.args.batch_size):
            q_value_overall.append(self.critic_network(o_1_fix_with_other_overall[i], u_1_fix_with_other_overall[i]).tolist())

        for i, row in enumerate(o_result_list):
            row.append(q_value_overall[i])

        return o_result_list


