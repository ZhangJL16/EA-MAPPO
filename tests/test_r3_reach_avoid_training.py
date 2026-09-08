import gymnasium as gym
import numpy as np
import pytest
import torch
from stable_baselines3.common.vec_env import DummyVecEnv

from experiments.r3_collision.core import shaped_transition
from experiments.r3_collision.training import ReachAvoidReplay, ReachAvoidSAC
from scripts.run_r3_reach_avoid_pair import save_checkpoint, parameter_hash


class TinyNavigation(gym.Env):
    observation_space=gym.spaces.Box(-10,10,(2,),dtype=np.float32)
    action_space=gym.spaces.Box(-1,1,(1,),dtype=np.float32)

    def __init__(self, timeout=False):
        self.timeout=timeout

    def reset(self,seed=None,options=None):
        super().reset(seed=seed)
        self.t=0
        return np.zeros(2,np.float32),{}

    def step(self,action):
        self.t+=1
        goal=self.t==3 and not self.timeout
        phi=-(4-self.t)/4
        after=0 if goal else -(3-self.t)/4
        reward,beta=shaped_transition(goal=goal,contact=False,correction_integral=.1,
            kappa=.2,gamma=.9,potential=phi,next_potential=after)
        return np.array([self.t,0],np.float32),reward,goal,self.t==3 and self.timeout,dict(
            ra_beta=beta,ra_phi=phi,ra_goal=goal,ra_q=float(np.exp(-.02)),ra_integral=.1)


def model_for(env,**kwargs):
    return ReachAvoidSAC('MlpPolicy',env,learning_starts=0,buffer_size=128,batch_size=8,
        policy_kwargs=dict(net_arch=[8,8],share_features_extractor=False),
        replay_buffer_class=ReachAvoidReplay,ent_coef=0.,seed=3,device='cpu',
        warmup_transitions=12,training_budget=24,**kwargs)


def test_timeout_uses_terminal_observation_and_does_not_zero_continuation():
    vec=DummyVecEnv([lambda:TinyNavigation(timeout=True)])
    model=model_for(vec,gradient_steps=0)
    model.learn(3)
    replay=model.replay_buffer
    np.testing.assert_array_equal(replay.next_observations[2,0],[3,0])
    assert replay.betas[2,0]>0
    assert replay.timeouts[2,0]==1
    assert np.isnan(replay.mc).all()
    batch=replay._get_samples(np.array([2]))
    assert batch.dones.item()==0
    assert batch.discounts.item()==pytest.approx(.9*np.exp(-.02))
    vec.close()


def test_mc_recursion_and_actor_frozen_until_value_initialized(tmp_path):
    torch.set_num_threads(1)
    vec=DummyVecEnv([TinyNavigation])
    model=model_for(vec)
    actor_before=parameter_hash(model.actor)
    critic_before=parameter_hash(model.critic)
    model.learn(12)
    assert parameter_hash(model.actor)==actor_before
    assert parameter_hash(model.critic)!=critic_before
    assert model.mc_updates>0 and model.actor_updates==0
    replay=model.replay_buffer
    q=np.exp(-.02)
    assert replay.mc[0,0]==pytest.approx(.9**2*q**3+.75)
    assert replay.betas[2,0]==0
    model.learn(12,reset_num_timesteps=False)
    assert model.actor_updates>0 and parameter_hash(model.actor)!=actor_before
    save_checkpoint(model,tmp_path)
    target=tmp_path/'checkpoints/00000024'
    restored=ReachAvoidSAC.load(target/'model.zip',env=vec,device='cpu')
    restored.load_replay_buffer(target/'replay.pkl')
    restored.replay_buffer.discard_unfinished_episodes()
    assert parameter_hash(restored.actor)==parameter_hash(model.actor)
    assert restored.num_timesteps==24
    np.testing.assert_array_equal(restored.replay_buffer.betas,model.replay_buffer.betas)
    assert len(restored.actor.optimizer.state)>0 and len(restored.critic.optimizer.state)>0
    restored.learn(3,reset_num_timesteps=False)
    assert restored.num_timesteps==27 and np.isfinite(restored.last_ra_metrics['critic_loss'])
    vec.close()


def test_actor_update_refuses_missing_complete_mc_initialization():
    vec=DummyVecEnv([lambda:TinyNavigation(timeout=True)])
    model=model_for(vec)
    with pytest.raises(RuntimeError,match='no complete-trajectory'):
        model.learn(14)
    vec.close()


def test_replay_rejects_invalid_terminal_beta():
    vec=DummyVecEnv([TinyNavigation])
    replay=ReachAvoidReplay(10,vec.observation_space,vec.action_space,device='cpu')
    with pytest.raises(ValueError,match='zero continuation'):
        replay.add(np.zeros((1,2)),np.zeros((1,2)),np.zeros((1,1)),np.zeros(1),
                   np.ones(1),[dict(ra_beta=.9,ra_phi=0)])
    vec.close()
