import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from cadapt.environment import load_task,DeploymentEnv,PhysicalGain,split_done,hover_feasibility
from cadapt.train import run


ROOT=Path(__file__).resolve().parents[1]
CONFIG=json.loads((ROOT/'configs/study_v1.json').read_text())
REFERENCE=os.environ.get('SCG_REFERENCE','/home/zjl/safe_control_gym_reference_20260919')


class InterfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.task=load_task(REFERENCE,CONFIG)

    def test_timeout_not_terminal(self):
        self.assertEqual(split_done(True,{'TimeLimit.truncated':True}),(False,True))
        self.assertEqual(split_done(True,{}),(True,False))
        self.assertEqual(split_done(False,{}),(False,False))

    def test_gain_and_hover(self):
        p=PhysicalGain([.95,1.05])
        np.testing.assert_allclose(p.apply(np.array([.12,.13]),None),[.114,.1365])
        for gain in CONFIG['deployment_gains'].values():
            self.assertTrue(hover_feasibility(gain)['within_action_box'])
        self.assertFalse(hover_feasibility([.8,.8])['within_action_box'])

    def test_native_parity_history_privacy(self):
        from safe_control_gym.envs.gym_pybullet_drones.quadrotor import Quadrotor
        task=copy.deepcopy(self.task)
        task['seed']=99
        original=Quadrotor(**task)
        # Upstream base_aviary.py changeDynamics omits physicsClientId.
        # Run serially: concurrent clients would silently have different damping.
        action=np.array([.03,-.02],dtype=np.float32)
        native_initial,_=original.reset(seed=99)
        trace=[]
        for _ in range(3):
            result=original.step(action)
            trace.append(result)
            if result[2]:
                break
        original.close()
        wrapped=DeploymentEnv(task,99)
        try:
            obs,safe=wrapped.reset()
            np.testing.assert_allclose(obs[:12],native_initial,atol=1e-6)
            self.assertEqual(safe,{})
            self.assertEqual(obs.shape,(21,))
            self.assertEqual(obs[-1],0)
            for native,r,done,info in trace:
                out,reward,term,trunc,safe=wrapped.step(action)
                np.testing.assert_allclose(out[:12],native,atol=1e-6)
                np.testing.assert_allclose(out[12:18],obs[:6],atol=1e-6)
                np.testing.assert_array_equal(out[18:20],action)
                self.assertEqual(out[-1],1)
                self.assertAlmostEqual(reward,r)
                self.assertEqual(done,term or trunc)
                self.assertFalse({'symbolic_model','physical_parameters','gain'} & set(safe))
                obs=out
                if done:
                    break
        finally:
            wrapped.close()

    def test_episode_boundary_reconstruction(self):
        a=DeploymentEnv(self.task,100,reset_index=7)
        b=DeploymentEnv(self.task,100,reset_index=7)
        try:
            x,_=a.reset()
            y,_=b.reset(seed=100)  # SB3 re-supplies root seed on load.
            np.testing.assert_array_equal(x,y)
        finally:
            a.close()
            b.close()

    def test_training_and_resume(self):
        cfg=copy.deepcopy(CONFIG)
        cfg.update(hidden=[8,8],learning_starts=0,batch_size=4,replay_size=100,
                   base_steps=100,first_checkpoint_steps=50,checkpoint_steps=50)
        short=copy.deepcopy(self.task)
        short.update(episode_len_sec=.08,randomized_init=False)
        with tempfile.TemporaryDirectory() as d, patch('cadapt.train.load_task',return_value=short):
            full,resume=Path(d)/'full',Path(d)/'resumed'
            run(cfg,REFERENCE,full,42,max_episodes=4)
            run(cfg,REFERENCE,resume,42,max_episodes=2)
            run(cfg,REFERENCE,resume,42,max_episodes=2)
            def latest(p):
                return p/json.loads((p/'latest.json').read_text())['checkpoint']
            from stable_baselines3 import SAC
            x=SAC.load(latest(full)/'model.zip',device='cpu')
            y=SAC.load(latest(resume)/'model.zip',device='cpu')
            self.assertEqual(x.num_timesteps,y.num_timesteps)
            self.assertEqual(x._n_updates,y._n_updates)
            for k,v in x.policy.state_dict().items():
                torch.testing.assert_close(v,y.policy.state_dict()[k],rtol=0,atol=0)
            sx=json.loads((latest(full)/'state.json').read_text())
            sy=json.loads((latest(resume)/'state.json').read_text())
            self.assertEqual(sx['history'],sy['history'])
            self.assertTrue(sx['finite_parameters'])
            self.assertEqual(sx['replay_size'],sy['replay_size'])


if __name__=='__main__':
    unittest.main()
