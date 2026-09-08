"""Only necessary regression checks for the internal hit-basis ablation."""

import ast
from argparse import Namespace
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
from stable_baselines3.common.vec_env import DummyVecEnv

from experiments.directional_navigation.deployable_observation import DeployableLidarExtractor, DeployableRecovery, observation_space
from experiments.directional_navigation.derived_hit_observation import DerivedHitLidarExtractor
from scripts import run_deployable_observation_v2 as v2
from scripts import run_deployable_derived_hit_v3 as v3


def test_preserved_initial_function_rng_and_trainable_new_channel():
    torch.set_num_threads(1)
    torch.manual_seed(9)
    old=DeployableLidarExtractor(observation_space())
    expected_rng=torch.get_rng_state().clone()
    torch.manual_seed(9)
    new=DerivedHitLidarExtractor(observation_space())
    assert torch.equal(torch.get_rng_state(),expected_rng)
    x=torch.rand(3,1039)
    x[:,7:10]=1
    torch.testing.assert_close(old(x),new(x),atol=1e-6,rtol=1e-6)
    assert sum(p.numel() for p in new.parameters())-sum(p.numel() for p in old.parameters()) == 144
    new(x).square().mean().backward()
    grad=new.lidar_convolutions[0].weight.grad[:,1]
    assert torch.isfinite(grad).all() and grad.abs().sum()>0


def test_current_sensor_mask_reconstruction_at_max_range_boundary():
    env=DeployableRecovery(battery_capacity=100,horizon=2,obstacles=0)
    try:
        env.reset(seed=9)
        maximum=np.float32(env.base.lidar_max_range)
        near=[maximum]
        for _ in range(32):
            near.append(np.nextafter(near[-1],np.float32(0)))
        ranges=np.random.default_rng(9).uniform(0,float(maximum),1024).astype(np.float32)
        ranges[:len(near)]=near
        env.base.agent.lasers=ranges
        distances,hits=env.base._lidar_features()
        obs=np.zeros((1,1039),np.float32)
        obs[0,7:1031]=distances
        channels=DerivedHitLidarExtractor.sensor_channels(torch.from_numpy(obs))
        np.testing.assert_array_equal(channels[0,0].numpy().ravel(),distances)
        np.testing.assert_array_equal(channels[0,1].numpy().ravel(),hits)
    finally:
        env.close()


def test_whole_sac_initial_parameters_match_and_checkpoint_loop_unchanged():
    torch.set_num_threads(1)
    env=DummyVecEnv([lambda:DeployableRecovery(battery_capacity=100,horizon=2,obstacles=0)])
    args=Namespace(batch_size=8,device="cpu",buffer_size=32,learning_starts=8)
    try:
        old,new=v2.make_model(env,args),v3.make_model(env,args)
        a,b=old.policy.state_dict(),new.policy.state_dict()
        assert a.keys()==b.keys()
        for key in a:
            if a[key].shape==b[key].shape:
                torch.testing.assert_close(a[key],b[key],atol=0,rtol=0)
            else:
                assert key.endswith("lidar_convolutions.0.weight")
                torch.testing.assert_close(a[key],b[key][:,:1],atol=0,rtol=0)
                assert torch.count_nonzero(b[key][:,1:])==0
        x=env.reset()
        np.testing.assert_allclose(old.predict(x,deterministic=True)[0],new.predict(x,deterministic=True)[0],atol=1e-6)
    finally:
        env.close()
    def training_ast(path):
        tree=ast.parse(Path(path).read_text())
        return ast.dump(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="train"))
    assert training_ast(v2.__file__)==training_ast(v3.__file__)
    assert v2.save_checkpoint is v3.save_checkpoint and v2.load_checkpoint is v3.load_checkpoint


def test_one_tiny_new_runner_smoke(tmp_path):
    root=tmp_path/"smoke"
    subprocess.run([sys.executable,"scripts/run_deployable_derived_hit_v3.py","--start",
        "--output-dir",str(root),"--battery-capacity","100","--capacity-source","unit_test",
        "--device","cpu","--num-envs","1","--rollout-steps","8","--target-steps","16",
        "--checkpoint-steps","8","--batch-size","8","--buffer-size","32","--learning-starts","8",
        "--horizon","8","--obstacles","0"],check=True,capture_output=True,text=True,timeout=90)
    status=json.loads((root/"status.json").read_text())
    contract=json.loads((root/"manifest.json").read_text())["contract"]
    assert status["status"]=="TRAINING_COMPLETE_AWAITING_USER" and status["transitions"]==16
    assert contract["protocol"]=="deployable_distance_derived_hit_v3"
    assert contract["observation_dim"]==1039 and contract["lidar_shape"]==[2,8,128]
    assert (root/"sac/latest.json").exists() and not (root/"ERROR.json").exists()
