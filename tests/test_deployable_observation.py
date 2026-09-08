"""Focused sensor, telemetry, locked-recovery, and checkpoint regressions."""

import json
import subprocess
import sys

import numpy as np
import pytest
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv

from envs.UAVEnergyDeliverySAC import StaticCylinderObstacle
from experiments.directional_navigation.deployable_observation import (
    DIM, DeployableRecovery, DeployableLidarExtractor, observation_space,
    pack_observation, policy_kwargs,
)
from experiments.directional_navigation.standard_baselines import (
    ContinuousRecovery, save_checkpoint, load_checkpoint,
)


OPTIONS = {"start_position": np.array([1000., 1000., 100.], np.float32),
           "task_point": np.array([1500., 1000., 100.], np.float32)}


def make_env():
    return DeployableRecovery(battery_capacity=100., horizon=20, obstacles=0)


def test_distance_only_ignores_flags_and_has_no_geometry_arguments():
    legacy = np.zeros(2055, np.float32)
    legacy[7:1031] = np.linspace(0, 1, 1024)
    kwargs = dict(charger_displacement=np.array([3., 4., 0.]), distance_scale=100.,
                  remaining_episode_fraction=.8, remaining_energy_fraction=.6,
                  return_command=False, previous_contact=True)
    obs = pack_observation(legacy, **kwargs)
    legacy[1031:] = np.nan  # Completely unused, not a sensor-health channel.
    np.testing.assert_array_equal(pack_observation(legacy, **kwargs), obs)
    assert observation_space().contains(obs) and obs.shape == (DIM,)
    np.testing.assert_array_equal(obs[7:1031], legacy[7:1031])
    np.testing.assert_allclose(obs[1032:1035], [.6, .8, 0.])
    assert obs[-1] == 1 and obs[1036] == pytest.approx(.6)
    with pytest.raises(ValueError):
        pack_observation(legacy, **{**kwargs, "remaining_energy_fraction": np.nan})


def test_telemetry_and_physics_reward_match_legacy():
    env, old = make_env(), ContinuousRecovery(seed_start=1, seed_stride=1,
                                             horizon=20, obstacles=0)
    try:
        obs, _ = env.reset(seed=91, options={**OPTIONS, "initial_soc": .7})
        legacy, _ = old.reset(seed=91, options=OPTIONS)
        np.testing.assert_array_equal(obs[:1031], legacy[:1031])
        for _ in range(3):
            a, b = env.step(np.array([.2, .1, 0.])), old.step(np.array([.2, .1, 0.]))
            assert a[1:] == b[1:]
            np.testing.assert_array_equal(env.base.agent.pos, old.base.agent.pos)
            np.testing.assert_array_equal(a[0][:1031], b[0][:1031])
        assert a[0][1036] == pytest.approx(.7 - env.energy / 100.)
        assert a[0][1036] < .7
        assert env.base.agent.energy == old.base.agent.energy == 1.0
        assert env.base.observation_space.shape == (2055,)
    finally:
        env.close()
        old.close()


def test_previous_contact_alignment_reward_lock_and_clean_reset():
    env = make_env()
    try:
        obs, _ = env.reset(seed=91, options=OPTIONS)
        assert obs[-1] == 0
        penalties = []
        for contact in (True, True, True, False, True):
            env.base.obstacles = [StaticCylinderObstacle(np.array([1001., 1000.]), 1.)] if contact else []
            env.base.agent.pos[:] = [1000., 1000., 100.]
            env.base.agent.vel[:] = 0
            obs, _, done, _, info = env.step(np.zeros(3))
            assert not done and obs[-1] == int(contact)
            assert info["cost"] == int(contact)
            assert not any(k in info for k in ("boundary_collision_count", "obstacle_collision_count"))
            penalties.append(info["raw_reward"] + env.base.time_penalty)
            if contact:
                np.testing.assert_array_equal(env.base.agent.vel, np.zeros(3))
        np.testing.assert_allclose(penalties, [-1.2, -.42, -.42, 0., -1.2], atol=1e-5)
        obs, _ = env.reset(seed=92, options=OPTIONS)
        assert obs[-1] == 0 and obs[1036] == 1
    finally:
        env.close()


def test_return_command_and_zero_budget_do_not_create_hidden_termination():
    env = make_env()
    try:
        options = {"start_position": OPTIONS["start_position"],
                   "mission_mode": "return", "initial_soc": 0.0}
        obs, _ = env.reset(seed=91, options=options)
        assert options["mission_mode"] == "return"  # Caller-owned options unmodified.
        assert obs[1037] == 1 and obs[1036] == 0
        np.testing.assert_array_equal(env.base.active_goal, env.base.charger_position)
        np.testing.assert_allclose(obs[3:6], obs[1032:1035], atol=1e-6)
        assert not env.step(np.zeros(3))[2]
        with pytest.raises(ValueError, match="conflicts"):
            env.reset(seed=91, options={**OPTIONS, "mission_mode": "return"})
    finally:
        env.close()


def test_single_channel_ordered_readout_context_and_gradient():
    torch.set_num_threads(1)
    torch.manual_seed(7)
    model = DeployableLidarExtractor(observation_space())
    obs = torch.zeros(2, DIM)
    obs[:, 7:1031] = 1
    obs[0, 7:15] = .2
    obs[1, 71:79] = .2
    obs[:, 1031:] = torch.arange(8) / 8
    obs.requires_grad_()
    result = model(obs)
    assert result.shape == (2, 2088)
    assert model.lidar_convolutions[0].in_channels == 1
    assert not torch.allclose(result[0, 32:-8], result[1, 32:-8])
    torch.testing.assert_close(result[:, -8:], obs[:, 1031:])
    result.square().sum().backward()
    assert torch.isfinite(obs.grad).all() and obs.grad[:, 7:1031].abs().sum() > 0
    with pytest.raises(ValueError):
        model(torch.zeros(1, 2056))


def test_new_context_restores_exactly_with_existing_checkpoint_code(tmp_path):
    torch.set_num_threads(1)
    vec, other = DummyVecEnv([make_env]), DummyVecEnv([make_env])
    try:
        kwargs = policy_kwargs()
        kwargs["net_arch"] = {"pi": [16], "qf": [16]}
        model = SAC("MlpPolicy", vec, device="cpu", seed=32, buffer_size=64,
                    learning_starts=8, batch_size=8, policy_kwargs=kwargs)
        vec._reset_seeds()
        model.learn(16, reset_num_timesteps=False)
        contract = {"test": "v2_exact_resume"}
        save_checkpoint(model, vec, tmp_path, contract, {"marker": 1})
        model.learn(8, reset_num_timesteps=False)
        expected = {k: v.clone() for k, v in model.policy.state_dict().items()}
        restored, records = load_checkpoint("sac", other, tmp_path, contract, "cpu")
        assert records == {"marker": 1}
        restored.learn(8, reset_num_timesteps=False)
        np.testing.assert_array_equal(restored._last_obs, model._last_obs)
        for key, value in restored.policy.state_dict().items():
            torch.testing.assert_close(value, expected[key], atol=0, rtol=0)
        assert other.get_attr("telemetry_remaining_energy") == vec.get_attr("telemetry_remaining_energy")
    finally:
        vec.close()
        other.close()


def test_prepare_does_not_construct_workers_or_train(tmp_path, monkeypatch):
    from scripts import run_deployable_observation_v2 as runner
    def forbidden(*args, **kwargs):
        raise AssertionError("preparation must not launch")
    monkeypatch.setattr(runner, "train", forbidden)
    root = tmp_path / "prepared"
    monkeypatch.setattr(sys, "argv", ["prepare", "--output-dir", str(root),
                                      "--battery-capacity", "100", "--capacity-source", "unit_test"])
    runner.main()
    status = json.loads((root / "status.json").read_text())
    assert status == {"status": "READY_NOT_STARTED", "transitions": 0}
    assert not (root / "sac").exists()


def test_tiny_runner_execution_smoke(tmp_path):
    root = tmp_path / "smoke"
    subprocess.run([
        sys.executable, "scripts/run_deployable_observation_v2.py", "--start",
        "--output-dir", str(root), "--battery-capacity", "100",
        "--capacity-source", "unit_test_only", "--device", "cpu",
        "--num-envs", "1", "--rollout-steps", "8", "--target-steps", "16",
        "--checkpoint-steps", "8", "--batch-size", "8", "--buffer-size", "64",
        "--learning-starts", "8", "--horizon", "8", "--obstacles", "0",
    ], check=True, timeout=90, capture_output=True, text=True)
    status = json.loads((root / "status.json").read_text())
    assert status["status"] == "TRAINING_COMPLETE_AWAITING_USER"
    assert status["transitions"] == 16 and not status["automatic_evaluation"]
    assert (root / "sac/latest.json").exists()
    assert not (root / "ERROR.json").exists()
