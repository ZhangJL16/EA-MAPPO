"""Focused auto-reset, frozen semantics, and full training continuation checks."""

import json

import numpy as np
import pytest
import torch
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv

from experiments.directional_navigation.recovery import RecoveryCohort
from experiments.directional_navigation.standard_baselines import (
    ContinuousRecovery,
    load_checkpoint,
    save_checkpoint,
)


def make_env():
    return ContinuousRecovery(seed_start=120001, seed_stride=1, horizon=7, obstacles=0)


def test_auto_reset_retains_locked_reward_and_contact_transition():
    old = RecoveryCohort(horizon=2, obstacles=0)
    new = ContinuousRecovery(seed_start=710, seed_stride=1, horizon=2, obstacles=0)
    try:
        # Start legally inside the map, close enough that the commanded
        # outward motion contacts the boundary during the policy step.
        options = {"start_position": np.array([1.51, 1000., 100.], np.float32),
                   "start_velocity": np.array([-20., 0., 0.], np.float32),
                   "task_point": np.array([1500., 1000., 100.], np.float32)}
        old.reset(seed=709, options=options)
        new.reset(seed=709, options=options)
        for _ in range(2):
            a, b = old.step(np.array([-1., 0., 0.])), new.step(np.array([-1., 0., 0.]))
            np.testing.assert_array_equal(a[0], b[0])
            assert a[1:] == b[1:]
        new.reset()
        assert new.episode_seed == 710 and not new.parked and new.steps == 0
        assert new.collision_count == 0
    finally:
        old.close()
        new.close()


@pytest.mark.parametrize("algorithm", ["sac", "ppo"])
def test_checkpoint_matches_uninterrupted_next_update(tmp_path, algorithm):
    torch.set_num_threads(1)
    vec = DummyVecEnv([make_env])
    restored_vec = DummyVecEnv([make_env])
    common = dict(policy="MlpPolicy", env=vec, seed=32, device="cpu",
                  learning_rate=3e-4, batch_size=8, policy_kwargs={"net_arch": [16, 16]})
    model = (SAC(**common, buffer_size=100, learning_starts=8)
             if algorithm == "sac" else PPO(**common, n_steps=16, n_epochs=2))
    vec._reset_seeds()
    try:
        model.learn(32, reset_num_timesteps=False)
        cp = save_checkpoint(model, vec, tmp_path, {"test": algorithm}, {"marker": 3})
        obs = model._last_obs.copy()
        worker_steps = vec.get_attr("steps")
        model.learn(16, reset_num_timesteps=False)
        expected = {k: v.clone() for k, v in model.policy.state_dict().items()}
        expected_obs = model._last_obs.copy()
        restored, records = load_checkpoint(algorithm, restored_vec, tmp_path,
                                            {"test": algorithm}, "cpu")
        assert records == {"marker": 3}
        assert restored_vec.get_attr("steps") == worker_steps
        np.testing.assert_array_equal(restored._last_obs, obs)
        restored.learn(16, reset_num_timesteps=False)
        np.testing.assert_array_equal(restored._last_obs, expected_obs)
        for k, v in restored.policy.state_dict().items():
            torch.testing.assert_close(v, expected[k], rtol=0, atol=0)
        if algorithm == "sac":
            torch.testing.assert_close(model.log_ent_coef, restored.log_ent_coef, rtol=0, atol=0)
            assert model.replay_buffer.pos == restored.replay_buffer.pos
        assert restored.num_timesteps == model.num_timesteps
        assert json.loads((cp / "metadata.json").read_text())["transitions"] == 32
        with pytest.raises(ValueError, match="contract"):
            load_checkpoint(algorithm, restored_vec, tmp_path, {}, "cpu")
    finally:
        vec.close()
        restored_vec.close()
