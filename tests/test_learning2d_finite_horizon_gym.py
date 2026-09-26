"""A fixed task horizon must not trigger PPO time-limit bootstrapping."""

import json

from stable_baselines3.common.vec_env import DummyVecEnv

from dual_constraint_2d.calibration import DEFAULT_OUTPUT
from learning2d.finite_horizon_gym import FiniteHorizonGym


def _environment():
    calibration = json.loads((DEFAULT_OUTPUT / "calibration.json").read_text())
    env = FiniteHorizonGym(
        (15,), calibration["capacity_synthetic_energy"],
        calibration["full_charge_seconds"], shielded=False,
    )
    env.reset(seed=15, options={"map_id": 15})
    return env


def test_fixed_window_is_terminated_and_observation_contains_remaining_time():
    env = _environment()
    try:
        env.env.time_s = env.horizon_s - 0.04929987730133689
        obs, _, terminated, truncated, info = env.step(4)
        assert terminated and not truncated
        assert info["finite_horizon_terminal"] is True
        assert info["failure_reason"] is None
        assert info["time_s"] == env.horizon_s
        assert obs[9] == 0.0
    finally:
        env.close()


def test_energy_depletion_remains_a_true_failure_terminal():
    env = _environment()
    try:
        env.env.energy = 0.001
        _, _, terminated, truncated, info = env.step(4)
        assert terminated and not truncated
        assert info["failure_reason"] == "depletion"
        assert "finite_horizon_terminal" not in info
    finally:
        env.close()


def test_sb3_does_not_classify_task_horizon_as_timeout():
    vector_env = DummyVecEnv([_environment])
    try:
        vector_env.envs[0].env.time_s = vector_env.envs[0].horizon_s - 0.01
        _, _, dones, infos = vector_env.step([4])
        assert bool(dones[0])
        assert infos[0]["TimeLimit.truncated"] is False
        assert infos[0]["finite_horizon_terminal"] is True
    finally:
        vector_env.close()
