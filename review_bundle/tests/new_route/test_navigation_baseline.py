from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC

from envs.navigation import NavigationEnv, NavigationRewardConfig


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "artifacts/phase1_sb3_sac_1m_gpu"


def test_deterministic_seed_contract() -> None:
    first = NavigationEnv(max_episode_steps=5)
    second = NavigationEnv(max_episode_steps=5)
    first_observation, _ = first.reset(seed=91)
    second_observation, _ = second.reset(seed=91)
    np.testing.assert_array_equal(first_observation, second_observation)
    action = np.array([0.25, -0.5, 0.75], dtype=np.float32)
    first_step = first.step(action)
    second_step = second.step(action)
    np.testing.assert_array_equal(first_step[0], second_step[0])
    assert first_step[1] == second_step[1]
    assert first_step[4]["reward_components"] == second_step[4]["reward_components"]


def test_old_baseline_observation_action_reward_contract() -> None:
    config = json.loads((ARTIFACT / "seed0/config.json").read_text(encoding="utf-8"))
    reward = NavigationRewardConfig(
        distance_potential_scale=config["distance_potential_scale"],
        gamma=config["gamma"],
        velocity_toward_goal_weight=config["velocity_reward_weight"],
        time_cost=config["time_cost"],
        task_completion_reward=config["completion_reward"],
        collision_penalty=config["collision_penalty"],
        energy_cost_weight=config["energy_cost_weight"],
        backup_intervention_cost=config["backup_intervention_cost"],
    )
    environment = NavigationEnv(
        config["scenario"],
        max_episode_steps=5,
        navigation_energy_capacity=config["navigation_energy_capacity"],
        goal_radius=config["goal_radius"],
        minimum_goal_separation=config["minimum_goal_separation"],
        sampling_margin=config["sampling_margin"],
        reward_config=reward,
    )
    assert list(environment.observation_fields) == config["observation_fields"]
    assert environment.observation_space.shape == (config["observation_dimension"],)
    np.testing.assert_array_equal(environment.action_space.low, config["action_space_low"])
    np.testing.assert_array_equal(environment.action_space.high, config["action_space_high"])
    np.testing.assert_allclose(environment.config.a_max, config["physical_acceleration_limit"])
    np.testing.assert_allclose(environment.config.v_max, config["velocity_limit"])
    assert environment.config.dt == config["dt"]


def test_all_retained_checkpoints_load_and_step() -> None:
    for seed in range(3):
        checkpoint = ARTIFACT / f"seed{seed}/checkpoint_step_1000000.zip"
        model = SAC.load(checkpoint, device="cpu")
        environment = NavigationEnv(max_episode_steps=2)
        observation, _ = environment.reset(seed=100)
        action, _ = model.predict(observation, deterministic=True)
        next_observation, reward, terminated, _, _ = environment.step(action)
        assert next_observation.shape == (77,)
        assert np.isfinite(reward)
        assert not terminated


def test_frozen_checkpoint_is_dimensionally_compatible_with_target_open_world() -> None:
    checkpoint = ARTIFACT / "seed0/checkpoint_step_1000000.zip"
    model = SAC.load(checkpoint, device="cpu")
    environment = NavigationEnv("target_open_16x16.json", max_episode_steps=2)
    observation, _ = environment.reset(seed=501)
    action, _ = model.predict(observation, deterministic=True)
    next_observation, reward, terminated, _, _ = environment.step(action)
    assert observation.shape == next_observation.shape == (77,)
    assert action.shape == (3,)
    assert np.isfinite(reward)
    assert not terminated
