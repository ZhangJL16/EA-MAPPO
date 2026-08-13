from __future__ import annotations

import numpy as np

from envs.navigation import NavigationEnv, OperationalEnergyConfig, OperationalEnergyWrapper


def test_operational_energy_does_not_change_navigation_observation_contract() -> None:
    baseline = NavigationEnv(max_episode_steps=3)
    wrapped = OperationalEnergyWrapper(
        NavigationEnv(max_episode_steps=3),
        OperationalEnergyConfig(capacity=0.5, enforce_exhaustion=False),
    )
    baseline_observation, _ = baseline.reset(seed=15)
    wrapped_observation, wrapped_info = wrapped.reset(seed=15)
    np.testing.assert_array_equal(wrapped_observation, baseline_observation)
    assert wrapped_observation.shape == (77,)
    assert wrapped_observation[12] == 1.0
    assert wrapped_info["operational_energy_remaining"] == 0.5
    action = np.array([0.3, -0.2, 0.1], dtype=np.float32)
    baseline_step = baseline.step(action)
    wrapped_step = wrapped.step(action)
    np.testing.assert_array_equal(wrapped_step[0], baseline_step[0])
    assert wrapped_step[4]["operational_soc"] < 1.0
    assert wrapped_step[0][12] > 0.99
    assert wrapped_step[4]["navigation_energy_is_compatibility_only"]


def test_operational_energy_can_enforce_target_domain_exhaustion() -> None:
    wrapped = OperationalEnergyWrapper(
        NavigationEnv(max_episode_steps=3),
        OperationalEnergyConfig(capacity=0.001, enforce_exhaustion=True),
    )
    wrapped.reset(seed=20)
    _, _, terminated, _, info = wrapped.step(np.zeros(3, dtype=np.float32))
    assert terminated
    assert info["operational_energy_exhausted"]
    assert info["operational_energy_remaining"] == 0.0


def test_random_station_and_counterfactual_goal_preserve_active_task_goal() -> None:
    environment = NavigationEnv(max_episode_steps=3)
    task_observation, info = environment.reset(seed=31, options={"randomize_station": True})
    original_goal = environment.goal.copy()
    assert np.linalg.norm(info["station_position"] - original_goal) >= environment.minimum_goal_separation
    charger_observation = environment.observation_for_goal(environment.station_position)
    np.testing.assert_allclose(charger_observation[6:9], charger_observation[9:12])
    np.testing.assert_array_equal(environment.goal, original_goal)
    np.testing.assert_allclose(task_observation[6:9], original_goal / environment.config.world_size)


def test_target_16x16_scenario_retains_sac_dimensions() -> None:
    environment = NavigationEnv("target_open_16x16.json", max_episode_steps=3)
    observation, info = environment.reset(seed=8)
    np.testing.assert_allclose(environment.config.world_size, np.array([16.0, 16.0, 2.0]))
    assert observation.shape == (77,)
    assert environment.action_space.shape == (3,)
    assert info["energy_semantics"] == "navigation_baseline_nonterminating_large_budget"
