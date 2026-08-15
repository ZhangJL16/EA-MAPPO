from __future__ import annotations

import numpy as np
import pytest

from envs.UAVEnergyDelivery import UAVEnv
from envs.UAVEnergyDeliverySAC import (
    OnlineScalarTDEnergyEstimator,
    SACTrainingPhase,
    UAVEnergyDeliverySACEnv,
    UAVEnv as SACUAVEnv,
)


class _ConstantEnergyEstimator:
    def __init__(self, prediction: float) -> None:
        self.prediction = float(prediction)
        self.observed = []

    def predict(self, charger_observation: np.ndarray, charger_action: np.ndarray) -> float:
        assert charger_observation.shape == (7,)
        assert charger_action.shape == (3,)
        return self.prediction

    def observe_transition(self, *transition) -> float:
        self.observed.append(transition)
        return 0.0


def _zero_charger_policy(observation: np.ndarray) -> np.ndarray:
    assert observation.shape == (7,)
    return np.zeros(3, dtype=np.float32)


def test_default_contract_and_old_environment_are_independent() -> None:
    old = UAVEnv(dim_actions=3)
    new = UAVEnergyDeliverySACEnv()
    assert old.length == 4.0
    assert old.width == 4.0
    assert old.time_step == 0.4
    assert old.v_max == 0.16
    assert old.a_max == 0.05
    assert new.length == 4000.0
    assert new.width == 4000.0
    assert new.time_step == 0.2
    assert new.num_agents == 1
    assert new.num_obstacle == 0
    assert new.observation_space.shape == (7,)
    assert new.action_space.shape == (3,)
    assert SACUAVEnv is UAVEnergyDeliverySACEnv


def test_observation_is_active_goal_relative_and_has_no_energy_identity() -> None:
    environment = UAVEnergyDeliverySACEnv()
    observation, _ = environment.reset(
        seed=4,
        options={
            "start_position": np.array([1000.0, 1000.0, 2.0]),
            "task_point": np.array([1100.0, 1000.0, 2.0]),
        },
    )
    np.testing.assert_allclose(observation[:3], 0.0)
    np.testing.assert_allclose(observation[3:6], [1.0, 0.0, 0.0])
    assert observation[6] == 1.0
    before = environment.charger_relative_observation()
    environment.agent.energy = 1.0
    environment.current_task_point = np.array([3500.0, 3500.0, 2.0], dtype=np.float32)
    after = environment.charger_relative_observation()
    np.testing.assert_allclose(before, after)


def test_task_completion_preserves_position_velocity_and_continues_episode() -> None:
    environment = UAVEnergyDeliverySACEnv(minimum_task_distance=80.0)
    start = np.array([1000.0, 1000.0, 2.0], dtype=np.float32)
    velocity = np.array([1.0, -0.5, 0.0], dtype=np.float32)
    environment.reset(
        seed=8,
        options={
            "start_position": start,
            "start_velocity": velocity,
            "task_point": np.array([1100.0, 1000.0, 2.0]),
        },
    )
    environment.current_task_point = start.copy()
    expected_position = start + velocity * environment.time_step
    _, _, terminated, truncated, info = environment.step(np.zeros(3, dtype=np.float32))
    assert info["task_completed_now"] is True
    assert info["tasks_completed"] == 1
    assert not terminated
    assert not truncated
    np.testing.assert_allclose(environment.agent.pos, expected_position)
    np.testing.assert_allclose(environment.agent.vel, velocity)
    assert np.linalg.norm(environment.current_task_point - environment.agent.pos) >= environment.minimum_task_distance


def test_sampler_returns_legal_well_separated_task_points() -> None:
    environment = UAVEnergyDeliverySACEnv(minimum_task_distance=250.0)
    for seed in range(100):
        environment.reset(seed=seed)
        task = environment.current_task_point
        assert environment.safe_radius <= task[0] <= environment.length - environment.safe_radius
        assert environment.safe_radius <= task[1] <= environment.width - environment.safe_radius
        assert environment.safe_radius <= task[2] <= environment.height - environment.safe_radius
        assert np.linalg.norm(task - environment.agent.pos) >= 250.0


def test_anisotropic_action_and_velocity_limits() -> None:
    environment = UAVEnergyDeliverySACEnv(height=1000.0)
    environment.reset(
        seed=1,
        options={
            "start_position": np.array([2000.0, 2000.0, 500.0]),
            "task_point": np.array([3000.0, 3000.0, 500.0]),
        },
    )
    _, _, _, _, info = environment.step(np.ones(3, dtype=np.float32))
    acceleration = np.asarray(info["physical_acceleration"])
    assert np.linalg.norm(acceleration[:2]) == pytest.approx(3.0)
    assert acceleration[2] == pytest.approx(2.0)
    for _ in range(100):
        environment.step(np.ones(3, dtype=np.float32))
    assert np.linalg.norm(environment.agent.vel[:2]) <= 12.0 + 1e-5
    assert abs(float(environment.agent.vel[2])) <= 3.0 + 1e-5


def test_phase_one_does_not_consume_operational_energy() -> None:
    environment = UAVEnergyDeliverySACEnv(phase=SACTrainingPhase.NAVIGATION)
    environment.reset(seed=3)
    initial = environment.agent.energy
    for _ in range(10):
        _, _, terminated, _, _ = environment.step(np.zeros(3, dtype=np.float32))
        assert not terminated
    assert environment.agent.energy == initial


def test_phase_two_switch_is_absorbing_and_charger_hit_ends_cycle() -> None:
    estimator = _ConstantEnergyEstimator(prediction=100.0)
    environment = UAVEnergyDeliverySACEnv(
        operational_energy_capacity=100.0,
        energy_cost_per_step=0.1,
    )
    environment.enable_phase_two(
        energy_estimator=estimator,
        charger_action_provider=_zero_charger_policy,
        reserve=1.0,
    )
    observation, info = environment.reset(seed=5)
    assert observation.shape == (7,)
    assert info["mode"] == "CHARGER_COMMITTED"
    assert info["task_to_charger_count"] == 1
    _, _, terminated, _, info = environment.step(np.zeros(3, dtype=np.float32))
    assert terminated
    assert info["mode"] == "CHARGER_COMMITTED"
    assert info["task_to_charger_count"] == 1
    assert info["battery_cycle_end_reason"] == "charger_reached"
    assert len(estimator.observed) == 1


def test_energy_exhaustion_ends_phase_two_cycle() -> None:
    estimator = _ConstantEnergyEstimator(prediction=0.0)
    environment = UAVEnergyDeliverySACEnv(
        operational_energy_capacity=0.1,
        energy_cost_per_step=0.1,
    )
    environment.enable_phase_two(
        energy_estimator=estimator,
        charger_action_provider=_zero_charger_policy,
        reserve=0.0,
    )
    environment.reset(seed=6)
    _, _, terminated, _, info = environment.step(np.zeros(3, dtype=np.float32))
    assert terminated
    assert info["battery_cycle_end_reason"] == "energy_exhausted"


def test_phase_two_cycle_is_not_time_limit_truncated() -> None:
    estimator = _ConstantEnergyEstimator(prediction=0.0)
    environment = UAVEnergyDeliverySACEnv(
        episode_limit=1,
        operational_energy_capacity=1.0,
        energy_cost_per_step=0.1,
    )
    environment.enable_phase_two(
        energy_estimator=estimator,
        charger_action_provider=_zero_charger_policy,
        reserve=0.0,
    )
    environment.reset(seed=7)
    _, _, terminated, truncated, _ = environment.step(np.zeros(3, dtype=np.float32))
    assert not terminated
    assert not truncated


def test_online_td_estimator_uses_terminal_cost_without_bootstrap() -> None:
    estimator = OnlineScalarTDEnergyEstimator(
        hidden_dim=16,
        learning_rate=2e-3,
        batch_size=8,
        replay_capacity=256,
        learning_starts=8,
        updates_per_transition=4,
        target_tau=0.1,
        initial_output=0.5,
        seed=2,
    )
    state = np.zeros(7, dtype=np.float32)
    action = np.zeros(3, dtype=np.float32)
    for _ in range(120):
        estimator.observe_transition(state, action, 2.0, state, action, True)
    assert estimator.predict(state, action) == pytest.approx(2.0, abs=0.15)


def test_trajectory_logging_and_rendering() -> None:
    environment = UAVEnergyDeliverySACEnv(render_mode="rgb_array")
    environment.reset(seed=10)
    environment.step(np.zeros(3, dtype=np.float32))
    log = environment.get_trajectory_log()
    assert len(log) == 1
    assert log[0]["position"].shape == (3,)
    image_xy = environment.render(view="xy")
    image_3d = environment.render(view="3d")
    assert image_xy.ndim == 3 and image_xy.shape[-1] == 4
    assert image_3d.ndim == 3 and image_3d.shape[-1] == 4
    environment.close()
