from __future__ import annotations

import json
import inspect
from collections import deque
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from stable_baselines3 import SAC

from envs.UAVEnergyDelivery import (
    UAVEnv,
    update_lasers_to_boundary,
    update_lasers_to_obstacle,
)
from envs.UAVEnergyDeliverySAC import (
    ENERGY_QUANTILES,
    ENERGY_UNIT,
    GoalConditionedQuantileTDEnergyEstimator,
    SACTrainingPhase,
    StaticCylinderObstacle,
    UAVEnergyDeliverySACEnv,
)
from review_bundle.envs.navigation.state import NavigationState
from review_bundle.envs.navigation.telemetry_cost import TelemetryCostConfig, TelemetryCostModel
from review_bundle.safety.switching import (
    DistanceEnergyReturnManager,
    FixedSOCThresholdReturnManager,
    QuantileEnergyReturnManager,
    ReturnDecisionContext,
    ReturnManagerDecision,
    SortieMode,
)
from scripts.train_uav_energy_delivery_sac import (
    HeuristicGoalPolicy,
    NavigationTask,
    aggregate_goal_evaluations,
    calculate_battery_capacities,
    evaluate_energy_tasks,
    evaluate_navigation_tasks,
    environment_from_args,
    freeze_navigation_and_start_td,
    generate_navigation_curves,
    generate_stratified_navigation_tasks,
    load_resumed_phase1,
    make_navigation_vec_env,
    navigation_energy_gate_passed,
    navigation_observation_dim,
    navigation_safety_gate_passed,
    parse_args,
    run_battery_calibration,
    run_battery_validation,
    run_energy_exhaustion_smoke,
    run_phase2_fixed_budget,
    run_phase2_state_machine_smoke,
    train_navigation_fixed_budget,
)
from scripts.evaluate_td_flight_gif import collect_td_flight, render_td_flight_gif
from scripts.evaluate_energy_managed_scene_gif import (
    audit_managed_lifecycle,
    collect_energy_managed_scene,
    render_energy_managed_gif,
)
from experiments.uav_energy_parallel import _worker_main


class RecordingEstimator:
    def __init__(self, value: float) -> None:
        self.value = float(value)
        self.replay: deque[object] = deque()
        self.update_count = 0

    def predict_quantiles(self, energy_state: np.ndarray, action: np.ndarray) -> np.ndarray:
        del energy_state, action
        return np.full(4, self.value, dtype=np.float64)

    def predict(self, energy_state: np.ndarray, action: np.ndarray, quantile: float = 0.95) -> float:
        del energy_state, action, quantile
        return self.value

    def observe_transition(self, *args, **kwargs):
        self.replay.append((args, kwargs))
        return None


def test_parallel_worker_keeps_backward_compatible_optional_estimator_argument() -> None:
    parameter = inspect.signature(_worker_main).parameters[
        "energy_estimator_checkpoint"
    ]
    assert parameter.default is None


class ZeroPredictPolicy:
    def predict(self, observation: np.ndarray, deterministic: bool = True):
        del observation, deterministic
        return np.zeros(3, dtype=np.float32), None


class ConstantActionPolicy:
    def __init__(self, action: np.ndarray) -> None:
        self.action = np.asarray(action, dtype=np.float32)

    def predict(self, observation: np.ndarray, deterministic: bool = True):
        del observation, deterministic
        return self.action.copy(), None


def zero_policy(observation: np.ndarray) -> np.ndarray:
    del observation
    return np.zeros(3, dtype=np.float32)


def test_default_contract_and_legacy_environment_are_independent() -> None:
    old = UAVEnv(dim_actions=3)
    new = UAVEnergyDeliverySACEnv()
    assert (old.length, old.width, old.time_step, old.v_max, old.a_max) == (4.0, 4.0, 0.4, 0.16, 0.05)
    assert (new.length, new.width, new.height) == (4000.0, 4000.0, 400.0)
    assert (new.policy_dt, new.physics_dt, new.physics_substeps_per_policy_step) == (0.2, 0.05, 4)
    assert (new.horizontal_v_max, new.vertical_v_max) == (20.0, 5.0)
    assert (new.horizontal_a_max, new.vertical_a_max) == (5.0, 3.0)
    assert new.observation_space.shape == (7,)
    assert new.action_space.shape == (3,)
    assert new.operational_energy_capacity is None
    assert new.lidar_enabled is False
    assert new.cbf_enabled is False
    assert (new.lidar_max_range, new.lidar_frequency) == (100.0, 10.0)
    assert (new.lidar_horizontal_sectors, new.lidar_vertical_sectors) == (128, 8)
    assert new.cbf_frequency == 20.0
    assert (new.obstacle_radius_min, new.obstacle_radius_max) == (50.0, 120.0)


def test_legacy_32_ray_contract_remains_available_without_action_filter() -> None:
    environment = UAVEnergyDeliverySACEnv(
        lidar_enabled=True,
        lidar_max_range=250.0,
        lidar_horizontal_sectors=32,
        lidar_vertical_sectors=1,
        num_obstacles=6,
    )
    observation, info = environment.reset(seed=700)
    energy_state = environment.energy_state_for_goal(environment.current_task_point)
    assert observation.shape == (71,)
    assert energy_state.shape == (71,)
    assert environment.cbf_enabled is False
    assert len(environment.obstacles) == 6
    np.testing.assert_allclose(observation[7:], energy_state[7:])
    assert np.all((0.0 <= observation[7:39]) & (observation[7:39] <= 1.0))
    assert set(np.unique(observation[39:])).issubset({0.0, 1.0})
    assert info["lidar_enabled"] is True
    assert info["num_obstacles"] == 6
    environment.close()


def test_static_obstacles_are_drawn_in_2d_and_3d() -> None:
    import matplotlib.pyplot as plt

    environment = UAVEnergyDeliverySACEnv(num_obstacles=3, render_mode="rgb_array")
    environment.reset(seed=123)

    figure_2d, axis_2d = plt.subplots()
    environment._render_2d(axis_2d)
    assert len(axis_2d.patches) == 3
    plt.close(figure_2d)

    figure_3d = plt.figure()
    axis_3d = figure_3d.add_subplot(111, projection="3d")
    collections_before = len(axis_3d.collections)
    environment._render_3d(axis_3d)
    assert len(axis_3d.collections) - collections_before >= 3
    plt.close(figure_3d)
    environment.close()


def test_obstacle_collision_is_penalized_but_nonterminal() -> None:
    environment = UAVEnergyDeliverySACEnv(
        lidar_enabled=True,
        lidar_horizontal_sectors=32,
        lidar_vertical_sectors=1,
        minimum_task_distance=5.0,
    )
    start = np.array([1000.0, 1000.0, 100.0], dtype=np.float32)
    environment.reset(
        seed=701,
        options={
            "start_position": start,
            "start_velocity": np.array([20.0, 0.0, 0.0], dtype=np.float32),
            "task_point": np.array([1500.0, 1000.0, 100.0], dtype=np.float32),
        },
    )
    environment.obstacles = [
        StaticCylinderObstacle(np.array([1001.0, 1000.0], dtype=np.float32), 1.0)
    ]
    environment._update_lidar()
    _, reward, terminated, truncated, info = environment.step(
        np.zeros(3, dtype=np.float32)
    )
    assert info["obstacle_collision"] is True
    assert info["reward_components"]["obstacle_penalty_component"] == pytest.approx(-1.2)
    assert reward > -10.0
    assert not terminated and not truncated
    environment.close()


def test_obstacle_training_cli_builds_71d_direct_sac_environment() -> None:
    args = parse_args(
        [
            "--output-dir",
            "/tmp/not-used",
            "--smoke",
            "--lidar-enabled",
            "--lidar-sectors",
            "32",
            "--lidar-range",
            "250",
            "--num-obstacles",
            "24",
        ]
    )
    assert args.legacy_lidar_alias_used is True
    assert args.hocbf_enabled is False
    assert args.buffer_size == 1_000_000
    assert navigation_observation_dim(args) == 71
    environment = environment_from_args(args, phase=SACTrainingPhase.NAVIGATION)
    observation, _ = environment.reset(seed=702)
    assert observation.shape == (71,)
    assert len(environment.obstacles) == 24
    assert environment.lidar_max_range == pytest.approx(250.0)
    assert environment.cbf_enabled is False
    environment.close()


def test_formal_obstacle_contract_uses_1024_ray_3d_lidar_and_hocbf() -> None:
    args = parse_args(
        [
            "--output-dir",
            "/tmp/not-used",
            "--smoke",
            "--lidar-enabled",
            "--num-obstacles",
            "4",
        ]
    )
    assert args.lidar_horizontal_sectors == 128
    assert args.lidar_vertical_sectors == 8
    assert args.lidar_sectors == 1024
    assert args.hocbf_enabled is True
    assert args.buffer_size == 200_000
    assert navigation_observation_dim(args) == 2055
    environment = environment_from_args(args, phase=SACTrainingPhase.NAVIGATION)
    observation, info = environment.reset(seed=704)
    assert observation.shape == (2055,)
    assert environment._lidar_packet is not None
    assert environment._lidar_packet.directions.shape == (1024, 3)
    assert info["lidar_num_sectors"] == 1024
    assert info["cbf_enabled"] is True
    environment.close()


def test_hocbf_leaves_safe_nominal_action_unchanged() -> None:
    environment = UAVEnergyDeliverySACEnv(
        lidar_enabled=True,
        cbf_enabled=True,
        minimum_task_distance=5.0,
    )
    environment.reset(
        seed=705,
        options={
            "start_position": np.array([2000.0, 2000.0, 200.0], dtype=np.float32),
            "task_point": np.array([2100.0, 2000.0, 200.0], dtype=np.float32),
        },
    )
    _, _, _, _, info = environment.step(np.zeros(3, dtype=np.float32))
    assert info["hocbf_intervened"] is False
    np.testing.assert_allclose(info["executed_action"], 0.0, atol=1e-7)
    environment.close()


def test_hocbf_does_not_duplicate_physical_speed_saturation() -> None:
    environment = UAVEnergyDeliverySACEnv(
        lidar_enabled=True,
        cbf_enabled=True,
        minimum_task_distance=5.0,
    )
    environment.reset(
        seed=7051,
        options={
            "start_position": np.array([2000.0, 2000.0, 200.0], dtype=np.float32),
            "start_velocity": np.array([20.0, 0.0, 0.0], dtype=np.float32),
            "task_point": np.array([2500.0, 2000.0, 200.0], dtype=np.float32),
        },
    )
    _, _, _, _, info = environment.step(
        np.array([1.0, 0.0, 0.0], dtype=np.float32)
    )
    assert info["hocbf_intervened"] is False
    assert info["executed_action"][0] == pytest.approx(1.0)
    assert environment.agent.vel[0] == pytest.approx(20.0)
    environment.close()


def test_hocbf_uses_reverse_braking_when_current_barrier_state_is_unsafe() -> None:
    environment = UAVEnergyDeliverySACEnv(
        lidar_enabled=True,
        cbf_enabled=True,
        minimum_task_distance=5.0,
    )
    environment.reset(
        seed=706,
        options={
            "start_position": np.array([1000.0, 1000.0, 100.0], dtype=np.float32),
            "start_velocity": np.array([20.0, 0.0, 0.0], dtype=np.float32),
            "task_point": np.array([1500.0, 1000.0, 100.0], dtype=np.float32),
        },
    )
    environment.obstacles = [
        StaticCylinderObstacle(np.array([1045.0, 1000.0], dtype=np.float32), 10.0)
    ]
    environment._update_lidar()
    _, _, terminated, truncated, info = environment.step(
        np.array([1.0, 0.0, 0.0], dtype=np.float32)
    )
    assert info["hocbf_intervened"] is True
    assert info["hocbf_emergency_brake"] is True
    assert info["executed_action"][0] < 0.0
    assert environment.agent.vel[0] < 20.0
    assert info["obstacle_collision"] is False
    assert not terminated and not truncated
    environment.close()


def test_vectorized_lidar_matches_legacy_sensor_geometry() -> None:
    environment = UAVEnergyDeliverySACEnv(
        lidar_enabled=True,
        lidar_horizontal_sectors=32,
        lidar_vertical_sectors=1,
        lidar_max_range=250.0,
    )
    environment.reset(seed=703)
    environment.agent.pos = np.array([1000.0, 1200.0, 100.0], dtype=np.float32)
    environment.obstacles = [
        StaticCylinderObstacle(np.array([1100.0, 1200.0], dtype=np.float32), 30.0),
        StaticCylinderObstacle(np.array([900.0, 1300.0], dtype=np.float32), 45.0),
    ]
    expected = update_lasers_to_boundary(
        environment.agent.pos[:2],
        environment.lidar_max_range,
        environment.lidar_horizontal_sectors,
        environment.length,
        environment.width,
    )
    for obstacle in environment.obstacles:
        expected = np.minimum(
            expected,
            update_lasers_to_obstacle(
                environment.agent.pos[:2],
                obstacle.pos,
                obstacle.radius,
                environment.lidar_max_range,
                environment.lidar_horizontal_sectors,
            ),
        )
    environment._update_lidar()
    np.testing.assert_allclose(environment.agent.lasers, expected, rtol=1e-5, atol=1e-4)
    environment.close()


def test_obstacles_without_lidar_are_rejected_by_training_cli() -> None:
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--output-dir",
                "/tmp/not-used",
                "--num-obstacles",
                "1",
            ]
        )


def test_phase1_reset_is_random_single_task_with_zero_velocity() -> None:
    environment = UAVEnergyDeliverySACEnv()
    starts = []
    for seed in range(4):
        _, _ = environment.reset(seed=seed)
        starts.append(environment.agent.pos.copy())
        np.testing.assert_allclose(environment.agent.vel, 0.0)
        assert np.linalg.norm(environment.current_task_point - environment.agent.pos) >= 100.0
        assert not np.allclose(environment.agent.pos, environment.charger_position)
    assert any(not np.allclose(starts[0], start) for start in starts[1:])


def test_phase1_goal_reached_is_success_terminal() -> None:
    environment = UAVEnergyDeliverySACEnv(minimum_task_distance=5.0)
    start = np.array([1000.0, 1000.0, 100.0], dtype=np.float32)
    environment.reset(
        seed=1,
        options={
            "start_position": start,
            "start_velocity": np.array([20.0, 0.0, 0.0], dtype=np.float32),
            "task_point": start + np.array([5.5, 0.0, 0.0], dtype=np.float32),
        },
    )
    _, _, terminated, truncated, info = environment.step(np.zeros(3, dtype=np.float32))
    assert terminated and not truncated
    assert info["is_success"] is True
    assert info["end_reason"] == "goal_reached"
    assert info["physics_substeps"] == 1
    assert info["transition_dt"] == pytest.approx(0.05)


def test_phase1_task_limit_is_truncation_not_terminal() -> None:
    environment = UAVEnergyDeliverySACEnv(
        max_steps_per_task=1,
        phase1_episode_max_policy_steps=1,
    )
    environment.reset(seed=2)
    _, _, terminated, truncated, info = environment.step(np.zeros(3, dtype=np.float32))
    assert not terminated and truncated
    assert info["is_success"] is False
    assert info["navigation_failure"] is True
    assert info["end_reason"] == "task_step_limit"


def test_boundary_contact_is_nonterminal() -> None:
    environment = UAVEnergyDeliverySACEnv()
    start = np.array([0.5, 1000.0, 100.0], dtype=np.float32)
    environment.reset(
        seed=3,
        options={
            "start_position": start,
            "task_point": np.array([1000.0, 1000.0, 100.0], dtype=np.float32),
        },
    )
    _, _, terminated, truncated, info = environment.step(np.array([-1.0, 0.0, 0.0], dtype=np.float32))
    assert info["boundary_contact"] is True
    assert not terminated and not truncated


def test_policy_step_accumulates_four_substeps_and_hover_costs_energy() -> None:
    environment = UAVEnergyDeliverySACEnv()
    environment.reset(seed=4)
    _, _, _, _, info = environment.step(np.zeros(3, dtype=np.float32))
    config = environment.telemetry_cost_model.config
    expected_substep = environment.physics_dt * (
        config.base_power + config.compute_power + config.communication_power
    )
    assert info["physics_substeps"] == 4
    assert info["transition_dt"] == pytest.approx(0.2)
    assert info["realized_energy_cost"] == pytest.approx(4.0 * expected_substep)
    assert info["realized_energy_cost"] > 0.0
    assert info["energy_unit"] == ENERGY_UNIT


def test_physical_norm_limits_are_enforced() -> None:
    environment = UAVEnergyDeliverySACEnv()
    environment.reset(seed=5)
    _, _, _, _, info = environment.step(np.ones(3, dtype=np.float32))
    acceleration = np.asarray(info["physical_acceleration"])
    assert np.linalg.norm(acceleration[:2]) == pytest.approx(5.0)
    assert acceleration[2] == pytest.approx(3.0)
    for _ in range(100):
        _, _, terminated, truncated, _ = environment.step(np.ones(3, dtype=np.float32))
        if terminated or truncated:
            break
    assert np.linalg.norm(environment.agent.vel[:2]) <= 20.0 + 1e-5
    assert abs(float(environment.agent.vel[2])) <= 5.0 + 1e-5


def test_speed_saturation_uses_realized_not_commanded_acceleration_for_energy() -> None:
    environment = UAVEnergyDeliverySACEnv()
    start = np.array([1000.0, 1000.0, 100.0], dtype=np.float32)
    environment.reset(
        seed=51,
        options={
            "start_position": start,
            "start_velocity": np.array([20.0, 0.0, 0.0], dtype=np.float32),
            "task_point": np.array([2000.0, 1000.0, 100.0], dtype=np.float32),
        },
    )
    _, _, _, _, info = environment.step(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    np.testing.assert_allclose(info["commanded_acceleration"], [5.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(info["realized_acceleration"], 0.0, atol=1e-5)
    np.testing.assert_allclose(info["physical_acceleration"], info["realized_acceleration"])
    state = NavigationState(start, np.array([20.0, 0.0, 0.0]), 1.0, 0.0)
    expected_substep = environment.telemetry_cost_model.realized_cost(
        state,
        np.zeros(3),
        environment.physics_dt,
    )
    assert info["realized_energy_cost"] == pytest.approx(4.0 * expected_substep)


def test_unsaturated_realized_acceleration_matches_velocity_delta() -> None:
    environment = UAVEnergyDeliverySACEnv()
    start = np.array([1000.0, 1000.0, 100.0], dtype=np.float32)
    environment.reset(
        seed=52,
        options={
            "start_position": start,
            "task_point": np.array([2000.0, 1000.0, 100.0], dtype=np.float32),
        },
    )
    velocity_before = environment.agent.vel.copy()
    _, _, _, _, info = environment.step(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    expected_policy_delta = np.array([5.0 * environment.policy_dt, 0.0, 0.0])
    np.testing.assert_allclose(environment.agent.vel - velocity_before, expected_policy_delta, atol=1e-5)
    np.testing.assert_allclose(info["realized_acceleration"], [5.0, 0.0, 0.0], atol=1e-5)


def test_boundary_projection_does_not_create_propulsion_acceleration_energy() -> None:
    environment = UAVEnergyDeliverySACEnv()
    start = np.array([0.5, 1000.0, 100.0], dtype=np.float32)
    environment.reset(
        seed=53,
        options={
            "start_position": start,
            "start_velocity": np.array([-20.0, 0.0, 0.0], dtype=np.float32),
            "task_point": np.array([1000.0, 1000.0, 100.0], dtype=np.float32),
        },
    )
    _, _, terminated, truncated, info = environment.step(np.zeros(3, dtype=np.float32))
    assert info["boundary_contact"] is True
    assert not terminated and not truncated
    np.testing.assert_allclose(info["commanded_acceleration"], 0.0)
    np.testing.assert_allclose(info["realized_acceleration"], 0.0)
    config = environment.telemetry_cost_model.config
    first_power = config.base_power + 20.0 * config.velocity_coefficients[0] + config.compute_power + config.communication_power
    hover_power = config.base_power + config.compute_power + config.communication_power
    expected = environment.physics_dt * (first_power + 3.0 * hover_power)
    assert info["realized_energy_cost"] == pytest.approx(expected)


def test_sac_log_distance_and_td_linear_distance_are_distinct() -> None:
    environment = UAVEnergyDeliverySACEnv()
    start = np.array([100.0, 100.0, 100.0], dtype=np.float32)
    environment.reset(
        seed=6,
        options={"start_position": start, "task_point": start + np.array([100.0, 0.0, 0.0])},
    )
    sac_values = []
    energy_values = []
    for distance in (100.0, 1000.0, 4000.0):
        goal = start + np.array([distance, 0.0, 0.0], dtype=np.float32)
        if goal[0] >= environment.length:
            goal = start + np.array([0.0, distance - 200.0, 0.0], dtype=np.float32)
        sac_values.append(float(environment.sac_observation_for_goal(goal)[6]))
        energy_values.append(float(environment.energy_state_for_goal(goal)[6]))
    assert len(set(sac_values)) == 3
    assert len(set(energy_values)) == 3
    assert sac_values != pytest.approx(energy_values)


def test_observation_excludes_soc_mode_identity_and_absolute_coordinates() -> None:
    environment = UAVEnergyDeliverySACEnv(operational_energy_capacity=10.0)
    start = np.array([1000.0, 1000.0, 100.0], dtype=np.float32)
    goal = np.array([1100.0, 1000.0, 100.0], dtype=np.float32)
    observation, _ = environment.reset(seed=7, options={"start_position": start, "task_point": goal})
    environment.agent.energy = 1.0
    environment.mode = SortieMode.CHARGER_COMMITTED
    same_geometry = environment.sac_observation_for_goal(goal)
    np.testing.assert_allclose(observation, same_geometry)
    assert observation.shape == (7,)


def test_telemetry_cost_orders_hover_cruise_and_acceleration() -> None:
    model = TelemetryCostModel()
    position = np.zeros(3)
    hover = model.realized_cost(NavigationState(position, np.zeros(3), 1.0, 0.0), np.zeros(3), 0.05)
    cruise_10 = model.realized_cost(
        NavigationState(position, np.array([10.0, 0.0, 0.0]), 1.0, 0.0), np.zeros(3), 0.05
    )
    cruise_20 = model.realized_cost(
        NavigationState(position, np.array([20.0, 0.0, 0.0]), 1.0, 0.0), np.zeros(3), 0.05
    )
    accelerating = model.realized_cost(
        NavigationState(position, np.array([10.0, 0.0, 0.0]), 1.0, 0.0), np.array([5.0, 0.0, 0.0]), 0.05
    )
    assert 0.0 < hover < cruise_10 < cruise_20
    assert accelerating > cruise_10


def test_phase2_task_service_is_nonterminal_and_has_no_fake_td_transition() -> None:
    estimator = RecordingEstimator(0.01)
    environment = UAVEnergyDeliverySACEnv(
        minimum_task_distance=5.0,
        operational_energy_capacity=10.0,
    )
    environment.bind_energy_learning(
        energy_estimator=estimator,
        goal_action_provider=zero_policy,
        training_enabled=True,
    )
    environment.enable_phase_two()
    environment.mission_switching_enabled = False
    start = np.array([1000.0, 1000.0, 100.0], dtype=np.float32)
    environment.reset(
        seed=8,
        options={
            "start_position": start,
            "start_velocity": np.array([20.0, 0.0, 0.0], dtype=np.float32),
            "task_point": start + np.array([5.5, 0.0, 0.0], dtype=np.float32),
        },
    )
    _, _, terminated, truncated, info = environment.step(np.zeros(3, dtype=np.float32))
    assert not terminated and not truncated
    assert info["task_completed_now"] is True
    assert info["instantaneous_service_reset"] is True
    np.testing.assert_allclose(environment.agent.vel, 0.0)
    assert len(estimator.replay) == 1
    _, kwargs = estimator.replay[0]
    assert kwargs["censored_goal"] is False


def test_phase2_charger_recharge_is_nonterminal_and_increments_cycle() -> None:
    estimator = RecordingEstimator(1.0)
    environment = UAVEnergyDeliverySACEnv(
        minimum_task_distance=5.0,
        operational_energy_capacity=12.0,
        energy_reserve_fraction=0.10,
    )
    environment.bind_energy_learning(
        energy_estimator=estimator,
        goal_action_provider=zero_policy,
        training_enabled=False,
    )
    environment.enable_phase_two()
    charger = environment.charger_position
    environment.reset(
        seed=9,
        options={
            "start_position": charger + np.array([5.5, 0.0, 0.0], dtype=np.float32),
            "start_velocity": np.array([-20.0, 0.0, 0.0], dtype=np.float32),
            "task_point": charger + np.array([120.0, 0.0, 0.0], dtype=np.float32),
        },
    )
    environment.mode = SortieMode.CHARGER_COMMITTED
    environment.agent.goal = charger.copy()
    environment.agent.energy = 4.0
    environment._start_goal_trajectory(charger)
    _, _, terminated, truncated, info = environment.step(np.zeros(3, dtype=np.float32))
    assert not terminated and not truncated
    assert info["charger_reached_now"] is True
    assert info["battery_cycle_end"] is True
    assert environment.agent.energy == pytest.approx(12.0)
    np.testing.assert_allclose(environment.agent.vel, 0.0)
    assert environment.mode is SortieMode.TASK
    assert environment.battery_cycle_id == 1
    assert environment.battery_cycles_completed == 1
    assert info["battery_cycle_record"]["segment_type"] == "completed_recharge_cycle"


def test_energy_exhaustion_is_failure_terminal() -> None:
    environment = UAVEnergyDeliverySACEnv(operational_energy_capacity=0.001)
    environment.enable_battery_validation()
    environment.reset(seed=10)
    _, _, terminated, truncated, info = environment.step(np.zeros(3, dtype=np.float32))
    assert terminated and not truncated
    assert info["remaining_energy"] <= 0.0
    assert info["end_reason"] == "energy_exhausted"
    assert info["charger_reached_now"] is False


def test_phase2_emergency_limit_is_only_truncation() -> None:
    environment = UAVEnergyDeliverySACEnv(
        operational_energy_capacity=100.0,
        phase2_episode_limit=1,
    )
    environment.enable_battery_validation()
    environment.reset(seed=11)
    _, _, terminated, truncated, info = environment.step(np.zeros(3, dtype=np.float32))
    assert not terminated and truncated
    assert info["end_reason"] == "episode_emergency_step_guard"


def test_quantile_td_contract_order_gamma_and_terminal_target() -> None:
    with pytest.raises(ValueError, match="gamma_energy=1.0"):
        GoalConditionedQuantileTDEnergyEstimator(gamma_energy=0.99)
    estimator = GoalConditionedQuantileTDEnergyEstimator(
        hidden_dim=16,
        learning_rate=5e-3,
        batch_size=8,
        replay_capacity=128,
        learning_starts=8,
        updates_per_transition=4,
        initial_base=0.5,
        initial_increment=0.05,
        seed=12,
    )
    state = np.zeros(7, dtype=np.float32)
    action = np.zeros(3, dtype=np.float32)
    for _ in range(300):
        estimator.observe_transition(state, action, 2.0, state, state, action, True, "TASK")
    quantiles = estimator.predict_quantiles(state, action)
    assert tuple(estimator.quantile_levels) == ENERGY_QUANTILES
    assert np.all(np.diff(quantiles) >= 0.0)
    assert quantiles[0] == pytest.approx(2.0, abs=0.4)


def test_td_recomputes_next_action_from_current_policy_provider() -> None:
    calls = []
    estimator = GoalConditionedQuantileTDEnergyEstimator(
        hidden_dim=8,
        batch_size=2,
        replay_capacity=16,
        learning_starts=2,
        seed=13,
    )

    def provider(observations: np.ndarray) -> np.ndarray:
        calls.append(observations.copy())
        return np.full((len(observations), 3), 0.25, dtype=np.float32)

    estimator.set_next_action_provider(provider)
    state = np.zeros(7, dtype=np.float32)
    action = np.zeros(3, dtype=np.float32)
    estimator.observe_transition(state, action, 1.0, state, state, action, False, "TASK")
    estimator.observe_transition(state, action, 1.0, state, state, action, False, "TASK")
    assert calls and calls[-1].shape == (2, 7)


def test_lidar_conditioned_energy_td_uses_matching_71d_state() -> None:
    estimator = GoalConditionedQuantileTDEnergyEstimator(
        state_dim=71,
        hidden_dim=8,
        batch_size=2,
        replay_capacity=16,
        learning_starts=2,
        seed=130,
    )
    state = np.zeros(71, dtype=np.float32)
    action = np.zeros(3, dtype=np.float32)
    estimator.observe_transition(state, action, 1.0, state, state, action, True, "TASK")
    estimator.observe_transition(state, action, 1.0, state, state, action, True, "TASK")
    assert estimator.update_count == 1
    payload = estimator.checkpoint_payload()
    assert payload["state_dim"] == 71
    assert "lidar_normalized32" in payload["energy_observation"]
    assert "lidar_valid_mask32" in payload["energy_observation"]


def test_mission_composition_and_commitment_are_correct_and_irreversible() -> None:
    estimator = RecordingEstimator(4.0)
    environment = UAVEnergyDeliverySACEnv(operational_energy_capacity=10.0, energy_reserve_fraction=0.10)
    environment.bind_energy_learning(
        energy_estimator=estimator,
        goal_action_provider=zero_policy,
        training_enabled=False,
    )
    environment.reset(seed=14)
    estimate = environment.mission_energy_estimate()
    assert estimate.task_q95 == 4.0
    assert estimate.return_after_task_q95 == 4.0
    assert estimate.mission_q95_composition == 8.0
    assert estimate.mission_upper_bound_semantics == (
        "sum_of_component_q95_not_joint_mission_q95"
    )
    assert estimate.mission_nominal_coverage_lower_bound is None
    assert estimate.return_now_q95 == 4.0
    environment.agent.energy = 9.0
    assert environment._refresh_mission_decision() is True
    assert environment.mode is SortieMode.CHARGER_COMMITTED
    environment.agent.energy = 10.0
    assert environment._refresh_mission_decision() is False
    assert environment.mode is SortieMode.CHARGER_COMMITTED


def test_quantile_return_manager_reproduces_legacy_two_boundary_rule() -> None:
    manager = QuantileEnergyReturnManager()
    common = dict(
        mode=SortieMode.TASK,
        battery_capacity=10.0,
        reserve=1.0,
        distance_to_charger=100.0,
        distance_to_task=100.0,
        task_to_charger_distance=100.0,
        return_now_requirement=4.0,
        task_then_return_requirement=8.0,
    )
    continue_decision = manager.decide(
        ReturnDecisionContext(remaining_energy=10.0, **common)
    )
    assert continue_decision.commit is False
    assert continue_decision.immediate_margin == pytest.approx(5.0)
    assert continue_decision.mission_margin == pytest.approx(1.0)
    mission_decision = manager.decide(
        ReturnDecisionContext(remaining_energy=9.0, **common)
    )
    assert mission_decision.commit is True
    assert mission_decision.reason == "task_then_return_energy_boundary"
    immediate_decision = manager.decide(
        ReturnDecisionContext(remaining_energy=5.0, **common)
    )
    assert immediate_decision.commit is True
    assert immediate_decision.reason == "immediate_return_energy_boundary"


def test_quantile_return_manager_preserves_seeded_legacy_switch_step() -> None:
    class LegacyInlineReturnManager:
        manager_type = "legacy_inline_two_boundary_rule"
        requires_energy_estimate = True

        def decide(self, context: ReturnDecisionContext) -> ReturnManagerDecision:
            immediate_margin = (
                context.remaining_energy
                - float(context.return_now_requirement)
                - context.reserve
            )
            mission_margin = (
                context.remaining_energy
                - float(context.task_then_return_requirement)
                - context.reserve
            )
            commit = immediate_margin <= 0.0 or mission_margin <= 0.0
            return ReturnManagerDecision(
                commit=commit,
                reason="legacy_inline_commit" if commit else "legacy_inline_continue",
                immediate_margin=immediate_margin,
                mission_margin=mission_margin,
                decision_statistic=float(context.task_then_return_requirement),
            )

    common = dict(
        operational_energy_capacity=10.0,
        energy_reserve_fraction=0.10,
        mission_decision_interval_policy_steps=1,
    )
    refactored = UAVEnergyDeliverySACEnv(**common)
    legacy = UAVEnergyDeliverySACEnv(**common)
    for environment, manager in (
        (refactored, QuantileEnergyReturnManager()),
        (legacy, LegacyInlineReturnManager()),
    ):
        environment.bind_energy_learning(
            energy_estimator=RecordingEstimator(4.0),
            goal_action_provider=zero_policy,
            training_enabled=False,
            return_manager=manager,
        )
        environment.enable_phase_two()

    observations = [environment.reset(seed=1401)[0] for environment in (refactored, legacy)]
    np.testing.assert_allclose(observations[0], observations[1])
    switch_steps: list[int | None] = [None, None]
    for _ in range(200):
        for index, environment in enumerate((refactored, legacy)):
            observation, _, terminated, truncated, _ = environment.step(
                np.zeros(3, dtype=np.float32)
            )
            observations[index] = observation
            assert terminated is False
            assert truncated is False
            if (
                switch_steps[index] is None
                and environment.mode is SortieMode.CHARGER_COMMITTED
            ):
                switch_steps[index] = environment.current_step
        np.testing.assert_allclose(refactored.agent.pos, legacy.agent.pos)
        np.testing.assert_allclose(refactored.agent.vel, legacy.agent.vel)
        assert refactored.agent.energy == pytest.approx(legacy.agent.energy)
        assert refactored.mode is legacy.mode
        if switch_steps[0] is not None or switch_steps[1] is not None:
            break

    assert switch_steps[0] is not None
    assert switch_steps[0] == switch_steps[1]
    assert refactored.switching_events[0]["global_step"] == legacy.switching_events[0][
        "global_step"
    ]
    np.testing.assert_allclose(
        refactored.switching_events[0]["position"],
        legacy.switching_events[0]["position"],
    )
    refactored.close()
    legacy.close()


def test_component_risk_allocation_uses_union_bound_not_fake_joint_q95() -> None:
    estimator = RecordingEstimator(4.0)
    estimator.coverage = 0.975
    environment = UAVEnergyDeliverySACEnv()
    environment.bind_energy_learning(
        energy_estimator=estimator,
        goal_action_provider=zero_policy,
        training_enabled=False,
    )
    environment.reset(seed=16)
    estimate = environment.mission_energy_estimate()
    assert estimate.mission_upper_bound_semantics == (
        "component_upper_bound_sum_with_union_bound"
    )
    assert estimate.mission_nominal_coverage_lower_bound == pytest.approx(0.95)
    environment.close()


def test_return_manager_baselines_share_one_way_decision_contract() -> None:
    context = ReturnDecisionContext(
        mode=SortieMode.TASK,
        remaining_energy=2.0,
        battery_capacity=10.0,
        reserve=0.0,
        distance_to_charger=10.0,
        distance_to_task=8.0,
        task_to_charger_distance=12.0,
    )
    soc = FixedSOCThresholdReturnManager(0.25).decide(context)
    assert soc.commit is True
    assert soc.reason == "soc_threshold_reached"
    distance = DistanceEnergyReturnManager(0.1).decide(context)
    assert distance.commit is True
    assert distance.reason == "distance_task_then_return_boundary"
    assert distance.mission_margin == pytest.approx(0.0)
    absorbing = FixedSOCThresholdReturnManager(0.0).decide(
        ReturnDecisionContext(
            **{**context.__dict__, "mode": SortieMode.CHARGER_COMMITTED}
        )
    )
    assert absorbing.commit is False
    assert absorbing.reason == "commitment_is_absorbing"


def test_soc_return_manager_runs_phase2_without_energy_estimator() -> None:
    environment = UAVEnergyDeliverySACEnv(
        operational_energy_capacity=10.0,
        energy_reserve_fraction=0.0,
    )
    environment.bind_navigation_policy(zero_policy)
    environment.bind_return_manager(FixedSOCThresholdReturnManager(0.25))
    environment.enable_phase_two()
    environment.reset(seed=24)
    environment.agent.energy = 2.0
    assert environment._refresh_mission_decision() is True
    event = environment.switching_events[-1]
    assert event["return_manager_type"] == "fixed_soc_threshold"
    assert event["decision_margin_unit"] == "fraction_of_capacity"
    assert event["governing_required_energy"] is None
    assert event["observed_decision_interval_energy_requirement_drift"] is None
    assert event["mission_energy_upper_bound"] is None
    assert event["reason"] == "soc_threshold_reached"
    environment.close()


def test_energy_return_manager_logs_decision_interval_overshoot() -> None:
    environment = UAVEnergyDeliverySACEnv(
        operational_energy_capacity=10.0,
        energy_reserve_fraction=0.0,
    )
    environment.bind_navigation_policy(zero_policy)
    environment.bind_return_manager(DistanceEnergyReturnManager(0.001))
    environment.enable_phase_two()
    environment.reset(seed=25)
    environment.current_step = 1
    environment.agent.energy = 10.0
    assert environment._refresh_mission_decision() is False
    previous_audit = dict(environment._last_return_decision_audit)

    environment.current_step = 5
    environment.agent.energy = 0.0
    assert environment._refresh_mission_decision() is True
    event = environment.switching_events[-1]
    current_requirement = float(event["governing_required_energy"])
    previous_requirement = float(previous_audit["governing_requirement"])

    assert event["decision_margin_unit"] == ENERGY_UNIT
    assert event["previous_decision_step"] == 1
    assert event["decision_check_interval_steps"] == 4
    assert event["energy_consumed_since_previous_decision_check"] == pytest.approx(10.0)
    assert event["required_energy_increase_since_previous_decision_check"] == pytest.approx(
        current_requirement - previous_requirement
    )
    assert event["observed_decision_interval_energy_requirement_drift"] == pytest.approx(
        10.0 + current_requirement - previous_requirement
    )
    assert event["threshold_crossing_overshoot"] == pytest.approx(
        -float(event["continuation_margin"])
    )
    assert event["reserve_covers_observed_decision_interval_drift"] is False
    environment.close()


def test_parallel_environment_seeds_and_transition_count() -> None:
    args = parse_args(["--output-dir", "/tmp/not-used", "--smoke"])
    vector_environment = make_navigation_vec_env(args)
    observations = vector_environment.reset()
    starts = vector_environment.get_attr("agent")
    positions = [agent.pos.copy() for agent in starts]
    assert observations.shape == (8, 7)
    assert len({tuple(position) for position in positions}) == 8
    vector_steps = 100
    for _ in range(vector_steps):
        actions = np.zeros((args.num_envs, 3), dtype=np.float32)
        vector_environment.step(actions)
    assert vector_steps * args.num_envs == 800
    vector_environment.close()


def test_formal_transition_budget_must_be_exactly_divisible() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--output-dir", "/tmp/x", "--phase1-transition-budget", "500001", "--num-envs", "8"])
    args = parse_args(["--output-dir", "/tmp/x"])
    assert args.phase1_transition_budget == 500_000
    assert args.phase1_transition_budget // args.num_envs == 62_500


def test_small_navigation_run_stops_at_exact_budget_without_finishing_episodes(tmp_path: Path) -> None:
    args = parse_args(["--output-dir", str(tmp_path / "unused"), "--smoke", "--device", "cpu"])
    args.phase1_transition_budget = 80
    args.phase1_episode_max_steps = 1000
    args.eval_freq_transitions = 8000
    args.checkpoint_freq_transitions = 40
    args.log_freq_transitions = 8
    args.learning_starts = 1000
    output = tmp_path / "run"
    (output / "phase1_navigation").mkdir(parents=True)
    (output / "eval").mkdir()
    model, audit, checkpoint = train_navigation_fixed_budget(args, output, [])
    assert model.num_timesteps == 80
    assert audit["global_env_transitions"] == 80
    assert audit["vector_env_steps"] == 10
    assert audit["training_stop_reason"] == "transition_budget_reached"
    assert audit["partial_episodes_at_budget_stop"] == 8
    assert audit["failed_training_episodes"] == 0
    assert checkpoint.exists()
    assert model.gradient_steps == -1


def test_gradient_steps_minus_one_updates_once_per_environment_transition(tmp_path: Path) -> None:
    args = parse_args(["--output-dir", str(tmp_path / "unused"), "--smoke", "--device", "cpu"])
    args.phase1_transition_budget = 32
    args.phase1_episode_max_steps = 1000
    args.eval_freq_transitions = 8000
    args.checkpoint_freq_transitions = 32
    args.log_freq_transitions = 8
    args.learning_starts = 0
    args.batch_size = 8
    output = tmp_path / "run"
    (output / "phase1_navigation").mkdir(parents=True)
    (output / "eval").mkdir()
    model, audit, _ = train_navigation_fixed_budget(args, output, [])
    assert model.gradient_steps == -1
    assert audit["actual_training_transitions"] == 32
    assert audit["actual_gradient_updates"] == 32
    assert audit["gradient_update_to_transition_ratio"] == pytest.approx(1.0)


def test_fixed_evaluation_tasks_are_reproducible_and_stratified() -> None:
    first = generate_stratified_navigation_tasks(num_tasks=10, seed=15)
    second = generate_stratified_navigation_tasks(num_tasks=10, seed=15)
    for left, right in zip(first, second):
        np.testing.assert_allclose(left.start_position, right.start_position)
        np.testing.assert_allclose(left.goal_position, right.goal_position)
    assert {task.distance_bucket for task in first} == {
        "100-500",
        "500-1500",
        "1500-2500",
        "2500-4000",
        ">4000",
    }


def test_evaluation_transitions_are_separate_from_training(tmp_path: Path) -> None:
    args = parse_args(["--output-dir", str(tmp_path / "unused"), "--smoke", "--device", "cpu"])
    policy = HeuristicGoalPolicy()
    start = np.array([1000.0, 1000.0, 100.0], dtype=np.float32)
    task = NavigationTask(start, start + np.array([120.0, 0.0, 0.0]), np.zeros(3), 120.0, "100-500")
    summary = evaluate_navigation_tasks(policy, args, [task], global_env_transitions=8000)
    assert summary["global_env_transitions"] == 8000
    assert summary["evaluation_env_transitions"] > 0
    assert summary["overall_success_rate"] == 1.0


def test_parallel_navigation_evaluation_matches_serial_contract(tmp_path: Path) -> None:
    tasks = [
        NavigationTask(
            np.asarray([1000.0, 1000.0, 100.0], dtype=np.float32),
            np.asarray([1120.0, 1000.0, 100.0], dtype=np.float32),
            np.zeros(3, dtype=np.float32),
            120.0,
            "100-500",
        ),
        NavigationTask(
            np.asarray([1500.0, 1500.0, 200.0], dtype=np.float32),
            np.asarray([1500.0, 1650.0, 200.0], dtype=np.float32),
            np.zeros(3, dtype=np.float32),
            150.0,
            "100-500",
        ),
    ]
    serial_args = parse_args(
        ["--output-dir", str(tmp_path / "serial-run"), "--smoke", "--device", "cpu"]
    )
    serial_args.evaluation_num_envs = 1
    serial_args.eval_task_seed = 701
    parallel_args = parse_args(
        ["--output-dir", str(tmp_path / "parallel-run"), "--smoke", "--device", "cpu"]
    )
    parallel_args.evaluation_num_envs = 2
    parallel_args.eval_task_seed = 701
    serial = evaluate_navigation_tasks(
        HeuristicGoalPolicy(),
        serial_args,
        tasks,
        global_env_transitions=8000,
    )
    output_path = tmp_path / "parallel.json"
    parallel = evaluate_navigation_tasks(
        HeuristicGoalPolicy(),
        parallel_args,
        tasks,
        global_env_transitions=8000,
        output_path=output_path,
    )
    for key in [
        "global_env_transitions",
        "evaluation_env_transitions",
        "num_tasks",
        "overall_success_rate",
        "mean_steps_per_task",
        "mean_path_ratio",
        "boundary_contact_step_rate",
        "obstacle_collision_steps",
        "hocbf_intervention_steps",
        "mean_reward",
    ]:
        assert parallel[key] == pytest.approx(serial[key])
    assert parallel["distance_bucket_success"] == serial["distance_bucket_success"]
    assert parallel["records"] == serial["records"]
    assert serial["execution"]["parallel"] is False
    assert parallel["execution"]["parallel"] is True
    assert parallel["execution"]["num_workers"] == 2
    assert output_path.is_file()
    progress_path = tmp_path / "parallel_progress.jsonl"
    assert progress_path.is_file()
    progress_rows = [
        json.loads(line)
        for line in progress_path.read_text(encoding="utf-8").splitlines()
    ]
    assert progress_rows[-1]["completed_tasks"] == len(tasks)
    assert progress_rows[-1]["formal_gate_evaluable"] is False
    assert progress_rows[-1]["progress_metric_scope"] == (
        "completed_tasks_only_length_biased_until_final"
    )
    assert progress_rows[-1]["partial_success_rate"] == pytest.approx(
        parallel["overall_success_rate"]
    )
    assert progress_rows[-1]["partial_obstacle_collision_steps"] == parallel[
        "obstacle_collision_steps"
    ]


def test_navigation_boundary_episode_metrics() -> None:
    args = parse_args(["--output-dir", "/tmp/not-used", "--smoke", "--device", "cpu"])
    args.phase1_episode_max_steps = 3

    class BoundaryPolicy:
        def predict(self, observation, deterministic=True):
            del observation, deterministic
            return np.array([-1.0, 0.0, 0.0], dtype=np.float32), None

    start = np.array([0.5, 1000.0, 100.0], dtype=np.float32)
    task = NavigationTask(start, np.array([1000.0, 1000.0, 100.0]), np.zeros(3), 999.5, "500-1500")
    summary = evaluate_navigation_tasks(BoundaryPolicy(), args, [task], global_env_transitions=0)
    record = summary["records"][0]
    assert record["had_boundary_contact"] is True
    assert record["boundary_contact_steps"] == 3
    assert record["max_consecutive_boundary_contacts"] == 3
    assert summary["boundary_contact_episode_rate"] == pytest.approx(1.0)
    assert summary["mean_boundary_contacts_per_episode"] == pytest.approx(3.0)
    assert summary["max_boundary_contacts_in_episode"] == 3
    assert summary["max_consecutive_boundary_contacts"] == 3
    assert summary["boundary_contact_step_rate"] == pytest.approx(1.0)
    assert navigation_energy_gate_passed(summary) is False
    assert navigation_safety_gate_passed(summary) is False


def test_navigation_energy_and_safety_readiness_are_independent() -> None:
    summary = {
        "overall_success_rate": 1.0,
        "distance_bucket_success": {
            "100-500": 1.0,
            "500-1500": 1.0,
            "1500-2500": 1.0,
            "2500-4000": 1.0,
            ">4000": 1.0,
        },
        "mean_path_ratio": 1.0315,
        "boundary_contact_step_rate": 0.02896,
    }
    assert navigation_energy_gate_passed(summary) is True
    assert navigation_safety_gate_passed(summary) is False


def test_td_metrics_are_split_by_boundary_contact() -> None:
    base = {
        "true_total_energy": 10.0,
        "finite_predictions": True,
        "quantile_ordering_valid": True,
        "td_mae": 1.0,
        "td_rmse": 1.0,
        "td_bias": 0.0,
        "td_underestimation_rate": 0.5,
        "td_overestimation_rate": 0.5,
        "td_mean_underestimation_magnitude": 0.5,
        "td_relative_error": 0.1,
        "q50_coverage": 0.5,
        "q90_coverage": 0.9,
        "q95_coverage": 0.95,
        "q99_coverage": 0.99,
        "distance_bucket": "100-500",
    }
    metrics = aggregate_goal_evaluations(
        [
            {**base, "had_boundary_contact": False},
            {**base, "had_boundary_contact": True, "td_mae": 3.0},
        ]
    )
    groups = metrics["by_boundary_contact"]
    assert groups["all_trajectories"]["num_completed_goals"] == 2
    assert groups["clean_trajectories"]["td_mae"] == pytest.approx(1.0)
    assert groups["boundary_contact_trajectories"]["td_mae"] == pytest.approx(3.0)


def test_resume_phase1_loads_and_freezes_existing_500k_checkpoint(tmp_path: Path) -> None:
    source = tmp_path / "source"
    phase1 = source / "phase1_navigation"
    phase1.mkdir(parents=True)
    environment = UAVEnergyDeliverySACEnv()
    model = SAC("MlpPolicy", environment, device="cpu", buffer_size=32, learning_starts=32)
    checkpoint = phase1 / "checkpoint_transition_500000.zip"
    model.save(checkpoint)
    final_evaluation = {
        "overall_success_rate": 1.0,
        "distance_bucket_success": {
            "100-500": 1.0,
            "500-1500": 1.0,
            "1500-2500": 1.0,
            "2500-4000": 1.0,
            ">4000": 1.0,
        },
        "mean_path_ratio": 1.03,
        "boundary_contact_step_rate": 0.03,
        "boundary_contact_episode_rate": 0.28,
        "max_consecutive_boundary_contacts": 667,
    }
    (phase1 / "summary.json").write_text(
        json.dumps(
            {
                "actual_training_transitions": 500000,
                "exact_budget_match": True,
                "final_evaluation": final_evaluation,
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "output"
    (output / "phase1_navigation").mkdir(parents=True)
    args = parse_args(
        [
            "--output-dir",
            str(output),
            "--smoke",
            "--device",
            "cpu",
            "--resume-after-phase1-checkpoint",
            str(checkpoint),
            "--source-phase1-artifact",
            str(source),
        ]
    )
    resumed, audit, resumed_checkpoint = load_resumed_phase1(args, output)
    assert resumed_checkpoint == checkpoint.resolve()
    assert audit["phase1_retrained"] is False
    assert audit["source_transition"] == 500000
    assert audit["navigation_energy_ready"] is True
    assert audit["navigation_safety_ready"] is False
    assert all(not parameter.requires_grad for parameter in resumed.policy.parameters())
    assert (output / "phase1_navigation" / "source_final_evaluation.json").exists()
    environment.close()


def test_intermediate_100k_resume_uses_checkpoint_transition_not_500k_summary(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    phase1 = source / "phase1_navigation"
    phase1.mkdir(parents=True)
    environment = UAVEnergyDeliverySACEnv()
    model = SAC("MlpPolicy", environment, device="cpu", buffer_size=32, learning_starts=32)
    model.num_timesteps = 100_000
    checkpoint = phase1 / "checkpoint_transition_100000.zip"
    model.save(checkpoint)
    (phase1 / "summary.json").write_text(
        json.dumps(
            {
                "actual_training_transitions": 500_000,
                "exact_budget_match": True,
                "gif_evaluation_env_transitions": 0,
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "output"
    (output / "phase1_navigation").mkdir(parents=True)
    args = parse_args(
        [
            "--output-dir",
            str(output),
            "--smoke",
            "--device",
            "cpu",
            "--resume-after-phase1-checkpoint",
            str(checkpoint),
            "--source-phase1-artifact",
            str(source),
            "--source-phase1-transition",
            "100000",
            "--intermediate-checkpoint-energy-ablation",
        ]
    )
    resumed, audit, _ = load_resumed_phase1(args, output)
    assert audit["actual_training_transitions"] == 100_000
    assert audit["final_evaluation"] is None
    assert audit["navigation_energy_ready"] is None
    assert audit["intermediate_checkpoint_energy_ablation"] is True
    assert "EXPLORATORY" in audit["claim_status"]
    assert not (output / "phase1_navigation" / "source_final_evaluation.json").exists()
    assert all(not parameter.requires_grad for parameter in resumed.policy.parameters())
    environment.close()


def test_formal_phase2_rejects_mock_energy_estimator(tmp_path: Path) -> None:
    args = parse_args(["--output-dir", str(tmp_path / "unused"), "--smoke", "--device", "cpu"])
    with pytest.raises(TypeError, match="trained_goal_conditioned_quantile_td"):
        run_phase2_fixed_budget(
            HeuristicGoalPolicy(),
            RecordingEstimator(1.0),
            args,
            battery_capacity=10.0,
            output=tmp_path,
        )


def test_td_checkpoint_round_trip_for_flight_evaluation(tmp_path: Path) -> None:
    estimator = GoalConditionedQuantileTDEnergyEstimator(
        hidden_dim=8,
        batch_size=2,
        replay_capacity=8,
        learning_starts=2,
        device="cpu",
    )
    state = np.linspace(0.0, 1.0, 7, dtype=np.float32)
    action = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    expected = estimator.predict_quantiles(state, action)
    checkpoint = tmp_path / "td.pt"
    estimator.save(checkpoint)
    restored = GoalConditionedQuantileTDEnergyEstimator.load(checkpoint, device="cpu")
    np.testing.assert_allclose(restored.predict_quantiles(state, action), expected)
    assert restored.update_count == estimator.update_count


def test_full_td_flight_gif_generation(tmp_path: Path) -> None:
    environment = UAVEnergyDeliverySACEnv()
    environment.reset(seed=4)
    rows = []
    for step in range(1, 4):
        rows.append(
            {
                "step": step,
                "position": np.array([100.0 + 20.0 * step, 200.0, 50.0]),
                "velocity": np.array([5.0, 0.0, 0.0]),
                "speed": 5.0,
                "distance_to_goal": 80.0 - 20.0 * step,
                "realized_energy": 0.2,
                "true_energy_to_go": 0.2 * (4 - step),
                "Q50": 0.3 * (4 - step),
                "Q90": 0.4 * (4 - step),
                "Q95": 0.5 * (4 - step),
                "Q99": 0.6 * (4 - step),
                "boundary_contact": False,
            }
        )
    summary = {
        "start_position": [100.0, 200.0, 50.0],
        "goal_position": [180.0, 200.0, 50.0],
    }
    output = tmp_path / "full.gif"
    frames = render_td_flight_gif(rows, summary, environment, output, fps=5)
    assert frames == 3
    assert output.exists() and output.stat().st_size > 0
    with Image.open(output) as image:
        assert image.n_frames == 3
    environment.close()


def test_collect_td_flight_records_every_transition_until_natural_goal_reach() -> None:
    task = {
        "start_position": [1000.0, 1000.0, 100.0],
        "goal_position": [1100.0, 1000.0, 100.0],
        "initial_velocity": [20.0, 0.0, 0.0],
    }
    rows, summary, environment = collect_td_flight(
        ConstantActionPolicy(np.array([1.0, 0.0, 0.0], dtype=np.float32)),
        RecordingEstimator(1.0),
        task,
        seed=17,
        max_steps=30,
    )
    assert 1 < len(rows) < 30
    assert summary["success"] is True
    assert summary["end_reason"] == "goal_reached"
    assert summary["policy_steps"] == len(rows)
    assert summary["boundary_contact_steps"] == 0
    assert summary["quantile_ordering_valid"] is True
    assert [row["step"] for row in rows] == list(range(1, len(rows) + 1))
    assert all(float(row["realized_energy"]) > 0.0 for row in rows)
    assert all(float(row["transition_dt"]) > 0.0 for row in rows)
    assert rows[-1]["true_energy_to_go"] == pytest.approx(rows[-1]["realized_energy"])
    assert rows[0]["true_energy_to_go"] == pytest.approx(summary["true_total_energy"])
    environment.close()


def test_energy_managed_scene_records_exhaustion_and_full_gif(tmp_path: Path) -> None:
    trace, summary, environment = collect_energy_managed_scene(
        ZeroPredictPolicy(),
        RecordingEstimator(1_000.0),
        capacity=1.0,
        seed=5,
        max_recording_steps=10,
        start_position=np.array([500.0, 500.0, 200.0], dtype=np.float32),
        task_point=np.array([3500.0, 3500.0, 200.0], dtype=np.float32),
        initial_energy_fraction=0.001,
    )
    assert len(trace) == 1
    assert summary["energy_exhausted"] is True
    assert summary["environment_terminated"] is True
    assert summary["recording_stop_reason"] == "energy_exhausted"
    output = tmp_path / "managed_exhaustion.gif"
    frames = render_energy_managed_gif(trace, summary, environment, output, fps=5)
    assert frames == 1
    assert output.exists() and output.stat().st_size > 0
    environment.close()


def test_energy_managed_scene_detects_zero_task_charger_loop() -> None:
    environment = UAVEnergyDeliverySACEnv(
        phase=SACTrainingPhase.ENERGY_MANAGED,
        operational_energy_capacity=10.0,
        minimum_task_distance=5.1,
    )
    trace, summary, environment = collect_energy_managed_scene(
        ZeroPredictPolicy(),
        RecordingEstimator(1_000.0),
        capacity=10.0,
        seed=6,
        max_recording_steps=10,
        start_position=environment.charger_position,
        task_point=environment.charger_position + np.array([6.0, 0.0, 0.0], dtype=np.float32),
        post_first_recharge_steps=2,
        environment=environment,
    )
    assert len(trace) == 3
    assert summary["battery_cycles_completed"] == 3
    assert summary["tasks_completed"] == 0
    assert summary["charger_loop_detected"] is True
    assert summary["successful_recharge_count"] == 3
    assert summary["resumed_task_after_recharge"] is False
    assert summary["requested_task_return_recharge_resume_exhaustion_lifecycle_observed"] is False
    assert summary["environment_terminated"] is False
    assert summary["recording_stop_reason"] == "post_first_recharge_recording_complete"
    environment.close()


def test_energy_managed_lifecycle_audit_requires_all_physical_events() -> None:
    trace = [
        {
            "step": 1,
            "tasks_completed": 1,
            "battery_cycle_end": False,
            "battery_cycle_record": None,
            "mode": "TASK",
            "switched_now": False,
            "terminated": False,
            "end_reason": "in_progress",
        },
        {
            "step": 2,
            "tasks_completed": 1,
            "battery_cycle_end": True,
            "battery_cycle_record": {"return_success": True},
            "mode": "TASK",
            "switched_now": False,
            "terminated": False,
            "end_reason": "charger_reached",
        },
        {
            "step": 3,
            "tasks_completed": 1,
            "battery_cycle_end": False,
            "battery_cycle_record": None,
            "mode": "TASK",
            "switched_now": False,
            "terminated": False,
            "end_reason": "in_progress",
        },
        {
            "step": 4,
            "tasks_completed": 1,
            "battery_cycle_end": True,
            "battery_cycle_record": {"return_success": False},
            "mode": "CHARGER_COMMITTED",
            "switched_now": False,
            "terminated": True,
            "end_reason": "energy_exhausted",
        },
    ]
    audit = audit_managed_lifecycle(trace)
    assert audit["tasks_before_first_recharge"] == 1
    assert audit["successful_recharge_count"] == 1
    assert audit["resumed_task_after_recharge"] is True
    assert audit["energy_exhausted_after_recharge"] is True
    assert audit["requested_task_return_recharge_resume_exhaustion_lifecycle_observed"] is True


def test_freeze_navigation_clears_only_td_replay() -> None:
    environment = UAVEnergyDeliverySACEnv()
    model = SAC("MlpPolicy", environment, device="cpu", buffer_size=100, learning_starts=100)
    estimator = GoalConditionedQuantileTDEnergyEstimator(
        hidden_dim=8,
        batch_size=2,
        replay_capacity=16,
        learning_starts=2,
    )
    state = np.zeros(7, dtype=np.float32)
    action = np.zeros(3, dtype=np.float32)
    estimator.observe_transition(state, action, 1.0, state, state, action, True, "TASK")
    result = freeze_navigation_and_start_td(model, estimator)
    assert result["sac_frozen"] is True
    assert result["td_replay_size"] == 0
    assert all(not parameter.requires_grad for parameter in model.policy.parameters())


def test_constant_power_capacity_and_duration_scaling() -> None:
    capacities = calculate_battery_capacities(2.0, 30.0)
    assert capacities["calibrated_battery_capacity"] == pytest.approx(3600.0)
    assert capacities["capacity_for_20min"] == pytest.approx(2400.0)
    assert capacities["capacity_for_30min"] == pytest.approx(3600.0)
    assert capacities["capacity_for_40min"] == pytest.approx(4800.0)
    assert calculate_battery_capacities(2.0, 40.0)["calibrated_battery_capacity"] == pytest.approx(4800.0)


def test_battery_calibration_uses_realized_telemetry_and_does_not_write_td(tmp_path: Path) -> None:
    args = parse_args(["--output-dir", str(tmp_path / "unused"), "--smoke", "--device", "cpu"])
    args.target_nominal_endurance_minutes = 1.0
    start = np.array([1000.0, 1000.0, 100.0], dtype=np.float32)
    task = NavigationTask(start, start + np.array([120.0, 0.0, 0.0]), np.zeros(3), 120.0, "100-500")
    output = tmp_path / "calibration"
    output.mkdir()
    summary = run_battery_calibration(HeuristicGoalPolicy(), args, [task], output)
    assert summary["energy_source"] == "TelemetryCostModel.realized_cost"
    assert summary["mean_power"] > 0.0
    assert summary["calibrated_battery_capacity"] > 0.0
    assert summary["td_replay_writes"] == 0
    assert summary["calibration_success_rate"] == pytest.approx(1.0)
    assert summary["mean_power_successful_only"] == pytest.approx(summary["mean_power"])
    assert summary["mean_power_all_rollouts"] > 0.0
    assert summary["battery_calibration_navigation_valid"] is True
    assert summary["battery_calibration_env_transitions"] > 0
    assert (output / "battery_calibration.json").exists()
    assert (output / "battery_calibration_power_distribution.png").exists()


def test_parallel_battery_calibration_matches_serial_task_results(tmp_path: Path) -> None:
    args = parse_args(["--output-dir", str(tmp_path / "unused"), "--smoke", "--device", "cpu"])
    args.target_nominal_endurance_minutes = 0.1
    args.minimum_task_distance = 5.0
    args.evaluation_progress_interval_tasks = 1
    starts = (
        np.array([1000.0, 1000.0, 100.0], dtype=np.float32),
        np.array([2000.0, 2000.0, 150.0], dtype=np.float32),
    )
    tasks = [
        NavigationTask(
            start,
            start + np.array([20.0, 0.0, 0.0], dtype=np.float32),
            np.zeros(3, dtype=np.float32),
            20.0,
            "100-500",
        )
        for start in starts
    ]
    serial_output = tmp_path / "serial"
    parallel_output = tmp_path / "parallel"
    serial_output.mkdir()
    parallel_output.mkdir()
    args.evaluation_num_envs = 1
    serial = run_battery_calibration(
        HeuristicGoalPolicy(),
        args,
        tasks,
        serial_output,
    )
    args.evaluation_num_envs = 2
    parallel = run_battery_calibration(
        HeuristicGoalPolicy(),
        args,
        tasks,
        parallel_output,
    )
    assert parallel["execution"]["parallel"] is True
    assert parallel["execution"]["num_workers"] == 2
    assert parallel["battery_calibration_env_transitions"] == serial[
        "battery_calibration_env_transitions"
    ]
    assert parallel["total_realized_energy"] == pytest.approx(
        serial["total_realized_energy"]
    )
    assert parallel["mean_power"] == pytest.approx(serial["mean_power"])
    assert (parallel_output / "battery_calibration_progress.jsonl").exists()


def test_calibration_audits_failed_rollouts_instead_of_dropping_them(tmp_path: Path) -> None:
    args = parse_args(["--output-dir", str(tmp_path / "unused"), "--smoke", "--device", "cpu"])
    args.minimum_task_distance = 5.0
    args.phase1_episode_max_steps = 1
    start = np.array([1000.0, 1000.0, 100.0], dtype=np.float32)
    tasks = [
        NavigationTask(
            start,
            start + np.array([5.5, 0.0, 0.0], dtype=np.float32),
            np.array([20.0, 0.0, 0.0], dtype=np.float32),
            5.5,
            "100-500",
        ),
        NavigationTask(
            start,
            start + np.array([100.0, 0.0, 0.0], dtype=np.float32),
            np.zeros(3, dtype=np.float32),
            100.0,
            "100-500",
        ),
    ]
    output = tmp_path / "calibration"
    output.mkdir()
    summary = run_battery_calibration(HeuristicGoalPolicy(), args, tasks, output)
    assert summary["successful_task_count"] == 1
    assert summary["failed_task_count"] == 1
    assert summary["calibration_success_rate"] == pytest.approx(0.5)
    assert summary["total_energy_failed"] > 0.0
    assert summary["mean_power_all_rollouts"] > 0.0
    assert summary["battery_calibration_navigation_valid"] is False


def test_heldout_energy_evaluation_is_read_only_and_seed_separate(tmp_path: Path) -> None:
    args = parse_args(["--output-dir", str(tmp_path / "unused"), "--smoke", "--device", "cpu"])
    args.minimum_task_distance = 5.0
    assert args.energy_eval_seed not in {
        args.seed,
        args.eval_task_seed,
        args.battery_calibration_seed,
        args.battery_validation_seed,
        args.td_collection_seed,
    }
    estimator = GoalConditionedQuantileTDEnergyEstimator(
        hidden_dim=8,
        batch_size=2,
        replay_capacity=16,
        learning_starts=2,
    )
    state = np.zeros(7, dtype=np.float32)
    action = np.zeros(3, dtype=np.float32)
    estimator.observe_transition(state, action, 1.0, state, state, action, True, "TASK")
    replay_before = len(estimator.replay)
    update_before = estimator.update_count
    start = np.array([1000.0, 1000.0, 100.0], dtype=np.float32)
    task = NavigationTask(
        start,
        start + np.array([5.5, 0.0, 0.0], dtype=np.float32),
        np.array([20.0, 0.0, 0.0], dtype=np.float32),
        5.5,
        "100-500",
    )
    output = tmp_path / "heldout.json"
    summary = evaluate_energy_tasks(
        HeuristicGoalPolicy(),
        estimator,
        args,
        [task],
        global_td_transitions=50_000,
        output_path=output,
    )
    assert summary["successful_tasks"] == 1
    assert summary["td_optimizer_enabled"] is False
    assert summary["td_replay_writes"] == 0
    assert len(estimator.replay) == replay_before
    assert estimator.update_count == update_before
    assert summary["metrics"]["overall"]["num_completed_goals"] == 1
    assert summary["MAE"] is not None
    assert summary["Q95_coverage"] is not None
    assert "100-500" in summary["distance_bucket_metrics"]
    assert output.exists()


def test_parallel_heldout_energy_evaluation_uses_frozen_worker_snapshots(
    tmp_path: Path,
) -> None:
    args = parse_args(
        ["--output-dir", str(tmp_path / "unused"), "--smoke", "--device", "cpu"]
    )
    args.minimum_task_distance = 5.0
    args.evaluation_num_envs = 2
    estimator = GoalConditionedQuantileTDEnergyEstimator(
        hidden_dim=8,
        batch_size=2,
        replay_capacity=16,
        learning_starts=2,
    )
    state = np.zeros(7, dtype=np.float32)
    action = np.zeros(3, dtype=np.float32)
    estimator.observe_transition(
        state,
        action,
        1.0,
        state,
        state,
        action,
        True,
        "TASK",
    )
    parameter_before = [
        parameter.detach().clone() for parameter in estimator.model.parameters()
    ]
    replay_before = len(estimator.replay)
    update_before = estimator.update_count
    tasks = []
    for task_index in range(2):
        start = np.array(
            [1000.0, 1000.0 + 20.0 * task_index, 100.0],
            dtype=np.float32,
        )
        tasks.append(
            NavigationTask(
                start,
                start + np.array([5.5, 0.0, 0.0], dtype=np.float32),
                np.array([20.0, 0.0, 0.0], dtype=np.float32),
                5.5,
                "100-500",
            )
        )
    output = tmp_path / "parallel_heldout.json"
    summary = evaluate_energy_tasks(
        HeuristicGoalPolicy(),
        estimator,
        args,
        tasks,
        global_td_transitions=50_000,
        output_path=output,
    )
    assert summary["successful_tasks"] == 2
    assert summary["execution"]["parallel"] is True
    assert summary["execution"]["num_workers"] == 2
    assert summary["execution"]["td_inference"] == "frozen_worker_snapshot_cpu"
    assert len(estimator.replay) == replay_before
    assert estimator.update_count == update_before
    for before, after in zip(
        parameter_before,
        estimator.model.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(after, before)
    assert output.exists()


def test_battery_validation_runs_to_depletion_and_writes_report(tmp_path: Path) -> None:
    args = parse_args(["--output-dir", str(tmp_path / "unused"), "--smoke", "--device", "cpu"])
    args.battery_validation_runs = 2
    args.target_nominal_endurance_minutes = 0.01
    output = tmp_path / "validation"
    output.mkdir()
    summary = run_battery_validation(
        HeuristicGoalPolicy(),
        args,
        battery_capacity=0.06,
        output=output,
    )
    assert summary["battery_validation_runs"] == 2
    assert summary["all_runs_depleted"] is True
    assert summary["battery_validation_env_transitions"] > 0
    assert (output / "battery_validation.json").exists()
    assert (output / "battery_validation_depletion_time.png").exists()


def test_parallel_battery_validation_uses_batched_workers_and_progress(tmp_path: Path) -> None:
    args = parse_args(["--output-dir", str(tmp_path / "unused"), "--smoke", "--device", "cpu"])
    args.battery_validation_runs = 2
    args.target_nominal_endurance_minutes = 0.01
    args.evaluation_num_envs = 2
    args.evaluation_progress_interval_tasks = 1
    output = tmp_path / "validation_parallel"
    output.mkdir()
    summary = run_battery_validation(
        HeuristicGoalPolicy(),
        args,
        battery_capacity=0.06,
        output=output,
    )
    assert summary["execution"]["parallel"] is True
    assert summary["execution"]["num_workers"] == 2
    assert summary["battery_validation_runs"] == 2
    assert summary["all_runs_depleted"] is True
    assert (output / "battery_validation_progress.jsonl").exists()


def test_capacity_relative_reserve_and_remaining_fraction() -> None:
    environment = UAVEnergyDeliverySACEnv()
    environment.configure_calibrated_battery(250.0, reserve_fraction=0.15)
    assert environment.energy_reserve == pytest.approx(37.5)
    environment.enable_battery_validation()
    _, info = environment.reset(seed=16)
    assert info["battery_capacity"] == 250.0
    assert info["remaining_energy_fraction"] == pytest.approx(1.0)
    assert info["battery_capacity_source"] == "phase1_frozen_policy_calibration"


def test_phase2_state_machine_smoke_recharges_two_cycles_and_writes_logs(tmp_path: Path) -> None:
    args = parse_args(["--output-dir", str(tmp_path / "unused"), "--smoke", "--device", "cpu"])
    result = run_phase2_state_machine_smoke(args, battery_capacity=20.0, output=tmp_path)
    assert result["task_observed"] is True
    assert result["commitment_observed"] is True
    assert result["successful_battery_cycles"] >= 2
    assert result["charger_reached_nonterminal"] is True
    assert result["post_cycle_energy"] == pytest.approx(20.0)
    assert (tmp_path / "battery_cycles.jsonl").exists()
    assert (tmp_path / "switching_events.jsonl").exists()
    for filename in (
        "battery_remaining_curve.png",
        "battery_cycle_energy_curve.png",
        "energy_per_policy_step_curve.png",
        "energy_power_curve.png",
    ):
        assert (tmp_path / filename).exists()


def test_energy_exhaustion_smoke_covers_failure_path(tmp_path: Path) -> None:
    args = parse_args(["--output-dir", str(tmp_path / "unused"), "--smoke", "--device", "cpu"])
    result = run_energy_exhaustion_smoke(args, tmp_path)
    assert result["terminated"] is True
    assert result["truncated"] is False
    assert result["end_reason"] == "energy_exhausted"


def test_navigation_curve_files_are_generated(tmp_path: Path) -> None:
    for transition in (800, 1600):
        row = {
            "global_env_transitions": transition,
            "vector_env_steps": transition // 8,
            "num_envs": 8,
            "completed_training_episodes": transition // 100,
            "successful_training_episodes": transition // 200,
            "failed_training_episodes": transition // 200,
            "rolling_mean_reward_100": float(transition) / 100.0,
            "rolling_mean_reward_1000": float(transition) / 100.0,
        }
        with (tmp_path / "training_curve.jsonl").open("a", encoding="utf-8") as handle:
            import json

            handle.write(json.dumps(row) + "\n")
    generate_navigation_curves(tmp_path)
    assert (tmp_path / "reward_curve.png").exists()
    assert (tmp_path / "tasks_curve.png").exists()


def test_fixed_energy_cost_per_step_argument_is_removed() -> None:
    with pytest.raises(TypeError):
        UAVEnergyDeliverySACEnv(energy_cost_per_step=1.0)


def test_phase1_has_no_td_collection_before_explicit_binding() -> None:
    environment = UAVEnergyDeliverySACEnv()
    environment.reset(seed=17)
    for _ in range(3):
        environment.step(np.zeros(3, dtype=np.float32))
    assert environment.energy_estimator is None
    assert environment.energy_learning_enabled is False


def test_calibration_failure_requires_explicit_override_flag() -> None:
    default = parse_args(["--output-dir", "/tmp/x"])
    override = parse_args(["--output-dir", "/tmp/x", "--allow-failed-battery-calibration"])
    assert default.allow_failed_battery_calibration is False
    assert override.allow_failed_battery_calibration is True


def test_formal_parallel_evaluation_defaults_fit_dual_run_cpu_budget() -> None:
    formal = parse_args(["--output-dir", "/tmp/formal-parallel-eval"])
    smoke = parse_args(["--output-dir", "/tmp/smoke-parallel-eval", "--smoke"])
    assert formal.evaluation_num_envs == 6
    assert formal.evaluation_progress_interval_tasks == 10
    assert smoke.evaluation_num_envs == 1
