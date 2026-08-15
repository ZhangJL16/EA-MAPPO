from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
import pytest
import torch
from stable_baselines3 import SAC

from envs.UAVEnergyDelivery import UAVEnv
from envs.UAVEnergyDeliverySAC import (
    ENERGY_QUANTILES,
    ENERGY_UNIT,
    GoalConditionedQuantileTDEnergyEstimator,
    SACTrainingPhase,
    UAVEnergyDeliverySACEnv,
)
from review_bundle.envs.navigation.state import NavigationState
from review_bundle.envs.navigation.telemetry_cost import TelemetryCostConfig, TelemetryCostModel
from review_bundle.safety.switching import SortieMode
from scripts.train_uav_energy_delivery_sac import (
    HeuristicGoalPolicy,
    NavigationTask,
    calculate_battery_capacities,
    evaluate_navigation_tasks,
    freeze_navigation_and_start_td,
    generate_navigation_curves,
    generate_stratified_navigation_tasks,
    make_navigation_vec_env,
    parse_args,
    run_battery_calibration,
    run_battery_validation,
    run_energy_exhaustion_smoke,
    run_phase2_state_machine_smoke,
    train_navigation_fixed_budget,
)


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
    assert estimate.return_now_q95 == 4.0
    environment.agent.energy = 9.0
    assert environment._refresh_mission_decision() is True
    assert environment.mode is SortieMode.CHARGER_COMMITTED
    environment.agent.energy = 10.0
    assert environment._refresh_mission_decision() is False
    assert environment.mode is SortieMode.CHARGER_COMMITTED


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
    assert summary["battery_calibration_env_transitions"] > 0
    assert (output / "battery_calibration.json").exists()
    assert (output / "battery_calibration_power_distribution.png").exists()


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
