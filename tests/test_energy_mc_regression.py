from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from stable_baselines3 import SAC

from envs.UAVEnergyDeliverySAC import (
    SACTrainingPhase,
    StaticCylinderObstacle,
    UAVEnergyDeliverySACEnv,
)
from experiments.energy_mc.core import (
    ENERGY_GOAL_TYPES,
    EnergyGoalSpec,
    EnergyTrajectory,
    PackedEnergyDataset,
    assert_disjoint_trajectory_splits,
    generate_energy_goal_specs,
    monte_carlo_energy_to_go,
)
from review_bundle.safety.energy.mc_regression import (
    EnergyToGoRegressor,
    DistanceEnergyEstimator,
    GoalEnergyPrediction,
    ModelBasedEnergyRolloutEstimator,
)
from scripts.train_uav_energy_mc import parse_args as parse_mc_args
from scripts.train_uav_energy_mc import run_phase2
from scripts.train_uav_energy_delivery_sac import HeuristicGoalPolicy


def _trajectory(trajectory_id: int, goal_type: str = "TASK") -> EnergyTrajectory:
    costs = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
    spec = EnergyGoalSpec(
        trajectory_id,
        goal_type,
        np.asarray([500.0, 500.0, 200.0], dtype=np.float32),
        np.asarray([600.0, 500.0, 200.0], dtype=np.float32),
        np.zeros(3, dtype=np.float32),
        100.0,
        "100-500",
    )
    return EnergyTrajectory(
        spec,
        np.zeros((3, 7), dtype=np.float32),
        costs,
        np.full(3, 0.2, dtype=np.float32),
        monte_carlo_energy_to_go(costs).astype(np.float32),
        6.0,
        100.0,
        0.6,
        3,
        True,
        "goal_reached",
        False,
        0,
    )


def test_mc_energy_returns_are_exact() -> None:
    assert monte_carlo_energy_to_go(np.asarray([1.0, 2.0, 3.0])).tolist() == [6.0, 5.0, 3.0]


def test_energy_goal_specs_cover_all_contexts_and_distance_buckets() -> None:
    specs = generate_energy_goal_specs(
        num_trajectories=15,
        seed=7,
        charger_position=np.asarray([2000.0, 2000.0, 200.0], dtype=np.float32),
    )
    assert {spec.goal_type for spec in specs} == set(ENERGY_GOAL_TYPES)
    assert {spec.distance_bucket for spec in specs} == {
        "100-500",
        "500-1500",
        "1500-2500",
        "2500-4000",
        ">4000",
    }
    endpoint = next(spec for spec in specs if spec.goal_type == "TASK_ENDPOINT_TO_CHARGER")
    assert np.allclose(endpoint.initial_velocity, 0.0)


def test_trajectory_splits_reject_leakage() -> None:
    train = PackedEnergyDataset.from_trajectories([_trajectory(1)])
    validation = PackedEnergyDataset.from_trajectories([_trajectory(2)])
    assert_disjoint_trajectory_splits({"train": train, "validation": validation})
    with pytest.raises(ValueError, match="trajectory leakage"):
        assert_disjoint_trajectory_splits({"train": train, "test": train})


def test_packed_dataset_round_trip_preserves_trajectory_split(tmp_path: Path) -> None:
    dataset = PackedEnergyDataset.from_trajectories(
        [_trajectory(1, "TASK"), _trajectory(2, "CHARGER")]
    )
    dataset.save(tmp_path / "split")
    loaded = PackedEnergyDataset.load(tmp_path / "split")
    assert loaded.successful_trajectory_ids == {1, 2}
    assert np.array_equal(loaded.targets, dataset.targets)
    assert (tmp_path / "split" / "manifest.json").exists()


def test_regressor_output_is_finite_nonnegative_and_normalization_reversible() -> None:
    estimator = EnergyToGoRegressor(battery_capacity=378.7263, hidden_dim=8, device="cpu")
    states = np.zeros((4, 7), dtype=np.float32)
    predictions = estimator.predict_batch(states)
    assert np.all(np.isfinite(predictions))
    assert np.all(predictions > 0.0)
    physical = np.asarray([0.0, 20.0, 50.0, 378.7263])
    assert np.allclose(
        estimator.denormalize_prediction(estimator.normalize_target(physical)),
        physical,
        atol=1e-5,
    )


def test_distance_energy_baseline_uses_linear_physical_distance() -> None:
    estimator = DistanceEnergyEstimator(energy_per_meter=0.01, d_max=5000.0)
    states = np.zeros((2, 7), dtype=np.float32)
    states[:, -1] = [0.1, 0.5]
    assert estimator.predict_batch(states).tolist() == pytest.approx([5.0, 25.0])


def test_supervised_fit_and_checkpoint_round_trip(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    states = rng.normal(size=(128, 7)).astype(np.float32)
    targets = (10.0 + 2.0 * np.abs(states[:, 0])).astype(np.float32)
    estimator = EnergyToGoRegressor(
        battery_capacity=100.0,
        hidden_dim=16,
        batch_size=32,
        learning_rate=1e-3,
        seed=3,
        device="cpu",
    )
    history = estimator.fit(states, targets, states, targets, max_epochs=20, patience=5)
    assert history.epochs_completed > 0
    checkpoint = tmp_path / "energy.pt"
    estimator.save(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["energy_estimator_type"] == "mc_supervised_energy_to_go"
    assert payload["bootstrapping"] is False
    assert payload["gamma"] is None
    assert payload["navigation_policy"] == "frozen_sac_500k"
    restored = EnergyToGoRegressor.load(checkpoint)
    assert np.allclose(restored.predict_batch(states[:4]), estimator.predict_batch(states[:4]))


def test_one_sided_calibration_uses_trajectory_scores_and_is_test_independent() -> None:
    estimator = EnergyToGoRegressor(battery_capacity=100.0, hidden_dim=8, device="cpu")
    estimator.predict_batch = lambda states: np.zeros(len(states), dtype=np.float64)  # type: ignore[method-assign]
    calibration_states = np.zeros((6, 7), dtype=np.float32)
    calibration_truth = np.asarray([1.0, 2.0, 4.0, 1.0, 3.0, 5.0])
    trajectory_ids = np.asarray([0, 0, 1, 1, 2, 2])
    metadata = estimator.calibrate_upper_bound(
        calibration_states,
        calibration_truth,
        trajectory_ids,
        coverage=0.80,
    )
    assert metadata["num_calibration_trajectories"] == 3
    assert estimator.upper_delta == 5.0
    test_truth = np.asarray([1000.0])
    assert estimator.upper_delta == 5.0
    assert test_truth[0] > estimator.upper_delta


def test_model_rollout_oracle_does_not_modify_live_environment() -> None:
    environment = UAVEnergyDeliverySACEnv(phase=SACTrainingPhase.TD_PRETRAINING)
    environment.reset(
        seed=11,
        options={
            "start_position": np.asarray([500.0, 500.0, 200.0], dtype=np.float32),
            "start_velocity": np.zeros(3, dtype=np.float32),
            "task_point": np.asarray([650.0, 500.0, 200.0], dtype=np.float32),
        },
    )
    position = environment.agent.pos.copy()
    velocity = environment.agent.vel.copy()
    step = environment.current_step
    oracle = ModelBasedEnergyRolloutEstimator(HeuristicGoalPolicy(), max_policy_steps=500)
    prediction = oracle.estimate_context(environment, environment.current_task_point)
    assert prediction.prediction > 0.0
    assert prediction.rollout_steps > 0
    assert np.array_equal(environment.agent.pos, position)
    assert np.array_equal(environment.agent.vel, velocity)
    assert environment.current_step == step
    environment.close()


def test_model_rollout_oracle_returns_zero_inside_goal_radius() -> None:
    environment = UAVEnergyDeliverySACEnv(phase=SACTrainingPhase.TD_PRETRAINING)
    environment.reset(seed=14)
    oracle = ModelBasedEnergyRolloutEstimator(HeuristicGoalPolicy(), max_policy_steps=100)
    prediction = oracle.estimate_context(environment, environment.agent.pos.copy())
    assert prediction == GoalEnergyPrediction(0.0, 0.0, 0, 0.0)
    environment.close()


def test_model_rollout_oracle_accepts_saturated_live_velocity() -> None:
    environment = UAVEnergyDeliverySACEnv(phase=SACTrainingPhase.TD_PRETRAINING)
    environment.reset(
        seed=15,
        options={
            "start_position": np.asarray([500.0, 500.0, 200.0], dtype=np.float32),
            "start_velocity": np.asarray([20.0, 0.0, 0.0], dtype=np.float32),
            "task_point": np.asarray([650.0, 500.0, 200.0], dtype=np.float32),
        },
    )
    environment.agent.vel[0] = np.nextafter(np.float32(20.0), np.float32(21.0))
    oracle = ModelBasedEnergyRolloutEstimator(HeuristicGoalPolicy(), max_policy_steps=500)
    prediction = oracle.estimate_context(environment, environment.current_task_point)
    assert prediction.prediction > 0.0
    environment.close()


def test_model_rollout_oracle_clone_preserves_obstacles_lidar_and_hocbf() -> None:
    environment = UAVEnergyDeliverySACEnv(
        phase=SACTrainingPhase.TD_PRETRAINING,
        lidar_enabled=True,
        lidar_horizontal_sectors=8,
        lidar_vertical_sectors=2,
        num_obstacles=0,
        cbf_enabled=True,
        hocbf_top_k=4,
        projection_geometry_enabled=True,
    )
    environment.reset(
        seed=17,
        options={
            "start_position": np.asarray([500.0, 500.0, 200.0], dtype=np.float32),
            "start_velocity": np.zeros(3, dtype=np.float32),
            "task_point": np.asarray([650.0, 500.0, 200.0], dtype=np.float32),
        },
    )
    environment.obstacles = [
        StaticCylinderObstacle(np.asarray([575.0, 550.0], dtype=np.float32), 20.0)
    ]
    environment._update_lidar()
    oracle = ModelBasedEnergyRolloutEstimator(HeuristicGoalPolicy(), max_policy_steps=500)
    clone = oracle._make_rollout_environment(
        environment,
        start_position=environment.agent.pos.copy(),
        start_velocity=environment.agent.vel.copy(),
        goal=environment.current_task_point.copy(),
    )
    assert clone is not environment
    assert clone.cbf_enabled is True
    assert clone.lidar_enabled is True
    assert clone.projection_geometry_enabled is True
    assert len(clone.obstacles) == 1
    assert clone.obstacles[0] is not environment.obstacles[0]
    np.testing.assert_allclose(clone.obstacles[0].pos, environment.obstacles[0].pos)
    clone.close()
    environment.close()


def test_model_rollout_oracle_supplies_direct_joint_mission_cost() -> None:
    environment = UAVEnergyDeliverySACEnv(
        phase=SACTrainingPhase.TD_PRETRAINING,
        charger_position=np.asarray([500.0, 500.0, 200.0], dtype=np.float32),
    )
    environment.reset(
        seed=19,
        options={
            "start_position": np.asarray([500.0, 500.0, 200.0], dtype=np.float32),
            "start_velocity": np.zeros(3, dtype=np.float32),
            "task_point": np.asarray([650.0, 500.0, 200.0], dtype=np.float32),
        },
    )
    oracle = ModelBasedEnergyRolloutEstimator(HeuristicGoalPolicy(), max_policy_steps=500)
    environment.bind_energy_learning(
        energy_estimator=oracle,
        goal_action_provider=lambda observation: np.zeros(3, dtype=np.float32),
        training_enabled=False,
    )
    estimate = environment.mission_energy_estimate()
    assert estimate.mission_prediction > 0.0
    assert estimate.mission_upper95 == pytest.approx(estimate.mission_prediction)
    assert estimate.mission_upper_bound_semantics == (
        "deterministic_oracle_joint_mission_cost"
    )
    assert estimate.mission_nominal_coverage_lower_bound == pytest.approx(1.0)
    assert oracle.update_count == 0
    assert len(oracle.replay) == 0
    environment.close()


def test_model_rollout_oracle_reuses_exact_task_suffix_without_changing_result() -> None:
    environment = UAVEnergyDeliverySACEnv(
        phase=SACTrainingPhase.TD_PRETRAINING,
        mission_decision_interval_policy_steps=10,
        charger_position=np.asarray([400.0, 400.0, 200.0], dtype=np.float32),
    )
    observation, _ = environment.reset(
        seed=23,
        options={
            "start_position": np.asarray([500.0, 500.0, 200.0], dtype=np.float32),
            "start_velocity": np.zeros(3, dtype=np.float32),
            "task_point": np.asarray([650.0, 500.0, 200.0], dtype=np.float32),
        },
    )
    policy = HeuristicGoalPolicy()
    cached_oracle = ModelBasedEnergyRolloutEstimator(policy, max_policy_steps=500)
    cached_oracle.estimate_mission_bundle(environment, environment.current_task_point)
    first_diagnostics = cached_oracle.cache_diagnostics()
    action, _ = policy.predict(observation, deterministic=True)
    environment.step(action)
    cached_bundle = cached_oracle.estimate_mission_bundle(
        environment,
        environment.current_task_point,
    )
    fresh_oracle = ModelBasedEnergyRolloutEstimator(
        policy,
        max_policy_steps=500,
        cache_mission_suffixes=False,
    )
    fresh_bundle = fresh_oracle.estimate_mission_bundle(
        environment,
        environment.current_task_point,
    )
    for cached_prediction, fresh_prediction in zip(cached_bundle, fresh_bundle):
        assert cached_prediction.prediction == pytest.approx(
            fresh_prediction.prediction,
            abs=1e-8,
        )
        assert cached_prediction.upper95 == pytest.approx(
            fresh_prediction.upper95,
            abs=1e-8,
        )
    diagnostics = cached_oracle.cache_diagnostics()
    assert first_diagnostics["mission_cache_misses"] == 1
    assert diagnostics["mission_cache_hits"] == 1
    assert diagnostics["mission_cache_misses"] == 1
    assert diagnostics["rollout_request_count"] == 4
    assert fresh_oracle.cache_diagnostics()["rollout_request_count"] == 3
    environment.close()


def test_model_rollout_oracle_disables_suffix_cache_for_per_step_decisions() -> None:
    environment = UAVEnergyDeliverySACEnv(
        phase=SACTrainingPhase.TD_PRETRAINING,
        mission_decision_interval_policy_steps=1,
    )
    environment.reset(seed=29)
    oracle = ModelBasedEnergyRolloutEstimator(HeuristicGoalPolicy(), max_policy_steps=1000)
    oracle.estimate_mission_bundle(environment, environment.current_task_point)
    oracle.estimate_mission_bundle(environment, environment.current_task_point)
    diagnostics = oracle.cache_diagnostics()
    assert diagnostics["mission_cache_hits"] == 0
    assert diagnostics["mission_cache_misses"] == 2
    assert diagnostics["rollout_request_count"] == 6
    environment.close()


class UpperOnlyEstimator:
    estimator_type = "upper_only_test"
    update_count = 0
    replay = ()

    def estimate_context(
        self,
        environment,
        goal,
        *,
        position=None,
        velocity=None,
        goal_type=None,
    ):
        del environment, goal, position, velocity, goal_type
        return GoalEnergyPrediction(1.0, 1_000.0)

    def predict_quantiles(self, state, action=None):
        del state, action
        return np.asarray([1.0, 1_000.0, 1_000.0, 1_000.0])


def test_switching_uses_upper95_not_point_prediction() -> None:
    environment = UAVEnergyDeliverySACEnv(
        phase=SACTrainingPhase.ENERGY_MANAGED,
        operational_energy_capacity=100.0,
    )
    environment.bind_energy_learning(
        energy_estimator=UpperOnlyEstimator(),
        goal_action_provider=lambda observation: np.zeros(3, dtype=np.float32),
        training_enabled=False,
    )
    environment.reset(seed=12)
    assert environment._refresh_mission_decision() is True
    assert environment.mode.value == "CHARGER_COMMITTED"
    event = environment.switching_events[-1]
    assert event["task_energy_prediction"] == 1.0
    assert event["task_energy_upper95"] == 1_000.0
    environment.close()


def test_sb3_can_bind_mc_estimator_without_online_updates() -> None:
    environment = UAVEnergyDeliverySACEnv()
    model = SAC("MlpPolicy", environment, buffer_size=16, learning_starts=16, device="cpu")
    estimator = EnergyToGoRegressor(battery_capacity=100.0, hidden_dim=8, device="cpu")
    environment.bind_energy_learning(
        energy_estimator=estimator,
        goal_action_provider=lambda observation: model.predict(observation, deterministic=True)[0],
        training_enabled=False,
    )
    observation, _ = environment.reset(seed=13)
    action, _ = model.predict(observation, deterministic=True)
    environment.step(action)
    assert estimator.update_count == 0
    assert len(estimator.replay) == 0
    environment.close()


def test_phase2_runner_initializes_task_counter(tmp_path: Path) -> None:
    args = parse_mc_args(
        [
            "--output-dir",
            str(tmp_path / "unused"),
            "--smoke",
            "--phase2-log-frequency",
            "1",
        ]
    )
    args.phase2_transition_budget = 2
    estimator = EnergyToGoRegressor(
        battery_capacity=100.0,
        hidden_dim=8,
        device="cpu",
    )
    summary = run_phase2(
        HeuristicGoalPolicy(),
        estimator,
        args,
        capacity=100.0,
        output=tmp_path / "phase2",
    )
    assert summary["actual_training_transitions"] == 2
    assert summary["total_delivery_tasks_completed"] == 0
