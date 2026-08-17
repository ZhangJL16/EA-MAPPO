from __future__ import annotations

import numpy as np

from envs.UAVEnergyDeliverySAC import SACTrainingPhase, UAVEnergyDeliverySACEnv
from experiments.energy_mc.adaptive_analysis import (
    evaluate_goal_upper,
    evaluate_mission_upper,
    state_residual_by_remaining_distance,
    stratified_mission_split,
)
from experiments.energy_mc.adaptive_uncertainty import (
    HeteroscedasticResidualModel,
    MCSupervisedQuantileModel,
    TrajectoryConformalCorrection,
)
from experiments.energy_mc.conditional_risk import (
    GoalRiskFeatureBuilder,
    HierarchicalScaledConformalCalibration,
    ScaledTrajectoryConformalCorrection,
    SeparatedGoalMissionRiskEstimator,
    positive_underestimation_target,
    trajectory_balanced_state_indices,
)
from experiments.energy_mc.conformal import PackedMissionDataset
from experiments.energy_mc.core import PackedEnergyDataset
from review_bundle.safety.energy.mc_regression import GoalEnergyPrediction
from review_bundle.safety.switching import SortieMode
from scripts.run_energy_uncertainty_v4 import assert_disjoint_named_sets


def test_positive_residual_target_keeps_only_underestimation() -> None:
    truth = np.asarray([1.0, 3.0, 2.0])
    point = np.asarray([2.0, 1.0, 2.5])
    np.testing.assert_allclose(
        positive_underestimation_target(truth, point),
        np.asarray([0.0, 2.0, 0.0]),
    )


def test_goal_risk_feature_ablation_contracts() -> None:
    states = np.zeros((2, 7), dtype=np.float32)
    positions = np.asarray([[1000.0, 2000.0, 100.0], [3000.0, 1000.0, 300.0]])
    dimensions = {
        "state7": 7,
        "absolute_position": 10,
        "boundary_distances": 13,
        "compact_position_boundary": 13,
    }
    for mode, dimension in dimensions.items():
        features = GoalRiskFeatureBuilder(mode).build(states, positions)
        assert features.shape == (2, dimension)
        np.testing.assert_allclose(features[:, :7], states)
    absolute = GoalRiskFeatureBuilder("absolute_position").build(states, positions)
    np.testing.assert_allclose(absolute[0, 7:], [0.25, 0.5, 0.25])


def test_scaled_conformal_uses_trajectory_max_normalized_underestimation() -> None:
    truth = np.asarray([2.0, 5.0, 2.0, 6.0])
    point = np.asarray([1.0, 1.0, 1.0, 2.0])
    scale = np.asarray([1.0, 2.0, 1.0, 4.0])
    ids = np.asarray([10, 10, 11, 11])
    correction = ScaledTrajectoryConformalCorrection.fit(
        truth,
        point,
        scale,
        ids,
        coverage=0.5,
    )
    assert correction.raw_multiplier == 2.0
    assert correction.num_trajectories == 2


def test_hierarchical_scaled_conformal_uses_intersection_when_available() -> None:
    ids = np.repeat(np.arange(80), 2)
    goal_types = np.repeat(np.asarray(["TASK"] * 40 + ["CHARGER"] * 40), 2)
    buckets = np.repeat(np.asarray(["100-500"] * 20 + ["500-1500"] * 20 + ["100-500"] * 20 + ["500-1500"] * 20), 2)
    point = np.zeros(160)
    scale = np.ones(160)
    truth = np.where(goal_types == "TASK", 2.0, 1.0)
    calibration = HierarchicalScaledConformalCalibration.fit(
        truth,
        point,
        scale,
        ids,
        goal_types,
        buckets,
        coverage=0.9,
        minimum_group_trajectories=10,
    )
    assert calibration.multiplier_for("TASK", 200.0) == 2.0
    assert calibration.multiplier_for("CHARGER", 200.0) == 1.0


def test_v4_split_audit_rejects_trajectory_leakage() -> None:
    assert_disjoint_named_sets({"train": {1, 2}, "calibration": {3}, "test": {4, 5}})
    with np.testing.assert_raises(RuntimeError):
        assert_disjoint_named_sets({"train": {1, 2}, "test": {2, 3}})


def test_trajectory_balanced_sampling_caps_each_trajectory_without_crossing_splits() -> None:
    ids = np.asarray([10] * 3 + [11] * 10 + [12] * 2)
    selected = trajectory_balanced_state_indices(ids, max_states_per_trajectory=4)
    selected_ids, counts = np.unique(ids[selected], return_counts=True)
    np.testing.assert_array_equal(selected_ids, [10, 11, 12])
    np.testing.assert_array_equal(counts, [3, 4, 2])
    assert selected[0] == 0
    assert selected[-1] == len(ids) - 1


def test_adaptive_correction_uses_trajectory_max_underestimation() -> None:
    truth = np.asarray([1.0, 3.0, 2.0, 8.0])
    base = np.asarray([1.0, 1.0, 2.0, 4.0])
    ids = np.asarray([10, 10, 11, 11])
    correction = TrajectoryConformalCorrection.fit(
        truth,
        base,
        ids,
        coverage=0.5,
    )
    assert correction.num_trajectories == 2
    assert correction.raw_correction == 4.0


def test_cqr_correction_can_tighten_overconservative_upper_quantile() -> None:
    truth = np.asarray([1.0, 2.0, 3.0, 4.0])
    base = truth + 2.0
    ids = np.asarray([10, 10, 11, 11])
    correction = TrajectoryConformalCorrection.fit(
        truth,
        base,
        ids,
        coverage=0.5,
        allow_negative=True,
    )
    assert correction.raw_correction == -2.0
    assert correction.correction == -2.0


def test_quantile_predictions_are_rearranged_non_crossing() -> None:
    model = MCSupervisedQuantileModel(
        battery_capacity=100.0,
        hidden_dim=8,
        batch_size=4,
        device="cpu",
    )
    predictions = model.predict_quantiles(np.zeros((4, 7), dtype=np.float32))
    assert np.all(predictions[:, 1:] >= predictions[:, :-1])
    assert np.all(predictions >= 0.0)


def test_heteroscedastic_scale_is_positive_and_input_dependent() -> None:
    model = HeteroscedasticResidualModel(
        hidden_dim=8,
        batch_size=4,
        device="cpu",
    )
    states = np.zeros((8, 7), dtype=np.float32)
    residuals = np.linspace(-0.2, 0.2, 8, dtype=np.float32)
    model.fit(states, residuals, states, residuals, max_epochs=2, patience=2)
    scale = model.predict_scale(states)
    assert scale.shape == (8,)
    assert np.all(scale > 0.0)


def test_stratified_mission_split_has_no_mission_leakage() -> None:
    rows = 15
    mission_ids = np.repeat(np.arange(rows), 2)
    buckets = np.repeat(
        np.asarray(["100-500", "500-1500", "1500-2500"] * 5),
        2,
    )
    dataset = PackedMissionDataset(
        task_states=np.zeros((rows * 2, 7), dtype=np.float32),
        return_after_states=np.zeros((rows * 2, 7), dtype=np.float32),
        true_mission_energy=np.ones(rows * 2, dtype=np.float32),
        mission_ids=mission_ids,
        initial_distance_buckets=buckets,
        metadata=[
            {
                "mission_id": identifier,
                "initial_task_distance_bucket": ["100-500", "500-1500", "1500-2500"][identifier % 3],
            }
            for identifier in range(rows)
        ],
    )
    train, validation, calibration = stratified_mission_split(dataset, seed=7)
    assert not (train.successful_mission_ids & validation.successful_mission_ids)
    assert not (train.successful_mission_ids & calibration.successful_mission_ids)
    assert not (validation.successful_mission_ids & calibration.successful_mission_ids)
    assert train.successful_mission_ids | validation.successful_mission_ids | calibration.successful_mission_ids == set(range(rows))


def test_goal_upper_metrics_distinguish_state_and_trajectory_coverage() -> None:
    dataset = PackedEnergyDataset(
        states=np.zeros((4, 7), dtype=np.float32),
        targets=np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        step_energy=np.ones(4, dtype=np.float32),
        transition_dt=np.full(4, 0.2, dtype=np.float32),
        trajectory_ids=np.asarray([0, 0, 1, 1]),
        goal_types=np.asarray(["TASK"] * 4),
        distance_buckets=np.asarray(["100-500"] * 4),
        boundary_contact_trajectories=np.zeros(4, dtype=bool),
        metadata=[],
    )
    point = dataset.targets.astype(np.float64)
    upper = np.asarray([1.0, 2.0, 3.0, 3.0])
    result = evaluate_goal_upper(dataset, point, upper)["overall"]
    assert result["pointwise_state_coverage"] == 0.75
    assert result["whole_trajectory_simultaneous_coverage"] == 0.5


def test_remaining_distance_diagnostic_preserves_state_and_trajectory_counts() -> None:
    dataset = PackedEnergyDataset(
        states=np.asarray(
            [
                [0, 0, 0, 1, 0, 0, 0.01],
                [0, 0, 0, 1, 0, 0, 0.20],
                [0, 0, 0, 1, 0, 0, 0.80],
            ],
            dtype=np.float32,
        ),
        targets=np.asarray([0.8, 1.0, 2.0], dtype=np.float32),
        step_energy=np.ones(3, dtype=np.float32),
        transition_dt=np.full(3, 0.2, dtype=np.float32),
        trajectory_ids=np.asarray([10, 10, 11], dtype=np.int64),
        goal_types=np.asarray(["TASK", "TASK", "CHARGER"]),
        distance_buckets=np.asarray(["100-500", "100-500", "2500-4000"]),
        boundary_contact_trajectories=np.zeros(3, dtype=bool),
        metadata=[],
    )
    summary = state_residual_by_remaining_distance(
        dataset,
        np.asarray([0.5, 1.1, 1.0], dtype=np.float64),
    )
    assert int(summary["state_count"].sum()) == 3
    assert int(summary["trajectory_count"].max()) == 1
    assert np.isclose(float(summary["maximum_underestimation"].max()), 1.0)
    assert set(summary["remaining_distance_bucket"].astype(str)).issubset(
        {"0-100", "100-500", "500-1500", "1500-2500", "2500-4000", ">4000"}
    )


def test_environment_prefers_direct_mission_context_when_available() -> None:
    class DirectMissionEstimator:
        def estimate_context(self, environment, goal, **kwargs):
            del environment, goal, kwargs
            return GoalEnergyPrediction(1.0, 2.0)

        def estimate_mission_context(self, environment, task_goal):
            del environment, task_goal
            return GoalEnergyPrediction(2.0, 7.0)

    environment = UAVEnergyDeliverySACEnv(phase=SACTrainingPhase.TD_PRETRAINING)
    environment.reset(seed=4)
    environment.bind_energy_learning(
        energy_estimator=DirectMissionEstimator(),
        goal_action_provider=lambda observation: np.zeros(3, dtype=np.float32),
        training_enabled=False,
    )
    estimate = environment.mission_energy_estimate()
    assert estimate.mission_prediction == 2.0
    assert estimate.mission_upper95 == 7.0
    environment.close()


def test_mission_tradeoff_reports_acceptance_and_false_rejection() -> None:
    dataset = PackedMissionDataset(
        task_states=np.zeros((2, 7), dtype=np.float32),
        return_after_states=np.zeros((2, 7), dtype=np.float32),
        true_mission_energy=np.asarray([2.0, 4.0], dtype=np.float32),
        mission_ids=np.asarray([0, 1]),
        initial_distance_buckets=np.asarray(["100-500", "100-500"]),
        metadata=[],
    )
    result = evaluate_mission_upper(
        dataset,
        point=np.asarray([2.0, 4.0]),
        upper=np.asarray([4.0, 5.0]),
        battery_capacity=10.0,
        reserve_fraction=0.0,
        remaining_energy_samples=np.asarray([3.0, 6.0]),
    )["overall"]
    assert result["true_feasible_opportunity_rate"] == 1.0
    assert result["estimated_task_acceptance_rate"] == 0.5
    assert result["unnecessary_return_proxy_rate"] == 0.5


def test_goal_and_mission_uncertainty_are_separate_and_switching_uses_direct_mission() -> None:
    class PointEstimator:
        def predict(self, state):
            del state
            return 1.0

        def checkpoint_payload(self):
            return {"model_type": "test_point"}

    goal_model = HeteroscedasticResidualModel(input_dim=7, hidden_dim=8, device="cpu")
    goal_model.predict_scale = lambda features: np.ones(len(features), dtype=np.float64)
    mission_model = HeteroscedasticResidualModel(input_dim=14, hidden_dim=8, device="cpu")
    mission_model.upper_offset = lambda features, coverage: np.full(len(features), 4.0)
    estimator = SeparatedGoalMissionRiskEstimator(
        PointEstimator(),
        GoalRiskFeatureBuilder("state7"),
        goal_model,
        ScaledTrajectoryConformalCorrection(0.95, 1.0, 1.0, 1, 10),
        mission_model,
        TrajectoryConformalCorrection(0.95, 1.0, 1.0, 1, 10),
        coverage=0.95,
    )
    environment = UAVEnergyDeliverySACEnv(
        operational_energy_capacity=10.0,
        energy_reserve_fraction=0.10,
    )
    environment.bind_energy_learning(
        energy_estimator=estimator,
        goal_action_provider=lambda observation: np.zeros(3, dtype=np.float32),
        training_enabled=False,
    )
    environment.reset(seed=91)
    estimate = environment.mission_energy_estimate()
    assert estimate.task_upper95 == 2.0
    assert estimate.return_now_upper95 == 2.0
    assert estimate.component_upper95_sum == 4.0
    assert estimate.mission_upper95 == 7.0
    environment.agent.energy = 7.5
    assert environment._refresh_mission_decision() is True
    assert environment.mode is SortieMode.CHARGER_COMMITTED
    assert environment.energy_reserve == 1.0
    environment.close()
