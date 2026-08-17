from __future__ import annotations

import numpy as np
import pytest

from envs.UAVEnergyDeliverySAC import UAVEnergyDeliverySACEnv
from experiments.energy_mc.conditional_risk import GoalRiskFeatureBuilder
from experiments.energy_mc.final_risk import (
    ContextualEnergyRegressor,
    MondrianMissionCalibration,
    MondrianTrajectoryCalibration,
    assert_decision_time_features,
    assert_fresh_test_isolation,
    attribute_switch,
    future_max_underestimation_target,
    unnecessary_return_indicator,
)
from review_bundle.safety.energy.mc_regression import EnergyToGoRegressor
from experiments.energy_mc.statistical_gate import (
    binomial_coverage_evidence,
    probability_empirical_meets_target,
    sample_size_power_table,
)
from experiments.energy_mc.trajectory_context import (
    ShortFrozenSACRolloutContext,
    ShortRolloutConfig,
)
from scripts.run_energy_risk_v5 import parser as v5_parser


class _ConstantPolicy:
    def __init__(self, action: tuple[float, float, float] = (0.5, 0.0, 0.0)) -> None:
        self.action = np.asarray(action, dtype=np.float32)

    def predict(self, observation: np.ndarray, deterministic: bool = True):
        assert deterministic
        batch = np.asarray(observation)
        return np.repeat(self.action[None, :], batch.shape[0], axis=0), None


def test_binomial_audit_matches_v4_worst_goal_group() -> None:
    evidence = binomial_coverage_evidence(215, 231, target=0.95)
    assert evidence.empirical_coverage == pytest.approx(0.9307359307359307)
    assert evidence.standard_error == pytest.approx(0.0167056, rel=1e-4)
    assert evidence.exact_interval_low == pytest.approx(0.889955, rel=1e-4)
    assert evidence.exact_interval_high == pytest.approx(0.959894, rel=1e-4)
    assert evidence.wilson_interval_low == pytest.approx(0.8905, rel=1e-3)
    assert evidence.wilson_interval_high == pytest.approx(0.9569, rel=1e-3)


def test_raw_empirical_gate_has_low_pass_probability_at_nominal_truth() -> None:
    probability = probability_empirical_meets_target(
        231,
        true_coverage=0.95,
        empirical_target=0.95,
    )
    assert probability == pytest.approx(0.512244, rel=1e-5)
    table = sample_size_power_table()
    assert [row["count"] for row in table] == [100, 200, 250, 500, 1000]
    assert all(
        row["true_coverage_for_90_percent_pass_probability"] > 0.95
        for row in table
    )


def test_future_max_target_is_suffix_max_within_each_trajectory() -> None:
    truth = np.asarray([1.0, 4.0, 2.0, 9.0, 3.0])
    point = np.asarray([0.0, 3.0, 2.0, 7.0, 2.5])
    ids = np.asarray([1, 1, 1, 2, 2])
    result = future_max_underestimation_target(truth, point, ids)
    np.testing.assert_allclose(result, [1.0, 1.0, 0.0, 2.0, 0.5])


def test_mondrian_goal_calibration_uses_direct_interaction_group() -> None:
    ids = np.repeat(np.arange(4), 2)
    truth = np.asarray([2, 1, 3, 1, 7, 2, 8, 2], dtype=np.float64)
    point = np.asarray([1, 1, 2, 1, 2, 2, 3, 2], dtype=np.float64)
    risk = np.zeros_like(truth)
    goals = np.asarray(["TASK"] * 4 + ["CHARGER"] * 4)
    buckets = np.asarray(["100-500"] * 4 + ["500-1500"] * 4)
    calibration = MondrianTrajectoryCalibration.fit(
        truth,
        point,
        risk,
        ids,
        goals,
        buckets,
        coverage=0.5,
        mode="additive",
        minimum_group_trajectories=2,
    )
    assert calibration.group_counts == {"CHARGER|500-1500": 2, "TASK|100-500": 2}
    task_value, task_supported, _ = calibration.value_for("TASK", 300.0)
    charger_value, charger_supported, _ = calibration.value_for(
        "TASK_ENDPOINT_TO_CHARGER", 900.0
    )
    assert task_supported and charger_supported
    assert charger_value > task_value


def test_scaled_mondrian_uses_normalized_trajectory_score() -> None:
    ids = np.repeat(np.arange(2), 2)
    truth = np.asarray([3.0, 1.0, 5.0, 1.0])
    point = np.asarray([1.0, 1.0, 1.0, 1.0])
    scale = np.asarray([2.0, 2.0, 2.0, 2.0])
    goals = np.asarray(["TASK"] * 4)
    buckets = np.asarray(["100-500"] * 4)
    calibration = MondrianTrajectoryCalibration.fit(
        truth,
        point,
        scale,
        ids,
        goals,
        buckets,
        coverage=0.5,
        mode="scaled",
        minimum_group_trajectories=2,
    )
    assert calibration.group_values["TASK|100-500"] == pytest.approx(2.0)


def test_mission_mondrian_calibrates_complete_mission_scores() -> None:
    ids = np.repeat(np.arange(4), 2)
    truth = np.asarray([5, 3, 6, 2, 9, 4, 10, 4], dtype=np.float64)
    upper = np.asarray([4, 3, 5, 2, 5, 4, 6, 4], dtype=np.float64)
    buckets = np.asarray(["100-500"] * 4 + [">4000"] * 4)
    calibration = MondrianMissionCalibration.fit(
        truth,
        upper,
        ids,
        buckets,
        coverage=0.5,
        minimum_group_missions=2,
    )
    assert calibration.distance_counts == {"100-500": 2, ">4000": 2}
    assert calibration.distance_corrections[">4000"] > calibration.distance_corrections["100-500"]


def test_posthoc_feature_leakage_is_rejected() -> None:
    assert_decision_time_features(["state7", "absolute_position", "rollout_path_efficiency"])
    with pytest.raises(ValueError, match="post-hoc"):
        assert_decision_time_features(["state7", "future_path_ratio"])
    with pytest.raises(ValueError, match="post-hoc"):
        assert_decision_time_features(["trajectory_steps"])


def test_fresh_test_ids_must_be_disjoint() -> None:
    assert_fresh_test_isolation({10, 11}, {"train": {1, 2}, "calibration": {3, 4}})
    with pytest.raises(ValueError, match="fresh-test leakage"):
        assert_fresh_test_isolation({10, 11}, {"train": {1, 10}})


@pytest.mark.parametrize(
    ("remaining", "point", "uncertainty", "reserve", "expected"),
    [
        (9.0, 10.0, 1.0, 2.0, "point_estimate"),
        (14.0, 10.0, 1.0, 5.0, "reserve"),
        (14.0, 10.0, 5.0, 1.0, "uncertainty_margin"),
        (15.0, 10.0, 3.0, 3.0, "both"),
    ],
)
def test_switch_attribution(remaining, point, uncertainty, reserve, expected) -> None:
    result = attribute_switch(
        remaining_energy=remaining,
        point_estimate=point,
        uncertainty_margin=uncertainty,
        reserve=reserve,
    )
    assert result.cause == expected


def test_unnecessary_return_metric_uses_same_reserve_on_truth_and_upper() -> None:
    result = unnecessary_return_indicator(
        np.asarray([15.0, 15.0, 12.5]),
        np.asarray([10.0, 12.0, 10.0]),
        np.asarray([12.0, 12.0, 12.0]),
        reserve=2.0,
    )
    np.testing.assert_array_equal(result, [False, False, True])


def test_feature_ablation_adds_compact_and_full_context_without_changing_history() -> None:
    states = np.zeros((2, 7), dtype=np.float32)
    positions = np.asarray([[100.0, 200.0, 20.0], [3900.0, 3800.0, 380.0]])
    expected = {
        "state7": 7,
        "absolute_position": 10,
        "boundary_distances": 13,
        "compact_position_boundary": 13,
        "compact_decision_context": 12,
        "position_and_boundary": 16,
    }
    for mode, width in expected.items():
        builder = GoalRiskFeatureBuilder(mode)
        assert builder.build(states, positions).shape == (2, width)


def test_contextual_point_warm_start_preserves_frozen_7d_predictions() -> None:
    frozen = EnergyToGoRegressor(battery_capacity=100.0, seed=7)
    contextual = ContextualEnergyRegressor(
        input_dim=12,
        battery_capacity=100.0,
        seed=8,
    )
    contextual.initialize_from_frozen_7d(frozen)
    states = np.random.default_rng(9).normal(size=(16, 7)).astype(np.float32)
    context = np.random.default_rng(10).normal(size=(16, 5)).astype(np.float32)
    np.testing.assert_allclose(
        contextual.predict_batch(np.concatenate([states, context], axis=1)),
        frozen.predict_batch(states),
        atol=1e-5,
    )


def test_short_rollout_context_does_not_mutate_live_environment() -> None:
    environment = UAVEnergyDeliverySACEnv()
    environment.reset(
        seed=123,
        options={
            "start_position": np.asarray([1000.0, 1000.0, 100.0]),
            "start_velocity": np.asarray([2.0, 0.0, 0.0]),
            "task_point": np.asarray([2000.0, 1000.0, 100.0]),
        },
    )
    position_before = environment.agent.pos.copy()
    velocity_before = environment.agent.vel.copy()
    goal_before = environment.current_task_point.copy()
    state = environment.energy_state_for_goal(goal_before)[None, :]
    builder = ShortFrozenSACRolloutContext(
        _ConstantPolicy(),
        ShortRolloutConfig(horizon=10),
    )
    features = builder.build(state, position_before[None, :])
    chunked = builder.build_chunked(
        np.repeat(state, 3, axis=0),
        np.repeat(position_before[None, :], 3, axis=0),
        chunk_size=2,
    )
    assert features.shape == (1, 9)
    np.testing.assert_allclose(chunked, np.repeat(features, 3, axis=0))
    assert np.all(np.isfinite(features))
    np.testing.assert_allclose(environment.agent.pos, position_before)
    np.testing.assert_allclose(environment.agent.vel, velocity_before)
    np.testing.assert_allclose(environment.current_task_point, goal_before)
    environment.close()


def test_v5_uses_independent_final_calibration_before_fresh_test(tmp_path) -> None:
    args = v5_parser().parse_args(["--output-dir", str(tmp_path / "run")])
    assert args.final_goal_calibration_trajectories == 2_500
    assert args.final_mission_calibration_trajectories == 1_000
    assert args.fresh_goal_trajectories == 5_000
    assert args.fresh_mission_trajectories == 3_000
    assert len(
        {
            args.model_seed,
            args.final_goal_calibration_seed,
            args.final_mission_calibration_seed,
            args.fresh_goal_seed,
            args.fresh_mission_seed,
        }
    ) == 5
