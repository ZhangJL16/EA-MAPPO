from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from envs.UAVEnergyDeliverySAC import SACTrainingPhase, UAVEnergyDeliverySACEnv
from experiments.energy_mc.conformal import (
    PackedMissionDataset,
    _zero_velocity_energy_state,
    mission_scores,
)
from experiments.energy_mc.core import (
    PackedEnergyDataset,
    generate_intersection_stratified_energy_goal_specs,
)
from review_bundle.safety.energy.mc_regression import (
    EnergyToGoRegressor,
    HierarchicalConformalCalibration,
    HierarchicalConformalEnergyEstimator,
    MissionConformalCalibration,
    finite_sample_conformal_margin,
)
from scripts.audit_formal_mc_guard_resets import audit_guard_resets
from scripts.train_uav_energy_delivery_sac import HeuristicGoalPolicy
from scripts.train_uav_energy_mc import parse_args as parse_mc_args
from scripts.train_uav_energy_mc import run_phase2
from scripts.run_uav_energy_conformal_v2 import parse_args as parse_conformal_args


def _hierarchical_calibration() -> HierarchicalConformalCalibration:
    predictions = np.zeros(6)
    targets = np.asarray([1.0, 2.0, 5.0, 1.0, 3.0, 4.0])
    trajectory_ids = np.asarray([0, 0, 1, 1, 2, 2])
    goal_types = np.asarray(["TASK", "TASK", "TASK", "TASK", "CHARGER", "CHARGER"])
    buckets = np.asarray(
        ["100-500", "100-500", ">4000", ">4000", "500-1500", "500-1500"]
    )
    return HierarchicalConformalCalibration.fit(
        predictions,
        targets,
        trajectory_ids,
        goal_types,
        buckets,
        coverage=0.5,
    )


def test_trajectory_conformal_score_uses_worst_underestimation() -> None:
    calibration = _hierarchical_calibration()
    assert calibration.num_trajectories == 3
    assert calibration.global_raw_margin == 4.0


def test_finite_sample_conformal_index_is_ceil_n_plus_one_coverage() -> None:
    margin, raw, rank = finite_sample_conformal_margin(
        np.asarray([-1.0, 2.0, 4.0]),
        coverage=0.8,
    )
    assert rank == 3
    assert raw == 4.0
    assert margin == 4.0


def test_group_margin_is_maximum_of_global_goal_and_distance() -> None:
    calibration = _hierarchical_calibration()
    expected = max(
        calibration.global_margin,
        calibration.goal_type_margins["TASK"],
        calibration.distance_margins[">4000"],
    )
    assert calibration.margin_for("TASK", 4500.0) == expected


def test_intersection_stratified_splits_are_disjoint_and_cover_feasible_cells() -> None:
    charger = np.asarray([2000.0, 2000.0, 200.0], dtype=np.float32)
    calibration = generate_intersection_stratified_energy_goal_specs(
        num_trajectories=65,
        seed=101,
        charger_position=charger,
        trajectory_id_offset=1_000,
    )
    test = generate_intersection_stratified_energy_goal_specs(
        num_trajectories=65,
        seed=202,
        charger_position=charger,
        trajectory_id_offset=2_000,
    )
    assert not ({row.trajectory_id for row in calibration} & {row.trajectory_id for row in test})
    cells = {(row.goal_type, row.distance_bucket) for row in calibration}
    assert len(cells) == 13
    assert ("CHARGER", ">4000") not in cells


def test_overlapping_trajectory_ids_cannot_enter_combined_calibration() -> None:
    def dataset(trajectory_id: int) -> PackedEnergyDataset:
        return PackedEnergyDataset(
            states=np.zeros((1, 7), dtype=np.float32),
            targets=np.ones(1, dtype=np.float32),
            step_energy=np.ones(1, dtype=np.float32),
            transition_dt=np.full(1, 0.2, dtype=np.float32),
            trajectory_ids=np.asarray([trajectory_id], dtype=np.int64),
            goal_types=np.asarray(["TASK"]),
            distance_buckets=np.asarray(["100-500"]),
            boundary_contact_trajectories=np.zeros(1, dtype=bool),
            metadata=[],
        )

    with pytest.raises(ValueError, match="overlapping trajectory ids"):
        PackedEnergyDataset.concatenate([dataset(7), dataset(7)])


def test_mission_return_prediction_state_uses_nominal_zero_velocity_endpoint() -> None:
    state = _zero_velocity_energy_state(
        np.asarray([1000.0, 1000.0, 200.0], dtype=np.float32),
        np.asarray([2000.0, 2000.0, 200.0], dtype=np.float32),
    )
    assert state.shape == (7,)
    assert state[:3].tolist() == [0.0, 0.0, 0.0]
    assert state[3:6] == pytest.approx([2**-0.5, 2**-0.5, 0.0])
    assert state[-1] == pytest.approx(
        np.sqrt(2.0) * 1000.0 / np.linalg.norm([4000.0, 4000.0, 400.0])
    )


def test_mission_residual_and_upper_bound_use_mission_calibration() -> None:
    point = EnergyToGoRegressor(battery_capacity=100.0, hidden_dim=8, device="cpu")
    point.predict_batch = lambda states: np.full(len(states), 2.0)  # type: ignore[method-assign]
    dataset = PackedMissionDataset(
        task_states=np.zeros((3, 7), dtype=np.float32),
        return_after_states=np.zeros((3, 7), dtype=np.float32),
        true_mission_energy=np.asarray([7.0, 6.0, 5.0], dtype=np.float32),
        mission_ids=np.asarray([1, 1, 1]),
        initial_distance_buckets=np.asarray(["100-500"] * 3),
        metadata=[],
    )
    prediction, scores, buckets = mission_scores(point, dataset)
    assert prediction.tolist() == [4.0, 4.0, 4.0]
    assert scores.tolist() == [3.0]
    mission = MissionConformalCalibration.fit(scores, buckets, coverage=0.5)
    estimator = HierarchicalConformalEnergyEstimator(point, _hierarchical_calibration(), mission)
    bound = estimator.estimate_mission(2.0, 2.0, task_distance=200.0)
    assert bound.prediction == 4.0
    assert bound.upper95 == 7.0


def test_return_now_uses_charger_group_calibration() -> None:
    point = EnergyToGoRegressor(battery_capacity=100.0, hidden_dim=8, device="cpu")
    point.predict_batch = lambda states: np.full(len(states), 2.0)  # type: ignore[method-assign]
    mission = MissionConformalCalibration.fit(
        np.asarray([1.0]),
        np.asarray(["100-500"]),
        coverage=0.5,
    )
    calibration = _hierarchical_calibration()
    estimator = HierarchicalConformalEnergyEstimator(point, calibration, mission)
    environment = UAVEnergyDeliverySACEnv(phase=SACTrainingPhase.TD_PRETRAINING)
    environment.reset(
        seed=9,
        options={
            "start_position": np.asarray([1000.0, 1000.0, 200.0], dtype=np.float32),
            "start_velocity": np.zeros(3, dtype=np.float32),
            "task_point": np.asarray([1200.0, 1000.0, 200.0], dtype=np.float32),
        },
    )
    distance = float(np.linalg.norm(environment.charger_position - environment.agent.pos))
    result = estimator.estimate_context(
        environment,
        environment.charger_position,
        goal_type="CHARGER",
    )
    assert result.upper95 - result.prediction == pytest.approx(
        calibration.margin_for("CHARGER", distance)
    )
    environment.close()


def test_group_conformal_checkpoint_round_trip_preserves_margins(tmp_path: Path) -> None:
    point = EnergyToGoRegressor(battery_capacity=100.0, hidden_dim=8, device="cpu")
    mission = MissionConformalCalibration.fit(
        np.asarray([1.0, 2.0]),
        np.asarray(["100-500", "500-1500"]),
        coverage=0.5,
    )
    estimator = HierarchicalConformalEnergyEstimator(
        point,
        _hierarchical_calibration(),
        mission,
    )
    checkpoint = tmp_path / "group_conformal.pt"
    estimator.save(checkpoint)
    restored = HierarchicalConformalEnergyEstimator.load(checkpoint, device="cpu")
    assert restored.trajectory_calibration.as_dict() == estimator.trajectory_calibration.as_dict()
    assert restored.mission_calibration.as_dict() == estimator.mission_calibration.as_dict()


def test_guard_truncation_is_partial_not_completed_cycle(tmp_path: Path) -> None:
    args = parse_mc_args(["--output-dir", str(tmp_path / "unused"), "--smoke"])
    args.phase2_transition_budget = 2
    args.phase2_episode_max_steps = 2
    args.phase2_log_frequency = 1
    estimator = EnergyToGoRegressor(battery_capacity=100.0, hidden_dim=8, device="cpu")
    estimator.predict_batch = lambda states: np.zeros(len(np.atleast_2d(states)))  # type: ignore[method-assign]
    summary = run_phase2(
        HeuristicGoalPolicy(),
        estimator,
        args,
        capacity=100.0,
        output=tmp_path / "phase2",
    )
    assert summary["completed_recharge_cycles"] == 0
    assert summary["guard_truncated_segments"] == 1
    assert summary["partial_battery_segments"] == 1


def test_historical_guard_audit_proves_full_battery_reset(tmp_path: Path) -> None:
    run = tmp_path / "run"
    (run / "phase2").mkdir(parents=True)
    record = {
        "emergency_time_limit": True,
        "remaining_energy_at_cycle_end": 25.0,
        "battery_cycle_id": 3,
        "cycle_end_transition": 20_000,
        "return_success": False,
    }
    (run / "phase2" / "battery_cycles.jsonl").write_text(
        __import__("json").dumps(record) + "\n",
        encoding="utf-8",
    )
    audit = audit_guard_resets(run, battery_capacity=100.0)
    assert audit["FORMAL_PHASE2_HAS_ARTIFICIAL_BATTERY_RESETS"] is True
    assert audit["events"][0]["energy_before_guard"] == 25.0
    assert audit["events"][0]["energy_immediately_after_reset"] == 100.0
    assert audit["events"][0]["battery_cycle_id_after"] == 0


def test_formal_conformal_runner_rejects_phase2_guard_within_budget() -> None:
    with pytest.raises(SystemExit):
        parse_conformal_args(
            [
                "--output-dir",
                "unused",
                "--phase2-transition-budget",
                "500000",
                "--phase2-episode-max-steps",
                "500000",
            ]
        )
