from __future__ import annotations

import argparse
import json
import os
import subprocess
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from experiments.energy_mc.adaptive_analysis import (
    contiguous_trajectory_slices,
    mission_component_predictions,
)
from experiments.energy_mc.adaptive_uncertainty import (
    HeteroscedasticResidualModel,
    chunked_point_predictions,
)
from experiments.energy_mc.conformal import PackedMissionDataset
from experiments.energy_mc.conditional_risk import (
    GoalRiskFeatureBuilder,
    PositiveResidualQuantileModel,
)
from experiments.energy_mc.core import PackedEnergyDataset
from experiments.energy_mc.final_risk import (
    MondrianGoalMissionRiskEstimator,
    MondrianMissionCalibration,
    MondrianTrajectoryCalibration,
    assert_fresh_test_isolation,
    canonical_goal_type,
    primary_goal_group,
)
from experiments.energy_mc.statistical_gate import (
    BinomialCoverageEvidence,
    binomial_coverage_evidence,
    holm_undercoverage_audit,
)
from review_bundle.safety.energy.mc_regression import (
    EnergyToGoRegressor,
    HierarchicalConformalEnergyEstimator,
)
from scripts.explore_adaptive_energy_uncertainty import environment_args, point_callable
from scripts.run_energy_risk_method_ladder import goal_metrics, mission_metrics
from scripts.run_energy_uncertainty_v4 import (
    BATTERY_CAPACITY,
    D_MAX,
    GROUP_BASELINE_CHECKPOINT,
    MISSION_MODEL_CHECKPOINT,
    POINT_CHECKPOINT,
    SAC_CHECKPOINT,
    SOURCE,
    V2,
    load_jsonl,
    phase2_counterfactual_audit,
)
from scripts.run_uav_energy_conformal_v2 import (
    collect_intersection_split,
    collect_mission_split,
)
from scripts.train_uav_energy_mc import (
    file_sha256,
    git_sha,
    load_frozen_sac,
    run_phase2,
)


ROOT = Path(__file__).resolve().parents[1]
DEVELOPMENT = ROOT / "artifacts/energy_risk_v5_development"
METHOD_LADDER = DEVELOPMENT / "method_ladder"
GOAL_MODEL_PATHS = {
    "k1_state": METHOD_LADDER / "models/k1_state_residual.pt",
    "k2_suffix": METHOD_LADDER / "models/k2_suffix_max_residual.pt",
}
DISTANCE_BUCKETS = ("100-500", "500-1500", "1500-2500", "2500-4000", ">4000")
EXPECTED_GOAL_INTERACTIONS = tuple(
    [f"TASK|{bucket}" for bucket in DISTANCE_BUCKETS]
    + [f"CHARGER|{bucket}" for bucket in DISTANCE_BUCKETS[:-1]]
)
SAFETY_TARGET = 0.95


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_value(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def report_stage(output: Path, stage: str, **details: object) -> None:
    payload = {"timestamp": utc_now(), "stage": stage, **details}
    with (output / "progress.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(json_value(payload), sort_keys=True) + "\n")
    print(f"[energy-risk-v5] {stage}", flush=True)


def assert_tracked_worktree_clean() -> None:
    result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    if result.stdout.strip():
        raise RuntimeError(
            "fresh v5 requires a clean tracked worktree; commit code changes first"
        )


def load_development_sets() -> tuple[
    PackedEnergyDataset,
    PackedMissionDataset,
    PackedMissionDataset,
]:
    goal_calibration = PackedEnergyDataset.concatenate(
        [
            PackedEnergyDataset.load(SOURCE / "energy_dataset/calibration"),
            PackedEnergyDataset.load(V2 / "calibration_v2/new_trajectories"),
        ]
    )
    mission_calibration = PackedMissionDataset.load(V2 / "mission_calibration/trajectories")
    mission_diagnostic = PackedMissionDataset.load(V2 / "mission_test_v2/trajectories")
    return goal_calibration, mission_calibration, mission_diagnostic


def fit_goal_calibration(
    point_model: EnergyToGoRegressor,
    risk_model: PositiveResidualQuantileModel,
    dataset: PackedEnergyDataset,
    *,
    coverage: float,
) -> tuple[MondrianTrajectoryCalibration, np.ndarray, np.ndarray]:
    point = chunked_point_predictions(point_model, dataset.states)
    builder = GoalRiskFeatureBuilder("compact_decision_context")
    risk = risk_model.predict_level(builder.build_from_dataset(dataset), coverage)
    calibration = MondrianTrajectoryCalibration.fit(
        dataset.targets,
        point,
        risk,
        dataset.trajectory_ids,
        dataset.goal_types,
        dataset.distance_buckets,
        coverage=coverage,
        mode="additive",
        minimum_group_trajectories=100,
    )
    missing = sorted(set(EXPECTED_GOAL_INTERACTIONS) - calibration.group_values.keys())
    if missing:
        raise RuntimeError(f"Goal primary groups lack direct calibration: {missing}")
    return calibration, point, risk


def fit_mission_calibration(
    point_model: EnergyToGoRegressor,
    mission_model: HeteroscedasticResidualModel,
    dataset: PackedMissionDataset,
    *,
    coverage: float,
) -> tuple[MondrianMissionCalibration, np.ndarray, np.ndarray]:
    _, _, point = mission_component_predictions(point_callable(point_model), dataset)
    features = np.concatenate([dataset.task_states, dataset.return_after_states], axis=1)
    risk = mission_model.upper_offset(features, coverage)
    calibration = MondrianMissionCalibration.fit(
        dataset.true_mission_energy,
        point + risk,
        dataset.mission_ids,
        dataset.initial_distance_buckets,
        coverage=coverage,
        minimum_group_missions=100,
    )
    missing = sorted(set(DISTANCE_BUCKETS) - calibration.distance_corrections.keys())
    if missing:
        raise RuntimeError(f"Mission primary groups lack direct calibration: {missing}")
    return calibration, point, risk


def evidence(successes: int, count: int) -> BinomialCoverageEvidence:
    return binomial_coverage_evidence(successes, count, target=SAFETY_TARGET)


def goal_coverage_report(
    dataset: PackedEnergyDataset,
    upper: np.ndarray,
) -> tuple[dict[str, object], dict[str, BinomialCoverageEvidence]]:
    rows: list[dict[str, object]] = []
    for trajectory_id, selected in contiguous_trajectory_slices(dataset.trajectory_ids):
        goal_type = canonical_goal_type(str(dataset.goal_types[selected.start]))
        distance = str(dataset.distance_buckets[selected.start])
        rows.append(
            {
                "trajectory_id": int(trajectory_id),
                "goal_type": goal_type,
                "distance": distance,
                "interaction": primary_goal_group(goal_type, distance),
                "covered": bool(np.all(dataset.targets[selected] <= upper[selected])),
            }
        )

    def summarize(selected: list[dict[str, object]]) -> BinomialCoverageEvidence:
        return evidence(sum(bool(row["covered"]) for row in selected), len(selected))

    groups: dict[str, BinomialCoverageEvidence] = {"overall": summarize(rows)}
    for goal_type in ("TASK", "CHARGER"):
        groups[f"goal_type:{goal_type}"] = summarize(
            [row for row in rows if row["goal_type"] == goal_type]
        )
    for bucket in DISTANCE_BUCKETS:
        selected = [row for row in rows if row["distance"] == bucket]
        if selected:
            groups[f"distance:{bucket}"] = summarize(selected)
    for interaction in EXPECTED_GOAL_INTERACTIONS:
        selected = [row for row in rows if row["interaction"] == interaction]
        if not selected:
            raise RuntimeError(f"fresh v5 lacks Goal primary group {interaction}")
        groups[f"interaction:{interaction}"] = summarize(selected)
    tested = {
        name: result
        for name, result in groups.items()
        if name != "overall"
    }
    return (
        {
            "groups": {name: result.as_dict() for name, result in groups.items()},
            "holm_undercoverage_test": holm_undercoverage_audit(tested),
            "raw_empirical_target_is_not_the_sole_gate": True,
        },
        groups,
    )


def mission_coverage_report(
    dataset: PackedMissionDataset,
    upper: np.ndarray,
) -> tuple[dict[str, object], dict[str, BinomialCoverageEvidence]]:
    rows: list[dict[str, object]] = []
    for mission_id, selected in contiguous_trajectory_slices(dataset.mission_ids):
        rows.append(
            {
                "mission_id": int(mission_id),
                "distance": str(dataset.initial_distance_buckets[selected.start]),
                "covered": bool(
                    np.all(dataset.true_mission_energy[selected] <= upper[selected])
                ),
            }
        )

    def summarize(selected: list[dict[str, object]]) -> BinomialCoverageEvidence:
        return evidence(sum(bool(row["covered"]) for row in selected), len(selected))

    groups: dict[str, BinomialCoverageEvidence] = {"overall": summarize(rows)}
    for bucket in DISTANCE_BUCKETS:
        selected = [row for row in rows if row["distance"] == bucket]
        if not selected:
            raise RuntimeError(f"fresh v5 lacks Mission primary group {bucket}")
        groups[f"distance:{bucket}"] = summarize(selected)
    tested = {name: result for name, result in groups.items() if name != "overall"}
    return (
        {
            "groups": {name: result.as_dict() for name, result in groups.items()},
            "holm_undercoverage_test": holm_undercoverage_audit(tested),
            "raw_empirical_target_is_not_the_sole_gate": True,
        },
        groups,
    )


def phase2_switch_attribution(output: Path) -> dict[str, object]:
    events = load_jsonl(output / "switching_events.jsonl")
    counts = {"point_estimate": 0, "uncertainty_margin": 0, "reserve": 0, "both": 0}
    reserve_ratios = []
    for event in events:
        point = float(event["mission_energy_prediction"])
        upper = float(event["mission_energy_upper95"])
        uncertainty = max(0.0, upper - point)
        reserve = float(event["reserve"])
        remaining = float(event["remaining_energy"])
        point_trigger = remaining <= point
        without_uncertainty = remaining <= point + reserve
        without_reserve = remaining <= point + uncertainty
        if point_trigger:
            cause = "point_estimate"
        elif without_uncertainty and not without_reserve:
            cause = "reserve"
        elif without_reserve and not without_uncertainty:
            cause = "uncertainty_margin"
        else:
            cause = "both"
        counts[cause] += 1
        reserve_ratios.append(reserve / max(uncertainty, 1e-12))
    return {
        "num_switches": len(events),
        "cause_counts": counts,
        "cause_fractions": {
            name: (0.0 if not events else value / len(events))
            for name, value in counts.items()
        },
        "mean_reserve_to_uncertainty_ratio": (
            None if not reserve_ratios else float(np.mean(reserve_ratios))
        ),
    }


def run_paired_phase2(
    output: Path,
    policy,
    candidate: MondrianGoalMissionRiskEstimator,
    args: argparse.Namespace,
) -> dict[str, object]:
    baseline = HierarchicalConformalEnergyEstimator.load(
        GROUP_BASELINE_CHECKPOINT,
        device=args.device,
    )
    results: dict[str, object] = {}
    for seed in args.phase2_seeds:
        seed_output = output / "phase2_100k" / f"seed{seed}"
        baseline_output = seed_output / "baseline_point_group"
        candidate_output = seed_output / "candidate_v5"
        baseline_output.mkdir(parents=True)
        candidate_output.mkdir(parents=True)
        phase_args = environment_args(seed=seed, phase2_budget=args.phase2_transitions)
        baseline_summary = run_phase2(
            policy,
            baseline,
            phase_args,
            capacity=BATTERY_CAPACITY,
            output=baseline_output,
        )
        baseline_summary.update(
            phase2_counterfactual_audit(
                policy,
                phase_args,
                baseline_output,
                capacity=BATTERY_CAPACITY,
            )
        )
        baseline_summary["switch_attribution"] = phase2_switch_attribution(baseline_output)
        write_json(baseline_output / "summary.json", baseline_summary)
        candidate_summary = run_phase2(
            policy,
            candidate,
            phase_args,
            capacity=BATTERY_CAPACITY,
            output=candidate_output,
        )
        candidate_summary.update(
            phase2_counterfactual_audit(
                policy,
                phase_args,
                candidate_output,
                capacity=BATTERY_CAPACITY,
            )
        )
        candidate_summary["switch_attribution"] = phase2_switch_attribution(candidate_output)
        write_json(candidate_output / "summary.json", candidate_summary)
        baseline_stream = load_jsonl(baseline_output / "completed_task_stream.jsonl")
        candidate_stream = load_jsonl(candidate_output / "completed_task_stream.jsonl")
        common = min(len(baseline_stream), len(candidate_stream))
        stream_match = all(
            np.allclose(
                baseline_stream[index]["task_goal"],
                candidate_stream[index]["task_goal"],
            )
            for index in range(common)
        )
        results[str(seed)] = {
            "baseline": baseline_summary,
            "candidate": candidate_summary,
            "common_task_prefix": common,
            "common_task_prefix_matches": stream_match,
        }
    write_json(output / "phase2_100k/summary.json", results)
    return results


def run(args: argparse.Namespace) -> dict[str, object]:
    assert_tracked_worktree_clean()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    for directory in ("models", "fresh_v5", "phase2_100k"):
        (output / directory).mkdir()
    write_json(
        output / "RUNNING.json",
        {"status": "RUNNING", "pid": os.getpid(), "started_at": utc_now()},
    )
    try:
        report_stage(output, "development_freeze_started")
        if args.goal_method not in GOAL_MODEL_PATHS:
            raise ValueError(f"unsupported Goal method {args.goal_method}")
        point_model = EnergyToGoRegressor.load(POINT_CHECKPOINT, device=args.device)
        risk_model = PositiveResidualQuantileModel.load(
            GOAL_MODEL_PATHS[args.goal_method],
            device=args.device,
        )
        mission_model = HeteroscedasticResidualModel.load(
            MISSION_MODEL_CHECKPOINT,
            device=args.device,
        )
        policy = load_frozen_sac(SAC_CHECKPOINT, args.navigation_device)
        goal_calibration_set, mission_calibration_set, _ = load_development_sets()
        goal_calibration, _, _ = fit_goal_calibration(
            point_model,
            risk_model,
            goal_calibration_set,
            coverage=args.goal_conformal_coverage,
        )
        mission_calibration, _, _ = fit_mission_calibration(
            point_model,
            mission_model,
            mission_calibration_set,
            coverage=args.mission_conformal_coverage,
        )
        estimator = MondrianGoalMissionRiskEstimator(
            point_model,
            GoalRiskFeatureBuilder("compact_decision_context"),
            risk_model,
            goal_calibration,
            mission_model,
            mission_calibration,
            goal_coverage=args.goal_conformal_coverage,
            mission_coverage=args.mission_conformal_coverage,
        )
        estimator.save(output / "models/preregistered_v5_estimator.pt")
        preregistration = {
            "frozen_at": utc_now(),
            "git_sha": git_sha(),
            "implementation_sha256": {
                str(path.relative_to(ROOT)): file_sha256(path)
                for path in (
                    ROOT / "scripts/run_energy_risk_v5.py",
                    ROOT / "experiments/energy_mc/final_risk.py",
                    ROOT / "experiments/energy_mc/conditional_risk.py",
                    ROOT / "experiments/energy_mc/statistical_gate.py",
                    ROOT / "experiments/energy_mc/conformal.py",
                    ROOT / "envs/UAVEnergyDeliverySAC.py",
                )
            },
            "safety_coverage_target": SAFETY_TARGET,
            "raw_empirical_coverage_is_sole_gate": False,
            "gate": {
                "name": "Gate B plus Gate C stress test",
                "construction": (
                    "direct predefined Mondrian split-conformal calibration with complete-trajectory units"
                ),
                "stress_test": (
                    "one-sided binomial undercoverage tests at p=0.95 with Holm FWER 0.05"
                ),
                "pass": (
                    "all preregistered direct calibration groups present and no Holm-rejected "
                    "primary group for Goal or Mission"
                ),
            },
            "point_model": {
                "path": str(POINT_CHECKPOINT.resolve()),
                "sha256": file_sha256(POINT_CHECKPOINT),
                "input": "original_7d_goal_conditioned_state",
                "retrained": False,
            },
            "sac": {
                "path": str(SAC_CHECKPOINT.resolve()),
                "sha256": file_sha256(SAC_CHECKPOINT),
                "frozen": True,
                "retrained": False,
            },
            "goal": {
                "method": args.goal_method,
                "model_path": str(GOAL_MODEL_PATHS[args.goal_method].resolve()),
                "model_sha256": file_sha256(GOAL_MODEL_PATHS[args.goal_method]),
                "features": GoalRiskFeatureBuilder("compact_decision_context").as_dict(),
                "target": "positive future suffix maximum point underestimation",
                "trajectory_score": "max_t(true_remaining_energy-point-risk)",
                "conformal_construction_coverage": args.goal_conformal_coverage,
                "calibration": goal_calibration.as_dict(),
                "primary_interaction_groups": list(EXPECTED_GOAL_INTERACTIONS),
            },
            "mission": {
                "method": "frozen_heteroscedastic_laplace_plus_distance_mondrian",
                "model_path": str(MISSION_MODEL_CHECKPOINT.resolve()),
                "model_sha256": file_sha256(MISSION_MODEL_CHECKPOINT),
                "conformal_construction_coverage": args.mission_conformal_coverage,
                "calibration": mission_calibration.as_dict(),
                "primary_distance_groups": list(DISTANCE_BUCKETS),
            },
            "fresh_v5": {
                "goal_seed": args.fresh_goal_seed,
                "goal_trajectories": args.fresh_goal_trajectories,
                "mission_seed": args.fresh_mission_seed,
                "mission_trajectories": args.fresh_mission_trajectories,
                "used_for_training_validation_calibration_or_selection": False,
                "one_shot": True,
            },
            "phase2": {
                "run_only_if_fresh_v5_gate_passes": True,
                "seeds": list(args.phase2_seeds),
                "transitions_per_method_per_seed": args.phase2_transitions,
                "reserve_fraction": 0.10,
                "battery_capacity": BATTERY_CAPACITY,
            },
            "formal_500k_phase2": "NOT_STARTED",
        }
        write_json(output / "PREREGISTERED_ENERGY_RISK_V5.json", preregistration)
        report_stage(output, "configuration_preregistered")

        development_goal_ids = {
            "point_train": PackedEnergyDataset.load(
                SOURCE / "energy_dataset/train"
            ).successful_trajectory_ids,
            "point_validation": PackedEnergyDataset.load(
                SOURCE / "energy_dataset/validation"
            ).successful_trajectory_ids,
            "risk_calibration": goal_calibration_set.successful_trajectory_ids,
            "v4_diagnostic": PackedEnergyDataset.load(
                V2 / "final_conformal_test_v2/trajectories"
            ).successful_trajectory_ids,
        }
        report_stage(output, "fresh_v5_collection_started")
        fresh_args = environment_args(
            seed=args.model_seed,
            phase2_budget=args.phase2_transitions,
        )
        fresh_goal = collect_intersection_split(
            policy,
            fresh_args,
            count=args.fresh_goal_trajectories,
            seed=args.fresh_goal_seed,
            trajectory_id_offset=12_000_000,
            output=output / "fresh_v5/goal_trajectories",
            label="fresh_v5_goal",
        )
        fresh_mission = collect_mission_split(
            policy,
            fresh_args,
            count=args.fresh_mission_trajectories,
            seed=args.fresh_mission_seed,
            trajectory_id_offset=13_000_000,
            output=output / "fresh_v5/mission_trajectories",
            label="fresh_v5_mission",
        )
        assert_fresh_test_isolation(
            fresh_goal.successful_trajectory_ids,
            development_goal_ids,
        )
        prior_mission_ids = (
            mission_calibration_set.successful_mission_ids
            | PackedMissionDataset.load(
                V2 / "mission_test_v2/trajectories"
            ).successful_mission_ids
        )
        if fresh_mission.successful_mission_ids & prior_mission_ids:
            raise RuntimeError("fresh v5 Mission IDs overlap development data")
        write_json(
            output / "FRESH_V5_CONSUMED.json",
            {
                "consumed_at": utc_now(),
                "goal_count": len(fresh_goal.successful_trajectory_ids),
                "mission_count": len(fresh_mission.successful_mission_ids),
                "post_read_tuning_forbidden": True,
            },
        )
        report_stage(output, "fresh_v5_collection_completed")

        fresh_goal_point = chunked_point_predictions(point_model, fresh_goal.states)
        fresh_goal_risk = risk_model.predict_level(
            GoalRiskFeatureBuilder("compact_decision_context").build_from_dataset(fresh_goal),
            args.goal_conformal_coverage,
        )
        fresh_goal_upper = goal_calibration.apply(
            fresh_goal_point,
            fresh_goal_risk,
            fresh_goal.goal_types,
            fresh_goal.states[:, -1].astype(np.float64) * D_MAX,
        )
        rng = np.random.default_rng(args.fresh_goal_seed + 12345)
        goal_remaining = rng.uniform(
            0.0,
            BATTERY_CAPACITY,
            size=fresh_goal.targets.shape,
        )
        fresh_goal_metrics = goal_metrics(
            fresh_goal,
            fresh_goal_point,
            fresh_goal_risk,
            fresh_goal_upper,
            remaining_energy=goal_remaining,
        )
        goal_statistics, _ = goal_coverage_report(fresh_goal, fresh_goal_upper)

        point_fn = point_callable(point_model)
        _, _, fresh_mission_point = mission_component_predictions(point_fn, fresh_mission)
        fresh_mission_features = np.concatenate(
            [fresh_mission.task_states, fresh_mission.return_after_states],
            axis=1,
        )
        fresh_mission_risk = mission_model.upper_offset(
            fresh_mission_features,
            args.mission_conformal_coverage,
        )
        fresh_mission_upper = mission_calibration.apply(
            fresh_mission_point + fresh_mission_risk,
            fresh_mission.initial_distance_buckets,
        )
        mission_remaining = rng.uniform(
            0.0,
            BATTERY_CAPACITY,
            size=fresh_mission.true_mission_energy.shape,
        )
        fresh_mission_metrics = mission_metrics(
            fresh_mission,
            fresh_mission_point,
            fresh_mission_risk,
            fresh_mission_upper,
            remaining_energy=mission_remaining,
        )
        mission_statistics, _ = mission_coverage_report(
            fresh_mission,
            fresh_mission_upper,
        )
        write_json(output / "fresh_v5/goal_results.json", fresh_goal_metrics)
        write_json(output / "fresh_v5/goal_statistical_evidence.json", goal_statistics)
        write_json(output / "fresh_v5/mission_results.json", fresh_mission_metrics)
        write_json(output / "fresh_v5/mission_statistical_evidence.json", mission_statistics)

        construction_complete = (
            set(EXPECTED_GOAL_INTERACTIONS) <= goal_calibration.group_values.keys()
            and set(DISTANCE_BUCKETS) <= mission_calibration.distance_corrections.keys()
        )
        readiness = {
            "safety_target": SAFETY_TARGET,
            "goal_conformal_construction_coverage": args.goal_conformal_coverage,
            "mission_conformal_construction_coverage": args.mission_conformal_coverage,
            "direct_group_calibration_complete": construction_complete,
            "goal_holm_stress_test_passed": goal_statistics[
                "holm_undercoverage_test"
            ]["passed"],
            "mission_holm_stress_test_passed": mission_statistics[
                "holm_undercoverage_test"
            ]["passed"],
        }
        readiness["offline_gate_passed"] = bool(
            readiness["direct_group_calibration_complete"]
            and readiness["goal_holm_stress_test_passed"]
            and readiness["mission_holm_stress_test_passed"]
        )
        readiness["fresh_v5_status"] = (
            "PASS" if readiness["offline_gate_passed"] else "FAILED"
        )
        readiness["no_post_v5_tuning_allowed"] = True
        write_json(output / "FRESH_V5_READINESS.json", readiness)
        report_stage(
            output,
            "fresh_v5_evaluation_completed",
            offline_gate_passed=readiness["offline_gate_passed"],
        )

        phase2_results = None
        if readiness["offline_gate_passed"] and args.run_phase2:
            report_stage(output, "paired_phase2_100k_started")
            phase2_results = run_paired_phase2(
                output,
                policy,
                estimator,
                args,
            )
            report_stage(output, "paired_phase2_100k_completed")
        else:
            write_json(
                output / "phase2_100k/SKIPPED.json",
                {
                    "reason": (
                        "fresh_v5_offline_gate_failed"
                        if not readiness["offline_gate_passed"]
                        else "run_phase2_flag_not_set"
                    )
                },
            )

        completed = {
            "status": "COMPLETED",
            "completed_at": utc_now(),
            "fresh_v5_readiness": readiness,
            "goal_results": fresh_goal_metrics,
            "mission_results": fresh_mission_metrics,
            "phase2_100k": phase2_results,
            "formal_500k_phase2": "NOT_STARTED",
            "sac_retrained": False,
            "point_estimator_retrained": False,
            "td_restored": False,
        }
        write_json(output / "COMPLETED.json", completed)
        (output / "RUNNING.json").unlink(missing_ok=True)
        return completed
    except Exception as error:
        write_json(
            output / "FAILED.json",
            {
                "status": "FAILED",
                "failed_at": utc_now(),
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "formal_500k_phase2": "NOT_STARTED",
            },
        )
        (output / "RUNNING.json").unlink(missing_ok=True)
        raise


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Preregistered one-shot Energy Risk v5")
    value.add_argument("--output-dir", type=Path, required=True)
    value.add_argument("--device", default="cuda")
    value.add_argument("--navigation-device", default="cuda")
    value.add_argument("--model-seed", type=int, default=940_001)
    value.add_argument("--fresh-goal-seed", type=int, default=950_001)
    value.add_argument("--fresh-mission-seed", type=int, default=960_001)
    value.add_argument("--fresh-goal-trajectories", type=int, default=5_000)
    value.add_argument("--fresh-mission-trajectories", type=int, default=3_000)
    value.add_argument("--goal-method", choices=sorted(GOAL_MODEL_PATHS), default="k2_suffix")
    value.add_argument("--goal-conformal-coverage", type=float, default=0.975)
    value.add_argument("--mission-conformal-coverage", type=float, default=0.99)
    value.add_argument("--run-phase2", action="store_true")
    value.add_argument("--phase2-transitions", type=int, default=100_000)
    value.add_argument(
        "--phase2-seeds",
        type=int,
        nargs="+",
        default=[970_001, 970_002, 970_003],
    )
    return value


if __name__ == "__main__":
    run(parser().parse_args())
