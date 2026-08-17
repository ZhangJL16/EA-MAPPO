from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch

from envs.UAVEnergyDeliverySAC import SACTrainingPhase
from experiments.energy_mc.adaptive_analysis import (
    D_MAX,
    correlation_table,
    evaluate_goal_upper,
    evaluate_mission_upper,
    mission_component_predictions,
    residual_group_summary,
    state_residual_by_remaining_distance,
    trajectory_diagnostic_frame,
)
from experiments.energy_mc.adaptive_uncertainty import (
    HeteroscedasticResidualModel,
    RISK_COVERAGE_LEVELS,
    TrajectoryConformalCorrection,
    chunked_point_predictions,
)
from experiments.energy_mc.conditional_risk import (
    GoalRiskFeatureBuilder,
    HierarchicalScaledConformalCalibration,
    PositiveResidualQuantileModel,
    ScaledTrajectoryConformalCorrection,
    SeparatedGoalMissionRiskEstimator,
    positive_underestimation_target,
    reconstruct_positions,
    trajectory_balanced_state_indices,
)
from experiments.energy_mc.conformal import PackedMissionDataset
from experiments.energy_mc.core import PackedEnergyDataset
from review_bundle.safety.energy.mc_regression import (
    EnergyToGoRegressor,
    HierarchicalConformalEnergyEstimator,
    ModelBasedEnergyRolloutEstimator,
)
from scripts.explore_adaptive_energy_uncertainty import environment_args, point_callable
from scripts.run_uav_energy_conformal_v2 import collect_intersection_split, collect_mission_split
from scripts.train_uav_energy_delivery_sac import environment_from_args
from scripts.train_uav_energy_mc import file_sha256, git_sha, load_frozen_sac, run_phase2


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts/uav_energy_delivery_mc_formal_20260817_154321"
V2 = ROOT / "artifacts/uav_energy_delivery_mc_conformal_v2_20260817_184046"
V3 = ROOT / "artifacts/uav_energy_adaptive_uncertainty_20260817_214303"
POINT_CHECKPOINT = SOURCE / "energy_model/best_validation.pt"
SAC_CHECKPOINT = ROOT / (
    "artifacts/uav_energy_delivery_v3_formal_20260816_004619/"
    "phase1_navigation/checkpoint_transition_500000.zip"
)
GROUP_BASELINE_CHECKPOINT = V2 / "model/group_conformal_energy_estimator.pt"
MISSION_MODEL_CHECKPOINT = V3 / "models/mission_heteroscedastic_laplace.pt"
MISSION_CALIBRATION_PATH = V3 / "calibration/mission_calibrations.json"
BATTERY_CAPACITY = 378.72626091628933
FEATURE_MODES = (
    "state7",
    "absolute_position",
    "boundary_distances",
    "compact_position_boundary",
)


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
    record = {"timestamp": utc_now(), "stage": stage, **details}
    with (output / "progress.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(json_value(record), sort_keys=True) + "\n")
    print(f"[energy-v4] {stage}", flush=True)


def mission_correction(coverage: float) -> TrajectoryConformalCorrection:
    payload = json.loads(MISSION_CALIBRATION_PATH.read_text(encoding="utf-8"))
    values = payload["mission_heteroscedastic_laplace"][str(float(coverage))]
    return TrajectoryConformalCorrection(**values)


def grouped_multipliers(
    calibration: HierarchicalScaledConformalCalibration,
    dataset: PackedEnergyDataset,
) -> np.ndarray:
    distances = dataset.states[:, -1].astype(np.float64) * D_MAX
    return np.asarray(
        [
            calibration.multiplier_for(str(goal_type), float(distance))
            for goal_type, distance in zip(dataset.goal_types, distances, strict=True)
        ],
        dtype=np.float64,
    )


def upper_from_scaled(
    point: np.ndarray,
    scale: np.ndarray,
    calibration: ScaledTrajectoryConformalCorrection | HierarchicalScaledConformalCalibration,
    dataset: PackedEnergyDataset,
) -> np.ndarray:
    if isinstance(calibration, HierarchicalScaledConformalCalibration):
        multiplier = grouped_multipliers(calibration, dataset)
    else:
        multiplier = np.full(point.shape, calibration.multiplier, dtype=np.float64)
    return np.asarray(point, dtype=np.float64) + np.asarray(scale, dtype=np.float64) * multiplier


def fit_scaled_calibration(
    point: np.ndarray,
    scale: np.ndarray,
    dataset: PackedEnergyDataset,
    *,
    coverage: float,
    grouped: bool,
) -> ScaledTrajectoryConformalCorrection | HierarchicalScaledConformalCalibration:
    if grouped:
        return HierarchicalScaledConformalCalibration.fit(
            dataset.targets,
            point,
            scale,
            dataset.trajectory_ids,
            dataset.goal_types,
            dataset.distance_buckets,
            coverage=coverage,
        )
    return ScaledTrajectoryConformalCorrection.fit(
        dataset.targets,
        point,
        scale,
        dataset.trajectory_ids,
        coverage=coverage,
    )


def primary_goal_groups(result: dict[str, object]) -> dict[str, dict[str, object]]:
    groups = {
        "overall": result["overall"],
        "goal_type_TASK": result["by_goal_type"]["TASK"],
        "goal_type_CHARGER": result["by_goal_type"]["CHARGER"],
    }
    groups.update(
        {f"distance_{name}": metrics for name, metrics in result["by_initial_distance_bucket"].items()}
    )
    for name, metrics in result["by_goal_type_and_initial_distance"].items():
        goal_type = name.split("|", 1)[0]
        if goal_type in {"TASK", "CHARGER"} and int(metrics["num_trajectories"]) >= 100:
            groups[f"intersection_{name}"] = metrics
    return groups


def goal_readiness(result: dict[str, object], target: float = 0.95) -> dict[str, object]:
    groups = primary_goal_groups(result)
    checks = {
        name: float(metrics["whole_trajectory_simultaneous_coverage"]) >= target
        for name, metrics in groups.items()
    }
    return {
        "target": target,
        "passed": all(checks.values()),
        "checks": checks,
        "group_counts": {name: int(metrics["num_trajectories"]) for name, metrics in groups.items()},
        "worst_primary_group": min(
            groups,
            key=lambda name: float(groups[name]["whole_trajectory_simultaneous_coverage"]),
        ),
        "worst_primary_coverage": min(
            float(metrics["whole_trajectory_simultaneous_coverage"])
            for metrics in groups.values()
        ),
    }


def mission_readiness(result: dict[str, object], target: float = 0.95) -> dict[str, object]:
    groups = {"overall": result["overall"]}
    groups.update(
        {f"distance_{name}": metrics for name, metrics in result["by_initial_task_distance_bucket"].items()}
    )
    checks = {
        name: float(metrics["whole_trajectory_simultaneous_coverage"]) >= target
        for name, metrics in groups.items()
    }
    return {
        "target": target,
        "passed": all(checks.values()),
        "checks": checks,
        "group_counts": {name: int(metrics["num_trajectories"]) for name, metrics in groups.items()},
        "worst_primary_group": min(
            groups,
            key=lambda name: float(groups[name]["whole_trajectory_simultaneous_coverage"]),
        ),
        "worst_primary_coverage": min(
            float(metrics["whole_trajectory_simultaneous_coverage"])
            for metrics in groups.values()
        ),
    }


def model_complexity(model, sample_features: np.ndarray) -> dict[str, object]:
    parameters = int(sum(parameter.numel() for parameter in model.model.parameters()))
    values = np.asarray(sample_features[: min(20_000, len(sample_features))], dtype=np.float32)
    for _ in range(3):
        if isinstance(model, HeteroscedasticResidualModel):
            model.predict_scale(values)
        else:
            model.predict_quantiles(values)
    if torch.cuda.is_available() and model.device.type == "cuda":
        torch.cuda.synchronize(model.device)
    start = time.perf_counter()
    repeats = 10
    for _ in range(repeats):
        if isinstance(model, HeteroscedasticResidualModel):
            model.predict_scale(values)
        else:
            model.predict_quantiles(values)
    if torch.cuda.is_available() and model.device.type == "cuda":
        torch.cuda.synchronize(model.device)
    elapsed = time.perf_counter() - start
    return {
        "parameters": parameters,
        "latency_microseconds_per_state": 1e6 * elapsed / (repeats * len(values)),
        "latency_batch_size": len(values),
    }


def reconstruct_mission_geometry(
    dataset: PackedMissionDataset,
    charger: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    return_distance = dataset.return_after_states[:, -1].astype(np.float64) * D_MAX
    task_goal = charger[None, :] - dataset.return_after_states[:, 3:6] * return_distance[:, None]
    task_distance = dataset.task_states[:, -1].astype(np.float64) * D_MAX
    start_position = task_goal - dataset.task_states[:, 3:6] * task_distance[:, None]
    return np.clip(start_position, 0.0, np.asarray([4000.0, 4000.0, 400.0])), np.clip(
        task_goal,
        0.0,
        np.asarray([4000.0, 4000.0, 400.0]),
    )


def selected_goal_upper(
    model,
    builder: GoalRiskFeatureBuilder,
    calibration,
    dataset: PackedEnergyDataset,
    point: np.ndarray,
    *,
    coverage: float,
) -> np.ndarray:
    features = builder.build_from_dataset(dataset)
    if isinstance(model, HeteroscedasticResidualModel):
        return upper_from_scaled(point, model.predict_scale(features), calibration, dataset)
    base = point + model.predict_level(features, coverage)
    return upper_from_scaled(base, np.ones(base.shape), calibration, dataset)


def mission_component_upper(
    point_estimator: EnergyToGoRegressor,
    model,
    builder: GoalRiskFeatureBuilder,
    calibration,
    dataset: PackedMissionDataset,
    charger: np.ndarray,
    *,
    coverage: float,
) -> tuple[np.ndarray, np.ndarray]:
    start, task_goal = reconstruct_mission_geometry(dataset, charger)
    task_point = chunked_point_predictions(point_estimator, dataset.task_states)
    return_point = chunked_point_predictions(point_estimator, dataset.return_after_states)

    def component_upper(states, positions, goal_types, buckets, point):
        packed = PackedEnergyDataset(
            states=states,
            targets=np.zeros(len(states), dtype=np.float32),
            step_energy=np.zeros(len(states), dtype=np.float32),
            transition_dt=np.zeros(len(states), dtype=np.float32),
            trajectory_ids=dataset.mission_ids,
            goal_types=goal_types,
            distance_buckets=buckets,
            boundary_contact_trajectories=np.zeros(len(states), dtype=bool),
            metadata=[],
        )
        features = builder.build(states, positions)
        if isinstance(model, HeteroscedasticResidualModel):
            return upper_from_scaled(point, model.predict_scale(features), calibration, packed)
        base = point + model.predict_level(features, coverage)
        return upper_from_scaled(base, np.ones(base.shape), calibration, packed)

    task_upper = component_upper(
        dataset.task_states,
        start,
        np.full(len(start), "TASK", dtype="U32"),
        dataset.initial_distance_buckets,
        task_point,
    )
    return_buckets = np.asarray(
        [
            "100-500" if distance < 500 else "500-1500" if distance < 1500 else "1500-2500" if distance < 2500 else "2500-4000"
            for distance in dataset.return_after_states[:, -1].astype(np.float64) * D_MAX
        ],
        dtype="U16",
    )
    return_upper = component_upper(
        dataset.return_after_states,
        task_goal,
        np.full(len(start), "TASK_ENDPOINT_TO_CHARGER", dtype="U32"),
        return_buckets,
        return_point,
    )
    return task_point + return_point, task_upper + return_upper


def load_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def assert_disjoint_named_sets(named_sets: dict[str, set[int]]) -> None:
    names = list(named_sets)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = named_sets[left] & named_sets[right]
            if overlap:
                raise RuntimeError(f"data leakage between {left} and {right}: {sorted(overlap)[:5]}")


def phase2_counterfactual_audit(
    policy,
    args: SimpleNamespace,
    output: Path,
    *,
    capacity: float,
) -> dict[str, object]:
    events = load_jsonl(output / "switching_events.jsonl")
    oracle = ModelBasedEnergyRolloutEstimator(policy, max_policy_steps=4000)
    rows = []
    for index, event in enumerate(events):
        position = np.asarray(event["position"], dtype=np.float32)
        velocity = np.asarray(event["velocity"], dtype=np.float32)
        task_goal = np.asarray(event["task_goal"], dtype=np.float32)
        environment = environment_from_args(args, phase=SACTrainingPhase.TD_PRETRAINING)
        environment.reset(
            seed=args.seed + 900_000 + index,
            options={
                "start_position": position,
                "start_velocity": velocity,
                "task_point": task_goal,
            },
        )
        task = oracle.estimate_context(environment, task_goal, goal_type="TASK")
        return_now = oracle.estimate_context(
            environment,
            environment.charger_position,
            goal_type="CHARGER",
        )
        return_after = oracle.estimate_context(
            environment,
            environment.charger_position,
            position=task_goal,
            velocity=np.zeros(3, dtype=np.float32),
            goal_type="TASK_ENDPOINT_TO_CHARGER",
        )
        environment.close()
        true_mission = task.prediction + return_after.prediction
        reserve = capacity * args.energy_reserve_fraction
        unnecessary = float(event["remaining_energy"]) >= true_mission + reserve
        rows.append(
            {
                "switch_index": index,
                "global_step": event["global_step"],
                "reason": event["reason"],
                "true_task_energy": task.prediction,
                "true_return_after_task_energy": return_after.prediction,
                "true_return_now_energy": return_now.prediction,
                "true_mission_energy": true_mission,
                "remaining_energy": event["remaining_energy"],
                "reserve": reserve,
                "unnecessary_return": unnecessary,
            }
        )
    write_json(output / "switch_counterfactual_audit.json", rows)
    return {
        "audited_switches": len(rows),
        "unnecessary_returns": int(sum(bool(row["unnecessary_return"]) for row in rows)),
        "unnecessary_return_rate": None
        if not rows
        else float(np.mean([bool(row["unnecessary_return"]) for row in rows])),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    for directory in ("diagnostics", "models", "selection", "fresh_v4", "phase2_100k"):
        (output / directory).mkdir()
    write_json(output / "RUNNING.json", {"status": "RUNNING", "pid": os.getpid(), "started_at": utc_now()})
    torch.set_num_threads(args.torch_num_threads)
    try:
        report_stage(output, "audit_started")
        point = EnergyToGoRegressor.load(POINT_CHECKPOINT, device=args.device)
        policy = load_frozen_sac(SAC_CHECKPOINT, args.navigation_device)
        train = PackedEnergyDataset.load(SOURCE / "energy_dataset/train")
        validation = PackedEnergyDataset.load(SOURCE / "energy_dataset/validation")
        calibration = PackedEnergyDataset.concatenate(
            [
                PackedEnergyDataset.load(SOURCE / "energy_dataset/calibration"),
                PackedEnergyDataset.load(V2 / "calibration_v2/new_trajectories"),
            ]
        )
        diagnostic = PackedEnergyDataset.load(V2 / "final_conformal_test_v2/trajectories")
        development_goal_ids = {
            "risk_train": train.successful_trajectory_ids,
            "risk_validation": validation.successful_trajectory_ids,
            "risk_calibration": calibration.successful_trajectory_ids,
            "architecture_diagnostic": diagnostic.successful_trajectory_ids,
        }
        assert_disjoint_named_sets(development_goal_ids)
        train_point = chunked_point_predictions(point, train.states)
        validation_point = chunked_point_predictions(point, validation.states)
        calibration_point = chunked_point_predictions(point, calibration.states)
        diagnostic_point = chunked_point_predictions(point, diagnostic.states)
        train_indices = trajectory_balanced_state_indices(
            train.trajectory_ids,
            max_states_per_trajectory=args.risk_train_states_per_trajectory,
        )
        validation_indices = trajectory_balanced_state_indices(
            validation.trajectory_ids,
            max_states_per_trajectory=args.risk_validation_states_per_trajectory,
        )
        train_positions = reconstruct_positions(train)[train_indices]
        validation_positions = reconstruct_positions(validation)[validation_indices]
        calibration_positions = reconstruct_positions(calibration)
        diagnostic_positions = reconstruct_positions(diagnostic)
        point_mae = float(np.mean(np.abs(diagnostic_point - diagnostic.targets)))
        diagnostic_frame = trajectory_diagnostic_frame(diagnostic, diagnostic_point)
        diagnostic_frame.to_csv(output / "diagnostics/trajectory_residuals.csv", index=False)
        residual_group_summary(diagnostic_frame, ["goal_type", "distance_bucket"]).to_csv(
            output / "diagnostics/trajectory_residual_summary.csv",
            index=False,
        )
        state_residual_by_remaining_distance(diagnostic, diagnostic_point).to_csv(
            output / "diagnostics/state_residual_by_remaining_distance.csv",
            index=False,
        )
        correlations = []
        for column in diagnostic_frame.select_dtypes(include=[np.number]).columns:
            if column in {"trajectory_id", "trajectory_max_underestimation"}:
                continue
            x = diagnostic_frame[column].to_numpy(dtype=np.float64)
            y = diagnostic_frame["trajectory_max_underestimation"].to_numpy(dtype=np.float64)
            if np.std(x) == 0.0 or np.std(y) == 0.0:
                continue
            from scipy.stats import spearmanr

            correlations.append(
                {
                    "feature": column,
                    "spearman": float(spearmanr(x, y).statistic),
                    "count": int(len(x)),
                }
            )
        pd.DataFrame(correlations).sort_values("spearman", ascending=False).to_csv(
            output / "diagnostics/max_underestimation_spearman.csv",
            index=False,
        )
        report_stage(output, "audit_completed", point_checkpoint_sha=file_sha256(POINT_CHECKPOINT))

        report_stage(output, "goal_ablation_training_started")
        models: dict[str, dict[str, object]] = {}
        rows = []
        calibration_payloads: dict[str, object] = {}

        g1_calibration = fit_scaled_calibration(
            calibration_point,
            np.ones(calibration_point.shape),
            calibration,
            coverage=0.95,
            grouped=True,
        )
        g1_upper = upper_from_scaled(
            diagnostic_point,
            np.ones(diagnostic_point.shape),
            g1_calibration,
            diagnostic,
        )
        g1_result = evaluate_goal_upper(diagnostic, diagnostic_point, g1_upper)
        g1_ready = goal_readiness(g1_result)
        rows.append(
            {
                "method": "g1_point_trajectory_group_conformal",
                "feature_mode": "state7",
                "risk_model": "none",
                "point_mae": point_mae,
                "mean_trajectory_max_underestimation": float(diagnostic_frame.trajectory_max_underestimation.mean()),
                "overall_coverage": g1_result["overall"]["whole_trajectory_simultaneous_coverage"],
                "worst_primary_coverage": g1_ready["worst_primary_coverage"],
                "mean_bound_width": g1_result["overall"]["mean_bound_width"],
                "p95_bound_width": g1_result["overall"]["p95_bound_width"],
                "parameters": 0,
                "latency_microseconds_per_state": 0.0,
                "diagnostic_gate_pass": g1_ready["passed"],
            }
        )
        write_json(output / "selection/g1_diagnostic_result.json", g1_result)

        for mode_index, mode in enumerate(FEATURE_MODES):
            builder = GoalRiskFeatureBuilder(mode)
            train_features = builder.build(train.states[train_indices], train_positions)
            validation_features = builder.build(
                validation.states[validation_indices],
                validation_positions,
            )
            g2 = HeteroscedasticResidualModel(
                input_dim=builder.input_dim,
                distribution="laplace",
                seed=args.model_seed + mode_index * 10,
                device=args.device,
            )
            g2_history = g2.fit(
                train_features,
                train.targets[train_indices] - train_point[train_indices],
                validation_features,
                validation.targets[validation_indices] - validation_point[validation_indices],
                max_epochs=args.max_epochs,
            )
            g3 = PositiveResidualQuantileModel(
                input_dim=builder.input_dim,
                seed=args.model_seed + mode_index * 10 + 1,
                device=args.device,
            )
            g3_history = g3.fit(
                train_features,
                positive_underestimation_target(
                    train.targets[train_indices],
                    train_point[train_indices],
                ),
                validation_features,
                positive_underestimation_target(
                    validation.targets[validation_indices],
                    validation_point[validation_indices],
                ),
                max_epochs=args.max_epochs,
            )
            del train_features, validation_features
            calibration_features = builder.build(calibration.states, calibration_positions)
            diagnostic_features = builder.build(diagnostic.states, diagnostic_positions)
            g2_scale_calibration = g2.predict_scale(calibration_features)
            g2_scale_diagnostic = g2.predict_scale(diagnostic_features)
            g3_base_calibration = calibration_point + g3.predict_level(calibration_features, 0.95)
            g3_base_diagnostic = diagnostic_point + g3.predict_level(diagnostic_features, 0.95)
            complexity_g2 = model_complexity(g2, diagnostic_features)
            complexity_g3 = model_complexity(g3, diagnostic_features)
            models[mode] = {"builder": builder, "g2": g2, "g3": g3}
            g2.save(output / f"models/g2_laplace_{mode}.pt")
            g3.save(output / f"models/g3_positive_residual_{mode}.pt")
            write_json(
                output / f"models/training_history_{mode}.json",
                {"g2": g2_history.as_dict(), "g3": g3_history.as_dict()},
            )
            for risk_name, base_calibration, base_diagnostic, scale_calibration, scale_diagnostic, complexity in (
                (
                    "g2",
                    calibration_point,
                    diagnostic_point,
                    g2_scale_calibration,
                    g2_scale_diagnostic,
                    complexity_g2,
                ),
                (
                    "g3",
                    g3_base_calibration,
                    g3_base_diagnostic,
                    np.ones(g3_base_calibration.shape),
                    np.ones(g3_base_diagnostic.shape),
                    complexity_g3,
                ),
            ):
                for grouped in (False, True):
                    calibration_fit = fit_scaled_calibration(
                        base_calibration,
                        scale_calibration,
                        calibration,
                        coverage=0.95,
                        grouped=grouped,
                    )
                    upper = upper_from_scaled(
                        base_diagnostic,
                        scale_diagnostic,
                        calibration_fit,
                        diagnostic,
                    )
                    result = evaluate_goal_upper(diagnostic, diagnostic_point, upper)
                    readiness = goal_readiness(result)
                    method = f"{risk_name}_{mode}_{'group' if grouped else 'global'}"
                    write_json(output / f"selection/{method}_diagnostic_result.json", result)
                    calibration_payloads[method] = calibration_fit.as_dict()
                    rows.append(
                        {
                            "method": method,
                            "feature_mode": mode,
                            "risk_model": risk_name,
                            "point_mae": point_mae,
                            "mean_trajectory_max_underestimation": float(
                                diagnostic_frame.trajectory_max_underestimation.mean()
                            ),
                            "overall_coverage": result["overall"]["whole_trajectory_simultaneous_coverage"],
                            "worst_primary_coverage": readiness["worst_primary_coverage"],
                            "mean_bound_width": result["overall"]["mean_bound_width"],
                            "p95_bound_width": result["overall"]["p95_bound_width"],
                            **complexity,
                            "diagnostic_gate_pass": readiness["passed"],
                        }
                    )
            del calibration_features, diagnostic_features

        comparison = pd.DataFrame(rows)
        comparison.to_csv(output / "selection/goal_candidate_matrix.csv", index=False)
        adaptive = comparison[comparison.risk_model.isin(["g2", "g3"])].copy()
        adaptive["coverage_shortfall"] = (
            np.maximum(0.95 - adaptive.overall_coverage, 0.0)
            + np.maximum(0.95 - adaptive.worst_primary_coverage, 0.0)
        )
        passing = adaptive[adaptive.diagnostic_gate_pass]
        ranked = (passing if not passing.empty else adaptive).sort_values(
            ["coverage_shortfall", "mean_bound_width", "p95_bound_width", "parameters"],
            ascending=True,
        )
        selected_method = str(ranked.iloc[0].method)
        selected_risk_name = selected_method.split("_", 1)[0]
        # Modes contain underscores; recover them without relying on split width.
        selected_calibration_kind = "group" if selected_method.endswith("_group") else "global"
        selected_mode = selected_method[len(selected_risk_name) + 1 : -len(selected_calibration_kind) - 1]
        selected_builder = models[selected_mode]["builder"]
        selected_model = models[selected_mode][selected_risk_name]
        report_stage(output, "goal_ablation_training_completed", selected_method=selected_method)

        selected_calibrations = {}
        selected_calibration_json = {}
        selected_calibration_features = selected_builder.build(
            calibration.states,
            calibration_positions,
        )
        for coverage in RISK_COVERAGE_LEVELS:
            if selected_risk_name == "g2":
                base = calibration_point
                scale = selected_model.predict_scale(selected_calibration_features)
            else:
                base = calibration_point + selected_model.predict_level(
                    selected_calibration_features,
                    coverage,
                )
                scale = np.ones(base.shape)
            fitted = fit_scaled_calibration(
                base,
                scale,
                calibration,
                coverage=coverage,
                grouped=selected_calibration_kind == "group",
            )
            selected_calibrations[coverage] = fitted
            selected_calibration_json[str(coverage)] = fitted.as_dict()
        write_json(output / "models/selected_goal_calibrations.json", selected_calibration_json)
        selected_model.save(output / "models/selected_goal_risk_model.pt")

        mission_model = HeteroscedasticResidualModel.load(MISSION_MODEL_CHECKPOINT, device=args.device)
        preregistration = {
            "frozen_at": utc_now(),
            "point_model": {
                "path": str(POINT_CHECKPOINT.resolve()),
                "sha256": file_sha256(POINT_CHECKPOINT),
                "retrained": False,
            },
            "sac": {
                "path": str(SAC_CHECKPOINT.resolve()),
                "sha256": file_sha256(SAC_CHECKPOINT),
                "retrained": False,
            },
            "goal_risk": {
                "selected_method": selected_method,
                "feature_builder": selected_builder.as_dict(),
                "model_path": str((output / "models/selected_goal_risk_model.pt").resolve()),
                "model_sha256": file_sha256(output / "models/selected_goal_risk_model.pt"),
                "architecture": type(selected_model).__name__,
                "hidden_dim": selected_model.hidden_dim,
                "learning_rate": selected_model.learning_rate,
                "batch_size": selected_model.batch_size,
                "max_epochs": args.max_epochs,
                "trajectory_balanced_training": {
                    "train_states_per_trajectory": args.risk_train_states_per_trajectory,
                    "validation_states_per_trajectory": args.risk_validation_states_per_trajectory,
                    "train_state_count": int(len(train_indices)),
                    "validation_state_count": int(len(validation_indices)),
                },
                "calibration": "one-sided complete-trajectory conformal",
                "calibration_scope": selected_calibration_kind,
            },
            "mission_risk": {
                "selected_method": "mission_heteroscedastic_laplace",
                "model_path": str(MISSION_MODEL_CHECKPOINT.resolve()),
                "model_sha256": file_sha256(MISSION_MODEL_CHECKPOINT),
                "architecture": "HeteroscedasticResidualModel(input_dim=14, hidden_dim=128, laplace)",
                "calibration_path": str(MISSION_CALIBRATION_PATH.resolve()),
                "calibration_sha256": file_sha256(MISSION_CALIBRATION_PATH),
                "retrained_or_tuned_in_v4_stage": False,
            },
            "alpha": 0.05,
            "coverage_target": 0.95,
            "risk_levels_reported": list(RISK_COVERAGE_LEVELS),
            "distance_bins": ["100-500", "500-1500", "1500-2500", "2500-4000", ">4000"],
            "reserve_fraction": 0.10,
            "battery_capacity": BATTERY_CAPACITY,
            "readiness_thresholds": {
                "goal_overall": 0.95,
                "goal_TASK": 0.95,
                "goal_CHARGER": 0.95,
                "goal_primary_distance_and_intersection_groups": 0.95,
                "mission_overall": 0.95,
                "mission_distance_groups": 0.95,
                "minimum_intersection_sample_count": 100,
            },
            "fresh_v4": {
                "goal_seed": args.fresh_goal_seed,
                "goal_trajectories": args.fresh_goal_trajectories,
                "mission_seed": args.fresh_mission_seed,
                "mission_trajectories": args.fresh_mission_trajectories,
            },
            "phase2_100k": {
                "seed": args.phase2_seed,
                "transitions_per_method": args.phase2_transitions,
                "run_only_if_v4_offline_gate_passes": True,
            },
            "selection_dataset": str((V2 / "final_conformal_test_v2/trajectories").resolve()),
            "fresh_v4_not_used_for_selection": True,
        }
        write_json(output / "PREREGISTERED_CONFIG.json", preregistration)
        write_json(
            output / "DATA_LEAKAGE_AUDIT.json",
            {
                "development_goal_id_ranges_disjoint": True,
                "development_goal_counts": {
                    name: len(values) for name, values in development_goal_ids.items()
                },
                "fresh_v4_collected": False,
                "all_seeds_distinct": len(
                    {
                        args.model_seed,
                        args.fresh_goal_seed,
                        args.fresh_mission_seed,
                        args.phase2_seed,
                    }
                )
                == 4,
            },
        )
        report_stage(output, "configuration_preregistered")

        fresh_args = environment_args(seed=args.model_seed, phase2_budget=args.phase2_transitions)
        report_stage(output, "fresh_v4_collection_started")
        fresh_goal = collect_intersection_split(
            policy,
            fresh_args,
            count=args.fresh_goal_trajectories,
            seed=args.fresh_goal_seed,
            trajectory_id_offset=8_000_000,
            output=output / "fresh_v4/goal_trajectories",
            label="fresh_v4_goal",
        )
        fresh_mission = collect_mission_split(
            policy,
            fresh_args,
            count=args.fresh_mission_trajectories,
            seed=args.fresh_mission_seed,
            trajectory_id_offset=9_000_000,
            output=output / "fresh_v4/mission_trajectories",
            label="fresh_v4_mission",
        )
        assert_disjoint_named_sets(
            {**development_goal_ids, "fresh_v4_goal": fresh_goal.successful_trajectory_ids}
        )
        write_json(
            output / "DATA_LEAKAGE_AUDIT.json",
            {
                "development_goal_id_ranges_disjoint": True,
                "fresh_v4_goal_disjoint_from_all_development_goal_splits": True,
                "fresh_v4_mission_ids_disjoint_from_prior_mission_ranges": not bool(
                    fresh_mission.successful_mission_ids
                    & set(range(4_000_000, 8_000_000))
                ),
                "development_goal_counts": {
                    name: len(values) for name, values in development_goal_ids.items()
                },
                "fresh_v4_goal_count": len(fresh_goal.successful_trajectory_ids),
                "fresh_v4_mission_count": len(fresh_mission.successful_mission_ids),
                "all_seeds_distinct": True,
                "fresh_v4_used_for_selection": False,
            },
        )
        report_stage(output, "fresh_v4_collection_completed")

        fresh_goal_point = chunked_point_predictions(point, fresh_goal.states)
        fresh_goal_features = selected_builder.build(
            fresh_goal.states,
            reconstruct_positions(fresh_goal),
        )
        goal_results = {}
        baseline_goal_results = {}
        g1_calibrations = {}
        for coverage in RISK_COVERAGE_LEVELS:
            calibration_g1 = fit_scaled_calibration(
                calibration_point,
                np.ones(calibration_point.shape),
                calibration,
                coverage=coverage,
                grouped=True,
            )
            g1_calibrations[coverage] = calibration_g1
            baseline_upper = upper_from_scaled(
                fresh_goal_point,
                np.ones(fresh_goal_point.shape),
                calibration_g1,
                fresh_goal,
            )
            baseline_goal_results[str(coverage)] = evaluate_goal_upper(
                fresh_goal,
                fresh_goal_point,
                baseline_upper,
            )
            if selected_risk_name == "g2":
                base = fresh_goal_point
                scale = selected_model.predict_scale(fresh_goal_features)
            else:
                base = fresh_goal_point + selected_model.predict_level(fresh_goal_features, coverage)
                scale = np.ones(base.shape)
            upper = upper_from_scaled(base, scale, selected_calibrations[coverage], fresh_goal)
            goal_results[str(coverage)] = evaluate_goal_upper(
                fresh_goal,
                fresh_goal_point,
                upper,
            )

        mission_point_fn = point_callable(point)
        _, _, fresh_mission_point = mission_component_predictions(mission_point_fn, fresh_mission)
        mission_features = np.concatenate(
            [fresh_mission.task_states, fresh_mission.return_after_states],
            axis=1,
        )
        rng = np.random.default_rng(args.fresh_mission_seed + 12345)
        remaining_samples = rng.uniform(0.0, BATTERY_CAPACITY, size=fresh_mission_point.shape)
        probe = environment_from_args(fresh_args, phase=SACTrainingPhase.TD_PRETRAINING)
        charger = probe.charger_position.copy().astype(np.float64)
        probe.close()
        mission_results = {}
        component_results = {}
        for coverage in RISK_COVERAGE_LEVELS:
            correction = mission_correction(coverage)
            mission_upper = (
                fresh_mission_point
                + mission_model.upper_offset(mission_features, coverage)
                + correction.correction
            )
            mission_results[str(coverage)] = evaluate_mission_upper(
                fresh_mission,
                fresh_mission_point,
                mission_upper,
                battery_capacity=BATTERY_CAPACITY,
                reserve_fraction=0.10,
                remaining_energy_samples=remaining_samples,
            )
            component_point, component_upper = mission_component_upper(
                point,
                selected_model,
                selected_builder,
                selected_calibrations[coverage],
                fresh_mission,
                charger,
                coverage=coverage,
            )
            component_results[str(coverage)] = evaluate_mission_upper(
                fresh_mission,
                component_point,
                component_upper,
                battery_capacity=BATTERY_CAPACITY,
                reserve_fraction=0.10,
                remaining_energy_samples=remaining_samples,
            )
        write_json(output / "fresh_v4/goal_selected_results.json", goal_results)
        write_json(output / "fresh_v4/goal_group_baseline_results.json", baseline_goal_results)
        write_json(output / "fresh_v4/mission_adaptive_results.json", mission_results)
        write_json(output / "fresh_v4/mission_component_sum_results.json", component_results)

        fresh_frame = trajectory_diagnostic_frame(fresh_goal, fresh_goal_point)
        fresh_frame.to_csv(output / "fresh_v4/trajectory_residual_diagnostics.csv", index=False)
        residual_group_summary(fresh_frame, ["goal_type", "distance_bucket"]).to_csv(
            output / "fresh_v4/trajectory_residual_group_summary.csv",
            index=False,
        )
        state_residual_by_remaining_distance(fresh_goal, fresh_goal_point).to_csv(
            output / "fresh_v4/state_residual_by_remaining_distance.csv",
            index=False,
        )
        correlation_table(
            fresh_frame,
            targets=["trajectory_max_underestimation"],
            excluded=["trajectory_id"],
        ).to_csv(output / "fresh_v4/max_underestimation_correlations.csv", index=False)
        goal_gate = goal_readiness(goal_results["0.95"])
        mission_gate = mission_readiness(mission_results["0.95"])
        readiness = {
            "goal": goal_gate,
            "mission": mission_gate,
            "offline_gate_passed": bool(goal_gate["passed"] and mission_gate["passed"]),
            "fresh_v4_consumed": True,
            "no_post_v4_tuning_allowed": True,
        }
        write_json(output / "FRESH_V4_READINESS.json", readiness)
        report_stage(output, "fresh_v4_evaluation_completed", offline_gate_passed=readiness["offline_gate_passed"])

        phase2_summary = None
        if readiness["offline_gate_passed"] and args.run_phase2:
            adaptive_estimator = SeparatedGoalMissionRiskEstimator(
                point,
                selected_builder,
                selected_model,
                selected_calibrations[0.95],
                mission_model,
                mission_correction(0.95),
                coverage=0.95,
            )
            adaptive_estimator.save(output / "models/preregistered_deployment_estimator.pt")
            phase_args = environment_args(seed=args.phase2_seed, phase2_budget=args.phase2_transitions)
            baseline_output = output / "phase2_100k/group_conformal_baseline"
            adaptive_output = output / "phase2_100k/adaptive_goal_mission"
            baseline_output.mkdir()
            adaptive_output.mkdir()
            baseline = HierarchicalConformalEnergyEstimator.load(
                GROUP_BASELINE_CHECKPOINT,
                device=args.device,
            )
            report_stage(output, "phase2_100k_baseline_started")
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
            write_json(baseline_output / "summary.json", baseline_summary)
            report_stage(output, "phase2_100k_adaptive_started")
            adaptive_summary = run_phase2(
                policy,
                adaptive_estimator,
                phase_args,
                capacity=BATTERY_CAPACITY,
                output=adaptive_output,
            )
            adaptive_summary.update(
                phase2_counterfactual_audit(
                    policy,
                    phase_args,
                    adaptive_output,
                    capacity=BATTERY_CAPACITY,
                )
            )
            write_json(adaptive_output / "summary.json", adaptive_summary)
            baseline_stream = load_jsonl(baseline_output / "completed_task_stream.jsonl")
            adaptive_stream = load_jsonl(adaptive_output / "completed_task_stream.jsonl")
            common = min(len(baseline_stream), len(adaptive_stream))
            stream_match = all(
                np.allclose(baseline_stream[index]["task_goal"], adaptive_stream[index]["task_goal"])
                for index in range(common)
            )
            phase2_summary = {
                "same_seed": args.phase2_seed,
                "same_battery_capacity": BATTERY_CAPACITY,
                "same_reserve_fraction": 0.10,
                "transitions_per_method": args.phase2_transitions,
                "paired_task_stream_common_prefix": common,
                "paired_task_stream_common_prefix_matches": stream_match,
                "group_conformal_baseline": baseline_summary,
                "adaptive_goal_mission": adaptive_summary,
            }
            write_json(output / "phase2_100k/summary.json", phase2_summary)
            report_stage(output, "phase2_100k_completed")
        else:
            write_json(
                output / "phase2_100k/SKIPPED.json",
                {
                    "reason": "fresh_v4_offline_gate_failed"
                    if not readiness["offline_gate_passed"]
                    else "run_phase2_flag_not_set",
                },
            )

        completed = {
            "status": "COMPLETED",
            "completed_at": utc_now(),
            "selected_goal_method": selected_method,
            "selected_mission_method": "mission_heteroscedastic_laplace",
            "fresh_v4_readiness": readiness,
            "phase2_100k": phase2_summary,
            "formal_500k_phase2_launched": False,
            "point_estimator_retrained": False,
            "sac_retrained": False,
        }
        write_json(output / "COMPLETED.json", completed)
        (output / "RUNNING.json").unlink(missing_ok=True)
        return completed
    except KeyboardInterrupt:
        write_json(
            output / "INTERRUPTED.json",
            {"status": "INTERRUPTED", "interrupted_at": utc_now()},
        )
        (output / "RUNNING.json").unlink(missing_ok=True)
        raise
    except Exception as error:
        write_json(
            output / "FAILED.json",
            {
                "status": "FAILED",
                "failed_at": utc_now(),
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        (output / "RUNNING.json").unlink(missing_ok=True)
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preregistered adaptive energy uncertainty v4 validation")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--navigation-device", default="cpu")
    parser.add_argument("--torch-num-threads", type=int, default=1)
    parser.add_argument("--model-seed", type=int, default=710_001)
    parser.add_argument("--fresh-goal-seed", type=int, default=720_001)
    parser.add_argument("--fresh-mission-seed", type=int, default=730_001)
    parser.add_argument("--phase2-seed", type=int, default=740_001)
    parser.add_argument("--fresh-goal-trajectories", type=int, default=3000)
    parser.add_argument("--fresh-mission-trajectories", type=int, default=2000)
    parser.add_argument("--phase2-transitions", type=int, default=100_000)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--risk-train-states-per-trajectory", type=int, default=128)
    parser.add_argument("--risk-validation-states-per-trajectory", type=int, default=128)
    parser.add_argument("--run-phase2", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    if len({args.model_seed, args.fresh_goal_seed, args.fresh_mission_seed, args.phase2_seed}) != 4:
        parser.error("model, fresh v4, and Phase2 seeds must be distinct")
    if args.smoke:
        args.fresh_goal_trajectories = 65
        args.fresh_mission_trajectories = 25
        args.phase2_transitions = min(args.phase2_transitions, 1000)
        args.max_epochs = min(args.max_epochs, 2)
    if args.fresh_goal_trajectories < 13:
        parser.error("fresh goal trajectories must cover all 13 feasible intersections")
    if args.fresh_mission_trajectories % 5 != 0:
        parser.error("fresh mission trajectories must be divisible by five")
    if args.risk_train_states_per_trajectory <= 0 or args.risk_validation_states_per_trajectory <= 0:
        parser.error("trajectory-balanced risk-model state counts must be positive")
    return args


def main() -> None:
    result = run(parse_args())
    print(json.dumps(json_value(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
