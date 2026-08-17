from __future__ import annotations

import argparse
import json
import os
import traceback
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.energy_mc.adaptive_analysis import (
    D_MAX,
    VELOCITY_SCALE,
    correlation_table,
    evaluate_goal_upper,
    evaluate_mission_upper,
    mission_component_predictions,
    mission_residual_dependence,
    nearest_neighbor_aliasing_analysis,
    residual_group_summary,
    stratified_mission_split,
    trajectory_diagnostic_frame,
)
from experiments.energy_mc.adaptive_uncertainty import (
    AdaptiveConformalEnergyEstimator,
    HeteroscedasticResidualModel,
    MCSupervisedQuantileModel,
    RISK_COVERAGE_LEVELS,
    TrajectoryConformalCorrection,
    chunked_point_predictions,
)
from experiments.energy_mc.conformal import PackedMissionDataset, mission_scores
from experiments.energy_mc.core import PackedEnergyDataset
from review_bundle.safety.energy.mc_regression import (
    EnergyToGoRegressor,
    HierarchicalConformalCalibration,
    HierarchicalConformalEnergyEstimator,
    MissionConformalCalibration,
    ModelBasedEnergyRolloutEstimator,
)
from envs.UAVEnergyDeliverySAC import SACTrainingPhase
from scripts.train_uav_energy_delivery_sac import environment_from_args
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
SOURCE = ROOT / "artifacts/uav_energy_delivery_mc_formal_20260817_154321"
V2 = ROOT / "artifacts/uav_energy_delivery_mc_conformal_v2_20260817_184046"
SAC_CHECKPOINT = ROOT / (
    "artifacts/uav_energy_delivery_v3_formal_20260816_004619/"
    "phase1_navigation/checkpoint_transition_500000.zip"
)
POINT_CHECKPOINT = SOURCE / "energy_model/best_validation.pt"
GROUP_CONFORMAL_CHECKPOINT = V2 / "model/group_conformal_energy_estimator.pt"
BATTERY_CAPACITY = 378.72626091628933


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
    print(f"[adaptive-energy] {stage}", flush=True)


def environment_args(*, seed: int = 0, phase2_budget: int = 20_000) -> SimpleNamespace:
    return SimpleNamespace(
        minimum_task_distance=100.0,
        xy_sampling_margin=100.0,
        task_z_min=20.0,
        task_z_max=380.0,
        phase1_episode_max_steps=4000,
        phase2_episode_max_steps=max(100_000, phase2_budget + 1),
        energy_reserve_fraction=0.10,
        render_vertical_exaggeration=4.0,
        base_power=0.05,
        velocity_coefficients=[0.005, 0.005, 0.005],
        acceleration_coefficients=[0.005, 0.005, 0.005],
        compute_power=0.005,
        communication_power=0.005,
        simulation_error=0.0,
        flight_energy_multiplier=1.0,
        target_nominal_endurance_minutes=30.0,
        phase2_transition_budget=phase2_budget,
        phase2_log_frequency=1000,
        seed=seed,
        energy_test_seed=850_001,
    )


def point_callable(estimator: EnergyToGoRegressor):
    return lambda states: chunked_point_predictions(estimator, states)


def mission_initial_rows(dataset: PackedMissionDataset) -> pd.DataFrame:
    metadata = {int(row["mission_id"]): row for row in dataset.metadata}
    rows = []
    for mission_id in np.unique(dataset.mission_ids):
        index = int(np.flatnonzero(dataset.mission_ids == mission_id)[0])
        row = metadata[int(mission_id)]
        rows.append(
            {
                "mission_id": int(mission_id),
                "initial_distance_bucket": row["initial_task_distance_bucket"],
                "initial_task_distance": float(row["initial_task_distance"]),
                "task_energy": float(row["task_energy"]),
                "return_energy": float(row["return_after_task_energy"]),
                "true_mission_energy": float(dataset.true_mission_energy[index]),
                "task_state_index": index,
            }
        )
    return pd.DataFrame(rows)


def calibrate_goal_methods(
    point: EnergyToGoRegressor,
    heteroscedastic: dict[str, HeteroscedasticResidualModel],
    quantile: MCSupervisedQuantileModel,
    calibration: PackedEnergyDataset,
) -> dict[str, dict[float, object]]:
    point_prediction = chunked_point_predictions(point, calibration.states)
    methods: dict[str, dict[float, object]] = {
        "point_global": {},
        "point_group": {},
        **{f"heteroscedastic_{name}": {} for name in heteroscedastic},
        "supervised_quantile_cqr": {},
    }
    for coverage in RISK_COVERAGE_LEVELS:
        methods["point_global"][coverage] = TrajectoryConformalCorrection.fit(
            calibration.targets,
            point_prediction,
            calibration.trajectory_ids,
            coverage=coverage,
        )
        methods["point_group"][coverage] = HierarchicalConformalCalibration.fit(
            point_prediction,
            calibration.targets,
            calibration.trajectory_ids,
            calibration.goal_types,
            calibration.distance_buckets,
            coverage=coverage,
        )
        for name, model in heteroscedastic.items():
            base = point_prediction + model.upper_offset(calibration.states, coverage)
            methods[f"heteroscedastic_{name}"][coverage] = TrajectoryConformalCorrection.fit(
                calibration.targets,
                base,
                calibration.trajectory_ids,
                coverage=coverage,
                allow_negative=True,
            )
        quantile_base = quantile.predict_level(calibration.states, coverage)
        methods["supervised_quantile_cqr"][coverage] = TrajectoryConformalCorrection.fit(
            calibration.targets,
            quantile_base,
            calibration.trajectory_ids,
            coverage=coverage,
            allow_negative=True,
        )
    return methods


def goal_method_predictions(
    point: EnergyToGoRegressor,
    heteroscedastic: dict[str, HeteroscedasticResidualModel],
    quantile: MCSupervisedQuantileModel,
    calibrations: dict[str, dict[float, object]],
    dataset: PackedEnergyDataset,
    coverage: float,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    point_prediction = chunked_point_predictions(point, dataset.states)
    results: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    global_correction = calibrations["point_global"][coverage]
    results["point_global"] = (
        point_prediction,
        point_prediction + global_correction.correction,
    )
    group = calibrations["point_group"][coverage]
    distances = dataset.states[:, -1].astype(np.float64) * D_MAX
    group_margin = np.asarray(
        [
            group.margin_for(goal_type, distance)
            for goal_type, distance in zip(dataset.goal_types, distances, strict=True)
        ]
    )
    results["point_group"] = (point_prediction, point_prediction + group_margin)
    for name, model in heteroscedastic.items():
        correction = calibrations[f"heteroscedastic_{name}"][coverage]
        upper = (
            point_prediction
            + model.upper_offset(dataset.states, coverage)
            + correction.correction
        )
        results[f"heteroscedastic_{name}"] = (point_prediction, upper)
    quantile_prediction = quantile.predict_quantiles(dataset.states)
    q50 = quantile_prediction[:, quantile.quantile_levels.index(0.50)]
    q_level = quantile_prediction[:, quantile.quantile_levels.index(coverage)]
    correction = calibrations["supervised_quantile_cqr"][coverage]
    results["supervised_quantile_cqr"] = (q50, q_level + correction.correction)
    return results


def calibrate_mission_methods(
    point: EnergyToGoRegressor,
    heteroscedastic: HeteroscedasticResidualModel,
    quantile: MCSupervisedQuantileModel,
    calibration: PackedMissionDataset,
) -> dict[str, dict[float, object]]:
    point_fn = point_callable(point)
    _, _, point_prediction = mission_component_predictions(point_fn, calibration)
    features = np.concatenate([calibration.task_states, calibration.return_after_states], axis=1)
    methods: dict[str, dict[float, object]] = {
        "mission_point_global": {},
        "mission_point_distance_group": {},
        "mission_heteroscedastic_laplace": {},
        "mission_direct_quantile_cqr": {},
    }
    _, mission_scores_values, buckets = mission_scores(point, calibration)
    for coverage in RISK_COVERAGE_LEVELS:
        methods["mission_point_global"][coverage] = TrajectoryConformalCorrection.fit(
            calibration.true_mission_energy,
            point_prediction,
            calibration.mission_ids,
            coverage=coverage,
        )
        methods["mission_point_distance_group"][coverage] = MissionConformalCalibration.fit(
            mission_scores_values,
            buckets,
            coverage=coverage,
        )
        hetero_base = point_prediction + heteroscedastic.upper_offset(features, coverage)
        methods["mission_heteroscedastic_laplace"][coverage] = TrajectoryConformalCorrection.fit(
            calibration.true_mission_energy,
            hetero_base,
            calibration.mission_ids,
            coverage=coverage,
            allow_negative=True,
        )
        quantile_base = quantile.predict_level(features, coverage)
        methods["mission_direct_quantile_cqr"][coverage] = TrajectoryConformalCorrection.fit(
            calibration.true_mission_energy,
            quantile_base,
            calibration.mission_ids,
            coverage=coverage,
            allow_negative=True,
        )
    return methods


def mission_method_predictions(
    point: EnergyToGoRegressor,
    heteroscedastic: HeteroscedasticResidualModel,
    quantile: MCSupervisedQuantileModel,
    calibrations: dict[str, dict[float, object]],
    dataset: PackedMissionDataset,
    coverage: float,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    point_fn = point_callable(point)
    _, _, point_prediction = mission_component_predictions(point_fn, dataset)
    features = np.concatenate([dataset.task_states, dataset.return_after_states], axis=1)
    results = {}
    correction = calibrations["mission_point_global"][coverage]
    results["mission_point_global"] = (
        point_prediction,
        point_prediction + correction.correction,
    )
    group = calibrations["mission_point_distance_group"][coverage]
    distances = dataset.task_states[:, -1].astype(np.float64) * D_MAX
    margins = np.asarray([group.margin_for(distance) for distance in distances])
    results["mission_point_distance_group"] = (
        point_prediction,
        point_prediction + margins,
    )
    correction = calibrations["mission_heteroscedastic_laplace"][coverage]
    results["mission_heteroscedastic_laplace"] = (
        point_prediction,
        point_prediction
        + heteroscedastic.upper_offset(features, coverage)
        + correction.correction,
    )
    quantile_prediction = quantile.predict_quantiles(features)
    q50 = quantile_prediction[:, quantile.quantile_levels.index(0.50)]
    q_level = quantile_prediction[:, quantile.quantile_levels.index(coverage)]
    correction = calibrations["mission_direct_quantile_cqr"][coverage]
    results["mission_direct_quantile_cqr"] = (
        q50,
        q_level + correction.correction,
    )
    return results


def summarize_method_results(
    goal_results: dict[str, dict[float, dict[str, object]]],
    mission_results: dict[str, dict[float, dict[str, object]]],
) -> pd.DataFrame:
    rows = []
    for method, by_coverage in goal_results.items():
        for coverage, result in by_coverage.items():
            intersections = result["by_goal_type_and_initial_distance"]
            rows.append(
                {
                    "scope": "goal",
                    "method": method,
                    "target_coverage": coverage,
                    "empirical_coverage": result["overall"]["whole_trajectory_simultaneous_coverage"],
                    "task_coverage": result["by_goal_type"].get("TASK", {}).get("whole_trajectory_simultaneous_coverage"),
                    "charger_coverage": result["by_goal_type"].get("CHARGER", {}).get("whole_trajectory_simultaneous_coverage"),
                    "worst_subgroup_coverage": min(
                        value["whole_trajectory_simultaneous_coverage"] for value in intersections.values()
                    ),
                    "mean_bound_width": result["overall"]["mean_bound_width"],
                    "p95_bound_width": result["overall"]["p95_bound_width"],
                    "conservatism_cost": result["overall"]["conservatism_cost"],
                    "unnecessary_return_proxy_rate": np.nan,
                    "estimated_task_acceptance_rate": np.nan,
                }
            )
    for method, by_coverage in mission_results.items():
        for coverage, result in by_coverage.items():
            buckets = result["by_initial_task_distance_bucket"]
            rows.append(
                {
                    "scope": "mission",
                    "method": method,
                    "target_coverage": coverage,
                    "empirical_coverage": result["overall"]["whole_trajectory_simultaneous_coverage"],
                    "task_coverage": np.nan,
                    "charger_coverage": np.nan,
                    "worst_subgroup_coverage": min(
                        value["whole_trajectory_simultaneous_coverage"] for value in buckets.values()
                    ),
                    "mean_bound_width": result["overall"]["mean_bound_width"],
                    "p95_bound_width": result["overall"]["p95_bound_width"],
                    "conservatism_cost": result["overall"]["conservatism_cost"],
                    "unnecessary_return_proxy_rate": result["overall"]["unnecessary_return_proxy_rate"],
                    "estimated_task_acceptance_rate": result["overall"]["estimated_task_acceptance_rate"],
                }
            )
    return pd.DataFrame(rows)


def select_method(
    frame: pd.DataFrame,
    scope: str,
    target: float = 0.95,
    allowed_methods: tuple[str, ...] | None = None,
) -> str:
    selected = frame[(frame.scope == scope) & np.isclose(frame.target_coverage, target)].copy()
    if allowed_methods is not None:
        selected = selected[selected.method.isin(allowed_methods)].copy()
    if selected.empty:
        raise ValueError("no methods remain after selection filtering")
    selected["coverage_shortfall"] = (
        np.maximum(target - selected.empirical_coverage, 0.0)
        + np.maximum(target - selected.worst_subgroup_coverage, 0.0)
    )
    selected = selected.sort_values(
        ["coverage_shortfall", "conservatism_cost", "mean_bound_width"],
        ascending=[True, True, True],
    )
    return str(selected.iloc[0].method)


def plot_tradeoff(frame: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    for scope, axis in zip(("goal", "mission"), axes):
        selected = frame[frame.scope == scope]
        for method, rows in selected.groupby("method"):
            axis.plot(
                rows["conservatism_cost"],
                rows["empirical_coverage"],
                marker="o",
                label=method,
            )
        axis.axhline(0.95, color="black", linestyle="--", linewidth=1)
        axis.set_title(scope)
        axis.set_xlabel("Mean upper minus true energy")
        axis.set_ylabel("Whole-trajectory coverage")
        axis.grid(alpha=0.25)
    axes[1].legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def oracle_forensic_analysis(
    policy,
    point: EnergyToGoRegressor,
    goal_dataset: PackedEnergyDataset,
    goal_frame: pd.DataFrame,
    mission_dataset: PackedMissionDataset,
    args: SimpleNamespace,
    *,
    max_cases: int = 50,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if max_cases < 0:
        raise ValueError("max_cases must be nonnegative")
    oracle = ModelBasedEnergyRolloutEstimator(policy, max_policy_steps=4000)
    goal_metadata = {int(row["trajectory_id"]): row for row in goal_dataset.metadata}
    goal_rows = []
    for row in goal_frame.sort_values("trajectory_max_underestimation", ascending=False).head(max_cases).to_dict("records"):
        metadata = goal_metadata[int(row["trajectory_id"])]
        environment = environment_from_args(args, phase=SACTrainingPhase.TD_PRETRAINING)
        environment.reset(
            seed=0,
            options={
                "start_position": np.asarray(metadata["start_position"], dtype=np.float32),
                "start_velocity": np.asarray(metadata["initial_velocity"], dtype=np.float32),
                "task_point": np.asarray(metadata["goal_position"], dtype=np.float32),
            },
        )
        result = oracle.estimate_context(
            environment,
            np.asarray(metadata["goal_position"], dtype=np.float32),
            goal_type=str(metadata["goal_type"]),
        )
        environment.close()
        goal_rows.append(
            {
                **row,
                "oracle_initial_energy": result.prediction,
                "oracle_rollout_steps": result.rollout_steps,
                "oracle_minus_actual_initial": result.prediction - float(row["true_initial_energy"]),
                "point_minus_oracle_initial": float(row["predicted_initial_energy"]) - result.prediction,
            }
        )

    point_fn = point_callable(point)
    task_prediction, return_prediction, mission_prediction = mission_component_predictions(
        point_fn,
        mission_dataset,
    )
    initial_rows = []
    for mission_id in np.unique(mission_dataset.mission_ids):
        index = int(np.flatnonzero(mission_dataset.mission_ids == mission_id)[0])
        initial_rows.append(
            {
                "mission_id": int(mission_id),
                "index": index,
                "true_mission_energy": float(mission_dataset.true_mission_energy[index]),
                "point_mission_energy": float(mission_prediction[index]),
                "mission_underestimation": float(
                    mission_dataset.true_mission_energy[index] - mission_prediction[index]
                ),
                "task_point_prediction": float(task_prediction[index]),
                "return_point_prediction": float(return_prediction[index]),
                "distance_bucket": str(mission_dataset.initial_distance_buckets[index]),
            }
        )
    probe = environment_from_args(args, phase=SACTrainingPhase.TD_PRETRAINING)
    charger = probe.charger_position.copy()
    probe.close()
    mission_rows = []
    for row in sorted(initial_rows, key=lambda item: item["mission_underestimation"], reverse=True)[:max_cases]:
        index = int(row["index"])
        task_state = mission_dataset.task_states[index].astype(np.float64)
        return_state = mission_dataset.return_after_states[index].astype(np.float64)
        task_goal = charger.astype(np.float64) - return_state[3:6] * return_state[-1] * D_MAX
        start = task_goal - task_state[3:6] * task_state[-1] * D_MAX
        velocity = task_state[:3] * VELOCITY_SCALE
        environment = environment_from_args(args, phase=SACTrainingPhase.TD_PRETRAINING)
        environment.reset(
            seed=0,
            options={
                "start_position": start.astype(np.float32),
                "start_velocity": velocity.astype(np.float32),
                "task_point": task_goal.astype(np.float32),
            },
        )
        oracle_task = oracle.estimate_context(environment, task_goal.astype(np.float32), goal_type="TASK")
        oracle_return = oracle.estimate_context(
            environment,
            charger,
            position=task_goal.astype(np.float32),
            velocity=np.zeros(3, dtype=np.float32),
            goal_type="TASK_ENDPOINT_TO_CHARGER",
        )
        environment.close()
        oracle_mission = oracle_task.prediction + oracle_return.prediction
        mission_rows.append(
            {
                **row,
                "oracle_task_energy": oracle_task.prediction,
                "oracle_return_energy": oracle_return.prediction,
                "oracle_mission_energy": oracle_mission,
                "oracle_minus_actual_mission": oracle_mission - float(row["true_mission_energy"]),
                "point_minus_oracle_mission": float(row["point_mission_energy"]) - oracle_mission,
            }
        )
    return goal_rows, mission_rows


def run(args: argparse.Namespace) -> dict[str, object]:
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    for directory in (
        "diagnostics",
        "models",
        "calibration",
        "diagnostic_comparison",
        "fresh_test_v3",
        "short_phase2",
    ):
        (output / directory).mkdir()
    write_json(output / "RUNNING.json", {"status": "RUNNING", "started_at": utc_now(), "pid": os.getpid()})
    torch.set_num_threads(args.torch_num_threads)
    try:
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
        diagnostic_test = PackedEnergyDataset.load(V2 / "final_conformal_test_v2/trajectories")
        mission_development = PackedMissionDataset.load(V2 / "mission_calibration/trajectories")
        mission_diagnostic = PackedMissionDataset.load(V2 / "mission_test_v2/trajectories")
        mission_train, mission_validation, mission_calibration = stratified_mission_split(
            mission_development,
            seed=args.model_seed + 10,
        )
        config = {
            "git_sha": git_sha(),
            "point_checkpoint": str(POINT_CHECKPOINT.resolve()),
            "point_checkpoint_sha256": file_sha256(POINT_CHECKPOINT),
            "group_conformal_checkpoint": str(GROUP_CONFORMAL_CHECKPOINT.resolve()),
            "group_conformal_checkpoint_sha256": file_sha256(GROUP_CONFORMAL_CHECKPOINT),
            "sac_checkpoint": str(SAC_CHECKPOINT.resolve()),
            "sac_checkpoint_sha256": file_sha256(SAC_CHECKPOINT),
            "point_estimator_retrained": False,
            "sac_retrained": False,
            "risk_levels": RISK_COVERAGE_LEVELS,
            "model_selection_test": str((V2 / "final_conformal_test_v2").resolve()),
            "fresh_test_v3_seed": args.fresh_goal_seed,
            "fresh_mission_v3_seed": args.fresh_mission_seed,
            "mission_development_split": {"train": 600, "validation": 200, "calibration": 200},
            "short_phase2_transitions": args.short_phase2_transitions,
            "forensic_cases": args.forensic_cases,
        }
        write_json(output / "config.json", config)

        report_stage(output, "diagnostics_started")
        diagnostic_prediction = chunked_point_predictions(point, diagnostic_test.states)
        trajectory_frame = trajectory_diagnostic_frame(diagnostic_test, diagnostic_prediction)
        trajectory_frame.to_csv(output / "diagnostics/trajectory_residuals.csv", index=False)
        correlations = correlation_table(
            trajectory_frame,
            targets=["trajectory_max_underestimation", "std_residual", "p95_residual"],
            excluded=["trajectory_id"],
        )
        correlations.to_csv(output / "diagnostics/trajectory_correlations.csv", index=False)
        group_summary = residual_group_summary(
            trajectory_frame,
            ["goal_type", "distance_bucket"],
        )
        group_summary.to_csv(output / "diagnostics/residual_group_summary.csv", index=False)
        aliasing, alias_context = nearest_neighbor_aliasing_analysis(
            diagnostic_test,
            sample_size=args.aliasing_sample_size,
            seed=args.model_seed,
        )
        write_json(output / "diagnostics/aliasing_summary.json", aliasing)
        alias_context.to_csv(output / "diagnostics/aliasing_context_explanations.csv", index=False)
        dependence = mission_residual_dependence(point_callable(point), mission_diagnostic)
        write_json(output / "diagnostics/mission_residual_dependence.json", dependence)
        trajectory_frame.sort_values("trajectory_max_underestimation", ascending=False).head(50).to_csv(
            output / "diagnostics/top50_trajectory_underestimations.csv",
            index=False,
        )
        report_stage(output, "oracle_forensics_started", max_cases=args.forensic_cases)
        oracle_goal, oracle_mission = oracle_forensic_analysis(
            policy,
            point,
            diagnostic_test,
            trajectory_frame,
            mission_diagnostic,
            environment_args(seed=args.model_seed),
            max_cases=args.forensic_cases,
        )
        write_json(output / "diagnostics/oracle_top50_goal_forensics.json", oracle_goal)
        write_json(output / "diagnostics/oracle_top50_mission_forensics.json", oracle_mission)
        report_stage(output, "diagnostics_completed")

        report_stage(output, "adaptive_training_started", max_epochs=args.max_epochs)
        train_prediction = chunked_point_predictions(point, train.states)
        validation_prediction = chunked_point_predictions(point, validation.states)
        heteroscedastic = {}
        hetero_histories = {}
        for index, distribution in enumerate(("gaussian", "laplace")):
            model = HeteroscedasticResidualModel(
                input_dim=7,
                distribution=distribution,
                seed=args.model_seed + index,
                device=args.device,
            )
            history = model.fit(
                train.states,
                train.targets - train_prediction,
                validation.states,
                validation.targets - validation_prediction,
                max_epochs=args.max_epochs,
            )
            model.save(output / f"models/goal_heteroscedastic_{distribution}.pt")
            heteroscedastic[distribution] = model
            hetero_histories[distribution] = history.as_dict()
        quantile = MCSupervisedQuantileModel(
            battery_capacity=BATTERY_CAPACITY,
            input_dim=7,
            seed=args.model_seed + 2,
            device=args.device,
        )
        quantile_history = quantile.fit(
            train.states,
            train.targets,
            validation.states,
            validation.targets,
            max_epochs=args.max_epochs,
        )
        quantile.save(output / "models/goal_supervised_quantiles.pt")
        write_json(
            output / "models/goal_training_history.json",
            {
                "heteroscedastic": hetero_histories,
                "quantile": quantile_history.as_dict(),
                "validation_raw_quantile_crossing_rate": quantile.raw_crossing_rate(validation.states),
            },
        )

        mission_point_fn = point_callable(point)
        _, _, mission_train_point = mission_component_predictions(mission_point_fn, mission_train)
        _, _, mission_validation_point = mission_component_predictions(mission_point_fn, mission_validation)
        mission_train_features = np.concatenate(
            [mission_train.task_states, mission_train.return_after_states], axis=1
        )
        mission_validation_features = np.concatenate(
            [mission_validation.task_states, mission_validation.return_after_states], axis=1
        )
        mission_hetero = HeteroscedasticResidualModel(
            input_dim=14,
            distribution="laplace",
            seed=args.model_seed + 3,
            device=args.device,
        )
        mission_hetero_history = mission_hetero.fit(
            mission_train_features,
            mission_train.true_mission_energy - mission_train_point,
            mission_validation_features,
            mission_validation.true_mission_energy - mission_validation_point,
            max_epochs=args.max_epochs,
        )
        mission_hetero.save(output / "models/mission_heteroscedastic_laplace.pt")
        mission_quantile = MCSupervisedQuantileModel(
            battery_capacity=BATTERY_CAPACITY,
            input_dim=14,
            seed=args.model_seed + 4,
            device=args.device,
        )
        mission_quantile_history = mission_quantile.fit(
            mission_train_features,
            mission_train.true_mission_energy,
            mission_validation_features,
            mission_validation.true_mission_energy,
            max_epochs=args.max_epochs,
        )
        mission_quantile.save(output / "models/mission_direct_quantiles.pt")
        write_json(
            output / "models/mission_training_history.json",
            {
                "heteroscedastic": mission_hetero_history.as_dict(),
                "quantile": mission_quantile_history.as_dict(),
                "validation_raw_quantile_crossing_rate": mission_quantile.raw_crossing_rate(
                    mission_validation_features
                ),
            },
        )
        report_stage(output, "adaptive_training_completed")

        report_stage(output, "calibration_started")
        goal_calibrations = calibrate_goal_methods(
            point,
            heteroscedastic,
            quantile,
            calibration,
        )
        mission_calibrations = calibrate_mission_methods(
            point,
            mission_hetero,
            mission_quantile,
            mission_calibration,
        )
        write_json(
            output / "calibration/goal_calibrations.json",
            {
                method: {str(level): value.as_dict() for level, value in levels.items()}
                for method, levels in goal_calibrations.items()
            },
        )
        write_json(
            output / "calibration/mission_calibrations.json",
            {
                method: {str(level): value.as_dict() for level, value in levels.items()}
                for method, levels in mission_calibrations.items()
            },
        )
        report_stage(output, "calibration_completed")

        report_stage(output, "diagnostic_method_comparison_started")
        rng = np.random.default_rng(args.model_seed + 100)
        diagnostic_remaining = rng.uniform(
            0.0,
            BATTERY_CAPACITY,
            size=mission_diagnostic.true_mission_energy.shape,
        )
        goal_diagnostic_results: dict[str, dict[float, dict[str, object]]] = {}
        mission_diagnostic_results: dict[str, dict[float, dict[str, object]]] = {}
        for coverage in RISK_COVERAGE_LEVELS:
            predictions = goal_method_predictions(
                point,
                heteroscedastic,
                quantile,
                goal_calibrations,
                diagnostic_test,
                coverage,
            )
            for method, (center, upper) in predictions.items():
                goal_diagnostic_results.setdefault(method, {})[coverage] = evaluate_goal_upper(
                    diagnostic_test,
                    center,
                    upper,
                )
            predictions = mission_method_predictions(
                point,
                mission_hetero,
                mission_quantile,
                mission_calibrations,
                mission_diagnostic,
                coverage,
            )
            for method, (center, upper) in predictions.items():
                mission_diagnostic_results.setdefault(method, {})[coverage] = evaluate_mission_upper(
                    mission_diagnostic,
                    center,
                    upper,
                    battery_capacity=BATTERY_CAPACITY,
                    reserve_fraction=0.10,
                    remaining_energy_samples=diagnostic_remaining,
                )
        write_json(output / "diagnostic_comparison/goal_results.json", goal_diagnostic_results)
        write_json(output / "diagnostic_comparison/mission_results.json", mission_diagnostic_results)
        diagnostic_matrix = summarize_method_results(
            goal_diagnostic_results,
            mission_diagnostic_results,
        )
        diagnostic_matrix.to_csv(output / "diagnostic_comparison/method_matrix.csv", index=False)
        plot_tradeoff(diagnostic_matrix, output / "diagnostic_comparison/safety_efficiency_tradeoff.png")
        selected_goal_overall = select_method(diagnostic_matrix, "goal")
        selected_mission_overall = select_method(diagnostic_matrix, "mission")
        selected_goal = select_method(
            diagnostic_matrix,
            "goal",
            allowed_methods=(
                "heteroscedastic_gaussian",
                "heteroscedastic_laplace",
                "supervised_quantile_cqr",
            ),
        )
        selected_mission = select_method(
            diagnostic_matrix,
            "mission",
            allowed_methods=(
                "mission_heteroscedastic_laplace",
                "mission_direct_quantile_cqr",
            ),
        )
        write_json(
            output / "METHOD_SELECTION.json",
            {
                "selection_data": "exposed diagnostic test v2; not final evidence",
                "best_goal_method_including_baselines": selected_goal_overall,
                "best_mission_method_including_baselines": selected_mission_overall,
                "selected_goal_method": selected_goal,
                "selected_mission_method": selected_mission,
            },
        )
        report_stage(
            output,
            "diagnostic_method_comparison_completed",
            selected_goal_method=selected_goal,
            selected_mission_method=selected_mission,
        )

        report_stage(output, "fresh_test_v3_collection_started")
        fresh_args = environment_args(seed=args.model_seed)
        fresh_goal = collect_intersection_split(
            policy,
            fresh_args,
            count=args.fresh_goal_trajectories,
            seed=args.fresh_goal_seed,
            trajectory_id_offset=6_000_000,
            output=output / "fresh_test_v3/goal_trajectories",
            label="fresh_goal_v3",
        )
        fresh_mission = collect_mission_split(
            policy,
            fresh_args,
            count=args.fresh_mission_trajectories,
            seed=args.fresh_mission_seed,
            trajectory_id_offset=7_000_000,
            output=output / "fresh_test_v3/mission_trajectories",
            label="fresh_mission_v3",
        )
        fresh_remaining = rng.uniform(
            0.0,
            BATTERY_CAPACITY,
            size=fresh_mission.true_mission_energy.shape,
        )
        goal_final_results: dict[str, dict[float, dict[str, object]]] = {}
        mission_final_results: dict[str, dict[float, dict[str, object]]] = {}
        for coverage in RISK_COVERAGE_LEVELS:
            for method, (center, upper) in goal_method_predictions(
                point,
                heteroscedastic,
                quantile,
                goal_calibrations,
                fresh_goal,
                coverage,
            ).items():
                goal_final_results.setdefault(method, {})[coverage] = evaluate_goal_upper(
                    fresh_goal,
                    center,
                    upper,
                )
            for method, (center, upper) in mission_method_predictions(
                point,
                mission_hetero,
                mission_quantile,
                mission_calibrations,
                fresh_mission,
                coverage,
            ).items():
                mission_final_results.setdefault(method, {})[coverage] = evaluate_mission_upper(
                    fresh_mission,
                    center,
                    upper,
                    battery_capacity=BATTERY_CAPACITY,
                    reserve_fraction=0.10,
                    remaining_energy_samples=fresh_remaining,
                )
        write_json(output / "fresh_test_v3/goal_results.json", goal_final_results)
        write_json(output / "fresh_test_v3/mission_results.json", mission_final_results)
        final_matrix = summarize_method_results(goal_final_results, mission_final_results)
        final_matrix.to_csv(output / "fresh_test_v3/method_matrix.csv", index=False)
        plot_tradeoff(final_matrix, output / "fresh_test_v3/safety_efficiency_tradeoff.png")
        report_stage(output, "fresh_test_v3_completed")

        selected_goal_model = quantile if selected_goal == "supervised_quantile_cqr" else heteroscedastic[
            selected_goal.removeprefix("heteroscedastic_")
        ]
        selected_mission_model = mission_quantile if selected_mission == "mission_direct_quantile_cqr" else mission_hetero
        adaptive = AdaptiveConformalEnergyEstimator(
            point,
            selected_goal_model,
            goal_calibrations[selected_goal][0.95],
            selected_mission_model,
            mission_calibrations[selected_mission][0.95],
            coverage=0.95,
        )
        adaptive.save(output / "models/selected_adaptive_estimator.pt")
        phase2_args = environment_args(
            seed=args.short_phase2_seed,
            phase2_budget=args.short_phase2_transitions,
        )
        report_stage(output, "short_phase2_started", transitions=args.short_phase2_transitions)
        baseline = HierarchicalConformalEnergyEstimator.load(
            GROUP_CONFORMAL_CHECKPOINT,
            device=args.device,
        )
        baseline_phase2 = run_phase2(
            policy,
            baseline,
            phase2_args,
            capacity=BATTERY_CAPACITY,
            output=output / "short_phase2/group_conformal_baseline",
        )
        adaptive_phase2 = run_phase2(
            policy,
            adaptive,
            phase2_args,
            capacity=BATTERY_CAPACITY,
            output=output / "short_phase2/selected_adaptive",
        )
        short_phase2 = {
            "transitions_per_method": args.short_phase2_transitions,
            "same_seed": args.short_phase2_seed,
            "group_conformal_baseline": baseline_phase2,
            "selected_adaptive": adaptive_phase2,
        }
        write_json(output / "short_phase2/summary.json", short_phase2)
        report_stage(output, "short_phase2_completed")

        completed = {
            "status": "COMPLETED",
            "completed_at": utc_now(),
            "selected_goal_method": selected_goal,
            "selected_mission_method": selected_mission,
            "short_phase2": short_phase2,
            "formal_500k_phase2_launched": False,
        }
        write_json(output / "COMPLETED.json", completed)
        (output / "RUNNING.json").unlink(missing_ok=True)
        return completed
    except KeyboardInterrupt:
        write_json(
            output / "INTERRUPTED.json",
            {
                "status": "INTERRUPTED",
                "interrupted_at": utc_now(),
            },
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
    parser = argparse.ArgumentParser(description="Explore adaptive MC Energy-to-Go uncertainty")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--navigation-device", default="cpu")
    parser.add_argument("--torch-num-threads", type=int, default=1)
    parser.add_argument("--model-seed", type=int, default=610_001)
    parser.add_argument("--fresh-goal-seed", type=int, default=620_001)
    parser.add_argument("--fresh-mission-seed", type=int, default=630_001)
    parser.add_argument("--short-phase2-seed", type=int, default=640_001)
    parser.add_argument("--fresh-goal-trajectories", type=int, default=2000)
    parser.add_argument("--fresh-mission-trajectories", type=int, default=1000)
    parser.add_argument("--aliasing-sample-size", type=int, default=100_000)
    parser.add_argument("--forensic-cases", type=int, default=50)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--short-phase2-transitions", type=int, default=20_000)
    args = parser.parse_args(argv)
    if args.forensic_cases < 0:
        parser.error("--forensic-cases must be nonnegative")
    if len({args.model_seed, args.fresh_goal_seed, args.fresh_mission_seed, args.short_phase2_seed}) != 4:
        parser.error("all exploration seeds must be distinct")
    return args


if __name__ == "__main__":
    run(parse_args())
