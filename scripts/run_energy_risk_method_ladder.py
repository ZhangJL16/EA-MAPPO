from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from experiments.energy_mc.adaptive_analysis import (
    contiguous_trajectory_slices,
    mission_component_predictions,
    mission_residual_dependence,
)
from experiments.energy_mc.adaptive_uncertainty import (
    HeteroscedasticResidualModel,
    chunked_point_predictions,
)
from experiments.energy_mc.conformal import PackedMissionDataset
from experiments.energy_mc.conditional_risk import (
    GoalRiskFeatureBuilder,
    PositiveResidualQuantileModel,
    positive_underestimation_target,
    trajectory_balanced_state_indices,
)
from experiments.energy_mc.core import PackedEnergyDataset
from experiments.energy_mc.final_risk import (
    MondrianMissionCalibration,
    MondrianTrajectoryCalibration,
    future_max_underestimation_target,
    primary_goal_group,
    unnecessary_return_indicator,
)
from review_bundle.safety.energy.mc_regression import EnergyToGoRegressor
from scripts.explore_adaptive_energy_uncertainty import point_callable
from scripts.run_energy_uncertainty_v4 import (
    BATTERY_CAPACITY,
    D_MAX,
    MISSION_MODEL_CHECKPOINT,
    POINT_CHECKPOINT,
    SOURCE,
    V2,
)


ROOT = Path(__file__).resolve().parents[1]
LEVELS = (0.90, 0.95, 0.975, 0.99)
RESERVE = 0.10 * BATTERY_CAPACITY


def correlation(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    return {
        "pearson": float(pearsonr(x, y).statistic),
        "spearman": float(spearmanr(x, y).statistic),
    }


def goal_metrics(
    dataset: PackedEnergyDataset,
    point: np.ndarray,
    risk: np.ndarray,
    upper: np.ndarray,
    *,
    remaining_energy: np.ndarray,
) -> dict[str, object]:
    rows = []
    for trajectory_id, selected in contiguous_trajectory_slices(dataset.trajectory_ids):
        group = primary_goal_group(
            str(dataset.goal_types[selected.start]),
            str(dataset.distance_buckets[selected.start]),
        )
        rows.append(
            {
                "trajectory_id": trajectory_id,
                "group": group,
                "goal_type": group.split("|", 1)[0],
                "covered": bool(np.all(dataset.targets[selected] <= upper[selected])),
            }
        )
    frame = pd.DataFrame(rows)
    grouped = {
        str(name): {
            "count": int(len(group)),
            "covered": int(group.covered.sum()),
            "coverage": float(group.covered.mean()),
        }
        for name, group in frame.groupby("group")
    }
    goal_type = {
        str(name): {
            "count": int(len(group)),
            "covered": int(group.covered.sum()),
            "coverage": float(group.covered.mean()),
        }
        for name, group in frame.groupby("goal_type")
    }
    truth = dataset.targets.astype(np.float64)
    unnecessary = unnecessary_return_indicator(
        remaining_energy,
        truth,
        upper,
        RESERVE,
    )
    return {
        "overall": {
            "count": int(len(frame)),
            "covered": int(frame.covered.sum()),
            "coverage": float(frame.covered.mean()),
        },
        "by_primary_group": grouped,
        "by_goal_type": goal_type,
        "worst_primary_group": min(grouped, key=lambda name: grouped[name]["coverage"]),
        "worst_primary_coverage": min(row["coverage"] for row in grouped.values()),
        "mean_bound_width": float(np.mean(upper - point)),
        "p95_bound_width": float(np.quantile(upper - point, 0.95)),
        "mean_conservatism": float(np.mean(upper - truth)),
        "unnecessary_return_proxy_rate": float(np.mean(unnecessary)),
        "point_mae": float(np.mean(np.abs(point - truth))),
        "risk_mean": float(np.mean(risk)),
        "risk_p95": float(np.quantile(risk, 0.95)),
    }


def mission_metrics(
    dataset: PackedMissionDataset,
    point: np.ndarray,
    risk: np.ndarray,
    upper: np.ndarray,
    *,
    remaining_energy: np.ndarray,
) -> dict[str, object]:
    rows = []
    for mission_id, selected in contiguous_trajectory_slices(dataset.mission_ids):
        rows.append(
            {
                "mission_id": mission_id,
                "group": str(dataset.initial_distance_buckets[selected.start]),
                "covered": bool(np.all(dataset.true_mission_energy[selected] <= upper[selected])),
            }
        )
    frame = pd.DataFrame(rows)
    grouped = {
        str(name): {
            "count": int(len(group)),
            "covered": int(group.covered.sum()),
            "coverage": float(group.covered.mean()),
        }
        for name, group in frame.groupby("group")
    }
    truth = dataset.true_mission_energy.astype(np.float64)
    unnecessary = unnecessary_return_indicator(
        remaining_energy,
        truth,
        upper,
        RESERVE,
    )
    return {
        "overall": {
            "count": int(len(frame)),
            "covered": int(frame.covered.sum()),
            "coverage": float(frame.covered.mean()),
        },
        "by_primary_group": grouped,
        "worst_primary_group": min(grouped, key=lambda name: grouped[name]["coverage"]),
        "worst_primary_coverage": min(row["coverage"] for row in grouped.values()),
        "mean_bound_width": float(np.mean(upper - point)),
        "p95_bound_width": float(np.quantile(upper - point, 0.95)),
        "mean_conservatism": float(np.mean(upper - truth)),
        "unnecessary_return_proxy_rate": float(np.mean(unnecessary)),
        "point_mae": float(np.mean(np.abs(point - truth))),
        "risk_mean": float(np.mean(risk)),
    }


def sigma_deciles(scale: np.ndarray, actual: np.ndarray) -> list[dict[str, float | int]]:
    frame = pd.DataFrame({"scale": scale, "actual": actual})
    frame["decile"] = pd.qcut(frame.scale, 10, labels=False, duplicates="drop")
    return [
        {
            "decile": int(name),
            "count": int(len(group)),
            "mean_predicted_scale": float(group.scale.mean()),
            "mean_actual_future_max_underestimation": float(group.actual.mean()),
            "p95_actual_future_max_underestimation": float(group.actual.quantile(0.95)),
        }
        for name, group in frame.groupby("decile")
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Controlled Energy Risk method ladder")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts/energy_risk_v5_development/method_ladder",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-epochs", type=int, default=16)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--states-per-trajectory", type=int, default=128)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "models").mkdir(exist_ok=True)

    point_model = EnergyToGoRegressor.load(POINT_CHECKPOINT, device=args.device)
    train = PackedEnergyDataset.load(SOURCE / "energy_dataset/train")
    validation = PackedEnergyDataset.load(SOURCE / "energy_dataset/validation")
    calibration = PackedEnergyDataset.concatenate(
        [
            PackedEnergyDataset.load(SOURCE / "energy_dataset/calibration"),
            PackedEnergyDataset.load(V2 / "calibration_v2/new_trajectories"),
        ]
    )
    diagnostic = PackedEnergyDataset.load(V2 / "final_conformal_test_v2/trajectories")
    point = {
        "train": chunked_point_predictions(point_model, train.states),
        "validation": chunked_point_predictions(point_model, validation.states),
        "calibration": chunked_point_predictions(point_model, calibration.states),
        "diagnostic": chunked_point_predictions(point_model, diagnostic.states),
    }
    builder = GoalRiskFeatureBuilder("compact_decision_context")
    features = {
        "train": builder.build_from_dataset(train),
        "validation": builder.build_from_dataset(validation),
        "calibration": builder.build_from_dataset(calibration),
        "diagnostic": builder.build_from_dataset(diagnostic),
    }
    train_indices = trajectory_balanced_state_indices(
        train.trajectory_ids, max_states_per_trajectory=args.states_per_trajectory
    )
    validation_indices = trajectory_balanced_state_indices(
        validation.trajectory_ids, max_states_per_trajectory=args.states_per_trajectory
    )
    state_target = {
        "train": positive_underestimation_target(train.targets, point["train"]),
        "validation": positive_underestimation_target(validation.targets, point["validation"]),
    }
    suffix_target = {
        "train": future_max_underestimation_target(
            train.targets, point["train"], train.trajectory_ids
        ),
        "validation": future_max_underestimation_target(
            validation.targets, point["validation"], validation.trajectory_ids
        ),
        "diagnostic": future_max_underestimation_target(
            diagnostic.targets, point["diagnostic"], diagnostic.trajectory_ids
        ),
    }
    k1 = PositiveResidualQuantileModel(
        input_dim=builder.input_dim, seed=820_101, device=args.device
    )
    k1_history = k1.fit(
        features["train"][train_indices],
        state_target["train"][train_indices],
        features["validation"][validation_indices],
        state_target["validation"][validation_indices],
        max_epochs=args.max_epochs,
        patience=args.patience,
    )
    k1.save(args.output_dir / "models/k1_state_residual.pt")
    k2 = PositiveResidualQuantileModel(
        input_dim=builder.input_dim, seed=820_102, device=args.device
    )
    k2_history = k2.fit(
        features["train"][train_indices],
        suffix_target["train"][train_indices],
        features["validation"][validation_indices],
        suffix_target["validation"][validation_indices],
        max_epochs=args.max_epochs,
        patience=args.patience,
    )
    k2.save(args.output_dir / "models/k2_suffix_max_residual.pt")
    scale_model = HeteroscedasticResidualModel(
        input_dim=builder.input_dim,
        distribution="laplace",
        seed=820_103,
        device=args.device,
    )
    scale_history = scale_model.fit(
        features["train"][train_indices],
        train.targets[train_indices] - point["train"][train_indices],
        features["validation"][validation_indices],
        validation.targets[validation_indices] - point["validation"][validation_indices],
        max_epochs=args.max_epochs,
        patience=args.patience,
    )
    scale_model.save(args.output_dir / "models/normalized_scale.pt")
    histories = {
        "k1": k1_history.as_dict(),
        "k2": k2_history.as_dict(),
        "normalized_scale": scale_history.as_dict(),
    }
    (args.output_dir / "training_histories.json").write_text(
        json.dumps(histories, indent=2, sort_keys=True) + "\n"
    )

    rng = np.random.default_rng(820_200)
    goal_remaining = rng.uniform(0.0, BATTERY_CAPACITY, size=diagnostic.targets.shape)
    goal_results: dict[str, object] = {}
    risk_predictions = {
        "point_only": {
            level: (np.zeros_like(point["calibration"]), np.zeros_like(point["diagnostic"]), "additive")
            for level in LEVELS
        },
        "normalized_scale": {
            level: (
                scale_model.predict_scale(features["calibration"]),
                scale_model.predict_scale(features["diagnostic"]),
                "scaled",
            )
            for level in LEVELS
        },
    }
    for name, model in (("k1_state", k1), ("k2_suffix", k2)):
        risk_predictions[name] = {
            level: (
                model.predict_level(features["calibration"], level),
                model.predict_level(features["diagnostic"], level),
                "additive",
            )
            for level in LEVELS
        }
    risk_predictions["k3_hybrid_max"] = {
        level: (
            np.maximum(
                risk_predictions["k1_state"][level][0],
                risk_predictions["k2_suffix"][level][0],
            ),
            np.maximum(
                risk_predictions["k1_state"][level][1],
                risk_predictions["k2_suffix"][level][1],
            ),
            "additive",
        )
        for level in LEVELS
    }
    for method, levels in risk_predictions.items():
        goal_results[method] = {}
        for level, (calibration_risk, diagnostic_risk, mode) in levels.items():
            fitted = MondrianTrajectoryCalibration.fit(
                calibration.targets,
                point["calibration"],
                calibration_risk,
                calibration.trajectory_ids,
                calibration.goal_types,
                calibration.distance_buckets,
                coverage=level,
                mode=mode,
                minimum_group_trajectories=100,
            )
            upper = fitted.apply(
                point["diagnostic"],
                diagnostic_risk,
                diagnostic.goal_types,
                diagnostic.states[:, -1].astype(np.float64) * D_MAX,
            )
            goal_results[method][str(level)] = {
                "metrics": goal_metrics(
                    diagnostic,
                    point["diagnostic"],
                    diagnostic_risk,
                    upper,
                    remaining_energy=goal_remaining,
                ),
                "calibration": fitted.as_dict(),
            }
    diagnostic_scale = scale_model.predict_scale(features["diagnostic"])
    goal_results["normalized_scale_diagnostics"] = {
        "correlation_with_suffix_max_underestimation": correlation(
            diagnostic_scale, suffix_target["diagnostic"]
        ),
        "deciles": sigma_deciles(diagnostic_scale, suffix_target["diagnostic"]),
    }
    goal_results["k1_correlation_with_suffix_max"] = correlation(
        risk_predictions["k1_state"][0.95][1], suffix_target["diagnostic"]
    )
    goal_results["k2_correlation_with_suffix_max"] = correlation(
        risk_predictions["k2_suffix"][0.95][1], suffix_target["diagnostic"]
    )
    (args.output_dir / "goal_method_results.json").write_text(
        json.dumps(goal_results, indent=2, sort_keys=True) + "\n"
    )

    mission_calibration = PackedMissionDataset.load(V2 / "mission_calibration/trajectories")
    mission_diagnostic = PackedMissionDataset.load(V2 / "mission_test_v2/trajectories")
    mission_model = HeteroscedasticResidualModel.load(
        MISSION_MODEL_CHECKPOINT, device=args.device
    )
    point_fn = point_callable(point_model)
    _, _, mission_calibration_point = mission_component_predictions(
        point_fn, mission_calibration
    )
    _, _, mission_diagnostic_point = mission_component_predictions(
        point_fn, mission_diagnostic
    )
    mission_calibration_features = np.concatenate(
        [mission_calibration.task_states, mission_calibration.return_after_states], axis=1
    )
    mission_diagnostic_features = np.concatenate(
        [mission_diagnostic.task_states, mission_diagnostic.return_after_states], axis=1
    )
    mission_results: dict[str, object] = {}
    mission_remaining = rng.uniform(
        0.0, BATTERY_CAPACITY, size=mission_diagnostic.true_mission_energy.shape
    )
    for level in LEVELS:
        calibration_risk = mission_model.upper_offset(mission_calibration_features, level)
        diagnostic_risk = mission_model.upper_offset(mission_diagnostic_features, level)
        fitted = MondrianMissionCalibration.fit(
            mission_calibration.true_mission_energy,
            mission_calibration_point + calibration_risk,
            mission_calibration.mission_ids,
            mission_calibration.initial_distance_buckets,
            coverage=level,
            minimum_group_missions=100,
        )
        upper = fitted.apply(
            mission_diagnostic_point + diagnostic_risk,
            mission_diagnostic.initial_distance_buckets,
        )
        mission_results[str(level)] = {
            "metrics": mission_metrics(
                mission_diagnostic,
                mission_diagnostic_point,
                diagnostic_risk,
                upper,
                remaining_energy=mission_remaining,
            ),
            "calibration": fitted.as_dict(),
        }
    mission_residual = (
        mission_diagnostic.true_mission_energy.astype(np.float64)
        - mission_diagnostic_point
    )
    mission_rows = []
    for bucket in sorted(np.unique(mission_diagnostic.initial_distance_buckets)):
        selected = mission_diagnostic.initial_distance_buckets == bucket
        residual = mission_residual[selected]
        scale = mission_model.predict_scale(mission_diagnostic_features[selected])
        mission_rows.append(
            {
                "distance_bucket": str(bucket),
                "count_states": int(np.sum(selected)),
                "residual_mean": float(np.mean(residual)),
                "residual_std": float(np.std(residual)),
                "residual_p95": float(np.quantile(residual, 0.95)),
                "residual_p99": float(np.quantile(residual, 0.99)),
                **{
                    f"scale_vs_positive_residual_{key}": value
                    for key, value in correlation(scale, np.maximum(residual, 0.0)).items()
                },
            }
        )
    mission_results["residual_by_distance"] = mission_rows
    mission_results["task_return_residual_dependence"] = mission_residual_dependence(
        point_fn, mission_diagnostic
    )
    (args.output_dir / "mission_method_results.json").write_text(
        json.dumps(mission_results, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
