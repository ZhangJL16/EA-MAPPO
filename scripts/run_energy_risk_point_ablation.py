from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.energy_mc.adaptive_analysis import contiguous_trajectory_slices
from experiments.energy_mc.adaptive_uncertainty import (
    chunked_point_predictions,
    trajectory_max_scores,
)
from experiments.energy_mc.conditional_risk import (
    GoalRiskFeatureBuilder,
    reconstruct_positions,
    trajectory_balanced_state_indices,
)
from experiments.energy_mc.core import PackedEnergyDataset
from experiments.energy_mc.final_risk import (
    ContextualEnergyRegressor,
    MondrianTrajectoryCalibration,
    canonical_goal_type,
    primary_goal_group,
)
from review_bundle.safety.energy.mc_regression import EnergyToGoRegressor
from scripts.run_energy_uncertainty_v4 import BATTERY_CAPACITY, D_MAX, POINT_CHECKPOINT, SOURCE, V2


ROOT = Path(__file__).resolve().parents[1]
FEATURE_MODES = (
    "state7",
    "absolute_position",
    "boundary_distances",
    "compact_decision_context",
    "position_and_boundary",
)


def _point_metrics(
    dataset: PackedEnergyDataset,
    prediction: np.ndarray,
) -> dict[str, float]:
    truth = dataset.targets.astype(np.float64)
    residual = truth - prediction
    _, trajectory_residual = trajectory_max_scores(truth, prediction, dataset.trajectory_ids)
    positive = np.maximum(residual, 0.0)
    trajectory_positive = np.maximum(trajectory_residual, 0.0)
    return {
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "bias": float(np.mean(prediction - truth)),
        "state_positive_underestimation_p95": float(np.quantile(positive, 0.95)),
        "state_positive_underestimation_p99": float(np.quantile(positive, 0.99)),
        "trajectory_max_underestimation_p95": float(np.quantile(trajectory_positive, 0.95)),
        "trajectory_max_underestimation_p99": float(np.quantile(trajectory_positive, 0.99)),
    }


def _trajectory_group_coverage(
    dataset: PackedEnergyDataset,
    upper: np.ndarray,
) -> dict[str, object]:
    rows = []
    for trajectory_id, selected in contiguous_trajectory_slices(dataset.trajectory_ids):
        goal = str(dataset.goal_types[selected.start])
        bucket = str(dataset.distance_buckets[selected.start])
        rows.append(
            {
                "trajectory_id": trajectory_id,
                "group": primary_goal_group(goal, bucket),
                "covered": bool(np.all(dataset.targets[selected] <= upper[selected])),
            }
        )
    frame = pd.DataFrame(rows)
    groups = {
        str(name): {
            "count": int(len(group)),
            "covered": int(group.covered.sum()),
            "coverage": float(group.covered.mean()),
        }
        for name, group in frame.groupby("group")
    }
    return {
        "overall": float(frame.covered.mean()),
        "groups": groups,
        "worst_group": min(groups, key=lambda name: groups[name]["coverage"]),
        "worst_group_coverage": min(row["coverage"] for row in groups.values()),
    }


def _dataset_features(
    dataset: PackedEnergyDataset,
    builder: GoalRiskFeatureBuilder,
) -> np.ndarray:
    return builder.build(dataset.states, reconstruct_positions(dataset))


def main() -> None:
    parser = argparse.ArgumentParser(description="Controlled contextual point-model ablation")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts/energy_risk_v5_development/point_ablation",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--states-per-trajectory", type=int, default=128)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "models").mkdir(exist_ok=True)

    train = PackedEnergyDataset.load(SOURCE / "energy_dataset/train")
    validation = PackedEnergyDataset.load(SOURCE / "energy_dataset/validation")
    calibration = PackedEnergyDataset.concatenate(
        [
            PackedEnergyDataset.load(SOURCE / "energy_dataset/calibration"),
            PackedEnergyDataset.load(V2 / "calibration_v2/new_trajectories"),
        ]
    )
    diagnostic = PackedEnergyDataset.load(V2 / "final_conformal_test_v2/trajectories")
    train_indices = trajectory_balanced_state_indices(
        train.trajectory_ids,
        max_states_per_trajectory=args.states_per_trajectory,
    )
    validation_indices = trajectory_balanced_state_indices(
        validation.trajectory_ids,
        max_states_per_trajectory=args.states_per_trajectory,
    )
    results: list[dict[str, object]] = []

    original = EnergyToGoRegressor.load(POINT_CHECKPOINT, device=args.device)
    original_calibration_prediction = chunked_point_predictions(original, calibration.states)
    original_diagnostic_prediction = chunked_point_predictions(original, diagnostic.states)
    original_calibration = MondrianTrajectoryCalibration.fit(
        calibration.targets,
        original_calibration_prediction,
        np.zeros_like(original_calibration_prediction),
        calibration.trajectory_ids,
        calibration.goal_types,
        calibration.distance_buckets,
        coverage=0.95,
        mode="additive",
        minimum_group_trajectories=100,
    )
    original_upper = original_calibration.apply(
        original_diagnostic_prediction,
        np.zeros_like(original_diagnostic_prediction),
        diagnostic.goal_types,
        diagnostic.states[:, -1].astype(np.float64) * D_MAX,
    )
    original_metrics = _point_metrics(diagnostic, original_diagnostic_prediction)
    original_coverage = _trajectory_group_coverage(diagnostic, original_upper)
    results.append(
        {
            "method": "original_frozen_7d_point",
            "feature_mode": "state7",
            "retrained": False,
            "parameters": sum(parameter.numel() for parameter in original.model.parameters()),
            **original_metrics,
            "mondrian_overall_coverage": original_coverage["overall"],
            "mondrian_worst_group": original_coverage["worst_group"],
            "mondrian_worst_group_coverage": original_coverage["worst_group_coverage"],
            "mean_bound_width": float(np.mean(original_upper - original_diagnostic_prediction)),
        }
    )

    for mode_index, mode in enumerate(FEATURE_MODES):
        builder = GoalRiskFeatureBuilder(mode)
        train_features = _dataset_features(train, builder)[train_indices]
        validation_features = _dataset_features(validation, builder)[validation_indices]
        model = ContextualEnergyRegressor(
            input_dim=builder.input_dim,
            battery_capacity=BATTERY_CAPACITY,
            seed=810_100 + mode_index,
            device=args.device,
        )
        model.initialize_from_frozen_7d(original)
        history = model.fit(
            train_features,
            train.targets[train_indices],
            validation_features,
            validation.targets[validation_indices],
            max_epochs=args.max_epochs,
            patience=args.patience,
        )
        model.save(args.output_dir / f"models/point_{mode}.pt", feature_mode=mode)
        calibration_prediction = model.predict_batch(_dataset_features(calibration, builder))
        diagnostic_prediction = model.predict_batch(_dataset_features(diagnostic, builder))
        mondrian = MondrianTrajectoryCalibration.fit(
            calibration.targets,
            calibration_prediction,
            np.zeros_like(calibration_prediction),
            calibration.trajectory_ids,
            calibration.goal_types,
            calibration.distance_buckets,
            coverage=0.95,
            mode="additive",
            minimum_group_trajectories=100,
        )
        upper = mondrian.apply(
            diagnostic_prediction,
            np.zeros_like(diagnostic_prediction),
            diagnostic.goal_types,
            diagnostic.states[:, -1].astype(np.float64) * D_MAX,
        )
        metrics = _point_metrics(diagnostic, diagnostic_prediction)
        coverage = _trajectory_group_coverage(diagnostic, upper)
        results.append(
            {
                "method": f"retrained_{mode}",
                "feature_mode": mode,
                "retrained": True,
                "parameters": model.parameter_count(),
                **metrics,
                "mondrian_overall_coverage": coverage["overall"],
                "mondrian_worst_group": coverage["worst_group"],
                "mondrian_worst_group_coverage": coverage["worst_group_coverage"],
                "mean_bound_width": float(np.mean(upper - diagnostic_prediction)),
                "best_validation_mae": history["best_validation_mae"],
                "best_epoch": history["best_epoch"],
            }
        )
        (args.output_dir / f"training_history_{mode}.json").write_text(
            json.dumps(history, indent=2, sort_keys=True) + "\n"
        )
        del train_features, validation_features

    frame = pd.DataFrame(results)
    frame.to_csv(args.output_dir / "point_feature_ablation.csv", index=False)
    best = frame.sort_values(["mae", "trajectory_max_underestimation_p99"]).iloc[0]
    original_row = frame[frame.method == "original_frozen_7d_point"].iloc[0]
    relative_mae_improvement = float((original_row.mae - best.mae) / original_row.mae)
    conclusion = {
        "best_method": str(best.method),
        "best_mae": float(best.mae),
        "original_mae": float(original_row.mae),
        "relative_mae_improvement": relative_mae_improvement,
        "material_improvement_threshold": 0.10,
        "contextual_point_retraining_material": relative_mae_improvement >= 0.10,
        "recommendation": (
            "retrain_contextual_point"
            if relative_mae_improvement >= 0.10
            else "keep_original_7d_point_and_contextualize_risk_only"
        ),
    }
    (args.output_dir / "POINT_MODEL_DECISION.json").write_text(
        json.dumps(conclusion, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
