from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.energy_mc.adaptive_analysis import (
    contiguous_trajectory_slices,
    evaluate_mission_upper,
    mission_component_predictions,
    nearest_neighbor_aliasing_analysis,
)
from experiments.energy_mc.adaptive_uncertainty import (
    HeteroscedasticResidualModel,
    MCSupervisedQuantileModel,
    RISK_COVERAGE_LEVELS,
    TrajectoryConformalCorrection,
    chunked_point_predictions,
)
from experiments.energy_mc.conformal import PackedMissionDataset
from experiments.energy_mc.core import PackedEnergyDataset
from review_bundle.safety.energy.mc_regression import (
    EnergyToGoRegressor,
    MissionConformalCalibration,
)
from scripts.explore_adaptive_energy_uncertainty import (
    BATTERY_CAPACITY,
    mission_method_predictions,
    summarize_method_results,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts/uav_energy_delivery_mc_formal_20260817_154321"
V2 = ROOT / "artifacts/uav_energy_delivery_mc_conformal_v2_20260817_184046"


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def correlation(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    return {
        "pearson": float(pearsonr(left, right).statistic),
        "spearman": float(spearmanr(left, right).statistic),
    }


def state_residual_groups(
    dataset: PackedEnergyDataset,
    point: np.ndarray,
) -> pd.DataFrame:
    residual = dataset.targets.astype(np.float64) - point
    rows = []
    for goal_type in np.unique(dataset.goal_types):
        for bucket in np.unique(dataset.distance_buckets):
            selected = (dataset.goal_types == goal_type) & (dataset.distance_buckets == bucket)
            if not np.any(selected):
                continue
            values = residual[selected]
            rows.append(
                {
                    "goal_type": str(goal_type),
                    "distance_bucket": str(bucket),
                    "state_count": int(np.sum(selected)),
                    "mean_residual": float(np.mean(values)),
                    "std_residual": float(np.std(values)),
                    "mae": float(np.mean(np.abs(values))),
                    "p90_residual": float(np.quantile(values, 0.90)),
                    "p95_residual": float(np.quantile(values, 0.95)),
                    "p99_residual": float(np.quantile(values, 0.99)),
                    "maximum_residual": float(np.max(values)),
                }
            )
    return pd.DataFrame(rows)


def scale_diagnostics(
    dataset: PackedEnergyDataset,
    point: np.ndarray,
    model: HeteroscedasticResidualModel,
) -> dict[str, object]:
    residual = dataset.targets.astype(np.float64) - point
    scale = model.predict_scale(dataset.states)
    trajectory_scale = []
    trajectory_error = []
    for _, selected in contiguous_trajectory_slices(dataset.trajectory_ids):
        trajectory_scale.append(float(np.max(scale[selected])))
        trajectory_error.append(float(np.max(residual[selected])))
    return {
        "scale_vs_absolute_residual": correlation(scale, np.abs(residual)),
        "scale_vs_positive_underestimation": correlation(scale, np.maximum(residual, 0.0)),
        "trajectory_max_scale_vs_max_underestimation": correlation(
            np.asarray(trajectory_scale),
            np.asarray(trajectory_error),
        ),
        "scale_p05": float(np.quantile(scale, 0.05)),
        "scale_p50": float(np.quantile(scale, 0.50)),
        "scale_p95": float(np.quantile(scale, 0.95)),
    }


def mission_residual_groups(
    dataset: PackedMissionDataset,
    point: EnergyToGoRegressor,
) -> pd.DataFrame:
    _, _, prediction = mission_component_predictions(
        lambda states: chunked_point_predictions(point, states),
        dataset,
    )
    residual = dataset.true_mission_energy.astype(np.float64) - prediction
    rows = []
    for bucket in np.unique(dataset.initial_distance_buckets):
        selected = dataset.initial_distance_buckets == bucket
        values = residual[selected]
        rows.append(
            {
                "distance_bucket": str(bucket),
                "state_count": int(np.sum(selected)),
                "mean_residual": float(np.mean(values)),
                "std_residual": float(np.std(values)),
                "mae": float(np.mean(np.abs(values))),
                "p90_residual": float(np.quantile(values, 0.90)),
                "p95_residual": float(np.quantile(values, 0.95)),
                "p99_residual": float(np.quantile(values, 0.99)),
                "maximum_residual": float(np.max(values)),
            }
        )
    return pd.DataFrame(rows)


def plot_residual_tails(frame: pd.DataFrame, output: Path) -> None:
    task = frame[frame.goal_type == "TASK"]
    figure, axis = plt.subplots(figsize=(9, 5))
    x = np.arange(len(task))
    axis.bar(x - 0.25, task.mean_residual, width=0.25, label="mean")
    axis.bar(x, task.p95_residual, width=0.25, label="P95")
    axis.bar(x + 0.25, task.p99_residual, width=0.25, label="P99")
    axis.set_xticks(x, task.distance_bucket)
    axis.set_ylabel("true minus point prediction")
    axis.set_title("TASK residual tails on fresh test v3")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def run(run_dir: Path, *, aliasing_sample_size: int) -> None:
    output = run_dir / "posthoc_analysis"
    output.mkdir(exist_ok=True)
    point = EnergyToGoRegressor.load(SOURCE / "energy_model/best_validation.pt", device="cuda")
    datasets = {
        "diagnostic_v2": PackedEnergyDataset.load(V2 / "final_conformal_test_v2/trajectories"),
        "fresh_test_v3": PackedEnergyDataset.load(run_dir / "fresh_test_v3/goal_trajectories"),
    }
    decisions: dict[str, object] = {}
    for label, dataset in datasets.items():
        prediction = chunked_point_predictions(point, dataset.states)
        groups = state_residual_groups(dataset, prediction)
        groups.to_csv(output / f"{label}_state_residual_groups.csv", index=False)
        scale_results = {}
        for distribution in ("gaussian", "laplace"):
            model = HeteroscedasticResidualModel.load(
                run_dir / f"models/goal_heteroscedastic_{distribution}.pt",
                device="cuda",
            )
            scale_results[distribution] = scale_diagnostics(dataset, prediction, model)
        write_json(output / f"{label}_scale_diagnostics.json", scale_results)
        if label == "fresh_test_v3":
            plot_residual_tails(groups, output / "fresh_task_residual_tails.png")
            task = groups[groups.goal_type == "TASK"].set_index("distance_bucket")
            short = task.loc[["500-1500", "1500-2500"]]
            long = task.loc[["2500-4000", ">4000"]]
            mean_near_zero = bool(np.max(np.abs(long.mean_residual.to_numpy())) < 0.02)
            variance_ratio = float(long.std_residual.mean() / short.std_residual.mean())
            p99_ratio = float(long.p99_residual.mean() / short.p99_residual.mean())
            decisions["heteroscedastic_tail"] = {
                "mean_residual_near_zero": mean_near_zero,
                "long_to_mid_std_ratio": variance_ratio,
                "long_to_mid_p99_ratio": p99_ratio,
                "HETEROSCEDASTIC_TAIL_RISK_CONFIRMED": bool(
                    mean_near_zero and variance_ratio > 1.4 and p99_ratio > 1.4
                ),
            }

    aliasing, context = nearest_neighbor_aliasing_analysis(
        datasets["fresh_test_v3"],
        sample_size=aliasing_sample_size,
        seed=710_001,
    )
    write_json(output / "fresh_test_v3_aliasing_summary.json", aliasing)
    context.to_csv(output / "fresh_test_v3_aliasing_context.csv", index=False)
    fresh_point_mae = float(
        np.mean(
            np.abs(
                datasets["fresh_test_v3"].targets
                - chunked_point_predictions(point, datasets["fresh_test_v3"].states)
            )
        )
    )
    decisions["state_aliasing"] = {
        "point_mae": fresh_point_mae,
        **aliasing,
        "empirical_near_neighbor_information_loss": bool(
            aliasing["p95_return_difference_closest_one_percent"] > 0.5 * fresh_point_mae
            and aliasing["maximum_return_difference_closest_one_percent"] > 2.0 * fresh_point_mae
        ),
        "structural_markov_sufficiency": False,
        "structural_reason": "absolute position affects boundary projection but is absent from 7D state",
    }

    mission = PackedMissionDataset.load(run_dir / "fresh_test_v3/mission_trajectories")
    mission_residual_groups(mission, point).to_csv(
        output / "fresh_test_v3_mission_residual_groups.csv",
        index=False,
    )
    calibration_payload = json.loads(
        (run_dir / "calibration/mission_calibrations.json").read_text(encoding="utf-8")
    )
    mission_calibrations: dict[str, dict[float, object]] = {}
    for method, levels in calibration_payload.items():
        mission_calibrations[method] = {}
        for level, payload in levels.items():
            if method == "mission_point_distance_group":
                value = MissionConformalCalibration.from_dict(payload)
            else:
                value = TrajectoryConformalCorrection(**payload)
            mission_calibrations[method][float(level)] = value
    mission_heteroscedastic = HeteroscedasticResidualModel.load(
        run_dir / "models/mission_heteroscedastic_laplace.pt",
        device="cuda",
    )
    mission_quantile = MCSupervisedQuantileModel.load(
        run_dir / "models/mission_direct_quantiles.pt",
        device="cuda",
    )
    diagnostic_mission = PackedMissionDataset.load(V2 / "mission_test_v2/trajectories")
    rng = np.random.default_rng(610_101)
    rng.uniform(0.0, BATTERY_CAPACITY, size=diagnostic_mission.true_mission_energy.shape)
    remaining = rng.uniform(0.0, BATTERY_CAPACITY, size=mission.true_mission_energy.shape)
    refreshed_results: dict[str, dict[float, dict[str, object]]] = {}
    for coverage in RISK_COVERAGE_LEVELS:
        predictions = mission_method_predictions(
            point,
            mission_heteroscedastic,
            mission_quantile,
            mission_calibrations,
            mission,
            coverage,
        )
        for method, (center, upper) in predictions.items():
            refreshed_results.setdefault(method, {})[coverage] = evaluate_mission_upper(
                mission,
                center,
                upper,
                battery_capacity=BATTERY_CAPACITY,
                reserve_fraction=0.10,
                remaining_energy_samples=remaining,
            )
    write_json(output / "fresh_mission_tradeoff_results.json", refreshed_results)
    mission_matrix = summarize_method_results({}, refreshed_results)
    mission_matrix.to_csv(output / "fresh_mission_tradeoff_matrix.csv", index=False)
    trajectory = pd.read_csv(run_dir / "diagnostics/trajectory_residuals.csv")
    features = [
        "trajectory_total_length",
        "path_ratio",
        "flight_steps",
        "mean_speed",
        "max_speed",
        "mean_acceleration",
        "max_acceleration",
        "vertical_displacement",
        "boundary_contact_duration_steps",
        "start_distance_to_boundary",
        "minimum_distance_to_boundary",
        "mean_residual",
        "std_residual",
        "p95_residual",
        "p99_residual",
        "trajectory_max_underestimation",
    ]
    trajectory[trajectory.goal_type == "TASK"].groupby("distance_bucket")[features].agg(
        ["mean", "std"]
    ).to_csv(output / "task_distance_feature_comparison.csv")
    write_json(output / "diagnostic_decisions.json", decisions)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--aliasing-sample-size", type=int, default=100_000)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(arguments.run_dir, aliasing_sample_size=arguments.aliasing_sample_size)
