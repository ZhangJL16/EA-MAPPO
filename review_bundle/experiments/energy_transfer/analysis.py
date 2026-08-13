from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METHOD_LABELS = {
    "B0_distance": "B0 distance",
    "B1_monte_carlo": "B1 Monte-Carlo",
    "B2_scalar_td": "B2 scalar TD",
    "B3_distributional_median": "B3 quantile median",
}


def regression_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    predicted = np.asarray(prediction, dtype=np.float64)
    observed = np.asarray(target, dtype=np.float64)
    if predicted.shape != observed.shape or predicted.ndim != 1 or predicted.size == 0:
        raise ValueError("prediction and target must be aligned nonempty vectors")
    error = predicted - observed
    under = error < 0.0
    under_magnitude = -error[under]
    denominator = np.maximum(np.abs(observed), 1e-6)
    return {
        "sample_count": int(observed.size),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "median_absolute_error": float(np.median(np.abs(error))),
        "mean_signed_error": float(np.mean(error)),
        "underestimation_rate": float(np.mean(under)),
        "mean_underestimation_magnitude": float(np.mean(under_magnitude)) if under_magnitude.size else 0.0,
        "worst_case_underestimation": float(np.max(under_magnitude)) if under_magnitude.size else 0.0,
        "mean_absolute_relative_error": float(np.mean(np.abs(error) / denominator)),
        "median_absolute_relative_error": float(np.median(np.abs(error) / denominator)),
    }


def quantile_metrics(
    predictions: np.ndarray,
    target: np.ndarray,
    levels: tuple[float, ...],
) -> dict[str, Any]:
    predicted = np.asarray(predictions, dtype=np.float64)
    observed = np.asarray(target, dtype=np.float64)
    if predicted.shape != (observed.size, len(levels)):
        raise ValueError("quantile predictions must align with outcomes and levels")
    crossing = np.diff(predicted, axis=1) < -1e-10
    coverage: dict[str, float] = {}
    under_coverage: dict[str, float] = {}
    upper_tail: dict[str, dict[str, float]] = {}
    for column, level in enumerate(levels):
        values = predicted[:, column]
        empirical = float(np.mean(observed <= values))
        coverage[str(level)] = empirical
        under_coverage[str(level)] = max(0.0, float(level) - empirical)
        residual = observed - values
        pinball = np.maximum(level * residual, (level - 1.0) * residual)
        threshold = float(np.quantile(observed, level))
        mask = observed >= threshold
        upper_tail[str(level)] = {
            "actual_tail_threshold": threshold,
            "tail_sample_count": int(mask.sum()),
            "mae_on_actual_upper_tail": float(np.mean(np.abs(values[mask] - observed[mask]))),
            "mean_signed_error_on_actual_upper_tail": float(np.mean(values[mask] - observed[mask])),
            "mean_pinball_loss_all_test": float(np.mean(pinball)),
        }
    return {
        "empirical_coverage": coverage,
        "under_coverage": under_coverage,
        "quantile_crossing_count": int(crossing.sum()),
        "quantile_crossing_rate": float(np.mean(crossing)) if crossing.size else 0.0,
        "quantile_crossing_occurred": bool(crossing.any()),
        "upper_tail_prediction_error": upper_tail,
    }


def safe_correlation(first: np.ndarray, second: np.ndarray) -> float | None:
    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    if left.shape != right.shape or left.size < 2:
        return None
    if np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def residual_relationships(
    residual: np.ndarray,
    variables: dict[str, np.ndarray],
) -> dict[str, dict[str, Any]]:
    error = np.asarray(residual, dtype=np.float64)
    report: dict[str, dict[str, Any]] = {}
    for name, raw_values in variables.items():
        values = np.asarray(raw_values, dtype=np.float64)
        if values.shape != error.shape:
            raise ValueError(f"residual variable {name} is not aligned")
        correlation = safe_correlation(error, values)
        if np.std(values) > 1e-12:
            slope = float(np.polyfit(values, error, 1)[0])
        else:
            slope = 0.0
        edges = np.unique(np.quantile(values, np.linspace(0.0, 1.0, 6)))
        bins: list[dict[str, float | int]] = []
        if edges.size >= 2:
            for index in range(edges.size - 1):
                lower = edges[index]
                upper = edges[index + 1]
                mask = (values >= lower) & (values <= upper if index == edges.size - 2 else values < upper)
                if mask.any():
                    bins.append(
                        {
                            "lower": float(lower),
                            "upper": float(upper),
                            "count": int(mask.sum()),
                            "mean_signed_error": float(np.mean(error[mask])),
                            "mae": float(np.mean(np.abs(error[mask]))),
                        }
                    )
        report[name] = {
            "pearson_correlation_with_signed_error": correlation,
            "linear_slope": slope,
            "quantile_bins": bins,
        }
    return report


def horizon_error_report(
    prediction: np.ndarray,
    target: np.ndarray,
    horizons: np.ndarray,
) -> dict[str, Any]:
    predicted = np.asarray(prediction, dtype=np.float64)
    observed = np.asarray(target, dtype=np.float64)
    steps = np.asarray(horizons, dtype=np.int64)
    if predicted.shape != observed.shape or predicted.shape != steps.shape:
        raise ValueError("prediction, target, and horizons must align")
    bins = (
        ("1", steps == 1),
        ("2", steps == 2),
        ("3", steps == 3),
        ("4-5", (steps >= 4) & (steps <= 5)),
        ("6-10", (steps >= 6) & (steps <= 10)),
        ("11-20", (steps >= 11) & (steps <= 20)),
        ("21-40", (steps >= 21) & (steps <= 40)),
        (">40", steps > 40),
    )
    report: dict[str, dict[str, float | int]] = {}
    for label, mask in bins:
        if not np.any(mask):
            continue
        error = predicted[mask] - observed[mask]
        report[label] = {
            "sample_count": int(mask.sum()),
            "mae": float(np.mean(np.abs(error))),
            "mean_signed_error": float(np.mean(error)),
            "mean_prediction": float(np.mean(predicted[mask])),
            "mean_target": float(np.mean(observed[mask])),
        }
    return {
        "bins": report,
        "signed_error_vs_horizon_correlation": safe_correlation(predicted - observed, steps),
        "absolute_error_vs_horizon_correlation": safe_correlation(np.abs(predicted - observed), steps),
    }


def paired_sortie_comparisons(
    predictions: dict[str, np.ndarray],
    target: np.ndarray,
    sortie_indices: np.ndarray,
    *,
    seed: int,
    bootstrap_samples: int = 5000,
) -> dict[str, dict[str, float | int]]:
    observed = np.asarray(target, dtype=np.float64)
    sorties = np.asarray(sortie_indices, dtype=np.int64)
    baseline = np.asarray(predictions["B0_distance"], dtype=np.float64)
    unique_sorties = np.unique(sorties)
    baseline_mae = np.asarray(
        [np.mean(np.abs(baseline[sorties == sortie] - observed[sorties == sortie])) for sortie in unique_sorties]
    )
    rng = np.random.default_rng(seed)
    report: dict[str, dict[str, float | int]] = {}
    for name, raw_prediction in predictions.items():
        if name == "B0_distance":
            continue
        prediction = np.asarray(raw_prediction, dtype=np.float64)
        learned_mae = np.asarray(
            [np.mean(np.abs(prediction[sorties == sortie] - observed[sorties == sortie])) for sortie in unique_sorties]
        )
        delta = learned_mae - baseline_mae
        sampled = rng.integers(0, unique_sorties.size, size=(bootstrap_samples, unique_sorties.size))
        bootstrap_means = delta[sampled].mean(axis=1)
        mean_delta = float(np.mean(delta))
        report[name] = {
            "test_sortie_count": int(unique_sorties.size),
            "mean_sortie_mae_delta_vs_B0": mean_delta,
            "bootstrap_95_ci_lower": float(np.quantile(bootstrap_means, 0.025)),
            "bootstrap_95_ci_upper": float(np.quantile(bootstrap_means, 0.975)),
            "fraction_test_sorties_better_than_B0": float(np.mean(delta < 0.0)),
            "relative_transition_mae_improvement_vs_B0": float(
                (np.mean(np.abs(baseline - observed)) - np.mean(np.abs(prediction - observed)))
                / max(np.mean(np.abs(baseline - observed)), 1e-12)
            ),
        }
    return report


def learned_value_decision(comparisons: dict[str, dict[str, float | int]]) -> dict[str, Any]:
    best_name = min(comparisons, key=lambda name: float(comparisons[name]["mean_sortie_mae_delta_vs_B0"]))
    best = comparisons[best_name]
    improvement = float(best["relative_transition_mae_improvement_vs_B0"])
    upper = float(best["bootstrap_95_ci_upper"])
    if improvement >= 0.05 and upper < 0.0:
        decision = "TRUE"
        reason = "best learned method improves transition MAE by at least 5% and its paired-sortie bootstrap CI excludes zero"
    elif float(best["mean_sortie_mae_delta_vs_B0"]) >= 0.0:
        decision = "FALSE"
        reason = "even the best learned method has no lower mean sortie MAE than B0"
    else:
        decision = "UNCLEAR"
        reason = "point estimate favors a learned method but the paired-sortie evidence is not decisive"
    return {"LEARNED_MODEL_ADDS_VALUE": decision, "best_learned_method": best_name, "reason": reason}


def coverage_report(
    return_start_positions: np.ndarray,
    return_start_velocities: np.ndarray,
    charger_distances: np.ndarray,
    prefix_lengths: np.ndarray,
    world_size: np.ndarray,
    *,
    grid_size: int = 8,
) -> dict[str, Any]:
    positions = np.asarray(return_start_positions, dtype=np.float64)
    velocities = np.asarray(return_start_velocities, dtype=np.float64)
    distances = np.asarray(charger_distances, dtype=np.float64)
    prefixes = np.asarray(prefix_lengths, dtype=np.float64)
    xy_bins = np.floor(positions[:, :2] / world_size[:2] * grid_size).astype(int)
    xy_bins = np.clip(xy_bins, 0, grid_size - 1)
    occupied = len({(int(x), int(y)) for x, y in xy_bins})
    speed = np.linalg.norm(velocities, axis=1)
    distance_span = float(np.ptp(distances))
    prefix_span = float(np.ptp(prefixes))
    occupancy = occupied / float(grid_size * grid_size)
    sufficient = bool(
        occupancy >= 0.50
        and distance_span >= 2.5
        and prefix_span >= 100.0
        and float(np.quantile(speed, 0.90)) >= 0.05
        and float(np.std(speed)) >= 0.01
    )
    return {
        "DATA_COVERAGE_SUFFICIENT": sufficient,
        "xy_grid_size": grid_size,
        "xy_occupied_cells": occupied,
        "xy_occupancy_fraction": occupancy,
        "position_min": positions.min(axis=0).tolist(),
        "position_max": positions.max(axis=0).tolist(),
        "charger_distance_min": float(distances.min()),
        "charger_distance_max": float(distances.max()),
        "charger_distance_span": distance_span,
        "prefix_min": int(prefixes.min()),
        "prefix_max": int(prefixes.max()),
        "prefix_span": prefix_span,
        "return_start_speed_mean": float(speed.mean()),
        "return_start_speed_std": float(speed.std()),
        "return_start_speed_p90": float(np.quantile(speed, 0.90)),
        "criterion": "8x8 XY occupancy >= 0.50, distance span >= 2.5 m, prefix span >= 100 steps, speed p90 >= 0.05 m/s, speed std >= 0.01 m/s",
    }


def build_figures(
    output_dir: str | Path,
    *,
    predictions: dict[str, np.ndarray],
    quantile_predictions: np.ndarray,
    quantile_levels: tuple[float, ...],
    target: np.ndarray,
    distances: np.ndarray,
    path_lengths: np.ndarray,
    return_start_positions: np.ndarray,
    world_size: np.ndarray,
) -> dict[str, str]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    observed = np.asarray(target)
    figure_paths: dict[str, str] = {}

    figure, axes = plt.subplots(2, 2, figsize=(10, 9), constrained_layout=True)
    lower = float(min(observed.min(), *(values.min() for values in predictions.values())))
    upper = float(max(observed.max(), *(values.max() for values in predictions.values())))
    for axis, (name, predicted) in zip(axes.ravel(), predictions.items(), strict=True):
        axis.scatter(observed, predicted, s=6, alpha=0.25)
        axis.plot([lower, upper], [lower, upper], "k--", linewidth=1)
        axis.set_title(METHOD_LABELS[name])
        axis.set_xlabel("actual return energy")
        axis.set_ylabel("predicted return energy")
    figure_paths["predicted_vs_actual"] = _save(figure, output / "predicted_vs_actual.png")

    figure, axis = plt.subplots(figsize=(7, 5), constrained_layout=True)
    axis.scatter(distances, observed, s=7, alpha=0.3, label="Euclidean distance")
    axis.scatter(path_lengths, observed, s=7, alpha=0.2, label="actual path-to-go")
    axis.set_xlabel("distance / path length (m)")
    axis.set_ylabel("actual return energy")
    axis.legend()
    figure_paths["energy_vs_distance"] = _save(figure, output / "return_energy_vs_distance.png")

    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    errors = [np.abs(predictions[name] - observed) for name in METHOD_LABELS]
    axis.boxplot(errors, labels=[METHOD_LABELS[name] for name in METHOD_LABELS], showfliers=False)
    axis.set_ylabel("absolute error")
    axis.tick_params(axis="x", rotation=18)
    figure_paths["absolute_error_comparison"] = _save(figure, output / "absolute_error_comparison.png")

    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    order = np.argsort(distances)
    chunks = np.array_split(order, 10)
    for name, predicted in predictions.items():
        centers = [float(np.mean(distances[chunk])) for chunk in chunks if chunk.size]
        means = [float(np.mean(np.abs(predicted[chunk] - observed[chunk]))) for chunk in chunks if chunk.size]
        axis.plot(centers, means, marker="o", label=METHOD_LABELS[name])
    axis.set_xlabel("charger distance (m)")
    axis.set_ylabel("mean absolute error")
    axis.legend()
    figure_paths["error_vs_distance"] = _save(figure, output / "error_vs_charger_distance.png")

    coverage = [float(np.mean(observed <= quantile_predictions[:, column])) for column in range(len(quantile_levels))]
    figure, axis = plt.subplots(figsize=(6, 5), constrained_layout=True)
    axis.plot(quantile_levels, quantile_levels, "k--", label="ideal")
    axis.plot(quantile_levels, coverage, marker="o", label="empirical")
    axis.set_xlim(0.45, 1.0)
    axis.set_ylim(0.45, 1.02)
    axis.set_xlabel("nominal quantile")
    axis.set_ylabel("empirical coverage")
    axis.legend()
    figure_paths["quantile_coverage"] = _save(figure, output / "quantile_coverage.png")

    figure, axis = plt.subplots(figsize=(6, 6), constrained_layout=True)
    histogram = axis.hist2d(
        return_start_positions[:, 0],
        return_start_positions[:, 1],
        bins=8,
        range=[[0.0, world_size[0]], [0.0, world_size[1]]],
        cmap="viridis",
    )
    figure.colorbar(histogram[3], ax=axis, label="return starts")
    axis.set_xlabel("x (m)")
    axis.set_ylabel("y (m)")
    axis.set_title("Return-start spatial coverage")
    figure_paths["return_start_heatmap"] = _save(figure, output / "return_start_heatmap.png")
    return figure_paths


def _save(figure: plt.Figure, path: Path) -> str:
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return str(path)
