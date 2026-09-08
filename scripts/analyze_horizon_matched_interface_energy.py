#!/usr/bin/env python3
"""Horizon-matched falsification audit for R3 executed-interface energy signal.

This consumes completed frozen-policy trajectories. It does not train a policy,
unlock a formal Gate, or create a policy/filter composition comparison.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np


RIDGE_GRID = (0.0, 1e-4, 1e-2, 1.0, 100.0)
FEATURE_SETS = (
    "horizon",
    "strong_confounder",
    "strong_plus_action_delta",
    "strong_plus_burden_rate",
    "strong_plus_interface",
)


def _safe_log(values: np.ndarray) -> np.ndarray:
    return np.log1p(np.maximum(np.asarray(values, dtype=np.float64), 0.0))


def _one_hot(values: np.ndarray) -> tuple[np.ndarray, list[str]]:
    labels = sorted({str(value) for value in values.tolist()})
    matrix = np.column_stack(
        [np.asarray([str(value) == label for value in values], dtype=np.float64) for label in labels]
    )
    return matrix, labels


def load_trajectory_features(trajectory_dir: Path) -> dict[str, np.ndarray]:
    manifest_path = trajectory_dir / "manifest.jsonl"
    records = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError(f"no trajectories in {manifest_path}")

    columns: dict[str, list[float] | list[str]] = {
        "trajectory_id": [],
        "total_energy": [],
        "horizon": [],
        "total_duration": [],
        "initial_distance": [],
        "path_length": [],
        "path_ratio": [],
        "goal_type": [],
        "distance_bucket": [],
        "nominal_norm_mean": [],
        "nominal_norm_q90": [],
        "nominal_delta_rms": [],
        "intervention_mean": [],
        "intervention_q90": [],
        "intervention_max": [],
        "intervention_rms": [],
        "intervention_nonzero_fraction": [],
        "safety_burden_rate": [],
        "energy_per_step": [],
    }
    max_energy_sum_error = 0.0
    for item in records:
        path = trajectory_dir / str(item["file"])
        with np.load(path, allow_pickle=False) as payload:
            nominal = np.asarray(payload["nominal_actions"], dtype=np.float64)
            executed = np.asarray(payload["executed_actions"], dtype=np.float64)
            positions = np.asarray(payload["positions"], dtype=np.float64)
            step_energy = np.asarray(payload["step_energy"], dtype=np.float64)
            transition_dt = np.asarray(payload["transition_dt"], dtype=np.float64)
        horizon = int(item["num_transitions"])
        if not (
            nominal.shape == executed.shape == positions.shape == (horizon, 3)
            and step_energy.shape == transition_dt.shape == (horizon,)
        ):
            raise ValueError(f"trajectory contract mismatch: {path}")
        if not np.all(
            np.isfinite(
                np.concatenate(
                    [
                        nominal.ravel(),
                        executed.ravel(),
                        positions.ravel(),
                        step_energy,
                        transition_dt,
                    ]
                )
            )
        ):
            raise ValueError(f"nonfinite trajectory payload: {path}")
        total_energy = float(item["total_realized_energy"])
        max_energy_sum_error = max(
            max_energy_sum_error,
            abs(float(np.sum(step_energy)) - total_energy),
        )
        intervention = np.linalg.norm(executed - nominal, axis=1)
        nominal_norm = np.linalg.norm(nominal, axis=1)
        nominal_delta = np.diff(nominal, axis=0)
        path_length = float(
            np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
        )
        initial_distance = float(item["initial_distance"])

        columns["trajectory_id"].append(float(item["trajectory_id"]))
        columns["total_energy"].append(total_energy)
        columns["horizon"].append(float(horizon))
        columns["total_duration"].append(float(np.sum(transition_dt)))
        columns["initial_distance"].append(initial_distance)
        columns["path_length"].append(path_length)
        columns["path_ratio"].append(
            path_length / max(initial_distance, np.finfo(np.float64).eps)
        )
        columns["goal_type"].append(str(item["goal_type"]))
        columns["distance_bucket"].append(str(item["distance_bucket"]))
        columns["nominal_norm_mean"].append(float(np.mean(nominal_norm)))
        columns["nominal_norm_q90"].append(float(np.quantile(nominal_norm, 0.90)))
        columns["nominal_delta_rms"].append(
            0.0
            if nominal_delta.size == 0
            else float(np.sqrt(np.mean(np.square(nominal_delta))))
        )
        columns["intervention_mean"].append(float(np.mean(intervention)))
        columns["intervention_q90"].append(float(np.quantile(intervention, 0.90)))
        columns["intervention_max"].append(float(np.max(intervention)))
        columns["intervention_rms"].append(
            float(np.sqrt(np.mean(np.square(intervention))))
        )
        columns["intervention_nonzero_fraction"].append(
            float(np.mean(intervention > 1e-8))
        )
        columns["safety_burden_rate"].append(
            float(item["total_safety_burden"]) / horizon
        )
        columns["energy_per_step"].append(total_energy / horizon)

    result: dict[str, np.ndarray] = {}
    for key, values in columns.items():
        if key in {"goal_type", "distance_bucket"}:
            result[key] = np.asarray(values, dtype=str)
        else:
            result[key] = np.asarray(values, dtype=np.float64)
    result["maximum_manifest_energy_sum_error"] = np.asarray(
        [max_energy_sum_error], dtype=np.float64
    )
    if np.unique(result["trajectory_id"]).size != len(records):
        raise ValueError("trajectory ids must be unique")
    return result


def build_feature_matrices(data: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, list[str]]]:
    horizon = _safe_log(data["horizon"])
    distance = _safe_log(data["initial_distance"])
    goal_hot, goal_labels = _one_hot(data["goal_type"])
    bucket_hot, bucket_labels = _one_hot(data["distance_bucket"])
    horizon_matrix = np.column_stack(
        [
            horizon,
            np.square(horizon),
            np.power(horizon, 3),
            distance,
            np.square(distance),
            horizon * distance,
            goal_hot,
            bucket_hot,
        ]
    )
    horizon_names = [
        "log_horizon",
        "log_horizon_sq",
        "log_horizon_cube",
        "log_initial_distance",
        "log_initial_distance_sq",
        "log_horizon_x_distance",
        *[f"goal={label}" for label in goal_labels],
        *[f"distance_bucket={label}" for label in bucket_labels],
    ]
    strong_extra_names = [
        "log_total_duration",
        "log_path_length",
        "log_path_ratio",
        "nominal_norm_mean",
        "nominal_norm_q90",
        "nominal_delta_rms",
    ]
    strong_extra = np.column_stack(
        [
            _safe_log(data["total_duration"]),
            _safe_log(data["path_length"]),
            _safe_log(data["path_ratio"]),
            data["nominal_norm_mean"],
            data["nominal_norm_q90"],
            data["nominal_delta_rms"],
        ]
    )
    action_delta_names = [
        "intervention_mean",
        "intervention_q90",
        "intervention_max",
        "intervention_rms",
        "intervention_nonzero_fraction",
    ]
    action_delta = np.column_stack([data[name] for name in action_delta_names])
    burden = data["safety_burden_rate"][:, None]
    interface_names = action_delta_names + ["safety_burden_rate"]
    interface = np.column_stack([action_delta, burden])
    matrices = {
        "horizon": horizon_matrix,
        "strong_confounder": np.column_stack([horizon_matrix, strong_extra]),
        "strong_plus_action_delta": np.column_stack(
            [horizon_matrix, strong_extra, action_delta]
        ),
        "strong_plus_burden_rate": np.column_stack(
            [horizon_matrix, strong_extra, burden]
        ),
        "strong_plus_interface": np.column_stack(
            [horizon_matrix, strong_extra, interface]
        ),
    }
    names = {
        "horizon": horizon_names,
        "strong_confounder": horizon_names + strong_extra_names,
        "strong_plus_action_delta": horizon_names
        + strong_extra_names
        + action_delta_names,
        "strong_plus_burden_rate": horizon_names
        + strong_extra_names
        + ["safety_burden_rate"],
        "strong_plus_interface": horizon_names + strong_extra_names + interface_names,
    }
    for key, matrix in matrices.items():
        if not np.all(np.isfinite(matrix)):
            raise ValueError(f"nonfinite design matrix: {key}")
    return matrices, names


def _outer_folds(num_rows: int, num_folds: int, scheme: str) -> list[np.ndarray]:
    indices = np.arange(num_rows, dtype=np.int64)
    if scheme == "contiguous":
        return [part for part in np.array_split(indices, num_folds) if part.size]
    if scheme == "interleaved":
        return [indices[indices % num_folds == fold] for fold in range(num_folds)]
    raise ValueError(f"unknown fold scheme: {scheme}")


def _standardize(
    train: np.ndarray,
    test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(train, axis=0)
    scale = np.std(train, axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    return (train - mean) / scale, (test - mean) / scale


def _ridge_fit_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    alpha: float,
) -> np.ndarray:
    train_design = np.column_stack([np.ones(train_x.shape[0]), train_x])
    test_design = np.column_stack([np.ones(test_x.shape[0]), test_x])
    penalty = np.eye(train_design.shape[1], dtype=np.float64) * float(alpha)
    penalty[0, 0] = 0.0
    coefficient = np.linalg.pinv(train_design.T @ train_design + penalty) @ (
        train_design.T @ train_y
    )
    return test_design @ coefficient


def _select_ridge_alpha(train_x: np.ndarray, train_y: np.ndarray) -> float:
    folds = _outer_folds(train_x.shape[0], min(4, train_x.shape[0]), "contiguous")
    losses: list[tuple[float, float]] = []
    all_indices = np.arange(train_x.shape[0])
    for alpha in RIDGE_GRID:
        squared_errors: list[np.ndarray] = []
        for validation in folds:
            mask = np.ones(train_x.shape[0], dtype=bool)
            mask[validation] = False
            fitting = all_indices[mask]
            if fitting.size == 0:
                continue
            prediction = _ridge_fit_predict(
                train_x[fitting], train_y[fitting], train_x[validation], alpha
            )
            squared_errors.append(np.square(train_y[validation] - prediction))
        losses.append((float(np.mean(np.concatenate(squared_errors))), alpha))
    return min(losses, key=lambda item: (item[0], item[1]))[1]


def cross_fitted_predictions(
    design: np.ndarray,
    target: np.ndarray,
    *,
    scheme: str,
    num_folds: int = 5,
) -> tuple[np.ndarray, list[float]]:
    matrix = np.asarray(design, dtype=np.float64)
    values = np.asarray(target, dtype=np.float64)
    if matrix.ndim != 2 or values.shape != (matrix.shape[0],):
        raise ValueError("cross-fitting inputs must align")
    predictions = np.full(values.shape, np.nan, dtype=np.float64)
    selected: list[float] = []
    indices = np.arange(values.size)
    for test_indices in _outer_folds(values.size, num_folds, scheme):
        mask = np.ones(values.size, dtype=bool)
        mask[test_indices] = False
        train_indices = indices[mask]
        train_x, test_x = _standardize(
            matrix[train_indices], matrix[test_indices]
        )
        alpha = _select_ridge_alpha(train_x, values[train_indices])
        predictions[test_indices] = _ridge_fit_predict(
            train_x,
            values[train_indices],
            test_x,
            alpha,
        )
        selected.append(float(alpha))
    if not np.all(np.isfinite(predictions)):
        raise ValueError("cross-fitted predictions are incomplete")
    return predictions, selected


def _regression_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    residual = target - prediction
    sse = float(np.sum(np.square(residual)))
    sst = float(np.sum(np.square(target - np.mean(target))))
    return {
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "r_squared": float(1.0 - sse / max(sst, np.finfo(np.float64).eps)),
    }


def _moving_block_bootstrap_mean_interval(
    values: np.ndarray,
    *,
    block_length: int,
    repetitions: int,
    seed: int,
) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < 2 or repetitions <= 0:
        raise ValueError("invalid block-bootstrap inputs")
    length = min(max(1, int(block_length)), array.size)
    blocks_needed = int(math.ceil(array.size / length))
    rng = np.random.default_rng(seed)
    means = np.empty(repetitions, dtype=np.float64)
    offsets = np.arange(length)
    for index in range(repetitions):
        starts = rng.integers(0, array.size, size=blocks_needed)
        sample_indices = ((starts[:, None] + offsets[None, :]) % array.size).ravel()[
            : array.size
        ]
        means[index] = float(np.mean(array[sample_indices]))
    return {
        "estimate": float(np.mean(array)),
        "ci95_lower": float(np.quantile(means, 0.025)),
        "ci95_upper": float(np.quantile(means, 0.975)),
        "block_length": int(length),
        "repetitions": int(repetitions),
    }


def _rank_bins(values: np.ndarray, num_bins: int) -> np.ndarray:
    order = np.argsort(np.asarray(values, dtype=np.float64), kind="mergesort")
    ranks = np.empty(order.size, dtype=np.int64)
    ranks[order] = np.arange(order.size)
    return np.minimum(num_bins - 1, ranks * num_bins // order.size)


def _weighted_cell_bootstrap(
    differences: np.ndarray,
    weights: np.ndarray,
    *,
    repetitions: int,
    seed: int,
) -> dict[str, float]:
    if differences.size == 0:
        return {
            "estimate": None,
            "ci95_lower": None,
            "ci95_upper": None,
        }
    estimate = float(np.average(differences, weights=weights))
    rng = np.random.default_rng(seed)
    draws = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        selected = rng.integers(0, differences.size, size=differences.size)
        draws[index] = float(
            np.average(differences[selected], weights=weights[selected])
        )
    return {
        "estimate": estimate,
        "ci95_lower": float(np.quantile(draws, 0.025)),
        "ci95_upper": float(np.quantile(draws, 0.975)),
    }


def coarsened_horizon_match(
    data: dict[str, np.ndarray],
    *,
    base_residual: np.ndarray,
    repetitions: int,
    seed: int,
) -> dict[str, object]:
    horizon_bin = _rank_bins(data["horizon"], 5)
    distance_bin = _rank_bins(data["initial_distance"], 3)
    cells: dict[tuple[int, int, str], list[int]] = {}
    for index, key in enumerate(
        zip(horizon_bin, distance_bin, data["goal_type"], strict=True)
    ):
        cells.setdefault((int(key[0]), int(key[1]), str(key[2])), []).append(index)

    residual_differences: list[float] = []
    rate_differences: list[float] = []
    intervention_differences: list[float] = []
    weights: list[float] = []
    cell_details: list[dict[str, object]] = []
    matched_trajectories = 0
    for indices_list in cells.values():
        indices = np.asarray(indices_list, dtype=np.int64)
        if indices.size < 12:
            continue
        ordered = indices[np.argsort(data["intervention_mean"][indices])]
        group_size = indices.size // 3
        low = ordered[:group_size]
        high = ordered[-group_size:]
        intervention_difference = float(
            np.mean(data["intervention_mean"][high])
            - np.mean(data["intervention_mean"][low])
        )
        if intervention_difference <= 1e-10:
            continue
        residual_differences.append(
            float(np.mean(base_residual[high]) - np.mean(base_residual[low]))
        )
        rate_differences.append(
            float(
                np.mean(data["energy_per_step"][high])
                - np.mean(data["energy_per_step"][low])
            )
        )
        intervention_differences.append(intervention_difference)
        weights.append(float(group_size))
        matched_trajectories += 2 * group_size
        cell_details.append(
            {
                "horizon_bin": int(horizon_bin[indices[0]]),
                "distance_bin": int(distance_bin[indices[0]]),
                "goal_type": str(data["goal_type"][indices[0]]),
                "cell_size": int(indices.size),
                "matched_per_group": int(group_size),
                "mean_horizon_low": float(np.mean(data["horizon"][low])),
                "mean_horizon_high": float(np.mean(data["horizon"][high])),
                "intervention_high_minus_low": intervention_difference,
                "strong_residual_high_minus_low": residual_differences[-1],
                "energy_rate_high_minus_low": rate_differences[-1],
            }
        )

    diff = np.asarray(residual_differences, dtype=np.float64)
    rate = np.asarray(rate_differences, dtype=np.float64)
    weight = np.asarray(weights, dtype=np.float64)
    return {
        "cell_definition": "horizon quintile x initial-distance tercile x goal type",
        "minimum_cell_size": 12,
        "retained_cells": int(diff.size),
        "matched_trajectories": int(matched_trajectories),
        "mean_high_minus_low_intervention": (
            None
            if diff.size == 0
            else float(np.average(intervention_differences, weights=weight))
        ),
        "strong_base_residual_high_minus_low": _weighted_cell_bootstrap(
            diff, weight, repetitions=repetitions, seed=seed
        ),
        "energy_rate_high_minus_low": _weighted_cell_bootstrap(
            rate, weight, repetitions=repetitions, seed=seed + 1
        ),
        "cell_details": cell_details,
    }


def _normalized_weights(log_weights: np.ndarray) -> np.ndarray:
    shifted = log_weights - float(np.max(log_weights))
    weights = np.exp(shifted)
    return weights / np.sum(weights)


def residual_risk_tilt(
    data: dict[str, np.ndarray],
    residual: np.ndarray,
    *,
    betas: Iterable[float] = (0.0, 0.5, 1.0, 2.0),
) -> list[dict[str, float]]:
    scale = float(np.std(residual, ddof=1))
    if scale <= 0.0 or not np.isfinite(scale):
        raise ValueError("residual risk tilt needs positive residual scale")
    ordinary_intervention = float(np.mean(data["intervention_mean"]))
    rows: list[dict[str, float]] = []
    for beta in betas:
        weights = _normalized_weights(float(beta) * residual / scale)
        ess = float(1.0 / np.sum(np.square(weights)))
        weighted_intervention = float(weights @ data["intervention_mean"])
        rows.append(
            {
                "beta": float(beta),
                "ess_fraction": ess / residual.size,
                "weighted_residual_energy": float(weights @ residual),
                "weighted_horizon": float(weights @ data["horizon"]),
                "weighted_intervention_mean": weighted_intervention,
                "intervention_enrichment_ratio": weighted_intervention
                / max(ordinary_intervention, np.finfo(np.float64).eps),
                "weighted_safety_burden_rate": float(
                    weights @ data["safety_burden_rate"]
                ),
            }
        )
    return rows


def residual_risk_by_intervention_quintile(
    data: dict[str, np.ndarray],
    residual: np.ndarray,
) -> list[dict[str, float]]:
    scale = float(np.std(residual, ddof=1))
    if scale <= 0.0 or not np.isfinite(scale):
        raise ValueError("residual quintile audit needs positive residual scale")
    bins = _rank_bins(data["intervention_mean"], 5)
    rows: list[dict[str, float]] = []
    for bin_index in range(5):
        selected = np.flatnonzero(bins == bin_index)
        values = residual[selected]
        normalized = values / scale
        log_weights = normalized - float(np.max(normalized))
        weights = np.exp(log_weights)
        weights /= np.sum(weights)
        rows.append(
            {
                "intervention_quintile": int(bin_index),
                "num_trajectories": int(selected.size),
                "mean_intervention": float(
                    np.mean(data["intervention_mean"][selected])
                ),
                "residual_mean": float(np.mean(values)),
                "residual_std": float(np.std(values, ddof=1)),
                "residual_q90": float(np.quantile(values, 0.90)),
                "residual_q95": float(np.quantile(values, 0.95)),
                "residual_q99": float(np.quantile(values, 0.99)),
                "normalized_log_mgf_beta1": float(
                    np.log(np.mean(np.exp(normalized)))
                ),
                "within_quintile_tilted_ess_fraction": float(
                    (1.0 / np.sum(np.square(weights))) / selected.size
                ),
            }
        )
    return rows


def conditional_interface_permutation(
    *,
    strong_design: np.ndarray,
    full_design: np.ndarray,
    intervention_mean: np.ndarray,
    target: np.ndarray,
    base_prediction: np.ndarray,
    observed_relative_mae_reduction: float,
    repetitions: int,
    seed: int,
) -> dict[str, object]:
    if repetitions <= 0:
        raise ValueError("permutation repetitions must be positive")
    propensity, _ = cross_fitted_predictions(
        strong_design,
        intervention_mean,
        scheme="contiguous",
    )
    propensity_bins = _rank_bins(propensity, 10)
    interface_design = full_design[:, strong_design.shape[1] :]
    base_mae = float(np.mean(np.abs(target - base_prediction)))
    rng = np.random.default_rng(seed)
    null_reductions = np.empty(repetitions, dtype=np.float64)
    for repetition in range(repetitions):
        permutation = np.arange(target.size)
        for bin_index in range(10):
            members = np.flatnonzero(propensity_bins == bin_index)
            permutation[members] = rng.permutation(members)
        permuted_design = np.column_stack(
            [strong_design, interface_design[permutation]]
        )
        prediction, _ = cross_fitted_predictions(
            permuted_design,
            target,
            scheme="contiguous",
        )
        permuted_mae = float(np.mean(np.abs(target - prediction)))
        null_reductions[repetition] = (base_mae - permuted_mae) / max(
            base_mae, np.finfo(np.float64).eps
        )
    exceedances = int(
        np.sum(null_reductions >= observed_relative_mae_reduction)
    )
    return {
        "conditioning": (
            "joint row permutation of all interface features within deciles of "
            "cross-fitted intervention predicted from the strong confounder design"
        ),
        "repetitions": int(repetitions),
        "observed_relative_mae_reduction": float(
            observed_relative_mae_reduction
        ),
        "null_median": float(np.median(null_reductions)),
        "null_q95": float(np.quantile(null_reductions, 0.95)),
        "null_max": float(np.max(null_reductions)),
        "empirical_upper_tail_probability": float(
            (1 + exceedances) / (1 + repetitions)
        ),
        "interpretation": (
            "Approximate conditional randomization diagnostic, not a formal p-value; "
            "the conditional interface distribution is represented only by bins."
        ),
    }


def analyze_feature_arrays(
    data: dict[str, np.ndarray],
    *,
    bootstrap_repetitions: int,
    permutation_repetitions: int,
    seed: int,
) -> dict[str, object]:
    matrices, feature_names = build_feature_matrices(data)
    responses = {
        "total_energy": np.asarray(data["total_energy"], dtype=np.float64),
        "energy_per_step": np.asarray(data["energy_per_step"], dtype=np.float64),
    }
    model_results: dict[str, object] = {}
    predictions: dict[tuple[str, str, str], np.ndarray] = {}
    for scheme in ("contiguous", "interleaved"):
        scheme_results: dict[str, object] = {}
        for response_name, target in responses.items():
            response_results: dict[str, object] = {}
            for feature_set in FEATURE_SETS:
                prediction, alphas = cross_fitted_predictions(
                    matrices[feature_set], target, scheme=scheme
                )
                predictions[(scheme, response_name, feature_set)] = prediction
                response_results[feature_set] = {
                    **_regression_metrics(target, prediction),
                    "selected_ridge_alphas": alphas,
                    "num_features": int(matrices[feature_set].shape[1]),
                }
            scheme_results[response_name] = response_results
        model_results[scheme] = scheme_results

    primary_target = responses["total_energy"]
    primary_base = predictions[("contiguous", "total_energy", "strong_confounder")]
    primary_augmented = predictions[
        ("contiguous", "total_energy", "strong_plus_interface")
    ]
    base_absolute = np.abs(primary_target - primary_base)
    augmented_absolute = np.abs(primary_target - primary_augmented)
    base_squared = np.square(primary_target - primary_base)
    augmented_squared = np.square(primary_target - primary_augmented)
    block_length = int(math.ceil(math.sqrt(primary_target.size)))
    mae_improvement = _moving_block_bootstrap_mean_interval(
        base_absolute - augmented_absolute,
        block_length=block_length,
        repetitions=bootstrap_repetitions,
        seed=seed,
    )
    mse_improvement = _moving_block_bootstrap_mean_interval(
        base_squared - augmented_squared,
        block_length=block_length,
        repetitions=bootstrap_repetitions,
        seed=seed + 1,
    )
    base_metrics = model_results["contiguous"]["total_energy"]["strong_confounder"]
    augmented_metrics = model_results["contiguous"]["total_energy"][
        "strong_plus_interface"
    ]
    relative_mae_reduction = mae_improvement["estimate"] / max(
        float(base_metrics["mae"]), np.finfo(np.float64).eps
    )
    delta_r_squared = float(
        augmented_metrics["r_squared"] - base_metrics["r_squared"]
    )
    fold_stability: list[dict[str, float]] = []
    for fold_index, fold_indices in enumerate(
        _outer_folds(primary_target.size, 5, "contiguous")
    ):
        fold_base_mae = float(np.mean(base_absolute[fold_indices]))
        fold_augmented_mae = float(np.mean(augmented_absolute[fold_indices]))
        fold_stability.append(
            {
                "fold": int(fold_index),
                "trajectory_id_min": int(np.min(data["trajectory_id"][fold_indices])),
                "trajectory_id_max": int(np.max(data["trajectory_id"][fold_indices])),
                "base_mae": fold_base_mae,
                "augmented_mae": fold_augmented_mae,
                "relative_mae_reduction": (
                    fold_base_mae - fold_augmented_mae
                )
                / max(fold_base_mae, np.finfo(np.float64).eps),
            }
        )
    subgroup_rows: list[dict[str, object]] = []
    horizon_subgroup = _rank_bins(data["horizon"], 5).astype(str)
    for grouping_name, labels in (
        ("goal_type", data["goal_type"]),
        ("distance_bucket", data["distance_bucket"]),
        ("horizon_quintile", horizon_subgroup),
    ):
        for label in sorted(np.unique(labels).tolist()):
            selected = np.flatnonzero(labels == label)
            base_mae = float(np.mean(base_absolute[selected]))
            augmented_mae = float(np.mean(augmented_absolute[selected]))
            subgroup_rows.append(
                {
                    "grouping": grouping_name,
                    "label": str(label),
                    "num_trajectories": int(selected.size),
                    "base_mae": base_mae,
                    "augmented_mae": augmented_mae,
                    "relative_mae_reduction": (base_mae - augmented_mae)
                    / max(base_mae, np.finfo(np.float64).eps),
                    "mean_intervention": float(
                        np.mean(data["intervention_mean"][selected])
                    ),
                }
            )
    strong_residual = primary_target - primary_base
    matching = coarsened_horizon_match(
        data,
        base_residual=strong_residual,
        repetitions=bootstrap_repetitions,
        seed=seed + 2,
    )
    tilt_rows = residual_risk_tilt(data, strong_residual)
    quintile_risk_rows = residual_risk_by_intervention_quintile(
        data, strong_residual
    )
    permutation = conditional_interface_permutation(
        strong_design=matrices["strong_confounder"],
        full_design=matrices["strong_plus_interface"],
        intervention_mean=data["intervention_mean"],
        target=primary_target,
        base_prediction=primary_base,
        observed_relative_mae_reduction=float(relative_mae_reduction),
        repetitions=permutation_repetitions,
        seed=seed + 3,
    )
    beta_one = next(row for row in tilt_rows if row["beta"] == 1.0)
    matched_interval = matching["strong_base_residual_high_minus_low"]
    matched_excludes_zero = bool(
        matched_interval["ci95_lower"] is not None
        and np.isfinite(matched_interval["ci95_lower"])
        and (
            matched_interval["ci95_lower"] > 0.0
            or matched_interval["ci95_upper"] < 0.0
        )
    )
    tilt_deviation = abs(math.log(max(beta_one["intervention_enrichment_ratio"], 1e-12)))
    strong_support = bool(
        relative_mae_reduction >= 0.02
        and mae_improvement["ci95_lower"] > 0.0
        and tilt_deviation >= math.log(1.10)
        and matched_excludes_zero
        and permutation["empirical_upper_tail_probability"] <= 0.05
    )
    no_support = bool(
        relative_mae_reduction < 0.01
        and mae_improvement["ci95_lower"] <= 0.0 <= mae_improvement["ci95_upper"]
        and tilt_deviation < math.log(1.05)
        and not matched_excludes_zero
    )
    verdict = (
        "SUPPORTED_EXPLORATORY_WITHIN_COMPOSITION"
        if strong_support
        else "NOT_SUPPORTED_AFTER_HORIZON_MATCHING"
        if no_support
        else "MIXED_OR_WEAK_WITHIN_COMPOSITION_SIGNAL"
    )
    ablation_rows: list[dict[str, object]] = []
    for feature_set in (
        "strong_plus_action_delta",
        "strong_plus_burden_rate",
        "strong_plus_interface",
    ):
        metrics = model_results["contiguous"]["total_energy"][feature_set]
        ablation_rows.append(
            {
                "feature_set": feature_set,
                "mae": float(metrics["mae"]),
                "relative_mae_reduction_over_strong": (
                    float(base_metrics["mae"]) - float(metrics["mae"])
                )
                / max(float(base_metrics["mae"]), np.finfo(np.float64).eps),
                "delta_r_squared_over_strong": float(metrics["r_squared"])
                - float(base_metrics["r_squared"]),
            }
        )
    return {
        "status": "EXPLORATORY_HORIZON_MATCHED_INTERFACE_AUDIT",
        "formal_gate_evidence": False,
        "verdict": verdict,
        "claim_scope": (
            "Within one frozen policy/filter composition, do post-trajectory "
            "intervention summaries add held-out association with total energy after "
            "horizon/geometric/nominal controls?"
        ),
        "non_claim": (
            "This cannot identify policy-filter composition shift, target support, "
            "Oracle headroom, Pareto improvement, or a deployable predictor: duration, "
            "path length, and aggregate intervention are future trajectory summaries."
        ),
        "num_trajectories": int(primary_target.size),
        "feature_contract": {
            "forbidden_as_predictors": ["step_energy", "energy_to_go", "burden_to_go"],
            "post_trajectory_mechanism_only": [
                "total_duration",
                "path_length",
                "path_ratio",
                "aggregate nominal-action effort",
                "aggregate intervention statistics",
            ],
            "feature_sets": feature_names,
            "primary_fold_scheme": "five contiguous trajectory-id blocks",
            "sensitivity_fold_scheme": "five interleaved trajectory-id folds",
        },
        "cross_fitted_models": model_results,
        "primary_interface_increment": {
            "base": "strong_confounder",
            "augmented": "strong_plus_interface",
            "delta_r_squared": delta_r_squared,
            "relative_mae_reduction": float(relative_mae_reduction),
            "paired_mae_improvement": mae_improvement,
            "paired_mse_improvement": mse_improvement,
            "contiguous_fold_stability": fold_stability,
        },
        "interface_group_ablation": ablation_rows,
        "subgroup_interface_increment": subgroup_rows,
        "coarsened_matching": matching,
        "residual_risk_tilt": tilt_rows,
        "residual_risk_by_intervention_quintile": quintile_risk_rows,
        "conditional_interface_permutation": permutation,
        "decision_rule": {
            "strong_support": (
                "relative MAE reduction >=2%, block-bootstrap lower CI >0, "
                "beta=1 intervention enrichment/depletion >=10%, and matched "
                "strong-base residual CI excludes zero, and approximate "
                "conditional-permutation upper-tail <=5%"
            ),
            "not_supported": (
                "relative MAE reduction <1%, MAE CI spans zero, beta=1 change <5%, "
                "and matched residual CI includes zero"
            ),
        },
    }


def render_markdown(report: dict[str, object]) -> str:
    increment = report["primary_interface_increment"]
    matching = report["coarsened_matching"]
    lines = [
        "# R3 Horizon-Matched Executed-Interface Energy Audit",
        "",
        "> Exploratory within-composition falsification analysis; not formal Gate evidence.",
        "",
        f"- Verdict: `{report['verdict']}`",
        f"- Complete trajectories: {report['num_trajectories']}",
        f"- Strong-base to +interface delta R-squared: {increment['delta_r_squared']:.6f}",
        f"- Relative held-out MAE reduction: {increment['relative_mae_reduction']:.4%}",
        f"- Paired block-bootstrap MAE improvement: "
        f"{increment['paired_mae_improvement']['estimate']:.6f} "
        f"[{increment['paired_mae_improvement']['ci95_lower']:.6f}, "
        f"{increment['paired_mae_improvement']['ci95_upper']:.6f}]",
        f"- Conditional-permutation upper-tail probability: "
        f"{report['conditional_interface_permutation']['empirical_upper_tail_probability']:.6f}",
        "",
        "## Cross-fitted prediction",
        "",
        "| fold scheme | response | features | MAE | RMSE | R-squared |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for scheme, scheme_rows in report["cross_fitted_models"].items():
        for response, response_rows in scheme_rows.items():
            for features, metrics in response_rows.items():
                lines.append(
                    f"| {scheme} | {response} | {features} | "
                    f"{metrics['mae']:.6f} | {metrics['rmse']:.6f} | "
                    f"{metrics['r_squared']:.6f} |"
                )
    residual_match = matching["strong_base_residual_high_minus_low"]
    rate_match = matching["energy_rate_high_minus_low"]
    lines.extend(
        [
            "",
            "## Coarsened horizon/distance/goal matching",
            "",
            f"- Retained cells / trajectories: {matching['retained_cells']} / "
            f"{matching['matched_trajectories']}",
            f"- High-minus-low intervention contrast: "
            f"{matching['mean_high_minus_low_intervention']}",
            f"- Strong-base residual difference: {residual_match['estimate']:.6f} "
            f"[{residual_match['ci95_lower']:.6f}, {residual_match['ci95_upper']:.6f}]",
            f"- Energy-rate difference: {rate_match['estimate']:.8f} "
            f"[{rate_match['ci95_lower']:.8f}, {rate_match['ci95_upper']:.8f}]",
            "",
            "## Residual-risk tilt",
            "",
            "| beta | ESS fraction | weighted residual energy | weighted horizon | weighted intervention | enrichment ratio | burden rate |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in report["residual_risk_tilt"]:
        lines.append(
            f"| {row['beta']:.2f} | {row['ess_fraction']:.4f} | "
            f"{row['weighted_residual_energy']:.6f} | {row['weighted_horizon']:.2f} | "
            f"{row['weighted_intervention_mean']:.6f} | "
            f"{row['intervention_enrichment_ratio']:.4f} | "
            f"{row['weighted_safety_burden_rate']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Residual risk by intervention quintile",
            "",
            "The log-MGF uses beta=1 after dividing residual energy by its global sample standard deviation.",
            "",
            "| intervention quintile | paths | mean intervention | residual mean | residual std | q90 | q95 | q99 | normalized log-MGF | tilted ESS frac. |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in report["residual_risk_by_intervention_quintile"]:
        lines.append(
            f"| {row['intervention_quintile']} | {row['num_trajectories']} | "
            f"{row['mean_intervention']:.6f} | {row['residual_mean']:.6f} | "
            f"{row['residual_std']:.6f} | {row['residual_q90']:.6f} | "
            f"{row['residual_q95']:.6f} | {row['residual_q99']:.6f} | "
            f"{row['normalized_log_mgf_beta1']:.6f} | "
            f"{row['within_quintile_tilted_ess_fraction']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Interface group ablation",
            "",
            "| added interface group | MAE | relative MAE reduction | delta R-squared |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in report["interface_group_ablation"]:
        lines.append(
            f"| {row['feature_set']} | {row['mae']:.6f} | "
            f"{row['relative_mae_reduction_over_strong']:.4%} | "
            f"{row['delta_r_squared_over_strong']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Subgroup stability",
            "",
            "| grouping | label | paths | base MAE | +interface MAE | relative reduction | mean intervention |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in report["subgroup_interface_increment"]:
        lines.append(
            f"| {row['grouping']} | {row['label']} | {row['num_trajectories']} | "
            f"{row['base_mae']:.6f} | {row['augmented_mae']:.6f} | "
            f"{row['relative_mae_reduction']:.4%} | {row['mean_intervention']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Evidence boundary",
            "",
            report["non_claim"],
            "The strong confounder model includes total duration, realized path length, "
            "path ratio, and nominal-action effort. These are mechanism controls and may "
            "also mediate intervention effects; both horizon-only and strong-control "
            "comparisons are therefore reported.",
            "All cross-fitted regressions are held-out association diagnostics. They use "
            "post-trajectory summaries and are not deployable Resource-to-Go predictors.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--permutation-repetitions", type=int, default=200)
    parser.add_argument("--seed", type=int, default=170001)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_trajectory_features(args.trajectory_dir)
    report = analyze_feature_arrays(
        data,
        bootstrap_repetitions=args.bootstrap_repetitions,
        permutation_repetitions=args.permutation_repetitions,
        seed=args.seed,
    )
    report["trajectory_dir"] = str(args.trajectory_dir.resolve())
    report["maximum_manifest_energy_sum_error"] = float(
        data["maximum_manifest_energy_sum_error"][0]
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "horizon_matched_interface_audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "horizon_matched_interface_audit.md").write_text(
        render_markdown(report), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
