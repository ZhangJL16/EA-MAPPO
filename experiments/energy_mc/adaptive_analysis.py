from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import pearsonr, spearmanr

from experiments.energy_mc.conformal import PackedMissionDataset
from experiments.energy_mc.core import PackedEnergyDataset
from review_bundle.safety.energy.mc_regression import energy_regression_metrics


MAP_EXTENT = np.asarray([4000.0, 4000.0, 400.0], dtype=np.float64)
D_MAX = float(np.linalg.norm(MAP_EXTENT))
VELOCITY_SCALE = np.asarray([20.0, 20.0, 5.0], dtype=np.float64)
REMAINING_DISTANCE_EDGES = np.asarray(
    [0.0, 100.0, 500.0, 1500.0, 2500.0, 4000.0, np.inf],
    dtype=np.float64,
)
REMAINING_DISTANCE_LABELS = (
    "0-100",
    "100-500",
    "500-1500",
    "1500-2500",
    "2500-4000",
    ">4000",
)


def contiguous_trajectory_slices(trajectory_ids: np.ndarray) -> list[tuple[int, slice]]:
    ids = np.asarray(trajectory_ids, dtype=np.int64)
    if ids.ndim != 1 or ids.size == 0:
        raise ValueError("trajectory ids must be a nonempty vector")
    starts = np.concatenate([[0], np.flatnonzero(ids[1:] != ids[:-1]) + 1])
    stops = np.concatenate([starts[1:], [ids.size]])
    rows = [(int(ids[start]), slice(int(start), int(stop))) for start, stop in zip(starts, stops)]
    if len(rows) != np.unique(ids).size:
        raise ValueError("each trajectory must occupy one contiguous block")
    return rows


def _boundary_distance(positions: np.ndarray) -> np.ndarray:
    values = np.asarray(positions, dtype=np.float64)
    return np.min(np.concatenate([values, MAP_EXTENT - values], axis=1), axis=1)


def trajectory_diagnostic_frame(
    dataset: PackedEnergyDataset,
    predictions: np.ndarray,
) -> pd.DataFrame:
    predicted = np.asarray(predictions, dtype=np.float64)
    if predicted.shape != dataset.targets.shape:
        raise ValueError("predictions do not align with the dataset")
    metadata = {int(row["trajectory_id"]): row for row in dataset.metadata}
    rows: list[dict[str, object]] = []
    for trajectory_id, selected in contiguous_trajectory_slices(dataset.trajectory_ids):
        record = metadata[trajectory_id]
        states = dataset.states[selected].astype(np.float64)
        truth = dataset.targets[selected].astype(np.float64)
        residual = truth - predicted[selected]
        velocity = states[:, :3] * VELOCITY_SCALE
        speed = np.linalg.norm(velocity, axis=1)
        distance = states[:, -1] * D_MAX
        goal = np.asarray(record["goal_position"], dtype=np.float64)
        position = goal[None, :] - states[:, 3:6] * distance[:, None]
        transition_dt = dataset.transition_dt[selected].astype(np.float64)
        if velocity.shape[0] > 1:
            acceleration = np.diff(velocity, axis=0) / np.maximum(
                transition_dt[:-1, None],
                1e-8,
            )
            acceleration_norm = np.linalg.norm(acceleration, axis=1)
        else:
            acceleration_norm = np.zeros(1, dtype=np.float64)
        start = np.asarray(record["start_position"], dtype=np.float64)
        initial_velocity = np.asarray(record["initial_velocity"], dtype=np.float64)
        initial_distance = float(record["initial_goal_distance"])
        path_length = float(record["path_length"])
        rows.append(
            {
                "trajectory_id": trajectory_id,
                "goal_type": str(record["goal_type"]),
                "distance_bucket": str(record["distance_bucket"]),
                "initial_goal_distance": initial_distance,
                "mean_remaining_distance": float(np.mean(distance)),
                "p90_remaining_distance": float(np.quantile(distance, 0.90)),
                "trajectory_total_length": path_length,
                "path_ratio": path_length / max(initial_distance, 1e-8),
                "flight_steps": int(record["steps"]),
                "flight_time": float(record["flight_time"]),
                "initial_velocity": float(np.linalg.norm(initial_velocity)),
                "mean_speed": float(np.mean(speed)),
                "max_speed": float(np.max(speed)),
                "mean_acceleration": float(np.mean(acceleration_norm)),
                "max_acceleration": float(np.max(acceleration_norm)),
                "vertical_displacement": float(abs(goal[2] - start[2])),
                "goal_direction_x": float(states[0, 3]),
                "goal_direction_y": float(states[0, 4]),
                "goal_direction_z": float(states[0, 5]),
                "boundary_contact": bool(record["had_boundary_contact"]),
                "boundary_contact_duration_steps": int(record["boundary_contact_steps"]),
                "max_consecutive_boundary_contacts": int(
                    record.get("maximum_consecutive_boundary_contacts", 0)
                ),
                "start_x": float(start[0]),
                "start_y": float(start[1]),
                "start_z": float(start[2]),
                "start_dx_min": float(start[0]),
                "start_dx_max": float(MAP_EXTENT[0] - start[0]),
                "start_dy_min": float(start[1]),
                "start_dy_max": float(MAP_EXTENT[1] - start[1]),
                "start_dz_min": float(start[2]),
                "start_dz_max": float(MAP_EXTENT[2] - start[2]),
                "start_distance_to_boundary": float(_boundary_distance(start[None, :])[0]),
                "mean_x": float(np.mean(position[:, 0])),
                "mean_y": float(np.mean(position[:, 1])),
                "mean_z": float(np.mean(position[:, 2])),
                "mean_distance_to_boundary": float(np.mean(_boundary_distance(position))),
                "minimum_distance_to_boundary": float(np.min(_boundary_distance(position))),
                "true_initial_energy": float(truth[0]),
                "predicted_initial_energy": float(predicted[selected][0]),
                "initial_residual": float(residual[0]),
                "mean_residual": float(np.mean(residual)),
                "std_residual": float(np.std(residual)),
                "p90_residual": float(np.quantile(residual, 0.90)),
                "p95_residual": float(np.quantile(residual, 0.95)),
                "p99_residual": float(np.quantile(residual, 0.99)),
                "trajectory_max_underestimation": float(np.max(residual)),
                "mean_absolute_residual": float(np.mean(np.abs(residual))),
            }
        )
    return pd.DataFrame(rows)


def state_residual_by_remaining_distance(
    dataset: PackedEnergyDataset,
    predictions: np.ndarray,
) -> pd.DataFrame:
    predicted = np.asarray(predictions, dtype=np.float64)
    if predicted.shape != dataset.targets.shape:
        raise ValueError("predictions do not align with the dataset")
    remaining_distance = dataset.states[:, -1].astype(np.float64) * D_MAX
    residual = dataset.targets.astype(np.float64) - predicted
    bucket = pd.cut(
        remaining_distance,
        bins=REMAINING_DISTANCE_EDGES,
        labels=REMAINING_DISTANCE_LABELS,
        right=False,
        include_lowest=True,
    )
    frame = pd.DataFrame(
        {
            "trajectory_id": dataset.trajectory_ids,
            "remaining_distance": remaining_distance,
            "residual": residual,
            "positive_underestimation": np.maximum(residual, 0.0),
            "remaining_distance_bucket": bucket,
        }
    )
    return (
        frame.groupby("remaining_distance_bucket", observed=True)
        .agg(
            state_count=("residual", "count"),
            trajectory_count=("trajectory_id", "nunique"),
            mean_remaining_distance=("remaining_distance", "mean"),
            mean_residual=("residual", "mean"),
            std_residual=("residual", "std"),
            underestimation_rate=("residual", lambda values: float(np.mean(values > 0.0))),
            mean_positive_underestimation=("positive_underestimation", "mean"),
            p90_positive_underestimation=(
                "positive_underestimation",
                lambda values: values.quantile(0.90),
            ),
            p95_positive_underestimation=(
                "positive_underestimation",
                lambda values: values.quantile(0.95),
            ),
            p99_positive_underestimation=(
                "positive_underestimation",
                lambda values: values.quantile(0.99),
            ),
            maximum_underestimation=("positive_underestimation", "max"),
        )
        .reset_index()
    )


def correlation_table(
    frame: pd.DataFrame,
    *,
    targets: Iterable[str],
    excluded: Iterable[str] = (),
) -> pd.DataFrame:
    excluded_names = set(excluded) | set(targets)
    numeric = [
        column
        for column in frame.select_dtypes(include=[np.number]).columns
        if column not in excluded_names
    ]
    rows = []
    for target in targets:
        for feature in numeric:
            x = frame[feature].to_numpy(dtype=np.float64)
            y = frame[target].to_numpy(dtype=np.float64)
            finite = np.isfinite(x) & np.isfinite(y)
            if np.sum(finite) < 3 or np.std(x[finite]) == 0.0 or np.std(y[finite]) == 0.0:
                continue
            rows.append(
                {
                    "target": target,
                    "feature": feature,
                    "pearson": float(pearsonr(x[finite], y[finite]).statistic),
                    "spearman": float(spearmanr(x[finite], y[finite]).statistic),
                    "count": int(np.sum(finite)),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["target", "spearman"],
        ascending=[True, False],
    )


def residual_group_summary(frame: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    return (
        frame.groupby(groups, dropna=False)["trajectory_max_underestimation"]
        .agg(
            count="count",
            mean="mean",
            std="std",
            p90=lambda values: values.quantile(0.90),
            p95=lambda values: values.quantile(0.95),
            p99=lambda values: values.quantile(0.99),
            maximum="max",
        )
        .reset_index()
    )


def state_context_arrays(dataset: PackedEnergyDataset) -> dict[str, np.ndarray]:
    metadata = {int(row["trajectory_id"]): row for row in dataset.metadata}
    count = dataset.states.shape[0]
    position = np.empty((count, 3), dtype=np.float64)
    path_ratio = np.empty(count, dtype=np.float64)
    boundary_contact = np.empty(count, dtype=np.float64)
    vertical_displacement = np.empty(count, dtype=np.float64)
    future_path_length = np.empty(count, dtype=np.float64)
    for trajectory_id, selected in contiguous_trajectory_slices(dataset.trajectory_ids):
        record = metadata[trajectory_id]
        states = dataset.states[selected].astype(np.float64)
        distance = states[:, -1] * D_MAX
        goal = np.asarray(record["goal_position"], dtype=np.float64)
        positions = goal[None, :] - states[:, 3:6] * distance[:, None]
        position[selected] = positions
        initial_distance = float(record["initial_goal_distance"])
        path_ratio[selected] = float(record["path_length"]) / max(initial_distance, 1e-8)
        boundary_contact[selected] = float(bool(record["had_boundary_contact"]))
        start = np.asarray(record["start_position"], dtype=np.float64)
        vertical_displacement[selected] = abs(goal[2] - start[2])
        segment = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        remaining = np.concatenate([np.cumsum(segment[::-1])[::-1], [0.0]])
        future_path_length[selected] = remaining
    return {
        "absolute_x": position[:, 0],
        "absolute_y": position[:, 1],
        "absolute_z": position[:, 2],
        "distance_to_boundary": _boundary_distance(position),
        "future_boundary_contact": boundary_contact,
        "trajectory_path_ratio": path_ratio,
        "vertical_displacement": vertical_displacement,
        "future_path_length": future_path_length,
    }


def nearest_neighbor_aliasing_analysis(
    dataset: PackedEnergyDataset,
    *,
    sample_size: int = 100_000,
    seed: int = 0,
) -> tuple[dict[str, object], pd.DataFrame]:
    rng = np.random.default_rng(seed)
    sample_size = min(int(sample_size), dataset.states.shape[0])
    selected = np.sort(rng.choice(dataset.states.shape[0], size=sample_size, replace=False))
    states = dataset.states[selected].astype(np.float64)
    truth = dataset.targets[selected].astype(np.float64)
    ids = dataset.trajectory_ids[selected]
    context = state_context_arrays(dataset)
    sampled_context = {name: values[selected] for name, values in context.items()}
    tree = cKDTree(states)
    distances, neighbors = tree.query(states, k=12, workers=-1)
    chosen_neighbor = np.full(sample_size, -1, dtype=np.int64)
    chosen_distance = np.full(sample_size, np.nan, dtype=np.float64)
    for row in range(sample_size):
        for distance, candidate in zip(distances[row, 1:], neighbors[row, 1:]):
            if ids[candidate] != ids[row]:
                chosen_neighbor[row] = int(candidate)
                chosen_distance[row] = float(distance)
                break
    valid = chosen_neighbor >= 0
    left = np.flatnonzero(valid)
    right = chosen_neighbor[valid]
    return_difference = np.abs(truth[left] - truth[right])
    close_threshold = float(np.quantile(chosen_distance[valid], 0.01))
    close = chosen_distance[valid] <= close_threshold
    pair_rows = []
    for name, values in sampled_context.items():
        difference = np.abs(values[left] - values[right])
        if np.std(difference) == 0.0 or np.std(return_difference) == 0.0:
            pearson = 0.0
            spearman = 0.0
        else:
            pearson = float(pearsonr(difference, return_difference).statistic)
            spearman = float(spearmanr(difference, return_difference).statistic)
        close_difference = difference[close]
        close_returns = return_difference[close]
        if (
            close_difference.size < 3
            or np.std(close_difference) == 0.0
            or np.std(close_returns) == 0.0
        ):
            close_pearson = 0.0
            close_spearman = 0.0
        else:
            close_pearson = float(pearsonr(close_difference, close_returns).statistic)
            close_spearman = float(spearmanr(close_difference, close_returns).statistic)
        pair_rows.append(
            {
                "omitted_context": name,
                "pearson_with_nn_return_difference": pearson,
                "spearman_with_nn_return_difference": spearman,
                "median_pair_difference": float(np.median(difference)),
                "close_pair_pearson": close_pearson,
                "close_pair_spearman": close_spearman,
                "close_pair_median_difference": float(np.median(close_difference)),
                "close_pair_p95_difference": float(np.quantile(close_difference, 0.95)),
            }
        )
    local_variances = []
    for row in range(sample_size):
        candidate = neighbors[row, 1:]
        candidate = candidate[ids[candidate] != ids[row]][:5]
        if candidate.size:
            local_variances.append(float(np.var(truth[candidate])))
    summary = {
        "sample_size": sample_size,
        "median_nearest_other_trajectory_feature_distance": float(
            np.median(chosen_distance[valid])
        ),
        "p01_feature_distance_threshold": close_threshold,
        "median_return_difference_all_nearest_pairs": float(np.median(return_difference)),
        "p95_return_difference_all_nearest_pairs": float(np.quantile(return_difference, 0.95)),
        "median_return_difference_closest_one_percent": float(np.median(return_difference[close])),
        "p95_return_difference_closest_one_percent": float(np.quantile(return_difference[close], 0.95)),
        "maximum_return_difference_closest_one_percent": float(np.max(return_difference[close])),
        "mean_local_conditional_variance_k5": float(np.mean(local_variances)),
        "p95_local_conditional_variance_k5": float(np.quantile(local_variances, 0.95)),
    }
    return summary, pd.DataFrame(pair_rows).sort_values(
        "spearman_with_nn_return_difference",
        ascending=False,
    )


def _trajectory_coverage(
    truth: np.ndarray,
    upper: np.ndarray,
    trajectory_ids: np.ndarray,
    mask: np.ndarray,
) -> tuple[int, int, float | None]:
    selected_ids = np.unique(trajectory_ids[mask])
    failed_ids = np.unique(trajectory_ids[mask & (truth > upper)])
    count = int(selected_ids.size)
    covered = count - int(failed_ids.size)
    return count, covered, None if count == 0 else float(covered / count)


def wilson_interval(successes: int, count: int, *, z: float = 1.959963984540054) -> tuple[float | None, float | None]:
    if count <= 0:
        return None, None
    proportion = successes / count
    denominator = 1.0 + z**2 / count
    center = (proportion + z**2 / (2.0 * count)) / denominator
    radius = z * np.sqrt(proportion * (1.0 - proportion) / count + z**2 / (4.0 * count**2)) / denominator
    return float(max(0.0, center - radius)), float(min(1.0, center + radius))


def upper_bound_metrics(
    point: np.ndarray,
    upper: np.ndarray,
    truth: np.ndarray,
    trajectory_ids: np.ndarray,
    mask: np.ndarray,
) -> dict[str, object]:
    if not np.any(mask):
        return {"count": 0, "num_trajectories": 0}
    result = dict(energy_regression_metrics(point[mask], truth[mask], upper_bounds=upper[mask]))
    result["pointwise_state_coverage"] = result.pop("upper95_coverage")
    count, covered, coverage = _trajectory_coverage(truth, upper, trajectory_ids, mask)
    interval_low, interval_high = wilson_interval(covered, count)
    width = upper[mask] - point[mask]
    conservatism = upper[mask] - truth[mask]
    result.update(
        {
            "num_trajectories": count,
            "covered_trajectories": covered,
            "whole_trajectory_simultaneous_coverage": coverage,
            "trajectory_coverage_wilson95_low": interval_low,
            "trajectory_coverage_wilson95_high": interval_high,
            "mean_bound_width": float(np.mean(width)),
            "median_bound_width": float(np.median(width)),
            "p95_bound_width": float(np.quantile(width, 0.95)),
            "mean_bound_over_true_ratio": float(np.mean(upper[mask] / np.maximum(truth[mask], 1e-8))),
            "conservatism_cost": float(np.mean(conservatism)),
        }
    )
    return result


def evaluate_goal_upper(
    dataset: PackedEnergyDataset,
    point: np.ndarray,
    upper: np.ndarray,
) -> dict[str, object]:
    truth = dataset.targets.astype(np.float64)
    ids = dataset.trajectory_ids
    all_mask = np.ones(truth.shape, dtype=bool)
    by_goal = {
        name: upper_bound_metrics(point, upper, truth, ids, dataset.goal_types == name)
        for name in np.unique(dataset.goal_types)
    }
    by_distance = {
        name: upper_bound_metrics(point, upper, truth, ids, dataset.distance_buckets == name)
        for name in np.unique(dataset.distance_buckets)
    }
    intersections = {}
    for goal_type in np.unique(dataset.goal_types):
        for bucket in np.unique(dataset.distance_buckets):
            mask = (dataset.goal_types == goal_type) & (dataset.distance_buckets == bucket)
            if np.any(mask):
                intersections[f"{goal_type}|{bucket}"] = upper_bound_metrics(
                    point,
                    upper,
                    truth,
                    ids,
                    mask,
                )
    return {
        "overall": upper_bound_metrics(point, upper, truth, ids, all_mask),
        "by_goal_type": by_goal,
        "by_initial_distance_bucket": by_distance,
        "by_goal_type_and_initial_distance": intersections,
    }


def mission_subset(dataset: PackedMissionDataset, mission_ids: np.ndarray) -> PackedMissionDataset:
    selected_ids = np.asarray(mission_ids, dtype=np.int64)
    mask = np.isin(dataset.mission_ids, selected_ids)
    metadata = [row for row in dataset.metadata if int(row["mission_id"]) in set(selected_ids.tolist())]
    return PackedMissionDataset(
        task_states=dataset.task_states[mask],
        return_after_states=dataset.return_after_states[mask],
        true_mission_energy=dataset.true_mission_energy[mask],
        mission_ids=dataset.mission_ids[mask],
        initial_distance_buckets=dataset.initial_distance_buckets[mask],
        metadata=metadata,
    )


def stratified_mission_split(
    dataset: PackedMissionDataset,
    *,
    seed: int,
    fractions: tuple[float, float, float] = (0.6, 0.2, 0.2),
) -> tuple[PackedMissionDataset, PackedMissionDataset, PackedMissionDataset]:
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError("mission split fractions must sum to one")
    rng = np.random.default_rng(seed)
    split_ids: list[list[int]] = [[], [], []]
    for bucket in np.unique(dataset.initial_distance_buckets):
        ids = np.unique(dataset.mission_ids[dataset.initial_distance_buckets == bucket])
        ids = rng.permutation(ids)
        first = int(round(fractions[0] * ids.size))
        second = first + int(round(fractions[1] * ids.size))
        for destination, values in zip(split_ids, (ids[:first], ids[first:second], ids[second:])):
            destination.extend(int(value) for value in values)
    return tuple(mission_subset(dataset, np.asarray(ids)) for ids in split_ids)  # type: ignore[return-value]


def mission_component_predictions(
    point_estimator,
    dataset: PackedMissionDataset,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    task = point_estimator(dataset.task_states)
    returned = point_estimator(dataset.return_after_states)
    return task, returned, task + returned


def mission_residual_dependence(
    point_estimator,
    dataset: PackedMissionDataset,
) -> dict[str, object]:
    task_prediction, return_prediction, _ = mission_component_predictions(point_estimator, dataset)
    metadata = {int(row["mission_id"]): row for row in dataset.metadata}
    task_residual = []
    return_residual = []
    buckets = []
    for mission_id, selected in contiguous_trajectory_slices(dataset.mission_ids):
        row = metadata[mission_id]
        task_residual.append(float(row["task_energy"] - task_prediction[selected][0]))
        return_residual.append(float(row["return_after_task_energy"] - return_prediction[selected][0]))
        buckets.append(str(row["initial_task_distance_bucket"]))
    task_values = np.asarray(task_residual)
    return_values = np.asarray(return_residual)

    def dependence(mask: np.ndarray) -> dict[str, object]:
        left = task_values[mask]
        right = return_values[mask]
        joint_tail = (left >= np.quantile(left, 0.9)) & (right >= np.quantile(right, 0.9))
        return {
            "count": int(mask.sum()),
            "pearson": float(pearsonr(left, right).statistic),
            "spearman": float(spearmanr(left, right).statistic),
            "joint_upper_decile_rate": float(np.mean(joint_tail)),
            "independence_reference": 0.01,
        }

    bucket_values = np.asarray(buckets)
    return {
        "overall": dependence(np.ones(task_values.shape, dtype=bool)),
        "by_initial_task_distance_bucket": {
            bucket: dependence(bucket_values == bucket) for bucket in np.unique(bucket_values)
        },
    }


def evaluate_mission_upper(
    dataset: PackedMissionDataset,
    point: np.ndarray,
    upper: np.ndarray,
    *,
    battery_capacity: float,
    reserve_fraction: float,
    remaining_energy_samples: np.ndarray,
) -> dict[str, object]:
    truth = dataset.true_mission_energy.astype(np.float64)
    ids = dataset.mission_ids
    reserve = float(battery_capacity * reserve_fraction)
    remaining = np.asarray(remaining_energy_samples, dtype=np.float64)
    if remaining.shape != truth.shape:
        raise ValueError("remaining-energy proxy samples must align with mission states")

    def metrics(mask: np.ndarray) -> dict[str, object]:
        result = upper_bound_metrics(point, upper, truth, ids, mask)
        feasible = remaining[mask] >= truth[mask] + reserve
        accepted = remaining[mask] >= upper[mask] + reserve
        unnecessary = feasible & (remaining[mask] < upper[mask] + reserve)
        unsafe_accept = accepted & ~feasible
        result["feasible_state_count"] = int(np.sum(feasible))
        result["estimated_accepted_state_count"] = int(np.sum(accepted))
        result["estimated_task_acceptance_rate"] = float(np.mean(accepted))
        result["true_feasible_opportunity_rate"] = float(np.mean(feasible))
        result["unnecessary_return_proxy_count"] = int(np.sum(unnecessary))
        result["unnecessary_return_proxy_rate"] = None if not np.any(feasible) else float(
            np.sum(unnecessary) / np.sum(feasible)
        )
        result["unsafe_accept_proxy_count"] = int(np.sum(unsafe_accept))
        result["unsafe_accept_proxy_rate"] = None if not np.any(accepted) else float(
            np.sum(unsafe_accept) / np.sum(accepted)
        )
        result["uniform_energy_opportunity_fraction"] = float(
            np.mean(np.maximum(upper[mask] - truth[mask], 0.0)) / battery_capacity
        )
        return result

    return {
        "overall": metrics(np.ones(truth.shape, dtype=bool)),
        "by_initial_task_distance_bucket": {
            bucket: metrics(dataset.initial_distance_buckets == bucket)
            for bucket in np.unique(dataset.initial_distance_buckets)
        },
    }


__all__ = [
    "D_MAX",
    "MAP_EXTENT",
    "VELOCITY_SCALE",
    "contiguous_trajectory_slices",
    "correlation_table",
    "evaluate_goal_upper",
    "evaluate_mission_upper",
    "mission_component_predictions",
    "mission_residual_dependence",
    "mission_subset",
    "nearest_neighbor_aliasing_analysis",
    "residual_group_summary",
    "state_context_arrays",
    "stratified_mission_split",
    "trajectory_diagnostic_frame",
    "upper_bound_metrics",
    "wilson_interval",
]
