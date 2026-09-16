#!/usr/bin/env python3
"""Exploratory C/R factor decomposition and legal-observability audit.

This analysis uses only the completed DEV outcomes.  It fits small linear ridge
diagnostics with physical-world cross-fitting; it does not train an actor, a
critic, or a larger history encoder.  Exact obstacle geometry is isolated in a
clearly labelled privileged-only feature channel.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.cmi_v3_models import fold_for_world
from scripts.validate_identification_contract import file_hash


ACTION_ORDER = ("R", "C")
TERMINATIONS = ("returned", "branch_guard", "return_deadline", "energy_exhausted")
CAPACITY = 378.72626091628933
WORKSPACE = np.asarray([4000.0, 4000.0, 400.0], dtype=np.float64)
D_MAX = float(np.linalg.norm(WORKSPACE))
ALPHAS = (0.1, 1.0, 10.0, 100.0)
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 2_026_091_501


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def decode_distance(observation: np.ndarray) -> np.ndarray:
    return np.expm1(np.clip(observation[..., 6], 0.0, 1.0) * np.log1p(D_MAX))


def lidar_summary(observation: np.ndarray) -> np.ndarray:
    ranges = np.asarray(observation[..., 7:1031], dtype=np.float64)
    hits = np.asarray(observation[..., 1031:2055], dtype=np.float64)
    masked = np.where(hits > 0.5, ranges, 1.0)
    return np.stack(
        (
            hits.mean(axis=-1),
            masked.min(axis=-1),
            np.quantile(masked, 0.1, axis=-1),
            masked.mean(axis=-1),
        ),
        axis=-1,
    )


def segment_geometry(a: np.ndarray, b: np.ndarray, obstacles: list[dict[str, Any]]) -> list[float]:
    """Privileged 2-D cylinder-layout summaries for one straight segment."""
    start, end = np.asarray(a[:2], np.float64), np.asarray(b[:2], np.float64)
    delta = end - start
    length2 = float(delta @ delta)
    signed = []
    for obstacle in obstacles:
        center = np.asarray(obstacle["position"], np.float64)
        t = 0.0 if length2 == 0.0 else float(np.clip((center - start) @ delta / length2, 0.0, 1.0))
        signed.append(float(np.linalg.norm(center - (start + t * delta)) - obstacle["radius"]))
    values = np.asarray(signed, np.float64)
    return [
        float(values.min() / 100.0),
        float(np.mean(values < 0.0)),
        float(np.mean(values < 50.0)),
        float(np.mean(values < 100.0)),
    ]


def legal_features(arrays: dict[str, np.ndarray], world: dict[str, Any], index: int) -> tuple[list[str], np.ndarray, list[str], np.ndarray, np.ndarray, np.ndarray]:
    nav = np.asarray(arrays["nav_observation"][index], np.float64)
    ret = np.asarray(arrays["return_observation"][index], np.float64)
    battery = np.asarray(arrays["battery"][index], np.float64)
    return_distance = np.asarray(arrays["distance_to_charger"][index], np.float64)
    task_distance = decode_distance(nav)
    velocity = nav[:, :3]
    task_direction, return_direction = nav[:, 3:6], ret[:, 3:6]
    task_progress = np.asarray(arrays["task_progress"][index], np.float64)
    task_clock = np.asarray(arrays["task_clock"][index], np.float64)
    step = np.asarray(arrays["step"][index], np.float64)
    correction = np.linalg.norm(
        np.asarray(arrays["previous_executed_action"][index], np.float64)
        - np.asarray(arrays["previous_nominal_action"][index], np.float64), axis=1
    )
    contacts = np.asarray(arrays["previous_contact"][index], np.float64)
    lidar = lidar_summary(nav)
    charger = np.asarray(world["charger_position"], np.float64)
    position = charger - return_direction[-1] * return_distance[-1]
    task = position + task_direction[-1] * task_distance[-1]
    task_projection = np.sum(velocity * task_direction, axis=1)
    return_projection = np.sum(velocity * return_direction, axis=1)

    latest_names = [
        "battery_fraction", "return_distance_over_dmax", "task_distance_over_dmax",
        "task_to_return_distance_ratio", "velocity_toward_task_normalized",
        "velocity_toward_charger_normalized", "horizontal_speed_normalized",
        "vertical_velocity_normalized", "task_clock_fraction", "task_progress",
        "prefix_step_fraction", "lidar_hit_fraction", "lidar_min_range",
        "lidar_q10_range", "lidar_mean_masked_range", "latest_action_correction_norm",
        "previous_contact", "position_x", "position_y", "position_z",
    ]
    latest = np.asarray([
        battery[-1] / CAPACITY,
        return_distance[-1] / D_MAX,
        task_distance[-1] / D_MAX,
        task_distance[-1] / max(return_distance[-1], 1.0),
        task_projection[-1], return_projection[-1],
        np.linalg.norm(velocity[-1, :2]), velocity[-1, 2],
        task_clock[-1] / 4000.0, task_progress[-1], step[-1] / 768.0,
        *lidar[-1], correction[-1], contacts[-1], *(position / WORKSPACE),
    ], np.float64)

    elapsed = max(step[-1] - step[0], 1.0)
    history_names = latest_names + [
        "battery_change_per_step", "return_distance_change_per_step",
        "task_distance_change_per_step", "task_progress_change",
        "task_clock_change_per_step", "mean_velocity_toward_task",
        "q10_velocity_toward_task", "mean_velocity_toward_charger",
        "mean_action_correction", "max_action_correction", "history_contact_rate",
        "mean_lidar_hit_fraction", "min_history_lidar_range",
        "lidar_min_range_change", "task_distance_stall_fraction",
    ]
    task_change = np.diff(task_distance)
    history_extra = np.asarray([
        (battery[-1] - battery[0]) / (CAPACITY * elapsed),
        (return_distance[-1] - return_distance[0]) / (D_MAX * elapsed),
        (task_distance[-1] - task_distance[0]) / (D_MAX * elapsed),
        task_progress[-1] - task_progress[0],
        (task_clock[-1] - task_clock[0]) / elapsed,
        task_projection.mean(), np.quantile(task_projection, 0.1), return_projection.mean(),
        correction.mean(), correction.max(), contacts.mean(), lidar[:, 0].mean(),
        lidar[:, 1].min(), lidar[-1, 1] - lidar[0, 1],
        np.mean(task_change >= -1e-3),
    ], np.float64)
    return latest_names, latest, history_names, np.concatenate((latest, history_extra)), position, task


def privileged_features(position: np.ndarray, task: np.ndarray, world: dict[str, Any]) -> tuple[list[str], np.ndarray]:
    charger = np.asarray(world["charger_position"], np.float64)
    obstacles = list(world["static_obstacles"])
    names, values = [], []
    for label, start, end in (
        ("current_to_task", position, task),
        ("current_to_charger", position, charger),
        ("task_to_charger", task, charger),
    ):
        names.extend([
            f"privileged_{label}_minimum_clearance_over_100m",
            f"privileged_{label}_intersect_fraction",
            f"privileged_{label}_within_50m_fraction",
            f"privileged_{label}_within_100m_fraction",
        ])
        values.extend(segment_geometry(start, end, obstacles))
    return names, np.asarray(values, np.float64)


def fit_predict_ridge(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, alpha: float) -> np.ndarray:
    mean, std = x_train.mean(axis=0), np.maximum(x_train.std(axis=0), 1e-8)
    design = (x_train - mean) / std
    test = (x_test - mean) / std
    target_mean = float(y_train.mean())
    coefficient = np.linalg.solve(design.T @ design + alpha * np.eye(design.shape[1]), design.T @ (y_train - target_mean))
    return test @ coefficient + target_mean


def select_alpha(x: np.ndarray, y: np.ndarray, worlds: np.ndarray) -> float:
    inner = np.asarray([fold_for_world(str(w), outer=False) for w in worlds])
    losses = {}
    for alpha in ALPHAS:
        errors = []
        for fold in range(4):
            train, test = inner != fold, inner == fold
            if not train.any() or not test.any():
                continue
            pred = fit_predict_ridge(x[train], y[train], x[test], alpha)
            errors.extend(np.square(pred - y[test]).tolist())
        losses[alpha] = float(np.mean(errors))
    return min(ALPHAS, key=lambda value: (losses[value], -value))


def crossfit(x: np.ndarray, y: np.ndarray, worlds: np.ndarray) -> np.ndarray:
    outer = np.asarray([fold_for_world(str(w), outer=True) for w in worlds])
    result = np.zeros(len(y), np.float64)
    for fold in range(6):
        train, test = outer != fold, outer == fold
        alpha = select_alpha(x[train], y[train], worlds[train])
        result[test] = fit_predict_ridge(x[train], y[train], x[test], alpha)
    return result


def world_bootstrap_interval(values: np.ndarray, worlds: np.ndarray, rng: np.random.Generator) -> list[float]:
    identities = list(dict.fromkeys(worlds.tolist()))
    groups = [np.flatnonzero(worlds == world) for world in identities]
    means = np.asarray([values[group].mean() for group in groups])
    draws = means[rng.integers(0, len(means), (BOOTSTRAP_DRAWS, len(means)))].mean(axis=1)
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    experiment = args.experiment_root.resolve()
    run = experiment / "pai_dev_raw_v1"
    results_path = run / "results.json"
    results = json.loads(results_path.read_text())

    worlds, steps = [], []
    features: dict[str, list[np.ndarray]] = {"legal_latest_core": [], "legal_history_core": [], "privileged_geometry_augmented": []}
    feature_names: dict[str, list[str]] = {}
    halves = {name: [] for name in ("task", "failure", "contact", "utility")}
    full_components, termination_rows = [], []
    branch_guard_task_increments = Counter()
    artifact_hashes = {"raw_results": file_hash(results_path)}

    for record in results["records"]:
        meta_path = run / "records" / record["json"]
        npz_path = run / "records" / record["npz"]
        if file_hash(meta_path).removeprefix("sha256:") != record["json_sha256"]:
            raise ValueError(f"raw JSON hash mismatch: {meta_path}")
        if file_hash(npz_path).removeprefix("sha256:") != record["npz_sha256"]:
            raise ValueError(f"raw NPZ hash mismatch: {npz_path}")
        meta = json.loads(meta_path.read_text())
        identity = meta["world_identity"]
        world_path = experiment / "worlds" / f"{identity.removeprefix('sha256:')}.json"
        world = json.loads(world_path.read_text())
        with np.load(npz_path, allow_pickle=False) as archive:
            arrays = {name: archive[name].copy() for name in archive.files}
        for anchor in meta["anchors"]:
            if not anchor.get("included"):
                continue
            index = int(anchor["history_index"])
            latest_names, latest, history_names, history, position, task = legal_features(arrays, world, index)
            privileged_names, privileged = privileged_features(position, task, world)
            feature_names["legal_latest_core"] = latest_names
            feature_names["legal_history_core"] = history_names
            feature_names["privileged_geometry_augmented"] = history_names + privileged_names
            features["legal_latest_core"].append(latest)
            features["legal_history_core"].append(history)
            features["privileged_geometry_augmented"].append(np.concatenate((history, privileged)))
            worlds.append(identity)
            steps.append(int(anchor["anchor_step"]))

            by_half = {name: np.zeros((2, 2), np.float64) for name in halves}
            counts = np.zeros((2, 2), np.int64)
            term = {action: Counter() for action in ACTION_ORDER}
            for row in anchor["resamples"]:
                half = int(row["replicate"]) % 2
                action = ACTION_ORDER.index(row["action"])
                outcome = row["outcome"]
                by_half["task"][half, action] += float(outcome["task_increment"])
                by_half["failure"][half, action] += -2.0 * float(outcome["operational_failure"])
                by_half["contact"][half, action] += -0.25 * float(outcome["collision_count"] > 0)
                by_half["utility"][half, action] += float(outcome["utility"])
                counts[half, action] += 1
                term[row["action"]][outcome["termination"]] += 1
                if row["action"] == "C" and outcome["termination"] == "branch_guard":
                    branch_guard_task_increments[int(outcome["task_increment"])] += 1
            if not np.all(counts == 32):
                raise ValueError("unexpected paired half cardinality")
            for name in halves:
                means = by_half[name] / counts
                halves[name].append(means[:, 1] - means[:, 0])
            component = [np.mean(halves[name][-1]) for name in ("task", "failure", "contact")]
            full_components.append(component)
            termination_rows.append({action: dict(term[action]) for action in ACTION_ORDER})

    worlds_array = np.asarray(worlds)
    x_channels = {name: np.asarray(value, np.float64) for name, value in features.items()}
    y_halves = {name: np.asarray(value, np.float64) for name, value in halves.items()}
    components = np.asarray(full_components, np.float64)
    full_utility = y_halves["utility"].mean(axis=1)
    if not np.allclose(components.sum(axis=1), full_utility, atol=1e-12):
        raise ValueError("component decomposition does not reconstruct utility advantage")

    variance = float(np.var(full_utility, ddof=1))
    component_names = ("task_increment", "operational_failure_penalty", "any_contact_penalty")
    factor = {}
    for index, name in enumerate(component_names):
        contribution = components[:, index]
        factor[name] = {
            "mean": float(contribution.mean()),
            "mean_absolute": float(np.mean(np.abs(contribution))),
            "variance": float(np.var(contribution, ddof=1)),
            "heterogeneity_share_covariance_over_total_variance": float(np.cov(contribution, full_utility, ddof=1)[0, 1] / variance),
            "nonzero_anchors": int(np.sum(np.abs(contribution) > 1e-12)),
        }

    negative = full_utility < 0
    term_all = {action: Counter() for action in ACTION_ORDER}
    term_negative = {action: Counter() for action in ACTION_ORDER}
    for is_negative, row in zip(negative, termination_rows):
        for action in ACTION_ORDER:
            term_all[action].update(row[action])
            if is_negative:
                term_negative[action].update(row[action])

    resamples_per_action = len(worlds) * 64
    failure_mode_contributions = {}
    for termination in ("branch_guard", "return_deadline", "energy_exhausted"):
        delta_probability = (
            term_all["C"].get(termination, 0) - term_all["R"].get(termination, 0)
        ) / resamples_per_action
        failure_mode_contributions[termination] = {
            "delta_probability_C_minus_R": float(delta_probability),
            "mean_utility_contribution": float(-2.0 * delta_probability),
        }

    feature_associations = {}
    for channel, x in x_channels.items():
        rows = []
        for index, name in enumerate(feature_names[channel]):
            scale = max(float(x[:, index].std()), 1e-12)
            rows.append({
                "feature": name,
                "negative_minus_positive_standardized_mean": float(
                    (x[negative, index].mean() - x[~negative, index].mean()) / scale
                ),
            })
        feature_associations[channel] = sorted(
            rows,
            key=lambda row: abs(row["negative_minus_positive_standardized_mean"]),
            reverse=True,
        )

    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    diagnostics = {}
    diagnostic_rows: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for channel, x in x_channels.items():
        channel_result = {"feature_count": int(x.shape[1]), "targets": {}}
        for target_name, target_halves in y_halves.items():
            predictions, constants = [], []
            for discovery, evaluation in ((0, 1), (1, 0)):
                prediction = crossfit(x, target_halves[:, discovery], worlds_array)
                constant = np.zeros(len(prediction), np.float64)
                outer = np.asarray([fold_for_world(str(w), outer=True) for w in worlds_array])
                for fold in range(6):
                    train, test = outer != fold, outer == fold
                    constant[test] = target_halves[train, discovery].mean()
                predictions.append(prediction)
                constants.append(constant)
            evals = (target_halves[:, 1], target_halves[:, 0])
            model_error = np.mean([np.square(p - y) for p, y in zip(predictions, evals)], axis=0)
            constant_error = np.mean([np.square(p - y) for p, y in zip(constants, evals)], axis=0)
            improvement = constant_error - model_error
            averaged_prediction = np.mean(predictions, axis=0)
            correlation = float(np.corrcoef(averaged_prediction, target_halves.mean(axis=1))[0, 1])
            row = {
                "mse": float(model_error.mean()),
                "constant_mse": float(constant_error.mean()),
                "constant_minus_model_mse": float(improvement.mean()),
                "constant_minus_model_mse_ci95_world_bootstrap": world_bootstrap_interval(improvement, worlds_array, rng),
                "correlation_with_full_64_pair_target": correlation,
            }
            if target_name == "utility":
                value_differences = []
                for swap, (discovery, evaluation) in enumerate(((0, 1), (1, 0))):
                    action = (predictions[swap] > 0).astype(int)
                    q_eval = np.column_stack((np.zeros(len(action)), target_halves[:, evaluation]))
                    outer = np.asarray([fold_for_world(str(w), outer=True) for w in worlds_array])
                    constant_action = np.zeros(len(action), np.int64)
                    for fold in range(6):
                        train, test = outer != fold, outer == fold
                        constant_action[test] = int(target_halves[train, discovery].mean() > 0)
                    value_differences.append(q_eval[np.arange(len(action)), action] - q_eval[np.arange(len(action)), constant_action])
                value_difference = np.mean(value_differences, axis=0)
                row.update({
                    "selected_utility_minus_crossfit_constant": float(value_difference.mean()),
                    "selected_utility_minus_crossfit_constant_ci95_world_bootstrap": world_bootstrap_interval(value_difference, worlds_array, rng),
                    "decision_c_rate": float(np.mean(np.mean(predictions, axis=0) > 0)),
                })
            diagnostic_rows.setdefault(channel, {})[target_name] = {
                "error": model_error,
                "selected_value_minus_constant": (
                    value_difference if target_name == "utility" else np.zeros(len(model_error))
                ),
            }
            channel_result["targets"][target_name] = row
        diagnostics[channel] = channel_result

    channel_contrasts = {}
    for name, baseline, candidate in (
        ("legal_history_minus_legal_latest", "legal_latest_core", "legal_history_core"),
        ("privileged_geometry_minus_legal_history", "legal_history_core", "privileged_geometry_augmented"),
    ):
        channel_contrasts[name] = {}
        for target_name in y_halves:
            improvement = (
                diagnostic_rows[baseline][target_name]["error"]
                - diagnostic_rows[candidate][target_name]["error"]
            )
            row = {
                "mse_improvement": float(improvement.mean()),
                "mse_improvement_ci95_world_bootstrap": world_bootstrap_interval(
                    improvement, worlds_array, rng
                ),
            }
            if target_name == "utility":
                value_gain = (
                    diagnostic_rows[candidate][target_name]["selected_value_minus_constant"]
                    - diagnostic_rows[baseline][target_name]["selected_value_minus_constant"]
                )
                row.update({
                    "selected_utility_gain": float(value_gain.mean()),
                    "selected_utility_gain_ci95_world_bootstrap": world_bootstrap_interval(
                        value_gain, worlds_array, rng
                    ),
                })
            channel_contrasts[name][target_name] = row

    output = {
        "schema_version": "latent-factor-observability-exploratory-v1",
        "role": "EXPLORATORY_DEV_ONLY_NO_CONFIRM_NO_METHOD_TRAIN",
        "worlds": len(set(worlds)),
        "anchors": len(worlds),
        "advantage_definition": "U(C)-U(R)",
        "utility_decomposition": "delta_task - 2*delta_operational_failure - 0.25*delta_any_contact",
        "factor_decomposition": factor,
        "advantage_sign": {"positive": int(np.sum(full_utility > 0)), "negative": int(np.sum(negative)), "zero": int(np.sum(full_utility == 0))},
        "termination_counts_all_anchors": {action: dict(term_all[action]) for action in ACTION_ORDER},
        "termination_counts_negative_advantage_anchors": {action: dict(term_negative[action]) for action in ACTION_ORDER},
        "C_branch_guard_task_increment_counts": dict(branch_guard_task_increments),
        "operational_failure_mode_decomposition": failure_mode_contributions,
        "descriptive_feature_associations": feature_associations,
        "observability_diagnostics": diagnostics,
        "observability_channel_contrasts": channel_contrasts,
        "feature_channels": {
            name: {
                "features": feature_names[name],
                "deployment_legal": name != "privileged_geometry_augmented",
                "exact_simulator_obstacle_geometry": name == "privileged_geometry_augmented",
            }
            for name in feature_names
        },
        "inference_limits": [
            "Post-branch component outcomes identify additive drivers, not pre-decision causal state.",
            "Ridge diagnostics are exploratory and cross-fitted by physical world; their intervals are not a preregistered confirmation.",
            "The privileged channel is a diagnostic control and is forbidden as a deployable input.",
            "No CONFIRM outcomes were accessed and no actor, critic, recurrent model, or larger history encoder was trained.",
        ],
        "bootstrap": {"draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED, "unit": "physical_world"},
        "input_hashes": artifact_hashes,
        "analysis_source_sha256": file_hash(Path(__file__).resolve()),
        "confirm_accessed": False,
        "method_train_authorized": False,
    }
    atomic_json(args.output.resolve(), output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
