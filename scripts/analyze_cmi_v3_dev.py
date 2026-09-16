#!/usr/bin/env python3
"""Frozen CMI-v3 DEV adequacy analysis; never emits an information verdict."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.cmi_v3_models import (
    ACTION_ORDER,
    CHANNELS,
    FAMILIES,
    OUTER_FOLDS,
    RFF_SEEDS,
    RidgeState,
    calibration_worlds,
    family_seeds,
    fold_for_world,
    legal_features,
    fit_ridge,
    fit_standardizer,
    predict_ridge,
    select_alpha,
)
from scripts.validate_identification_contract import file_hash


BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED = 2_026_091_224
MIN_RELATIVE_RMSE_REDUCTION = 0.05
MAX_FAMILY_RMSE_SPREAD = 0.05
MAX_FAMILY_ACTION_DISAGREEMENT = 0.20
MIN_CALIBRATION_COVERAGE = 0.85
MC_SE_MEDIAN_MAX = 0.10
MC_SE_P95_MAX = 0.20
MIN_WORLDS = 90
MIN_ANCHORS_PER_STEP = 85


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def conformal_quantile(residuals: np.ndarray, coverage: float = 0.90) -> float:
    values = np.sort(np.asarray(residuals, np.float64).reshape(-1))
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("invalid CMI-v3 calibration residuals")
    rank = min(len(values) - 1, int(math.ceil((len(values) + 1) * coverage)) - 1)
    return float(values[rank])


def world_weights(worlds: np.ndarray) -> np.ndarray:
    counts = Counter(worlds.tolist())
    weight = np.asarray([1.0 / counts[str(world)] for world in worlds], np.float64)
    return weight * (len(weight) / weight.sum())


def load_dev(contract_path: Path, run_dir: Path) -> dict[str, Any]:
    contract = json.loads(contract_path.read_text())
    pre = contract["pre_h"]
    if pre.get("contract_id") != "conditional-history-mc-q-20260912-v3":
        raise ValueError("not the frozen CMI-v3 contract")
    status = json.loads((run_dir / "status.json").read_text())
    results = json.loads((run_dir / "results.json").read_text())
    assigned = pre["master_splits"]["CMI_V3_DEV"]
    if status.get("status") != "COMPLETE_AWAITING_CMI_V3_DEV_ANALYSIS":
        raise ValueError("CMI-v3 DEV is not complete")
    if results.get("completed") != len(assigned):
        raise ValueError("CMI-v3 DEV result count is incomplete")
    if [row["world_identity"] for row in results["records"]] != assigned:
        raise ValueError("CMI-v3 DEV record order differs from assignment")
    expected_replicates = pre["resamples_per_anchor_action"]["CMI_V3_DEV"]
    feature_rows: dict[str, list[np.ndarray]] = {channel: [] for channel in CHANNELS}
    targets: list[list[float]] = []
    sample_variances: list[list[float]] = []
    standard_errors: list[list[float]] = []
    worlds: list[str] = []
    steps: list[int] = []
    for record in results["records"]:
        json_path = run_dir / "records" / record["json"]
        npz_path = run_dir / "records" / record["npz"]
        if file_hash(json_path).removeprefix("sha256:") != record["json_sha256"]:
            raise ValueError("CMI-v3 DEV JSON record hash mismatch")
        if file_hash(npz_path).removeprefix("sha256:") != record["npz_sha256"]:
            raise ValueError("CMI-v3 DEV NPZ record hash mismatch")
        meta = json.loads(json_path.read_text())
        if meta.get("world_identity") != record["world_identity"]:
            raise ValueError("CMI-v3 world identity mismatch")
        included = [anchor for anchor in meta["anchors"] if anchor.get("included")]
        with np.load(npz_path, allow_pickle=False) as archive:
            features = legal_features({name: archive[name] for name in archive.files})
        if any(len(features[channel]) != len(included) for channel in CHANNELS):
            raise ValueError("CMI-v3 legal feature/anchor mismatch")
        for index, anchor in enumerate(included):
            by_action = {action: [] for action in ACTION_ORDER}
            paired_seed = {replicate: {} for replicate in range(expected_replicates)}
            for row in anchor["resamples"]:
                action = row.get("action")
                replicate = row.get("replicate")
                if action not in ACTION_ORDER or not isinstance(replicate, int):
                    raise ValueError("invalid CMI-v3 action/replicate")
                if row.get("block") != "evaluation":
                    raise ValueError("CMI-v3 DEV contains a non-evaluation block")
                if not 0 <= row["outcome"]["collision_count"] <= row["policy_steps"]:
                    raise ValueError("invalid unified CMI-v3 collision count")
                by_action[action].append(float(row["outcome"]["utility"]))
                paired_seed[replicate][action] = row["disturbance_seed"]
            if any(len(by_action[action]) != expected_replicates for action in ACTION_ORDER):
                raise ValueError("CMI-v3 DEV resample count mismatch")
            if any(
                set(pair) != set(ACTION_ORDER) or pair["R"] != pair["C"]
                for pair in paired_seed.values()
            ):
                raise ValueError("CMI-v3 DEV common-random-number pairing mismatch")
            for channel in CHANNELS:
                feature_rows[channel].append(features[channel][index])
            means = [float(np.mean(by_action[action])) for action in ACTION_ORDER]
            variances = [float(np.var(by_action[action], ddof=1)) for action in ACTION_ORDER]
            targets.append(means)
            sample_variances.append(variances)
            standard_errors.append([math.sqrt(value / expected_replicates) for value in variances])
            worlds.append(meta["world_identity"])
            steps.append(int(anchor["anchor_step"]))
    return {
        "contract": contract,
        "features": {
            channel: np.asarray(feature_rows[channel], np.float64) for channel in CHANNELS
        },
        "targets": np.asarray(targets, np.float64),
        "sample_variances": np.asarray(sample_variances, np.float64),
        "standard_errors": np.asarray(standard_errors, np.float64),
        "worlds": np.asarray(worlds),
        "steps": np.asarray(steps, np.int64),
        "weights": world_weights(np.asarray(worlds)),
    }


def fit_predict_family(
    x_train: np.ndarray,
    y_train: np.ndarray,
    weights_train: np.ndarray,
    worlds_train: np.ndarray,
    x_test: np.ndarray,
    family: str,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    predictions, selected = [], []
    for seed in family_seeds(family):
        alpha = select_alpha(
            x_train, y_train, weights_train, worlds_train,
            family=family, seed=seed,
        )
        state = fit_ridge(
            x_train, y_train, weights_train,
            family=family, seed=seed, alpha=alpha,
        )
        predictions.append(predict_ridge(state, x_test))
        selected.append({"seed": seed, "alpha": alpha})
    return np.mean(predictions, axis=0), selected


def crossfit_channel(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    worlds: np.ndarray,
) -> dict[str, Any]:
    outer = np.asarray([fold_for_world(str(world), outer=True) for world in worlds])
    family_predictions = {
        family: np.zeros_like(y, dtype=np.float64) for family in FAMILIES
    }
    ensemble = np.zeros_like(y, dtype=np.float64)
    constant = np.zeros_like(y, dtype=np.float64)
    half_width = np.zeros_like(y, dtype=np.float64)
    constant_half_width = np.zeros_like(y, dtype=np.float64)
    fold_records: list[dict[str, Any]] = []
    for fold in range(OUTER_FOLDS):
        train, test = outer != fold, outer == fold
        if not train.any() or not test.any():
            raise ValueError("empty CMI-v3 outer world fold")
        constant[test] = np.average(y[train], axis=0, weights=weights[train])
        family_fold: dict[str, np.ndarray] = {}
        selections: dict[str, list[dict[str, Any]]] = {}
        for family in FAMILIES:
            family_fold[family], selections[family] = fit_predict_family(
                x[train], y[train], weights[train], worlds[train], x[test], family
            )
            family_predictions[family][test] = family_fold[family]
        ensemble[test] = np.mean(list(family_fold.values()), axis=0)

        calibration_ids = calibration_worlds(worlds[train])
        calibration = train & np.isin(worlds, list(calibration_ids))
        fit = train & ~calibration
        if len(set(worlds[fit].tolist())) < 20 or not calibration.any():
            raise ValueError("insufficient CMI-v3 split-calibration worlds")
        calibration_family = []
        for family in FAMILIES:
            prediction, _ = fit_predict_family(
                x[fit], y[fit], weights[fit], worlds[fit], x[calibration], family
            )
            calibration_family.append(prediction)
        calibration_prediction = np.mean(calibration_family, axis=0)
        model_width = conformal_quantile(
            np.abs(calibration_prediction - y[calibration])
        )
        baseline_fit = np.average(y[fit], axis=0, weights=weights[fit])
        baseline_width = conformal_quantile(
            np.abs(y[calibration] - baseline_fit)
        )
        half_width[test] = model_width
        constant_half_width[test] = baseline_width
        fold_records.append({
            "fold": fold,
            "train_worlds": sorted(set(worlds[train].tolist())),
            "test_worlds": sorted(set(worlds[test].tolist())),
            "calibration_worlds": sorted(calibration_ids),
            "selected_ridge": selections,
            "conformal_half_width": model_width,
            "constant_conformal_half_width": baseline_width,
        })
    return {
        "family_predictions": family_predictions,
        "ensemble": ensemble,
        "constant": constant,
        "half_width": half_width,
        "constant_half_width": constant_half_width,
        "fold_records": fold_records,
    }


def weighted_rmse(prediction: np.ndarray, target: np.ndarray, weights: np.ndarray) -> float:
    repeated = np.repeat(weights, target.shape[1]).reshape(target.shape)
    return float(np.sqrt(np.average(np.square(prediction - target), weights=repeated)))


def by_world(values: np.ndarray, worlds: np.ndarray) -> tuple[list[str], np.ndarray]:
    identities = list(dict.fromkeys(worlds.tolist()))
    return identities, np.asarray([values[worlds == world].mean() for world in identities])


def simultaneous_improvement_intervals(
    improvements: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    estimate = improvements.mean(axis=0)
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    centered = improvements - estimate
    bootstrap = np.empty((BOOTSTRAP_REPLICATES, improvements.shape[1]), np.float64)
    for draw in range(BOOTSTRAP_REPLICATES):
        multiplier = rng.standard_normal(len(improvements))
        bootstrap[draw] = estimate + np.mean(centered * multiplier[:, None], axis=0)
    scale = bootstrap.std(axis=0, ddof=1)
    safe = np.where(scale > 0, scale, 1.0)
    standardized = np.abs((bootstrap - bootstrap.mean(axis=0)) / safe)
    standardized[:, scale == 0] = 0.0
    critical = float(np.quantile(standardized.max(axis=1), 0.95, method="linear"))
    return estimate - critical * scale, estimate + critical * scale, critical


def save_states(path: Path, states: list[tuple[str, str, RidgeState]]) -> list[dict[str, Any]]:
    arrays: dict[str, np.ndarray] = {}
    manifest: list[dict[str, Any]] = []
    for index, (channel, family, state) in enumerate(states):
        prefix = f"state_{index:02d}"
        arrays[f"{prefix}_mean"] = state.mean
        arrays[f"{prefix}_std"] = state.std
        arrays[f"{prefix}_target_mean"] = state.target_mean
        arrays[f"{prefix}_coefficient"] = state.coefficient
        if state.rff_weight is not None:
            arrays[f"{prefix}_rff_weight"] = state.rff_weight
            arrays[f"{prefix}_rff_phase"] = state.rff_phase
        manifest.append({
            "prefix": prefix,
            "channel": channel,
            "family": family,
            "seed": state.seed,
            "alpha": state.alpha,
            "has_rff": state.rff_weight is not None,
        })
    np.savez_compressed(path, **arrays)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    data = load_dev(args.contract.resolve(), args.run_dir.resolve())
    output.mkdir(parents=True)
    y, worlds, weights = data["targets"], data["worlds"], data["weights"]
    channel_results = {
        channel: crossfit_channel(data["features"][channel], y, weights, worlds)
        for channel in CHANNELS
    }

    channel_metrics: dict[str, Any] = {}
    improvement_columns = []
    family_metrics: dict[str, dict[str, Any]] = {family: {} for family in FAMILIES}
    for channel in CHANNELS:
        result = channel_results[channel]
        baseline_rmse = weighted_rmse(result["constant"], y, weights)
        model_rmse = weighted_rmse(result["ensemble"], y, weights)
        point_error = np.square(result["constant"] - y).mean(axis=1) - np.square(
            result["ensemble"] - y
        ).mean(axis=1)
        identities, world_improvement = by_world(point_error, worlds)
        improvement_columns.append(world_improvement)
        coverage = float(np.mean(np.abs(result["ensemble"] - y) <= result["half_width"]))
        family_choice = {}
        for family in FAMILIES:
            prediction = result["family_predictions"][family]
            family_rmse = weighted_rmse(prediction, y, weights)
            family_improvement = float(np.average(
                np.square(result["constant"] - y).mean(axis=1)
                - np.square(prediction - y).mean(axis=1),
                weights=weights,
            ))
            family_metrics[family][channel] = {
                "rmse": family_rmse,
                "mse_improvement": family_improvement,
            }
            family_choice[family] = np.argmax(prediction, axis=1)
        disagreement = float(np.mean(family_choice[FAMILIES[0]] != family_choice[FAMILIES[1]]))
        family_rmses = [family_metrics[family][channel]["rmse"] for family in FAMILIES]
        channel_metrics[channel] = {
            "constant_crossfit_rmse": baseline_rmse,
            "ensemble_crossfit_rmse": model_rmse,
            "relative_rmse_reduction": (baseline_rmse - model_rmse) / baseline_rmse,
            "calibration_coverage_90": coverage,
            "mean_conformal_half_width": float(result["half_width"].mean()),
            "mean_constant_conformal_half_width": float(result["constant_half_width"].mean()),
            "family_rmse_spread": max(family_rmses) - min(family_rmses),
            "family_action_disagreement": disagreement,
            "worlds": len(identities),
        }
    improvements = np.stack(improvement_columns, axis=1)
    lower, upper, critical = simultaneous_improvement_intervals(improvements)
    for index, channel in enumerate(CHANNELS):
        channel_metrics[channel]["world_mse_improvement"] = float(improvements[:, index].mean())
        channel_metrics[channel]["simultaneous_lcb"] = float(lower[index])
        channel_metrics[channel]["simultaneous_ucb"] = float(upper[index])

    final_states: list[tuple[str, str, RidgeState]] = []
    support_meta: dict[str, Any] = {
        "schema_version": "cmi-v3-dev-support-v1", "channels": {}
    }
    support_arrays: dict[str, np.ndarray] = {}
    support_checks: dict[str, bool] = {}
    for channel in CHANNELS:
        x = data["features"][channel]
        for family in FAMILIES:
            for seed in family_seeds(family):
                alpha = select_alpha(x, y, weights, worlds, family=family, seed=seed)
                final_states.append((
                    channel, family,
                    fit_ridge(x, y, weights, family=family, seed=seed, alpha=alpha),
                ))
        mean, std = fit_standardizer(x)
        standardized = (x - mean) / std
        _, singular, right = np.linalg.svd(standardized, full_matrices=False)
        positive = int(np.sum(singular > max(float(singular[0]) * 1e-10, 1e-12)))
        components = right[: min(32, len(right))]
        representation = standardized @ components.T
        distances = []
        for index, world in enumerate(worlds):
            candidates = np.flatnonzero(worlds != world)
            distances.append(float(np.linalg.norm(
                representation[candidates] - representation[index], axis=1
            ).min()))
        radius = float(np.quantile(distances, 0.99, method="linear") * 1.15)
        passed = bool(
            np.isfinite(representation).all() and positive >= 8
            and math.isfinite(radius) and radius > 0
        )
        support_checks[channel] = passed
        support_meta["channels"][channel] = {
            "positive_singular_values": positive,
            "components": int(len(components)),
            "confirmation_radius": radius,
            "passed": passed,
        }
        support_arrays[f"{channel}_mean"] = mean
        support_arrays[f"{channel}_std"] = std
        support_arrays[f"{channel}_components"] = components
        support_arrays[f"{channel}_representation"] = representation
        support_arrays[f"{channel}_worlds"] = worlds.astype("U71")
        support_arrays[f"{channel}_steps"] = data["steps"]
        support_arrays[f"{channel}_loo_distances"] = np.asarray(distances)

    model_manifest = save_states(output / "frozen_models.npz", final_states)
    atomic_json(output / "frozen_models.json", {
        "schema_version": "cmi-v3-frozen-models-v1",
        "action_order": list(ACTION_ORDER),
        "exact_tie_action": "R",
        "states": model_manifest,
    })
    np.savez_compressed(output / "support_calibration.npz", **support_arrays)
    atomic_json(output / "support_calibration.json", support_meta)
    calibration = {
        "schema_version": "cmi-v3-dev-calibration-v1",
        "channels": {
            channel: {
                "coverage_90": channel_metrics[channel]["calibration_coverage_90"],
                "mean_half_width": channel_metrics[channel]["mean_conformal_half_width"],
                "constant_mean_half_width": channel_metrics[channel]["mean_constant_conformal_half_width"],
                "confirmation_half_width": max(
                    row["conformal_half_width"]
                    for row in channel_results[channel]["fold_records"]
                ),
                "folds": channel_results[channel]["fold_records"],
            }
            for channel in CHANNELS
        },
    }
    atomic_json(output / "calibration.json", calibration)

    se = data["standard_errors"]
    mc_checks = {
        action: {
            "median": float(np.median(se[:, index])),
            "p95": float(np.quantile(se[:, index], 0.95, method="linear")),
            "passed": bool(
                np.median(se[:, index]) <= MC_SE_MEDIAN_MAX
                and np.quantile(se[:, index], 0.95, method="linear") <= MC_SE_P95_MAX
            ),
        }
        for index, action in enumerate(ACTION_ORDER)
    }
    step_counts = {str(step): int(np.sum(data["steps"] == step)) for step in (256, 768)}
    contributing_worlds = len(set(worlds.tolist()))
    raw_target_passed = bool(
        np.isfinite(y).all() and np.isfinite(se).all()
        and all(value["passed"] for value in mc_checks.values())
        and contributing_worlds >= MIN_WORLDS
        and all(value >= MIN_ANCHORS_PER_STEP for value in step_counts.values())
    )
    improvement_passed = all(
        channel_metrics[channel]["simultaneous_lcb"] > 0
        and channel_metrics[channel]["relative_rmse_reduction"] >= MIN_RELATIVE_RMSE_REDUCTION
        for channel in CHANNELS
    )
    stability_passed = all(
        all(family_metrics[family][channel]["mse_improvement"] > 0 for family in FAMILIES)
        and channel_metrics[channel]["family_rmse_spread"] <= MAX_FAMILY_RMSE_SPREAD
        and channel_metrics[channel]["family_action_disagreement"] <= MAX_FAMILY_ACTION_DISAGREEMENT
        for channel in CHANNELS
    )
    calibration_passed = all(
        channel_metrics[channel]["calibration_coverage_90"] >= MIN_CALIBRATION_COVERAGE
        and channel_metrics[channel]["mean_conformal_half_width"]
        <= channel_metrics[channel]["mean_constant_conformal_half_width"]
        for channel in CHANNELS
    )
    uncertainty_passed = bool(
        np.isfinite(improvements).all() and contributing_worlds >= MIN_WORLDS
        and 1.0 / contributing_worlds <= 1.0 / 90.0
        and math.isfinite(critical)
    )
    checks = {
        "raw_and_mc_target_validity": raw_target_passed,
        "significant_constant_q_improvement": improvement_passed,
        "cross_model_stability": stability_passed,
        "heldout_calibration": calibration_passed,
        "support": all(support_checks.values()),
        "world_level_uncertainty": uncertainty_passed,
    }
    adequacy = {
        "schema_version": "cmi-v3-dev-adequacy-v1",
        "passed": all(checks.values()),
        "checks": checks,
        "mc_standard_error": mc_checks,
        "step_anchor_counts": step_counts,
        "contributing_worlds": contributing_worlds,
        "channel_metrics": channel_metrics,
        "family_metrics": family_metrics,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "simultaneous_max_t_critical": critical,
        "failure_effect": "CMI_V3_CONFIRM remains unopened; no absence claim and no METHOD_TRAIN",
    }
    atomic_json(output / "adequacy.json", adequacy)
    artifact_names = (
        "adequacy.json", "calibration.json", "support_calibration.json",
        "support_calibration.npz", "frozen_models.json", "frozen_models.npz",
    )
    atomic_json(output / "analysis.json", {
        "schema_version": "cmi-v3-dev-analysis-v1",
        "role": "DEVELOPMENT_ESTIMATOR_ADEQUACY_ONLY",
        "worlds": contributing_worlds,
        "anchors": len(worlds),
        "adequacy_passed": adequacy["passed"],
        "artifact_hashes": {name: file_hash(output / name) for name in artifact_names},
        "method_train_authorized": False,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
