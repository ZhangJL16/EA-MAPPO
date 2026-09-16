#!/usr/bin/env python3
"""Frozen DEV analyzer for paired-advantage Gate H then Gate I."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_cmi_paired_advantage_exploratory import paired_advantages
from scripts.analyze_cmi_v3_dev import (
    ACTION_ORDER, CHANNELS, FAMILIES, atomic_json, crossfit_channel, load_dev,
    save_states, simultaneous_improvement_intervals,
)
from scripts.cmi_v3_models import (
    family_seeds, fit_ridge, fit_standardizer, fold_for_world, select_alpha,
)
from scripts.validate_identification_contract import file_hash

BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 2_026_091_421


def load_halves(run_dir: Path, results: dict) -> tuple[np.ndarray, np.ndarray]:
    values = []
    for record in results["records"]:
        meta = json.loads((run_dir / "records" / record["json"]).read_text())
        for anchor in meta["anchors"]:
            if not anchor.get("included"):
                continue
            by_half = np.zeros((2, 2), np.float64)
            counts = np.zeros((2, 2), np.int64)
            for row in anchor["resamples"]:
                half = int(row["replicate"]) % 2
                action = ACTION_ORDER.index(row["action"])
                by_half[half, action] += float(row["outcome"]["utility"])
                counts[half, action] += 1
            if not np.all(counts == 32):
                raise ValueError("paired-advantage half cardinality changed")
            values.append(by_half / counts)
    halves = np.asarray(values)
    return halves, halves[:, :, 1] - halves[:, :, 0]


def world_means(values: np.ndarray, worlds: np.ndarray) -> np.ndarray:
    identities = list(dict.fromkeys(worlds.tolist()))
    return np.asarray([values[worlds == world].mean(axis=0) for world in identities])


def constant_actions(q_discovery: np.ndarray, worlds: np.ndarray) -> np.ndarray:
    outer = np.asarray([fold_for_world(str(world), outer=True) for world in worlds])
    result = np.zeros(len(worlds), np.int64)
    for fold in range(6):
        train, test = outer != fold, outer == fold
        result[test] = int(np.argmax(q_discovery[train].mean(axis=0)))
    return result


def bootstrap_interval(metric, world_count: int) -> list[float]:
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    draws = np.asarray([
        metric(rng.integers(0, world_count, world_count)) for _ in range(BOOTSTRAP_DRAWS)
    ])
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


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
    results = json.loads((args.run_dir / "results.json").read_text())
    halves_q, halves_delta = load_halves(args.run_dir.resolve(), results)
    full_delta, paired_variance = paired_advantages(args.run_dir.resolve(), results)
    worlds, steps = data["worlds"], data["steps"]
    identities = list(dict.fromkeys(worlds.tolist()))
    if len(full_delta) != len(worlds):
        raise ValueError("paired-advantage target/feature mismatch")

    selector_gain_swaps = []
    contrast_swaps = []
    latest_mse_swaps, constant_mse_swaps = [], []
    full_utility_swaps, latest_utility_swaps, constant_utility_swaps = [], [], []
    action_fractions, action_world_counts = [], []
    family_improvements = {family: [] for family in FAMILIES}
    disagreements = []
    final_predictions = {}
    for discovery, evaluation in ((0, 1), (1, 0)):
        delta_train = halves_delta[:, discovery]
        delta_eval = halves_delta[:, evaluation]
        q_train, q_eval = halves_q[:, discovery], halves_q[:, evaluation]
        duplicated = np.column_stack((delta_train, delta_train))
        fits = {
            channel: crossfit_channel(data["features"][channel], duplicated, data["weights"], worlds)
            for channel in CHANNELS
        }
        predictions = {channel: fits[channel]["ensemble"][:, 0] for channel in CHANNELS}
        constant_prediction = fits["latest"]["constant"][:, 0]
        latest_action = (predictions["latest"] > 0).astype(int)
        full_action = (predictions["full"] > 0).astype(int)
        discovery_action = (delta_train > 0).astype(int)
        constant_action = constant_actions(q_train, worlds)
        row = np.arange(len(worlds))
        selector_gain = q_eval[row, discovery_action] - q_eval[row, constant_action]
        full_utility = q_eval[row, full_action]
        latest_utility = q_eval[row, latest_action]
        constant_utility = q_eval[row, constant_action]
        latest_error = np.square(predictions["latest"] - delta_eval)
        full_error = np.square(predictions["full"] - delta_eval)
        constant_error = np.square(constant_prediction - delta_eval)
        selector_gain_swaps.append(selector_gain)
        contrast_swaps.append(np.column_stack((
            latest_error - full_error,
            constant_error - full_error,
            full_utility - latest_utility,
            full_utility - constant_utility,
        )))
        latest_mse_swaps.append(latest_error)
        constant_mse_swaps.append(constant_error)
        full_utility_swaps.append(full_utility)
        latest_utility_swaps.append(latest_utility)
        constant_utility_swaps.append(constant_utility)
        disagreements.append(full_action != latest_action)
        action_fractions.append({a: float(np.mean(discovery_action == i)) for i, a in enumerate(ACTION_ORDER)})
        action_world_counts.append({
            a: len(set(worlds[discovery_action == i].tolist())) for i, a in enumerate(ACTION_ORDER)
        })
        for family in FAMILIES:
            latest_family = fits["latest"]["family_predictions"][family][:, 0]
            full_family = fits["full"]["family_predictions"][family][:, 0]
            family_improvements[family].append(
                np.square(latest_family - delta_eval) - np.square(full_family - delta_eval)
            )
        final_predictions = predictions

    selector_gain = np.mean(selector_gain_swaps, axis=0)
    contrasts = np.mean(contrast_swaps, axis=0)
    world_selector = world_means(selector_gain, worlds)
    world_contrasts = world_means(contrasts, worlds)
    lower, upper, critical = simultaneous_improvement_intervals(world_contrasts)
    contrast_names = (
        "full_minus_latest_advantage_mse_improvement",
        "full_minus_constant_advantage_mse_improvement",
        "full_minus_latest_selected_utility",
        "full_minus_best_constant_action_utility",
    )
    contrast_result = {
        name: {"estimate": float(world_contrasts[:, i].mean()), "lcb": float(lower[i]), "ucb": float(upper[i])}
        for i, name in enumerate(contrast_names)
    }

    observed_variance = float(np.var(full_delta, ddof=1))
    noise_variance = float(np.mean(paired_variance / 64.0))
    corrected_variance = max(0.0, observed_variance - noise_variance)
    world_delta = [np.flatnonzero(worlds == world) for world in identities]
    variance_ci = bootstrap_interval(
        lambda sample: max(0.0, float(np.var(
            np.concatenate([full_delta[world_delta[i]] for i in sample]), ddof=1
        ) - np.mean(np.concatenate([paired_variance[world_delta[i]] / 64.0 for i in sample])))),
        len(identities),
    )
    selector_ci = bootstrap_interval(lambda sample: float(world_selector[sample].mean()), len(identities))
    paired_se = np.sqrt(paired_variance / 64.0)
    step_counts = {str(step): int(np.sum(steps == step)) for step in (256, 768)}
    raw_pass = bool(
        len(identities) >= 90 and all(v >= 85 for v in step_counts.values())
        and np.isfinite(halves_q).all() and np.median(paired_se) <= 0.10
        and np.quantile(paired_se, 0.95) <= 0.15
    )
    action_mix_pass = all(
        row[a] >= 0.05 and worlds_row[a] >= 5
        for row, worlds_row in zip(action_fractions, action_world_counts)
        for a in ACTION_ORDER
    )
    gate_h_checks = {
        "raw_and_mc_validity": raw_pass,
        "heldout_selector_gain": selector_ci[0] > 0 and float(world_selector.mean()) >= 0.05,
        "both_actions_represented": action_mix_pass,
        "noise_corrected_variance": variance_ci[0] > 0,
    }
    gate_h_passed = all(gate_h_checks.values())

    latest_mse = float(np.mean(latest_mse_swaps))
    constant_mse = float(np.mean(constant_mse_swaps))
    full_mse = latest_mse - contrast_result[contrast_names[0]]["estimate"]
    family_points = {family: float(np.mean(values)) for family, values in family_improvements.items()}
    gate_i_checks = {
        "simultaneous_positive_lcbs": all(contrast_result[name]["lcb"] > 0 for name in contrast_names),
        "full_vs_latest_relative_mse": (latest_mse - full_mse) / latest_mse >= 0.05,
        "full_vs_constant_relative_mse": (constant_mse - full_mse) / constant_mse >= 0.05,
        "full_vs_constant_utility_point": contrast_result[contrast_names[3]]["estimate"] >= 0.05,
        "action_disagreement": float(np.mean(disagreements)) >= 0.05,
        "family_direction": all(value > 0 for value in family_points.values()),
    }

    support = {"schema_version": "paired-advantage-dev-support-v1", "channels": {}}
    support_arrays = {}
    support_pass = True
    for channel in CHANNELS:
        x = data["features"][channel]
        mean, std = fit_standardizer(x)
        standardized = (x - mean) / std
        _, singular, right = np.linalg.svd(standardized, full_matrices=False)
        positive = int(np.sum(singular > max(float(singular[0]) * 1e-10, 1e-12)))
        components = right[: min(32, len(right))]
        representation = standardized @ components.T
        distances = []
        for index, world in enumerate(worlds):
            candidates = np.flatnonzero(worlds != world)
            distances.append(float(np.linalg.norm(representation[candidates] - representation[index], axis=1).min()))
        radius = float(np.quantile(distances, 0.99) * 1.15)
        passed = bool(positive >= 8 and radius > 0 and np.isfinite(representation).all())
        support_pass &= passed
        support["channels"][channel] = {"positive_singular_values": positive, "components": len(components), "confirmation_radius": radius, "passed": passed}
        support_arrays[f"{channel}_mean"] = mean
        support_arrays[f"{channel}_std"] = std
        support_arrays[f"{channel}_components"] = components
    gate_i_checks["finite_and_support"] = support_pass and all(np.isfinite(v).all() for v in final_predictions.values())
    gate_i_passed_blind = all(gate_i_checks.values())

    output.mkdir(parents=True)
    np.savez_compressed(output / "support.npz", **support_arrays)
    atomic_json(output / "support.json", support)
    states = []
    duplicated_full = np.column_stack((full_delta, full_delta))
    for channel in CHANNELS:
        for family in FAMILIES:
            for seed in family_seeds(family):
                alpha = select_alpha(data["features"][channel], duplicated_full, data["weights"], worlds, family=family, seed=seed)
                states.append((channel, family, fit_ridge(data["features"][channel], duplicated_full, data["weights"], family=family, seed=seed, alpha=alpha)))
    manifest = save_states(output / "models.npz", states)
    atomic_json(output / "models.json", {"schema_version": "paired-advantage-dev-models-v1", "states": manifest, "duplicated_scalar_target": True})
    result = {
        "schema_version": "paired-advantage-dev-gates-v1",
        "role": "BLIND_DEV_H_THEN_I",
        "gate_h": {"passed": gate_h_passed, "checks": gate_h_checks, "selector_gain": float(world_selector.mean()), "selector_gain_ci95": selector_ci, "action_fractions": action_fractions, "action_world_counts": action_world_counts, "noise_corrected_variance": corrected_variance, "noise_corrected_variance_ci95": variance_ci},
        "gate_i": {"interpreted": gate_h_passed, "passed_blind": gate_i_passed_blind, "passed": gate_h_passed and gate_i_passed_blind, "checks": gate_i_checks, "contrasts": contrast_result, "latest_mse": latest_mse, "full_mse": full_mse, "constant_mse": constant_mse, "family_full_minus_latest_mse_improvement": family_points, "action_disagreement": float(np.mean(disagreements))},
        "step_anchor_counts": step_counts,
        "worlds": len(identities), "anchors": len(worlds),
        "simultaneous_max_t_critical": critical,
        "bootstrap": {"draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED, "unit": "physical_world"},
        "confirmation_authorized": gate_h_passed and gate_i_passed_blind,
        "method_train_authorized": False,
    }
    atomic_json(output / "gates.json", result)
    artifact_names = ("gates.json", "support.json", "support.npz", "models.json", "models.npz")
    atomic_json(output / "analysis.json", {"schema_version": "paired-advantage-dev-analysis-v1", "gate_h_passed": gate_h_passed, "gate_i_passed": result["gate_i"]["passed"], "artifact_hashes": {name: file_hash(output / name) for name in artifact_names}, "method_train_authorized": False})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
