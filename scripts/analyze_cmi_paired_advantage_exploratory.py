#!/usr/bin/env python3
"""Exploratory paired C-minus-R advantage diagnostic for completed CMI DEV."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_cmi_v3_dev import (
    ACTION_ORDER,
    CHANNELS,
    FAMILIES,
    atomic_json,
    crossfit_channel,
    load_dev,
    weighted_rmse,
)


BOOTSTRAP_SEED = 2_026_091_401
BOOTSTRAP_DRAWS = 20_000


def percentile_interval(values: np.ndarray) -> list[float]:
    return [
        float(np.quantile(values, 0.025, method="linear")),
        float(np.quantile(values, 0.975, method="linear")),
    ]


def paired_advantages(run_dir: Path, results: dict) -> tuple[np.ndarray, np.ndarray]:
    means, variances = [], []
    for record in results["records"]:
        meta = json.loads((run_dir / "records" / record["json"]).read_text())
        for anchor in meta["anchors"]:
            if not anchor.get("included"):
                continue
            pairs: dict[int, dict[str, float]] = {}
            for row in anchor["resamples"]:
                pairs.setdefault(int(row["replicate"]), {})[row["action"]] = float(
                    row["outcome"]["utility"]
                )
            if len(pairs) != 64 or any(set(pair) != set(ACTION_ORDER) for pair in pairs.values()):
                raise ValueError("invalid paired advantage replicate set")
            delta = np.asarray([pairs[index]["C"] - pairs[index]["R"] for index in range(64)])
            means.append(float(delta.mean()))
            variances.append(float(delta.var(ddof=1)))
    return np.asarray(means), np.asarray(variances)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    data = load_dev(args.contract.resolve(), args.run_dir.resolve())
    results = json.loads((args.run_dir / "results.json").read_text())
    delta, paired_variance = paired_advantages(args.run_dir, results)
    target_delta = data["targets"][:, 1] - data["targets"][:, 0]
    if not np.array_equal(delta, target_delta):
        raise ValueError("paired and action-mean advantage differ")

    worlds = data["worlds"]
    unique_worlds = list(dict.fromkeys(worlds.tolist()))
    world_indices = [np.flatnonzero(worlds == world) for world in unique_worlds]
    paired_se = np.sqrt(paired_variance / 64.0)
    observed_variance = float(np.var(delta, ddof=1))
    noise_variance = float(np.mean(paired_variance / 64.0))
    corrected_variance = max(0.0, observed_variance - noise_variance)

    q = data["targets"]
    constant_action = int(np.argmax(q.mean(axis=0)))
    oracle_gain_rows = q.max(axis=1) - q[:, constant_action]

    predictions = {}
    channel_metrics = {}
    # The frozen ridge helper requires two target columns. Duplicate the scalar
    # advantage exactly; the two fitted columns must therefore remain equal.
    duplicated_delta = np.column_stack((delta, delta))
    for channel in CHANNELS:
        fit = crossfit_channel(
            data["features"][channel], duplicated_delta, data["weights"], worlds
        )
        prediction = fit["ensemble"][:, 0]
        if not np.array_equal(fit["ensemble"][:, 0], fit["ensemble"][:, 1]):
            raise ValueError("duplicated scalar advantage predictions diverged")
        predictions[channel] = prediction
        chosen = (prediction > 0).astype(int)
        selected = q[np.arange(len(q)), chosen]
        family = {}
        for name in FAMILIES:
            family_prediction = fit["family_predictions"][name][:, 0]
            family[name] = {
                "rmse": weighted_rmse(
                    np.column_stack((family_prediction, family_prediction)),
                    duplicated_delta,
                    data["weights"],
                ),
                "decision_c_rate": float(np.mean(family_prediction > 0)),
            }
        channel_metrics[channel] = {
            "advantage_rmse": weighted_rmse(
                np.column_stack((prediction, prediction)), duplicated_delta, data["weights"]
            ),
            "constant_advantage_rmse": weighted_rmse(
                fit["constant"], duplicated_delta, data["weights"]
            ),
            "decision_c_rate": float(np.mean(chosen)),
            "selected_utility": float(np.mean(selected)),
            "families": family,
        }

    latest_error = np.square(predictions["latest"] - delta)
    full_error = np.square(predictions["full"] - delta)
    latest_action = (predictions["latest"] > 0).astype(int)
    full_action = (predictions["full"] > 0).astype(int)
    latest_utility = q[np.arange(len(q)), latest_action]
    full_utility = q[np.arange(len(q)), full_action]

    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    boot_oracle, boot_variance, boot_mse, boot_value = [], [], [], []
    for _ in range(BOOTSTRAP_DRAWS):
        sampled_worlds = rng.integers(0, len(world_indices), len(world_indices))
        sampled = np.concatenate([world_indices[index] for index in sampled_worlds])
        sampled_delta = delta[sampled]
        sampled_q = q[sampled]
        selected_constant = int(np.argmax(sampled_q.mean(axis=0)))
        boot_oracle.append(float(np.mean(sampled_q.max(axis=1) - sampled_q[:, selected_constant])))
        boot_variance.append(max(0.0, float(
            np.var(sampled_delta, ddof=1) - np.mean(paired_variance[sampled] / 64.0)
        )))
        boot_mse.append(float(np.mean(latest_error[sampled] - full_error[sampled])))
        boot_value.append(float(np.mean(full_utility[sampled] - latest_utility[sampled])))

    confident_positive = int(np.sum(delta - 1.96 * paired_se > 0))
    confident_negative = int(np.sum(delta + 1.96 * paired_se < 0))
    output = {
        "schema_version": "cmi-paired-advantage-exploratory-v1",
        "role": "POST_HOC_DESIGN_AND_POWER_DIAGNOSTIC_ONLY",
        "worlds": len(unique_worlds),
        "anchors": len(delta),
        "paired_resamples_per_anchor": 64,
        "advantage_definition": "U(C)-U(R) under paired disturbance seed",
        "paired_mc_se": {
            "median": float(np.median(paired_se)),
            "p95": float(np.quantile(paired_se, 0.95, method="linear")),
        },
        "heterogeneity": {
            "mean_advantage": float(np.mean(delta)),
            "observed_anchor_variance": observed_variance,
            "mean_mc_noise_variance": noise_variance,
            "noise_corrected_anchor_variance": corrected_variance,
            "noise_corrected_variance_ci95_world_bootstrap": percentile_interval(np.asarray(boot_variance)),
            "anchors_confident_positive_approx95": confident_positive,
            "anchors_confident_negative_approx95": confident_negative,
            "oracle_gain_over_best_constant_action": float(np.mean(oracle_gain_rows)),
            "oracle_gain_ci95_world_bootstrap": percentile_interval(np.asarray(boot_oracle)),
            "best_constant_action": ACTION_ORDER[constant_action],
        },
        "predictability": {
            "channels": channel_metrics,
            "full_minus_latest_mse_improvement": float(np.mean(latest_error - full_error)),
            "full_minus_latest_mse_improvement_ci95_world_bootstrap": percentile_interval(np.asarray(boot_mse)),
            "full_minus_latest_selected_utility": float(np.mean(full_utility - latest_utility)),
            "full_minus_latest_selected_utility_ci95_world_bootstrap": percentile_interval(np.asarray(boot_value)),
            "action_disagreement": float(np.mean(full_action != latest_action)),
        },
        "bootstrap": {"unit": "physical_world", "draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED},
        "formal_claim_authorized": False,
        "method_train_authorized": False,
    }
    atomic_json(args.output.resolve(), output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
