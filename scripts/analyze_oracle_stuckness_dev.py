#!/usr/bin/env python3
"""Registered whole-world analysis; refuses incomplete 24-world DEV grids."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


METHODS = tuple(range(5))
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 20260916


def load_complete_grid(raw: Path, contract: dict[str, Any]) -> dict[int, dict[int, dict[str, Any]]]:
    dev = [row for row in contract["worlds"] if row["split"] == "DEV"]
    if len(dev) != 24:
        raise RuntimeError("registered DEV set is not 24 worlds")
    grid: dict[int, dict[int, dict[str, Any]]] = {}
    for world in dev:
        index = int(world["world_index"])
        directory = raw / f"world_{index:03d}_{int(world['world_seed'])}"
        rows: dict[int, dict[str, Any]] = {}
        for method in METHODS:
            path = directory / f"M{method}.json"
            if not path.exists():
                raise RuntimeError(f"formal analysis forbidden: missing {path}")
            row = json.loads(path.read_text())
            if not row.get("completed") or row["split"] != "DEV":
                raise RuntimeError(f"formal analysis forbidden: incomplete/invalid {path}")
            if row["physical_world_identity"] != world["physical_world_identity"]:
                raise RuntimeError(f"identity mismatch: {path}")
            if float(row["configured_denominator_seconds"]) != 7200.0:
                raise RuntimeError(f"denominator mismatch: {path}")
            rows[method] = row
        grid[index] = rows
    return grid


def cross_fitted_simple(grid: dict[int, dict[int, dict[str, Any]]]) -> tuple[np.ndarray, list[int]]:
    indices = sorted(grid)
    values, choices = [], []
    for held_out in indices:
        training = [i for i in indices if i != held_out]
        means = {
            method: np.mean([grid[i][method]["tasks_per_hour"] for i in training])
            for method in (1, 2, 3)
        }
        chosen = min(means, key=lambda m: (-means[m], m))
        choices.append(chosen)
        values.append(float(grid[held_out][chosen]["tasks_per_hour"]))
    return np.asarray(values), choices


def simultaneous_max_t(contrasts: np.ndarray) -> dict[str, Any]:
    """Studentized paired world bootstrap for two pre-registered contrasts."""
    n, k = contrasts.shape
    means = contrasts.mean(axis=0)
    ses = contrasts.std(axis=0, ddof=1) / np.sqrt(n)
    if np.any(ses <= 0):
        raise RuntimeError("degenerate contrast prevents max-t interval")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    max_t = np.empty(BOOTSTRAP_DRAWS)
    for draw in range(BOOTSTRAP_DRAWS):
        sample = contrasts[rng.integers(0, n, n)]
        sample_se = sample.std(axis=0, ddof=1) / np.sqrt(n)
        sample_se = np.where(sample_se > 0, sample_se, ses)
        max_t[draw] = np.max(np.abs((sample.mean(axis=0) - means) / sample_se))
    critical = float(np.quantile(max_t, 0.95))
    return {
        "means": means.tolist(),
        "ses": ses.tolist(),
        "critical_value": critical,
        "lower": (means - critical * ses).tolist(),
        "upper": (means + critical * ses).tolist(),
        "draws": BOOTSTRAP_DRAWS,
        "seed": BOOTSTRAP_SEED,
    }


def analyze(grid: dict[int, dict[int, dict[str, Any]]]) -> dict[str, Any]:
    indices = sorted(grid)
    base = np.asarray([grid[i][0]["tasks_per_hour"] for i in indices])
    oracle = np.asarray([grid[i][4]["tasks_per_hour"] for i in indices])
    simple, choices = cross_fitted_simple(grid)
    contrasts = np.column_stack((oracle - base, oracle - simple))
    intervals = simultaneous_max_t(contrasts)
    oracle_gain = float(np.mean(oracle - base))
    simple_gain = float(np.mean(simple - base))
    capture = simple_gain / oracle_gain if oracle_gain > 0 else float("inf")
    interventions = np.asarray([
        grid[i][4]["intervention_counts"]["oracle_return"] for i in indices
    ])
    base_dwell = np.asarray([grid[i][0]["unproductive_task_dwell_seconds"] for i in indices])
    oracle_dwell = np.asarray([grid[i][4]["unproductive_task_dwell_seconds"] for i in indices])
    dwell_reduction = 1.0 - float(oracle_dwell.mean() / base_dwell.mean()) if base_dwell.mean() else 0.0
    added_tasks = np.asarray([grid[i][4]["tasks_completed"] - grid[i][0]["tasks_completed"] for i in indices])
    avoided_energy_failures = np.asarray([
        int(grid[i][0]["energy_exhausted"]) - int(grid[i][4]["energy_exhausted"]) for i in indices
    ])
    failure_accounting_fraction = (
        float(np.maximum(avoided_energy_failures, 0).sum() / np.maximum(added_tasks, 0).sum())
        if np.maximum(added_tasks, 0).sum() else float("inf")
    )
    gate_o = bool(
        intervals["lower"][0] > 0
        and oracle_gain / float(base.mean()) >= 0.10
        and float(np.mean(oracle > base)) >= 0.60
        and int(np.sum(interventions > 0)) >= 5
    )
    collision_delta = float(np.mean([
        grid[i][4]["collision_count"] - grid[i][0]["collision_count"] for i in indices
    ]))
    gate_m = bool(gate_o and dwell_reduction >= 0.20 and failure_accounting_fraction <= 0.50)
    gate_n = bool(gate_m and intervals["lower"][1] > 0 and capture < 0.80)
    return {
        "schema_version": "oracle-stuckness-analysis-v1",
        "worlds": len(indices),
        "simultaneous_max_t": intervals,
        "oracle_minus_base": {
            "mean": oracle_gain,
            "median": float(np.median(oracle - base)),
            "mean_relative_gain": oracle_gain / float(base.mean()),
            "positive_world_fraction": float(np.mean(oracle > base)),
        },
        "oracle_minus_cross_fitted_simple": {
            "mean": float(np.mean(oracle - simple)),
            "choices": choices,
            "capture_fraction": capture,
        },
        "mechanism": {
            "unproductive_dwell_reduction_fraction": dwell_reduction,
            "energy_failure_accounting_fraction": failure_accounting_fraction,
            "mean_collision_delta": collision_delta,
        },
        "gates": {"O": gate_o, "M": gate_m, "N": gate_n},
        "warning": "M3 is an experiment-only action outside the original Continue/Return space.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    contract = json.loads(args.contract.read_text())
    result = analyze(load_complete_grid(args.raw, contract))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
