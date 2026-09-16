#!/usr/bin/env python3
"""Frozen v1a four-contrast analysis; refuses incomplete 24-world grids."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 20260916


def load_grid(raw: Path, contract: dict[str, Any]) -> dict[int, dict[int, dict[str, Any]]]:
    worlds = [w for w in contract["worlds"] if w["split"] == "DEV"]
    if len(worlds) != 24:
        raise RuntimeError("v1a DEV registry is not 24 worlds")
    grid = {}
    for world in worlds:
        i = int(world["world_index"])
        directory = raw / f"world_{i:03d}_{world['world_seed']}"
        rows = {}
        for method in range(5):
            path = directory / f"M{method}.json"
            if not path.exists():
                raise RuntimeError(f"formal v1a analysis forbidden: missing {path}")
            row = json.loads(path.read_text())
            if (not row.get("completed") or row.get("schema_version") !=
                    "oracle-stuckness-method-result-v1a"):
                raise RuntimeError(f"invalid v1a result: {path}")
            if row["physical_world_identity"] != world["physical_world_identity"]:
                raise RuntimeError(f"identity mismatch: {path}")
            if row["configured_denominator_seconds"] != 7200.0:
                raise RuntimeError(f"denominator mismatch: {path}")
            if row["tasks_completed"] != len(row["task_completion_timestamps"]):
                raise RuntimeError(f"timestamp/count mismatch: {path}")
            rows[method] = row
        grid[i] = rows
    return grid


def simultaneous_max_t(contrasts: np.ndarray) -> dict[str, Any]:
    n = contrasts.shape[0]
    means = contrasts.mean(axis=0)
    ses = contrasts.std(axis=0, ddof=1) / np.sqrt(n)
    if np.any(ses <= 0):
        raise RuntimeError("degenerate registered contrast")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    maximum = np.empty(BOOTSTRAP_DRAWS)
    for draw in range(BOOTSTRAP_DRAWS):
        sample = contrasts[rng.integers(0, n, n)]
        sample_se = sample.std(axis=0, ddof=1) / np.sqrt(n)
        sample_se = np.where(sample_se > 0, sample_se, ses)
        maximum[draw] = np.max(np.abs((sample.mean(axis=0) - means) / sample_se))
    critical = float(np.quantile(maximum, .95))
    return {"labels": ["M4-M0", "M4-M1", "M4-M2", "M4-M3"],
            "means": means.tolist(), "ses": ses.tolist(), "critical_value": critical,
            "lower": (means - critical * ses).tolist(),
            "upper": (means + critical * ses).tolist(),
            "draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED}


def count_at(row: dict[str, Any], horizon: float) -> int:
    return sum(float(t) <= horizon for t in row["task_completion_timestamps"])


def survival_explanation(grid: dict[int, dict[int, dict[str, Any]]]) -> tuple[float | None, float]:
    total_delta, common_delta = [], []
    for i in sorted(grid):
        base, oracle = grid[i][0], grid[i][4]
        common = min(base["time_alive_seconds"], oracle["time_alive_seconds"])
        total_delta.append(count_at(oracle, 7200.0) - count_at(base, 7200.0))
        common_delta.append(count_at(oracle, common) - count_at(base, common))
    total_delta = np.asarray(total_delta)
    common_delta = np.asarray(common_delta)
    denominator = float(np.maximum(total_delta, 0).sum())
    if denominator <= 0:
        return None, denominator
    fraction = float(np.maximum(total_delta - common_delta, 0).sum()) / denominator
    return fraction, denominator


def descriptive_loo(grid: dict[int, dict[int, dict[str, Any]]]) -> dict[str, Any]:
    indices = sorted(grid)
    choices, held_out = [], []
    for target in indices:
        training = [i for i in indices if i != target]
        means = {m: np.mean([grid[i][m]["tasks_per_hour"] for i in training]) for m in (1, 2, 3)}
        chosen = min(means, key=lambda m: (-means[m], m))
        choices.append(chosen)
        held_out.append(grid[target][chosen]["tasks_per_hour"])
    return {"gate_role": "descriptive_only", "choices": choices,
            "mean_held_out_tasks_per_hour": float(np.mean(held_out))}


def analyze(grid: dict[int, dict[int, dict[str, Any]]]) -> dict[str, Any]:
    indices = sorted(grid)
    y = np.asarray([[grid[i][m]["tasks_per_hour"] for m in range(5)] for i in indices])
    contrasts = np.column_stack([y[:, 4] - y[:, m] for m in range(4)])
    intervals = simultaneous_max_t(contrasts)
    oracle_gain = float(np.mean(y[:, 4] - y[:, 0]))
    simple_gain = float(max(np.mean(y[:, m] - y[:, 0]) for m in (1, 2, 3)))
    capture = simple_gain / oracle_gain if oracle_gain > 0 else None
    survival_fraction, denominator = survival_explanation(grid)
    base_dwell = np.mean([grid[i][0]["unproductive_task_dwell_seconds"] for i in indices])
    oracle_dwell = np.mean([grid[i][4]["unproductive_task_dwell_seconds"] for i in indices])
    dwell_reduction = 1.0 - oracle_dwell / base_dwell if base_dwell > 0 else 0.0
    interventions = sum(grid[i][4]["intervention_counts"]["oracle_return"] > 0 for i in indices)
    gate_o = bool(intervals["lower"][0] > 0 and oracle_gain / np.mean(y[:, 0]) >= .10
                  and np.mean(y[:, 4] > y[:, 0]) >= .60 and interventions >= 5)
    gate_m = bool(gate_o and dwell_reduction >= .20 and survival_fraction is not None
                  and survival_fraction <= .50)
    gate_n = bool(gate_m and all(value > 0 for value in intervals["lower"][1:])
                  and capture is not None and capture < .80)
    return {"schema_version": "oracle-stuckness-analysis-v1a", "worlds": 24,
            "simultaneous_max_t": intervals,
            "gate_o": {"mean_oracle_gain": oracle_gain,
                       "mean_relative_gain": oracle_gain / float(np.mean(y[:, 0])),
                       "positive_world_fraction": float(np.mean(y[:, 4] > y[:, 0])),
                       "worlds_with_oracle_intervention": int(interventions)},
            "gate_m": {"common_alive_survival_explanation_fraction": survival_fraction,
                       "unproductive_dwell_reduction_fraction": float(dwell_reduction),
                       "positive_total_gain_denominator": denominator},
            "gate_n": {"best_simple_gain": simple_gain, "oracle_gain": oracle_gain,
                       "simple_capture_fraction": capture,
                       "all_three_simultaneous_lcbs_positive": bool(
                           all(value > 0 for value in intervals["lower"][1:]))},
            "descriptive_cross_fitted_simple": descriptive_loo(grid),
            "gates": {"O": gate_o, "M": gate_m, "N": gate_n},
            "warning": "M3 is outside the original Continue/Return action space."}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(load_grid(args.raw, json.loads(args.contract.read_text())))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
