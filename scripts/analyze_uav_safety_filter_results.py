from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


METRICS = (
    "realized_energy",
    "path_ratio",
    "steps",
    "mean_intervention",
    "minimum_clearance",
)


def two_sided_sign_test(positive: int, negative: int) -> float:
    trials = positive + negative
    if trials == 0:
        return 1.0
    extreme = min(positive, negative)
    lower_tail = sum(math.comb(trials, index) for index in range(extreme + 1))
    return float(min(1.0, 2.0 * lower_tail / 2.0**trials))


def paired_summary(
    baseline: np.ndarray,
    comparator: np.ndarray,
    *,
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> dict[str, object]:
    difference = comparator - baseline
    indices = rng.integers(
        0,
        difference.size,
        size=(bootstrap_samples, difference.size),
    )
    bootstrap_means = np.mean(difference[indices], axis=1)
    positive = int(np.sum(difference > 1e-12))
    negative = int(np.sum(difference < -1e-12))
    return {
        "pairs": int(difference.size),
        "baseline_mean": float(np.mean(baseline)),
        "comparator_mean": float(np.mean(comparator)),
        "mean_difference_comparator_minus_baseline": float(np.mean(difference)),
        "relative_mean_difference": float(
            np.mean(difference) / max(abs(float(np.mean(baseline))), 1e-12)
        ),
        "median_difference": float(np.median(difference)),
        "standard_deviation_of_difference": float(np.std(difference, ddof=1)),
        "bootstrap_mean_difference_95_percent_interval": [
            float(np.quantile(bootstrap_means, 0.025)),
            float(np.quantile(bootstrap_means, 0.975)),
        ],
        "comparator_lower_count": negative,
        "comparator_equal_count": int(difference.size - positive - negative),
        "comparator_higher_count": positive,
        "two_sided_sign_test_p": two_sided_sign_test(positive, negative),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired controlled-rollout analysis")
    parser.add_argument("--rollouts-csv", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--comparators", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.bootstrap_samples <= 0:
        raise ValueError("bootstrap sample count must be positive")

    with Path(args.rollouts_csv).open() as handle:
        rows = list(csv.DictReader(handle))
    by_method: dict[str, dict[str, dict[str, str]]] = {}
    for row in rows:
        by_method.setdefault(row["method"], {})[row["scenario_id"]] = row
    if args.baseline not in by_method:
        raise ValueError(f"baseline method not found: {args.baseline}")

    rng = np.random.default_rng(args.seed)
    output: dict[str, object] = {
        "rollouts_csv": str(Path(args.rollouts_csv).resolve()),
        "baseline": args.baseline,
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "comparisons": {},
    }
    baseline_rows = by_method[args.baseline]
    for comparator in args.comparators:
        if comparator not in by_method:
            raise ValueError(f"comparator method not found: {comparator}")
        comparator_rows = by_method[comparator]
        scenario_ids = sorted(set(baseline_rows) & set(comparator_rows))
        if len(scenario_ids) != len(baseline_rows) or len(scenario_ids) != len(
            comparator_rows
        ):
            raise ValueError("paired methods do not have identical scenario support")
        metrics = {}
        for metric in METRICS:
            baseline = np.asarray(
                [float(baseline_rows[key][metric]) for key in scenario_ids]
            )
            candidate = np.asarray(
                [float(comparator_rows[key][metric]) for key in scenario_ids]
            )
            metrics[metric] = paired_summary(
                baseline,
                candidate,
                bootstrap_samples=args.bootstrap_samples,
                rng=rng,
            )
        output["comparisons"][comparator] = {
            "scenario_count": len(scenario_ids),
            "metrics": metrics,
        }

    Path(args.output).write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
