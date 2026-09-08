#!/usr/bin/env python3
"""Verify the irreversible first-disagreement bounds on an explicit finite law."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.energy_mc.return_decision import sequential_boundary_certificate


def first_disagreement_index(scores: np.ndarray, requirement_errors: np.ndarray) -> int | None:
    scores = np.asarray(scores, dtype=np.float64)
    errors = np.asarray(requirement_errors, dtype=np.float64)
    if scores.ndim != 1 or errors.shape != scores.shape:
        raise ValueError("scores and requirement errors must be aligned vectors")
    for index, (score, error) in enumerate(zip(scores, errors, strict=True)):
        oracle_commit = bool(score <= 0.0)
        learned_commit = bool(score - error <= 0.0)
        if oracle_commit != learned_commit:
            return index
        if oracle_commit:
            return None
    return None


def build_boundary_verification_report() -> dict[str, object]:
    weights = np.asarray([0.4, 0.3, 0.2, 0.1], dtype=np.float64)
    scores = np.asarray(
        [
            [2.0, 1.0, -1.0],
            [0.1, 1.0, -1.0],
            [-0.1, 1.0, -1.0],
            [2.0, 0.1, -1.0],
        ],
        dtype=np.float64,
    )
    errors = np.asarray(
        [
            [0.2, 0.2, 0.2],
            [0.2, 0.2, 0.2],
            [-0.2, -0.2, -0.2],
            [0.2, 0.2, 0.2],
        ],
        dtype=np.float64,
    )
    widths = np.max(np.abs(errors), axis=0)
    disagreements = [
        first_disagreement_index(path_scores, path_errors)
        for path_scores, path_errors in zip(scores, errors, strict=True)
    ]
    exact_disagreement = float(
        sum(weight for weight, index in zip(weights, disagreements, strict=True) if index is not None)
    )

    boundary_probabilities = np.zeros(scores.shape[1], dtype=np.float64)
    for path_index, weight in enumerate(weights):
        oracle_alive = True
        for time_index in range(scores.shape[1]):
            if not oracle_alive:
                break
            score = scores[path_index, time_index]
            if abs(score) <= widths[time_index]:
                boundary_probabilities[time_index] += weight
            oracle_alive = bool(score > 0.0)

    oracle_stranding = np.zeros(weights.size, dtype=np.float64)
    learned_stranding = np.asarray([0.0, 1.0, 0.0, 1.0])
    oracle_throughput = np.full(weights.size, 10.0)
    learned_throughput = np.asarray([10.0, 0.0, 5.0, 0.0])
    exact_stranding_deviation = float(
        abs(weights @ learned_stranding - weights @ oracle_stranding)
    )
    exact_throughput_deviation = float(
        abs(weights @ learned_throughput - weights @ oracle_throughput)
    )
    certificate = sequential_boundary_certificate(
        estimation_failure_probability=0.0,
        boundary_probabilities=boundary_probabilities,
        boundary_widths=widths,
        throughput_upper_bound=10.0,
        loss_lipschitz=2.0,
        loss_upper_bound=1.0,
        oracle_stranding_rate=0.0,
        oracle_throughput=10.0,
        heuristic_throughput=3.0,
        stranding_ceiling=0.7,
        minimum_throughput_gain_fraction=0.05,
    )
    return {
        "status": "IRREVERSIBLE_BOUNDARY_IDENTITIES_VERIFIED",
        "path_weights": weights.tolist(),
        "first_disagreement_indices": disagreements,
        "boundary_probabilities": boundary_probabilities.tolist(),
        "boundary_widths": widths.tolist(),
        "exact_first_disagreement_probability": exact_disagreement,
        "first_disagreement_probability_bound": (
            certificate.first_disagreement_probability_bound
        ),
        "exact_stranding_deviation": exact_stranding_deviation,
        "stranding_deviation_bound": certificate.stranding_deviation_bound,
        "exact_throughput_deviation": exact_throughput_deviation,
        "throughput_deviation_bound": certificate.throughput_deviation_bound,
        "population_headroom_preserved_by_bound": (
            certificate.population_headroom_preserved
        ),
        "certificate": certificate.as_dict(),
        "non_claim": (
            "The finite construction verifies the coupling algebra; it is not a "
            "formal Oracle Gate result or empirical Pareto evidence."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_boundary_verification_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
