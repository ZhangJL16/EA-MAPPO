#!/usr/bin/env python3
"""Verify the predictable first-hit martingale quotient certificate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def predictable_first_hit_radius(
    *,
    samples_per_query: int,
    query_count: int,
    witness_dimension: int,
    witness_bound: float,
    holder_bias: float,
    failure_probability: float,
) -> float:
    count = int(samples_per_query)
    queries = int(query_count)
    dimension = int(witness_dimension)
    bound = float(witness_bound)
    bias = float(holder_bias)
    delta = float(failure_probability)
    if min(count, queries, dimension) <= 0:
        raise ValueError("sample, query, and witness counts must be positive")
    if bound <= 0.0 or bias < 0.0:
        raise ValueError("witness bound must be positive and bias nonnegative")
    if not 0.0 < delta < 1.0:
        raise ValueError("failure probability must lie in (0, 1)")
    stochastic = bound * np.sqrt(
        2.0 * np.log(2.0 * queries * dimension / delta) / count
    )
    return float(bias + stochastic)


def _adaptive_trial(
    *, rng: np.random.Generator, centers: np.ndarray, bandwidth: float, hits: int
) -> tuple[np.ndarray, list[np.ndarray], int]:
    selected: list[list[float]] = [[] for _ in centers]
    z = 0.5
    transitions = 0
    while min(len(values) for values in selected) < hits:
        probability = 0.2 + 0.6 * z
        witness = float(rng.random() < probability)
        for query_index, center in enumerate(centers):
            if len(selected[query_index]) < hits and abs(z - center) <= bandwidth:
                selected[query_index].append(witness)
        # The next interface depends on the realized past witness, while random
        # exploration prevents a coverage dead end. It is known before the next
        # primitive outcome is sampled.
        if rng.random() < 0.35:
            z = float(rng.random())
        else:
            z = float(np.clip(0.65 * z + 0.25 * witness + 0.10 * rng.random(), 0.0, 1.0))
        transitions += 1
        if transitions > 200_000:
            raise RuntimeError("adaptive construction failed to hit every query")
    estimates = np.asarray([np.mean(values) for values in selected])
    return estimates, [np.asarray(values) for values in selected], transitions


def build_eirr_predictable_martingale_quotient_report() -> dict[str, object]:
    centers = np.asarray([0.25, 0.50, 0.75])
    bandwidth = 0.10
    holder_constant = 0.60
    hits = 200
    delta = 0.05
    radius = predictable_first_hit_radius(
        samples_per_query=hits,
        query_count=centers.size,
        witness_dimension=1,
        witness_bound=1.0,
        holder_bias=holder_constant * bandwidth,
        failure_probability=delta,
    )
    true_centers = 0.2 + holder_constant * centers

    rng = np.random.default_rng(30082026)
    trials = 200
    simultaneous_coverages = 0
    maximum_error = 0.0
    transition_counts: list[int] = []
    exemplar_values: list[np.ndarray] | None = None
    for trial in range(trials):
        estimates, selected, transitions = _adaptive_trial(
            rng=rng, centers=centers, bandwidth=bandwidth, hits=hits
        )
        errors = np.abs(estimates - true_centers)
        maximum_error = max(maximum_error, float(np.max(errors)))
        simultaneous_coverages += int(np.all(errors <= radius))
        transition_counts.append(transitions)
        if trial == 0:
            exemplar_values = selected

    assert exemplar_values is not None
    replay_factor = 16
    unique_values = exemplar_values[1]
    replayed_values = np.repeat(unique_values, replay_factor)
    unique_mean = float(np.mean(unique_values))
    replay_mean = float(np.mean(replayed_values))
    correct_stochastic_radius = predictable_first_hit_radius(
        samples_per_query=hits,
        query_count=centers.size,
        witness_dimension=1,
        witness_bound=1.0,
        holder_bias=0.0,
        failure_probability=delta,
    )
    naive_replay_radius = predictable_first_hit_radius(
        samples_per_query=hits * replay_factor,
        query_count=centers.size,
        witness_dimension=1,
        witness_bound=1.0,
        holder_bias=0.0,
        failure_probability=delta,
    )

    sample_grid = np.asarray([100, 400, 1600, 6400], dtype=float)
    stochastic_grid = np.asarray(
        [
            predictable_first_hit_radius(
                samples_per_query=int(value),
                query_count=1,
                witness_dimension=1,
                witness_bound=1.0,
                holder_bias=0.0,
                failure_probability=delta,
            )
            for value in sample_grid
        ]
    )
    fitted_slope = float(np.polyfit(np.log(sample_grid), np.log(stochastic_grid), 1)[0])

    checks = {
        "adaptive_trajectory_is_simultaneously_covered": bool(
            simultaneous_coverages == trials
        ),
        "every_query_uses_exact_unique_hit_count": bool(
            all(len(values) == hits for values in exemplar_values)
        ),
        "replay_duplication_does_not_change_estimate": bool(
            abs(unique_mean - replay_mean) < 1e-12
        ),
        "replay_duplication_cannot_shrink_valid_radius": bool(
            abs(naive_replay_radius * np.sqrt(replay_factor) - correct_stochastic_radius)
            < 1e-12
        ),
        "unique_transition_radius_has_root_count_slope": bool(
            abs(fitted_slope + 0.5) < 1e-12
        ),
    }
    status = (
        "EIRR_PREDICTABLE_MARTINGALE_QUOTIENT_VERIFIED"
        if all(checks.values())
        else "EIRR_PREDICTABLE_MARTINGALE_QUOTIENT_VERIFICATION_FAILED"
    )
    return {
        "status": status,
        "adaptive_first_hit": {
            "queries": centers.tolist(),
            "bandwidth": bandwidth,
            "unique_hits_per_query": hits,
            "simultaneous_radius": radius,
            "trials": trials,
            "simultaneous_coverages": simultaneous_coverages,
            "maximum_observed_error": maximum_error,
            "mean_raw_transitions_to_coverage": float(np.mean(transition_counts)),
            "maximum_raw_transitions_to_coverage": int(max(transition_counts)),
        },
        "replay_duplicate_audit": {
            "replay_factor": replay_factor,
            "unique_mean": unique_mean,
            "replayed_mean": replay_mean,
            "valid_unique_transition_radius": correct_stochastic_radius,
            "invalid_naive_replay_radius": naive_replay_radius,
        },
        "rate_check": {
            "sample_grid": sample_grid.astype(int).tolist(),
            "stochastic_radii": stochastic_grid.tolist(),
            "fitted_log_slope": fitted_slope,
        },
        "checks": checks,
        "non_claim": (
            "The certificate assumes each primitive outcome is fresh conditional "
            "on a predictable executed interface. Replaying one stored outcome "
            "many times does not create new statistical samples."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_eirr_predictable_martingale_quotient_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
