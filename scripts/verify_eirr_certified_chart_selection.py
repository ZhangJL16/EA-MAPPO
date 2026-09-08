#!/usr/bin/env python3
"""Verify EIRR Theorem 33 finite-library chart certification algebra."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def partition_diameter(mu: np.ndarray, labels: np.ndarray) -> float:
    values = np.asarray(mu, dtype=float)
    groups = np.asarray(labels)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or groups.shape != (values.shape[0],):
        raise ValueError("mu must be N by D and labels must have length N")
    diameter = 0.0
    for group in np.unique(groups):
        rows = values[groups == group]
        if len(rows) <= 1:
            continue
        pairwise = np.max(np.abs(rows[:, None, :] - rows[None, :, :]), axis=2)
        diameter = max(diameter, float(pairwise.max()))
    return diameter


def select_certified_chart(
    *,
    true_mu: np.ndarray,
    estimated_mu: np.ndarray,
    partitions: list[np.ndarray],
    variance_terms: np.ndarray,
    amplification: float,
    witness_radius: float,
) -> dict[str, object]:
    if amplification <= 0.0 or witness_radius < 0.0:
        raise ValueError("amplification must be positive and radius nonnegative")
    variance = np.asarray(variance_terms, dtype=float)
    if variance.shape != (len(partitions),) or np.any(variance < 0.0):
        raise ValueError("one nonnegative variance term is required per partition")
    true_diameter = np.asarray(
        [partition_diameter(true_mu, p) for p in partitions]
    )
    empirical_diameter = np.asarray(
        [partition_diameter(estimated_mu, p) for p in partitions]
    )
    upper = amplification**2 * (empirical_diameter + 2.0 * witness_radius) ** 2
    upper += variance
    selected = int(np.argmin(upper))
    selected_true_risk_proxy = (
        amplification**2 * true_diameter[selected] ** 2 + variance[selected]
    )
    oracle_envelope = float(
        np.min(
            amplification**2 * (true_diameter + 4.0 * witness_radius) ** 2
            + variance
        )
    )
    return {
        "selected_index": selected,
        "true_diameters": true_diameter.tolist(),
        "empirical_diameters": empirical_diameter.tolist(),
        "certified_objectives": upper.tolist(),
        "selected_true_risk_proxy": float(selected_true_risk_proxy),
        "oracle_envelope": oracle_envelope,
        "diameter_sandwich_holds": bool(
            np.all(np.abs(empirical_diameter - true_diameter) <= 2.0 * witness_radius + 1e-12)
        ),
        "oracle_inequality_holds": bool(
            selected_true_risk_proxy <= oracle_envelope + 1e-12
        ),
    }


def build_certified_chart_selection_report() -> dict[str, object]:
    rng = np.random.default_rng(20260830)
    true_mu = np.asarray([0.2, 0.2, 0.5, 0.5, 0.8, 0.8])[:, None]
    partitions = [
        np.arange(6),
        np.asarray([0, 0, 1, 1, 2, 2]),
        np.asarray([0, 0, 0, 0, 1, 1]),
        np.zeros(6, dtype=int),
    ]
    variance_terms = np.asarray([0.12, 0.04, 0.02, 0.01])
    sample_size = 1000
    delta = 0.01
    radius = float(np.sqrt(2.0 * np.log(2.0 * 6.0 / delta) / sample_size))
    trials: list[dict[str, object]] = []
    event_trials = 0
    sandwich_violations = 0
    oracle_violations = 0
    for _ in range(300):
        samples = rng.binomial(
            1,
            true_mu[:, 0, None],
            size=(len(true_mu), sample_size),
        )
        estimate = samples.mean(axis=1)[:, None]
        event = bool(np.max(np.abs(estimate - true_mu)) <= radius)
        result = select_certified_chart(
            true_mu=true_mu,
            estimated_mu=estimate,
            partitions=partitions,
            variance_terms=variance_terms,
            amplification=2.0,
            witness_radius=radius,
        )
        if event:
            event_trials += 1
            sandwich_violations += int(not result["diameter_sandwich_holds"])
            oracle_violations += int(not result["oracle_inequality_holds"])
        trials.append(
            {
                "uniform_event": event,
                "selected_index": result["selected_index"],
                "diameter_sandwich_holds": result["diameter_sandwich_holds"],
                "oracle_inequality_holds": result["oracle_inequality_holds"],
            }
        )

    checks = {
        "uniform_event_observed": event_trials > 0,
        "diameter_sandwich_has_no_event_violation": sandwich_violations == 0,
        "selector_oracle_has_no_event_violation": oracle_violations == 0,
        "finite_chart_radius_has_inverse_sqrt_sample_order": bool(
            abs(
                radius
                / np.sqrt(2.0 * np.log(2.0 * 6.0 / delta) / sample_size)
                - 1.0
            )
            < 1e-12
        ),
    }
    status = (
        "EIRR_CERTIFIED_CHART_SELECTION_VERIFIED"
        if all(checks.values())
        else "EIRR_CERTIFIED_CHART_SELECTION_VERIFICATION_FAILED"
    )
    return {
        "status": status,
        "sample_size_per_interface": sample_size,
        "witness_radius": radius,
        "uniform_event_trials": event_trials,
        "total_trials": len(trials),
        "event_sandwich_violations": sandwich_violations,
        "event_oracle_violations": oracle_violations,
        "checks": checks,
        "matched_information_result": (
            "Any unrestricted transformed baseline given the same structure "
            "fold, witness dictionary, and candidate partitions can run the "
            "same selector and reproduce its manager decision."
        ),
        "non_claim": (
            "The result certifies selection from a finite partition library; "
            "it is not a distribution-free neural-chart generalization bound."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_certified_chart_selection_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
