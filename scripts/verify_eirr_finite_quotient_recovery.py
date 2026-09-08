#!/usr/bin/env python3
"""Verify the finite quotient separation construction in EIRR Theorem 26."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _components(points: np.ndarray, threshold: float) -> list[list[int]]:
    count = points.shape[0]
    unseen = set(range(count))
    components: list[list[int]] = []
    while unseen:
        seed = unseen.pop()
        component = {seed}
        frontier = [seed]
        while frontier:
            current = frontier.pop()
            neighbors = {
                other
                for other in unseen
                if float(np.max(np.abs(points[current] - points[other])))
                < threshold
            }
            unseen -= neighbors
            component |= neighbors
            frontier.extend(neighbors)
        components.append(sorted(component))
    return sorted(components)


def build_eirr_finite_quotient_recovery_report() -> dict[str, object]:
    prototypes = np.asarray(
        [
            [0.20, 0.35, 0.50],
            [0.45, 0.35, 0.50],
            [0.45, 0.60, 0.50],
        ],
        dtype=np.float64,
    )
    means = np.repeat(prototypes, 2, axis=0)
    true_components = [[0, 1], [2, 3], [4, 5]]
    interface_count, dimension = means.shape
    separation = 0.25
    delta = 0.05
    bound = int(
        np.ceil(
            1.05
            * 32.0
            * np.log(2.0 * interface_count * dimension / delta)
            / separation**2
        )
    )
    radius = float(
        np.sqrt(
            2.0
            * np.log(2.0 * interface_count * dimension / delta)
            / bound
        )
    )

    rng = np.random.default_rng(26082026)
    recoveries = 0
    trials = 300
    for _ in range(trials):
        estimates = rng.binomial(
            bound, means, size=(interface_count, dimension)
        ) / bound
        if _components(estimates, separation / 2.0) == true_components:
            recoveries += 1

    lower_rows: list[dict[str, float]] = []
    for gamma in [0.10, 0.05, 0.025]:
        samples = int(np.floor(0.05 / gamma**2))
        kl = float(-0.5 * samples * np.log(1.0 - 4.0 * gamma**2))
        pinsker_tv = float(np.sqrt(kl / 2.0))
        minimax_error_lower_bound = 0.5 * (1.0 - pinsker_tv)
        lower_rows.append(
            {
                "separation": gamma,
                "samples": samples,
                "samples_times_separation_squared": samples * gamma**2,
                "exact_kl": kl,
                "pinsker_tv_upper_bound": pinsker_tv,
                "minimax_error_lower_bound": minimax_error_lower_bound,
            }
        )

    perturbation = np.asarray(
        [
            [0.02, -0.01, 0.01],
            [-0.02, 0.01, -0.01],
            [0.01, -0.02, 0.02],
            [-0.01, 0.02, -0.02],
            [0.02, 0.01, -0.01],
            [-0.02, -0.01, 0.01],
        ],
        dtype=np.float64,
    )
    decoded = means + perturbation
    decoded_components = _components(decoded, separation / 2.0)

    checks = {
        "hoeffding_radius_is_below_quarter_separation": bool(
            radius < separation / 4.0
        ),
        "finite_quotient_is_recovered_in_seeded_construction": bool(
            recoveries == trials
        ),
        "two_point_kl_stays_constant_at_inverse_square_scale": bool(
            max(row["exact_kl"] for row in lower_rows) < 0.11
        ),
        "two_point_error_floor_is_nonvanishing": bool(
            min(row["minimax_error_lower_bound"] for row in lower_rows) > 0.38
        ),
        "decoded_witness_moments_recover_quotient": bool(
            decoded_components == true_components
        ),
    }
    status = (
        "EIRR_FINITE_QUOTIENT_RECOVERY_VERIFIED"
        if all(checks.values())
        else "EIRR_FINITE_QUOTIENT_RECOVERY_VERIFICATION_FAILED"
    )
    return {
        "status": status,
        "upper_recovery": {
            "interfaces": interface_count,
            "witness_dimension": dimension,
            "separation": separation,
            "samples_per_interface": bound,
            "uniform_radius": radius,
            "trials": trials,
            "recoveries": recoveries,
        },
        "lower_separation_boundary": lower_rows,
        "neural_decoder_corollary": {
            "maximum_reconstruction_error": float(
                np.max(np.abs(perturbation))
            ),
            "quarter_separation": separation / 4.0,
            "recovered_components": decoded_components,
        },
        "checks": checks,
        "non_claim": (
            "This is a finite bounded-interface construction. It does not "
            "supply a continuous neural complexity rate or justify reusing "
            "the same outcomes for quotient selection and value estimation."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_eirr_finite_quotient_recovery_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
