#!/usr/bin/env python3
"""Verify the scaling identities in continuous EIRR Theorem 27.

The report is a deterministic finite construction.  It checks the local-mass
Hölder upper radius, the matching *point-query* bump-testing scale, the
quotient-distance sandwich, and the quotient/transience/stopped-margin upper
phase exponents.  The distributed stopped-law lower construction is verified
separately by ``verify_first_passage_assouad_phase.py``.  It is not empirical
mission evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def holder_pointwise_radius(
    *,
    samples: int,
    smoothness: float,
    quotient_dimension: float,
    holder_constant: float,
    witness_bound: float,
    lower_mass_constant: float,
    log_factor: float,
) -> tuple[float, float]:
    """Return the balanced bandwidth and local-average confidence radius."""

    n = int(samples)
    alpha = float(smoothness)
    dimension = float(quotient_dimension)
    lipschitz = float(holder_constant)
    bound = float(witness_bound)
    mass = float(lower_mass_constant)
    logarithm = float(log_factor)
    if n <= 0:
        raise ValueError("samples must be positive")
    if alpha <= 0.0 or dimension <= 0.0:
        raise ValueError("smoothness and quotient dimension must be positive")
    if min(lipschitz, bound, mass, logarithm) <= 0.0:
        raise ValueError("radius constants must be positive")

    bandwidth = (
        bound**2 * logarithm / (mass * lipschitz**2 * n)
    ) ** (1.0 / (2.0 * alpha + dimension))
    radius = lipschitz * bandwidth**alpha + 2.0 * bound * np.sqrt(
        logarithm / (n * mass * bandwidth**dimension)
    )
    return float(bandwidth), float(radius)


def _log_slope(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.polyfit(np.log(x), np.log(y), deg=1)[0])


def build_eirr_continuous_joint_phase_report() -> dict[str, object]:
    alpha = 1.0
    dimension = 2.0
    rate_exponent = alpha / (2.0 * alpha + dimension)
    sample_sizes = np.asarray([2**power for power in range(14, 27, 2)], dtype=float)
    log_factor = 8.0

    radii: list[float] = []
    bandwidths: list[float] = []
    for sample_size in sample_sizes.astype(int):
        bandwidth, radius = holder_pointwise_radius(
            samples=sample_size,
            smoothness=alpha,
            quotient_dimension=dimension,
            holder_constant=0.4,
            witness_bound=1.0,
            lower_mass_constant=0.5,
            log_factor=log_factor,
        )
        bandwidths.append(bandwidth)
        radii.append(radius)
    radii_array = np.asarray(radii)
    radius_slope = _log_slope(sample_sizes, radii_array)

    # A Bernoulli Hölder bump has height a_n and support volume O(h_n^d).
    # At h_n=n^{-1/(2 alpha+d)}, n h_n^d a_n^2 is constant.
    lower_rows: list[dict[str, float]] = []
    for sample_size in sample_sizes.astype(int):
        h_n = sample_size ** (-1.0 / (2.0 * alpha + dimension))
        amplitude = 0.02 * h_n**alpha
        point_separation = amplitude
        support_probability_upper = (2.0 * h_n) ** dimension
        conditional_kl_upper = -0.5 * np.log(1.0 - 4.0 * amplitude**2)
        total_kl_upper = sample_size * support_probability_upper * conditional_kl_upper
        pinsker_tv = min(1.0, np.sqrt(total_kl_upper / 2.0))
        lower_rows.append(
            {
                "samples": float(sample_size),
                "bandwidth": float(h_n),
                "point_separation": float(point_separation),
                "total_kl_upper": float(total_kl_upper),
                "le_cam_error_lower": float(0.5 * (1.0 - pinsker_tv)),
            }
        )

    # Any decoded moment error at most epsilon perturbs every induced
    # l_infinity pseudometric distance by at most 2 epsilon.
    rng = np.random.default_rng(27082026)
    true_moments = rng.uniform(0.1, 0.9, size=(16, 5))
    epsilon = 0.02
    decoded_moments = true_moments + rng.uniform(
        -epsilon, epsilon, size=true_moments.shape
    )
    true_distances = np.max(
        np.abs(true_moments[:, None, :] - true_moments[None, :, :]), axis=2
    )
    decoded_distances = np.max(
        np.abs(decoded_moments[:, None, :] - decoded_moments[None, :, :]), axis=2
    )
    maximum_distance_error = float(
        np.max(np.abs(decoded_distances - true_distances))
    )
    pooling_threshold = 0.30
    pooled_mask = decoded_distances <= pooling_threshold
    maximum_true_pooled_distance = float(np.max(true_distances[pooled_mask]))

    # The nonparametric error is r_n=n^{-alpha/(2 alpha+d)}.  Resolvent and
    # Perron perturbations amplify it by Delta^{-1} and g^{-1}.  A stopped
    # margin with exponent kappa raises disagreement to the kappa power.
    nonparametric_error = sample_sizes ** (-rate_exponent)
    good_beta = 0.10
    critical_beta = rate_exponent
    good_gap = sample_sizes ** (-good_beta)
    critical_gap = sample_sizes ** (-critical_beta)
    good_ratio = nonparametric_error / good_gap
    critical_ratio = nonparametric_error / critical_gap
    kappa = 2.0
    good_disagreement = good_ratio**kappa
    critical_disagreement = np.minimum(critical_ratio, 1.0) ** kappa
    ratio_slope = _log_slope(sample_sizes, good_ratio)
    disagreement_slope = _log_slope(sample_sizes, good_disagreement)

    checks = {
        "holder_radius_has_declared_intrinsic_rate": bool(
            abs(radius_slope + rate_exponent) < 1e-10
        ),
        "bump_kl_stays_bounded_at_matching_rate": bool(
            max(row["total_kl_upper"] for row in lower_rows) < 0.01
        ),
        "bump_testing_error_floor_is_nonvanishing": bool(
            min(row["le_cam_error_lower"] for row in lower_rows) > 0.46
        ),
        "quotient_distance_obeys_two_epsilon_sandwich": bool(
            maximum_distance_error <= 2.0 * epsilon + 1e-12
            and maximum_true_pooled_distance
            <= pooling_threshold + 2.0 * epsilon + 1e-12
        ),
        "stable_gap_phase_has_predicted_slope": bool(
            abs(ratio_slope + rate_exponent - good_beta) < 1e-10
            and abs(disagreement_slope + kappa * (rate_exponent - good_beta))
            < 1e-10
        ),
        "critical_gap_phase_has_constant_error": bool(
            np.max(np.abs(critical_ratio - 1.0)) < 1e-12
            and np.max(np.abs(critical_disagreement - 1.0)) < 1e-12
        ),
    }
    status = (
        "EIRR_CONTINUOUS_JOINT_PHASE_VERIFIED"
        if all(checks.values())
        else "EIRR_CONTINUOUS_JOINT_PHASE_VERIFICATION_FAILED"
    )
    return {
        "status": status,
        "declared_class": {
            "smoothness": alpha,
            "quotient_dimension": dimension,
            "pointwise_rate_exponent": rate_exponent,
            "sample_sizes": [int(value) for value in sample_sizes],
            "bandwidths": bandwidths,
            "confidence_radii": radii,
            "fitted_radius_slope": radius_slope,
        },
        "matching_bump_lower_slice": lower_rows,
        "distance_sandwich": {
            "decoded_moment_error_bound": epsilon,
            "maximum_distance_error": maximum_distance_error,
            "pooling_threshold": pooling_threshold,
            "maximum_true_pooled_distance": maximum_true_pooled_distance,
        },
        "joint_phase": {
            "margin_exponent": kappa,
            "stable_gap_exponent": good_beta,
            "critical_gap_exponent": critical_beta,
            "stable_error_to_gap_slope": ratio_slope,
            "stable_disagreement_slope": disagreement_slope,
            "critical_error_to_gap_ratios": critical_ratio.tolist(),
        },
        "checks": checks,
        "non_claim": (
            "The bump rows prove only the point-query lower scale; they do not "
            "by themselves create stopped margin mass. The distributed "
            "first-passage Assouad verifier checks the additional algebra. "
            "Neither verifier proves arbitrary neural-encoder consistency, "
            "general covariate shift, navigation readiness, or empirical Pareto gain."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_eirr_continuous_joint_phase_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
