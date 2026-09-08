#!/usr/bin/env python3
"""Verify Theorem 16's finite linear risk-witness quotient formulas."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def build_linear_risk_quotient_report() -> dict[str, object]:
    n = 20
    sigma = 0.4
    source_interfaces = np.concatenate(
        [np.full(n // 2, -1.0), np.full(n // 2, 1.0)]
    )
    design = np.column_stack(
        [np.ones(n, dtype=np.float64), source_interfaces]
    )
    target_interface = 0.0
    target_feature_mean = np.asarray([1.0, target_interface])
    gram = design.T @ design
    gram_pinv = np.linalg.pinv(gram)
    variance_leverage = float(target_feature_mean @ gram_pinv @ target_feature_mean)
    quotient_coefficient = n * variance_leverage
    exact_minimax_mse = sigma**2 * variance_leverage

    estimator_weights = design @ gram_pinv @ target_feature_mean
    estimator_variance = sigma**2 * float(estimator_weights @ estimator_weights)
    unbiased_feature = design.T @ estimator_weights

    # Non-identifiable representation slice: source features expose only the
    # intercept, while the target functional asks for the hidden second axis.
    collapsed_design = np.column_stack(
        [np.ones(n, dtype=np.float64), np.zeros(n, dtype=np.float64)]
    )
    hidden_target_mean = np.asarray([0.0, 1.0])
    null_projector = np.eye(2) - np.linalg.pinv(collapsed_design) @ collapsed_design
    hidden_projection = null_projector @ hidden_target_mean
    parameter_radius = 2.0
    indistinguishable_mse_lower_bound = (
        parameter_radius**2 * float(hidden_projection @ hidden_projection)
    )
    theta_plus_hidden = parameter_radius * hidden_projection / np.linalg.norm(
        hidden_projection
    )
    theta_minus_hidden = -theta_plus_hidden

    # Least-favourable identifiable pair from Claim D.
    least_direction = (
        gram_pinv @ target_feature_mean / math.sqrt(variance_leverage)
    )
    theta_plus = 0.5 * sigma * least_direction
    theta_minus = -theta_plus
    source_mean_difference = design @ (theta_plus - theta_minus)
    mahalanobis_distance = float(
        np.linalg.norm(source_mean_difference) / sigma
    )
    target_functional_half_gap = float(
        target_feature_mean @ theta_plus
    )
    exact_equal_prior_testing_error = 0.5 * math.erfc(
        mahalanobis_distance / (2.0 * math.sqrt(2.0))
    )

    checks = {
        "balanced_gram_is_n_identity": bool(
            np.allclose(gram, n * np.eye(2), atol=1e-13)
        ),
        "target_feature_is_identifiable": bool(
            np.allclose(unbiased_feature, target_feature_mean, atol=1e-13)
        ),
        "quotient_coefficient_equals_one": bool(
            np.isclose(quotient_coefficient, 1.0, atol=1e-13)
        ),
        "minimax_variance_matches_estimator": bool(
            np.isclose(estimator_variance, exact_minimax_mse, atol=1e-13)
        ),
        "raw_target_support_is_absent": bool(
            target_interface not in set(source_interfaces.tolist())
        ),
        "collapsed_target_is_not_identifiable": bool(
            np.linalg.norm(hidden_projection) > 0.0
            and np.allclose(
                collapsed_design @ theta_plus_hidden,
                collapsed_design @ theta_minus_hidden,
                atol=1e-13,
            )
        ),
        "least_pair_mahalanobis_distance_is_one": bool(
            np.isclose(mahalanobis_distance, 1.0, atol=1e-13)
        ),
        "least_pair_target_scale_matches_rate": bool(
            np.isclose(
                target_functional_half_gap,
                0.5 * sigma * math.sqrt(variance_leverage),
                atol=1e-13,
            )
        ),
    }
    status = (
        "LINEAR_RISK_QUOTIENT_THEOREM_VERIFIED"
        if all(checks.values())
        else "LINEAR_RISK_QUOTIENT_VERIFICATION_FAILED"
    )
    return {
        "status": status,
        "sample_size": n,
        "noise_standard_deviation": sigma,
        "raw_support_example": {
            "source_interfaces": sorted(set(source_interfaces.tolist())),
            "target_interface": target_interface,
            "target_absolutely_continuous_wrt_source_support": False,
            "gram": gram.tolist(),
            "target_feature_mean": target_feature_mean.tolist(),
            "variance_leverage": variance_leverage,
            "quotient_coefficient": quotient_coefficient,
            "exact_minimax_mse": exact_minimax_mse,
            "estimator_variance": estimator_variance,
        },
        "nonidentifiable_slice": {
            "hidden_target_projection_norm": float(
                np.linalg.norm(hidden_projection)
            ),
            "parameter_radius": parameter_radius,
            "indistinguishable_mse_lower_bound": (
                indistinguishable_mse_lower_bound
            ),
            "source_mean_difference_norm": float(
                np.linalg.norm(
                    collapsed_design
                    @ (theta_plus_hidden - theta_minus_hidden)
                )
            ),
        },
        "decision_lower_pair": {
            "mahalanobis_distance": mahalanobis_distance,
            "target_functional_half_gap": target_functional_half_gap,
            "exact_equal_prior_testing_error": exact_equal_prior_testing_error,
        },
        "checks": checks,
        "non_claim": (
            "This verifies a classical fixed-design Gaussian slice with fixed "
            "risk-occupation and continuation nuisances. It does not prove a "
            "neural, dependent-trajectory, or end-to-end Oracle/Pareto result."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_linear_risk_quotient_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
