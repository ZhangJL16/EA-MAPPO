#!/usr/bin/env python3
"""Verify Theorem 19's exact quotient--transience phase-transition formulas."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def build_eirr_critical_phase_report() -> dict[str, object]:
    multiplier = 1.25
    risk_parameter = math.log(multiplier)
    quotient_coverage = 0.20
    gaps = np.asarray([0.20, 0.10, 0.05, 0.025], dtype=np.float64)
    termination_probability = (gaps + multiplier - 1.0) / multiplier

    psi = multiplier * termination_probability / gaps
    raw_variance = (
        termination_probability
        * (1.0 - termination_probability)
        * multiplier**2
        * (multiplier - 1.0) ** 2
        / (quotient_coverage * gaps**4)
    )
    log_variance = (
        (1.0 - termination_probability)
        * (multiplier - 1.0) ** 2
        / (quotient_coverage * termination_probability * gaps**2)
    )
    requirement_variance = log_variance / risk_parameter**2

    dpsi_dp = -multiplier * (multiplier - 1.0) / gaps**2
    dlog_dp = -(multiplier - 1.0) / (
        termination_probability * gaps
    )
    bernoulli_variance = (
        termination_probability
        * (1.0 - termination_probability)
        / quotient_coverage
    )
    raw_delta_variance = np.square(dpsi_dp) * bernoulli_variance
    log_delta_variance = np.square(dlog_dp) * bernoulli_variance

    normalized_raw = raw_variance / (
        termination_probability * (1.0 - termination_probability)
    )
    normalized_log = log_variance * termination_probability / (
        1.0 - termination_probability
    )
    raw_slope = float(
        np.polyfit(np.log(gaps), np.log(normalized_raw), 1)[0]
    )
    log_slope = float(
        np.polyfit(np.log(gaps), np.log(normalized_log), 1)[0]
    )

    sample_size = 50_000
    local_requirement_scale = np.sqrt(requirement_variance / sample_size)
    direct_scale = (
        (multiplier - 1.0)
        / (risk_parameter * gaps)
        * np.sqrt(
            (1.0 - termination_probability)
            / (
                sample_size
                * quotient_coverage
                * termination_probability
            )
        )
    )

    checks = {
        "all_models_are_exponentially_transient": bool(
            np.all(gaps > 0.0)
        ),
        "raw_efficiency_matches_bernoulli_delta_method": bool(
            np.allclose(raw_variance, raw_delta_variance, atol=1e-12)
        ),
        "log_efficiency_matches_bernoulli_delta_method": bool(
            np.allclose(log_variance, log_delta_variance, atol=1e-12)
        ),
        "requirement_scale_formula_matches": bool(
            np.allclose(local_requirement_scale, direct_scale, atol=1e-13)
        ),
        "normalized_raw_gap_exponent_is_minus_four": bool(
            np.isclose(raw_slope, -4.0, atol=1e-12)
        ),
        "normalized_log_gap_exponent_is_minus_two": bool(
            np.isclose(log_slope, -2.0, atol=1e-12)
        ),
        "halving_gap_increases_normalized_raw_variance_by_sixteen": bool(
            np.allclose(normalized_raw[1:] / normalized_raw[:-1], 16.0)
        ),
        "halving_gap_increases_normalized_log_variance_by_four": bool(
            np.allclose(normalized_log[1:] / normalized_log[:-1], 4.0)
        ),
    }
    status = (
        "EIRR_QUOTIENT_TRANSIENCE_PHASE_VERIFIED"
        if all(checks.values())
        else "EIRR_QUOTIENT_TRANSIENCE_PHASE_VERIFICATION_FAILED"
    )
    rows = []
    for index, gap in enumerate(gaps):
        rows.append(
            {
                "gap": float(gap),
                "termination_probability": float(
                    termination_probability[index]
                ),
                "psi": float(psi[index]),
                "raw_efficiency_variance": float(raw_variance[index]),
                "log_efficiency_variance": float(log_variance[index]),
                "requirement_efficiency_variance": float(
                    requirement_variance[index]
                ),
                "requirement_local_scale_at_n": float(
                    local_requirement_scale[index]
                ),
            }
        )
    return {
        "status": status,
        "model": {
            "multiplicative_factor": multiplier,
            "risk_parameter_for_unit_cost": risk_parameter,
            "quotient_class_source_coverage": quotient_coverage,
            "sample_size_for_local_scale": sample_size,
            "raw_target_cell_source_coverage_can_be_zero": True,
            "pooled_coverage_requires_exact_shared_primitive": True,
        },
        "critical_rows": rows,
        "scaling": {
            "normalized_raw_log_log_slope": raw_slope,
            "normalized_log_risk_log_log_slope": log_slope,
            "raw_consistency_information": "n * mu_h * Delta^4",
            "log_requirement_consistency_information": (
                "n * mu_h * Delta^2"
            ),
        },
        "checks": checks,
        "non_claim": (
            "This verifies an exact one-state Bernoulli information slice. "
            "It does not prove the proposed multi-state Perron-Frobenius phase "
            "transition or establish oral-level novelty."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_eirr_critical_phase_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
