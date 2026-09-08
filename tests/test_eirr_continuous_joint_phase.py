from __future__ import annotations

import json

import pytest

from scripts.verify_eirr_continuous_joint_phase import (
    build_eirr_continuous_joint_phase_report,
    holder_pointwise_radius,
)


def test_holder_pointwise_radius_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="samples"):
        holder_pointwise_radius(
            samples=0,
            smoothness=1.0,
            quotient_dimension=2.0,
            holder_constant=1.0,
            witness_bound=1.0,
            lower_mass_constant=1.0,
            log_factor=1.0,
        )


def test_continuous_upper_and_matching_bump_scales() -> None:
    report = build_eirr_continuous_joint_phase_report()
    declared = report["declared_class"]
    assert declared["fitted_radius_slope"] == pytest.approx(
        -declared["pointwise_rate_exponent"], abs=1e-10
    )
    lower = report["matching_bump_lower_slice"]
    assert max(row["total_kl_upper"] for row in lower) < 0.01
    assert min(row["le_cam_error_lower"] for row in lower) > 0.46


def test_distance_sandwich_and_joint_phase() -> None:
    report = build_eirr_continuous_joint_phase_report()
    sandwich = report["distance_sandwich"]
    assert sandwich["maximum_distance_error"] <= (
        2.0 * sandwich["decoded_moment_error_bound"] + 1e-12
    )
    assert sandwich["maximum_true_pooled_distance"] <= (
        sandwich["pooling_threshold"]
        + 2.0 * sandwich["decoded_moment_error_bound"]
        + 1e-12
    )
    phase = report["joint_phase"]
    expected = -2.0 * (
        report["declared_class"]["pointwise_rate_exponent"]
        - phase["stable_gap_exponent"]
    )
    assert phase["stable_disagreement_slope"] == pytest.approx(expected, abs=1e-10)
    assert all(value == pytest.approx(1.0) for value in phase["critical_error_to_gap_ratios"])
    assert report["status"] == "EIRR_CONTINUOUS_JOINT_PHASE_VERIFIED"
    assert all(report["checks"].values())
    json.dumps(report, allow_nan=False)
