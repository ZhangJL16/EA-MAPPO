from __future__ import annotations

import json

from scripts.verify_eirr_collapsing_design_phase import (
    build_eirr_collapsing_design_phase_report,
)


def test_mode_conditioning_derivative_is_exact() -> None:
    report = build_eirr_collapsing_design_phase_report()
    conditioning = report["mode_conditioning"]
    assert conditioning["maximum_normalized_derivative_error"] < 1e-6


def test_local_two_point_boundary_has_constant_kl_and_separation() -> None:
    report = build_eirr_collapsing_design_phase_report()
    rows = report["local_two_point_boundary"]["rows"]
    assert min(row["log_priority_separation"] for row in rows) > 0.18
    assert max(row["bounded_rademacher_pilot_kl"] for row in rows) < 0.021


def test_projected_noise_difficulty_is_not_determined_by_variance() -> None:
    report = build_eirr_collapsing_design_phase_report()
    phase = report["projected_variance_nonuniversality"]
    rows = phase["rare_bernoulli_rows"]
    assert phase["gaussian_scale_log_sigma_fisher_information_per_sample"] == 2.0
    assert min(row["no_rare_event_probability"] for row in rows) > 0.34
    assert rows[-1]["standardized_fourth_moment"] > rows[0][
        "standardized_fourth_moment"
    ]
    assert report["status"] == "EIRR_COLLAPSING_DESIGN_PHASE_VERIFIED"
    assert all(report["checks"].values())
    json.dumps(report, allow_nan=False)
