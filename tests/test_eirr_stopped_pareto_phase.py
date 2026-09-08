from __future__ import annotations

import json

from scripts.verify_eirr_stopped_pareto_phase import (
    build_eirr_stopped_pareto_phase_report,
)


def test_stopped_disagreement_and_loss_have_predicted_powers() -> None:
    report = build_eirr_stopped_pareto_phase_report()
    assert report["maximum_disagreement_slope_error"] < 2e-4
    assert report["maximum_loss_slope_error"] < 2e-4


def test_gaussian_slice_has_the_exact_leading_constants() -> None:
    report = build_eirr_stopped_pareto_phase_report()
    assert report["maximum_leading_constant_relative_error"] < 2e-4


def test_adaptive_variance_ratio_reaches_pareto_radius() -> None:
    report = build_eirr_stopped_pareto_phase_report()
    assert report["status"] == "EIRR_STOPPED_PARETO_PHASE_VERIFIED"
    assert all(report["checks"].values())
    json.dumps(report, allow_nan=False)
