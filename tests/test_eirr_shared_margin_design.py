from __future__ import annotations

import json

from scripts.verify_eirr_shared_margin_design import (
    build_eirr_shared_margin_design_report,
)


def test_shared_margin_objective_is_convex_and_kkt_optimal() -> None:
    report = build_eirr_shared_margin_design_report()
    assert report["minimum_directional_curvature"] > 0.0
    assert report["maximum_kkt_relative_error"] < 2e-5
    assert report["minimum_random_objective_ratio"] >= 1.0 - 1e-10


def test_common_critical_singularity_cancels_from_shared_design() -> None:
    report = build_eirr_shared_margin_design_report()
    assert report["common_critical_gap"]["maximum_relative_spread"] < 2e-6


def test_query_specific_criticality_changes_shared_allocation() -> None:
    report = build_eirr_shared_margin_design_report()
    allocation = report["heterogeneous_critical_gaps"]["allocation"]
    assert allocation[0] > allocation[1]
    assert report["status"] == "EIRR_SHARED_MARGIN_DESIGN_VERIFIED"
    assert all(report["checks"].values())
    json.dumps(report, allow_nan=False)
