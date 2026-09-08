from __future__ import annotations

import json

from scripts.verify_eirr_critical_design import (
    build_eirr_critical_design_report,
)


def test_cost_aware_oracle_allocation_is_exact() -> None:
    report = build_eirr_critical_design_report()
    assert report["status"] == "EIRR_CRITICAL_MODE_DESIGN_VERIFIED"
    assert report["oracle"]["minimum_random_allocation_variance_ratio"] >= 1.0


def test_critical_mode_limit_matches_perron_prediction() -> None:
    report = build_eirr_critical_design_report()
    assert report["critical_limit"]["relative_error_at_smallest_gap"] < 0.02


def test_plugin_efficiency_bound_and_json_contract() -> None:
    report = build_eirr_critical_design_report()
    plugin = report["plugin"]
    assert (
        plugin["stage_two_efficiency_ratio"]
        <= plugin["stage_two_efficiency_upper_bound"]
    )
    assert all(report["checks"].values())
    json.dumps(report, allow_nan=False)
