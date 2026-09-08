from __future__ import annotations

import json

from scripts.verify_eirr_adaptive_critical_design import (
    build_eirr_adaptive_critical_design_report,
)


def test_pilot_is_below_risk_value_critical_scale() -> None:
    report = build_eirr_adaptive_critical_design_report()
    rows = report["rows"]
    assert rows[-1]["pilot_per_interface"] > rows[0]["pilot_per_interface"]
    assert rows[-1]["pilot_gap2"] < rows[0]["pilot_gap2"]
    assert rows[-1]["pilot_gap2"] < 0.07


def test_adaptive_priority_still_approaches_oracle() -> None:
    report = build_eirr_adaptive_critical_design_report()
    assert report["status"] == "EIRR_SINGULARITY_FREE_ADAPTIVE_DESIGN_VERIFIED"
    assert report["rows"][-1]["median_oracle_variance_ratio"] < 1.0002
    assert max(row["q90_oracle_variance_ratio"] for row in report["rows"]) < 1.02


def test_adaptive_design_checks_and_json_contract() -> None:
    report = build_eirr_adaptive_critical_design_report()
    assert all(report["checks"].values())
    json.dumps(report, allow_nan=False)
