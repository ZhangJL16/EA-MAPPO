from __future__ import annotations

import json

import pytest

from scripts.verify_eirr_critical_phase import (
    build_eirr_critical_phase_report,
)


def test_critical_phase_matches_bernoulli_information() -> None:
    report = build_eirr_critical_phase_report()
    assert report["status"] == "EIRR_QUOTIENT_TRANSIENCE_PHASE_VERIFIED"
    assert all(report["checks"].values())


def test_raw_and_log_risk_have_distinct_gap_exponents() -> None:
    report = build_eirr_critical_phase_report()
    scaling = report["scaling"]
    assert scaling["normalized_raw_log_log_slope"] == pytest.approx(-4.0)
    assert scaling["normalized_log_risk_log_log_slope"] == pytest.approx(-2.0)
    assert scaling["raw_consistency_information"] == "n * mu_h * Delta^4"
    assert scaling["log_requirement_consistency_information"] == (
        "n * mu_h * Delta^2"
    )


def test_critical_phase_json_contract_and_structural_support_boundary() -> None:
    report = build_eirr_critical_phase_report()
    model = report["model"]
    assert model["raw_target_cell_source_coverage_can_be_zero"] is True
    assert model["pooled_coverage_requires_exact_shared_primitive"] is True
    json.dumps(report, allow_nan=False)
