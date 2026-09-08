from __future__ import annotations

import json

import pytest

from scripts.verify_eirr_perron_phase import (
    build_eirr_perron_phase_report,
)


def test_perron_nondegenerate_limits_and_exponents() -> None:
    report = build_eirr_perron_phase_report()
    case = report["nondegenerate"]
    assert report["status"] == "EIRR_PERRON_PHASE_SPLIT_VERIFIED"
    assert case["raw_limit_relative_error_at_smallest_gap"] < 0.02
    assert case["log_limit_relative_error_at_smallest_gap"] < 0.02
    assert case["tail_raw_log_log_slope"] == pytest.approx(-4.0, abs=0.08)
    assert case["tail_log_risk_log_log_slope"] == pytest.approx(
        -2.0, abs=0.08
    )


def test_projected_noise_degeneracy_changes_the_phase() -> None:
    report = build_eirr_perron_phase_report()
    case = report["degenerate"]
    assert case["maximum_projected_noise_variance"] < 1e-25
    assert case["tail_raw_log_log_slope"] == pytest.approx(-2.0, abs=0.08)
    assert case["tail_log_risk_log_log_slope"] == pytest.approx(0.0, abs=0.08)


def test_perron_phase_checks_and_json_contract() -> None:
    report = build_eirr_perron_phase_report()
    assert all(report["checks"].values())
    json.dumps(report, allow_nan=False)
