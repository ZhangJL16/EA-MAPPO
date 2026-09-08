from __future__ import annotations

import json

import pytest

from scripts.verify_stopped_pareto_contraction import (
    build_stopped_pareto_contraction_report,
    dual_stopped_pareto_certificate,
    stopped_pareto_certificate,
)


def test_stopped_pareto_certificate_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="MSE"):
        stopped_pareto_certificate(
            score_mse_bound=-0.1,
            margin_constant=1.0,
            margin_exponent=1.0,
            margin_radius=1.0,
            throughput_range=1.0,
        )


def test_zero_stopped_score_risk_has_zero_pareto_rectangle() -> None:
    result = stopped_pareto_certificate(
        score_mse_bound=0.0,
        margin_constant=1.0,
        margin_exponent=1.0,
        margin_radius=1.0,
        throughput_range=3.0,
    )
    assert result["first_disagreement_probability_bound"] == 0.0
    assert result["stranding_deviation_bound"] == 0.0
    assert result["throughput_deviation_bound"] == 0.0


def test_contraction_report_separates_uniform_and_integrated_regimes() -> None:
    report = build_stopped_pareto_contraction_report()
    assert report["status"] == "STOPPED_PARETO_CONTRACTION_VERIFIED"
    assert all(report["checks"].values())
    assert all(
        row["integrated_and_uniform_regimes_are_distinct"]
        for row in report["information_regime_rows"]
    )
    json.dumps(report, allow_nan=False)


def test_dual_certificate_uses_uniform_route_only_inside_margin_radius() -> None:
    strong = dual_stopped_pareto_certificate(
        score_mse_bound=1e-2,
        uniform_score_error_bound=1e-3,
        margin_constant=1.0,
        margin_exponent=1.0,
        margin_radius=0.1,
        throughput_range=2.0,
    )
    weak = dual_stopped_pareto_certificate(
        score_mse_bound=1e-8,
        uniform_score_error_bound=0.2,
        margin_constant=1.0,
        margin_exponent=1.0,
        margin_radius=0.1,
        throughput_range=2.0,
    )
    assert strong["selected_disagreement_route"] == "uniform_active_cell"
    assert weak["selected_disagreement_route"] == "integrated_stopped_l2"
    assert weak["uniform_route_valid"] is False
