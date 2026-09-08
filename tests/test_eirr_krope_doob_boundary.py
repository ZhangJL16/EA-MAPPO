from __future__ import annotations

import json

import pytest

from scripts.verify_eirr_krope_doob_boundary import (
    build_eirr_krope_doob_boundary_report,
)


def test_same_mean_interfaces_have_different_exponential_risk() -> None:
    report = build_eirr_krope_doob_boundary_report()
    collapse = report["risk_neutral_collapse"]
    assert collapse["deterministic_mean_cost"] == pytest.approx(
        collapse["mixture_mean_cost"]
    )
    assert collapse["mixture_exponential_moment"] > (
        collapse["deterministic_exponential_moment"]
    )
    assert collapse["exact_exponential_moment_gap"] == pytest.approx(
        collapse["mixture_exponential_moment"]
        - collapse["deterministic_exponential_moment"]
    )


def test_raw_nuisance_coverage_diverges_but_quotient_does_not() -> None:
    rows = build_eirr_krope_doob_boundary_report()["raw_vs_quotient_coverage"]
    assert [row["raw_chi_square"] for row in rows] == pytest.approx(
        [3.0, 15.0, 63.0, 255.0]
    )
    assert all(row["risk_quotient_chi_square"] == 0.0 for row in rows)


def test_oracle_doob_reduction_closes_the_baseline_boundary() -> None:
    report = build_eirr_krope_doob_boundary_report()
    reduction = report["oracle_doob_reduction"]
    assert reduction["matrix_spectral_radius"] < reduction["discount"]
    assert reduction["row_sums_with_cemetery"] == pytest.approx([1.0, 1.0])
    assert reduction["similarity_reconstruction_error"] < 1e-12
    assert report["status"] == "EIRR_KROPE_DOOB_BOUNDARY_VERIFIED"
    assert all(report["checks"].values())
    json.dumps(report, allow_nan=False)
