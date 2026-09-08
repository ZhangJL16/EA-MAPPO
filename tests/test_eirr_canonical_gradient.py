from __future__ import annotations

import json

import pytest

from scripts.verify_eirr_canonical_gradient import (
    build_eirr_canonical_gradient_report,
)


def test_canonical_gradient_matches_pathwise_derivative() -> None:
    report = build_eirr_canonical_gradient_report()
    derivative = report["pathwise_derivative"]
    assert report["status"] == "EIRR_CANONICAL_GRADIENT_IDENTITIES_VERIFIED"
    assert derivative["central_finite_difference"] == pytest.approx(
        derivative["canonical_gradient_inner_product"], abs=1e-9
    )


def test_efficiency_norm_and_least_favourable_direction() -> None:
    report = build_eirr_canonical_gradient_report()
    efficiency = report["efficiency"]
    assert efficiency["squared_gradient_norm"] == pytest.approx(
        efficiency["conditional_variance_formula"], abs=1e-13
    )
    assert efficiency["least_favourable_finite_derivative"] == pytest.approx(
        efficiency["sqrt_efficiency_variance"], abs=1e-9
    )
    assert efficiency["marginal_design_inner_product"] == pytest.approx(
        0.0, abs=1e-13
    )


def test_canonical_gradient_checks_and_json_contract() -> None:
    report = build_eirr_canonical_gradient_report()
    assert all(report["checks"].values())
    assert report["model"]["spectral_radius"] < 1.0
    json.dumps(report, allow_nan=False)
