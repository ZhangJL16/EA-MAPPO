from __future__ import annotations

import json

import pytest

from scripts.verify_two_layer_orthogonal_eirr import (
    build_two_layer_orthogonal_eirr_report,
)


def test_two_layer_score_has_block_double_robustness() -> None:
    report = build_two_layer_orthogonal_eirr_report()
    robust = report["robustness"]
    assert report["status"] == "TWO_LAYER_ORTHOGONAL_EIRR_IDENTITIES_VERIFIED"
    assert robust["exact_ratio_arbitrary_model_score"] == pytest.approx(
        robust["target"], abs=1e-13
    )
    assert robust["exact_model_arbitrary_ratio_score"] == pytest.approx(
        robust["target"], abs=1e-13
    )


def test_product_bias_identity_and_l2_bound() -> None:
    report = build_two_layer_orthogonal_eirr_report()
    bias = report["product_bias"]
    assert bias["actual_bias"] == pytest.approx(
        bias["identity_sum"], abs=1e-13
    )
    assert abs(bias["actual_bias"]) <= bias["l2_product_bound"] + 1e-13
    assert abs(bias["actual_bias"]) > 1e-8


def test_two_layer_verifier_checks_and_json_contract() -> None:
    report = build_two_layer_orthogonal_eirr_report()
    assert all(report["checks"].values())
    assert report["model"]["spectral_radius"] < 1.0
    json.dumps(report, allow_nan=False)
