from __future__ import annotations

import json

import pytest

from scripts.verify_linear_risk_quotient import (
    build_linear_risk_quotient_report,
)


def test_linear_risk_quotient_exact_rate_and_support_separation() -> None:
    report = build_linear_risk_quotient_report()
    assert report["status"] == "LINEAR_RISK_QUOTIENT_THEOREM_VERIFIED"
    assert all(report["checks"].values())
    example = report["raw_support_example"]
    assert example["target_absolutely_continuous_wrt_source_support"] is False
    assert example["quotient_coefficient"] == pytest.approx(1.0)
    assert example["exact_minimax_mse"] == pytest.approx(
        example["estimator_variance"]
    )


def test_nonidentifiable_representation_has_positive_floor() -> None:
    report = build_linear_risk_quotient_report()
    blocked = report["nonidentifiable_slice"]
    assert blocked["source_mean_difference_norm"] == pytest.approx(0.0)
    assert blocked["indistinguishable_mse_lower_bound"] > 0.0


def test_decision_pair_and_json_contract() -> None:
    report = build_linear_risk_quotient_report()
    pair = report["decision_lower_pair"]
    assert pair["mahalanobis_distance"] == pytest.approx(1.0)
    assert pair["exact_equal_prior_testing_error"] == pytest.approx(
        0.3085375387259869
    )
    json.dumps(report, allow_nan=False)
