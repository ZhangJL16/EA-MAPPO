from __future__ import annotations

import numpy as np
import pytest

from scripts.verify_eirr_finite_state import (
    build_verification_report,
    clipped_certificate_penalty,
    doob_discount_reduction,
    common_support_sample_lower_bound,
    no_support_target_mgf,
    perfect_support_risk_ratio,
    risk_resolvent,
)


def test_finite_state_eirr_identities_hold_to_machine_precision() -> None:
    report = build_verification_report()
    assert report["status"] == "FINITE_STATE_IDENTITIES_VERIFIED"
    assert report["spectral_radius"] < 1.0
    assert report["perturbation_identity_max_error"] < 1e-12
    assert report["residual_identity_error"] < 1e-12
    assert report["chernoff_grid_example"]["bound_satisfied"] is True
    assert report["doob_discount_reduction"]["row_sum_error"] < 1e-12
    assert report["doob_discount_reduction"]["matrix_reconstruction_error"] < 1e-12
    assert report["doob_discount_reduction"]["resolvent_reconstruction_error"] < 1e-12


def test_doob_discount_reduction_rejects_invalid_gamma() -> None:
    matrix = np.asarray([[0.2, 0.1], [0.05, 0.25]])
    with pytest.raises(ValueError, match="gamma"):
        doob_discount_reduction(matrix, gamma=0.3)


def test_no_support_separation_is_exponentially_cost_sensitive() -> None:
    low = no_support_target_mgf(
        branch_probability=0.05,
        risk_lambda=0.5,
        terminal_cost=0.0,
    )
    high = no_support_target_mgf(
        branch_probability=0.05,
        risk_lambda=0.5,
        terminal_cost=8.0,
    )
    assert low == pytest.approx(1.0)
    assert high - low == pytest.approx(0.05 * (np.exp(4.0) - 1.0))


def test_perfect_ordinary_support_does_not_bound_risk_concentration() -> None:
    small = perfect_support_risk_ratio(
        high_prefix_probability=0.05,
        risk_lambda=0.5,
        prefix_cost=2.0,
    )
    large = perfect_support_risk_ratio(
        high_prefix_probability=0.05,
        risk_lambda=0.5,
        prefix_cost=12.0,
    )
    assert small > 1.0
    assert large > 10.0 * small


def test_common_support_can_require_exponentially_many_trajectories() -> None:
    moderate = common_support_sample_lower_bound(risk_lambda=0.5, high_cost=8.0)
    severe = common_support_sample_lower_bound(risk_lambda=0.5, high_cost=16.0)
    assert severe["maximum_trajectory_count"] > 50.0 * moderate["maximum_trajectory_count"]
    assert moderate["mgf_absolute_loss_lower_bound"] == pytest.approx(9.0 / 32.0)
    assert moderate["log_mgf_separation"] > moderate["log_mgf_absolute_loss_lower_bound"]


def test_optimized_clipped_certificate_has_inverse_sqrt_rate() -> None:
    small = clipped_certificate_penalty(
        sample_count=1000,
        residual_bound=1.0,
        ratio_second_moment=4.0,
        ratio_l1_error=0.0,
        alpha=0.05,
    )
    large = clipped_certificate_penalty(
        sample_count=4000,
        residual_bound=1.0,
        ratio_second_moment=4.0,
        ratio_l1_error=0.0,
        alpha=0.05,
    )
    assert large["total_nonempirical_penalty"] == pytest.approx(
        small["total_nonempirical_penalty"] / 2.0
    )
    assert small["clipping_bias_penalty"] == pytest.approx(small["range_penalty"])


def test_resolvent_rejects_nontransient_exponential_operator() -> None:
    with pytest.raises(ValueError, match="spectral radius"):
        risk_resolvent(np.asarray([[1.0]]), np.asarray([1.0]))
