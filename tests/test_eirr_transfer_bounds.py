from __future__ import annotations

import numpy as np
import pytest

from experiments.energy_mc.eirr_transfer import (
    interface_transfer_minimax_lower_bound,
    resolvent_plugin_certificate,
)
from scripts.verify_eirr_transfer_bounds import (
    build_eirr_transfer_verification_report,
)


def test_finite_interface_upper_and_lower_bounds() -> None:
    report = build_eirr_transfer_verification_report()
    assert report["status"] == "EIRR_TRANSFER_BOUNDS_VERIFIED"
    assert report["upper_bound_holds"] is True
    assert report["constant_error_floor_9_over_32_holds"] is True
    assert report["decision_error_floor_3_over_8_holds"] is True
    assert report["variance_is_linear_order_in_exponential_weight"] is True
    assert report["lower_bound"]["constant_error_regime"] is True
    assert report["lower_bound"]["maximum_transitions_for_constant_error"] == pytest.approx(50.0)


def test_transfer_lower_bound_exposes_inverse_coverage_scaling() -> None:
    common = dict(
        risk_parameter=1.0,
        high_resource=float(np.log(20.0)),
        number_of_source_transitions=10,
    )
    half = interface_transfer_minimax_lower_bound(
        source_interface_probability=0.5,
        **common,
    )
    tenth = interface_transfer_minimax_lower_bound(
        source_interface_probability=0.1,
        **common,
    )
    assert tenth.maximum_transitions_for_constant_error == pytest.approx(
        5.0 * half.maximum_transitions_for_constant_error
    )
    assert tenth.minimax_absolute_error_lower > half.minimax_absolute_error_lower


def test_plugin_certificate_fails_closed_when_robust_radius_is_too_large() -> None:
    with pytest.raises(ValueError, match="strictly below one"):
        resolvent_plugin_certificate(
            learned_transient_operator=np.asarray([[0.5]]),
            learned_terminal_vector=np.asarray([1.0]),
            transient_operator_radius_inf=0.5,
            terminal_vector_radius_inf=0.1,
        )
