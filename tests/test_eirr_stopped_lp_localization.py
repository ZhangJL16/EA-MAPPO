from __future__ import annotations

import json

import pytest

from scripts.verify_eirr_stopped_lp_localization import (
    build_eirr_stopped_lp_localization_report,
    stopped_lp_bounds,
)


def test_stopped_lp_bound_requires_power_above_one() -> None:
    with pytest.raises(ValueError, match="p>1"):
        stopped_lp_bounds(
            error_p_moment=0.1,
            moment_power=1.0,
            margin_exponent=1.0,
            margin_constant=1.0,
        )


def test_zero_moment_has_zero_decision_bounds() -> None:
    assert stopped_lp_bounds(
        error_p_moment=0.0,
        moment_power=2.0,
        margin_exponent=1.0,
        margin_constant=1.0,
    ) == (0.0, 0.0, 0.0, 0.0)


def test_lp_to_stopped_margin_powers_are_sharp() -> None:
    report = build_eirr_stopped_lp_localization_report()
    for row in report["sharp_power_rows"]:
        assert row["fitted_disagreement_power"] == pytest.approx(
            row["expected_disagreement_power"], abs=1e-12
        )
        assert row["fitted_boundary_loss_power"] == pytest.approx(
            row["expected_boundary_loss_power"], abs=1e-12
        )
    assert report["status"] == "EIRR_STOPPED_LP_LOCALIZATION_VERIFIED"
    assert all(report["checks"].values())
    json.dumps(report, allow_nan=False)
