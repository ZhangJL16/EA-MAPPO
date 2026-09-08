from __future__ import annotations

import json

import numpy as np
import pytest

from scripts.verify_eirr_crossfit_stopped_quotient import (
    build_crossfit_stopped_quotient_report,
    holder_rate_powers,
    local_shift_coefficient,
)


def test_local_shift_coefficient_rejects_missing_source_support() -> None:
    value = local_shift_coefficient(
        np.asarray([1.0, 0.0]),
        np.asarray([0.0, 1.0]),
        np.asarray([0, 1]),
    )
    assert value == float("inf")


def test_holder_rate_powers_require_positive_parameters() -> None:
    with pytest.raises(ValueError, match="positive"):
        holder_rate_powers(
            smoothness=1.0, dimension=0.0, margin_exponent=1.0
        )


def test_crossfit_quotient_rate_and_overlap_algebra() -> None:
    report = build_crossfit_stopped_quotient_report()
    assert report["status"] == "EIRR_CROSSFIT_STOPPED_QUOTIENT_VERIFIED"
    assert all(report["checks"].values())
    assert report["overlap_rows"][-1]["raw_to_quotient_ratio"] == pytest.approx(
        64.0
    )
    json.dumps(report, allow_nan=False)
