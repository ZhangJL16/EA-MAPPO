from __future__ import annotations

import json

from scripts.verify_eirr_finite_quotient_recovery import (
    build_eirr_finite_quotient_recovery_report,
)


def test_finite_quotient_recovers_above_separation_rate() -> None:
    report = build_eirr_finite_quotient_recovery_report()
    upper = report["upper_recovery"]
    assert upper["uniform_radius"] < upper["separation"] / 4.0
    assert upper["recoveries"] == upper["trials"]


def test_inverse_square_separation_has_nonvanishing_error_floor() -> None:
    report = build_eirr_finite_quotient_recovery_report()
    rows = report["lower_separation_boundary"]
    assert max(row["exact_kl"] for row in rows) < 0.11
    assert min(row["minimax_error_lower_bound"] for row in rows) > 0.38


def test_decoded_risk_witness_moments_define_neural_target() -> None:
    report = build_eirr_finite_quotient_recovery_report()
    neural = report["neural_decoder_corollary"]
    assert neural["maximum_reconstruction_error"] < neural["quarter_separation"]
    assert report["status"] == "EIRR_FINITE_QUOTIENT_RECOVERY_VERIFIED"
    assert all(report["checks"].values())
    json.dumps(report, allow_nan=False)
