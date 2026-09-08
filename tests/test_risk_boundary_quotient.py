from __future__ import annotations

import json

import numpy as np

from scripts.verify_risk_boundary_quotient import (
    build_risk_boundary_quotient_report,
)


def test_risk_boundary_quotient_algebra() -> None:
    report = build_risk_boundary_quotient_report()
    assert report["status"] == "RISK_BOUNDARY_QUOTIENT_IDENTITIES_VERIFIED"
    assert all(report["checks"].values())
    assert report["exact_quotient"]["fiber_witness_diameter"] == 0.0
    assert report["approximate_quotient"]["fiber_witness_diameter"] > 0.0


def test_representation_error_bound_is_nontrivial() -> None:
    report = build_risk_boundary_quotient_report()
    approximate = report["approximate_quotient"]
    error = np.asarray(approximate["value_error"])
    envelope = np.asarray(approximate["pointwise_envelope"])
    assert np.max(np.abs(error)) > 0.0
    assert np.all(np.abs(error) <= envelope + 1e-13)
    assert np.max(envelope) < approximate["uniform_resolvent_bound"]


def test_report_is_strict_json_serializable() -> None:
    report = build_risk_boundary_quotient_report()
    encoded = json.dumps(report, allow_nan=False)
    assert "RISK_BOUNDARY_QUOTIENT_IDENTITIES_VERIFIED" in encoded
