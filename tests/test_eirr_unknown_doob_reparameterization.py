from __future__ import annotations

import json

import numpy as np
import pytest

from scripts.verify_eirr_unknown_doob_reparameterization import (
    build_eirr_unknown_doob_reparameterization_report,
    doob_transform,
    inverse_doob_transform,
)


def test_doob_transform_rejects_invalid_discount() -> None:
    matrix = np.asarray([[0.8]])
    with pytest.raises(ValueError, match="discount"):
        doob_transform(matrix, 0.7)


def test_doob_transform_is_invertible_and_value_preserving() -> None:
    matrix = np.asarray([[0.2, 0.1], [0.05, 0.3]])
    scale, transformed, cemetery = doob_transform(matrix, 0.8)
    reconstructed = inverse_doob_transform(
        value_scale=scale, transformed=transformed, discount=0.8
    )
    assert reconstructed == pytest.approx(matrix)
    assert np.sum(transformed, axis=1) + cemetery == pytest.approx([1.0, 1.0])


def test_unknown_doob_plugin_has_no_algebraic_rate_gain() -> None:
    report = build_eirr_unknown_doob_reparameterization_report()
    identity = report["finite_matrix_identity"]
    assert identity["inverse_error"] < 1e-12
    assert identity["value_identity_error"] < 1e-12
    assert identity["plugin_identity_error"] < 1e-12
    assert report["status"] == "EIRR_UNKNOWN_DOOB_REPARAMETERIZATION_VERIFIED"
    assert all(report["checks"].values())
    json.dumps(report, allow_nan=False)
