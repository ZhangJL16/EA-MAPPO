from __future__ import annotations

import json

import pytest

from scripts.verify_first_passage_assouad_phase import (
    build_first_passage_assouad_report,
    killed_requirement,
    termination_for_requirement,
)


def test_killed_requirement_inverse_round_trip() -> None:
    requirement = killed_requirement(
        cost_multiplier=1.5,
        termination_probability=0.4,
        risk_lambda=0.7,
    )
    probability = termination_for_requirement(
        cost_multiplier=1.5,
        requirement=requirement,
        risk_lambda=0.7,
    )
    assert probability == pytest.approx(0.4, abs=1e-14)


def test_killed_requirement_rejects_improper_chain() -> None:
    with pytest.raises(ValueError, match="proper killed"):
        killed_requirement(
            cost_multiplier=2.0,
            termination_probability=0.4,
            risk_lambda=1.0,
        )


def test_first_passage_assouad_phase_algebra_and_exponents() -> None:
    report = build_first_passage_assouad_report()
    assert report["status"] == "FIRST_PASSAGE_ASSOUAD_PHASE_VERIFIED"
    assert all(report["checks"].values())
    assert report["parameters"]["compatibility_alpha_kappa_le_dimension"] is True
    json.dumps(report, allow_nan=False)
