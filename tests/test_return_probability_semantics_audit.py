from __future__ import annotations

import pytest

from scripts.audit_return_probability_semantics import (
    parse_args,
    probability_semantics_payload,
)


def variance_payload(*, nondegenerate: bool) -> dict[str, object]:
    return {
        "num_rollouts": 3,
        "mean": 12.5,
        "variance": 0.2 if nondegenerate else 0.0,
        "standard_deviation": 0.447 if nondegenerate else 0.0,
        "minimum": 12.0 if nondegenerate else 12.5,
        "maximum": 13.0 if nondegenerate else 12.5,
        "q50": 12.5,
        "q90": 12.9 if nondegenerate else 12.5,
        "q95": 12.95 if nondegenerate else 12.5,
        "unique_values_at_tolerance": 2 if nondegenerate else 1,
        "nondegenerate_conditional_distribution": nondegenerate,
        "distributional_headline_supported": nondegenerate,
        "interpretation": "synthetic",
    }


def test_fixed_snapshot_exact_oracle_selects_point_etg_semantics() -> None:
    payload = probability_semantics_payload(
        variance_audit=variance_payload(nondegenerate=False),
        estimator_declares_exact_deterministic_return=True,
        snapshot_sha256="a" * 64,
        repeated_rollout_seconds=1.5,
    )
    assert payload["status"] == "PASS_DETERMINISTIC_POINT_ETG_SEMANTICS"
    assert payload["passed"] is True
    assert payload["aleatoric_q90_q95_claim_authorized"] is False
    assert (
        payload["randomness_audit"]["repetition_index_is_not_a_disturbance_seed"]
        is True
    )


def test_nondegenerate_repetition_cannot_be_called_exact_deterministic() -> None:
    payload = probability_semantics_payload(
        variance_audit=variance_payload(nondegenerate=True),
        estimator_declares_exact_deterministic_return=True,
        snapshot_sha256="b" * 64,
        repeated_rollout_seconds=1.5,
    )
    assert payload["status"] == "FAIL_UNEXPECTED_CONDITIONAL_NONDETERMINISM"
    assert payload["passed"] is False
    assert payload["aleatoric_q90_q95_claim_authorized"] is False


def test_cli_requires_multiple_repetitions(tmp_path) -> None:
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--navigation-artifact",
                str(tmp_path / "artifact"),
                "--output-json",
                str(tmp_path / "audit.json"),
                "--num-rollouts",
                "1",
            ]
        )
