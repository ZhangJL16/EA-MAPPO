from __future__ import annotations

import json

import pytest

from scripts.verify_eirr_predictable_martingale_quotient import (
    build_eirr_predictable_martingale_quotient_report,
    predictable_first_hit_radius,
)


def test_predictable_radius_rejects_invalid_count() -> None:
    with pytest.raises(ValueError, match="counts"):
        predictable_first_hit_radius(
            samples_per_query=0,
            query_count=1,
            witness_dimension=1,
            witness_bound=1.0,
            holder_bias=0.0,
            failure_probability=0.05,
        )


def test_adaptive_first_hit_trajectory_obeys_certificate() -> None:
    report = build_eirr_predictable_martingale_quotient_report()
    adaptive = report["adaptive_first_hit"]
    assert adaptive["simultaneous_coverages"] == adaptive["trials"]
    assert adaptive["maximum_observed_error"] < adaptive["simultaneous_radius"]


def test_replay_duplicates_do_not_create_information() -> None:
    report = build_eirr_predictable_martingale_quotient_report()
    replay = report["replay_duplicate_audit"]
    assert replay["unique_mean"] == pytest.approx(replay["replayed_mean"])
    assert replay["invalid_naive_replay_radius"] == pytest.approx(
        replay["valid_unique_transition_radius"] / replay["replay_factor"] ** 0.5
    )
    assert report["rate_check"]["fitted_log_slope"] == pytest.approx(-0.5)
    assert report["status"] == "EIRR_PREDICTABLE_MARTINGALE_QUOTIENT_VERIFIED"
    assert all(report["checks"].values())
    json.dumps(report, allow_nan=False)
