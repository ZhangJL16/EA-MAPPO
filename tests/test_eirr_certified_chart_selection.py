from __future__ import annotations

import json

import numpy as np
import pytest

from scripts.verify_eirr_certified_chart_selection import (
    build_certified_chart_selection_report,
    partition_diameter,
    select_certified_chart,
)


def test_partition_diameter_matches_within_group_maximum() -> None:
    mu = np.asarray([[0.0], [0.2], [0.7]])
    labels = np.asarray([0, 0, 1])
    assert partition_diameter(mu, labels) == pytest.approx(0.2)


def test_selector_rejects_negative_variance_term() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        select_certified_chart(
            true_mu=np.asarray([[0.0], [1.0]]),
            estimated_mu=np.asarray([[0.0], [1.0]]),
            partitions=[np.asarray([0, 1])],
            variance_terms=np.asarray([-1.0]),
            amplification=1.0,
            witness_radius=0.1,
        )


def test_certified_chart_selection_report() -> None:
    report = build_certified_chart_selection_report()
    assert report["status"] == "EIRR_CERTIFIED_CHART_SELECTION_VERIFIED"
    assert all(report["checks"].values())
    assert report["event_sandwich_violations"] == 0
    assert report["event_oracle_violations"] == 0
    json.dumps(report, allow_nan=False)
