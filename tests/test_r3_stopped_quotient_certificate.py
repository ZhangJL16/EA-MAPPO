from __future__ import annotations

import numpy as np
import pytest

from scripts.analyze_r3_stopped_quotient_certificate import (
    partition_diameter_from_atoms,
    pushed_forward_overlap,
)


def test_pushforward_overlap_removes_irrelevant_nuisance() -> None:
    source_raw = np.asarray(["z0v0", "z0v1", "z1v0", "z1v1"])
    target_raw = np.asarray(["z0v0", "z1v0"])
    weights = np.ones(2)
    raw = pushed_forward_overlap(source_raw, target_raw, weights)
    source_quotient = np.asarray(["z0", "z0", "z1", "z1"])
    target_quotient = np.asarray(["z0", "z1"])
    quotient = pushed_forward_overlap(source_quotient, target_quotient, weights)
    assert raw == pytest.approx(4.0)
    assert quotient == pytest.approx(2.0)


def test_pushforward_overlap_reports_missing_support() -> None:
    value = pushed_forward_overlap(
        np.asarray(["a"]), np.asarray(["b"]), np.asarray([1.0])
    )
    assert value == float("inf")


def test_partition_diameter_uses_only_merged_atoms() -> None:
    means = {
        "a": np.asarray([0.0, 0.2]),
        "b": np.asarray([0.3, 0.1]),
        "c": np.asarray([0.9, 0.9]),
    }
    mapping = {"a": "x", "b": "x", "c": "y"}
    assert partition_diameter_from_atoms(means, mapping) == pytest.approx(0.3)
