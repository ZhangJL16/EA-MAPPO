from __future__ import annotations

import numpy as np
import torch

from scripts.explore_resource_cdf_learnability import (
    assess,
    binary_metrics,
    cdf_probability,
    energy_class,
    parse_args,
    sanitize_snapshot_velocity,
)


def test_failure_atom_and_finite_energy_bins_are_distinct() -> None:
    assert energy_class(None, capacity=100.0, finite_bins=8, support_capacity_multiple=4.0) == 8
    assert energy_class(25.0, capacity=100.0, finite_bins=8, support_capacity_multiple=4.0) == 0
    assert energy_class(450.0, capacity=100.0, finite_bins=8, support_capacity_multiple=4.0) == 7


def test_cdf_excludes_failure_atom_and_is_budget_monotone() -> None:
    probabilities = np.asarray([[0.1, 0.2, 0.3, 0.1, 0.3]])
    low = cdf_probability(probabilities, budget_fraction=1.0, support_capacity_multiple=4.0)
    high = cdf_probability(probabilities, budget_fraction=3.0, support_capacity_multiple=4.0)
    np.testing.assert_allclose(low, [0.1])
    np.testing.assert_allclose(high, [0.6])
    assert high[0] >= low[0]
    assert high[0] <= 1.0 - probabilities[0, -1] + 1e-12


def test_binary_metrics_are_exact_for_perfect_predictions() -> None:
    metrics = binary_metrics(np.asarray([0, 0, 1, 1]), np.asarray([0.0, 0.1, 0.9, 1.0]))
    assert metrics["auroc"] == 1.0
    assert metrics["auprc"] == 1.0
    assert metrics["accuracy_at_0_5"] == 1.0
    assert metrics["brier"] < 0.01


def test_assessment_does_not_promote_a_constant_head() -> None:
    folds = []
    labels = [0, 1] * 10
    for fold in range(3):
        folds.append(
            {
                "labels": labels,
                "cdf_probabilities": [0.5] * len(labels),
                "success_labels": labels,
                "success_probabilities": [0.5] * len(labels),
                "constant_probabilities": [0.5] * len(labels),
                "geometry_probabilities": [0.5] * len(labels),
            }
        )
    result = assess(folds)
    assert result["promotable"] is False
    assert result["status"] == "DO_NOT_PROMOTE_FROM_LEARNABILITY_PILOT"


def test_smoke_protocol_is_small_and_valid() -> None:
    args = parse_args(
        [
            "--artifact", "/tmp/a", "--checkpoint", "/tmp/c",
            "--upstream-dir", "/tmp/u", "--output-dir", "/tmp/o", "--smoke",
        ]
    )
    assert args.num_tasks == 5
    assert args.max_policy_steps == 20
    assert args.epochs == 2


def test_waiting_protocol_accepts_a_positive_upstream_pid() -> None:
    args = parse_args(
        [
            "--artifact", "/tmp/a", "--checkpoint", "/tmp/c",
            "--upstream-dir", "/tmp/u", "--output-dir", "/tmp/o",
            "--wait-for-upstream", "--upstream-pid", "123",
        ]
    )
    assert args.wait_for_upstream is True
    assert args.upstream_pid == 123


def test_bare_cuda_device_has_an_explicit_default_index() -> None:
    selected = torch.device("cuda")
    assert (0 if selected.index is None else selected.index) == 0


def test_snapshot_velocity_projects_only_float32_scale_overshoot() -> None:
    velocity = np.asarray([20.000002, 0.0, 10.000002], dtype=np.float32)
    projected = sanitize_snapshot_velocity(
        velocity,
        horizontal_limit=20.0,
        vertical_limit=10.0,
    )
    assert np.linalg.norm(projected[:2].astype(np.float64)) <= 20.0 + 1e-6
    assert abs(float(projected[2])) <= 10.0


def test_snapshot_velocity_rejects_real_physical_overshoot() -> None:
    with np.testing.assert_raises_regex(ValueError, "non-numerical"):
        sanitize_snapshot_velocity(
            np.asarray([20.01, 0.0, 0.0], dtype=np.float32),
            horizontal_limit=20.0,
            vertical_limit=10.0,
        )
