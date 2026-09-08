from __future__ import annotations

import numpy as np

from scripts.run_resource_cdf_scaling_experiment import (
    decision,
    expand_budget_queries,
    parse_args,
    representation_arrays,
    task_weights,
)


def test_budget_queries_create_finite_energy_boundary_cases() -> None:
    labels = expand_budget_queries(
        np.asarray([True, True, False]),
        np.asarray([0.10, 0.30, np.nan]),
        np.asarray([0.08, 0.20, 0.36]),
    )
    np.testing.assert_array_equal(
        labels,
        np.asarray([[0, 1, 1], [0, 0, 1], [0, 0, 0]]),
    )


def test_representation_layout_preserves_raw_and_builds_compact() -> None:
    raw = np.arange(2 * 21, dtype=np.float32).reshape(2, 21)
    arrays = representation_arrays(raw, observation_dim=10)
    assert arrays["raw"].shape == (2, 21)
    assert arrays["task_observation"].shape == (2, 10)
    assert arrays["charger_compact"].shape == (2, 7)
    assert arrays["position_and_horizon"].shape == (2, 4)
    assert arrays["compact"].shape == (2, 18)


def test_task_weights_equalize_tasks_with_different_snapshot_counts() -> None:
    task = np.asarray([0, 0, 0, 1])
    weights = task_weights(task)
    assert weights.mean() == np.float32(1.0)
    assert np.isclose(weights[task == 0].sum(), weights[task == 1].sum())


def test_default_scaling_protocol_is_fixed() -> None:
    args = parse_args(
        [
            "--artifact", "/tmp/a",
            "--checkpoint", "/tmp/c",
            "--upstream-result", "/tmp/u",
            "--output-dir", "/tmp/o",
        ]
    )
    assert args.num_tasks == 300
    assert args.epoch_checkpoints == [25, 100, 400]
    assert args.representations == ["raw", "frozen_encoder", "compact"]
    assert len(args.budget_fractions) == 8


def test_smoke_decision_can_name_its_reduced_primary() -> None:
    metric = {
        "count": 20,
        "positive_count": 0,
        "auroc": None,
        "ece_10_bin": 0.0,
        "brier": 0.1,
    }
    result = decision(
        {"compact@2": {"overall": metric}},
        {"geometry": {"brier": 0.1}},
        primary_key="compact@2",
    )
    assert result["primary_configuration"] == "compact@2"
    assert result["promotable"] is False
