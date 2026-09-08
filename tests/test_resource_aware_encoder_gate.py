from __future__ import annotations

import numpy as np
import torch

from scripts.run_resource_aware_encoder_gate import (
    calibration_aware_loss,
    cdf_from_logits,
    cdf_query_mask,
    energy_targets,
    gate_decision,
    nested_task_split,
    select_best_epoch,
    threshold_aligned_edges,
)


def test_nested_split_is_disjoint_complete_and_bucket_balanced() -> None:
    tasks = np.arange(300)
    buckets = {index: f"bucket-{index // 60}" for index in tasks}
    split = nested_task_split(
        tasks,
        buckets,
        fold=1,
        split_seed=123,
        validation_per_bucket=6,
        calibration_per_bucket=6,
    )
    assert [len(split[name]) for name in ("selection_train", "validation", "calibration", "test")] == [140, 30, 30, 100]
    joined = [index for values in split.values() for index in values]
    assert len(joined) == len(set(joined)) == 300
    for name, expected in (("validation", 6), ("calibration", 6), ("test", 20)):
        for bucket in set(buckets.values()):
            assert sum(buckets[index] == bucket for index in split[name]) == expected


def test_threshold_edges_include_every_budget_exactly() -> None:
    budgets = np.asarray([0.08, 0.20, 0.36])
    edges = threshold_aligned_edges(support=1.5, uniform_bins=64, budgets=budgets)
    for budget in budgets:
        assert np.any(edges == budget)
    assert edges[-1] == 1.5
    assert np.all(np.diff(edges) > 0.0)


def test_energy_targets_preserve_failure_atom_and_budget_boundary() -> None:
    budgets = np.asarray([0.08, 0.20])
    edges = threshold_aligned_edges(support=1.5, uniform_bins=64, budgets=budgets)
    targets = energy_targets(np.asarray([0.079, 0.08, 0.081, np.nan]), edges)
    assert targets[-1] == edges.size
    mask = cdf_query_mask(edges, budgets)
    assert mask[targets[0], 0] == 1.0
    assert mask[targets[1], 0] == 1.0
    assert mask[targets[2], 0] == 0.0


def test_cdf_queries_are_monotone_and_exclude_failure_atom() -> None:
    edges = np.asarray([0.1, 0.2, 0.3])
    budgets = np.asarray([0.1, 0.2, 0.3])
    mask = torch.as_tensor(cdf_query_mask(edges, budgets))
    logits = torch.log(torch.tensor([[0.1, 0.2, 0.3, 0.4]]))
    probability = cdf_from_logits(logits, mask)
    torch.testing.assert_close(probability, torch.tensor([[0.1, 0.3, 0.6]]))
    assert torch.all(probability[:, 1:] >= probability[:, :-1])


def test_calibration_loss_is_finite_and_has_gradients() -> None:
    logits = torch.tensor(
        [[1.0, 0.0, -1.0], [0.0, 1.0, -1.0]], requires_grad=True
    )
    targets = torch.tensor([0, 1])
    labels = torch.tensor([[1.0, 1.0], [0.0, 1.0]])
    weights = torch.ones(2)
    mask = torch.tensor([[1.0, 1.0], [0.0, 1.0]])
    loss, pieces = calibration_aware_loss(
        logits,
        targets,
        labels,
        weights,
        mask,
        calibration_weight=1.0,
        normalize_nll=True,
    )
    assert torch.isfinite(loss)
    assert pieces["nll"] > 0.0 and pieces["brier"] >= 0.0
    loss.backward()
    assert logits.grad is not None and torch.all(torch.isfinite(logits.grad))


def test_best_epoch_uses_validation_brier_then_earliest_epoch() -> None:
    history = [
        {"epoch": 10, "validation_brier": 0.2},
        {"epoch": 20, "validation_brier": 0.1},
        {"epoch": 30, "validation_brier": 0.1},
    ]
    assert select_best_epoch(history) == 20


def test_gate_requires_fold_consistency_and_geometry_comparison() -> None:
    metric = {
        "count": 9600,
        "positive_count": 5000,
        "brier": 0.08,
        "ece_10_bin": 0.05,
        "auroc": 0.92,
    }
    matrix = {"trainable_calibrated": {"overall": metric}}
    baselines = {"geometry": {"brier": 0.10, "auroc": 0.925}}
    fold_results = []
    baseline_folds = []
    for fold, model_probability in enumerate((0.8, 0.8, 0.4)):
        labels = [1, 0]
        fold_results.append(
            {
                "variant": "trainable_calibrated",
                "fold": fold,
                "test": {"labels": labels, "probability": [model_probability, 0.2]},
            }
        )
        baseline_folds.append(
            {"labels": labels, "geometry_probability": [0.7, 0.3]}
        )
    result = gate_decision(
        matrix,
        baselines,
        fold_results,
        baseline_folds,
        primary_variant="trainable_calibrated",
    )
    assert result["promotable"] is True
    assert result["checks"]["primary_brier_better_on_at_least_two_folds"] is True
