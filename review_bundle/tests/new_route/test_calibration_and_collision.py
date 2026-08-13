from __future__ import annotations

import numpy as np
import pytest
import torch

from safety.calibration import SplitConformalUpperBound
from safety.collision import (
    CollisionRiskBudget,
    CollisionRiskEnsemble,
    OneSidedRiskCalibrator,
    short_horizon_labels,
)


def test_split_conformal_upper_bound_and_report() -> None:
    predicted = np.arange(20, dtype=np.float64)
    observed = predicted + np.linspace(-0.2, 0.8, 20)
    calibrator = SplitConformalUpperBound(alpha=0.1).fit(predicted, observed)
    bounds = calibrator.predict(predicted)
    report = calibrator.report(bounds, observed)
    assert report.sample_count == 20
    assert report.coverage >= 0.9


def test_small_calibration_set_can_yield_infinite_valid_bound() -> None:
    calibrator = SplitConformalUpperBound(alpha=0.01).fit(np.array([0.0]), np.array([1.0]))
    assert np.isinf(calibrator.predict(np.array([0.2]))[0])


def test_collision_calibrator_outputs_probability_domain() -> None:
    model = OneSidedRiskCalibrator(alpha=0.2).fit(
        np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
        np.array([0, 0, 0, 1, 1]),
    )
    upper = model.upper(np.array([0.0, 0.5, 1.0]))
    assert np.all((0.0 <= upper) & (upper <= 1.0))


def test_collision_risk_budget_never_overspends() -> None:
    ledger = CollisionRiskBudget(0.1)
    allocations = [ledger.geometric_allocation(step, 0.5) for step in range(10)]
    for allocation in allocations:
        ledger.spend(allocation)
    assert sum(allocations) <= 0.1 + 1e-12
    with pytest.raises(ValueError, match="exceeds"):
        ledger.spend(0.1)


def test_short_horizon_collision_labels_include_near_events() -> None:
    labels = short_horizon_labels(
        collision=np.array([False, False, False, True, False]),
        near_collision=np.array([False, True, False, False, False]),
        horizon=2,
    )
    np.testing.assert_array_equal(labels, np.array([1, 1, 1, 1, 0], dtype=np.float32))


def test_collision_ensemble_separates_uncertainty_terms() -> None:
    torch.manual_seed(4)
    ensemble = CollisionRiskEnsemble(6, members=3)
    estimate = ensemble.estimate(torch.randn(7, 6))
    assert estimate.mean_probability.shape == (7,)
    assert np.all(estimate.aleatoric_std >= 0.0)
    assert np.all(estimate.epistemic_std >= 0.0)
    assert np.all(estimate.uncalibrated_upper >= estimate.mean_probability)
