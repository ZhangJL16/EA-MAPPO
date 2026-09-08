from __future__ import annotations

import numpy as np

from experiments.rechargeability_safety.risk_control import (
    hbb_upper_mean,
    select_rcps_threshold,
)
from scripts.calibrate_rcps_meet_threshold import coverage_curve, scene_loss_table


def test_hbb_zero_loss_exposes_75_scene_impossibility() -> None:
    assert np.isclose(hbb_upper_mean(np.zeros(75)), 0.06060421539057684)
    assert hbb_upper_mean(np.zeros(100)) < 0.05


def test_rcps_selects_first_fully_valid_conservative_suffix() -> None:
    thresholds = np.asarray([0.0, 0.5, 1.0, np.inf])
    losses = np.zeros((100, 4), dtype=np.float64)
    losses[:, 0] = 0.20
    losses[:6, 1] = 1.0
    coverage = np.asarray([1.0, 0.7, 0.2, 0.0])
    selected, bounds = select_rcps_threshold(losses, coverage, thresholds)
    assert selected.grid_index == 2
    assert selected.threshold == 1.0
    assert selected.nonvacuous
    assert bounds[1] > 0.05
    assert bounds[2] < 0.05


def test_scene_loss_and_coverage_curves_use_inclusive_acceptance() -> None:
    labels = np.asarray([[0.0, 1.0], [0.0, 0.0], [1.0, 1.0]])
    scores = np.asarray([[0.2, 0.8], [0.5, 0.9], [0.4, 0.7]])
    scenes = np.asarray([0, 0, 1])
    thresholds = np.asarray([0.5, 0.9, np.inf])
    table = scene_loss_table(labels, scores, scenes, thresholds)
    np.testing.assert_allclose(table[0], [2.0 / 3.0, 1.0 / 3.0, 0.0])
    np.testing.assert_allclose(table[1], 0.0)
    np.testing.assert_allclose(coverage_curve(scores, thresholds), [4.0 / 6.0, 1.0 / 6.0, 0.0])
