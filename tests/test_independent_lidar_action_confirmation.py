from __future__ import annotations

import numpy as np

from scripts.run_independent_lidar_action_confirmation import (
    acceptance_metrics,
    paired_bootstrap_interval,
    scene_paired_differences,
)


def test_acceptance_metrics_separate_joint_conditional_and_coverage() -> None:
    labels = np.asarray([0, 0, 1, 1], dtype=np.float32)
    probability = np.asarray([0.95, 0.20, 0.99, 0.50], dtype=np.float32)
    result = acceptance_metrics(labels, probability)
    assert result["joint_dangerous_false_safe"] == 0.25
    assert result["conditional_false_safe_given_infeasible"] == 0.5
    assert result["unsafe_among_accepted"] == 0.5
    assert result["declaration_coverage"] == 0.5


def test_scene_paired_bootstrap_preserves_scene_as_unit() -> None:
    labels = np.asarray([[0.0], [1.0], [0.0], [1.0]])
    candidate = np.asarray([[0.2], [0.8], [0.1], [0.9]])
    baseline = np.asarray([[0.4], [0.6], [0.3], [0.7]])
    scenes = np.asarray([10, 10, 30, 30])
    brier, dangerous = scene_paired_differences(labels, candidate, baseline, scenes)
    assert brier.shape == dangerous.shape == (2,)
    assert np.all(brier < 0.0)
    first = paired_bootstrap_interval(brier, seed=7, draws=500)
    second = paired_bootstrap_interval(brier, seed=7, draws=500)
    assert first == second
    assert first["scene_count"] == 2
    assert first["upper_95"] < 0.0
