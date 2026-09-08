from __future__ import annotations

import numpy as np

from scripts.fit_certified_meet_fusion import development_split
from scripts.evaluate_certified_meet_fusion import paired_bootstrap, selective_metrics


def test_development_split_is_scene_grouped_and_90_30_30() -> None:
    scenes = np.repeat(np.arange(150), 40)
    train, validation, calibration = development_split(scenes)
    sets = [set(scenes[mask]) for mask in (train, validation, calibration)]
    assert [len(item) for item in sets] == [90, 30, 30]
    assert set.union(*sets) == set(range(150))
    assert all(not sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3))


def test_meet_is_dominated_and_monotone() -> None:
    geometry = np.asarray([[0.1, 0.4, 0.7], [0.2, 0.3, 0.9]])
    direct = np.asarray([[0.2, 0.3, 0.8], [0.1, 0.6, 0.7]])
    meet = np.minimum(geometry, direct)
    assert np.all(meet <= geometry)
    assert np.all(meet <= direct)
    assert np.all(np.diff(meet, axis=1) >= 0.0)


def test_selective_metrics_are_scene_grouped() -> None:
    labels = np.asarray([[0, 1], [0, 1], [0, 0], [1, 1]], dtype=np.float32)
    prediction = np.asarray([[0.9, 0.9], [0.1, 0.9], [0.8, 0.1], [0.9, 0.9]])
    scenes = np.asarray([0, 0, 1, 1])
    report, risk, coverage = selective_metrics(labels, prediction, scenes, 0.5)
    np.testing.assert_allclose(risk, [0.5, 0.5])
    np.testing.assert_allclose(coverage, [0.75, 0.75])
    assert report["scene_false_safe"] == 0.5
    assert report["coverage"] == 0.75


def test_paired_bootstrap_preserves_constant_scene_difference() -> None:
    result = paired_bootstrap(np.full(20, 0.03), seed=7, draws=100)
    assert abs(result["lower_95"] - 0.03) < 1e-12
    assert abs(result["upper_95"] - 0.03) < 1e-12
