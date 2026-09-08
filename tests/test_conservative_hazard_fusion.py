from __future__ import annotations

import numpy as np

from scripts.run_conservative_hazard_fusion_gate import (
    conformal_risk_threshold,
    four_way_scene_split,
    per_scene_false_safe,
)


def test_four_way_split_keeps_complete_scenes_and_roles_disjoint() -> None:
    scene_ids = np.asarray([0, 1, 30, 31, 60, 61, 90, 91, 120, 121] * 4)
    scenes = np.repeat(scene_ids, 3)
    # Use a dense 150-scene case for the role-count contract.
    scenes = np.repeat(np.arange(150), 7)
    for fold in range(5):
        masks = four_way_scene_split(scenes, outer_fold=fold)
        sets = [set(scenes[mask]) for mask in masks]
        assert [len(value) for value in sets] == [60, 30, 30, 30]
        assert set.union(*sets) == set(range(150))
        assert all(not sets[left] & sets[right] for left in range(4) for right in range(left + 1, 4))


def test_crc_threshold_uses_scene_loss_and_finite_sample_correction() -> None:
    labels = np.zeros((40, 2), dtype=np.float32)
    prediction = np.linspace(0.0, 1.0, 80, dtype=np.float32).reshape(40, 2)
    scenes = np.repeat(np.arange(20), 2)
    result = conformal_risk_threshold(labels, prediction, scenes, alpha=0.10)
    losses = per_scene_false_safe(labels, prediction, scenes, float(result["threshold"]))
    corrected = 20.0 / 21.0 * float(np.mean(losses)) + 1.0 / 21.0
    assert corrected <= 0.10 + 1e-12
    assert abs(corrected - float(result["corrected_risk"])) < 1e-12


def test_crc_can_return_complete_abstention_without_claiming_coverage() -> None:
    labels = np.zeros((10, 1), dtype=np.float32)
    prediction = np.ones((10, 1), dtype=np.float32)
    scenes = np.arange(10)
    result = conformal_risk_threshold(labels, prediction, scenes, alpha=0.10)
    assert float(result["threshold"]) > 1.0
    assert not np.any(prediction >= float(result["threshold"]))
