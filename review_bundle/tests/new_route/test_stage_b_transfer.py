from __future__ import annotations

from pathlib import Path

import numpy as np

from agents.goal_conditioned_sac import FrozenGoalConditionedSAC
from envs.navigation import NavigationEnv
from experiments.energy_transfer.stage_b_navigation_transfer import (
    DISTANCE_BINS,
    StageBNavigationConfig,
    distance_bin_label,
    navigation_schedule,
)


ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = ROOT / "artifacts/phase1_sb3_sac_1m_gpu/seed0/checkpoint_step_1000000.zip"


def test_target_open_16x16_reset_and_normalization_contract() -> None:
    environment = NavigationEnv("target_open_16x16.json", max_episode_steps=2)
    start = np.array([4.0, 8.0, 1.0])
    goal = np.array([12.0, 4.0, 1.0])
    station = np.array([8.0, 12.0, 1.0])
    observation, _ = environment.reset(
        seed=301,
        options={"start_position": start, "goal_position": goal, "station_position": station},
    )
    assert observation.shape == (77,)
    np.testing.assert_allclose(observation[:3], start / [16.0, 16.0, 2.0])
    np.testing.assert_allclose(observation[6:9], goal / [16.0, 16.0, 2.0])
    np.testing.assert_allclose(observation[9:12], station / [16.0, 16.0, 2.0])
    assert environment.observation_space.contains(observation)


def test_frozen_sac_predicts_on_target_observation() -> None:
    policy = FrozenGoalConditionedSAC(CHECKPOINT, device="cpu")
    environment = NavigationEnv("target_open_16x16.json", max_episode_steps=2)
    observation, _ = environment.reset(seed=302)
    action = policy.action(observation)
    assert action.shape == (3,)
    assert np.all(np.isfinite(action))
    assert np.all(action >= -1.0) and np.all(action <= 1.0)


def test_navigation_schedule_is_reproducible_and_stratified(tmp_path: Path) -> None:
    config = StageBNavigationConfig(
        seed=7,
        checkpoint=str(CHECKPOINT),
        output_dir=str(tmp_path / "unused"),
        sorties=140,
    )
    world = np.array([16.0, 16.0, 2.0])
    first = navigation_schedule(config, world)
    second = navigation_schedule(config, world)
    assert len(first) == len(second) == 140
    assert [item["distance_bin"] for item in first] == [item["distance_bin"] for item in second]
    for left, right in zip(first, second, strict=True):
        np.testing.assert_array_equal(left["start"], right["start"])
        np.testing.assert_array_equal(left["goal"], right["goal"])
        distance = float(np.linalg.norm(left["goal"] - left["start"]))
        assert distance_bin_label(distance) == left["distance_bin"]
    counts = {label: sum(item["distance_bin"] == label for item in first) for label, _, _ in DISTANCE_BINS}
    assert set(counts.values()) == {20}

