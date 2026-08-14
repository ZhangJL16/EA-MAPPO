from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC

from envs.navigation import D_NEAR, RelativeGoalNavigationEnv
from experiments.navigation_scale import evaluate_relative_goal_policy


ROOT = Path(__file__).resolve().parents[2]
OLD_CHECKPOINT = ROOT / "artifacts/phase1_sb3_sac_1m_gpu/seed0/checkpoint_step_1000000.zip"
OLD_CHECKPOINT_SHA256 = "56f50be556888a2b567ea76b3c9726319187802e1327a78db48082de195c6fce"


def _explicit_observation(
    environment: RelativeGoalNavigationEnv,
    start: np.ndarray,
    delta: np.ndarray,
    velocity: np.ndarray | None = None,
) -> np.ndarray:
    observation, _ = environment.reset(
        seed=17,
        options={
            "start_position": start,
            "goal_position": start + delta,
            "start_velocity": np.zeros(3) if velocity is None else velocity,
        },
    )
    return observation


def test_relative_goal_observation_is_exactly_seven_dimensional_and_lidar_free() -> None:
    environment = RelativeGoalNavigationEnv()
    observation, info = environment.reset(seed=1)
    assert observation.shape == environment.observation_space.shape == (7,)
    assert environment.observation_fields == (
        "normalized_velocity",
        "relative_goal_direction",
        "relative_goal_distance",
    )
    assert info["lidar_observed_by_policy"] is False
    excluded = set(environment.observation_definition["excluded"])
    assert {
        "absolute_position",
        "absolute_goal_position",
        "station_position",
        "state_of_charge",
        "lidar_distances",
        "lidar_valid",
    } <= excluded


def test_velocity_and_direction_features_are_world_size_independent() -> None:
    velocity = np.array([0.03, -0.06, 0.012])
    source = RelativeGoalNavigationEnv(world_size_xy=4.0)
    target = RelativeGoalNavigationEnv(world_size_xy=16.0)
    source_observation = _explicit_observation(
        source,
        np.array([0.3, 1.0, 1.0]),
        np.array([3.4, 0.0, 0.0]),
        velocity,
    )
    target_observation = _explicit_observation(
        target,
        np.array([1.0, 1.0, 1.0]),
        np.array([12.0, 0.0, 0.0]),
        velocity,
    )
    np.testing.assert_allclose(source_observation, target_observation)
    np.testing.assert_allclose(source_observation[:3], velocity / [0.30, 0.30, 0.12])
    np.testing.assert_array_equal(source_observation[3:6], [1.0, 0.0, 0.0])
    assert source_observation[6] == 1.0


def test_goal_distance_feature_uses_fixed_two_meter_scale() -> None:
    environment = RelativeGoalNavigationEnv(world_size_xy=16.0)
    start = np.array([1.0, 1.0, 1.0])
    for distance, expected in ((1.0, 0.5), (2.0, 1.0), (4.0, 1.0), (8.0, 1.0), (12.0, 1.0)):
        observation = _explicit_observation(environment, start, np.array([distance, 0.0, 0.0]))
        assert observation[6] == expected
    assert D_NEAR == 2.0


def test_fixed_physics_do_not_scale_with_world_size() -> None:
    source = RelativeGoalNavigationEnv(world_size_xy=4.0)
    target = RelativeGoalNavigationEnv(world_size_xy=16.0)
    np.testing.assert_array_equal(source.config.v_max, target.config.v_max)
    np.testing.assert_array_equal(source.config.a_max, target.config.a_max)
    assert source.config.dt == target.config.dt == 0.2
    assert source.config.body_radius == target.config.body_radius == 0.05
    assert source.goal_radius == target.goal_radius == 0.20


def test_near_far_intervals_share_the_sampler_effective_bounds() -> None:
    environment = RelativeGoalNavigationEnv(world_size_xy=4.0)
    intervals = environment.effective_distance_intervals()
    assert set(intervals) == {"NEAR", "FAR"}
    assert intervals["NEAR"] == (0.30, 2.0)
    assert intervals["FAR"][0] == 2.0
    assert all(upper > lower for lower, upper in intervals.values())
    for label, interval in intervals.items():
        for seed in range(100):
            environment.np_random = np.random.default_rng(seed)
            start, goal, sampled_label, distance = environment._sample_start_goal(interval)
            assert sampled_label == label
            assert interval[0] <= distance <= interval[1]
            assert environment._is_legal_position(start)
            assert environment._is_legal_position(goal)


def test_sampler_survives_ten_thousand_resets_without_invalid_geometry() -> None:
    environment = RelativeGoalNavigationEnv(world_size_xy=4.0, near_probability=0.5)
    counts = {"NEAR": 0, "FAR": 0}
    for seed in range(10_000):
        observation, info = environment.reset(seed=seed)
        counts[info["distance_group"]] += 1
        assert observation.shape == (7,)
        assert environment._is_legal_position(environment.state.position)
        assert environment._is_legal_position(environment.goal)
        lower, upper = environment.effective_distance_intervals()[info["distance_group"]]
        assert lower <= info["initial_goal_distance"] <= upper
    assert 0.47 <= counts["NEAR"] / 10_000 <= 0.53
    assert 0.47 <= counts["FAR"] / 10_000 <= 0.53


def test_old_frozen_checkpoint_hash_remains_unchanged() -> None:
    assert hashlib.sha256(OLD_CHECKPOINT.read_bytes()).hexdigest() == OLD_CHECKPOINT_SHA256


def test_sb3_can_update_save_and_reload_relative_goal_policy(tmp_path: Path) -> None:
    environment = RelativeGoalNavigationEnv(world_size_xy=4.0, max_episode_steps=20)
    model = SAC(
        "MlpPolicy",
        environment,
        learning_starts=1,
        buffer_size=100,
        batch_size=2,
        train_freq=1,
        gradient_steps=1,
        device="cpu",
        seed=9,
        verbose=0,
    )
    model.learn(total_timesteps=24)
    path = tmp_path / "relative_goal_sac"
    model.save(path)
    loaded = SAC.load(path.with_suffix(".zip"), device="cpu")
    observation, _ = environment.reset(seed=9)
    action, _ = loaded.predict(observation, deterministic=True)
    next_observation, reward, terminated, truncated, _ = environment.step(action)
    assert action.shape == (3,)
    assert next_observation.shape == (7,)
    assert np.isfinite(reward)
    assert isinstance(terminated, bool) and isinstance(truncated, bool)


def test_zero_shot_evaluator_uses_one_frozen_policy_across_world_sizes() -> None:
    class DirectionPolicy:
        def predict(self, observation: np.ndarray, *, deterministic: bool):
            del deterministic
            return np.asarray(observation[3:6], dtype=np.float32), None

    result = evaluate_relative_goal_policy(
        DirectionPolicy(),
        world_sizes=(4.0, 8.0, 16.0),
        sorties_per_bin=1,
        max_steps=3,
        seed=700,
    )
    assert result["world_sizes"] == [4.0, 8.0, 16.0]
    assert result["policy_update_during_evaluation"] is False
    assert set(result["by_world_size"]) == {"4.0", "8.0", "16.0"}
    assert result["NEAR_COMPLETION_RATE"] is not None
    assert result["FAR_COMPLETION_RATE"] is not None
