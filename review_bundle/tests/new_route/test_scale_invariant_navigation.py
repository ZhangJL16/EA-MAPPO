from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC

from envs.navigation import ScaleInvariantNavigationEnv
from experiments.navigation_scale import evaluate_policy


ROOT = Path(__file__).resolve().parents[2]
OLD_CHECKPOINT = ROOT / "artifacts/phase1_sb3_sac_1m_gpu/seed0/checkpoint_step_1000000.zip"
OLD_CHECKPOINT_SHA256 = "56f50be556888a2b567ea76b3c9726319187802e1327a78db48082de195c6fce"


def _reset_geometry(
    environment: ScaleInvariantNavigationEnv,
    *,
    width: float,
    start: np.ndarray,
    goal_delta: np.ndarray,
    velocity: np.ndarray,
) -> np.ndarray:
    observation, _ = environment.reset(
        seed=91,
        options={
            "world_size_xy": width,
            "start_position": start,
            "goal_position": start + goal_delta,
            "start_velocity": velocity,
        },
    )
    return observation


def test_observation_dimension_is_constant_across_map_scales() -> None:
    environment = ScaleInvariantNavigationEnv()
    for width in (4.0, 8.0, 12.0, 16.0):
        observation, info = environment.reset(seed=int(width), options={"world_size_xy": width})
        assert observation.shape == (71,)
        assert environment.observation_space.shape == (71,)
        assert info["world_size"].tolist() == [width, width, 2.0]
        assert environment.observation_space.contains(observation)


def test_observation_contract_excludes_absolute_energy_and_station_fields() -> None:
    environment = ScaleInvariantNavigationEnv()
    assert environment.observation_fields == (
        "normalized_velocity",
        "relative_goal_direction",
        "relative_goal_distance",
        "lidar_distances",
        "lidar_valid",
    )
    excluded = set(environment.observation_definition["excluded"])
    assert {"absolute_position", "absolute_goal", "station_position", "state_of_charge"} <= excluded


def test_same_relative_goal_velocity_and_local_lidar_scene_match_across_maps() -> None:
    environment = ScaleInvariantNavigationEnv()
    delta = np.array([1.0, -0.5, 0.1])
    velocity = np.array([0.03, -0.06, 0.012])
    observation_12 = _reset_geometry(
        environment,
        width=12.0,
        start=np.array([6.0, 6.0, 1.0]),
        goal_delta=delta,
        velocity=velocity,
    )
    observation_16 = _reset_geometry(
        environment,
        width=16.0,
        start=np.array([8.0, 8.0, 1.0]),
        goal_delta=delta,
        velocity=velocity,
    )
    np.testing.assert_allclose(observation_12, observation_16, atol=1e-7)
    np.testing.assert_allclose(observation_12[:3], velocity / [0.30, 0.30, 0.12])
    np.testing.assert_allclose(observation_12[3:6], delta / np.linalg.norm(delta))
    assert observation_12[6] == np.float32(np.linalg.norm(delta) / 6.0)


def test_fixed_physics_and_lidar_semantics_across_map_scales() -> None:
    environment = ScaleInvariantNavigationEnv()
    records = []
    for width in (4.0, 8.0, 12.0, 16.0):
        environment.reset(seed=int(width) + 20, options={"world_size_xy": width})
        records.append(
            (
                environment.config.v_max.copy(),
                environment.config.a_max.copy(),
                environment.config.dt,
                environment.config.lidar_range,
                environment.config.body_radius,
                environment.goal_radius,
            )
        )
    for record in records[1:]:
        np.testing.assert_array_equal(record[0], records[0][0])
        np.testing.assert_array_equal(record[1], records[0][1])
        assert record[2:] == records[0][2:]
    assert records[0][2:] == (0.2, 6.0, 0.05, 0.20)


def test_world_and_distance_randomization_is_deterministic_and_bounded() -> None:
    first = ScaleInvariantNavigationEnv()
    second = ScaleInvariantNavigationEnv()
    seen_bins = set()
    widths = []
    for seed in range(40):
        first_observation, first_info = first.reset(seed=seed)
        second_observation, second_info = second.reset(seed=seed)
        np.testing.assert_array_equal(first_observation, second_observation)
        assert first_info["distance_bin"] == second_info["distance_bin"]
        assert first_info["initial_goal_distance"] == second_info["initial_goal_distance"]
        width = float(first_info["world_size"][0])
        assert 4.0 <= width <= 16.0
        widths.append(width)
        seen_bins.add(first_info["distance_bin"])
    assert min(widths) < 6.0 and max(widths) > 14.0
    assert {"0.5-2", "2-4", "4-8", "8-12"} <= seen_bins


def test_old_checkpoint_hash_is_unchanged() -> None:
    assert hashlib.sha256(OLD_CHECKPOINT.read_bytes()).hexdigest() == OLD_CHECKPOINT_SHA256


def test_sb3_sac_initializes_with_new_observation_contract() -> None:
    environment = ScaleInvariantNavigationEnv(max_episode_steps=10)
    model = SAC(
        "MlpPolicy",
        environment,
        learning_starts=1,
        buffer_size=100,
        batch_size=2,
        device="cpu",
        seed=3,
        verbose=0,
    )
    observation, _ = environment.reset(seed=3)
    action, _ = model.predict(observation, deterministic=True)
    next_observation, reward, terminated, truncated, _ = environment.step(action)
    assert next_observation.shape == (71,)
    assert np.isfinite(reward)
    assert isinstance(terminated, bool) and isinstance(truncated, bool)


def test_evaluation_protocol_separates_training_and_heldout_scales() -> None:
    class DirectionPolicy:
        def predict(self, observation: np.ndarray, *, deterministic: bool):
            del deterministic
            return np.asarray(observation[3:6], dtype=np.float32), None

    results = evaluate_policy(
        DirectionPolicy(),
        scales=(4.0, 6.0),
        sorties_per_distance_bin=1,
        max_steps=3,
        seed=500,
    )
    assert set(results["by_scale"]) == {"4.0", "6.0"}
    assert set(results["by_scale_split"]) == {"TRAIN_SCALE", "HELD_OUT_SCALE"}
    assert results["overall"]["sorties"] == len(results["records"])
