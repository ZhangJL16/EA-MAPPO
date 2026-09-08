from __future__ import annotations

import hashlib

import numpy as np

from experiments.forked_action_safety.core import (
    analytic_clearance,
    candidate_actions,
    sanitize_snapshot_position,
    sanitize_snapshot_velocity,
    select_anchor_indices,
    summarize_paired_gate,
)
from experiments.uav_energy_parallel import ParallelUAVEnvPool, WorkerForkReset
from scripts.run_lidar_forked_action_data_gate import selected_scene_indices


def test_scene_selection_can_reserve_disjoint_bucket_halves() -> None:
    development = selected_scene_indices(150, 75, offset_per_bucket=0)
    confirmation = selected_scene_indices(150, 75, offset_per_bucket=15)
    assert len(development) == len(confirmation) == 75
    assert not set(development) & set(confirmation)
    assert set(development) | set(confirmation) == set(range(150))


def test_candidate_set_has_registered_ten_actions() -> None:
    actions = candidate_actions(
        np.asarray([0.1, -0.2, 0.3], dtype=np.float32),
        position=np.asarray([50.0, 50.0, 20.0]),
        velocity=np.asarray([4.0, -2.0, 1.0]),
        obstacles=[{"position": [80.0, 50.0], "radius": 10.0}],
        perturbation=0.45,
        hold_steps=8,
        policy_dt=0.2,
        horizontal_a_max=5.0,
        vertical_a_max=3.0,
    )
    assert len(actions) == 10
    assert len({item.name for item in actions}) == 10
    assert all(item.action.shape == (3,) for item in actions)
    np.testing.assert_allclose(actions[-2].action, [1.0, 0.0, 0.0])
    np.testing.assert_allclose(actions[-1].action, [-1.0, 0.0, 0.0])


def test_clearance_and_anchor_selection_prioritize_near_obstacle() -> None:
    positions = np.asarray([[10.0 + i * 5.0, 50.0, 20.0] for i in range(12)])
    legs = np.asarray([0] * 6 + [1] * 6, dtype=np.int8)
    obstacles = [{"position": [42.0, 50.0], "radius": 5.0}]
    clearance = analytic_clearance(positions, obstacles, safe_radius=1.0)
    selected = select_anchor_indices(legs, clearance, anchors_per_scene=4)
    assert selected.size == 4
    assert set(legs[selected]) == {0, 1}
    # Endpoints 5 and 6 are deliberately excluded; the nearest interior state
    # from each leg must still be selected.
    assert 4 in selected
    assert 7 in selected


def test_paired_gate_requires_class_and_energy_variation() -> None:
    rows = []
    for anchor in range(5):
        for branch in range(10):
            rows.append(
                {
                    "anchor_id": f"a{anchor}",
                    "first_executed_action": [branch * 0.1, 0.0, 0.0],
                    "safe_recharge": not (branch == 0 and anchor < 2),
                    "total_realized_energy": 10.0 + branch,
                    "collision": branch == 0 and anchor < 2,
                    "boundary_contact": False,
                    "anchor_observation_sha256": "obs",
                    "obstacle_layout_sha256": "scene",
                }
            )
    report = summarize_paired_gate(rows)
    assert report["checks"]["all_anchors_have_registered_branches"]
    assert report["checks"]["executed_action_diversity_at_80_percent"]
    assert report["checks"]["unsafe_prevalence_between_2_and_40_percent"]
    assert report["checks"]["paired_variation_at_20_percent"]


def test_worker_fork_reset_restores_anchor_and_raw_step() -> None:
    kwargs = {
        "length": 200.0,
        "width": 200.0,
        "height": 100.0,
        "minimum_task_distance": 5.0,
        "xy_sampling_margin": 10.0,
        "task_z_min": 10.0,
        "task_z_max": 90.0,
        "max_steps_per_task": 20,
        "phase1_episode_max_policy_steps": 20,
        "num_obstacles": 1,
        "obstacle_sampling_margin": 2.0,
        "obstacle_radius_min": 10.0,
        "obstacle_radius_max": 10.0,
        "lidar_enabled": True,
        "lidar_horizontal_sectors": 8,
        "lidar_vertical_sectors": 2,
        "cbf_enabled": True,
    }
    start = np.asarray([20.0, 20.0, 20.0], dtype=np.float32)
    anchor = np.asarray([75.0, 100.0, 30.0], dtype=np.float32)
    goal = np.asarray([160.0, 160.0, 40.0], dtype=np.float32)
    obstacles = [{"position": [100.0, 100.0], "radius": 10.0}]
    with ParallelUAVEnvPool(kwargs, num_workers=1) as pool:
        reset = pool.fork_reset_many(
            [
                WorkerForkReset(
                    worker_id=0,
                    seed=123,
                    validation_start_position=start,
                    anchor_position=anchor,
                    anchor_velocity=np.zeros(3, dtype=np.float32),
                    goal_position=goal,
                    static_obstacles=obstacles,
                    elapsed_policy_steps=4,
                )
            ]
        )[0]
        stepped = pool.step_many(
            [0], np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32), disable_cbf=True
        )[0]
    assert reset.current_step == 4
    np.testing.assert_allclose(reset.position, anchor)
    assert reset.observation.shape == (7 + 32,)
    assert hashlib.sha256(reset.observation.tobytes()).hexdigest()
    np.testing.assert_allclose(stepped.info["executed_action"], [1.0, 0.0, 0.0])


def test_worker_fork_reset_does_not_recanonicalize_registered_velocity() -> None:
    kwargs = {
        "length": 200.0,
        "width": 200.0,
        "height": 100.0,
        "minimum_task_distance": 5.0,
        "xy_sampling_margin": 10.0,
        "task_z_min": 10.0,
        "task_z_max": 90.0,
        "max_steps_per_task": 20,
        "phase1_episode_max_policy_steps": 20,
        "num_obstacles": 0,
        "lidar_enabled": False,
        "cbf_enabled": False,
    }
    # This registered legacy snapshot is 2.03e-7 m/s above 20 in float64,
    # while the environment's float32 physical validator accepts it.  Four
    # branches already exist for the corresponding formal anchor, so a resume
    # must validate and restore its exact bits rather than recanonicalize it.
    velocity = np.asarray(
        [9.640527725219727, -17.523134231567383, 0.835464596748352],
        dtype=np.float32,
    )
    anchor = np.asarray([75.0, 100.0, 30.0], dtype=np.float32)
    with ParallelUAVEnvPool(kwargs, num_workers=1) as pool:
        reset = pool.fork_reset_many(
            [
                WorkerForkReset(
                    worker_id=0,
                    seed=123,
                    validation_start_position=np.asarray(
                        [20.0, 20.0, 20.0], dtype=np.float32
                    ),
                    anchor_position=anchor,
                    anchor_velocity=velocity,
                    goal_position=np.asarray([160.0, 160.0, 40.0], dtype=np.float32),
                    static_obstacles=[],
                    elapsed_policy_steps=4,
                )
            ]
        )[0]
    np.testing.assert_array_equal(reset.velocity, velocity)


def test_snapshot_velocity_projects_only_float32_scale_excess() -> None:
    velocity = np.asarray([19.09295845, -5.95474625, 1.0], dtype=np.float32)
    sanitized = sanitize_snapshot_velocity(
        velocity, horizontal_limit=20.0, vertical_limit=5.0
    )
    assert float(np.linalg.norm(sanitized[:2].astype(np.float64))) <= 20.0 + 1e-6
    with np.testing.assert_raises(ValueError):
        sanitize_snapshot_velocity(
            np.asarray([20.01, 0.0, 0.0], dtype=np.float32),
            horizontal_limit=20.0,
            vertical_limit=5.0,
        )


def test_snapshot_velocity_projection_is_bitwise_idempotent() -> None:
    # Exact vector that stopped the 150-scene RCPS confirmation: the first
    # float32 projection remained 6.22e-7 m/s outside the horizontal sphere.
    failing = np.asarray(
        [2.344048023223877, -19.862163543701172, -1.6375644207000732],
        dtype=np.float32,
    )
    once = sanitize_snapshot_velocity(
        failing, horizontal_limit=20.0, vertical_limit=5.0
    )
    twice = sanitize_snapshot_velocity(
        once, horizontal_limit=20.0, vertical_limit=5.0
    )
    np.testing.assert_array_equal(once, twice)
    assert float(np.linalg.norm(once[:2].astype(np.float64))) <= 20.0

    # Cover directions rather than only the observed vector: casting a radial
    # projection back to float32 must never create a second-stage change.
    angles = np.linspace(-np.pi, np.pi, 721, endpoint=False)
    radius = np.nextafter(np.float32(20.0), np.float32(np.inf))
    for angle in angles:
        velocity = np.asarray(
            [radius * np.cos(angle), radius * np.sin(angle), 0.0],
            dtype=np.float32,
        )
        projected = sanitize_snapshot_velocity(
            velocity, horizontal_limit=20.0, vertical_limit=5.0
        )
        repeated = sanitize_snapshot_velocity(
            projected, horizontal_limit=20.0, vertical_limit=5.0
        )
        np.testing.assert_array_equal(projected, repeated)
        assert float(np.linalg.norm(projected[:2].astype(np.float64))) <= 20.0


def test_snapshot_position_projects_only_float32_scale_excess() -> None:
    extent = np.asarray([4000.0, 4000.0, 400.0], dtype=np.float32)
    position = np.asarray([1000.0, 2000.0, 0.49994898], dtype=np.float32)
    sanitized = sanitize_snapshot_position(
        position, world_extent=extent, safe_radius=0.5
    )
    np.testing.assert_allclose(sanitized, [1000.0, 2000.0, 0.5])
    with np.testing.assert_raises(ValueError):
        sanitize_snapshot_position(
            np.asarray([1000.0, 2000.0, 0.49], dtype=np.float32),
            world_extent=extent,
            safe_radius=0.5,
        )
