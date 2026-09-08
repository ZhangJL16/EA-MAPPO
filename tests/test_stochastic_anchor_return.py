from __future__ import annotations

import numpy as np

from experiments.directional_navigation.stochastic_anchor_return import (
    StochasticExecutionPlant,
)
from scripts.collect_stochastic_defective_return import disturbance_seed


def build_plant(*, hocbf: bool = False) -> StochasticExecutionPlant:
    return StochasticExecutionPlant(
        execution_error_sigma=0.04,
        execution_error_rho=0.95,
        execution_error_clip_sigma=3.0,
        lidar_enabled=True,
        lidar_horizontal_sectors=128,
        lidar_vertical_sectors=8,
        num_obstacles=0,
        cbf_enabled=hocbf,
        projection_geometry_enabled=False,
        phase1_episode_max_policy_steps=16,
        max_steps_per_task=16,
    )


def test_disturbance_seed_is_schedule_independent() -> None:
    assert disturbance_seed(730001, 7, 3) == disturbance_seed(730001, 7, 3)
    assert len({disturbance_seed(730001, anchor, rep) for anchor in range(4) for rep in range(8)}) == 32


def test_raw_and_hocbf_receive_same_postfilter_noise_prefix_without_constraints() -> None:
    raw = build_plant(hocbf=False)
    hocbf = build_plant(hocbf=True)
    try:
        options = {
            "start_position": np.asarray([500.0, 500.0, 100.0], dtype=np.float32),
            "start_velocity": np.zeros(3, dtype=np.float32),
            "task_point": np.asarray([1000.0, 1000.0, 100.0], dtype=np.float32),
            "static_obstacles": [],
        }
        raw.reset(seed=1, options=options)
        hocbf.reset(seed=1, options=options)
        raw.configure_execution_disturbance(99)
        hocbf.configure_execution_disturbance(99)
        action = np.asarray([0.1, -0.2, 0.05], dtype=np.float32)
        for step in range(1, 6):
            raw.current_step = step
            hocbf.current_step = step
            raw_action, _ = raw._safety_filtered_action(action)
            hocbf_action, _ = hocbf._safety_filtered_action(action)
            np.testing.assert_array_equal(
                raw.last_requested_execution_error,
                hocbf.last_requested_execution_error,
            )
            np.testing.assert_array_equal(raw_action, hocbf_action)
            assert np.max(np.abs(raw.last_requested_execution_error)) <= 0.12
    finally:
        raw.close()
        hocbf.close()


def test_disturbance_is_constant_within_policy_step() -> None:
    plant = build_plant()
    try:
        plant.reset(seed=2)
        plant.configure_execution_disturbance(101)
        plant.current_step = 1
        plant._safety_filtered_action(np.zeros(3, dtype=np.float32))
        first = plant.last_requested_execution_error.copy()
        plant._safety_filtered_action(np.zeros(3, dtype=np.float32))
        np.testing.assert_array_equal(first, plant.last_requested_execution_error)
        plant.current_step = 2
        plant._safety_filtered_action(np.zeros(3, dtype=np.float32))
        assert not np.array_equal(first, plant.last_requested_execution_error)
    finally:
        plant.close()
