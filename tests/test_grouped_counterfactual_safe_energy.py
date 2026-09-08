from __future__ import annotations

from pathlib import Path

import numpy as np

from experiments.counterfactual_safe_energy.core import (
    CorrelatedActionIntervention,
    InterventionSpec,
    grouped_outer_fold,
    intervention_seed,
    relative_range,
)
from experiments.uav_energy_parallel import (
    ParallelUAVEnvPool,
    WorkerReset,
    WorkerRetarget,
)


def test_nominal_intervention_is_exact_identity() -> None:
    policy = CorrelatedActionIntervention(
        InterventionSpec("nominal", 1.0, 0.0, 0.0), seed=7
    )
    action = np.asarray([0.25, -0.5, 0.75], dtype=np.float32)
    np.testing.assert_array_equal(policy(action), action)


def test_correlated_intervention_is_keyed_and_leg_reset_only_resets_state() -> None:
    spec = InterventionSpec("test", 0.9, 0.2, 0.8)
    seed = intervention_seed(11, 3, 2)
    first = CorrelatedActionIntervention(spec, seed=seed)
    second = CorrelatedActionIntervention(spec, seed=seed)
    nominal = np.zeros(3, dtype=np.float32)
    sequence_a = [first(nominal) for _ in range(5)]
    sequence_b = [second(nominal) for _ in range(5)]
    np.testing.assert_allclose(sequence_a, sequence_b)
    first.reset_leg()
    assert np.all(first.residual == 0.0)
    assert not np.allclose(first(nominal), sequence_a[0])


def test_scene_grouping_never_splits_interventions() -> None:
    scenes = np.repeat(np.arange(15, dtype=np.int64), 5)
    folds = grouped_outer_fold(scenes)
    for scene in np.unique(scenes):
        assert np.unique(folds[scenes == scene]).size == 1
    assert set(np.unique(folds)) == {0, 1, 2}


def test_relative_range_requires_two_finite_values() -> None:
    assert np.isnan(relative_range(np.asarray([1.0, np.nan])))
    assert np.isclose(relative_range(np.asarray([1.0, 1.2, np.nan])), 0.2 / 1.1)


def test_parallel_retarget_command_runs_without_resampling_contract_break() -> None:
    kwargs = {
        "length": 200.0,
        "width": 200.0,
        "height": 100.0,
        "minimum_task_distance": 5.0,
        "xy_sampling_margin": 10.0,
        "task_z_min": 10.0,
        "task_z_max": 90.0,
        "max_steps_per_task": 10,
        "phase1_episode_max_policy_steps": 10,
        "num_obstacles": 0,
        "lidar_enabled": False,
        "cbf_enabled": False,
    }
    start = np.asarray([20.0, 20.0, 20.0], dtype=np.float32)
    goal = np.asarray([100.0, 100.0, 40.0], dtype=np.float32)
    second_goal = np.asarray([150.0, 150.0, 60.0], dtype=np.float32)
    with ParallelUAVEnvPool(kwargs, num_workers=1) as pool:
        reset = pool.reset_many(
            [
                WorkerReset(
                    0,
                    123,
                    {
                        "start_position": start,
                        "start_velocity": np.zeros(3, dtype=np.float32),
                        "task_point": goal,
                    },
                )
            ]
        )[0]
        retarget = pool.retarget_many(
            [
                WorkerRetarget(
                    0,
                    123,
                    start_position=start,
                    start_velocity=np.zeros(3, dtype=np.float32),
                    goal_position=second_goal,
                )
            ]
        )[0]
    assert reset.observation.shape == retarget.observation.shape == (7,)
    assert retarget.current_step == 0
    assert np.isfinite(retarget.observation).all()
