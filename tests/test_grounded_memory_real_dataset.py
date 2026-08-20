from __future__ import annotations

import numpy as np

from experiments.memory_safety.grounded_real_dataset import (
    ENERGY_STATE_DIM,
    FEATURE_DIM,
    GroundedDatasetConfig,
    _energy_state,
    _route_labels,
    _transition_feature,
    make_scenario,
)


def test_real_dataset_protocol_requires_one_hundred_thousand_transitions() -> None:
    try:
        GroundedDatasetConfig(minimum_controlled_transitions=99_999)
    except ValueError as error:
        assert "100k" in str(error)
    else:
        raise AssertionError("sub-100k real diagnostic protocol must be rejected")


def test_real_features_have_declared_dimensions_and_physical_energy_distance() -> None:
    position = np.array([100.0, 100.0, 50.0])
    velocity = np.array([2.0, -1.0, 0.5])
    goal = np.array([500.0, 300.0, 100.0])
    obstacle = np.array([250.0, 200.0, 50.0])
    feature = _transition_feature(
        position,
        velocity,
        goal,
        obstacle,
        np.zeros(3),
        np.zeros(3),
        np.zeros(3),
        0.0,
        1.0,
        True,
        False,
    )
    state = _energy_state(position, velocity, goal)
    assert feature.shape == (FEATURE_DIM,)
    assert state.shape == (ENERGY_STATE_DIM,)
    assert np.isclose(state[-1], np.linalg.norm(goal - position) / np.linalg.norm([4000.0, 4000.0, 400.0]))


def test_scenario_family_cycle_includes_required_real_conditions() -> None:
    families = {make_scenario(index, 7, 20.0).family for index in range(8)}
    assert families == {
        "static",
        "constant_velocity",
        "accelerating",
        "turning",
        "abrupt_change",
        "dropout",
        "dense_multi_obstacle",
        "boundary_interaction",
    }


def test_route_labels_are_future_trajectory_derived() -> None:
    positions = np.asarray([[float(index), 0.2 * index, 0.0] for index in range(12)])
    velocities = np.tile(np.array([1.0, 0.2, 0.0]), (12, 1))
    interventions = np.ones(12)
    obstacles = np.tile(np.array([6.0, 0.0, 0.0]), (12, 1))
    labels = _route_labels(
        positions,
        velocities,
        interventions,
        np.array([20.0, 0.0, 0.0]),
        obstacles,
        horizon=10,
    )
    assert labels["route_class"].shape == (12,)
    assert labels["future_safe_path_length"][0] > 0.0
    assert labels["future_intervention_count"][0] > 0.0
    assert labels["future_side"][0] != 0
