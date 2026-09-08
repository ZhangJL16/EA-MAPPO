from __future__ import annotations

import numpy as np

from experiments.rechargeability_safety.closed_loop import (
    geometry_action_features,
    temperature_scale,
)


def test_online_geometry_action_features_have_registered_semantics() -> None:
    nominal = np.asarray([0.2, -0.3, 0.4], dtype=np.float32)
    features = geometry_action_features(
        position=np.asarray([1000.0, 2000.0, 100.0]),
        velocity=np.asarray([10.0, -5.0, 2.5]),
        task_goal=np.asarray([2000.0, 2000.0, 100.0]),
        charger_goal=np.asarray([1000.0, 3000.0, 100.0]),
        active_goal=np.asarray([2000.0, 2000.0, 100.0]),
        obstacle_layout=[{"position": [1100.0, 2000.0], "radius": 50.0}],
        nearest_clearance=49.5,
        leg=0,
        elapsed_policy_steps=100,
        maximum_steps=4000,
        nominal_action=nominal,
        proposed_action=nominal,
        world_extent=np.asarray([4000.0, 4000.0, 400.0]),
        d_max=6000.0,
        horizontal_v_max=20.0,
        vertical_v_max=5.0,
        lidar_max_range=500.0,
    )
    assert features.shape == (26,)
    np.testing.assert_allclose(features[:6], [0.25, 0.5, 0.25, 0.5, -0.25, 0.5])
    np.testing.assert_array_equal(features[17:20], nominal)
    np.testing.assert_array_equal(features[20:23], np.zeros(3, dtype=np.float32))
    assert np.isclose(features[23], nominal[0])


def test_temperature_scaling_is_finite_and_order_preserving() -> None:
    values = np.asarray([0.01, 0.5, 0.99], dtype=np.float32)
    scaled = temperature_scale(values, 1.7)
    assert np.all(np.isfinite(scaled))
    assert np.all(np.diff(scaled) > 0.0)
    np.testing.assert_array_equal(temperature_scale(values, 1.0), values)
