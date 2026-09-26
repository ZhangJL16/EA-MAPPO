"""Learning and MPC decode the same current-target action set."""

import unittest

import numpy as np

from dual_constraint_2d.action_adapter import ACTION_COUNT, FEATURE_COUNT, ActionAdapter
from dual_constraint_2d.environment import DualConstraintEnv


class ActionAdapterTest(unittest.TestCase):
    def test_training_position_outside_planner_clearance_has_finite_fallback(self) -> None:
        env = DualConstraintEnv(0, 85.39121788182787, 29.9)
        adapter = ActionAdapter(env.case)
        blocked = env.case.world.obstacles[0].center_xy
        direction, distance = adapter.route_direction(blocked, env.target.position_xy)
        self.assertTrue(np.isfinite(direction).all())
        self.assertGreater(distance, 0.0)

    def test_current_goal_features_and_actions(self) -> None:
        env = DualConstraintEnv(0, 85.39121788182787, 29.9)
        adapter = ActionAdapter(env.case)
        observation = env.observe()
        self.assertEqual(adapter.features(observation).shape, (FEATURE_COUNT,))
        self.assertEqual(ACTION_COUNT, 13)
        self.assertIsNone(adapter.decode(4, observation).charge_target_fraction)
        env.energy *= 0.2
        self.assertEqual(adapter.decode(12, env.observe()).charge_target_fraction, 1.0)
        env.mode = "flight"
        desired = adapter.decode(9, env.observe()).desired_velocity_xy
        toward_station = np.asarray(env.case.station_xy) - env.state.position[:2]
        self.assertAlmostEqual(np.linalg.norm(desired), 0.0)
        self.assertAlmostEqual(np.linalg.norm(toward_station), 0.0)


if __name__ == "__main__":
    unittest.main()
