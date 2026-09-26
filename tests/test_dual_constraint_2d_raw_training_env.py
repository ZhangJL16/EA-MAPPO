"""Training failures remain visible while contact repair continues the task."""

import unittest

import numpy as np

from nav3d.simulation import FlightState

from dual_constraint_2d.environment import EnvironmentAction
from dual_constraint_2d.raw_training_env import RawTrainingEnv


class RawTrainingTest(unittest.TestCase):
    def test_contact_keeps_same_target_and_unified_cost(self) -> None:
        env = RawTrainingEnv(0, 85.39121788182787, 29.9, horizon_s=10)
        env.state = FlightState(env.case.xyz((0.5, 100.0)), np.array((-5.0, 0.0, 0.0)))
        env.mode = "flight"
        target = env.target.position_xy
        _, _, done, info = env.step(EnvironmentAction((-7.0, 0.0)))
        self.assertFalse(done)
        self.assertEqual(env.target.position_xy, target)
        self.assertGreater(info["new_collision_steps"], 0)
        self.assertEqual(env.state.collision_count, env.state.safety_cost)
        self.assertLess(info["raw_contact_penalty"], 0)

    def test_depletion_terminates_without_reset_or_rescue(self) -> None:
        env = RawTrainingEnv(0, 85.39121788182787, 29.9, horizon_s=10)
        env.energy = 0.001
        _, _, done, info = env.step(EnvironmentAction((5.0, 0.0)))
        self.assertTrue(done)
        self.assertEqual(info["failure_reason"], "depletion")
        self.assertEqual(env.energy, 0.0)
        self.assertEqual(env.charge_events, 0)


if __name__ == "__main__":
    unittest.main()
