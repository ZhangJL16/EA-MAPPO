"""MPC searches the current target and returns a shared action ID."""

import unittest

from dual_constraint_2d.environment import DualConstraintEnv
from dual_constraint_2d.mpc import FullMapMPC


class MPCTest(unittest.TestCase):
    def test_depth_one_decision_is_executable(self) -> None:
        env = DualConstraintEnv(0, 85.39121788182787, 29.9, horizon_s=3.0)
        planner = FullMapMPC(env.case, depth=1)
        action = planner.act(env)
        self.assertEqual(action, 4)
        env.energy *= 0.2
        action = planner.act(env)
        self.assertIn(action, (10, 11, 12))
        command = planner.adapter.decode(action, env.observe())
        _, _, _, info = env.step(command)
        self.assertEqual(info["event"], "charged")
        self.assertGreater(planner.stats.decisions, 0)


if __name__ == "__main__":
    unittest.main()
