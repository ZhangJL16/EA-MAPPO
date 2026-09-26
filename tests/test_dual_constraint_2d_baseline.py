"""The nonlearning baseline must see only the current goal and recharge explicitly."""

import unittest

from dual_constraint_2d.baseline import FullMapRouteBaseline
from dual_constraint_2d.environment import DualConstraintEnv


class FullMapBaselineTest(unittest.TestCase):
    def test_full_mode_charges_after_return_and_departs_when_full(self) -> None:
        env = DualConstraintEnv(0, 85.39121788182787, 29.9)
        policy = FullMapRouteBaseline(env.case)
        initial = policy.act(env.observe())
        self.assertIsNone(initial.charge_target_fraction)
        self.assertGreater(sum(x * x for x in initial.desired_velocity_xy), 0)
        env.energy *= 0.4
        charging = policy.act(env.observe())
        self.assertEqual(charging.charge_target_fraction, 1.0)

    def test_partial_mode_uses_shared_reference_and_never_previews_next_goal(self) -> None:
        env = DualConstraintEnv(0, 85.39121788182787, 29.9)
        policy = FullMapRouteBaseline(env.case, charge_mode="reference_minimum")
        env.energy = 1.0
        action = policy.act(env.observe())
        self.assertIsNotNone(action.charge_target_fraction)
        self.assertGreater(action.charge_target_fraction, env.energy / env.charger.capacity)
        self.assertLessEqual(action.charge_target_fraction, 1.0)
        self.assertEqual(list(policy._reference_energy), [env.target.position_xy])
        env.mode = "return"
        self.assertIsNone(policy.act(env.observe()))


if __name__ == "__main__":
    unittest.main()
