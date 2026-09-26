"""Focused dock/flight integration checks for the continuing task state."""

import unittest

from dual_constraint_2d.environment import DualConstraintEnv, EnvironmentAction


class EnvironmentTest(unittest.TestCase):
    def test_dock_idle_is_free_and_partial_charge_advances_clock(self) -> None:
        env = DualConstraintEnv(0, 85.39121788182787, 29.9, horizon_s=60.0)
        initial_target = env.target.position_xy
        env.energy = env.charger.capacity * 0.2
        observation, reward, done, info = env.step(EnvironmentAction())
        self.assertEqual(info["event"], "dock_idle")
        self.assertAlmostEqual(observation["energy"], env.charger.capacity * 0.2)
        self.assertAlmostEqual(observation["time_s"], 0.5)
        self.assertEqual(observation["target_xy"], initial_target)
        self.assertEqual(reward, 0.0)
        self.assertFalse(done)
        observation, _, _, info = env.step(EnvironmentAction(charge_target_fraction=0.8))
        self.assertEqual(info["event"], "charged")
        self.assertAlmostEqual(observation["energy"], env.charger.capacity * 0.8)
        self.assertAlmostEqual(observation["time_s"], 0.5 + 29.9 * 0.6)
        self.assertEqual(observation["mode"], "docked")

    def test_low_battery_rejected_departure_does_not_teleport_or_autocharge(self) -> None:
        env = DualConstraintEnv(0, 85.39121788182787, 29.9, horizon_s=60.0)
        env.energy = 0.001
        before = env.observe()
        after, reward, done, info = env.step(EnvironmentAction((5.0, 0.0)))
        self.assertEqual(info["event"], "departure_rejected")
        self.assertEqual(after["position_xy"], before["position_xy"])
        self.assertEqual(after["energy"], before["energy"])
        self.assertEqual(after["mode"], "docked")
        self.assertEqual(info["energy_filter_events"], 1)
        self.assertEqual(reward, 0.0)
        self.assertFalse(done)


if __name__ == "__main__":
    unittest.main()
