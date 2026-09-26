"""The dock-only diagnostic keeps flight actions and records charge repairs."""

import unittest

from diagnostics2d.dock_supervision import supervised_action
from dual_constraint_2d.environment import DualConstraintEnv


class DockSupervisorTest(unittest.TestCase):
    def test_low_energy_dock_proposal_is_charged_and_flight_is_untouched(self) -> None:
        env = DualConstraintEnv(0, 85.39121788182787, 29.9)
        env.energy = 1.0
        action = supervised_action(env, 9)
        self.assertIn(action, (10, 11, 12))
        env.mode = "flight"
        self.assertEqual(supervised_action(env, 9), 9)

    def test_sufficient_energy_dock_proposal_is_untouched(self) -> None:
        env = DualConstraintEnv(0, 85.39121788182787, 29.9)
        self.assertEqual(supervised_action(env, 4), 4)

    def test_valid_partial_charge_proposal_is_untouched(self) -> None:
        env = DualConstraintEnv(0, 85.39121788182787, 29.9)
        env.energy = 1.0
        self.assertEqual(supervised_action(env, 10), 10)


if __name__ == "__main__":
    unittest.main()
