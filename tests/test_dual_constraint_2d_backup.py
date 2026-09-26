"""Return viability must use velocity and energy from the actual state."""

import unittest

import numpy as np

from nav3d.simulation import FlightState

from dual_constraint_2d.backup import return_rollout
from dual_constraint_2d.routes import PlanarRouter
from dual_constraint_2d.synthetic import make_synthetic_map


class BackupTest(unittest.TestCase):
    def test_executable_return_and_insufficient_energy(self) -> None:
        case = make_synthetic_map(0)
        router = PlanarRouter(case)
        start = FlightState(case.xyz((56.91138312784211, 71.06819714784767)), np.zeros(3))
        sufficient = return_rollout(case, start, 100.0, router=router)
        self.assertTrue(sufficient.feasible, sufficient.reason)
        self.assertEqual(sufficient.reason, "arrived")
        self.assertGreater(sufficient.energy_needed, 0)
        self.assertEqual(len(sufficient.positions_xy), sufficient.steps + 1)
        self.assertEqual(len(sufficient.cumulative_energy), sufficient.steps + 1)
        insufficient = return_rollout(case, start, 1.0, router=router)
        self.assertFalse(insufficient.feasible)
        self.assertEqual(insufficient.reason, "insufficient_energy")


if __name__ == "__main__":
    unittest.main()
