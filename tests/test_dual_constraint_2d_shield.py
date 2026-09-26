"""Check that accepted blocks preserve an executable return continuation."""

import unittest

import numpy as np

from nav3d.simulation import FlightState

from dual_constraint_2d.backup import return_rollout
from dual_constraint_2d.routes import PlanarRouter
from dual_constraint_2d.shield import certify_policy_block, execute_certified_block
from dual_constraint_2d.synthetic import make_synthetic_map


class ShieldBlockTest(unittest.TestCase):
    def test_accepted_block_matches_execution_and_keeps_return_budget(self) -> None:
        case = make_synthetic_map(0)
        router = PlanarRouter(case)
        state = FlightState(case.xyz(case.station_xy), np.zeros(3))
        goal = np.array((56.91138312784211, 71.06819714784767))
        direction = goal - np.asarray(case.station_xy)
        desired = direction / np.linalg.norm(direction) * 5.0
        block = certify_policy_block(case, state, 85.39121788182787, desired, router=router)
        self.assertTrue(block.accepted_policy, block.reason)
        self.assertEqual(len(block.actions), case.config.policy_hold_steps)
        next_state, remaining = execute_certified_block(case, state, 85.39121788182787, block)
        np.testing.assert_allclose(next_state.position[:2], block.positions_xy[-1])
        self.assertGreaterEqual(remaining + 1e-9, block.endpoint_backup.energy_needed)
        self.assertEqual(next_state.collision_count, 0)

    def test_low_energy_at_station_does_not_authorize_departure(self) -> None:
        case = make_synthetic_map(0)
        router = PlanarRouter(case)
        state = FlightState(case.xyz(case.station_xy), np.zeros(3))
        block = certify_policy_block(case, state, 0.001, [5.0, 0.0], router=router)
        self.assertFalse(block.accepted_policy)
        self.assertEqual(block.reason, "energy_filter")
        self.assertEqual(block.actions, ())

    def test_unsafe_departure_executes_existing_return_suffix(self) -> None:
        case = make_synthetic_map(0)
        router = PlanarRouter(case)
        point = np.array((56.91138312784211, 71.06819714784767))
        state = FlightState(case.xyz(point), np.zeros(3))
        backup = return_rollout(case, state, 100.0, router=router)
        self.assertTrue(backup.feasible)
        near_limit = backup.energy_needed + 0.05
        direction = point - np.asarray(case.station_xy)
        desired = direction / np.linalg.norm(direction) * 5.0
        block = certify_policy_block(
            case, state, near_limit, desired, router=router, current_backup=backup
        )
        self.assertFalse(block.accepted_policy)
        self.assertEqual(block.reason, "energy_filter")
        self.assertEqual(len(block.actions), case.config.policy_hold_steps)
        next_state, remaining = execute_certified_block(case, state, near_limit, block)
        self.assertEqual(next_state.collision_count, 0)
        self.assertGreaterEqual(remaining + 1e-9, block.endpoint_backup.energy_needed)


if __name__ == "__main__":
    unittest.main()
