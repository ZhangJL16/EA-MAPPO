"""The accepted goal stream must be policy-independent and reproducible."""

import unittest

from dual_constraint_2d.routes import PlanarRouter
from dual_constraint_2d.synthetic import make_synthetic_map
from dual_constraint_2d.targets import select_next_target


class GoalStreamTest(unittest.TestCase):
    def test_same_candidate_cursor_produces_same_full_battery_goal(self) -> None:
        case = make_synthetic_map(0)
        router = PlanarRouter(case)
        first = select_next_target(case, router, 1_000.0, max_candidates=2)
        again = select_next_target(case, router, 1_000.0, max_candidates=2)
        self.assertEqual(first, again)
        self.assertEqual(first.accepted_candidate_id, 0)
        self.assertTrue(case.is_clear(first.position_xy))
        self.assertLess(first.reference_round_trip_energy, 1_000.0)


if __name__ == "__main__":
    unittest.main()
