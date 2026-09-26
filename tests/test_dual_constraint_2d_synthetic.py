"""Focused mechanics checks; not a navigation or safety-qualification test."""

import unittest

import numpy as np

from dual_constraint_2d.synthetic import (
    SYNTHETIC_V0, SyntheticChargeState, SyntheticCharger,
    SyntheticEnergy, make_synthetic_map,
)


class SyntheticMechanicsTest(unittest.TestCase):
    def test_map_is_deterministic_and_planar_obstacles_block_segments(self) -> None:
        first = make_synthetic_map(0)
        again = make_synthetic_map(0)
        self.assertEqual(first.station_xy, again.station_xy)
        self.assertEqual(len(first.world.obstacles), 8)
        for a, b in zip(first.world.obstacles, again.world.obstacles):
            np.testing.assert_array_equal(a.center_xy, b.center_xy)
            self.assertEqual(a.radius, b.radius)
            self.assertFalse(first.segment_clear(a.center_xy - [a.radius + 2, 0], a.center_xy + [a.radius + 2, 0]))
        self.assertTrue(first.is_clear(first.station_xy))
        self.assertEqual(SYNTHETIC_V0.policy_hold_steps, 10)

    def test_energy_varies_with_motion_and_docked_charge_jumps_time(self) -> None:
        model = SyntheticEnergy()
        hover = model.flight_cost([0, 0], [0, 0], 0.05)
        cruise = model.flight_cost([7, 0], [0, 0], 0.05)
        turn = model.flight_cost([7, 0], [0, 4], 0.05)
        self.assertLess(hover, cruise)
        self.assertLess(cruise, turn)
        charger = SyntheticCharger(100.0, 60.0)
        state = charger.charge_to_fraction(SyntheticChargeState(20.0, 10.0), 0.8)
        self.assertAlmostEqual(state.energy, 80.0)
        self.assertAlmostEqual(state.elapsed_s, 46.0)


if __name__ == "__main__":
    unittest.main()
