"""SOC-accounting checks for the event-driven 2D charging interface."""

import unittest

from delivery_1km.charging import CHARGING_SENSITIVITY
from delivery_1km.energy_profiles import BASE_ENERGY_ASSUMPTION
from dual_constraint_2d.battery import BatteryClock, BatteryModel


class BatteryClockTest(unittest.TestCase):
    def setUp(self) -> None:
        self.model = BatteryModel(BASE_ENERGY_ASSUMPTION, CHARGING_SENSITIVITY[1])

    def test_operating_cutoff_and_charge_clock(self) -> None:
        self.assertAlmostEqual(self.model.minimum_physical_soc, 0.1)
        full = BatteryClock(1.0)
        self.assertAlmostEqual(self.model.usable_wh(full), 89.91)
        depleted, reached_cutoff = self.model.spend(full, 89.91, 600.0)
        self.assertTrue(reached_cutoff)
        self.assertAlmostEqual(depleted.physical_soc, 0.1)
        restored = self.model.charge_to(depleted, 1.0)
        self.assertAlmostEqual(restored.elapsed_s - depleted.elapsed_s, 4405.0, delta=0.1)
        self.assertAlmostEqual(self.model.usable_wh(restored), 89.91)

    def test_partial_charge_advances_time_without_flight(self) -> None:
        at_station = BatteryClock(0.2, 10.0)
        partial = self.model.charge_to(at_station, 0.8)
        self.assertAlmostEqual(partial.elapsed_s, 2407.6)
        self.assertAlmostEqual(self.model.usable_wh(partial), 69.93)
        self.assertEqual(at_station, BatteryClock(0.2, 10.0))

    def test_flight_energy_is_not_reinterpreted_as_operating_soc(self) -> None:
        next_state, depleted = self.model.spend(BatteryClock(0.3), 9.99, 0.05)
        self.assertFalse(depleted)
        self.assertAlmostEqual(next_state.physical_soc, 0.2)
        self.assertAlmostEqual(next_state.elapsed_s, 0.05)


if __name__ == "__main__":
    unittest.main()
