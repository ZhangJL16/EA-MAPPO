"""Check that the stated charging assumption is internally consistent."""

import unittest

from delivery_1km import CHARGING_SENSITIVITY, M100_1KM, ChargingAssumption


class ChargingTests(unittest.TestCase):
    def test_partial_charge_is_additive_and_never_free(self) -> None:
        model = CHARGING_SENSITIVITY[1]
        battery = M100_1KM.nominal_battery_wh
        first = model.seconds_to_soc(0.2, 0.6, battery)
        second = model.seconds_to_soc(0.6, 0.9, battery)
        self.assertGreater(first, 0)
        self.assertGreater(second, 0)
        self.assertAlmostEqual(first + second, model.seconds_to_soc(0.2, 0.9, battery))
        self.assertEqual(model.seconds_to_soc(0.6, 0.6, battery), 0)

    def test_taper_slows_high_soc_charging(self) -> None:
        model = CHARGING_SENSITIVITY[1]
        battery = M100_1KM.nominal_battery_wh
        self.assertGreater(model.seconds_to_soc(0.8, 1.0, battery), model.seconds_to_soc(0.6, 0.8, battery))
        self.assertGreater(model.battery_power_w(0.5), model.battery_power_w(1.0))
        self.assertLessEqual(model.battery_power_w(0.5), model.charger_input_limit_w)

    def test_sensitivity_order_and_invalid_inputs(self) -> None:
        battery = M100_1KM.nominal_battery_wh
        times = [model.seconds_to_soc(0.3, 0.95, battery) for model in CHARGING_SENSITIVITY]
        self.assertLess(times[0], times[1])
        self.assertLess(times[1], times[2])
        with self.assertRaises(ValueError):
            ChargingAssumption("invalid", battery_efficiency=1.1)
        with self.assertRaises(ValueError):
            CHARGING_SENSITIVITY[1].seconds_to_soc(0.8, 0.2, battery)


if __name__ == "__main__":
    unittest.main()
