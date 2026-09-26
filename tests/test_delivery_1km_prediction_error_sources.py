"""Check mutually exclusive labels at the energy and execution boundaries."""

import unittest

from delivery_1km.return_prediction_error_sources import false_safe_category


class PredictionErrorSourceTest(unittest.TestCase):
    def test_base_residual_suffices_even_when_profile_also_raises_energy(self):
        self.assertEqual(false_safe_category(
            predicted_wh=9, base_wh=11, actual_wh=12,
            remaining_wh=10, physical_valid=True),
            "energy_base_error_sufficient")

    def test_profile_shift_crosses_a_valid_base_reserve(self):
        self.assertEqual(false_safe_category(
            predicted_wh=9, base_wh=9.5, actual_wh=11,
            remaining_wh=10, physical_valid=True),
            "energy_profile_shift_needed")

    def test_execution_failure_is_not_relabelled_as_energy_error(self):
        self.assertEqual(false_safe_category(
            predicted_wh=9, base_wh=9.5, actual_wh=9.7,
            remaining_wh=10, physical_valid=False),
            "physical_only")
        self.assertIsNone(false_safe_category(
            predicted_wh=11, base_wh=12, actual_wh=13,
            remaining_wh=10, physical_valid=False))


if __name__ == "__main__":
    unittest.main()
