"""Decision-boundary checks for internal SOC stress accounting."""

import unittest

import numpy as np

from delivery_1km.return_soc_audit import SOC_GRID, classify, fit_excluding_map


class ReturnSocAuditTests(unittest.TestCase):
    def test_soc_grid_keeps_full_charge_primary_and_boundary_cases(self) -> None:
        self.assertEqual(SOC_GRID[-1], 1.0)
        self.assertEqual(tuple(sorted(set(SOC_GRID))), SOC_GRID)
        self.assertTrue(all(0 < soc <= 1 for soc in SOC_GRID))

    def test_depleted_state_is_not_a_false_safe_decision(self) -> None:
        self.assertEqual(classify(0.0, 0.0, 0.0, True), (False, False, False))
        self.assertEqual(classify(5.0, 4.0, 6.0, True), (True, True, False))
        self.assertEqual(classify(5.0, 6.0, 4.0, True), (True, False, True))
        self.assertEqual(classify(5.0, 4.0, 4.0, False), (True, True, False))
        self.assertEqual(classify(5.0, None, None, False), (True, False, False))

    def test_fitting_excludes_heldout_map(self) -> None:
        def row(map_id: int, length: float, target: float) -> dict:
            return {"map_id": map_id, "route_features": [1, length, 0, 0, 1],
                    "base_energy_wh_by_payload": {str(payload): target for payload in (0.0, 0.25, 0.5)}}

        training = [row(101, 10, 2), row(102, 20, 3), row(103, 30, 4)]
        poisoned = [row(100, 10, 1000), *training]
        expected = fit_excluding_map(training, 100)
        actual = fit_excluding_map(poisoned, 100)
        for payload in (0.0, 0.25, 0.5):
            np.testing.assert_allclose(actual[payload], expected[payload])


if __name__ == "__main__":
    unittest.main()
