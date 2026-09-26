"""Cross-check every frozen energy profile against scalar accounting."""

import unittest

import numpy as np

from delivery_1km.energy import account_flight_trace
from delivery_1km.energy_profiles import ENERGY_PROFILES
from delivery_1km.energy_vectorized import account_flight_trace_vectorized


class VectorizedEnergyTests(unittest.TestCase):
    def test_matches_all_profiles_on_mixed_flight_states(self) -> None:
        rng = np.random.default_rng(20260926)
        velocities = rng.uniform((-7, -7, -4), (7, 7, 5), size=(129, 3))
        accelerations = rng.uniform((-4, -4, -3), (4, 4, 3), size=(128, 3))
        for payload in (0.0, 0.25, 0.5):
            for profile in ENERGY_PROFILES:
                with self.subTest(payload=payload, profile=profile.name):
                    original = account_flight_trace(velocities, accelerations, payload, profile)
                    vectorized = account_flight_trace_vectorized(velocities, accelerations, payload, profile)
                    for field in ("energy_wh", "max_power_w", "max_thrust_n", "max_tilt_deg"):
                        self.assertAlmostEqual(getattr(original, field), getattr(vectorized, field), delta=1e-9)
                    self.assertEqual(original.infeasible_steps, vectorized.infeasible_steps)
                    self.assertEqual(original.first_infeasible_step, vectorized.first_infeasible_step)
                    self.assertEqual(original.violated, vectorized.violated)

    def test_matches_saved_moving_return_trace(self) -> None:
        import json
        from pathlib import Path

        output = Path(__file__).resolve().parents[1] / "artifacts/delivery_1km_return_audit_evidence_arm_20260926"
        result = json.loads((output / "results/step_00400.json").read_text(encoding="utf-8"))
        with np.load(output / result["return_trace"], allow_pickle=False) as trace:
            for payload in (0.0, 0.25, 0.5):
                for profile in ENERGY_PROFILES:
                    original = account_flight_trace(trace["velocities"], trace["accelerations"], payload, profile)
                    vectorized = account_flight_trace_vectorized(trace["velocities"], trace["accelerations"], payload, profile)
                    self.assertAlmostEqual(original.energy_wh, vectorized.energy_wh, delta=1e-9)
                    self.assertEqual(original.infeasible_steps, vectorized.infeasible_steps)


if __name__ == "__main__":
    unittest.main()
