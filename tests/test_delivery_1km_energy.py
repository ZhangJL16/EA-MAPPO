"""Physical sanity and trace-accounting checks for the assumed power model."""

import unittest
from dataclasses import fields, replace

import numpy as np

from delivery_1km.energy import EnergyAssumption, account_flight_trace, rated_hover_equivalent_power_w, step_energy
from delivery_1km.energy_profiles import BASE_ENERGY_ASSUMPTION, ENERGY_PROFILES
from delivery_1km.energy_audit import build_manifest
from delivery_1km.scenario import M100_1KM


REFERENCE_FOR_TESTS = EnergyAssumption(
    name="illustrative_test_only",
    usable_energy_fraction=0.90,
    hover_power_multiplier=1.0,
    induced_power_fraction=0.45,
    profile_power_fraction=0.45,
    propeller_diameter_m=0.3302,
    rotor_tip_speed_mps=60.0,
    drag_area_m2=0.10,
    propulsive_efficiency=0.70,
    additional_electronics_w=21.5,
    thrust_headroom_over_mtow=1.35,
)


class EnergyTests(unittest.TestCase):
    def test_frozen_sensitivity_changes_one_parameter_at_a_time(self) -> None:
        self.assertEqual(len(ENERGY_PROFILES), 19)
        self.assertEqual(len({profile.name for profile in ENERGY_PROFILES}), 19)
        for profile in ENERGY_PROFILES[1:]:
            changed = [field.name for field in fields(profile)
                       if field.name != "name" and getattr(profile, field.name) != getattr(BASE_ENERGY_ASSUMPTION, field.name)]
            self.assertEqual(changed, [profile.name.rsplit("_", 1)[0]])

    def test_energy_audit_manifest_references_only_completed_navigation(self) -> None:
        manifest = build_manifest()
        self.assertEqual(len(manifest["jobs"]), 96)
        self.assertEqual(len(manifest["profiles"]), 19)
        self.assertEqual(manifest["payloads_kg"], [0.0, 0.25, 0.5])

    def test_manufacturer_hover_anchors_increase_with_payload(self) -> None:
        base = M100_1KM.base_mass_kg
        powers = [rated_hover_equivalent_power_w(base + load) for load in (0, 0.5, 1.0)]
        self.assertTrue(powers[0] < powers[1] < powers[2])
        self.assertAlmostEqual(powers[0] * 22.0 / 60.0, M100_1KM.nominal_battery_wh)
        available = REFERENCE_FOR_TESTS.usable_energy_fraction * M100_1KM.nominal_battery_wh
        calibrated_power = REFERENCE_FOR_TESTS.usable_energy_fraction * powers[0]
        self.assertAlmostEqual(60.0 * available / calibrated_power, 22.0)

    def test_payload_climb_and_acceleration_have_distinct_energy_effects(self) -> None:
        zero = (0.0, 0.0, 0.0)
        hover = step_energy(zero, zero, 0.0, REFERENCE_FOR_TESTS)
        loaded = step_energy(zero, zero, 0.5, REFERENCE_FOR_TESTS)
        climbing = step_energy((0.0, 0.0, 2.0), zero, 0.0, REFERENCE_FOR_TESTS)
        accelerating = step_energy(zero, (3.0, 0.0, 2.0), 0.0, REFERENCE_FOR_TESTS)
        moving = step_energy((7.0, 0.0, 0.0), zero, 0.0, REFERENCE_FOR_TESTS)
        self.assertLess(hover.power_w, loaded.power_w)
        self.assertLess(hover.power_w, climbing.power_w)
        self.assertLess(hover.power_w, accelerating.power_w)
        self.assertNotAlmostEqual(hover.power_w, moving.power_w)
        self.assertGreater(moving.tilt_deg, 0.0)

    def test_assumed_thrust_and_published_tilt_are_checked_separately(self) -> None:
        tight_headroom = replace(REFERENCE_FOR_TESTS, thrust_headroom_over_mtow=1.10)
        limit = step_energy((0.0, 0.0, 0.0), (0.0, 0.0, 3.0), 0.5, tight_headroom)
        self.assertIn("assumed_thrust_limit", limit.violated)
        tilt = step_energy((0.0, 0.0, 0.0), (8.0, 0.0, 0.0), 0.0, REFERENCE_FOR_TESTS)
        self.assertIn("published_tilt_limit", tilt.violated)

    def test_goal_arrival_velocity_zeroing_does_not_create_motor_spike(self) -> None:
        start = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        alternate_terminal = start.copy()
        alternate_terminal[-1] = (1.0, 0.0, 0.0)
        controls = np.zeros((2, 3))
        zeroed = account_flight_trace(start, controls, 0.0, REFERENCE_FOR_TESTS)
        retained = account_flight_trace(alternate_terminal, controls, 0.0, REFERENCE_FOR_TESTS)
        self.assertAlmostEqual(zeroed.energy_wh, retained.energy_wh)
        self.assertEqual(zeroed.infeasible_steps, retained.infeasible_steps)


if __name__ == "__main__":
    unittest.main()
