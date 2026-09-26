"""Only the newly selected platform and world configuration are checked here."""

from dataclasses import replace
from math import hypot
import unittest

import numpy as np

from delivery_1km import M100_1KM, make_map, make_world, sample_site, station_for_map
from nav3d.geometry import World3D


class ScenarioTests(unittest.TestCase):
    def test_m100_mass_and_navigation_limits(self) -> None:
        spec = M100_1KM
        self.assertEqual(spec.size_m, (1000.0, 1000.0, 100.0))
        self.assertAlmostEqual(spec.maximum_loaded_mass_kg, 3.355)
        self.assertLess(spec.maximum_loaded_mass_kg, spec.max_takeoff_mass_kg)
        self.assertAlmostEqual(spec.nominal_battery_wh, 99.9)
        self.assertEqual(spec.navigation_guard_steps, 16000)
        cfg = spec.navigation_config()
        self.assertEqual((cfg.max_horizontal_speed, cfg.max_vertical_speed), (7.0, 4.0))
        self.assertEqual((cfg.max_horizontal_acceleration, cfg.max_vertical_acceleration), (4.0, 3.0))


    def test_overweight_candidate_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "takeoff limit"):
            replace(M100_1KM, cargo_masses_kg=(0.0, 1.0))


    def test_finite_height_map_is_deterministic_and_station_clear(self) -> None:
        first_case, second_case = make_map(2), make_map(2)
        first, second = first_case.world, second_case.world
        self.assertEqual(first.size.tolist(), [1000.0, 1000.0, 100.0])
        self.assertEqual(len(first.obstacles), M100_1KM.obstacle_count)
        self.assertEqual(first_case.station_m, second_case.station_m)
        self.assertNotEqual(first_case.station_m[:2], station_for_map(3)[:2])
        self.assertGreater(first.clearance(first_case.station_m, M100_1KM.navigation_config().inflation), 0)
        height_bands = [0, 0, 0]
        for left, right in zip(first.obstacles, second.obstacles):
            np.testing.assert_array_equal(left.center_xy, right.center_xy)
            self.assertEqual(left.radius, right.radius)
            self.assertEqual(left.z_high, right.z_high)
            self.assertTrue(15.0 <= left.z_high < 75.0)
            height_bands[min(int((left.z_high - 15.0) / 20.0), 2)] += 1
            self.assertTrue(12.5 <= left.radius <= 30.0)
            self.assertGreaterEqual(
                hypot(*(left.center_xy - np.asarray(first_case.station_m[:2]))),
                left.radius + M100_1KM.station_obstacle_clearance_m,
            )
        self.assertEqual(height_bands, [8, 8, 8])
        cylinder = first.obstacles[0]
        overflight_z = cylinder.z_high + 2.0
        local_world = World3D(first.size, (cylinder,))
        self.assertTrue(local_world.segment_clear(
            (cylinder.center_xy[0] - cylinder.radius - 2.0, cylinder.center_xy[1], overflight_z),
            (cylinder.center_xy[0] + cylinder.radius + 2.0, cylinder.center_xy[1], overflight_z),
            M100_1KM.navigation_config().inflation,
        ))


    def test_map_id_validation(self) -> None:
        with self.assertRaisesRegex(ValueError, "map_id"):
            make_world(-1)
        with self.assertRaisesRegex(ValueError, "map_id"):
            station_for_map(-1)

    def test_ground_and_roof_sites_are_reachable_and_repeatable(self) -> None:
        case = make_map(0)
        for kind in ("ground", "roof"):
            first = sample_site(case, 7, kind)
            second = sample_site(case, 7, kind)
            self.assertEqual(first, second)
            self.assertEqual(first.kind, kind)
            self.assertGreater(case.world.clearance(first.position_m, M100_1KM.body_radius_m + 1.0), 0)
            if kind == "ground":
                self.assertEqual(first.position_m[2], M100_1KM.station_handoff_height_m)
                self.assertIsNone(first.support_name)
            else:
                obstacle = next(item for item in case.world.obstacles if item.name == first.support_name)
                self.assertAlmostEqual(first.position_m[2], obstacle.z_high + 2.0)


if __name__ == "__main__":
    unittest.main()
