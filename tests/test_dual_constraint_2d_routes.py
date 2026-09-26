"""Exact-edge checks for a fixed-height route around a synthetic disc."""

import unittest

import numpy as np

from dual_constraint_2d.routes import PlanarRouter
from dual_constraint_2d.synthetic import make_synthetic_map


class PlanarRouteTest(unittest.TestCase):
    def test_route_never_uses_overflight_and_edges_are_clear(self) -> None:
        case = make_synthetic_map(0)
        obstacle = case.world.obstacles[0]
        endpoints = None
        for axis in (np.array((1.0, 0.0)), np.array((0.0, 1.0))):
            a = obstacle.center_xy - axis * (obstacle.radius + 3.0)
            b = obstacle.center_xy + axis * (obstacle.radius + 3.0)
            if case.is_clear(a) and case.is_clear(b) and not case.segment_clear(a, b):
                endpoints = a, b
                break
        self.assertIsNotNone(endpoints)
        router = PlanarRouter(case)
        route = router.plan(*endpoints)
        self.assertGreater(len(route.waypoints_xy), 2)
        self.assertGreater(route.length_m, np.linalg.norm(endpoints[1] - endpoints[0]))
        self.assertTrue(all(case.segment_clear(a, b) for a, b in zip(route.waypoints_xy, route.waypoints_xy[1:])))
        self.assertTrue(all(case.xyz(p)[2] == 2.0 for p in route.waypoints_xy))


if __name__ == "__main__":
    unittest.main()
