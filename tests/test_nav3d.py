import unittest

import numpy as np

from nav3d import (
    BoxObstacle,
    CBFConfig,
    CBFController,
    CylinderObstacle,
    VisibilityPlanner,
    World3D,
    simulate_flight,
)
from nav3d.planner import NoRouteError
from nav3d.simulation import FlightState, step_flight


class GeometryTests(unittest.TestCase):
    def test_finite_cylinder_allows_overflight_but_blocks_level_crossing(self):
        world = World3D([20, 20, 12], (CylinderObstacle(np.array([10, 10]), 2, 0, 6),))
        self.assertFalse(world.segment_clear([4, 10, 3], [16, 10, 3], 0.5))
        self.assertTrue(world.segment_clear([4, 10, 8], [16, 10, 8], 0.5))

    def test_box_segment_and_clearance(self):
        world = World3D([20, 20, 12], (BoxObstacle(np.array([8, 8, 2]), np.array([12, 12, 9])),))
        self.assertFalse(world.segment_clear([4, 10, 5], [16, 10, 5], 0.5))
        self.assertTrue(world.segment_clear([4, 10, 10], [16, 10, 10], 0.5))
        self.assertGreater(world.clearance([4, 10, 5], 0.5), 0)


class PlannerControllerTests(unittest.TestCase):
    def test_wall_requires_three_dimensional_overflight(self):
        wall = BoxObstacle(np.array([10, 0, 0]), np.array([18, 30, 7]), "wall")
        world = World3D([30, 30, 15], (wall,))
        planner = VisibilityPlanner(world, body_radius=0.4, clearance_margin=0.2)
        route = planner.plan([3, 15, 2], [25, 15, 2])
        self.assertGreater(max(p[2] for p in route.waypoints), 7.6)
        self.assertTrue(all(world.segment_clear(a, b, 0.6) for a, b in zip(route.waypoints, route.waypoints[1:])))
        config = CBFConfig(
            body_radius=0.4,
            clearance_margin=0.2,
            max_horizontal_speed=7,
            max_vertical_speed=4,
            max_horizontal_acceleration=4,
            max_vertical_acceleration=3,
            activation_distance=15,
        )
        result = simulate_flight(world, [3, 15, 2], [25, 15, 2], config=config, max_steps=500)
        self.assertTrue(result.arrived)
        self.assertEqual(result.final_state.collision_count, 0)
        self.assertEqual(result.controller_infeasible_steps, 0)

    def test_cbf_intervenes_before_box_contact(self):
        world = World3D([20, 20, 12], (BoxObstacle(np.array([8, 4, 4]), np.array([10, 6, 6])),))
        controller = CBFController(world, CBFConfig(body_radius=0.4, clearance_margin=0.2, max_horizontal_speed=5, max_horizontal_acceleration=4, activation_distance=12))
        result = controller.action([6.8, 5, 5], [1, 0, 0], [15, 5, 5])
        self.assertTrue(result.feasible)
        self.assertTrue(result.intervened)
        self.assertLess(result.acceleration[0], 0)
        self.assertGreaterEqual(result.min_hold_clearance, 0)

    def test_fully_sealed_wall_reports_no_route(self):
        wall = BoxObstacle(np.array([10, 0, 0]), np.array([18, 30, 15]))
        world = World3D([30, 30, 15], (wall,))
        with self.assertRaises(NoRouteError):
            VisibilityPlanner(world, body_radius=0.4, clearance_margin=0.2).plan([3, 15, 2], [25, 15, 2])


class CollisionProtocolTests(unittest.TestCase):
    def test_contact_repairs_without_ending_task_and_clean_step_resets_penalty(self):
        world = World3D([10, 10, 10])
        config = CBFConfig(dt=0.2, body_radius=0.5, max_horizontal_speed=10, max_horizontal_acceleration=5)
        state = FlightState(np.array([0.51, 5, 5]), np.array([-1, 0, 0]))
        state, first = step_flight(world, state, [0, 0, 0], config)
        self.assertTrue(first.unified_contact)
        self.assertEqual(first.raw_penalty, -1.2)
        np.testing.assert_allclose(state.velocity, 0)
        np.testing.assert_allclose(state.position, [0.51, 5, 5])
        state, second = step_flight(world, state, [-5, 0, 0], config)
        self.assertTrue(second.unified_contact)
        self.assertAlmostEqual(second.raw_penalty, -0.42)
        for acceleration in ([5, 0, 0], [-5, 0, 0], [-5, 0, 0]):
            state, clean = step_flight(world, state, acceleration, config)
            self.assertFalse(clean.unified_contact)
        state, third = step_flight(world, state, [0, 0, 0], config)
        self.assertTrue(third.unified_contact)
        self.assertEqual(third.raw_penalty, -1.2)
        self.assertEqual(state.collision_count, 3)
        self.assertEqual(state.safety_cost, 3)
        self.assertAlmostEqual(state.raw_contact_penalty, -2.82)


if __name__ == "__main__":
    unittest.main()
