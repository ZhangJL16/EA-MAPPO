from __future__ import annotations

import unittest

import numpy as np

from experiments.uav_recoverable_safety.diagnostics import (
    DynamicObstacleSpec,
    HardScenario,
    PerceptionMode,
    RecoverabilityDiagnosticConfig,
    VisibilityWindow,
    compute_pointwise_margin,
    make_hand_scenarios,
    make_random_hard_scenarios,
    run_scenario,
    search_counterexamples,
)
from review_bundle.safety.collision.hocbf import SphericalObstacle


def zero_policy(observation: np.ndarray) -> np.ndarray:
    del observation
    return np.zeros(3, dtype=np.float64)


class RecoverableSafetyDiagnosticTests(unittest.TestCase):
    def test_pointwise_margin_separates_feasible_and_infeasible_rows(self) -> None:
        config = RecoverabilityDiagnosticConfig()
        feasible = compute_pointwise_margin(
            np.zeros(3),
            np.zeros(3),
            [SphericalObstacle(np.array([20.0, 0.0, 0.0]), 1.0)],
            config,
        )
        infeasible = compute_pointwise_margin(
            np.zeros(3),
            np.array([20.0, 0.0, 0.0]),
            [SphericalObstacle(np.array([12.0, 0.0, 0.0]), 2.0)],
            config,
        )
        self.assertTrue(feasible.feasible)
        self.assertGreater(feasible.rho, 0.0)
        self.assertFalse(infeasible.feasible)
        self.assertLess(infeasible.rho, 0.0)

    def test_current_frame_constraint_disappears_during_dropout(self) -> None:
        scenario = HardScenario(
            "dropout_test",
            "dropout",
            np.array([100.0, 100.0, 100.0]),
            np.array([200.0, 100.0, 100.0]),
            np.zeros(3),
            (
                DynamicObstacleSpec(
                    "hidden",
                    np.array([120.0, 100.0, 100.0]),
                    2.0,
                ),
            ),
            (VisibilityWindow(2, 6, ("hidden",)),),
            max_steps=8,
        )
        records = run_scenario(
            scenario,
            zero_policy,
            perception_mode=PerceptionMode.CURRENT_FRAME,
        )
        self.assertTrue(records[0].perceived_obstacles)
        self.assertFalse(records[2].perceived_obstacles)
        self.assertIsNone(records[2].rho_perceived)
        self.assertIsNotNone(records[2].rho_true)

    def test_bounded_acceleration_track_expands_during_dropout(self) -> None:
        scenario = HardScenario(
            "bounded_track",
            "dropout",
            np.array([100.0, 100.0, 100.0]),
            np.array([200.0, 100.0, 100.0]),
            np.zeros(3),
            (
                DynamicObstacleSpec(
                    "hidden",
                    np.array([120.0, 100.0, 100.0]),
                    2.0,
                    velocity=np.array([1.0, 0.0, 0.0]),
                ),
            ),
            (VisibilityWindow(2, 8, ("hidden",)),),
            max_steps=8,
        )
        records = run_scenario(
            scenario,
            zero_policy,
            perception_mode=PerceptionMode.BOUNDED_ACCELERATION,
        )
        initial_radius = records[0].perceived_obstacles[0]["radius"]
        stale_radius = records[6].perceived_obstacles[0]["radius"]
        self.assertGreater(stale_radius, initial_radius)
        self.assertGreater(records[6].stale_obstacle_ages["hidden"], 0.0)

    def test_counterexample_search_detects_positive_to_negative_margin(self) -> None:
        scenario = HardScenario(
            "approach",
            "braking",
            np.array([100.0, 100.0, 100.0]),
            np.array([200.0, 100.0, 100.0]),
            np.array([5.0, 0.0, 0.0]),
            (
                DynamicObstacleSpec(
                    "blocking",
                    np.array([110.0, 100.0, 100.0]),
                    2.0,
                ),
            ),
            (VisibilityWindow(0, 20, ("blocking",)),),
            max_steps=20,
        )
        config = RecoverabilityDiagnosticConfig(lookahead_steps=12)
        records = run_scenario(scenario, zero_policy, config=config)
        found = search_counterexamples(
            scenario,
            records,
            perception_mode=PerceptionMode.CURRENT_FRAME,
            config=config,
        )
        self.assertTrue(found)
        self.assertGreater(found[0].rho_t, 0.0)
        self.assertLess(found[0].rho_future[-1], 0.0)
        self.assertTrue(found[0].relative_velocities)
        self.assertTrue(found[0].state["hocbf_rows"])
        self.assertTrue(found[0].state["hocbf_bounds"])
        self.assertEqual(
            found[0].state["actuator_constraint_representation"]["filter_solver_set"],
            "inscribed_polygon",
        )
        actuator = found[0].state["actuator_constraint_representation"]
        self.assertEqual(len(actuator["actuator_rows"]), 34)
        self.assertEqual(len(actuator["actuator_rows"]), len(actuator["actuator_bounds"]))
        self.assertEqual(len(actuator["velocity_rows"]), 34)
        self.assertEqual(len(actuator["velocity_rows"]), len(actuator["velocity_bounds"]))
        self.assertIsInstance(found[0].sampled_recovery_action_exists, bool)
        self.assertTrue(np.isfinite(found[0].best_sampled_next_rho))

    def test_strengthened_and_raw_margins_are_logged_separately(self) -> None:
        scenario = HardScenario(
            "strengthening",
            "sampling",
            np.array([100.0, 100.0, 100.0]),
            np.array([200.0, 100.0, 100.0]),
            np.array([20.0, 0.0, 0.0]),
            (
                DynamicObstacleSpec(
                    "blocking",
                    np.array([112.0, 100.0, 100.0]),
                    2.0,
                ),
            ),
            max_steps=4,
        )
        records = run_scenario(scenario, zero_policy)
        negative = [
            record
            for record in records
            if record.rho_true is not None and record.rho_true < 0.0
        ]
        self.assertTrue(negative)
        self.assertTrue(
            all(record.rho_true_unstrengthened is not None for record in negative)
        )

    def test_all_required_hand_families_exist(self) -> None:
        families = {scenario.family for scenario in make_hand_scenarios()}
        self.assertEqual(len(families), 10)
        self.assertIn("two_obstacle_squeeze", families)
        self.assertIn("vertical_horizontal_simultaneous_conflict", families)

    def test_random_generation_is_reproducible(self) -> None:
        first = make_random_hard_scenarios(20, 7)
        second = make_random_hard_scenarios(20, 7)
        for left, right in zip(first, second):
            np.testing.assert_allclose(left.start, right.start)
            np.testing.assert_allclose(left.goal, right.goal)
            for left_obstacle, right_obstacle in zip(left.obstacles, right.obstacles):
                np.testing.assert_allclose(left_obstacle.center, right_obstacle.center)


if __name__ == "__main__":
    unittest.main()
