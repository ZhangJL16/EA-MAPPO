from __future__ import annotations

import unittest

import numpy as np
import torch

from envs.UAVEnergyDeliverySAC import UAVEnergyDeliverySACEnv
from experiments.uav_safety_filter.benchmark import (
    make_scenarios,
    normalized_action_to_acceleration,
    perturb_perceived_obstacles,
    run_rollout,
    sac_observation,
)
from review_bundle.safety.collision.geometry3d import (
    BoxGeometry,
    Lidar3DConfig,
    Lidar3DModel,
    SphereGeometry,
    VerticalCylinderGeometry,
    raw_lidar_point_obstacles,
)
from review_bundle.safety.collision.filter import (
    SafetyFilterConfig,
    SafetyFilterMethod,
    UAVSafetyActionFilter,
)
from review_bundle.safety.collision.feasibility import (
    cylindrical_input_support,
    energy_optimal_safe_action,
    energy_to_go_smooth_upper_bound,
    exact_zoh_sphere_clearance,
    joint_feasibility_margin,
    single_constraint_feasibility_margin,
)
from review_bundle.safety.collision.hocbf import (
    HOCBFConfig,
    SphericalObstacle,
    actuator_polygon_constraints,
    aggregate_hocbf_constraint,
    energy_aware_quadratic,
    energy_to_go_action_gradient,
    energy_viability_residual,
    hocbf_sampled_data_residual_bound,
    one_step_supporting_constraint,
    project_polyhedral_qp,
    project_single_halfspace,
    select_top_k_constraints,
    sphere_hocbf_constraint,
    strengthen_constraint_for_sample_hold,
    velocity_polygon_constraints,
)
from review_bundle.safety.energy.gradients import (
    mc_energy_physical_gradient,
    physical_energy_state,
)
from review_bundle.safety.energy.mc_regression import EnergyToGoRegressor


class HOCBFDerivationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = HOCBFConfig(
            k1=1.3,
            k2=0.8,
            uav_radius=0.5,
            uncertainty_margin=0.2,
        )

    def test_second_order_formula_matches_direct_derivatives(self) -> None:
        position = np.array([4.0, -1.0, 2.0])
        velocity = np.array([-2.0, 0.7, -0.3])
        acceleration = np.array([1.2, -0.4, 0.5])
        obstacle = SphericalObstacle(
            center=np.array([1.0, -0.5, 1.5]),
            radius=1.1,
            velocity=np.array([0.2, -0.1, 0.0]),
            acceleration=np.array([0.1, 0.05, -0.02]),
        )
        constraint = sphere_hocbf_constraint(
            position,
            velocity,
            obstacle,
            self.config,
        )
        relative = position - obstacle.center
        relative_velocity = velocity - obstacle.velocity
        h = relative @ relative - constraint.safe_distance**2
        h_dot = 2.0 * relative @ relative_velocity
        h_ddot = (
            2.0 * relative_velocity @ relative_velocity
            + 2.0 * relative @ (acceleration - obstacle.acceleration)
        )
        psi2 = (
            h_ddot
            + (self.config.k1 + self.config.k2) * h_dot
            + self.config.k1 * self.config.k2 * h
        )
        self.assertAlmostEqual(constraint.h, h)
        self.assertAlmostEqual(constraint.h_dot, h_dot)
        self.assertAlmostEqual(
            constraint.row @ acceleration - constraint.lower_bound,
            psi2,
        )

    def test_moving_obstacle_acceleration_sign(self) -> None:
        position = np.array([3.0, 0.0, 0.0])
        velocity = np.zeros(3)
        outward_accelerating_obstacle = SphericalObstacle(
            center=np.zeros(3),
            radius=1.0,
            acceleration=np.array([2.0, 0.0, 0.0]),
        )
        static = SphericalObstacle(center=np.zeros(3), radius=1.0)
        moving_constraint = sphere_hocbf_constraint(
            position,
            velocity,
            outward_accelerating_obstacle,
            self.config,
        )
        static_constraint = sphere_hocbf_constraint(
            position,
            velocity,
            static,
            self.config,
        )
        self.assertGreater(moving_constraint.lower_bound, static_constraint.lower_bound)

    def test_aggregate_barrier_derivatives_match_finite_difference(self) -> None:
        position = np.array([0.5, 0.3, 0.2])
        velocity = np.array([0.4, -0.2, 0.1])
        acceleration = np.array([0.3, 0.15, -0.1])
        obstacles = [
            SphericalObstacle(np.array([3.0, 0.0, 0.0]), 0.5, identifier="a"),
            SphericalObstacle(np.array([0.0, 2.8, 0.4]), 0.7, identifier="b"),
            SphericalObstacle(np.array([-2.5, -1.0, 0.0]), 0.6, identifier="c"),
        ]

        def aggregate_h(time: float) -> float:
            p = position + velocity * time + 0.5 * acceleration * time**2
            values = []
            for obstacle in obstacles:
                distance = obstacle.radius + self.config.uav_radius + self.config.uncertainty_margin
                relative = p - obstacle.center
                values.append(relative @ relative - distance**2)
            values = np.asarray(values)
            rho = 0.7
            shifted = -rho * values
            shift = np.max(shifted)
            return float(-(np.log(np.exp(shifted - shift).sum()) + shift) / rho)

        constraints = [
            sphere_hocbf_constraint(position, velocity, obstacle, self.config)
            for obstacle in obstacles
        ]
        aggregate = aggregate_hocbf_constraint(
            constraints,
            position,
            velocity,
            obstacles,
            self.config,
            rho=0.7,
        )
        step = 1e-4
        numeric_first = (aggregate_h(step) - aggregate_h(-step)) / (2.0 * step)
        numeric_second = (
            aggregate_h(step) - 2.0 * aggregate_h(0.0) + aggregate_h(-step)
        ) / step**2
        expected_psi2 = (
            numeric_second
            + (self.config.k1 + self.config.k2) * numeric_first
            + self.config.k1 * self.config.k2 * aggregate_h(0.0)
        )
        self.assertAlmostEqual(aggregate.h, aggregate_h(0.0), places=8)
        self.assertAlmostEqual(aggregate.h_dot, numeric_first, places=5)
        self.assertAlmostEqual(
            aggregate.row @ acceleration - aggregate.lower_bound,
            expected_psi2,
            places=3,
        )

    def test_sample_hold_bound_dominates_observed_residual_change(self) -> None:
        rng = np.random.default_rng(91)
        obstacle = SphericalObstacle(
            center=np.array([12.0, -4.0, 1.0]),
            radius=2.0,
            velocity=np.array([0.2, -0.1, 0.0]),
            acceleration=np.array([0.1, 0.05, -0.02]),
        )
        dt = 0.05
        for _ in range(100):
            position = rng.uniform(-2.0, 2.0, size=3)
            velocity = rng.uniform(-4.0, 4.0, size=3)
            horizontal = rng.normal(size=2)
            horizontal *= rng.uniform(0.0, 5.0) / max(np.linalg.norm(horizontal), 1e-12)
            acceleration = np.array(
                [horizontal[0], horizontal[1], rng.uniform(-3.0, 3.0)]
            )
            initial = sphere_hocbf_constraint(position, velocity, obstacle, self.config)
            residual_zero = initial.row @ acceleration - initial.lower_bound
            bound = hocbf_sampled_data_residual_bound(
                position,
                velocity,
                obstacle,
                self.config,
                hold_dt=dt,
                horizontal_acceleration_limit=5.0,
                vertical_acceleration_limit=3.0,
            )
            for time in np.linspace(0.0, dt, 21):
                relative_acceleration = acceleration - obstacle.acceleration
                sample_position = (
                    position
                    + velocity * time
                    + 0.5 * acceleration * time**2
                )
                sample_velocity = velocity + acceleration * time
                sample_obstacle = SphericalObstacle(
                    center=(
                        obstacle.center
                        + obstacle.velocity * time
                        + 0.5 * obstacle.acceleration * time**2
                    ),
                    radius=obstacle.radius,
                    velocity=obstacle.velocity + obstacle.acceleration * time,
                    acceleration=obstacle.acceleration,
                )
                residual = (
                    sphere_hocbf_constraint(
                        sample_position,
                        sample_velocity,
                        sample_obstacle,
                        self.config,
                    ).row
                    @ acceleration
                    - sphere_hocbf_constraint(
                        sample_position,
                        sample_velocity,
                        sample_obstacle,
                        self.config,
                    ).lower_bound
                )
                self.assertLessEqual(abs(residual - residual_zero), bound + 1e-8)
                self.assertTrue(np.all(np.isfinite(relative_acceleration)))

    def test_sample_hold_strengthening_increases_required_slack(self) -> None:
        obstacle = SphericalObstacle(np.array([10.0, 0.0, 0.0]), 1.0)
        constraint = sphere_hocbf_constraint(
            np.zeros(3),
            np.array([5.0, 0.0, 0.0]),
            obstacle,
            self.config,
        )
        strengthened = strengthen_constraint_for_sample_hold(constraint, 7.5)
        self.assertAlmostEqual(
            strengthened.lower_bound,
            constraint.lower_bound + 7.5,
        )
        self.assertAlmostEqual(strengthened.sampled_data_margin, 7.5)


class ProjectionTests(unittest.TestCase):
    def test_closed_form_projection_is_feasible_and_optimal(self) -> None:
        nominal = np.array([-1.0, 0.4, -0.2])
        row = np.array([2.0, 0.0, 0.0])
        projected = project_single_halfspace(nominal, row, 1.0)
        np.testing.assert_allclose(projected, np.array([0.5, 0.4, -0.2]))
        self.assertAlmostEqual(row @ projected, 1.0)
        tangent = np.array([0.0, 0.3, -0.7])
        competitor = projected + tangent
        self.assertGreater(
            np.linalg.norm(competitor - nominal),
            np.linalg.norm(projected - nominal),
        )

    def test_polyhedral_projection_respects_physical_input_limits(self) -> None:
        actuator_rows, actuator_bounds = actuator_polygon_constraints(5.0, 3.0, facets=32)
        result = project_polyhedral_qp(
            center=np.array([9.0, 7.0, 6.0]),
            hessian=np.eye(3),
            rows=actuator_rows,
            lower_bounds=actuator_bounds,
        )
        self.assertTrue(result.feasible)
        self.assertLessEqual(np.linalg.norm(result.acceleration[:2]), 5.0 + 1e-7)
        self.assertLessEqual(abs(result.acceleration[2]), 3.0 + 1e-7)

    def test_velocity_constraints_prevent_post_step_speed_saturation(self) -> None:
        rows, bounds = velocity_polygon_constraints(
            np.array([20.0, 0.0, 5.0]),
            dt=0.05,
            horizontal_limit=20.0,
            vertical_limit=5.0,
            facets=32,
        )
        actuator_rows, actuator_bounds = actuator_polygon_constraints(5.0, 3.0)
        result = project_polyhedral_qp(
            center=np.array([5.0, 0.0, 3.0]),
            hessian=np.eye(3),
            rows=np.vstack((rows, actuator_rows)),
            lower_bounds=np.concatenate((bounds, actuator_bounds)),
        )
        self.assertTrue(result.feasible)
        next_velocity = np.array([20.0, 0.0, 5.0]) + 0.05 * result.acceleration
        self.assertLessEqual(np.linalg.norm(next_velocity[:2]), 20.0 + 1e-7)
        self.assertLessEqual(abs(next_velocity[2]), 5.0 + 1e-7)

    def test_multiple_barrier_projection(self) -> None:
        barrier_rows = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        barrier_bounds = np.array([0.5, -0.2, 0.1])
        actuator_rows, actuator_bounds = actuator_polygon_constraints(5.0, 3.0)
        rows = np.vstack((barrier_rows, actuator_rows))
        bounds = np.concatenate((barrier_bounds, actuator_bounds))
        result = project_polyhedral_qp(
            center=np.array([-1.0, -1.0, -1.0]),
            hessian=np.eye(3),
            rows=rows,
            lower_bounds=bounds,
        )
        self.assertTrue(result.feasible)
        self.assertLessEqual(result.max_violation, 1e-7)
        self.assertGreaterEqual(result.acceleration[0], 0.5 - 1e-7)
        self.assertGreaterEqual(result.acceleration[1], -0.2 - 1e-7)
        self.assertGreaterEqual(result.acceleration[2], 0.1 - 1e-7)

    def test_infeasible_constraints_are_not_silently_accepted(self) -> None:
        actuator_rows, actuator_bounds = actuator_polygon_constraints(1.0, 1.0, facets=16)
        rows = np.vstack((np.array([[1.0, 0.0, 0.0]]), actuator_rows))
        bounds = np.concatenate((np.array([2.0]), actuator_bounds))
        result = project_polyhedral_qp(
            center=np.zeros(3),
            hessian=np.eye(3),
            rows=rows,
            lower_bounds=bounds,
            max_iterations=100,
        )
        self.assertFalse(result.feasible)
        self.assertGreater(result.max_violation, 0.1)

    def test_top_k_uses_hocbf_nominal_slack(self) -> None:
        config = HOCBFConfig(k1=1.0, k2=1.0, uav_radius=0.5)
        position = np.zeros(3)
        velocity = np.array([2.0, 0.0, 0.0])
        obstacles = [
            SphericalObstacle(np.array([10.0, 0.0, 0.0]), 1.0, identifier="ahead"),
            SphericalObstacle(np.array([-3.0, 0.0, 0.0]), 1.0, identifier="behind"),
        ]
        constraints = [
            sphere_hocbf_constraint(position, velocity, obstacle, config)
            for obstacle in obstacles
        ]
        selected = select_top_k_constraints(constraints, np.zeros(3), top_k=1)
        self.assertEqual(selected[0].identifier, "ahead")

    def test_ttc_priority_keeps_far_closing_before_near_receding(self) -> None:
        position = np.zeros(3)
        velocity = np.array([10.0, 0.0, 0.0])
        constraints = [
            sphere_hocbf_constraint(
                position,
                velocity,
                SphericalObstacle(
                    np.array([-5.0, 0.0, 0.0]),
                    1.0,
                    identifier="near_receding",
                ),
                HOCBFConfig(),
            ),
            sphere_hocbf_constraint(
                position,
                velocity,
                SphericalObstacle(
                    np.array([20.0, 0.0, 0.0]),
                    1.0,
                    identifier="far_closing",
                ),
                HOCBFConfig(),
            ),
        ]
        distance_selected = select_top_k_constraints(
            constraints,
            np.zeros(3),
            1,
            priority="distance",
        )
        ttc_selected = select_top_k_constraints(
            constraints,
            np.zeros(3),
            1,
            priority="ttc",
        )
        self.assertEqual(distance_selected[0].identifier, "near_receding")
        self.assertEqual(ttc_selected[0].identifier, "far_closing")

    def test_one_step_constraint_places_prediction_outside_tangent_plane(self) -> None:
        obstacle = SphericalObstacle(np.zeros(3), 1.0)
        position = np.array([2.0, 0.0, 0.0])
        velocity = np.array([-3.0, 0.0, 0.0])
        nominal = np.zeros(3)
        constraint = one_step_supporting_constraint(
            position,
            velocity,
            nominal,
            obstacle,
            uav_radius=0.5,
            uncertainty_margin=0.0,
            dt=0.2,
        )
        safe = project_single_halfspace(nominal, constraint.row, constraint.lower_bound)
        relative_next = position + velocity * 0.2 + 0.5 * safe * 0.2**2
        self.assertGreaterEqual(constraint.row @ relative_next, 1.5 - 1e-9)


class EnergyObjectiveTests(unittest.TestCase):
    def test_energy_aware_objective_prefers_lower_acceleration_energy(self) -> None:
        nominal = np.array([1.0, 0.0, 0.0])
        intervention = np.eye(3)
        energy = np.diag([1.0, 1.0, 1.0])
        hessian, center = energy_aware_quadratic(
            nominal,
            intervention,
            energy,
            energy_weight=5.0,
            dt=0.2,
        )
        rows = np.array([[1.0, 0.0, 0.0]])
        bounds = np.array([0.2])
        result = project_polyhedral_qp(center, hessian, rows, bounds)
        self.assertTrue(result.feasible)
        self.assertLess(result.acceleration @ energy @ result.acceleration, nominal @ energy @ nominal)
        self.assertGreaterEqual(result.acceleration[0], 0.2 - 1e-8)

    def test_energy_quadratic_is_strictly_convex(self) -> None:
        hessian, _ = energy_aware_quadratic(
            np.zeros(3),
            np.diag([1.0, 2.0, 3.0]),
            np.diag([0.0, 0.5, 1.0]),
            energy_weight=4.0,
            dt=0.05,
        )
        self.assertGreater(np.min(np.linalg.eigvalsh(hessian)), 0.0)


class FeasibilityCertificateTests(unittest.TestCase):
    def test_cylindrical_support_matches_constructed_maximizer(self) -> None:
        row = np.array([3.0, -4.0, -2.0])
        horizontal_limit = 5.0
        vertical_limit = 3.0
        horizontal = horizontal_limit * row[:2] / np.linalg.norm(row[:2])
        action = np.array([horizontal[0], horizontal[1], -vertical_limit])
        self.assertAlmostEqual(
            cylindrical_input_support(row, horizontal_limit, vertical_limit),
            float(row @ action),
            places=10,
        )

    def test_single_constraint_margin_is_exact(self) -> None:
        row = np.array([3.0, 4.0, -2.0])
        expected = 5.0 * 5.0 + 3.0 * 2.0 - 7.0
        self.assertAlmostEqual(
            single_constraint_feasibility_margin(row, 7.0, 5.0, 3.0),
            expected,
        )

    def test_single_constraint_global_validation_uses_closed_form(self) -> None:
        row = np.array([[120.0, -45.0, 30.0]])
        bound = np.array([17_500.0])
        result = joint_feasibility_margin(
            row,
            bound,
            5.0,
            3.0,
            global_certificate_fallback=True,
        )
        expected = single_constraint_feasibility_margin(
            row[0],
            bound[0],
            5.0,
            3.0,
        )
        self.assertTrue(result.primal_success)
        self.assertTrue(result.dual_success)
        self.assertFalse(result.global_certificate_fallback_used)
        self.assertAlmostEqual(result.margin, expected, places=10)
        self.assertLess(result.duality_gap, 1e-10)

    def test_joint_margin_matches_primal_dual_and_feasibility(self) -> None:
        rows = np.array(
            [
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        bounds = np.array([0.25, -0.75, -0.5])
        result = joint_feasibility_margin(rows, bounds, 1.0, 1.0)
        self.assertTrue(result.primal_success)
        self.assertTrue(result.dual_success)
        self.assertLess(result.duality_gap, 1e-7)
        self.assertGreaterEqual(result.margin, 0.0)
        self.assertGreaterEqual(np.min(rows @ result.maximizing_action - bounds), -1e-7)

        impossible = joint_feasibility_margin(
            np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]),
            np.array([0.75, 0.75]),
            1.0,
            1.0,
        )
        self.assertLess(impossible.margin, 0.0)
        self.assertFalse(impossible.feasible)

    def test_joint_margin_recovers_nonsmooth_dual_certificate(self) -> None:
        rows = np.array(
            [
                [-0.0370233831, -0.2683135045, -1.1113529451],
                [-1.0400429512, -0.8145510118, -1.1164597946],
                [-0.9680552793, -0.5823834828, 1.9594341928],
                [0.6737767323, -1.0049471226, 0.9190446480],
                [0.3634919643, 0.5711847196, 0.2054252251],
            ]
        )
        bounds = np.array(
            [-0.8754951207, -4.5543677933, 7.3943157823, -3.9561774386, 6.1133062187]
        )
        result = joint_feasibility_margin(rows, bounds, 5.0, 3.0)
        self.assertTrue(result.primal_success)
        self.assertTrue(result.dual_success)
        self.assertLess(result.duality_gap, 1e-7)
        self.assertLess(result.margin, 0.0)

    def test_joint_margin_optional_global_certificate_closes_rare_dual_gap(self) -> None:
        rows = np.array(
            [
                [1.3684045728, 1.0036580212, -1.3950892986],
                [-0.8367658068, 0.4660336531, 0.1356025892],
                [0.2131299547, 1.5501874029, 0.9280276682],
                [-0.0618033466, -0.9685920348, 1.0572259916],
                [-1.1301113936, 0.9311215171, 0.7897176904],
            ]
        )
        bounds = np.array(
            [0.3921339214, -7.5818253808, 6.7613451535, -4.8640068274, -7.2305714423]
        )
        result = joint_feasibility_margin(
            rows,
            bounds,
            5.0,
            3.0,
            global_certificate_fallback=True,
        )
        self.assertTrue(result.primal_success)
        self.assertTrue(result.dual_success)
        self.assertTrue(result.global_certificate_fallback_used)
        self.assertLess(result.duality_gap, 1e-7)

    def test_exact_zoh_clearance_matches_dense_time_grid(self) -> None:
        position = np.array([2.0, 0.6, 0.0])
        velocity = np.array([-8.0, -0.5, 0.0])
        acceleration = np.array([4.0, 0.0, 0.0])
        result = exact_zoh_sphere_clearance(
            position,
            velocity,
            acceleration,
            safe_distance=1.0,
            hold_dt=0.5,
        )
        times = np.linspace(0.0, 0.5, 200_001)
        relative = (
            position[None, :]
            + times[:, None] * velocity[None, :]
            + 0.5 * times[:, None] ** 2 * acceleration[None, :]
        )
        dense = np.min(np.sum(relative * relative, axis=1) - 1.0)
        self.assertAlmostEqual(result.minimum_barrier, float(dense), places=8)

    def test_endpoints_can_be_safe_during_intersample_collision(self) -> None:
        result = exact_zoh_sphere_clearance(
            np.array([-2.0, 0.0, 0.0]),
            np.array([8.0, 0.0, 0.0]),
            np.zeros(3),
            safe_distance=0.5,
            hold_dt=0.5,
        )
        self.assertGreater(result.endpoint_minimum_barrier, 0.0)
        self.assertLess(result.minimum_barrier, 0.0)
        self.assertAlmostEqual(result.minimizing_time, 0.25)

    def test_energy_selector_has_pointwise_dominance_only(self) -> None:
        rows = np.array([[1.0, 0.0, 0.0]])
        bounds = np.array([0.2])
        energy = np.diag([2.0, 1.0, 1.0])
        standard_action = np.array([0.8, 0.2, 0.0])
        result = energy_optimal_safe_action(
            rows,
            bounds,
            energy,
            1.0,
            1.0,
            progress_direction=np.array([1.0, 0.0, 0.0]),
            minimum_progress=0.2,
            initial_action=standard_action,
        )
        self.assertTrue(result.feasible)
        self.assertLessEqual(result.objective, float(standard_action @ energy @ standard_action) + 1e-8)
        self.assertGreaterEqual(result.action[0], 0.2 - 1e-7)

    def test_smooth_energy_upper_bound_is_exact_for_quadratic(self) -> None:
        jacobian = np.vstack((0.5 * 0.2**2 * np.eye(3), 0.2 * np.eye(3)))
        reference_state = np.array([1.0, -0.5, 0.2, 0.3, -0.1, 0.4])
        delta = np.array([0.2, -0.3, 0.1])
        state_delta = jacobian @ delta
        reference_value = 0.5 * float(reference_state @ reference_state)
        upper = energy_to_go_smooth_upper_bound(
            reference_value,
            reference_state,
            jacobian,
            delta,
            gradient_lipschitz_constant=1.0,
        )
        actual = 0.5 * float((reference_state + state_delta) @ (reference_state + state_delta))
        self.assertAlmostEqual(upper, actual, places=12)

    def test_energy_viability_residual_matches_derived_inequality(self) -> None:
        acceleration = np.array([0.5, -0.2, 0.1])
        velocity = np.array([2.0, 0.0, -0.5])
        matrix = np.diag([0.3, 0.4, 0.5])
        grad_p = np.array([0.1, 0.2, -0.1])
        grad_v = np.array([0.4, -0.3, 0.2])
        residual = energy_viability_residual(
            acceleration,
            velocity,
            energy_matrix=matrix,
            fixed_power=0.7,
            grad_position=grad_p,
            grad_velocity=grad_v,
            battery_margin=2.0,
            gain=0.8,
        )
        expected = (
            0.8 * 2.0
            - 0.7
            - grad_p @ velocity
            - acceleration @ matrix @ acceleration
            - grad_v @ acceleration
        )
        self.assertAlmostEqual(residual, expected)

    def test_energy_to_go_action_gradient_matches_finite_difference(self) -> None:
        rng = np.random.default_rng(72)
        dt = 0.05
        relative_errors = []
        for _ in range(1000):
            position = rng.normal(size=3)
            velocity = rng.normal(size=3)
            action = rng.uniform(-1.0, 1.0, size=3)
            grad_position = 2.0 * position
            grad_velocity = 0.6 * velocity
            analytic = energy_to_go_action_gradient(
                grad_position,
                grad_velocity,
                dt,
            )

            def next_energy(candidate: np.ndarray) -> float:
                next_position = position + velocity * dt + 0.5 * candidate * dt**2
                next_velocity = velocity + candidate * dt
                return float(next_position @ next_position + 0.3 * next_velocity @ next_velocity)

            numeric = np.zeros(3)
            epsilon = 1e-5
            for index in range(3):
                perturbation = np.zeros(3)
                perturbation[index] = epsilon
                numeric[index] = (
                    next_energy(action + perturbation)
                    - next_energy(action - perturbation)
                ) / (2.0 * epsilon)
            next_position = position + velocity * dt + 0.5 * action * dt**2
            next_velocity = velocity + action * dt
            exact = energy_to_go_action_gradient(
                2.0 * next_position,
                0.6 * next_velocity,
                dt,
            )
            relative_errors.append(
                np.linalg.norm(exact - numeric) / max(np.linalg.norm(numeric), 1e-12)
            )
            self.assertEqual(analytic.shape, (3,))
        self.assertLess(max(relative_errors), 1e-7)

    def test_physical_energy_state_matches_live_environment(self) -> None:
        environment = UAVEnergyDeliverySACEnv()
        environment.reset(seed=811)
        position = np.array([1200.0, 1300.0, 140.0])
        velocity = np.array([3.0, -4.0, 1.0])
        goal = np.array([2100.0, 2200.0, 250.0])
        expected = environment.energy_state_for_goal(
            goal,
            position=position,
            velocity=velocity,
        )
        actual = physical_energy_state(
            position,
            velocity,
            goal,
            horizontal_velocity_limit=20.0,
            vertical_velocity_limit=5.0,
            distance_scale=float(np.linalg.norm([4000.0, 4000.0, 400.0])),
        )
        np.testing.assert_allclose(actual, expected, atol=1e-7)

    def test_mc_physical_gradient_matches_network_finite_difference(self) -> None:
        torch.manual_seed(19)
        estimator = EnergyToGoRegressor(
            battery_capacity=378.0,
            hidden_dim=16,
            seed=19,
            device="cpu",
        )
        position = np.array([1000.0, 1200.0, 100.0])
        velocity = np.array([4.0, -2.0, 0.5])
        goal = np.array([2600.0, 2300.0, 240.0])
        scale = float(np.linalg.norm([4000.0, 4000.0, 400.0]))
        result = mc_energy_physical_gradient(
            estimator,
            position,
            velocity,
            goal,
            horizontal_velocity_limit=20.0,
            vertical_velocity_limit=5.0,
            distance_scale=scale,
        )
        epsilon = 1.0
        numeric = np.zeros(3)
        for axis in range(3):
            delta = np.zeros(3)
            delta[axis] = epsilon
            plus = estimator.predict(
                physical_energy_state(
                    position + delta,
                    velocity,
                    goal,
                    horizontal_velocity_limit=20.0,
                    vertical_velocity_limit=5.0,
                    distance_scale=scale,
                )
            )
            minus = estimator.predict(
                physical_energy_state(
                    position - delta,
                    velocity,
                    goal,
                    horizontal_velocity_limit=20.0,
                    vertical_velocity_limit=5.0,
                    distance_scale=scale,
                )
            )
            numeric[axis] = (plus - minus) / (2.0 * epsilon)
        np.testing.assert_allclose(result.grad_position, numeric, rtol=2e-2, atol=2e-5)


class LidarGeometryTests(unittest.TestCase):
    def test_default_lidar_has_1024_fixed_range_sectors(self) -> None:
        config = Lidar3DConfig()
        lidar = Lidar3DModel(config)
        self.assertEqual(config.horizontal_sectors, 128)
        self.assertEqual(config.vertical_sectors, 8)
        self.assertEqual(config.num_sectors, 1024)
        self.assertEqual(lidar.directions.shape, (1024, 3))
        np.testing.assert_allclose(np.linalg.norm(lidar.directions, axis=1), 1.0)

    def test_lidar_detects_sphere_cylinder_and_box(self) -> None:
        lidar = Lidar3DModel(
            Lidar3DConfig(
                horizontal_sectors=128,
                vertical_sectors=9,
                max_range=100.0,
                vertical_fov_degrees=60.0,
            )
        )
        obstacles = [
            SphereGeometry(np.array([20.0, 0.0, 0.0]), 2.0),
            VerticalCylinderGeometry(np.array([0.0, 20.0]), 2.0, -5.0, 5.0),
            BoxGeometry(np.array([-22.0, -2.0, -2.0]), np.array([-18.0, 2.0, 2.0])),
        ]
        packet = lidar.measure(np.zeros(3), obstacles, timestamp=0.0)
        hit_indices = set(packet.obstacle_indices[packet.hit].tolist())
        self.assertEqual(hit_indices, {0, 1, 2})
        self.assertTrue(np.all(packet.distances[packet.hit] < 100.0))

    def test_same_local_scene_is_map_size_independent(self) -> None:
        lidar = Lidar3DModel(Lidar3DConfig())
        obstacles = [SphereGeometry(np.array([20.0, 0.0, 0.0]), 2.0)]
        first = lidar.measure(np.zeros(3), obstacles, timestamp=1.0)
        second = lidar.measure(np.zeros(3), obstacles, timestamp=2.0)
        np.testing.assert_array_equal(first.hit, second.hit)
        np.testing.assert_allclose(first.distances, second.distances)

    def test_raw_lidar_points_create_only_hit_constraints(self) -> None:
        lidar = Lidar3DModel(Lidar3DConfig())
        packet = lidar.measure(
            np.zeros(3),
            [SphereGeometry(np.array([20.0, 0.0, 0.0]), 2.0)],
            timestamp=0.0,
        )
        proxies = raw_lidar_point_obstacles(packet, point_radius=0.1)
        self.assertEqual(len(proxies), int(np.sum(packet.hit)))
        self.assertTrue(all(proxy.radius == 0.1 for proxy in proxies))

    def test_geometry_clearance_sign(self) -> None:
        sphere = SphereGeometry(np.zeros(3), 2.0)
        cylinder = VerticalCylinderGeometry(np.zeros(2), 2.0, -1.0, 1.0)
        box = BoxGeometry(-np.ones(3), np.ones(3))
        self.assertLess(sphere.clearance(np.zeros(3)), 0.0)
        self.assertLess(cylinder.clearance(np.zeros(3)), 0.0)
        self.assertLess(box.clearance(np.zeros(3)), 0.0)
        self.assertGreater(sphere.clearance(np.array([3.0, 0.0, 0.0])), 0.0)
        self.assertGreater(cylinder.clearance(np.array([3.0, 0.0, 0.0])), 0.0)
        self.assertGreater(box.clearance(np.array([3.0, 0.0, 0.0])), 0.0)


class SafetyFilterIntegrationTests(unittest.TestCase):
    def test_lexicographic_filter_preserves_feasible_nominal_progress(self) -> None:
        telemetry_energy = np.diag([0.04, 0.04, 0.08])
        action_filter = UAVSafetyActionFilter(
            SafetyFilterConfig(
                method=SafetyFilterMethod.SAMPLED_DATA_LEXICOGRAPHIC_ENERGY_HOCBF,
                sampled_data_robust=True,
                energy_matrix=telemetry_energy,
                energy_characteristic=1.0,
                safety_dt=0.05,
            ),
            HOCBFConfig(k1=1.0, k2=1.0, uav_radius=0.5),
        )
        nominal = np.array([2.0, 1.0, 0.5])
        result = action_filter.filter(
            np.array([0.0, 0.0, 0.0]),
            np.zeros(3),
            nominal,
            [],
            progress_direction=np.array([1.0, 0.0, 0.0]),
        )
        self.assertTrue(result.diagnostics.feasible)
        self.assertAlmostEqual(result.diagnostics.required_progress, 2.0, places=7)
        self.assertGreaterEqual(result.acceleration[0], 2.0 - 1e-7)
        self.assertAlmostEqual(result.acceleration[1], 0.0, places=4)
        self.assertAlmostEqual(result.acceleration[2], 0.0, places=4)

    def test_energy_gradient_filter_moves_against_positive_cost_gradient(self) -> None:
        nominal = np.array([1.0, 0.0, 0.0])
        action_filter = UAVSafetyActionFilter(
            SafetyFilterConfig(
                method=SafetyFilterMethod.ENERGY_GRADIENT_HOCBF,
                energy_matrix=np.zeros((3, 3)),
                energy_weight=0.0,
                energy_gradient_weight=0.1,
            )
        )
        output = action_filter.filter(
            np.zeros(3),
            np.zeros(3),
            nominal,
            [],
            energy_gradient=np.array([2.0, 0.0, 0.0]),
        )
        self.assertLess(output.acceleration[0], nominal[0])
        self.assertAlmostEqual(output.acceleration[1], 0.0, places=8)
        self.assertAlmostEqual(output.acceleration[2], 0.0, places=8)

    def test_energy_gradient_filter_requires_finite_gradient(self) -> None:
        action_filter = UAVSafetyActionFilter(
            SafetyFilterConfig(method=SafetyFilterMethod.ENERGY_GRADIENT_HOCBF)
        )
        with self.assertRaisesRegex(ValueError, "requires energy_gradient"):
            action_filter.filter(np.zeros(3), np.zeros(3), np.zeros(3), [])
        with self.assertRaisesRegex(ValueError, "finite"):
            action_filter.filter(
                np.zeros(3),
                np.zeros(3),
                np.zeros(3),
                [],
                energy_gradient=np.array([np.nan, 0.0, 0.0]),
            )

    def test_fixed_gradient_scale_preserves_magnitude_but_unit_direction_does_not(self) -> None:
        def execute(normalization: str, magnitude: float) -> float:
            output = UAVSafetyActionFilter(
                SafetyFilterConfig(
                    method=SafetyFilterMethod.ENERGY_GRADIENT_HOCBF,
                    energy_matrix=np.zeros((3, 3)),
                    energy_weight=0.0,
                    energy_gradient_weight=0.01,
                    energy_gradient_characteristic=1.0,
                    energy_gradient_normalization=normalization,
                )
            ).filter(
                np.zeros(3),
                np.zeros(3),
                np.zeros(3),
                [],
                energy_gradient=np.array([magnitude, 0.0, 0.0]),
            )
            return float(output.acceleration[0])

        fixed_one = execute("fixed", 1.0)
        fixed_two = execute("fixed", 2.0)
        directional_one = execute("unit_direction", 1.0)
        directional_two = execute("unit_direction", 2.0)
        self.assertAlmostEqual(fixed_two, 2.0 * fixed_one, places=6)
        self.assertAlmostEqual(directional_two, directional_one, places=6)

    def test_sampled_data_energy_gradient_combines_both_terms(self) -> None:
        obstacle = [SphericalObstacle(np.array([20.0, 0.0, 0.0]), 3.0)]
        combined = UAVSafetyActionFilter(
            SafetyFilterConfig(
                method=SafetyFilterMethod.SAMPLED_DATA_ENERGY_GRADIENT_HOCBF,
                sampled_data_robust=True,
                energy_matrix=np.zeros((3, 3)),
                energy_weight=0.0,
                energy_gradient_weight=0.1,
            )
        ).filter(
            np.zeros(3),
            np.array([10.0, 0.0, 0.0]),
            np.array([2.0, 0.0, 0.0]),
            obstacle,
            energy_gradient=np.array([1.0, 0.0, 0.0]),
        )
        gradient_only = UAVSafetyActionFilter(
            SafetyFilterConfig(
                method=SafetyFilterMethod.ENERGY_GRADIENT_HOCBF,
                energy_matrix=np.zeros((3, 3)),
                energy_weight=0.0,
                energy_gradient_weight=0.1,
            )
        ).filter(
            np.zeros(3),
            np.array([10.0, 0.0, 0.0]),
            np.array([2.0, 0.0, 0.0]),
            obstacle,
            energy_gradient=np.array([1.0, 0.0, 0.0]),
        )
        self.assertGreaterEqual(
            combined.diagnostics.intervention_norm,
            gradient_only.diagnostics.intervention_norm,
        )
        self.assertTrue(np.all(np.isfinite(combined.acceleration)))

    def test_sampled_data_instantaneous_energy_does_not_require_gradient(self) -> None:
        output = UAVSafetyActionFilter(
            SafetyFilterConfig(
                method=SafetyFilterMethod.SAMPLED_DATA_ENERGY_AWARE_HOCBF,
                sampled_data_robust=True,
                energy_matrix=np.diag([0.005, 0.005, 0.005]),
                energy_characteristic=0.0085,
                energy_weight=0.1,
            )
        ).filter(
            np.zeros(3),
            np.array([10.0, 0.0, 0.0]),
            np.array([2.0, 0.0, 0.0]),
            [SphericalObstacle(np.array([20.0, 0.0, 0.0]), 3.0)],
        )
        self.assertTrue(np.all(np.isfinite(output.acceleration)))

    def test_nominal_action_is_unchanged_when_safe(self) -> None:
        action_filter = UAVSafetyActionFilter(
            SafetyFilterConfig(method=SafetyFilterMethod.HOCBF),
            HOCBFConfig(k1=1.0, k2=1.0, uav_radius=0.5),
        )
        nominal = np.array([0.4, -0.2, 0.1])
        output = action_filter.filter(
            np.zeros(3),
            np.zeros(3),
            nominal,
            [SphericalObstacle(np.array([50.0, 0.0, 0.0]), 1.0)],
        )
        self.assertTrue(output.diagnostics.feasible)
        np.testing.assert_allclose(output.acceleration, nominal, atol=1e-7)

    def test_hocbf_intervenes_for_closing_obstacle(self) -> None:
        action_filter = UAVSafetyActionFilter(
            SafetyFilterConfig(method=SafetyFilterMethod.HOCBF),
            HOCBFConfig(k1=1.0, k2=1.0, uav_radius=0.5),
        )
        nominal = np.array([2.0, 0.0, 0.0])
        output = action_filter.filter(
            np.zeros(3),
            np.array([10.0, 0.0, 0.0]),
            nominal,
            [SphericalObstacle(np.array([8.0, 0.0, 0.0]), 1.0)],
        )
        self.assertTrue(output.diagnostics.feasible)
        self.assertGreater(output.diagnostics.intervention_norm, 0.0)
        self.assertGreaterEqual(output.diagnostics.minimum_executed_slack, -1e-7)

    def test_aggregate_filter_builds_one_collision_constraint(self) -> None:
        action_filter = UAVSafetyActionFilter(
            SafetyFilterConfig(
                method=SafetyFilterMethod.AGGREGATE_HOCBF,
                aggregate_rho=0.5,
            ),
            HOCBFConfig(k1=1.0, k2=1.0, uav_radius=0.5),
        )
        output = action_filter.filter(
            np.zeros(3),
            np.zeros(3),
            np.zeros(3),
            [
                SphericalObstacle(np.array([5.0, 0.0, 0.0]), 1.0),
                SphericalObstacle(np.array([0.0, 5.0, 0.0]), 1.0),
            ],
        )
        self.assertEqual(output.diagnostics.candidate_constraints, 1)

    def test_top_k_drops_are_reported(self) -> None:
        action_filter = UAVSafetyActionFilter(
            SafetyFilterConfig(method=SafetyFilterMethod.HOCBF, top_k=1),
        )
        output = action_filter.filter(
            np.zeros(3),
            np.zeros(3),
            np.zeros(3),
            [
                SphericalObstacle(np.array([5.0, 0.0, 0.0]), 1.0),
                SphericalObstacle(np.array([0.0, 6.0, 0.0]), 1.0),
            ],
        )
        self.assertEqual(output.diagnostics.candidate_constraints, 2)
        self.assertEqual(output.diagnostics.dropped_constraints, 1)

    def test_energy_filter_reduces_acceleration_when_safety_allows(self) -> None:
        standard = UAVSafetyActionFilter(
            SafetyFilterConfig(method=SafetyFilterMethod.HOCBF),
        )
        energy = UAVSafetyActionFilter(
            SafetyFilterConfig(
                method=SafetyFilterMethod.ENERGY_AWARE_HOCBF,
                energy_matrix=np.diag([0.005, 0.005, 0.005]),
                energy_characteristic=0.0085,
                energy_weight=1.0,
            ),
        )
        nominal = np.array([3.0, 0.0, 0.0])
        standard_output = standard.filter(
            np.zeros(3), np.zeros(3), nominal, []
        )
        energy_output = energy.filter(
            np.zeros(3), np.zeros(3), nominal, []
        )
        self.assertLess(
            np.linalg.norm(energy_output.acceleration),
            np.linalg.norm(standard_output.acceleration),
        )

    def test_solver_latency_is_instrumented(self) -> None:
        action_filter = UAVSafetyActionFilter(SafetyFilterConfig())
        output = action_filter.filter(
            np.zeros(3), np.zeros(3), np.zeros(3), []
        )
        self.assertGreaterEqual(output.diagnostics.constraint_build_seconds, 0.0)
        self.assertGreaterEqual(output.diagnostics.solver_seconds, 0.0)
        self.assertGreaterEqual(
            output.diagnostics.total_seconds,
            output.diagnostics.solver_seconds,
        )


class HandDesignedPhysicsTests(unittest.TestCase):
    def make_filter(
        self,
        method: SafetyFilterMethod = SafetyFilterMethod.HOCBF,
        **kwargs,
    ) -> UAVSafetyActionFilter:
        return UAVSafetyActionFilter(
            SafetyFilterConfig(method=method, **kwargs),
            HOCBFConfig(k1=1.0, k2=1.0, uav_radius=0.5),
        )

    def test_stationary_uav_near_obstacle_rejects_toward_action(self) -> None:
        output = self.make_filter().filter(
            np.zeros(3),
            np.zeros(3),
            np.array([5.0, 0.0, 0.0]),
            [SphericalObstacle(np.array([3.0, 0.0, 0.0]), 1.0)],
        )
        self.assertTrue(output.diagnostics.feasible)
        self.assertLess(output.acceleration[0], 5.0)
        self.assertGreaterEqual(output.diagnostics.minimum_executed_slack, -1e-6)

    def test_moving_away_obstacle_does_not_require_extra_braking(self) -> None:
        output = self.make_filter().filter(
            np.zeros(3),
            np.array([-5.0, 0.0, 0.0]),
            np.array([-1.0, 0.0, 0.0]),
            [SphericalObstacle(np.array([8.0, 0.0, 0.0]), 1.0)],
        )
        self.assertTrue(output.diagnostics.feasible)
        np.testing.assert_allclose(output.acceleration, np.array([-1.0, 0.0, 0.0]))

    def test_max_speed_toward_obstacle_triggers_intervention(self) -> None:
        output = self.make_filter().filter(
            np.zeros(3),
            np.array([20.0, 0.0, 0.0]),
            np.array([5.0, 0.0, 0.0]),
            [SphericalObstacle(np.array([60.0, 0.0, 0.0]), 10.0)],
        )
        self.assertGreater(output.diagnostics.intervention_norm, 0.0)
        self.assertLess(output.acceleration[0], 5.0)

    def test_side_pass_constraint_can_induce_lateral_action(self) -> None:
        output = self.make_filter().filter(
            np.zeros(3),
            np.array([4.0, 0.0, 0.0]),
            np.array([2.0, 0.0, 0.0]),
            [SphericalObstacle(np.array([10.0, 3.0, 0.0]), 2.0)],
        )
        self.assertTrue(np.all(np.isfinite(output.acceleration)))
        self.assertGreater(output.diagnostics.intervention_norm, 0.0)
        self.assertNotEqual(output.acceleration[1], 0.0)

    def test_two_obstacle_corridor_has_two_candidate_rows(self) -> None:
        output = self.make_filter().filter(
            np.zeros(3),
            np.array([5.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
            [
                SphericalObstacle(np.array([10.0, 4.0, 0.0]), 3.0),
                SphericalObstacle(np.array([10.0, -4.0, 0.0]), 3.0),
            ],
        )
        self.assertEqual(output.diagnostics.candidate_constraints, 2)
        self.assertTrue(np.all(np.isfinite(output.acceleration)))

    def test_three_constraint_conflict_is_explicit(self) -> None:
        output = self.make_filter().filter(
            np.zeros(3),
            np.zeros(3),
            np.zeros(3),
            [
                SphericalObstacle(np.array([1.0, 0.0, 0.0]), 1.0),
                SphericalObstacle(np.array([-0.5, 0.87, 0.0]), 1.0),
                SphericalObstacle(np.array([-0.5, -0.87, 0.0]), 1.0),
            ],
        )
        self.assertFalse(output.diagnostics.feasible)
        self.assertTrue(output.diagnostics.fallback_used)

    def test_vertical_obstacle_can_change_vertical_acceleration(self) -> None:
        output = self.make_filter().filter(
            np.zeros(3),
            np.array([10.0, 0.0, 0.0]),
            np.array([2.0, 0.0, 0.0]),
            [SphericalObstacle(np.array([10.0, 0.0, 3.0]), 3.0)],
        )
        self.assertTrue(np.all(np.isfinite(output.acceleration)))
        self.assertNotEqual(output.acceleration[2], 0.0)

    def test_lidar_packet_can_be_held_for_one_static_sensor_cycle(self) -> None:
        lidar = Lidar3DModel(Lidar3DConfig())
        geometry = SphereGeometry(np.array([20.0, 0.0, 0.0]), 5.0)
        packet = lidar.measure(np.zeros(3), [geometry], timestamp=0.0)
        self.assertGreater(np.sum(packet.hit), 0)
        obstacle = SphericalObstacle(geometry.center, geometry.radius)
        output = self.make_filter().filter(
            np.array([0.5, 0.0, 0.0]),
            np.array([5.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
            [obstacle],
        )
        self.assertTrue(np.all(np.isfinite(output.acceleration)))
        self.assertEqual(packet.timestamp, 0.0)

    def test_solver_delay_deadline_miss_is_reported(self) -> None:
        output = self.make_filter(deadline_seconds=1e-12).filter(
            np.zeros(3),
            np.zeros(3),
            np.zeros(3),
            [SphericalObstacle(np.array([5.0, 0.0, 0.0]), 1.0)],
        )
        self.assertTrue(output.diagnostics.deadline_missed)

    def test_sampled_data_mode_is_stricter_than_continuous_row(self) -> None:
        obstacle = [SphericalObstacle(np.array([20.0, 0.0, 0.0]), 3.0)]
        standard = self.make_filter().filter(
            np.zeros(3),
            np.array([10.0, 0.0, 0.0]),
            np.array([2.0, 0.0, 0.0]),
            obstacle,
        )
        robust = self.make_filter(sampled_data_robust=True).filter(
            np.zeros(3),
            np.array([10.0, 0.0, 0.0]),
            np.array([2.0, 0.0, 0.0]),
            obstacle,
        )
        self.assertGreaterEqual(
            robust.diagnostics.intervention_norm,
            standard.diagnostics.intervention_norm,
        )

    def test_aggregate_sample_hold_mode_is_rejected_without_a_bound(self) -> None:
        with self.assertRaisesRegex(ValueError, "not derived"):
            self.make_filter(
                SafetyFilterMethod.AGGREGATE_HOCBF,
                sampled_data_robust=True,
            ).filter(
                np.zeros(3),
                np.zeros(3),
                np.zeros(3),
                [SphericalObstacle(np.array([5.0, 0.0, 0.0]), 1.0)],
            )


class ControlledBenchmarkContractTests(unittest.TestCase):
    def test_bounded_range_perturbation_respects_declared_clip(self) -> None:
        obstacle = SphericalObstacle(
            np.array([20.0, 0.0, 0.0]),
            2.0,
            identifier="bounded_noise",
        )
        perceived = perturb_perceived_obstacles(
            np.zeros(3),
            np.array([0]),
            (obstacle,),
            range_noise_std=0.5,
            range_noise_clip_sigma=3.0,
            obstacle_dropout_probability=0.0,
            rng=np.random.default_rng(4),
        )
        self.assertEqual(len(perceived), 1)
        self.assertLessEqual(
            np.linalg.norm(perceived[0].center - obstacle.center),
            1.5 + 1e-12,
        )

    def test_obstacle_level_dropout_is_explicit(self) -> None:
        obstacles = tuple(
            SphericalObstacle(np.array([20.0 + index, 0.0, 0.0]), 2.0)
            for index in range(100)
        )
        perceived = perturb_perceived_obstacles(
            np.zeros(3),
            np.arange(100),
            obstacles,
            range_noise_std=0.0,
            range_noise_clip_sigma=3.0,
            obstacle_dropout_probability=0.5,
            rng=np.random.default_rng(8),
        )
        self.assertGreater(len(perceived), 0)
        self.assertLess(len(perceived), len(obstacles))

    def test_benchmark_observation_matches_live_environment_without_mutation(self) -> None:
        environment = UAVEnergyDeliverySACEnv()
        environment.reset(seed=11)
        original_position = environment.agent.pos.copy()
        original_velocity = environment.agent.vel.copy()
        original_goal = environment.current_task_point.copy()
        diagnostic_position = np.array([1200.0, 1400.0, 150.0])
        diagnostic_velocity = np.array([4.0, -3.0, 1.0])
        diagnostic_goal = np.array([2200.0, 2400.0, 250.0])
        expected = environment.sac_observation_for_goal(
            diagnostic_goal,
            position=diagnostic_position,
            velocity=diagnostic_velocity,
        )
        actual = sac_observation(
            diagnostic_position,
            diagnostic_velocity,
            diagnostic_goal,
        )
        np.testing.assert_allclose(actual, expected, atol=1e-7)
        np.testing.assert_array_equal(environment.agent.pos, original_position)
        np.testing.assert_array_equal(environment.agent.vel, original_velocity)
        np.testing.assert_array_equal(environment.current_task_point, original_goal)

    def test_normalized_action_mapping_matches_agent_dynamics_before_saturation(self) -> None:
        environment = UAVEnergyDeliverySACEnv()
        environment.reset(seed=12)
        action = np.array([0.6, 0.8, -0.5])
        expected = normalized_action_to_acceleration(action)
        commanded, realized = environment.agent.update_velocity(action, environment.physics_dt)
        np.testing.assert_allclose(commanded, expected, atol=1e-7)
        np.testing.assert_allclose(realized, expected, atol=1e-5)

    def test_scenario_generation_is_reproducible(self) -> None:
        first = make_scenarios(12, 9)
        second = make_scenarios(12, 9)
        self.assertEqual([row.identifier for row in first], [row.identifier for row in second])
        for left, right in zip(first, second):
            np.testing.assert_allclose(left.start, right.start)
            np.testing.assert_allclose(left.goal, right.goal)

    def test_short_rollout_produces_all_audit_fields(self) -> None:
        scenario = make_scenarios(1, 3)[0]

        def direct_policy(observation: np.ndarray) -> np.ndarray:
            return np.array([observation[3], observation[4], observation[5]])

        result = run_rollout(
            direct_policy,
            scenario,
            "second_order_hocbf",
        )
        self.assertGreater(result.steps, 0)
        self.assertTrue(np.isfinite(result.realized_energy))
        self.assertTrue(np.isfinite(result.minimum_clearance))
        self.assertGreaterEqual(result.mean_solver_seconds, 0.0)


if __name__ == "__main__":
    unittest.main()
