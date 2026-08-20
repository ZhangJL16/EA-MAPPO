from __future__ import annotations

import unittest
from tempfile import TemporaryDirectory

import numpy as np
import torch

from review_bundle.envs.navigation.telemetry_cost import TelemetryCostConfig, TelemetryCostModel
from review_bundle.safety.collision.hocbf import SphericalObstacle
from review_bundle.safety.trajectory import (
    CandidateEvaluation,
    CoarseScreenMode,
    FeatureProvenance,
    HistoryBuffer,
    HistoryFrame,
    HistoryTubeMode,
    PhysicsRolloutConfig,
    ProposalConfig,
    TerminalEnergyEstimate,
    adaptive_history_set_membership,
    bounded_jerk_future_tube_lower_bound,
    certify_trajectory,
    coarse_safety_screen,
    future_position_error_tube,
    generate_trajectory_proposals,
    jerk_bounded_set_membership,
    lidar_temporal_flow,
    relative_motion_features,
    rollout_action_sequences,
    rollout_action_sequences_torch,
    rollout_single_sequence,
    select_safe_trajectory,
    trajectory_energy,
    trajectory_progress,
)
from experiments.history_safe_trajectory.history_ablation import (
    HistoryAblationConfig,
    run_history_ablation,
)


class HistoryAndMotionTests(unittest.TestCase):
    def frame(self, timestamp: float, control: np.ndarray | None = None) -> HistoryFrame:
        return HistoryFrame(
            timestamp=timestamp,
            position=np.array([timestamp, 0.0, 0.0]),
            velocity=np.array([1.0, 0.0, 0.0]),
            realized_acceleration=np.zeros(3),
            executed_control=control,
            lidar_ranges=np.array([10.0, 20.0]),
            lidar_valid=np.array([True, False]),
        )

    def test_history_buffer_lengths_order_and_capacity(self) -> None:
        for length in (2, 4, 8, 16):
            buffer = HistoryBuffer(length)
            for index in range(length + 2):
                buffer.append(self.frame(float(index)))
            self.assertEqual(len(buffer), length)
            self.assertEqual(buffer.snapshot()[0].timestamp, 2.0)
            self.assertTrue(buffer.full)
        with self.assertRaises(ValueError):
            HistoryBuffer(3)

    def test_history_rejects_noncausal_timestamp(self) -> None:
        buffer = HistoryBuffer(2)
        buffer.append(self.frame(1.0))
        with self.assertRaises(ValueError):
            buffer.append(self.frame(1.0))

    def test_relative_motion_closing_speed_ttc_and_provenance(self) -> None:
        features = relative_motion_features(
            np.zeros(3),
            np.array([2.0, 0.0, 0.0]),
            np.zeros(3),
            np.array([10.0, 0.0, 0.0]),
            contact_distance=1.0,
            braking_acceleration=2.0,
            sampling_period=0.2,
        )
        np.testing.assert_allclose(features.relative_position, [-10.0, 0.0, 0.0])
        self.assertAlmostEqual(features.closing_speed, 2.0)
        self.assertAlmostEqual(features.time_to_collision, 4.5)
        self.assertAlmostEqual(features.stopping_distance, 1.0)
        self.assertAlmostEqual(features.sampling_delay_distance, 0.4)
        self.assertTrue(
            all(value is FeatureProvenance.PHYSICALLY_DERIVED for value in features.provenance.values())
        )
        self.assertNotIn(FeatureProvenance.FUTURE_INFORMATION, features.provenance.values())

    def test_lidar_ego_motion_compensation_static_obstacle(self) -> None:
        flow = lidar_temporal_flow(
            previous_ranges=np.array([10.0]),
            current_ranges=np.array([9.0]),
            previous_valid=np.array([True]),
            current_valid=np.array([True]),
            ray_directions=np.array([[1.0, 0.0, 0.0]]),
            previous_uav_position=np.zeros(3),
            current_uav_position=np.array([1.0, 0.0, 0.0]),
            delta_time=0.1,
        )
        self.assertAlmostEqual(flow.raw_range_delta[0], -1.0)
        self.assertAlmostEqual(flow.ego_range_delta[0], 1.0)
        self.assertAlmostEqual(flow.motion_compensated_range_delta[0], 0.0)

    def test_history_ablation_captures_noise_reduction_and_staleness(self) -> None:
        with TemporaryDirectory() as directory:
            summary = run_history_ablation(
                HistoryAblationConfig(
                    cases=2000,
                    seed=91,
                    output_dir=f"{directory}/ablation",
                )
            )
        constant = summary["regimes"]["constant_velocity"]
        abrupt = summary["regimes"]["abrupt_velocity_change"]
        self.assertLess(
            constant["16"]["history_plus_physical_ego_compensation"]["velocity_mae"],
            constant["4"]["history_plus_physical_ego_compensation"]["velocity_mae"],
        )
        self.assertLess(
            constant["16"]["history_plus_physical_ego_compensation"]["velocity_mae"],
            constant["16"]["raw_lidar_history"]["velocity_mae"],
        )
        self.assertGreater(
            abrupt["16"]["history_plus_physical_ego_compensation"]["velocity_mae"],
            abrupt["4"]["history_plus_physical_ego_compensation"]["velocity_mae"],
        )

    def test_set_membership_history_contracts_current_state_bounds(self) -> None:
        times = np.arange(16, dtype=np.float64) * 0.1
        positions = np.stack((2.0 + times, 3.0 - 0.5 * times, np.ones(16)), axis=1)
        short = jerk_bounded_set_membership(
            times[-4:],
            positions[-4:],
            sensor_error=0.02,
            velocity_bound=5.0,
            acceleration_bound=3.0,
            jerk_bound=0.2,
        )
        long = jerk_bounded_set_membership(
            times,
            positions,
            sensor_error=0.02,
            velocity_bound=5.0,
            acceleration_bound=3.0,
            jerk_bound=0.2,
        )
        self.assertTrue(short.feasible)
        self.assertTrue(long.feasible)
        self.assertTrue(np.all(long.radius_position <= short.radius_position + 1e-9))
        self.assertTrue(np.all(long.radius_velocity <= short.radius_velocity + 1e-9))
        self.assertTrue(np.all(long.radius_acceleration <= short.radius_acceleration + 1e-9))

    def test_adaptive_history_resets_after_piecewise_velocity_change(self) -> None:
        times = np.arange(-15, 1, dtype=np.float64) * 0.1
        positions_x = np.where(times <= -0.2, times - 0.6, 4.0 * times)
        positions = np.stack((positions_x, np.zeros_like(times), np.zeros_like(times)), axis=1)
        result = adaptive_history_set_membership(
            times,
            positions,
            sensor_error=0.005,
            velocity_bound=10.0,
            acceleration_bound=6.0,
            history_jerk_bound=0.2,
            robust_jerk_bound=100.0,
            minimum_confident_window=4,
        )
        self.assertEqual(result.mode, HistoryTubeMode.ROBUST_BASE)
        self.assertEqual(result.bounds.window_length, 2)
        self.assertFalse(result.requires_history_motion_bound_assumption)
        self.assertTrue(result.certified_under_declared_robust_model)
        np.testing.assert_allclose(result.jerk_bound_used, 100.0)

    def test_infeasible_result_is_not_robustly_certified(self) -> None:
        result = adaptive_history_set_membership(
            np.array([0.0, 1.0]),
            np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]]),
            sensor_error=0.0,
            velocity_bound=1.0,
            acceleration_bound=1.0,
            history_jerk_bound=0.0,
            robust_jerk_bound=1.0,
            candidate_windows=(2,),
            minimum_confident_window=4,
        )
        self.assertEqual(result.mode, HistoryTubeMode.INFEASIBLE)
        self.assertFalse(result.bounds.feasible)
        self.assertFalse(result.certified_under_declared_robust_model)

    def test_feasible_fast_history_set_need_not_contain_true_state(self) -> None:
        times = np.arange(4, dtype=np.float64)
        measurements = np.zeros((4, 3), dtype=np.float64)
        true_position = np.zeros(3)
        true_velocity = np.array([-0.25, 0.0, 0.0])
        true_acceleration = np.array([-5.0 / 6.0, 0.0, 0.0])
        result = adaptive_history_set_membership(
            times,
            measurements,
            sensor_error=0.0,
            velocity_bound=1.0,
            acceleration_bound=1.0,
            history_jerk_bound=0.0,
            robust_jerk_bound=1.0,
            candidate_windows=(4,),
            minimum_confident_window=4,
        )

        self.assertEqual(result.mode, HistoryTubeMode.HISTORY_INFORMED)
        self.assertTrue(result.bounds.feasible)
        self.assertTrue(result.requires_history_motion_bound_assumption)
        current_state_contained = (
            np.all(true_position >= result.bounds.position[:, 0])
            and np.all(true_position <= result.bounds.position[:, 1])
            and np.all(true_velocity >= result.bounds.velocity[:, 0])
            and np.all(true_velocity <= result.bounds.velocity[:, 1])
            and np.all(true_acceleration >= result.bounds.acceleration[:, 0])
            and np.all(true_acceleration <= result.bounds.acceleration[:, 1])
        )
        self.assertFalse(current_state_contained)
        with self.assertRaisesRegex(ValueError, "true present state"):
            future_position_error_tube(
                np.array([0.1]),
                result.bounds,
                future_jerk_bound=1.0,
                true_state_containment_verified=False,
            )

    def test_feedback_interval_tube_contains_jerk_bounded_motion(self) -> None:
        times = np.arange(16, dtype=np.float64) * 0.1
        jerk = np.array([0.1, -0.05, 0.02])
        initial_position = np.array([1.0, 2.0, 3.0])
        initial_velocity = np.array([0.5, -0.3, 0.2])
        initial_acceleration = np.array([0.2, 0.1, -0.05])
        positions = (
            initial_position[None, :]
            + times[:, None] * initial_velocity[None, :]
            + 0.5 * times[:, None] ** 2 * initial_acceleration[None, :]
            + (times[:, None] ** 3 / 6.0) * jerk[None, :]
        )
        bounds = jerk_bounded_set_membership(
            times,
            positions,
            sensor_error=0.01,
            velocity_bound=5.0,
            acceleration_bound=3.0,
            jerk_bound=0.2,
        )
        horizons = np.array([0.05, 0.1, 0.2])
        component_radius, _ = future_position_error_tube(
            horizons,
            bounds,
            future_jerk_bound=0.2,
            true_state_containment_verified=True,
        )
        current_time = times[-1]
        current_position = positions[-1]
        current_velocity = initial_velocity + initial_acceleration * current_time + 0.5 * jerk * current_time**2
        current_acceleration = initial_acceleration + jerk * current_time
        true_future = (
            current_position[None, :]
            + horizons[:, None] * current_velocity[None, :]
            + 0.5 * horizons[:, None] ** 2 * current_acceleration[None, :]
            + horizons[:, None] ** 3 / 6.0 * jerk[None, :]
        )
        center_future = (
            bounds.center_position[None, :]
            + horizons[:, None] * bounds.center_velocity[None, :]
            + 0.5 * horizons[:, None] ** 2 * bounds.center_acceleration[None, :]
        )
        self.assertTrue(np.all(np.abs(true_future - center_future) <= component_radius + 1e-8))

    def test_identical_history_cannot_shrink_unannounced_future_jerk(self) -> None:
        horizons = np.array([0.1, 0.2, 1.0])
        jerk_bound = np.array([3.0, 2.0, 1.0])
        lower_bound = bounded_jerk_future_tube_lower_bound(
            horizons, jerk_bound
        )
        common_position = np.array([1.0, -2.0, 0.5])
        common_velocity = np.array([0.2, -0.3, 0.1])
        common_acceleration = np.array([0.1, 0.0, -0.1])
        center_dynamics = (
            common_position[None, :]
            + horizons[:, None] * common_velocity[None, :]
            + 0.5 * horizons[:, None] ** 2 * common_acceleration[None, :]
        )
        positive_future = center_dynamics + lower_bound
        negative_future = center_dynamics - lower_bound
        arbitrary_center = np.array([[0.01, -0.02, 0.03]])
        worst_error = np.maximum(
            np.abs(positive_future - (center_dynamics + arbitrary_center)),
            np.abs(negative_future - (center_dynamics + arbitrary_center)),
        )
        self.assertTrue(np.all(worst_error >= lower_bound - 1e-12))


class PhysicsAndCertificateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = PhysicsRolloutConfig(
            policy_dt=0.2,
            physics_dt=0.05,
            horizontal_speed_limit=20.0,
            vertical_speed_limit=5.0,
            horizontal_acceleration_limit=5.0,
            vertical_acceleration_limit=3.0,
        )

    def test_scalar_and_batch_rollouts_are_equivalent(self) -> None:
        rng = np.random.default_rng(7)
        sequence = rng.normal(size=(10, 3))
        scalar = rollout_single_sequence(np.array([1.0, 2.0, 3.0]), np.array([0.5, -0.2, 0.1]), sequence, self.config)
        batch = rollout_action_sequences(
            np.array([1.0, 2.0, 3.0]),
            np.array([0.5, -0.2, 0.1]),
            np.stack((sequence, sequence)),
            self.config,
        )
        np.testing.assert_allclose(scalar.positions[0], batch.positions[0])
        np.testing.assert_allclose(batch.positions[0], batch.positions[1])
        np.testing.assert_allclose(scalar.step_energy[0], batch.step_energy[0])

    def test_zoh_position_formula_without_clipping(self) -> None:
        sequence = np.array([[2.0, 0.0, 0.0]])
        rollout = rollout_single_sequence(np.zeros(3), np.array([1.0, 0.0, 0.0]), sequence, self.config)
        expected = 1.0 * 0.2 + 0.5 * 2.0 * 0.2**2
        self.assertAlmostEqual(rollout.positions[0, -1, 0], expected)
        self.assertAlmostEqual(rollout.velocities[0, -1, 0], 1.4)

    def test_speed_saturation_uses_realized_acceleration(self) -> None:
        sequence = np.array([[5.0, 0.0, 0.0]])
        rollout = rollout_single_sequence(np.zeros(3), np.array([20.0, 0.0, 0.0]), sequence, self.config)
        np.testing.assert_allclose(rollout.substep_realized_accelerations, 0.0, atol=1e-12)
        self.assertAlmostEqual(rollout.velocities[0, -1, 0], 20.0)

    def test_boundary_projection_does_not_change_propulsion_acceleration(self) -> None:
        config = PhysicsRolloutConfig(
            policy_dt=0.2,
            physics_dt=0.05,
            world_min=np.zeros(3),
            world_max=np.full(3, 10.0),
            horizontal_speed_limit=20.0,
            vertical_speed_limit=5.0,
            horizontal_acceleration_limit=5.0,
            vertical_acceleration_limit=3.0,
        )
        rollout = rollout_single_sequence(
            np.array([9.99, 5.0, 5.0]), np.array([2.0, 0.0, 0.0]), np.zeros((1, 3)), config
        )
        self.assertTrue(rollout.boundary_contacts[0, 0])
        np.testing.assert_allclose(rollout.substep_realized_accelerations[0, 0, 0], 0.0)

    def test_exact_intersample_certificate_catches_crossing(self) -> None:
        config = PhysicsRolloutConfig(
            policy_dt=1.0,
            physics_dt=1.0,
            horizontal_speed_limit=10.0,
            vertical_speed_limit=10.0,
            horizontal_acceleration_limit=5.0,
            vertical_acceleration_limit=5.0,
        )
        rollout = rollout_single_sequence(
            np.array([-2.0, 0.0, 0.0]), np.array([4.0, 0.0, 0.0]), np.zeros((1, 3)), config
        )
        self.assertGreater(np.linalg.norm(rollout.positions[0, 0]), 1.0)
        self.assertGreater(np.linalg.norm(rollout.positions[0, 1]), 1.0)
        certificate = certify_trajectory(
            rollout,
            [SphericalObstacle(np.zeros(3), radius=0.5)],
            config,
            uav_radius=0.5,
        )
        self.assertFalse(certificate.certified[0])
        self.assertGreater(certificate.intersample_violations[0], 0)

    def test_continuous_boundary_certificate_catches_internal_extremum(self) -> None:
        config = PhysicsRolloutConfig(
            policy_dt=1.0,
            physics_dt=1.0,
            world_min=np.zeros(3),
            world_max=np.full(3, 10.0),
            horizontal_speed_limit=10.0,
            vertical_speed_limit=10.0,
            horizontal_acceleration_limit=5.0,
            vertical_acceleration_limit=5.0,
        )
        rollout = rollout_single_sequence(
            np.array([9.9, 5.0, 5.0]),
            np.array([2.0, 0.0, 0.0]),
            np.array([[-4.0, 0.0, 0.0]]),
            config,
        )
        self.assertFalse(rollout.boundary_contacts[0, 0])
        certificate = certify_trajectory(
            rollout, [], config, uav_radius=0.0
        )
        self.assertFalse(certificate.boundary_safe[0])
        self.assertFalse(certificate.certified[0])

    def test_numpy_torch_cpu_and_optional_gpu_consistency(self) -> None:
        rng = np.random.default_rng(10)
        sequences = rng.normal(size=(8, 5, 3))
        numpy_result = rollout_action_sequences(np.zeros(3), np.zeros(3), sequences, self.config)
        torch_result = rollout_action_sequences_torch(
            np.zeros(3), np.zeros(3), sequences, self.config, device="cpu"
        )
        np.testing.assert_allclose(numpy_result.positions, torch_result.positions, atol=1e-10)
        np.testing.assert_allclose(numpy_result.step_energy, torch_result.step_energy, atol=1e-10)
        if torch.cuda.is_available():
            gpu_result = rollout_action_sequences_torch(
                np.zeros(3), np.zeros(3), sequences, self.config, device="cuda"
            )
            np.testing.assert_allclose(numpy_result.positions, gpu_result.positions, atol=1e-9)
            np.testing.assert_allclose(numpy_result.step_energy, gpu_result.step_energy, atol=1e-9)


class ScreeningEnergyAndSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = PhysicsRolloutConfig()

    def test_coarse_screen_does_not_mutate_inputs(self) -> None:
        positions = np.array(
            [
                [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]],
                [[5.0, 0.0, 0.0], [6.0, 0.0, 0.0]],
            ]
        )
        original = positions.copy()
        result = coarse_safety_screen(
            positions,
            np.array([[0.0, 0.0, 0.0]]),
            np.array([1.0]),
            keep_count=1,
            mode=CoarseScreenMode.CERTIFIED_REJECT,
            position_error_bound=0.1,
        )
        np.testing.assert_array_equal(positions, original)
        self.assertTrue(result.certified_rejection[0])
        self.assertEqual(result.survivor_indices.tolist(), [1])

    def test_clf_progress(self) -> None:
        positions = np.array(
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
            ]
        )
        velocities = np.zeros_like(positions)
        result = trajectory_progress(positions, velocities, np.array([3.0, 0.0, 0.0]))
        self.assertTrue(result.admissible[0])
        self.assertFalse(result.admissible[1])

    def test_trajectory_energy_sum_and_terminal(self) -> None:
        cost = TelemetryCostModel(
            TelemetryCostConfig(
                base_power=1.0,
                velocity_coefficients=np.zeros(3),
                acceleration_coefficients=np.zeros(3),
                compute_power=0.0,
                communication_power=0.0,
            )
        )
        rollout = rollout_action_sequences(
            np.zeros(3), np.zeros(3), np.zeros((2, 5, 3)), self.config, telemetry_cost_model=cost
        )

        def terminal(position: np.ndarray, velocity: np.ndarray, goal: np.ndarray) -> TerminalEnergyEstimate:
            del velocity
            point = np.linalg.norm(position - goal, axis=1)
            return TerminalEnergyEstimate(point=point, upper=point + 2.0)

        result = trajectory_energy(rollout, np.array([1.0, 0.0, 0.0]), terminal)
        np.testing.assert_allclose(result.rollout_energy, 1.0)
        np.testing.assert_allclose(result.total_upper, result.rollout_energy + result.terminal_point + 2.0)

    def test_candidate_ranking_and_fallback(self) -> None:
        actions = np.zeros((3, 5, 3))
        actions[:, :, 0] = np.arange(3)[:, None]
        evaluation = CandidateEvaluation(
            action_sequences=actions,
            certified=np.array([True, True, False]),
            progress_admissible=np.array([True, True, True]),
            progress_decrease=np.array([2.0, 1.0, 5.0]),
            energy_upper=np.array([3.0, 2.0, 1.0]),
            minimum_clearance=np.array([1.0, 1.0, 1.0]),
        )
        selected = select_safe_trajectory(evaluation, np.zeros((5, 3)))
        self.assertEqual(selected.selected_index, 1)
        self.assertFalse(selected.used_fallback)
        fallback_eval = CandidateEvaluation(
            action_sequences=actions,
            certified=np.array([False, False, False]),
            progress_admissible=np.ones(3, dtype=bool),
            progress_decrease=np.ones(3),
            energy_upper=np.ones(3),
            minimum_clearance=np.ones(3),
        )
        fallback = select_safe_trajectory(fallback_eval, np.ones((5, 3)))
        self.assertTrue(fallback.used_fallback)
        np.testing.assert_array_equal(fallback.selected_sequence, 1.0)

    def test_sampling_reproducibility_and_limits(self) -> None:
        config = ProposalConfig(horizon=5, count=100)
        kwargs = dict(
            position=np.zeros(3),
            velocity=np.array([1.0, 0.0, 0.0]),
            goal=np.array([10.0, 0.0, 0.0]),
            nominal_action=np.array([1.0, 0.0, 0.0]),
            config=config,
        )
        first = generate_trajectory_proposals(**kwargs, rng=np.random.default_rng(42))
        second = generate_trajectory_proposals(**kwargs, rng=np.random.default_rng(42))
        np.testing.assert_array_equal(first.action_sequences, second.action_sequences)
        self.assertLessEqual(
            np.max(np.linalg.norm(first.action_sequences[..., :2], axis=2)),
            config.horizontal_acceleration_limit + 1e-12,
        )


if __name__ == "__main__":
    unittest.main()
