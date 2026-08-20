from __future__ import annotations

import numpy as np
import torch

from experiments.memory_safety.closed_loop import (
    ClosedLoopConfig,
    METHODS,
    NUMERICAL_CERTIFICATION_RESERVE,
    SCENARIO_OBSTACLE_ACCELERATION_LIMIT,
    _apply_velocity_limits,
    _advance_obstacle_state,
    _calibrated_state_radii,
    _finite_horizon_reset_velocity_bound,
    _nominal_reference,
    _obstacle_displacement,
    _jerk,
    _history_split,
    _model_prediction,
    _perception_draw,
    _probabilistic_state_update,
    _projection_is_certifiable,
    _robust_filter,
    align_scenarios_to_nominal_path,
    make_scenarios,
)
from review_bundle.safety.collision.hocbf import ProjectionResult
from review_bundle.safety.memory import IntervalObserverState
from review_bundle.envs.navigation.telemetry_cost import TelemetryCostModel


def test_closed_loop_budget_and_bounded_jerk() -> None:
    config = ClosedLoopConfig(scenarios=4, max_steps=100)
    assert config.scenarios * len(METHODS) * config.max_steps <= 100_000
    for scenario in make_scenarios(config):
        acceleration = scenario.obstacle_acceleration.copy()
        for step in range(config.max_steps):
            jerk = _jerk(scenario, step, acceleration)
            assert np.all(np.abs(jerk) <= config.obstacle_jerk_bound + 1e-10)
            acceleration = np.clip(acceleration + jerk * 0.05, -6.0, 6.0)


def test_underdeclared_scenario_jerk_bound_is_rejected() -> None:
    with np.testing.assert_raises_regex(ValueError, "obstacle_jerk_bound"):
        ClosedLoopConfig(obstacle_jerk_bound=11.999)


def test_underdeclared_scenario_acceleration_bound_is_rejected() -> None:
    with np.testing.assert_raises_regex(ValueError, "obstacle_acceleration_bound"):
        ClosedLoopConfig(
            obstacle_acceleration_bound=(
                SCENARIO_OBSTACLE_ACCELERATION_LIMIT - 1e-3
            )
        )


def test_finite_horizon_reset_velocity_certificate_covers_all_scenarios() -> None:
    config = ClosedLoopConfig(scenarios=12, max_steps=800)
    for scenario in make_scenarios(config):
        bound = _finite_horizon_reset_velocity_bound(scenario, config)
        position = scenario.obstacle_position.copy()
        velocity = scenario.obstacle_velocity.copy()
        acceleration = scenario.obstacle_acceleration.copy()
        assert np.all(np.abs(velocity) <= bound)
        for step in range(config.max_steps):
            jerk = _jerk(scenario, step, acceleration)
            position, velocity, acceleration, _ = _advance_obstacle_state(
                position,
                velocity,
                acceleration,
                jerk,
                0.05,
            )
            assert np.all(np.abs(velocity) <= bound + 1e-10)
            assert np.all(
                np.abs(acceleration)
                <= config.obstacle_acceleration_bound + 1e-10
            )


def test_obstacle_state_uses_constant_realized_jerk_integration() -> None:
    position = np.array([1.0, 2.0, 3.0])
    velocity = np.array([4.0, -1.0, 0.5])
    acceleration = np.array([0.2, 0.4, -0.1])
    jerk = np.array([1.0, -2.0, 0.5])
    dt = 0.05
    next_position, next_velocity, next_acceleration, realized_jerk = (
        _advance_obstacle_state(position, velocity, acceleration, jerk, dt)
    )
    np.testing.assert_allclose(realized_jerk, jerk)
    np.testing.assert_allclose(next_acceleration, acceleration + jerk * dt)
    np.testing.assert_allclose(
        next_velocity,
        velocity + acceleration * dt + 0.5 * jerk * dt**2,
    )
    np.testing.assert_allclose(
        next_position,
        position + velocity * dt + 0.5 * acceleration * dt**2 + jerk * dt**3 / 6.0,
    )


def test_postsolve_certificate_rejects_solver_tolerance_violation() -> None:
    rows = np.eye(3)
    bounds = np.zeros(3)
    result = ProjectionResult(
        acceleration=np.array([-5e-7, 1.0, 1.0]),
        feasible=True,
        converged=True,
        iterations=1,
        max_violation=5e-7,
        intervention_norm=0.0,
        solver_time_seconds=0.0,
        active_constraints=1,
        reason="solver_tolerance",
    )
    certifiable, minimum_slack = _projection_is_certifiable(result, rows, bounds)
    assert minimum_slack == -5e-7
    assert not certifiable
    positive = ProjectionResult(
        acceleration=np.full(3, NUMERICAL_CERTIFICATION_RESERVE),
        feasible=True,
        converged=True,
        iterations=1,
        max_violation=0.0,
        intervention_norm=0.0,
        solver_time_seconds=0.0,
        active_constraints=1,
        reason="converged",
    )
    certifiable, _ = _projection_is_certifiable(positive, rows, bounds)
    assert certifiable


def test_velocity_saturation_exposes_actuation_model_mismatch() -> None:
    velocity = np.array([20.0, 0.0, 0.0])
    commanded = np.array([5.0, 0.0, 0.0])
    next_velocity, realized, matches = _apply_velocity_limits(
        velocity,
        commanded,
        0.05,
    )
    np.testing.assert_allclose(next_velocity, velocity)
    np.testing.assert_allclose(realized, np.zeros(3))
    assert not matches


def test_unsaturated_actuation_matches_certificate_model() -> None:
    velocity = np.array([1.0, -2.0, 0.5])
    commanded = np.array([3.0, 1.0, -0.25])
    next_velocity, realized, matches = _apply_velocity_limits(
        velocity,
        commanded,
        0.05,
    )
    np.testing.assert_allclose(next_velocity, velocity + commanded * 0.05)
    np.testing.assert_allclose(realized, commanded)
    assert matches


def test_state_radius_calibration_uses_development_validation_not_test(tmp_path) -> None:
    count = 200
    target_velocity = np.zeros((count, 1, 3), dtype=np.float32)
    target_acceleration = np.zeros((count, 1, 3), dtype=np.float32)
    np.savez(
        tmp_path / "validation.npz",
        obstacle_velocity=target_velocity,
        obstacle_acceleration=target_acceleration,
    )
    sources = {
        "A_current_only",
        "CA_Kalman",
        "IMM",
        "C_ego_L16_MLP",
        "E_GRU",
        "I_Physics_GRU",
        "M_Contractive_Physics_Memory",
    }
    predictions = {
        source: np.concatenate(
            (
                np.full((count // 2, 6), 0.2, dtype=np.float32),
                np.full((count // 2, 6), 0.4, dtype=np.float32),
            )
        )
        for source in sources
    }
    np.savez(tmp_path / "validation_predictions.npz", **predictions)
    radii = _calibrated_state_radii(tmp_path)
    for velocity_radius, acceleration_radius in radii.values():
        np.testing.assert_allclose(velocity_radius, 0.4, rtol=1e-6)
        np.testing.assert_allclose(acceleration_radius, 0.4, rtol=1e-6)


def test_scenarios_are_matched_and_dynamic() -> None:
    config = ClosedLoopConfig(scenarios=8, max_steps=100)
    scenarios = make_scenarios(config)
    assert len(scenarios) == 8
    assert len({item.identifier for item in scenarios}) == 8
    assert len({item.family for item in scenarios}) == 4
    assert all(np.linalg.norm(item.obstacle_velocity) > 0.0 for item in scenarios)


def test_obstacle_is_aligned_to_nominal_reference_crossing() -> None:
    class ZeroPolicy:
        def predict(self, observation, deterministic):
            del observation, deterministic
            return np.zeros(3), None

    config = ClosedLoopConfig(scenarios=1, max_steps=100)
    scenario = make_scenarios(config)[0]
    aligned = align_scenarios_to_nominal_path(ZeroPolicy(), [scenario], 100)[0]
    reference = _nominal_reference(ZeroPolicy(), scenario, 100)
    crossing_position = (
        aligned.obstacle_position
        + _obstacle_displacement(aligned, aligned.crossing_step)
    )
    np.testing.assert_allclose(
        crossing_position,
        reference[aligned.crossing_step],
        atol=1e-10,
    )


def test_perception_stream_is_method_independent() -> None:
    config = ClosedLoopConfig(scenarios=1, max_steps=10)
    scenario = make_scenarios(config)[0]
    first = [_perception_draw(config, scenario, index) for index in range(10)]
    second = [_perception_draw(config, scenario, index) for index in range(10)]
    for (first_noise, first_dropout), (second_noise, second_dropout) in zip(
        first, second
    ):
        np.testing.assert_array_equal(first_noise, second_noise)
        assert first_dropout == second_dropout


def test_dropout_update_expands_without_truth_access() -> None:
    config = ClosedLoopConfig(scenarios=1, max_steps=10)
    previous = IntervalObserverState(
        position=np.array([1.0, 2.0, 3.0]),
        velocity=np.array([2.0, 0.0, 0.0]),
        acceleration=np.array([0.5, 0.0, 0.0]),
        position_radius=np.full(3, 0.1),
        velocity_radius=np.full(3, 0.2),
        acceleration_radius=np.full(3, 0.3),
        track_id="probabilistic-baseline",
        certification_valid=False,
        invalid_reason="empirical_max_residual_box",
    )
    updated = _probabilistic_state_update(
        previous,
        None,
        np.array([2.1, 0.0, 0.0]),
        np.array([0.4, 0.0, 0.0]),
        np.full(3, 0.15),
        np.full(3, 0.25),
        config,
        dt=0.05,
    )
    dt = 0.05
    np.testing.assert_allclose(
        updated.position,
        previous.position
        + previous.velocity * dt
        + 0.5 * previous.acceleration * dt**2,
    )
    assert np.all(updated.position_radius > previous.position_radius)
    assert np.all(updated.velocity_radius >= previous.velocity_radius)
    assert np.all(updated.acceleration_radius >= previous.acceleration_radius)
    assert updated.missed_frames == 1
    assert not updated.certification_valid


def test_probabilistic_baseline_is_never_labeled_deterministically_certified() -> None:
    config = ClosedLoopConfig(scenarios=1, max_steps=10)
    previous = IntervalObserverState(
        position=np.zeros(3),
        velocity=np.zeros(3),
        acceleration=np.zeros(3),
        position_radius=np.full(3, 0.1),
        velocity_radius=np.full(3, 0.2),
        acceleration_radius=np.full(3, 0.3),
        track_id="probabilistic-baseline",
        certification_valid=False,
        invalid_reason="empirical_max_residual_box",
    )
    updated = _probabilistic_state_update(
        previous,
        np.array([1.0, 0.0, 0.0]),
        np.zeros(3),
        np.zeros(3),
        np.full(3, 0.2),
        np.full(3, 0.3),
        config,
        dt=0.05,
    )
    assert not updated.certification_valid
    assert updated.invalid_reason == "empirical_max_residual_box"


def test_invalid_observer_state_uses_fail_closed_braking_fallback() -> None:
    invalid = IntervalObserverState(
        position=np.array([10.0, 0.0, 0.0]),
        velocity=np.zeros(3),
        acceleration=np.zeros(3),
        position_radius=np.ones(3),
        velocity_radius=np.ones(3),
        acceleration_radius=np.ones(3),
        track_id="ambiguous-track",
        certification_valid=False,
        invalid_reason="association_uncertain",
    )
    (
        action,
        feasible,
        certified,
        intervention,
        safety_intervention,
        _,
        tube_seconds,
        qp_seconds,
    ) = (
        _robust_filter(
            np.zeros(3),
            np.array([3.0, 0.0, 0.0]),
            np.zeros(3),
            invalid,
            1.0,
            12.0,
            TelemetryCostModel(),
            allow_uncertified_interval=False,
        )
    )
    assert action[0] < 0.0
    assert not feasible
    assert not certified
    assert intervention > 0.0
    assert safety_intervention
    assert tube_seconds == 0.0
    assert qp_seconds == 0.0


def test_history_warm_start_is_finite_and_masked() -> None:
    split = _history_split(
        [np.array([1.0, 2.0, 3.0])],
        [np.array([4.0, 5.0, 6.0])],
        [True],
        16,
    )
    assert np.all(np.isfinite(split["relative_measurement"]))
    assert int(np.sum(split["valid"])) == 1
    np.testing.assert_array_equal(split["relative_measurement"][0, 0], [1.0, 2.0, 3.0])


def test_history_window_carries_valid_anchor_across_leading_dropout() -> None:
    relative = [np.array([float(index), 0.0, 0.0]) for index in range(20)]
    ego = [np.zeros(3) for _ in range(20)]
    valid = [True] * 4 + [False] * 16
    for index in range(4, 20):
        relative[index] = np.full(3, np.nan)
    split = _history_split(relative, ego, valid, 16)
    assert np.all(np.isfinite(split["relative_measurement"]))
    np.testing.assert_array_equal(
        split["relative_measurement"][0, 0],
        np.array([3.0, 0.0, 0.0]),
    )
    assert not np.any(split["valid"])


def test_estimator_holds_previous_state_when_window_has_no_measurement() -> None:
    split = {
        "relative_measurement": np.zeros((1, 16, 3), dtype=np.float32),
        "ego_position": np.zeros((1, 16, 3), dtype=np.float32),
        "valid": np.zeros((1, 16), dtype=bool),
    }
    velocity, acceleration = _model_prediction(
        "D_IMM",
        split,
        {},
        torch.device("cpu"),
        np.array([1.0, 2.0, 3.0]),
        np.array([4.0, 5.0, 6.0]),
    )
    np.testing.assert_array_equal(velocity, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(acceleration, [4.0, 5.0, 6.0])
