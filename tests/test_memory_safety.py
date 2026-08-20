from __future__ import annotations

import itertools

import numpy as np
import torch

from review_bundle.safety.collision.feasibility import (
    exact_constant_jerk_sphere_clearance,
    exact_zoh_sphere_clearance,
)
from review_bundle.safety.collision.hocbf import (
    HOCBFConfig,
    project_single_halfspace,
    strengthen_constraint_for_sample_hold,
)
from review_bundle.safety.memory import (
    ContractiveResidualMemory,
    IntervalObserverConfig,
    IntervalObserverState,
    PhysicsMemoryObserver,
    TrustedBaseIntervalCertificate,
    box_support,
    directional_hocbf_preconditions_hold,
    directional_hocbf_sample_hold_margin,
    directional_interval_hocbf_constraint,
    future_position_box,
    isotropic_enclosing_radius,
    position_box_contains,
)


def _base_certificate(
    config: IntervalObserverConfig,
    track_id: str = "track-a",
    measurement_epoch: int = 0,
) -> TrustedBaseIntervalCertificate:
    return TrustedBaseIntervalCertificate.from_config(
        config,
        track_id=track_id,
        provenance="deterministic unit-test bounds",
        measurement_epoch=measurement_epoch,
    )


def test_interval_observer_contains_bounded_motion_and_dropout() -> None:
    config = IntervalObserverConfig(
        dt=0.1,
        sensor_error_bound=np.array([0.02, 0.03, 0.01]),
        jerk_residual_bound=np.array([0.5, 0.4, 0.3]),
    )
    observer = PhysicsMemoryObserver(config)
    position = np.array([10.0, -4.0, 2.0])
    velocity = np.array([1.0, -0.5, 0.2])
    acceleration = np.array([0.1, 0.2, -0.05])
    noise = np.array([0.01, -0.02, 0.005])
    observer.initialize(
        position + noise,
        trusted_base_certificate=_base_certificate(config),
    )
    previous_radius = observer.state.position_radius.copy()
    for step in range(1, 21):
        jerk = np.array([0.3, -0.2, 0.1])
        dt = config.dt
        position = position + velocity * dt + 0.5 * acceleration * dt**2 + jerk * dt**3 / 6.0
        velocity = velocity + acceleration * dt + jerk * dt**2 / 2.0
        acceleration = acceleration + jerk * dt
        measurement = None if step in {7, 8, 9} else position + noise
        result = observer.step(
            measurement,
            track_id="track-a" if measurement is not None else None,
        )
        assert position_box_contains(
            position,
            result.state.position,
            result.state.position_radius,
        )
        assert np.all(
            np.abs(velocity - result.state.velocity)
            <= result.state.velocity_radius + 1e-10
        )
        assert np.all(
            np.abs(acceleration - result.state.acceleration)
            <= result.state.acceleration_radius + 1e-10
        )
        if measurement is None:
            assert np.all(result.state.position_radius >= previous_radius)
        previous_radius = result.state.position_radius.copy()


def test_innovation_reset_uses_declared_base_set() -> None:
    observer = PhysicsMemoryObserver(
        IntervalObserverConfig(
            sensor_error_bound=0.05,
            reset_velocity_bound=15.0,
            reset_acceleration_bound=10.0,
        )
    )
    observer.initialize(
        np.zeros(3),
        trusted_base_certificate=_base_certificate(observer.config),
    )
    result = observer.step(
        np.array([20.0, 0.0, 0.0]),
        track_id="track-a",
        trusted_base_certificate=_base_certificate(
            observer.config,
            measurement_epoch=1,
        ),
    )
    assert result.reset
    assert np.allclose(result.state.position, [20.0, 0.0, 0.0])
    assert np.allclose(result.state.position_radius, 0.05)
    assert np.allclose(result.state.velocity_radius, 15.0)
    assert np.allclose(result.state.acceleration_radius, 10.0)
    assert result.state.certification_valid


def test_uncertain_association_invalidates_certificate_and_hocbf_fails_closed() -> None:
    observer = PhysicsMemoryObserver(
        IntervalObserverConfig(sensor_error_bound=0.05)
    )
    observer.initialize(
        np.zeros(3),
        trusted_base_certificate=_base_certificate(observer.config),
    )
    result = observer.step(
        np.array([100.0, 0.0, 0.0]),
        track_id="track-b",
        association_uncertain=True,
    )
    assert result.reset
    assert not result.state.certification_valid
    assert result.state.invalid_reason == "association_uncertain"
    with np.testing.assert_raises_regex(ValueError, "invalid observer state"):
        directional_interval_hocbf_constraint(
            np.zeros(3),
            np.zeros(3),
            result.state,
            1.0,
            HOCBFConfig(),
        )


def test_same_track_measurement_can_reestablish_base_certificate() -> None:
    observer = PhysicsMemoryObserver(
        IntervalObserverConfig(sensor_error_bound=0.05)
    )
    observer.initialize(
        np.zeros(3),
        trusted_base_certificate=_base_certificate(observer.config),
    )
    invalid = observer.step(
        np.array([100.0, 0.0, 0.0]),
        track_id="track-b",
        association_uncertain=True,
    )
    assert not invalid.state.certification_valid
    recovered = observer.step(
        np.array([1.0, 0.0, 0.0]),
        track_id="track-a",
        trusted_base_certificate=_base_certificate(
            observer.config,
            measurement_epoch=1,
        ),
        association_uncertain=False,
    )
    assert recovered.reset
    assert recovered.state.certification_valid
    np.testing.assert_allclose(recovered.state.position, [1.0, 0.0, 0.0])


def test_track_identity_mismatch_fails_closed_without_new_base_certificate() -> None:
    config = IntervalObserverConfig(sensor_error_bound=0.05)
    observer = PhysicsMemoryObserver(config)
    observer.initialize(
        np.zeros(3),
        trusted_base_certificate=_base_certificate(config, "track-a"),
    )
    swapped = observer.step(
        np.array([20.0, 0.0, 0.0]),
        track_id="track-b",
    )
    assert swapped.reset
    assert not swapped.state.certification_valid
    assert swapped.state.invalid_reason == "track_identity_mismatch"


def test_contradictory_innovation_requires_a_fresh_reset_certificate() -> None:
    config = IntervalObserverConfig(sensor_error_bound=0.05)
    observer = PhysicsMemoryObserver(config)
    observer.initialize(
        np.zeros(3),
        trusted_base_certificate=_base_certificate(config, measurement_epoch=0),
    )
    contradicted = observer.step(
        np.array([60.0, 0.0, 0.0]),
        track_id="track-a",
    )
    assert contradicted.reset
    assert not contradicted.state.certification_valid
    assert contradicted.state.invalid_reason == "fresh_reset_certificate_required"


def test_invalid_reset_cannot_erase_epoch_and_replay_stale_certificate() -> None:
    config = IntervalObserverConfig(sensor_error_bound=0.05)
    observer = PhysicsMemoryObserver(config)
    observer.initialize(
        np.zeros(3),
        trusted_base_certificate=_base_certificate(config, measurement_epoch=10),
    )
    contradicted = observer.step(
        np.array([60.0, 0.0, 0.0]),
        track_id="track-a",
    )
    assert not contradicted.state.certification_valid
    assert contradicted.state.certificate_epoch == 10
    stale = observer.step(
        np.array([61.0, 0.0, 0.0]),
        track_id="track-a",
        trusted_base_certificate=_base_certificate(config, measurement_epoch=0),
    )
    assert not stale.state.certification_valid
    assert stale.state.certificate_epoch == 10
    fresh = observer.step(
        np.array([62.0, 0.0, 0.0]),
        track_id="track-a",
        trusted_base_certificate=_base_certificate(config, measurement_epoch=11),
    )
    assert fresh.state.certification_valid
    assert fresh.state.certificate_epoch == 11


def test_positive_or_nonfinite_innovation_tolerance_is_rejected() -> None:
    for value in (1e-12, np.inf, np.nan):
        with np.testing.assert_raises(ValueError):
            IntervalObserverConfig(innovation_tolerance=value)


def test_prediction_only_hold_advances_interval_to_the_control_timestamp() -> None:
    config = IntervalObserverConfig(
        dt=0.1,
        sensor_error_bound=0.0,
        jerk_residual_bound=0.0,
        reset_velocity_bound=12.0,
        reset_acceleration_bound=0.0,
    )
    observer = PhysicsMemoryObserver(config)
    observer.initialize(
        np.array([100.0, 0.0, 0.0]),
        trusted_base_certificate=_base_certificate(config),
    )
    propagated = observer.step(None)
    true_position = np.array([98.8, 0.0, 0.0])
    true_velocity = np.array([-12.0, 0.0, 0.0])
    assert position_box_contains(
        true_position,
        propagated.state.position,
        propagated.state.position_radius,
    )
    assert np.all(
        np.abs(true_velocity - propagated.state.velocity)
        <= propagated.state.velocity_radius
    )


def test_contractive_memory_hidden_state_bound() -> None:
    torch.manual_seed(4)
    memory = ContractiveResidualMemory(
        input_size=6,
        hidden_size=12,
        recurrent_norm_bound=0.6,
        leak=0.4,
    )
    features = torch.randn(64, 6)
    first = torch.randn(64, 12)
    second = torch.randn(64, 12)
    _, next_first = memory(features, first)
    _, next_second = memory(features, second)
    ratio = torch.linalg.vector_norm(next_first - next_second, dim=1) / torch.clamp(
        torch.linalg.vector_norm(first - second, dim=1), min=1e-12
    )
    assert float(torch.max(ratio)) <= memory.contraction_factor + 1e-5
    assert memory.contraction_factor < 1.0


def test_float64_residual_memory_observer_uses_parameter_dtype() -> None:
    memory = ContractiveResidualMemory(input_size=4, hidden_size=8).double()
    config = IntervalObserverConfig(sensor_error_bound=0.05)
    observer = PhysicsMemoryObserver(config, memory)
    observer.initialize(
        np.zeros(3),
        trusted_base_certificate=_base_certificate(config),
    )
    result = observer.step(
        np.zeros(3),
        features=np.zeros(4, dtype=np.float64),
        track_id="track-a",
    )
    assert result.nominal_jerk.dtype == np.float64
    assert np.all(np.isfinite(result.nominal_jerk))


def test_contraction_is_enforced_after_optimizer_can_change_raw_weight() -> None:
    torch.manual_seed(8)
    memory = ContractiveResidualMemory(
        input_size=4,
        hidden_size=8,
        recurrent_norm_bound=0.5,
        leak=0.7,
    )
    with torch.no_grad():
        memory.recurrent.weight.mul_(50.0)
    assert torch.linalg.matrix_norm(memory.recurrent.weight, ord=2) > 1.0
    assert torch.linalg.matrix_norm(memory.effective_recurrent_weight(), ord=2) <= 0.50001
    features = torch.randn(128, 4)
    first = torch.randn(128, 8)
    second = torch.randn(128, 8)
    _, next_first = memory(features, first)
    _, next_second = memory(features, second)
    ratio = torch.linalg.vector_norm(next_first - next_second, dim=1) / torch.clamp(
        torch.linalg.vector_norm(first - second, dim=1), min=1e-12
    )
    assert float(torch.max(ratio)) <= memory.contraction_factor + 1e-5


def test_execution_dtype_requires_a_strict_contraction_gap() -> None:
    with np.testing.assert_raises_regex(ValueError, "contraction factor"):
        ContractiveResidualMemory(
            input_size=4,
            recurrent_norm_bound=0.8,
            leak=1e-20,
        )


def test_contracting_memory_rejects_unsupported_float16_execution() -> None:
    memory = ContractiveResidualMemory(input_size=4, hidden_size=8).half()
    with np.testing.assert_raises_regex(TypeError, "float32 or float64"):
        memory(
            torch.zeros(1, 4, dtype=torch.float16),
            torch.zeros(1, 8, dtype=torch.float16),
        )


def test_constant_jerk_clearance_matches_dense_polynomial_minimum() -> None:
    position = np.array([2.0, -1.2, 0.4])
    velocity = np.array([-3.0, 1.4, -0.3])
    acceleration = np.array([1.0, -0.4, 0.2])
    jerk = np.array([2.0, 0.8, -0.5])
    hold = 0.7
    safe_distance = 0.6
    exact = exact_constant_jerk_sphere_clearance(
        position,
        velocity,
        acceleration,
        jerk,
        safe_distance,
        hold,
    )
    times = np.linspace(0.0, hold, 200_001)
    relative = (
        position[None, :]
        + times[:, None] * velocity[None, :]
        + 0.5 * times[:, None] ** 2 * acceleration[None, :]
        + times[:, None] ** 3 * jerk[None, :] / 6.0
    )
    dense = float(np.min(np.linalg.norm(relative, axis=1) - safe_distance))
    assert abs(exact.minimum_clearance - dense) < 1e-8


def test_constant_jerk_clearance_reduces_to_zoh_when_jerk_is_zero() -> None:
    arguments = (
        np.array([1.5, -0.2, 0.1]),
        np.array([-1.0, 0.3, 0.0]),
        np.array([0.4, 0.1, -0.2]),
        0.5,
        0.8,
    )
    zoh = exact_zoh_sphere_clearance(*arguments)
    jerk = exact_constant_jerk_sphere_clearance(
        arguments[0],
        arguments[1],
        arguments[2],
        np.zeros(3),
        arguments[3],
        arguments[4],
    )
    assert abs(zoh.minimum_barrier - jerk.minimum_barrier) < 1e-12
    assert abs(zoh.minimizing_time - jerk.minimizing_time) < 1e-10


def test_future_tube_grows_with_horizon_and_directional_support_is_tighter() -> None:
    radius = future_position_box(
        np.array([0.1, 0.2, 0.4]),
        np.array([0.2, 0.3, 0.5]),
        np.array([0.1, 0.1, 0.2]),
        np.array([0.5, 0.5, 1.0]),
        np.array([0.0, 0.25, 0.5, 1.0, 2.0]),
    )
    assert np.all(np.diff(radius, axis=0) >= 0.0)
    direction = np.array([1.0, 0.0, 0.0])
    assert box_support(radius[-1], direction) < isotropic_enclosing_radius(radius[-1])


def test_directional_hocbf_bounds_every_interval_corner() -> None:
    obstacle = IntervalObserverState(
        position=np.array([5.0, 0.5, 0.0]),
        velocity=np.array([-0.5, 0.2, 0.0]),
        acceleration=np.array([0.1, -0.1, 0.0]),
        position_radius=np.array([0.2, 0.5, 0.1]),
        velocity_radius=np.array([0.3, 0.4, 0.2]),
        acceleration_radius=np.array([0.5, 0.6, 0.2]),
        track_id="test-obstacle",
        certification_valid=True,
        invalid_reason=None,
        certificate_epoch=0,
    )
    position = np.zeros(3)
    velocity = np.array([1.0, 0.0, 0.0])
    config = HOCBFConfig(k1=1.3, k2=0.9, uav_radius=0.5)
    constraint = directional_interval_hocbf_constraint(
        position,
        velocity,
        obstacle,
        obstacle_radius=1.0,
        config=config,
    )
    assert directional_hocbf_preconditions_hold(constraint)
    action = project_single_halfspace(np.zeros(3), constraint.row, constraint.lower_bound)
    direction = constraint.row
    safe_distance = 1.5
    for signs in itertools.product((-1.0, 1.0), repeat=9):
        error = np.asarray(signs).reshape(3, 3)
        true_position = obstacle.position + error[0] * obstacle.position_radius
        true_velocity = obstacle.velocity + error[1] * obstacle.velocity_radius
        true_acceleration = obstacle.acceleration + error[2] * obstacle.acceleration_radius
        h = float(direction @ (position - true_position) - safe_distance)
        h_dot = float(direction @ (velocity - true_velocity))
        psi2 = float(
            direction @ (action - true_acceleration)
            + (config.k1 + config.k2) * h_dot
            + config.k1 * config.k2 * h
        )
        assert psi2 >= -1e-9


def test_directional_hocbf_rejects_negative_physical_radius() -> None:
    state = IntervalObserverState(
        position=np.array([5.0, 0.0, 0.0]),
        velocity=np.zeros(3),
        acceleration=np.zeros(3),
        position_radius=np.zeros(3),
        velocity_radius=np.zeros(3),
        acceleration_radius=np.zeros(3),
        track_id="test-obstacle",
        certification_valid=True,
        invalid_reason=None,
        certificate_epoch=0,
    )
    with np.testing.assert_raises_regex(ValueError, "nonnegative"):
        directional_interval_hocbf_constraint(
            np.zeros(3),
            np.zeros(3),
            state,
            -1.0,
            HOCBFConfig(),
        )


def test_directional_hocbf_preconditions_never_accept_negative_values() -> None:
    state = IntervalObserverState(
        position=np.array([1.5 - 5e-11, 0.0, 0.0]),
        velocity=np.zeros(3),
        acceleration=np.zeros(3),
        position_radius=np.zeros(3),
        velocity_radius=np.zeros(3),
        acceleration_radius=np.zeros(3),
        track_id="test-obstacle",
        certification_valid=True,
        invalid_reason=None,
        certificate_epoch=0,
    )
    constraint = directional_interval_hocbf_constraint(
        np.zeros(3),
        np.zeros(3),
        state,
        1.0,
        HOCBFConfig(uav_radius=0.5),
    )
    assert constraint.h < 0.0
    assert not directional_hocbf_preconditions_hold(constraint)
    with np.testing.assert_raises(ValueError):
        directional_hocbf_preconditions_hold(constraint, tolerance=-1e-10)


def test_uncertainty_inclusion_expands_safe_set_and_reduces_projection_cost() -> None:
    base = dict(
        position=np.array([6.0, 1.0, 0.0]),
        velocity=np.array([-0.2, 0.0, 0.0]),
        acceleration=np.zeros(3),
    )
    small = IntervalObserverState(
        **base,
        position_radius=np.array([0.1, 0.2, 0.1]),
        velocity_radius=np.array([0.1, 0.2, 0.1]),
        acceleration_radius=np.array([0.2, 0.3, 0.2]),
        track_id="test-obstacle",
        certification_valid=True,
        invalid_reason=None,
        certificate_epoch=0,
    )
    large = IntervalObserverState(
        **base,
        position_radius=np.array([0.5, 0.8, 0.4]),
        velocity_radius=np.array([0.4, 0.7, 0.3]),
        acceleration_radius=np.array([0.8, 1.0, 0.6]),
        track_id="test-obstacle",
        certification_valid=True,
        invalid_reason=None,
        certificate_epoch=0,
    )
    config = HOCBFConfig(k1=1.0, k2=1.0, uav_radius=0.5)
    small_constraint = directional_interval_hocbf_constraint(
        np.zeros(3), np.array([2.0, 0.0, 0.0]), small, 1.0, config
    )
    large_constraint = directional_interval_hocbf_constraint(
        np.zeros(3), np.array([2.0, 0.0, 0.0]), large, 1.0, config
    )
    assert np.allclose(small_constraint.row, large_constraint.row)
    assert small_constraint.lower_bound <= large_constraint.lower_bound
    nominal = np.zeros(3)
    projected_small = project_single_halfspace(
        nominal, small_constraint.row, small_constraint.lower_bound
    )
    projected_large = project_single_halfspace(
        nominal, large_constraint.row, large_constraint.lower_bound
    )
    assert np.linalg.norm(projected_small - nominal) <= np.linalg.norm(
        projected_large - nominal
    ) + 1e-10


def test_directional_sample_hold_margin_keeps_true_psi2_nonnegative() -> None:
    rng = np.random.default_rng(12)
    obstacle = IntervalObserverState(
        position=np.array([8.0, 1.0, 0.5]),
        velocity=np.array([-1.0, 0.5, 0.0]),
        acceleration=np.array([0.2, -0.1, 0.0]),
        position_radius=np.array([0.2, 0.3, 0.1]),
        velocity_radius=np.array([0.3, 0.2, 0.1]),
        acceleration_radius=np.array([0.4, 0.3, 0.2]),
        track_id="test-obstacle",
        certification_valid=True,
        invalid_reason=None,
        certificate_epoch=0,
    )
    uav_position = np.zeros(3)
    uav_velocity = np.array([2.0, 0.0, 0.0])
    config = HOCBFConfig(k1=1.0, k2=1.0, uav_radius=0.5)
    constraint = directional_interval_hocbf_constraint(
        uav_position,
        uav_velocity,
        obstacle,
        1.0,
        config,
    )
    hold_dt = 0.05
    jerk_bound = np.array([2.0, 2.0, 1.0])
    margin = directional_hocbf_sample_hold_margin(
        uav_velocity,
        obstacle,
        constraint.row,
        config,
        hold_dt=hold_dt,
        horizontal_acceleration_limit=5.0,
        vertical_acceleration_limit=3.0,
        true_obstacle_jerk_bound=jerk_bound,
    )
    strengthened = strengthen_constraint_for_sample_hold(constraint, margin)
    action = project_single_halfspace(
        np.zeros(3),
        strengthened.row,
        strengthened.lower_bound,
    )
    direction = constraint.row
    for _ in range(64):
        position_error = rng.uniform(-1.0, 1.0, 3) * obstacle.position_radius
        velocity_error = rng.uniform(-1.0, 1.0, 3) * obstacle.velocity_radius
        acceleration_error = rng.uniform(-1.0, 1.0, 3) * obstacle.acceleration_radius
        jerk = rng.uniform(-1.0, 1.0, 3) * jerk_bound
        true_position = obstacle.position + position_error
        true_velocity = obstacle.velocity + velocity_error
        true_acceleration = obstacle.acceleration + acceleration_error
        for time in np.linspace(0.0, hold_dt, 101):
            uav_p = uav_position + uav_velocity * time + 0.5 * action * time**2
            uav_v = uav_velocity + action * time
            obstacle_p = (
                true_position
                + true_velocity * time
                + 0.5 * true_acceleration * time**2
                + jerk * time**3 / 6.0
            )
            obstacle_v = true_velocity + true_acceleration * time + 0.5 * jerk * time**2
            obstacle_a = true_acceleration + jerk * time
            h = float(direction @ (uav_p - obstacle_p) - 1.5)
            h_dot = float(direction @ (uav_v - obstacle_v))
            psi2 = float(
                direction @ (action - obstacle_a)
                + (config.k1 + config.k2) * h_dot
                + config.k1 * config.k2 * h
            )
            assert psi2 >= -1e-9


def test_auditor_nonzero_jerk_counterexample_is_covered_by_hold_margin() -> None:
    config = HOCBFConfig(k1=1.0, k2=1.0, uav_radius=0.0)
    hold_dt = 0.05
    relative_position = np.array([23.5118, 0.0, 0.0])
    relative_velocity = np.array([-0.0602, 0.0, 0.0])
    obstacle_acceleration = np.array([5.2312, 0.0, 0.0])
    obstacle = IntervalObserverState(
        position=np.zeros(3),
        velocity=np.zeros(3),
        acceleration=obstacle_acceleration,
        position_radius=np.zeros(3),
        velocity_radius=np.zeros(3),
        acceleration_radius=np.zeros(3),
        track_id="audit-track",
        certification_valid=True,
        invalid_reason=None,
        certificate_epoch=0,
    )
    constraint = directional_interval_hocbf_constraint(
        relative_position,
        relative_velocity,
        obstacle,
        0.0,
        config,
    )
    margin = directional_hocbf_sample_hold_margin(
        relative_velocity,
        obstacle,
        constraint.row,
        config,
        hold_dt=hold_dt,
        horizontal_acceleration_limit=5.0,
        vertical_acceleration_limit=2.0,
        true_obstacle_jerk_bound=np.array([12.0, 0.0, 0.0]),
    )
    strengthened = strengthen_constraint_for_sample_hold(constraint, margin)
    action = np.array([-4.8123, 0.0, 0.0])
    assert float(strengthened.row @ action) >= strengthened.lower_bound
    jerk = np.array([12.0, 0.0, 0.0])
    for time in np.linspace(0.0, hold_dt, 101):
        position = (
            relative_position
            + relative_velocity * time
            + 0.5 * (action - obstacle_acceleration) * time**2
            - jerk * time**3 / 6.0
        )
        velocity = (
            relative_velocity
            + (action - obstacle_acceleration) * time
            - 0.5 * jerk * time**2
        )
        acceleration = obstacle_acceleration + jerk * time
        psi2 = float(
            constraint.row @ (action - acceleration)
            + (config.k1 + config.k2) * constraint.row @ velocity
            + config.k1 * config.k2 * constraint.row @ position
        )
        assert psi2 >= -1e-9
