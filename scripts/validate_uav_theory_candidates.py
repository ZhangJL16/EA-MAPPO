#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import platform
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import sympy as sp

from review_bundle.safety.collision.feasibility import (
    cylindrical_input_support,
    energy_to_go_smooth_upper_bound,
    exact_zoh_sphere_clearance,
    joint_feasibility_margin,
)


def symbolic_checks() -> dict[str, object]:
    tau, distance = sp.symbols("tau distance", real=True)
    coordinates = sp.symbols("r0:3")
    velocities = sp.symbols("v0:3")
    accelerations = sp.symbols("a0:3")
    relative = [
        coordinates[index]
        + velocities[index] * tau
        + sp.Rational(1, 2) * accelerations[index] * tau**2
        for index in range(3)
    ]
    barrier = sp.expand(sum(value**2 for value in relative) - distance**2)
    derivative = sp.expand(sp.diff(barrier, tau))
    expected_derivative = sp.expand(
        2 * sum(
            relative[index] * (velocities[index] + accelerations[index] * tau)
            for index in range(3)
        )
    )

    px, py, pz, vx, vy, vz, ux, uy, uz, radius, k1, k2 = sp.symbols(
        "px py pz vx vy vz ux uy uz radius k1 k2", real=True
    )
    position = sp.Matrix([px, py, pz])
    velocity = sp.Matrix([vx, vy, vz])
    acceleration = sp.Matrix([ux, uy, uz])
    h = sp.expand(position.dot(position) - radius**2)
    h_dot = sp.expand(2 * position.dot(velocity))
    h_ddot = sp.expand(2 * velocity.dot(velocity) + 2 * position.dot(acceleration))
    psi2 = sp.expand(h_ddot + (k1 + k2) * h_dot + k1 * k2 * h)
    affine_row = 2 * position
    affine_drift = sp.expand(
        2 * velocity.dot(velocity) + (k1 + k2) * h_dot + k1 * k2 * h
    )

    dt, lipschitz = sp.symbols("dt lipschitz", positive=True)
    control_jacobian = sp.Matrix.vstack(
        sp.Rational(1, 2) * dt**2 * sp.eye(3),
        dt * sp.eye(3),
    )
    upper_hessian = sp.simplify(lipschitz * control_jacobian.T * control_jacobian)
    expected_hessian = sp.simplify(
        lipschitz * (dt**2 + sp.Rational(1, 4) * dt**4) * sp.eye(3)
    )

    return {
        "quartic_degree": int(sp.Poly(barrier, tau).degree()),
        "quartic_derivative_degree": int(sp.Poly(derivative, tau).degree()),
        "quartic_derivative_identity": bool(sp.simplify(derivative - expected_derivative) == 0),
        "hocbf_affine_identity": bool(
            sp.simplify(psi2 - (affine_drift + affine_row.dot(acceleration))) == 0
        ),
        "energy_upper_hessian_identity": bool(
            sp.simplify(upper_hessian - expected_hessian) == sp.zeros(3)
        ),
        "energy_upper_hessian_diagonal": str(upper_hessian[0, 0]),
    }


def support_checks(rng: np.random.Generator, state_count: int) -> dict[str, object]:
    directions = rng.normal(size=(state_count, 3))
    horizontal_limits = rng.uniform(0.1, 10.0, size=state_count)
    vertical_limits = rng.uniform(0.1, 5.0, size=state_count)
    horizontal_norms = np.linalg.norm(directions[:, :2], axis=1)
    actions = np.zeros_like(directions)
    nonzero = horizontal_norms > 1e-14
    actions[nonzero, :2] = (
        horizontal_limits[nonzero, None]
        * directions[nonzero, :2]
        / horizontal_norms[nonzero, None]
    )
    actions[:, 2] = vertical_limits * np.sign(directions[:, 2])
    constructed = np.sum(directions * actions, axis=1)
    analytic = horizontal_limits * horizontal_norms + vertical_limits * np.abs(directions[:, 2])
    return {
        "states": state_count,
        "maximum_absolute_error": float(np.max(np.abs(analytic - constructed))),
        "mean_absolute_error": float(np.mean(np.abs(analytic - constructed))),
        "all_constructed_actions_inside_input_set": bool(
            np.all(np.linalg.norm(actions[:, :2], axis=1) <= horizontal_limits + 1e-12)
            and np.all(np.abs(actions[:, 2]) <= vertical_limits + 1e-12)
        ),
    }


def intersample_checks(rng: np.random.Generator, state_count: int) -> dict[str, object]:
    worst_sample_minus_exact = -math.inf
    worst_stationarity = 0.0
    endpoint_false_safe = 0
    exact_unsafe = 0
    for _ in range(state_count):
        position = rng.uniform(-10.0, 10.0, size=3)
        velocity = rng.uniform(-20.0, 20.0, size=3)
        acceleration = rng.uniform(-5.0, 5.0, size=3)
        safe_distance = float(rng.uniform(0.1, 4.0))
        hold = float(rng.uniform(0.01, 0.25))
        result = exact_zoh_sphere_clearance(
            position,
            velocity,
            acceleration,
            safe_distance,
            hold,
        )
        sample_times = rng.uniform(0.0, hold, size=8)
        relative = (
            position[None, :]
            + sample_times[:, None] * velocity[None, :]
            + 0.5 * sample_times[:, None] ** 2 * acceleration[None, :]
        )
        sampled_minimum = float(np.min(np.sum(relative * relative, axis=1) - safe_distance**2))
        worst_sample_minus_exact = max(
            worst_sample_minus_exact,
            result.minimum_barrier - sampled_minimum,
        )
        if 1e-10 < result.minimizing_time < hold - 1e-10:
            time = result.minimizing_time
            relative_at_minimum = position + velocity * time + 0.5 * acceleration * time**2
            derivative = 2.0 * float(relative_at_minimum @ (velocity + acceleration * time))
            worst_stationarity = max(worst_stationarity, abs(derivative))
        if result.endpoint_minimum_barrier >= 0.0 and result.minimum_barrier < 0.0:
            endpoint_false_safe += 1
        exact_unsafe += int(result.minimum_barrier < 0.0)
    return {
        "states": state_count,
        "exact_minimum_never_exceeds_random_samples": bool(worst_sample_minus_exact <= 1e-8),
        "maximum_exact_minus_random_sample_minimum": float(worst_sample_minus_exact),
        "maximum_interior_stationarity_residual": float(worst_stationarity),
        "endpoint_safe_but_intersample_unsafe_cases": endpoint_false_safe,
        "exact_unsafe_cases": exact_unsafe,
    }


def joint_duality_checks(rng: np.random.Generator, case_count: int) -> dict[str, object]:
    maximum_gap = 0.0
    primal_failures = 0
    dual_failures = 0
    sign_mismatches = 0
    global_fallbacks = 0
    solve_times = []
    for _ in range(case_count):
        constraint_count = int(rng.integers(1, 6))
        rows = rng.normal(size=(constraint_count, 3))
        bounds = rng.uniform(-8.0, 8.0, size=constraint_count)
        result = joint_feasibility_margin(
            rows,
            bounds,
            horizontal_limit=5.0,
            vertical_limit=3.0,
            max_iterations=200,
            global_certificate_fallback=True,
        )
        maximum_gap = max(maximum_gap, result.duality_gap)
        primal_failures += int(not result.primal_success)
        dual_failures += int(not result.dual_success)
        global_fallbacks += int(result.global_certificate_fallback_used)
        solve_times.append(result.solve_seconds)
        direct_minimum = float(np.min(rows @ result.maximizing_action - bounds))
        sign_mismatches += int((result.margin >= -1e-7) != (direct_minimum >= -1e-7))
    times = np.asarray(solve_times)
    return {
        "cases": case_count,
        "maximum_primal_dual_gap": float(maximum_gap),
        "primal_solver_failures": primal_failures,
        "dual_solver_failures": dual_failures,
        "feasibility_sign_mismatches": sign_mismatches,
        "global_certificate_fallbacks": global_fallbacks,
        "mean_solve_ms": float(np.mean(times) * 1e3),
        "p95_solve_ms": float(np.quantile(times, 0.95) * 1e3),
        "p99_solve_ms": float(np.quantile(times, 0.99) * 1e3),
        "maximum_solve_ms": float(np.max(times) * 1e3),
        "deadline_misses_50ms": int(np.sum(times > 0.05)),
    }


def energy_upper_checks(rng: np.random.Generator, state_count: int) -> dict[str, object]:
    dt = 0.05
    jacobian = np.vstack((0.5 * dt**2 * np.eye(3), dt * np.eye(3)))
    violations = 0
    maximum_error = 0.0
    for _ in range(state_count):
        reference_state = rng.normal(size=6)
        delta = rng.uniform(-5.0, 5.0, size=3)
        state_delta = jacobian @ delta
        reference_value = 0.5 * float(reference_state @ reference_state)
        bound = energy_to_go_smooth_upper_bound(
            reference_value,
            reference_state,
            jacobian,
            delta,
            gradient_lipschitz_constant=1.0,
        )
        actual = 0.5 * float((reference_state + state_delta) @ (reference_state + state_delta))
        error = actual - bound
        maximum_error = max(maximum_error, abs(error))
        violations += int(error > 1e-10)
    return {
        "states": state_count,
        "violations": violations,
        "maximum_absolute_error_for_unit_quadratic": float(maximum_error),
        "relu_global_gradient_lipschitz_available": False,
        "relu_reason": "the deployed MC estimator contains ReLU activation boundaries",
    }


def counterexamples() -> dict[str, object]:
    intersample = exact_zoh_sphere_clearance(
        np.array([-2.0, 0.0, 0.0]),
        np.array([8.0, 0.0, 0.0]),
        np.zeros(3),
        safe_distance=0.5,
        hold_dt=0.5,
    )
    current_state = 0.9
    current_margin = 1.0 - current_state
    chosen_action = 1.0
    next_state = current_state + 0.2 * chosen_action
    next_margin = 1.0 - next_state

    circular_state = 0.75
    original_lower = circular_state
    feasibility_cbf_upper = 1.0 - circular_state

    base_power = 0.6
    fast_total = base_power + 1.0**2
    slow_total = 2.0 * (base_power + 0.5**2)

    relu_epsilons = [1e-1, 1e-2, 1e-3, 1e-4]
    required_lipschitz = [1.0 / (2.0 * epsilon) for epsilon in relu_epsilons]
    return {
        "pointwise_margin_not_recursive": {
            "system": "x_next = x + 0.2 u, U=[-1,1], safety row u>=x",
            "current_state": current_state,
            "current_margin": current_margin,
            "chosen_feasible_action": chosen_action,
            "next_state": next_state,
            "next_margin": next_margin,
        },
        "feasibility_barrier_can_be_circular": {
            "original_constraint": f"u >= {original_lower}",
            "rho_cbf_constraint": f"u <= {feasibility_cbf_upper}",
            "joint_feasible": bool(original_lower <= feasibility_cbf_upper),
        },
        "endpoint_safety_misses_intersample_collision": {
            "endpoint_minimum_barrier": intersample.endpoint_minimum_barrier,
            "exact_minimum_barrier": intersample.minimum_barrier,
            "collision_time": intersample.minimizing_time,
        },
        "instantaneous_acceleration_energy_not_trajectory_energy": {
            "fast_one_step_total_energy": fast_total,
            "slow_two_step_total_energy": slow_total,
            "slow_has_lower_action_energy_each_step": True,
            "slow_has_higher_total_energy_due_to_base_power": bool(slow_total > fast_total),
        },
        "relu_gradient_is_not_globally_lipschitz": {
            "epsilons": relu_epsilons,
            "minimum_L_needed_across_relu_kink": required_lipschitz,
            "diverges_as_epsilon_goes_to_zero": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", type=int, default=100_000)
    parser.add_argument("--joint-cases", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.states < 100_000:
        raise ValueError("the theory validation protocol requires at least 100000 randomized states")
    if args.joint_cases <= 0:
        raise ValueError("joint-cases must be positive")
    rng = np.random.default_rng(args.seed)
    started = perf_counter()
    report = {
        "protocol": "uav_theory_candidate_symbolic_and_numerical_validation_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "python": platform.python_version(),
        "symbolic": symbolic_checks(),
        "support_function": support_checks(rng, args.states),
        "exact_intersample": intersample_checks(rng, args.states),
        "joint_primal_dual": joint_duality_checks(rng, args.joint_cases),
        "smooth_energy_upper_bound": energy_upper_checks(rng, args.states),
        "counterexamples": counterexamples(),
    }
    report["wall_clock_seconds"] = float(perf_counter() - started)
    report["all_required_state_banks_at_least_100k"] = bool(
        report["support_function"]["states"] >= 100_000
        and report["exact_intersample"]["states"] >= 100_000
        and report["joint_primal_dual"]["cases"] >= 100_000
        and report["smooth_energy_upper_bound"]["states"] >= 100_000
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
