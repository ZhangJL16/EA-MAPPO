from __future__ import annotations

import numpy as np

from review_bundle.safety.collision.hocbf import project_polyhedral_qp
from review_bundle.safety.collision.projection_geometry import (
    ProjectionProblem,
    analyze_projection_geometry,
    normalized_to_physical_action_jacobian,
    stable_pseudo_inverse,
    weighted_projection_jacobian,
)


def _problem(
    center: np.ndarray,
    rows: np.ndarray,
    bounds: np.ndarray,
    *,
    barrier_count: int | None = None,
) -> ProjectionProblem:
    metric = np.eye(3, dtype=np.float64)
    projection = project_polyhedral_qp(center, metric, rows, bounds)
    assert projection.feasible
    assert projection.converged
    return ProjectionProblem(
        center=center,
        hessian=metric,
        rows=rows,
        lower_bounds=bounds,
        projected_acceleration=projection.acceleration,
        barrier_constraint_count=(rows.shape[0] if barrier_count is None else barrier_count),
        feasible=projection.feasible,
        converged=projection.converged,
        solver_reason=projection.reason,
    )


def _geometry(problem: ProjectionProblem, nominal: np.ndarray):
    return analyze_projection_geometry(
        problem,
        nominal_normalized_action=nominal,
        executed_normalized_action=problem.projected_acceleration,
        horizontal_acceleration_limit=1.0,
        vertical_acceleration_limit=1.0,
    )


def test_no_active_constraints_has_identity_jacobian() -> None:
    center = np.array([0.2, -0.1, 0.3])
    rows = np.array([[1.0, 0.0, 0.0]])
    bounds = np.array([-2.0])
    geometry = _geometry(_problem(center, rows, bounds), center)
    assert geometry.valid
    assert geometry.active_constraint_indices == ()
    np.testing.assert_allclose(geometry.jacobian_total, np.eye(3), atol=1e-10)
    assert geometry.rank == 3
    assert geometry.action_authority == 1.0


def test_one_active_halfspace_matches_analytic_projector() -> None:
    center = np.array([-0.5, 0.2, 0.3])
    normal = np.array([1.0, 0.0, 0.0])
    problem = _problem(center, normal[None, :], np.array([0.0]))
    geometry = _geometry(problem, center)
    expected = np.eye(3) - np.outer(normal, normal) / (normal @ normal)
    assert geometry.valid
    np.testing.assert_allclose(geometry.jacobian_total, expected, atol=1e-8)
    assert geometry.rank == 2
    assert geometry.active_barrier_constraints == 1


def test_two_independent_active_constraints_reduce_rank_twice() -> None:
    center = np.array([-0.5, -0.25, 0.2])
    rows = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    problem = _problem(center, rows, np.zeros(2))
    geometry = _geometry(problem, center)
    assert geometry.valid
    np.testing.assert_allclose(
        geometry.jacobian_total,
        np.diag([0.0, 0.0, 1.0]),
        atol=1e-8,
    )
    assert geometry.rank == 1
    assert geometry.active_barrier_constraints == 2


def test_redundant_active_rows_use_stable_pseudo_inverse() -> None:
    center = np.array([-0.5, 0.2, 0.1])
    rows = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    problem = _problem(center, rows, np.zeros(2))
    geometry = _geometry(problem, center)
    expected = np.diag([0.0, 1.0, 1.0])
    assert geometry.valid
    np.testing.assert_allclose(geometry.jacobian_total, expected, atol=1e-8)
    gram = rows @ rows.T
    np.testing.assert_allclose(
        gram @ stable_pseudo_inverse(gram) @ gram,
        gram,
        atol=1e-10,
    )


def test_fixed_active_set_finite_difference_matches_jacobian() -> None:
    center = np.array([-0.5, 0.2, 0.1])
    rows = np.array([[1.0, 0.0, 0.0]])
    bounds = np.array([0.0])
    problem = _problem(center, rows, bounds)
    geometry = _geometry(problem, center)
    perturbation = np.array([2e-5, -3e-5, 4e-5])
    perturbed = project_polyhedral_qp(
        center + perturbation,
        np.eye(3),
        rows,
        bounds,
    ).acceleration
    observed = perturbed - problem.projected_acceleration
    predicted = geometry.jacobian_total @ perturbation
    absolute_error = float(np.linalg.norm(observed - predicted))
    relative_error = absolute_error / max(float(np.linalg.norm(observed)), 1e-12)
    cosine = float(
        observed @ predicted
        / max(float(np.linalg.norm(observed) * np.linalg.norm(predicted)), 1e-12)
    )
    assert absolute_error < 1e-9
    assert relative_error < 1e-5
    assert cosine > 0.999999


def test_active_set_switching_boundary_is_masked() -> None:
    center = np.zeros(3)
    rows = np.array([[1.0, 0.0, 0.0]])
    problem = _problem(center, rows, np.zeros(1))
    geometry = _geometry(problem, center)
    assert not geometry.valid
    assert not geometry.active_set_stable
    assert geometry.reason == "inactive_constraint_near_switch"


def test_normalized_physical_coordinate_transform_matches_diagonal_formula() -> None:
    scales = np.diag([5.0, 5.0, 3.0])
    normalized = np.array([-0.2, 0.1, 0.15])
    center = scales @ normalized
    normal = np.array([1.0, 2.0, 0.5])
    lower = float(normal @ center + 0.4)
    problem = _problem(center, normal[None, :], np.array([lower]))
    executed_normalized = np.linalg.inv(scales) @ problem.projected_acceleration
    geometry = analyze_projection_geometry(
        problem,
        nominal_normalized_action=normalized,
        executed_normalized_action=executed_normalized,
        horizontal_acceleration_limit=5.0,
        vertical_acceleration_limit=3.0,
    )
    expected_physical = weighted_projection_jacobian(normal[None, :], np.eye(3))
    expected_normalized = np.linalg.inv(scales) @ expected_physical @ scales
    assert geometry.valid
    np.testing.assert_allclose(
        geometry.jacobian_physical_total,
        expected_physical,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        geometry.jacobian_total,
        expected_normalized,
        atol=1e-8,
    )


def test_horizontal_radial_coordinate_switch_is_explicitly_unstable() -> None:
    _, stable, reason = normalized_to_physical_action_jacobian(
        np.array([1.0, 0.0, 0.0]),
        horizontal_acceleration_limit=5.0,
        vertical_acceleration_limit=3.0,
    )
    assert not stable
    assert reason in {
        "normalized_box_switching_boundary",
        "horizontal_radial_switching_boundary",
    }
