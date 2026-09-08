#!/usr/bin/env python3
"""Numerically verify the finite-state EIRR identities on explicit examples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def risk_resolvent(matrix: np.ndarray, terminal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(matrix, dtype=np.float64)
    terminal = np.asarray(terminal, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")
    if terminal.shape != (matrix.shape[0],):
        raise ValueError("terminal vector shape mismatch")
    if np.any(matrix < 0.0) or np.any(terminal < 0.0):
        raise ValueError("Feynman--Kac entries must be nonnegative")
    radius = float(np.max(np.abs(np.linalg.eigvals(matrix))))
    if not np.isfinite(radius) or radius >= 1.0:
        raise ValueError("spectral radius must be strictly below one")
    resolvent = np.linalg.inv(np.eye(matrix.shape[0]) - matrix)
    return resolvent, resolvent @ terminal


def doob_discount_reduction(matrix: np.ndarray, gamma: float) -> dict[str, object]:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")
    if np.any(matrix < 0.0):
        raise ValueError("matrix entries must be nonnegative")
    radius = float(np.max(np.abs(np.linalg.eigvals(matrix))))
    if not radius < gamma < 1.0:
        raise ValueError("gamma must lie strictly between the spectral radius and one")
    identity = np.eye(matrix.shape[0])
    ones = np.ones(matrix.shape[0], dtype=np.float64)
    scaling = np.linalg.solve(identity - matrix / gamma, ones)
    transient = matrix * scaling[np.newaxis, :] / (gamma * scaling[:, np.newaxis])
    cemetery = 1.0 / scaling
    diagonal = np.diag(scaling)
    inverse_diagonal = np.diag(1.0 / scaling)
    reconstructed_matrix = gamma * diagonal @ transient @ inverse_diagonal
    original_resolvent = np.linalg.inv(identity - matrix)
    reduced_resolvent = diagonal @ np.linalg.inv(identity - gamma * transient) @ inverse_diagonal
    return {
        "gamma": float(gamma),
        "scaling": scaling.tolist(),
        "transient_kernel": transient.tolist(),
        "cemetery_probability": cemetery.tolist(),
        "row_sum_error": float(np.max(np.abs(np.sum(transient, axis=1) + cemetery - 1.0))),
        "matrix_reconstruction_error": float(
            np.max(np.abs(reconstructed_matrix - matrix))
        ),
        "resolvent_reconstruction_error": float(
            np.max(np.abs(reduced_resolvent - original_resolvent))
        ),
    }


def perturbation_identity_error(
    matrix: np.ndarray,
    terminal: np.ndarray,
    learned_matrix: np.ndarray,
    learned_terminal: np.ndarray,
) -> float:
    resolvent, value = risk_resolvent(matrix, terminal)
    _, learned_value = risk_resolvent(learned_matrix, learned_terminal)
    right = resolvent @ (
        terminal
        - learned_terminal
        + (matrix - learned_matrix) @ learned_value
    )
    return float(np.max(np.abs((value - learned_value) - right)))


def residual_identity_error(
    matrix: np.ndarray,
    terminal: np.ndarray,
    initial: np.ndarray,
    critic: np.ndarray,
) -> float:
    resolvent, value = risk_resolvent(matrix, terminal)
    initial = np.asarray(initial, dtype=np.float64)
    critic = np.asarray(critic, dtype=np.float64)
    occupation = initial @ resolvent
    bellman_residual = terminal + matrix @ critic - critic
    left = float(initial @ (value - critic))
    right = float(occupation @ bellman_residual)
    return abs(left - right)


def no_support_target_mgf(
    *,
    branch_probability: float,
    risk_lambda: float,
    terminal_cost: float,
) -> float:
    if not 0.0 < branch_probability <= 1.0:
        raise ValueError("branch probability must lie in (0, 1]")
    return float(
        1.0
        - branch_probability
        + branch_probability * np.exp(risk_lambda * terminal_cost)
    )


def perfect_support_risk_ratio(
    *,
    high_prefix_probability: float,
    risk_lambda: float,
    prefix_cost: float,
) -> float:
    if not 0.0 < high_prefix_probability < 1.0:
        raise ValueError("high-prefix probability must lie in (0, 1)")
    return float(
        1.0
        - high_prefix_probability
        + high_prefix_probability * np.exp(risk_lambda * prefix_cost)
    )


def common_support_sample_lower_bound(
    *,
    risk_lambda: float,
    high_cost: float,
) -> dict[str, float]:
    q = float(np.exp(-risk_lambda * high_cost))
    if q > 0.25:
        raise ValueError("the explicit lower bound requires exp(-lambda L) <= 1/4")
    theta_0 = 2.0 - q
    theta_1 = 3.0 - 2.0 * q
    return {
        "rare_probability_model_0": q,
        "rare_probability_model_1": 2.0 * q,
        "maximum_trajectory_count": float(np.floor(1.0 / (4.0 * q))),
        "mgf_separation": theta_1 - theta_0,
        "mgf_absolute_loss_lower_bound": 9.0 / 32.0,
        "log_mgf_separation": float(np.log(theta_1 / theta_0)),
        "log_mgf_absolute_loss_lower_bound": float(3.0 / 8.0 * np.log(10.0 / 7.0)),
    }


def clipped_certificate_penalty(
    *,
    sample_count: int,
    residual_bound: float,
    ratio_second_moment: float,
    ratio_l1_error: float,
    alpha: float,
    clipping_threshold: float | None = None,
) -> dict[str, float]:
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if residual_bound < 0.0 or ratio_second_moment < 0.0 or ratio_l1_error < 0.0:
        raise ValueError("bounds must be nonnegative")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    log_term = float(np.log(2.0 / alpha))
    optimal_threshold = float(
        np.sqrt(3.0 * ratio_second_moment * sample_count / (2.0 * log_term))
    )
    threshold = optimal_threshold if clipping_threshold is None else clipping_threshold
    if threshold <= 0.0:
        raise ValueError("clipping threshold must be positive")
    nuisance = residual_bound * ratio_l1_error
    clipping_bias = residual_bound * ratio_second_moment / threshold
    variance = residual_bound * np.sqrt(
        2.0 * ratio_second_moment * log_term / sample_count
    )
    range_term = 2.0 * residual_bound * threshold * log_term / (3.0 * sample_count)
    return {
        "optimal_clipping_threshold": optimal_threshold,
        "used_clipping_threshold": float(threshold),
        "nuisance_penalty": float(nuisance),
        "clipping_bias_penalty": float(clipping_bias),
        "variance_penalty": float(variance),
        "range_penalty": float(range_term),
        "total_nonempirical_penalty": float(
            nuisance + clipping_bias + variance + range_term
        ),
    }


def exact_chernoff_grid_requirement(
    values: np.ndarray,
    probabilities: np.ndarray,
    lambdas: np.ndarray,
    delta: float,
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    lambdas = np.asarray(lambdas, dtype=np.float64)
    if values.ndim != 1 or probabilities.shape != values.shape:
        raise ValueError("values and probabilities must be aligned vectors")
    if not np.isclose(np.sum(probabilities), 1.0) or np.any(probabilities < 0.0):
        raise ValueError("probabilities must be nonnegative and sum to one")
    if lambdas.ndim != 1 or lambdas.size == 0 or np.any(lambdas <= 0.0):
        raise ValueError("lambdas must be a positive nonempty vector")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie in (0, 1)")
    log_mgfs = np.asarray(
        [np.log(np.sum(probabilities * np.exp(risk_lambda * values))) for risk_lambda in lambdas]
    )
    candidates = (log_mgfs + np.log(1.0 / delta)) / lambdas
    index = int(np.argmin(candidates))
    requirement = float(candidates[index])
    exact_tail = float(np.sum(probabilities[values > requirement]))
    return requirement, float(lambdas[index]), exact_tail


def build_verification_report() -> dict[str, object]:
    matrix = np.asarray([[0.20, 0.10], [0.05, 0.25]], dtype=np.float64)
    terminal = np.asarray([1.15, 1.05], dtype=np.float64)
    learned_matrix = matrix + np.asarray([[0.01, -0.005], [0.0, -0.01]])
    learned_terminal = terminal + np.asarray([0.02, -0.01])
    initial = np.asarray([0.7, 0.3])
    critic = np.asarray([1.4, 1.2])
    resolvent, value = risk_resolvent(matrix, terminal)
    doob_reduction = doob_discount_reduction(matrix, gamma=0.8)

    p = 0.05
    risk_lambda = 0.5
    cost_0 = 0.0
    cost_1 = 8.0
    theta_0 = no_support_target_mgf(
        branch_probability=p,
        risk_lambda=risk_lambda,
        terminal_cost=cost_0,
    )
    theta_1 = no_support_target_mgf(
        branch_probability=p,
        risk_lambda=risk_lambda,
        terminal_cost=cost_1,
    )
    ratio_small = perfect_support_risk_ratio(
        high_prefix_probability=p,
        risk_lambda=risk_lambda,
        prefix_cost=2.0,
    )
    ratio_large = perfect_support_risk_ratio(
        high_prefix_probability=p,
        risk_lambda=risk_lambda,
        prefix_cost=12.0,
    )
    requirement, selected_lambda, exact_tail = exact_chernoff_grid_requirement(
        np.asarray([0.0, 3.0, 8.0]),
        np.asarray([0.80, 0.15, 0.05]),
        np.asarray([0.1, 0.25, 0.5, 1.0]),
        0.10,
    )
    return {
        "status": "FINITE_STATE_IDENTITIES_VERIFIED",
        "spectral_radius": float(np.max(np.abs(np.linalg.eigvals(matrix)))),
        "resolvent": resolvent.tolist(),
        "value": value.tolist(),
        "doob_discount_reduction": doob_reduction,
        "perturbation_identity_max_error": perturbation_identity_error(
            matrix,
            terminal,
            learned_matrix,
            learned_terminal,
        ),
        "residual_identity_error": residual_identity_error(
            matrix,
            terminal,
            initial,
            critic,
        ),
        "no_support_example": {
            "branch_probability": p,
            "lambda": risk_lambda,
            "cost_0": cost_0,
            "cost_1": cost_1,
            "theta_0": theta_0,
            "theta_1": theta_1,
            "absolute_separation": abs(theta_1 - theta_0),
            "absolute_loss_minimax_lower_bound": abs(theta_1 - theta_0) / 2.0,
            "log_mgf_minimax_lower_bound": abs(np.log(theta_1) - np.log(theta_0)) / 2.0,
        },
        "perfect_support_concentration_example": {
            "ordinary_occupation_ratio": 1.0,
            "risk_ratio_prefix_cost_2": ratio_small,
            "risk_ratio_prefix_cost_12": ratio_large,
        },
        "common_support_sample_complexity_example": common_support_sample_lower_bound(
            risk_lambda=risk_lambda,
            high_cost=12.0,
        ),
        "clipped_certificate_example": clipped_certificate_penalty(
            sample_count=1000,
            residual_bound=1.0,
            ratio_second_moment=4.0,
            ratio_l1_error=0.02,
            alpha=0.05,
        ),
        "chernoff_grid_example": {
            "delta": 0.10,
            "selected_lambda": selected_lambda,
            "requirement": requirement,
            "exact_tail_probability": exact_tail,
            "bound_satisfied": exact_tail <= 0.10,
        },
        "non_claim": (
            "Numerical verification checks algebra on explicit finite examples; "
            "it is not a proof of neural estimation consistency or a formal "
            "Oracle/Pareto result."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_verification_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
