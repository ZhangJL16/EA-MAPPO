#!/usr/bin/env python3
"""Verify Theorem 18's canonical-gradient identities in a finite model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def build_eirr_canonical_gradient_report() -> dict[str, object]:
    rho_x = np.asarray([0.55, 0.45], dtype=np.float64)
    source_interface = np.asarray([[0.70, 0.30], [0.40, 0.60]])
    target_interface = np.asarray([[0.25, 0.75], [0.65, 0.35]])
    rho_q = (rho_x[:, None] * source_interface).reshape(-1)
    nu = np.asarray([0.60, 0.40], dtype=np.float64)

    # Each q has three outcomes: charger terminal, nonterminal state 0, and
    # nonterminal state 1. Multipliers stand for exp(lambda * cost).
    base_probability = np.asarray(
        [
            [0.50, 0.30, 0.20],
            [0.35, 0.40, 0.25],
            [0.45, 0.15, 0.40],
            [0.55, 0.25, 0.20],
        ],
        dtype=np.float64,
    )
    multiplier = np.asarray(
        [
            [0.95, 1.08, 1.02],
            [1.04, 1.01, 1.10],
            [0.98, 1.12, 1.03],
            [1.06, 1.05, 1.09],
        ],
        dtype=np.float64,
    )

    def solve(probability: np.ndarray) -> dict[str, np.ndarray | float]:
        reward_q = probability[:, 0] * multiplier[:, 0]
        continuation_q = np.column_stack(
            [
                probability[:, 1] * multiplier[:, 1],
                probability[:, 2] * multiplier[:, 2],
            ]
        )
        reward = np.sum(
            target_interface * reward_q.reshape(2, 2), axis=1
        )
        continuation = np.einsum(
            "xu,xuy->xy",
            target_interface,
            continuation_q.reshape(2, 2, 2),
        )
        resolvent = np.linalg.inv(np.eye(2) - continuation)
        psi = resolvent @ reward
        eta_x = nu @ resolvent
        eta_q = (eta_x[:, None] * target_interface).reshape(-1)
        return {
            "theta": float(nu @ psi),
            "psi": psi,
            "eta_x": eta_x,
            "eta_q": eta_q,
            "continuation": continuation,
        }

    base = solve(base_probability)
    psi = np.asarray(base["psi"])
    eta_q = np.asarray(base["eta_q"])
    risk_ratio = eta_q / rho_q
    gamma_psi = np.column_stack(
        [
            multiplier[:, 0],
            multiplier[:, 1] * psi[0],
            multiplier[:, 2] * psi[1],
        ]
    )
    conditional_mean = np.sum(base_probability * gamma_psi, axis=1)
    phi = risk_ratio[:, None] * (
        gamma_psi - conditional_mean[:, None]
    )

    raw_score = np.asarray(
        [
            [0.40, -0.25, 0.10],
            [-0.15, 0.35, -0.20],
            [0.20, -0.30, 0.45],
            [-0.35, 0.15, 0.25],
        ]
    )
    score = raw_score - np.sum(
        base_probability * raw_score, axis=1, keepdims=True
    )

    def tilt(score_direction: np.ndarray, t: float) -> np.ndarray:
        unnormalized = base_probability * np.exp(t * score_direction)
        return unnormalized / unnormalized.sum(axis=1, keepdims=True)

    derivative_from_gradient = float(
        np.sum(rho_q[:, None] * base_probability * phi * score)
    )
    step = 1e-6
    finite_derivative = (
        float(solve(tilt(score, step))["theta"])
        - float(solve(tilt(score, -step))["theta"])
    ) / (2.0 * step)

    efficiency_variance = float(
        np.sum(rho_q[:, None] * base_probability * np.square(phi))
    )
    conditional_variance_formula = float(
        np.sum(
            rho_q
            * np.square(risk_ratio)
            * np.sum(
                base_probability
                * np.square(gamma_psi - conditional_mean[:, None]),
                axis=1,
            )
        )
    )
    least_favourable_score = phi / np.sqrt(efficiency_variance)
    least_favourable_derivative = float(
        np.sum(
            rho_q[:, None]
            * base_probability
            * phi
            * least_favourable_score
        )
    )
    finite_least_favourable_derivative = (
        float(solve(tilt(least_favourable_score, step))["theta"])
        - float(solve(tilt(least_favourable_score, -step))["theta"])
    ) / (2.0 * step)

    marginal_score = np.asarray([0.3, -0.2, 0.1, -0.4])
    marginal_score -= float(rho_q @ marginal_score)
    marginal_inner_product = float(
        np.sum(
            rho_q[:, None]
            * base_probability
            * phi
            * marginal_score[:, None]
        )
    )
    conditional_phi_mean = np.sum(base_probability * phi, axis=1)
    spectral_radius = float(
        np.max(np.abs(np.linalg.eigvals(np.asarray(base["continuation"]))))
    )
    derivative_error = abs(finite_derivative - derivative_from_gradient)
    least_favourable_error = abs(
        finite_least_favourable_derivative
        - np.sqrt(efficiency_variance)
    )
    checks = {
        "conditional_probabilities_normalized": bool(
            np.allclose(base_probability.sum(axis=1), 1.0)
        ),
        "killed_operator_is_transient": bool(spectral_radius < 1.0),
        "canonical_gradient_is_conditionally_centered": bool(
            np.allclose(conditional_phi_mean, 0.0, atol=1e-13)
        ),
        "pathwise_derivative_matches_finite_difference": bool(
            derivative_error < 1e-9
        ),
        "variance_formula_matches_squared_gradient_norm": bool(
            np.isclose(
                efficiency_variance,
                conditional_variance_formula,
                atol=1e-13,
            )
        ),
        "marginal_design_tangent_is_orthogonal": bool(
            abs(marginal_inner_product) < 1e-13
        ),
        "least_favourable_derivative_matches_sqrt_information": bool(
            least_favourable_error < 1e-9
        ),
    }
    status = (
        "EIRR_CANONICAL_GRADIENT_IDENTITIES_VERIFIED"
        if all(checks.values())
        else "EIRR_CANONICAL_GRADIENT_VERIFICATION_FAILED"
    )
    return {
        "status": status,
        "model": {
            "number_of_states": 2,
            "number_of_interfaces": 4,
            "outcomes_per_interface": 3,
            "spectral_radius": spectral_radius,
            "theta": float(base["theta"]),
            "psi": psi.tolist(),
        },
        "pathwise_derivative": {
            "canonical_gradient_inner_product": derivative_from_gradient,
            "central_finite_difference": finite_derivative,
            "absolute_error": derivative_error,
        },
        "efficiency": {
            "squared_gradient_norm": efficiency_variance,
            "conditional_variance_formula": conditional_variance_formula,
            "sqrt_efficiency_variance": float(
                np.sqrt(efficiency_variance)
            ),
            "least_favourable_finite_derivative": (
                finite_least_favourable_derivative
            ),
            "least_favourable_absolute_error": least_favourable_error,
            "marginal_design_inner_product": marginal_inner_product,
        },
        "checks": checks,
        "non_claim": (
            "This finite example checks the pathwise derivative, canonical "
            "gradient norm, and least-favourable direction. It does not prove "
            "dependent-replay inference, a quotient/transience phase transition, "
            "or oral-level novelty."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_eirr_canonical_gradient_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
