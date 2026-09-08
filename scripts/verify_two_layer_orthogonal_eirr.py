#!/usr/bin/env python3
"""Verify the finite two-layer orthogonal EIRR population identities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _weighted_l2(vector: np.ndarray, measure: np.ndarray) -> float:
    return float(np.sqrt(np.sum(measure * np.square(vector))))


def build_two_layer_orthogonal_eirr_report() -> dict[str, object]:
    """Construct and verify a two-state, two-interface-action example."""

    rho_x = np.asarray([0.55, 0.45], dtype=np.float64)
    source_interface = np.asarray(
        [[0.70, 0.30], [0.40, 0.60]], dtype=np.float64
    )
    target_interface = np.asarray(
        [[0.25, 0.75], [0.65, 0.35]], dtype=np.float64
    )
    nu = np.asarray([0.60, 0.40], dtype=np.float64)

    # K_lambda f(q) = r(q) + M(q, :) f is the conditional mean of the
    # exponential charger-hitting witness at executed interface q=(x,u).
    primitive_reward = np.asarray(
        [[0.72, 0.88], [0.81, 0.76]], dtype=np.float64
    )
    primitive_continuation = np.asarray(
        [
            [[0.12, 0.08], [0.05, 0.18]],
            [[0.16, 0.04], [0.07, 0.11]],
        ],
        dtype=np.float64,
    )

    target_reward = np.sum(target_interface * primitive_reward, axis=1)
    target_continuation = np.einsum(
        "xu,xuy->xy", target_interface, primitive_continuation
    )
    resolvent = np.linalg.inv(np.eye(2) - target_continuation)
    psi = resolvent @ target_reward
    theta = float(nu @ psi)

    eta_x = nu @ resolvent
    rho_q = (rho_x[:, None] * source_interface).reshape(-1)
    eta_q = (eta_x[:, None] * target_interface).reshape(-1)
    exact_v = eta_x / rho_x
    exact_w = eta_q / rho_q

    def k_operator(f: np.ndarray) -> np.ndarray:
        return (
            primitive_reward
            + np.einsum("xuy,y->xu", primitive_continuation, f)
        ).reshape(-1)

    def l_operator(g: np.ndarray) -> np.ndarray:
        return np.sum(target_interface * g.reshape(2, 2), axis=1)

    def score(
        f: np.ndarray,
        g: np.ndarray,
        v: np.ndarray,
        w: np.ndarray,
    ) -> float:
        state_residual = l_operator(g) - f
        primitive_residual = k_operator(f) - g
        return float(
            nu @ f
            + np.sum(rho_x * v * state_residual)
            + np.sum(rho_q * w * primitive_residual)
        )

    candidate_f = np.asarray([0.45, 1.35], dtype=np.float64)
    candidate_g = np.asarray([1.12, 0.74, 0.93, 1.28], dtype=np.float64)
    candidate_v = exact_v + np.asarray([0.18, -0.11])
    candidate_w = exact_w + np.asarray([0.12, -0.08, 0.19, -0.14])

    exact_g = k_operator(psi)
    ratio_robust_score = score(
        candidate_f, candidate_g, exact_v, exact_w
    )
    model_robust_score = score(psi, exact_g, candidate_v, candidate_w)
    candidate_score = score(
        candidate_f, candidate_g, candidate_v, candidate_w
    )

    state_residual = l_operator(candidate_g) - candidate_f
    primitive_residual = k_operator(candidate_f) - candidate_g
    state_product = float(
        np.sum(rho_x * (candidate_v - exact_v) * state_residual)
    )
    primitive_product = float(
        np.sum(rho_q * (candidate_w - exact_w) * primitive_residual)
    )
    product_bias = state_product + primitive_product
    actual_bias = candidate_score - theta

    state_bound = _weighted_l2(candidate_v - exact_v, rho_x) * _weighted_l2(
        state_residual, rho_x
    )
    primitive_bound = _weighted_l2(
        candidate_w - exact_w, rho_q
    ) * _weighted_l2(primitive_residual, rho_q)
    l2_bound = state_bound + primitive_bound

    # An explicit symmetric two-outcome realization shows that the second
    # residual is conditionally centered when the model block is exact.
    noise_amplitude = np.asarray([0.09, 0.12, 0.07, 0.10])
    outcome_witness = np.stack(
        [exact_g - noise_amplitude, exact_g + noise_amplitude], axis=1
    )
    conditional_model_residual = np.mean(
        outcome_witness - exact_g[:, None], axis=1
    )

    spectral_radius = float(
        np.max(np.abs(np.linalg.eigvals(target_continuation)))
    )
    checks = {
        "source_and_target_interface_rows_normalized": bool(
            np.allclose(source_interface.sum(axis=1), 1.0)
            and np.allclose(target_interface.sum(axis=1), 1.0)
        ),
        "positive_source_domination": bool(
            np.all(rho_x > 0.0) and np.all(rho_q > 0.0)
        ),
        "killed_operator_is_transient": spectral_radius < 1.0,
        "fixed_point_holds": bool(
            np.allclose(
                psi,
                target_reward + target_continuation @ psi,
                atol=1e-13,
            )
        ),
        "risk_measure_balance_holds": bool(
            np.allclose(
                eta_x,
                nu + eta_x @ target_continuation,
                atol=1e-13,
            )
        ),
        "exact_ratios_are_model_invariant": bool(
            np.isclose(ratio_robust_score, theta, atol=1e-13)
        ),
        "exact_models_are_ratio_invariant": bool(
            np.isclose(model_robust_score, theta, atol=1e-13)
        ),
        "product_bias_identity_holds": bool(
            np.isclose(actual_bias, product_bias, atol=1e-13)
        ),
        "l2_product_bound_holds": abs(actual_bias) <= l2_bound + 1e-13,
        "primitive_model_residual_is_conditionally_centered": bool(
            np.allclose(conditional_model_residual, 0.0, atol=1e-13)
        ),
        "neither_block_exact_has_nonzero_bias": abs(actual_bias) > 1e-8,
    }
    status = (
        "TWO_LAYER_ORTHOGONAL_EIRR_IDENTITIES_VERIFIED"
        if all(checks.values())
        else "TWO_LAYER_ORTHOGONAL_EIRR_VERIFICATION_FAILED"
    )
    return {
        "status": status,
        "model": {
            "number_of_states": 2,
            "number_of_interfaces": 4,
            "spectral_radius": spectral_radius,
            "theta": theta,
            "psi": psi.tolist(),
            "risk_state_measure": eta_x.tolist(),
            "risk_state_ratio": exact_v.tolist(),
            "risk_interface_ratio": exact_w.tolist(),
        },
        "robustness": {
            "exact_ratio_arbitrary_model_score": ratio_robust_score,
            "exact_model_arbitrary_ratio_score": model_robust_score,
            "target": theta,
        },
        "product_bias": {
            "actual_bias": actual_bias,
            "state_layer_product": state_product,
            "primitive_layer_product": primitive_product,
            "identity_sum": product_bias,
            "l2_product_bound": l2_bound,
        },
        "checks": checks,
        "non_claim": (
            "This finite population verifier checks algebraic double robustness, "
            "the exact product-bias identity, and its L2 bound. It does not prove "
            "nuisance-learning rates, dependent-trajectory asymptotic normality, "
            "neural consistency, Oracle headroom, or Pareto improvement."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_two_layer_orthogonal_eirr_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
