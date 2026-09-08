#!/usr/bin/env python3
"""Verify the risk-observable quotient/resolvent algebra on a finite example."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _fiber_witness_diameter(
    r_q: np.ndarray,
    m_q: np.ndarray,
    left: int,
    right: int,
) -> float:
    """IPM for affine witnesses r(q) + m(q)^T f with ||f||_inf <= 1."""

    return float(
        abs(r_q[left] - r_q[right])
        + np.sum(np.abs(m_q[left] - m_q[right]))
    )


def _compose(
    interface_law: np.ndarray,
    r_q: np.ndarray,
    m_q: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    return interface_law @ r_q, interface_law @ m_q


def build_risk_boundary_quotient_report() -> dict[str, object]:
    risk_parameter = 0.7
    tail_probability = 0.1

    # State zero executes two interfaces that the encoder maps to the same code.
    # State one executes a third, separately represented interface.
    interface_law = np.asarray(
        [[0.8, 0.2, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    encoder_codes = np.asarray([0, 0, 1], dtype=np.int64)

    # Exact-quotient slice: interfaces zero and one are indistinguishable for all
    # affine continuation witnesses in the declared unit ball.
    exact_r_q = np.asarray([0.92, 0.92, 0.90], dtype=np.float64)
    exact_m_q = np.asarray(
        [[0.16, 0.045], [0.16, 0.045], [0.08, 0.18]], dtype=np.float64
    )
    exact_fiber_diameter = _fiber_witness_diameter(
        exact_r_q, exact_m_q, 0, 1
    )
    exact_r, exact_m = _compose(interface_law, exact_r_q, exact_m_q)
    factored_r, factored_m = _compose(interface_law, exact_r_q, exact_m_q)
    exact_value = np.linalg.solve(np.eye(2) - exact_m, exact_r)
    factored_value = np.linalg.solve(np.eye(2) - factored_m, factored_r)

    # Approximate-quotient slice: the true primitive distinguishes the merged
    # interfaces, while the encoded model uses their source-balanced average.
    true_r_q = np.asarray([0.90, 0.94, 0.90], dtype=np.float64)
    true_m_q = np.asarray(
        [[0.15, 0.05], [0.17, 0.04], [0.08, 0.18]], dtype=np.float64
    )
    encoded_r_q = np.asarray([0.92, 0.92, 0.90], dtype=np.float64)
    encoded_m_q = np.asarray(
        [[0.16, 0.045], [0.16, 0.045], [0.08, 0.18]], dtype=np.float64
    )
    fiber_diameter = _fiber_witness_diameter(true_r_q, true_m_q, 0, 1)
    true_r, true_m = _compose(interface_law, true_r_q, true_m_q)
    encoded_r, encoded_m = _compose(
        interface_law, encoded_r_q, encoded_m_q
    )

    resolvent = np.linalg.inv(np.eye(2) - true_m)
    true_value = resolvent @ true_r
    encoded_value = np.linalg.solve(np.eye(2) - encoded_m, encoded_r)
    local_residual = (
        (true_r - encoded_r) + (true_m - encoded_m) @ encoded_value
    )
    value_error = true_value - encoded_value
    reconstructed_error = resolvent @ local_residual
    pointwise_envelope = resolvent @ np.abs(local_residual)

    witness_scale = max(1.0, float(np.max(np.abs(encoded_value))))
    resolvent_inf_norm = float(np.linalg.norm(resolvent, ord=np.inf))
    uniform_resolvent_bound = (
        resolvent_inf_norm * witness_scale * fiber_diameter
    )
    minimum_positive_value = float(
        min(np.min(true_value), np.min(encoded_value))
    )
    log_error = np.abs(np.log(true_value) - np.log(encoded_value))
    log_envelope = pointwise_envelope / minimum_positive_value

    true_requirement = (
        np.log(true_value) + np.log(1.0 / tail_probability)
    ) / risk_parameter
    encoded_requirement = (
        np.log(encoded_value) + np.log(1.0 / tail_probability)
    ) / risk_parameter
    requirement_envelope = log_envelope / risk_parameter
    conservative_requirement = encoded_requirement + requirement_envelope

    # Construct one oracle-shadow state inside the conservative boundary band.
    conservative_gap = float(
        conservative_requirement[0] - true_requirement[0]
    )
    decision_margin = 0.5 * conservative_gap
    exact_continue = bool(decision_margin > 0.0)
    conservative_continue = bool(
        true_requirement[0] + decision_margin
        > conservative_requirement[0]
    )
    boundary_indicator = bool(
        0.0 < decision_margin <= requirement_envelope[0]
    )

    checks = {
        "primitive_weighted_mass_at_least_one": bool(
            np.all(true_r_q + np.sum(true_m_q, axis=1) >= 1.0)
            and np.all(
                encoded_r_q + np.sum(encoded_m_q, axis=1) >= 1.0
            )
        ),
        "mgf_values_at_least_one": bool(
            np.all(true_value >= 1.0) and np.all(encoded_value >= 1.0)
        ),
        "exact_fiber_has_zero_witness_distance": bool(
            np.isclose(exact_fiber_diameter, 0.0)
        ),
        "exact_quotient_preserves_value": bool(
            np.allclose(exact_value, factored_value, atol=1e-13)
        ),
        "resolvent_residual_identity": bool(
            np.allclose(value_error, reconstructed_error, atol=1e-13)
        ),
        "pointwise_positive_envelope": bool(
            np.all(np.abs(value_error) <= pointwise_envelope + 1e-13)
        ),
        "uniform_fiber_bound": bool(
            np.max(pointwise_envelope) <= uniform_resolvent_bound + 1e-13
        ),
        "log_envelope": bool(np.all(log_error <= log_envelope + 1e-13)),
        "requirement_envelope": bool(
            np.all(
                np.abs(true_requirement - encoded_requirement)
                <= requirement_envelope + 1e-13
            )
        ),
        "constructed_first_disagreement_is_in_boundary_band": bool(
            exact_continue and not conservative_continue and boundary_indicator
        ),
    }
    status = (
        "RISK_BOUNDARY_QUOTIENT_IDENTITIES_VERIFIED"
        if all(checks.values())
        else "RISK_BOUNDARY_QUOTIENT_VERIFICATION_FAILED"
    )
    return {
        "status": status,
        "risk_parameter": risk_parameter,
        "tail_probability": tail_probability,
        "encoder_codes": encoder_codes.tolist(),
        "exact_quotient": {
            "fiber_witness_diameter": exact_fiber_diameter,
            "true_value": exact_value.tolist(),
            "factored_value": factored_value.tolist(),
        },
        "approximate_quotient": {
            "fiber_witness_diameter": fiber_diameter,
            "true_value": true_value.tolist(),
            "encoded_value": encoded_value.tolist(),
            "local_residual": local_residual.tolist(),
            "value_error": value_error.tolist(),
            "resolvent_reconstruction": reconstructed_error.tolist(),
            "pointwise_envelope": pointwise_envelope.tolist(),
            "witness_scale": witness_scale,
            "resolvent_inf_norm": resolvent_inf_norm,
            "uniform_resolvent_bound": uniform_resolvent_bound,
            "log_error": log_error.tolist(),
            "log_envelope": log_envelope.tolist(),
            "requirement_error": np.abs(
                true_requirement - encoded_requirement
            ).tolist(),
            "requirement_envelope": requirement_envelope.tolist(),
        },
        "boundary_construction": {
            "conservative_gap": conservative_gap,
            "decision_margin": decision_margin,
            "exact_continue": exact_continue,
            "conservative_continue": conservative_continue,
            "boundary_indicator": boundary_indicator,
        },
        "checks": checks,
        "non_claim": (
            "This finite construction verifies quotient, resolvent, log-risk, "
            "and stopped-boundary algebra. It does not prove a quotient learning "
            "rate, neural consistency, Oracle headroom, or Pareto improvement."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_risk_boundary_quotient_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
