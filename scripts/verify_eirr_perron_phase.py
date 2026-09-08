#!/usr/bin/env python3
"""Verify the multi-state Perron EIRR information phase split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _weighted_variance(values: np.ndarray, probabilities: np.ndarray) -> float:
    mean = float(probabilities @ values)
    return float(probabilities @ np.square(values - mean))


def build_eirr_perron_phase_report() -> dict[str, object]:
    left = np.asarray([0.40, 0.60], dtype=np.float64)
    right = np.asarray([1.30, 0.80], dtype=np.float64)
    projector = np.outer(right, left)
    beta = 0.20
    forcing = np.asarray([0.70, 0.90], dtype=np.float64)
    nu = np.asarray([0.70, 0.30], dtype=np.float64)
    source_state = np.asarray([0.58, 0.42], dtype=np.float64)
    gaps = np.asarray(
        [0.10, 0.05, 0.025, 0.0125, 0.00625, 0.003125],
        dtype=np.float64,
    )
    a = float(nu @ right)
    b = float(left @ forcing)

    def killed_operator(gap: float) -> np.ndarray:
        return beta * np.eye(2) + (1.0 - gap - beta) * projector

    def solve(gap: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        operator = killed_operator(gap)
        resolvent = np.linalg.inv(np.eye(2) - operator)
        return operator, resolvent @ forcing, nu @ resolvent

    # Nondegenerate primitive: one quotient interface per state. The terminal
    # and two continuation coefficients are normalized into outcome
    # probabilities with one common positive multiplier.
    nondegenerate_rows: list[dict[str, float]] = []
    for gap in gaps:
        operator, psi, eta = solve(float(gap))
        conditional_variance = np.zeros(2)
        projected_variance = np.zeros(2)
        for state in range(2):
            coefficients = np.concatenate(
                ([forcing[state]], operator[state])
            )
            common_multiplier = float(coefficients.sum())
            probabilities = coefficients / common_multiplier
            gamma_values = common_multiplier * np.asarray(
                [1.0, psi[0], psi[1]]
            )
            projected_values = common_multiplier * np.asarray(
                [0.0, right[0], right[1]]
            )
            conditional_variance[state] = _weighted_variance(
                gamma_values, probabilities
            )
            projected_variance[state] = _weighted_variance(
                projected_values, probabilities
            )
        ratio = eta / source_state
        raw_variance = float(
            np.sum(source_state * np.square(ratio) * conditional_variance)
        )
        theta = float(nu @ psi)
        log_variance = raw_variance / theta**2
        coefficient = float(
            np.sum(np.square(left) * projected_variance / source_state)
        )
        nondegenerate_rows.append(
            {
                "gap": float(gap),
                "theta": theta,
                "raw_efficiency_variance": raw_variance,
                "log_efficiency_variance": log_variance,
                "projected_noise_coefficient": coefficient,
                "gap4_raw_variance": float(gap**4 * raw_variance),
                "gap2_log_variance": float(gap**2 * log_variance),
            }
        )

    # Exact limiting coefficient at Delta=0.
    operator_zero = killed_operator(0.0)
    projected_variance_zero = np.zeros(2)
    for state in range(2):
        coefficients = np.concatenate(
            ([forcing[state]], operator_zero[state])
        )
        multiplier = float(coefficients.sum())
        probabilities = coefficients / multiplier
        projected_values = multiplier * np.asarray(
            [0.0, right[0], right[1]]
        )
        projected_variance_zero[state] = _weighted_variance(
            projected_values, probabilities
        )
    coefficient_limit = float(
        np.sum(np.square(left) * projected_variance_zero / source_state)
    )
    raw_limit = a**2 * b**2 * coefficient_limit

    # Degenerate primitive: known target composition mixes a deterministic
    # terminal interface and a continuation interface. Continuation outcome
    # multipliers are chosen so multiplier * z(next) is constant conditional on
    # each interface, while their target composition still equals M_Delta.
    terminal_probability = 0.50
    continuation_probability = 1.0 - terminal_probability
    source_action = np.asarray(
        [[0.45, 0.55], [0.35, 0.65]], dtype=np.float64
    )
    source_q = (source_state[:, None] * source_action).reshape(-1)
    target_action = np.asarray(
        [terminal_probability, continuation_probability]
    )
    degenerate_rows: list[dict[str, float]] = []
    max_projected_variance = 0.0
    for gap in gaps:
        operator, psi, eta = solve(float(gap))
        eta_q = (eta[:, None] * target_action[None, :]).reshape(-1)
        ratio = eta_q / source_q
        conditional_variance = np.zeros(4)
        projected_variance = np.zeros(4)
        for state in range(2):
            terminal_index = 2 * state
            continuation_index = terminal_index + 1
            conditional_variance[terminal_index] = 0.0
            projected_variance[terminal_index] = 0.0

            eigen_scale = (1.0 - gap) * right[state]
            probabilities = (
                operator[state] * right / eigen_scale
            )
            multipliers = (
                eigen_scale
                / (continuation_probability * right)
            )
            gamma_values = multipliers * psi
            projected_values = multipliers * right
            conditional_variance[continuation_index] = _weighted_variance(
                gamma_values, probabilities
            )
            projected_variance[continuation_index] = _weighted_variance(
                projected_values, probabilities
            )
        max_projected_variance = max(
            max_projected_variance, float(projected_variance.max())
        )
        raw_variance = float(
            np.sum(source_q * np.square(ratio) * conditional_variance)
        )
        theta = float(nu @ psi)
        log_variance = raw_variance / theta**2
        degenerate_rows.append(
            {
                "gap": float(gap),
                "theta": theta,
                "raw_efficiency_variance": raw_variance,
                "log_efficiency_variance": log_variance,
                "gap2_raw_variance": float(gap**2 * raw_variance),
            }
        )

    nondegenerate_gaps = np.asarray(
        [row["gap"] for row in nondegenerate_rows]
    )
    nondegenerate_raw = np.asarray(
        [row["raw_efficiency_variance"] for row in nondegenerate_rows]
    )
    nondegenerate_log = np.asarray(
        [row["log_efficiency_variance"] for row in nondegenerate_rows]
    )
    degenerate_raw = np.asarray(
        [row["raw_efficiency_variance"] for row in degenerate_rows]
    )
    degenerate_log = np.asarray(
        [row["log_efficiency_variance"] for row in degenerate_rows]
    )
    tail = slice(-4, None)
    nondegenerate_raw_slope = float(
        np.polyfit(
            np.log(nondegenerate_gaps[tail]),
            np.log(nondegenerate_raw[tail]),
            1,
        )[0]
    )
    nondegenerate_log_slope = float(
        np.polyfit(
            np.log(nondegenerate_gaps[tail]),
            np.log(nondegenerate_log[tail]),
            1,
        )[0]
    )
    degenerate_raw_slope = float(
        np.polyfit(
            np.log(nondegenerate_gaps[tail]),
            np.log(degenerate_raw[tail]),
            1,
        )[0]
    )
    degenerate_log_slope = float(
        np.polyfit(
            np.log(nondegenerate_gaps[tail]),
            np.log(degenerate_log[tail]),
            1,
        )[0]
    )
    last_nondegenerate = nondegenerate_rows[-1]
    raw_limit_relative_error = abs(
        last_nondegenerate["gap4_raw_variance"] - raw_limit
    ) / raw_limit
    log_limit_relative_error = abs(
        last_nondegenerate["gap2_log_variance"] - coefficient_limit
    ) / coefficient_limit

    smallest_operator = killed_operator(float(gaps[-1]))
    reduced_resolvent = (
        np.linalg.inv(np.eye(2) - smallest_operator)
        - projector / gaps[-1]
    )
    exact_reduced_resolvent = (np.eye(2) - projector) / (1.0 - beta)
    checks = {
        "perron_normalization_holds": bool(
            np.isclose(left.sum(), 1.0)
            and np.isclose(left @ right, 1.0)
        ),
        "perron_eigen_relations_hold": bool(
            np.allclose(smallest_operator @ right, (1.0 - gaps[-1]) * right)
            and np.allclose(
                left @ smallest_operator, (1.0 - gaps[-1]) * left
            )
        ),
        "reduced_resolvent_is_exact_and_bounded": bool(
            np.allclose(
                reduced_resolvent, exact_reduced_resolvent, atol=1e-11
            )
        ),
        "nondegenerate_raw_limit_matches": raw_limit_relative_error < 0.02,
        "nondegenerate_log_limit_matches": log_limit_relative_error < 0.02,
        "nondegenerate_raw_slope_is_minus_four": abs(
            nondegenerate_raw_slope + 4.0
        ) < 0.08,
        "nondegenerate_log_slope_is_minus_two": abs(
            nondegenerate_log_slope + 2.0
        ) < 0.08,
        "degenerate_projected_noise_is_zero": max_projected_variance < 1e-25,
        "degenerate_raw_slope_is_minus_two": abs(
            degenerate_raw_slope + 2.0
        ) < 0.08,
        "degenerate_log_slope_is_zero": abs(degenerate_log_slope) < 0.08,
    }
    status = (
        "EIRR_PERRON_PHASE_SPLIT_VERIFIED"
        if all(checks.values())
        else "EIRR_PERRON_PHASE_SPLIT_VERIFICATION_FAILED"
    )
    return {
        "status": status,
        "model": {
            "left_perron_vector": left.tolist(),
            "right_perron_vector": right.tolist(),
            "subleading_eigenvalue": beta,
            "initial_perron_mass": a,
            "forcing_perron_mass": b,
            "source_state_design": source_state.tolist(),
        },
        "nondegenerate": {
            "rows": nondegenerate_rows,
            "coefficient_limit": coefficient_limit,
            "predicted_gap4_raw_limit": raw_limit,
            "raw_limit_relative_error_at_smallest_gap": (
                raw_limit_relative_error
            ),
            "log_limit_relative_error_at_smallest_gap": (
                log_limit_relative_error
            ),
            "tail_raw_log_log_slope": nondegenerate_raw_slope,
            "tail_log_risk_log_log_slope": nondegenerate_log_slope,
        },
        "degenerate": {
            "rows": degenerate_rows,
            "maximum_projected_noise_variance": max_projected_variance,
            "tail_raw_log_log_slope": degenerate_raw_slope,
            "tail_log_risk_log_log_slope": degenerate_log_slope,
        },
        "checks": checks,
        "non_claim": (
            "This verifier checks a two-state non-symmetric positive-operator "
            "family with fixed quotient design. It supports the finite Perron "
            "asymptotics but is not a neural, dependent-replay, or formal Pareto "
            "result."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_eirr_perron_phase_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
