#!/usr/bin/env python3
"""Verify singularity-free pilot learning of critical-mode EIRR allocation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _weighted_variance(values: np.ndarray, probabilities: np.ndarray) -> float:
    mean = float(probabilities @ values)
    return float(probabilities @ np.square(values - mean))


def _perron_modes(operator: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    eigenvalues, right_vectors = np.linalg.eig(operator)
    index = int(np.argmax(eigenvalues.real))
    eigenvalue = eigenvalues[index]
    right = right_vectors[:, index].real
    left_values, left_vectors = np.linalg.eig(operator.T)
    left_index = int(np.argmin(np.abs(left_values - eigenvalue)))
    left = left_vectors[:, left_index].real
    if float(left.sum()) < 0.0:
        left = -left
    left = left / float(left.sum())
    if float(left @ right) < 0.0:
        right = -right
    right = right / float(left @ right)
    return left, right


def _oracle_variance(
    sensitivity: np.ndarray, costs: np.ndarray, budget: float
) -> float:
    return float(
        np.square(np.sum(sensitivity * np.sqrt(costs))) / budget
    )


def _allocation_variance(
    true_sensitivity: np.ndarray,
    allocation_priority: np.ndarray,
    costs: np.ndarray,
    budget: float,
) -> float:
    scale = float(np.sum(allocation_priority * np.sqrt(costs)))
    counts = (
        budget * allocation_priority / np.sqrt(costs) / scale
    )
    return float(np.sum(np.square(true_sensitivity) / counts))


def build_eirr_adaptive_critical_design_report() -> dict[str, object]:
    left = np.asarray([0.40, 0.60], dtype=np.float64)
    right = np.asarray([1.30, 0.80], dtype=np.float64)
    projector = np.outer(right, left)
    beta = 0.20
    forcing = np.asarray([0.70, 0.90], dtype=np.float64)
    nu = np.asarray([0.70, 0.30], dtype=np.float64)
    costs = np.asarray([1.0, 4.0], dtype=np.float64)
    budget = 100_000.0
    gaps = np.asarray(
        [0.05, 0.025, 0.0125, 0.00625, 0.003125],
        dtype=np.float64,
    )
    pilot_scale = 20.0
    repetitions = 200
    rng = np.random.default_rng(20260830)

    def killed_operator(gap: float) -> np.ndarray:
        return beta * np.eye(2) + (1.0 - gap - beta) * projector

    rows: list[dict[str, object]] = []
    for gap in gaps:
        operator = killed_operator(float(gap))
        resolvent = np.linalg.inv(np.eye(2) - operator)
        psi = resolvent @ forcing
        eta = nu @ resolvent
        theta = float(nu @ psi)
        true_sensitivity = np.zeros(2)
        multipliers = np.zeros(2)
        primitive_probabilities: list[np.ndarray] = []
        for state in range(2):
            coefficients = np.concatenate(
                ([forcing[state]], operator[state])
            )
            multiplier = float(coefficients.sum())
            probabilities = coefficients / multiplier
            gamma_values = multiplier * np.asarray(
                [1.0, psi[0], psi[1]]
            )
            true_sensitivity[state] = (
                eta[state]
                / theta
                * np.sqrt(
                    _weighted_variance(gamma_values, probabilities)
                )
            )
            multipliers[state] = multiplier
            primitive_probabilities.append(probabilities)

        oracle_variance = _oracle_variance(
            true_sensitivity, costs, budget
        )
        pilot_per_interface = int(np.ceil(pilot_scale / gap))
        efficiency_ratios: list[float] = []
        mode_errors: list[float] = []
        for _ in range(repetitions):
            empirical_operator = np.zeros((2, 2), dtype=np.float64)
            empirical_probabilities: list[np.ndarray] = []
            for state in range(2):
                counts = rng.multinomial(
                    pilot_per_interface,
                    primitive_probabilities[state],
                )
                probabilities = counts / pilot_per_interface
                empirical_probabilities.append(probabilities)
                empirical_operator[state] = (
                    multipliers[state] * probabilities[1:]
                )

            estimated_left, estimated_right = _perron_modes(
                empirical_operator
            )
            estimated_priority = np.zeros(2)
            for state in range(2):
                projected_values = multipliers[state] * np.asarray(
                    [0.0, estimated_right[0], estimated_right[1]]
                )
                projected_sd = np.sqrt(
                    _weighted_variance(
                        projected_values,
                        empirical_probabilities[state],
                    )
                )
                estimated_priority[state] = (
                    estimated_left[state] * projected_sd
                )
            estimated_priority = np.maximum(
                estimated_priority, np.finfo(np.float64).tiny
            )
            variance = _allocation_variance(
                true_sensitivity,
                estimated_priority,
                costs,
                budget,
            )
            efficiency_ratios.append(variance / oracle_variance)
            mode_errors.append(
                float(
                    np.max(np.abs(estimated_left - left))
                    + np.max(np.abs(estimated_right - right))
                )
            )

        rows.append(
            {
                "gap": float(gap),
                "pilot_per_interface": pilot_per_interface,
                "pilot_gap2": float(pilot_per_interface * gap**2),
                "median_oracle_variance_ratio": float(
                    np.median(efficiency_ratios)
                ),
                "q90_oracle_variance_ratio": float(
                    np.quantile(efficiency_ratios, 0.90)
                ),
                "maximum_oracle_variance_ratio": float(
                    np.max(efficiency_ratios)
                ),
                "median_perron_mode_error": float(
                    np.median(mode_errors)
                ),
            }
        )

    pilot_sizes = np.asarray([row["pilot_per_interface"] for row in rows])
    pilot_gap2 = np.asarray([row["pilot_gap2"] for row in rows])
    medians = np.asarray(
        [row["median_oracle_variance_ratio"] for row in rows]
    )
    q90s = np.asarray([row["q90_oracle_variance_ratio"] for row in rows])
    checks = {
        "pilot_size_diverges": bool(np.all(np.diff(pilot_sizes) > 0)),
        "pilot_gap2_decreases_toward_zero": bool(
            np.all(np.diff(pilot_gap2) < 0) and pilot_gap2[-1] < 0.07
        ),
        "all_median_ratios_are_near_oracle": bool(
            np.all(medians < 1.01)
        ),
        "all_q90_ratios_are_near_oracle": bool(np.all(q90s < 1.02)),
        "design_efficiency_improves_as_pilot_grows": bool(
            medians[-1] < medians[0]
        ),
        "smallest_gap_median_regret_is_tiny": bool(
            medians[-1] - 1.0 < 2e-4
        ),
    }
    status = (
        "EIRR_SINGULARITY_FREE_ADAPTIVE_DESIGN_VERIFIED"
        if all(checks.values())
        else "EIRR_SINGULARITY_FREE_ADAPTIVE_DESIGN_VERIFICATION_FAILED"
    )
    return {
        "status": status,
        "model": {
            "left_perron_vector": left.tolist(),
            "right_perron_vector": right.tolist(),
            "subleading_eigenvalue": beta,
            "interface_costs": costs.tolist(),
            "pilot_rule": "ceil(20 / gap) per interface",
            "repetitions_per_gap": repetitions,
        },
        "rows": rows,
        "checks": checks,
        "interpretation": (
            "The pilot sizes diverge while m*gap^2 decreases to zero. The "
            "pilot is therefore below the critical scale for estimating the "
            "risk value itself, yet its normalized Perron-mode allocation "
            "approaches oracle efficiency because the common 1/gap factor "
            "cancels from allocation proportions."
        ),
        "non_claim": (
            "This finite simulation assumes a known exact quotient, known "
            "target composition, independent strata, bounded primitives, and "
            "a uniformly separated non-Perron spectrum. It is not a learned "
            "neural quotient, dependent replay, or Pareto result."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_eirr_adaptive_critical_design_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
