#!/usr/bin/env python3
"""Verify cost-aware critical-mode allocation for stopped log-EIRR."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _weighted_variance(values: np.ndarray, probabilities: np.ndarray) -> float:
    mean = float(probabilities @ values)
    return float(probabilities @ np.square(values - mean))


def _oracle_counts(
    sensitivity: np.ndarray, costs: np.ndarray, budget: float
) -> np.ndarray:
    scale = float(np.sum(sensitivity * np.sqrt(costs)))
    return budget * sensitivity / np.sqrt(costs) / scale


def _stratified_variance(
    sensitivity: np.ndarray, counts: np.ndarray
) -> float:
    return float(np.sum(np.square(sensitivity) / counts))


def build_eirr_critical_design_report() -> dict[str, object]:
    left = np.asarray([0.40, 0.60], dtype=np.float64)
    right = np.asarray([1.30, 0.80], dtype=np.float64)
    projector = np.outer(right, left)
    beta = 0.20
    forcing = np.asarray([0.70, 0.90], dtype=np.float64)
    nu = np.asarray([0.70, 0.30], dtype=np.float64)
    costs = np.asarray([1.0, 4.0], dtype=np.float64)
    budget = 100_000.0
    gaps = np.asarray(
        [0.10, 0.05, 0.025, 0.0125, 0.00625, 0.003125],
        dtype=np.float64,
    )

    def killed_operator(gap: float) -> np.ndarray:
        return beta * np.eye(2) + (1.0 - gap - beta) * projector

    rows: list[dict[str, object]] = []
    last_sensitivity = np.zeros(2)
    for gap in gaps:
        operator = killed_operator(float(gap))
        resolvent = np.linalg.inv(np.eye(2) - operator)
        psi = resolvent @ forcing
        eta = nu @ resolvent
        theta = float(nu @ psi)
        conditional_variance = np.zeros(2)
        for state in range(2):
            coefficients = np.concatenate(([forcing[state]], operator[state]))
            multiplier = float(coefficients.sum())
            probabilities = coefficients / multiplier
            gamma_values = multiplier * np.asarray([1.0, psi[0], psi[1]])
            conditional_variance[state] = _weighted_variance(
                gamma_values, probabilities
            )
        sensitivity = eta / theta * np.sqrt(conditional_variance)
        counts = _oracle_counts(sensitivity, costs, budget)
        variance = _stratified_variance(sensitivity, counts)
        limiting_scaled_constant = float(gap**2 * budget * variance)
        rows.append(
            {
                "gap": float(gap),
                "log_risk_sensitivity": sensitivity.tolist(),
                "oracle_counts": counts.tolist(),
                "oracle_cost_fractions": (costs * counts / budget).tolist(),
                "oracle_variance": variance,
                "gap2_budget_variance": limiting_scaled_constant,
            }
        )
        last_sensitivity = sensitivity

    operator_zero = killed_operator(0.0)
    projected_variance = np.zeros(2)
    for state in range(2):
        coefficients = np.concatenate(([forcing[state]], operator_zero[state]))
        multiplier = float(coefficients.sum())
        probabilities = coefficients / multiplier
        projected_values = multiplier * np.asarray([0.0, right[0], right[1]])
        projected_variance[state] = _weighted_variance(
            projected_values, probabilities
        )
    critical_priority = left * np.sqrt(projected_variance)
    predicted_limit = float(
        np.square(np.sum(critical_priority * np.sqrt(costs)))
    )
    observed_limit = float(rows[-1]["gap2_budget_variance"])
    critical_relative_error = abs(observed_limit - predicted_limit) / predicted_limit

    oracle_counts = _oracle_counts(last_sensitivity, costs, budget)
    oracle_variance = _stratified_variance(last_sensitivity, oracle_counts)
    rng = np.random.default_rng(20260830)
    random_ratios: list[float] = []
    for proportions in rng.dirichlet(np.ones(2), size=2000):
        counts = budget * proportions / float(costs @ proportions)
        random_ratios.append(
            _stratified_variance(last_sensitivity, counts) / oracle_variance
        )

    relative_multipliers = np.asarray([0.90, 1.08], dtype=np.float64)
    estimated_sensitivity = last_sensitivity * relative_multipliers
    pilot_budget = 5_000.0
    stage_two_budget = budget - pilot_budget
    plugin_counts = _oracle_counts(
        estimated_sensitivity, costs, stage_two_budget
    )
    plugin_variance = _stratified_variance(last_sensitivity, plugin_counts)
    stage_two_oracle_counts = _oracle_counts(
        last_sensitivity, costs, stage_two_budget
    )
    stage_two_oracle_variance = _stratified_variance(
        last_sensitivity, stage_two_oracle_counts
    )
    epsilon = float(np.max(np.abs(relative_multipliers - 1.0)))
    plugin_stage_two_ratio = plugin_variance / stage_two_oracle_variance
    plugin_bound = (1.0 + epsilon) / (1.0 - epsilon)
    plugin_full_budget_ratio = plugin_variance / oracle_variance
    plugin_full_budget_bound = budget / stage_two_budget * plugin_bound

    checks = {
        "oracle_spends_exact_budget": bool(
            np.isclose(costs @ oracle_counts, budget)
        ),
        "oracle_closed_form_matches_cauchy_bound": bool(
            np.isclose(
                budget * oracle_variance,
                np.square(
                    np.sum(last_sensitivity * np.sqrt(costs))
                ),
            )
        ),
        "random_allocations_do_not_beat_oracle": bool(
            min(random_ratios) >= 1.0 - 1e-12
        ),
        "critical_limit_matches": critical_relative_error < 0.02,
        "plugin_ratio_obeys_finite_error_bound": bool(
            plugin_stage_two_ratio <= plugin_bound + 1e-12
        ),
        "pilot_adjusted_full_budget_bound_holds": bool(
            plugin_full_budget_ratio <= plugin_full_budget_bound + 1e-12
        ),
    }
    status = (
        "EIRR_CRITICAL_MODE_DESIGN_VERIFIED"
        if all(checks.values())
        else "EIRR_CRITICAL_MODE_DESIGN_VERIFICATION_FAILED"
    )
    return {
        "status": status,
        "model": {
            "left_perron_vector": left.tolist(),
            "right_perron_vector": right.tolist(),
            "interface_costs": costs.tolist(),
            "budget": budget,
            "gaps": gaps.tolist(),
        },
        "critical_limit": {
            "projected_standard_deviations": np.sqrt(
                projected_variance
            ).tolist(),
            "critical_priority_without_cost": critical_priority.tolist(),
            "predicted_gap2_budget_variance": predicted_limit,
            "observed_gap2_budget_variance_at_smallest_gap": observed_limit,
            "relative_error_at_smallest_gap": critical_relative_error,
            "rows": rows,
        },
        "oracle": {
            "counts_at_smallest_gap": oracle_counts.tolist(),
            "variance_at_smallest_gap": oracle_variance,
            "minimum_random_allocation_variance_ratio": min(random_ratios),
            "median_random_allocation_variance_ratio": float(
                np.median(random_ratios)
            ),
        },
        "plugin": {
            "pilot_budget": pilot_budget,
            "relative_sensitivity_multipliers": relative_multipliers.tolist(),
            "maximum_relative_error": epsilon,
            "stage_two_efficiency_ratio": plugin_stage_two_ratio,
            "stage_two_efficiency_upper_bound": plugin_bound,
            "full_budget_efficiency_ratio": plugin_full_budget_ratio,
            "pilot_adjusted_full_budget_upper_bound": plugin_full_budget_bound,
        },
        "checks": checks,
        "non_claim": (
            "The oracle is a cost-aware Neyman allocation specialized to the "
            "stopped log-EIRR canonical gradient. The verifier does not prove "
            "learned quotient recovery, pilot relative-consistency rates, "
            "dependent replay inference, or a navigation Pareto gain."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_eirr_critical_design_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
