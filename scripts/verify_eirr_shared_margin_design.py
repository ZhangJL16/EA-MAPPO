#!/usr/bin/env python3
"""Verify convexity and KKT structure of EIRR Theorem 25."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize


def _variances(allocation: np.ndarray, sensitivities: np.ndarray) -> np.ndarray:
    return np.sum(sensitivities**2 / allocation[None, :], axis=1)


def _objective(
    allocation: np.ndarray,
    sensitivities: np.ndarray,
    weights: np.ndarray,
    power: float,
) -> float:
    return float(np.sum(weights * _variances(allocation, sensitivities) ** power))


def _gradient(
    allocation: np.ndarray,
    sensitivities: np.ndarray,
    weights: np.ndarray,
    power: float,
) -> np.ndarray:
    variances = _variances(allocation, sensitivities)
    return -power * np.sum(
        weights[:, None]
        * variances[:, None] ** (power - 1.0)
        * sensitivities**2
        / allocation[None, :] ** 2,
        axis=0,
    )


def _solve(
    sensitivities: np.ndarray,
    weights: np.ndarray,
    costs: np.ndarray,
    budget: float,
    power: float,
) -> np.ndarray:
    initial = np.full(costs.size, 1.0 / costs.size)

    def allocation_from_shares(shares: np.ndarray) -> np.ndarray:
        return budget * shares / costs

    def scaled_objective(shares: np.ndarray) -> float:
        return budget**power * _objective(
            allocation_from_shares(shares),
            sensitivities,
            weights,
            power,
        )

    def scaled_gradient(shares: np.ndarray) -> np.ndarray:
        allocation = allocation_from_shares(shares)
        return (
            budget**power
            * _gradient(allocation, sensitivities, weights, power)
            * budget
            / costs
        )

    result = minimize(
        scaled_objective,
        initial,
        jac=scaled_gradient,
        method="SLSQP",
        bounds=[(1e-10, None)] * costs.size,
        constraints={
            "type": "eq",
            "fun": lambda shares: float(np.sum(shares) - 1.0),
            "jac": lambda shares: np.ones_like(shares),
        },
        options={"ftol": 1e-12, "maxiter": 2000},
    )
    if not result.success:
        raise RuntimeError(result.message)
    return allocation_from_shares(np.asarray(result.x, dtype=np.float64))


def _kkt_fixed_point(
    allocation: np.ndarray,
    sensitivities: np.ndarray,
    weights: np.ndarray,
    costs: np.ndarray,
    budget: float,
    power: float,
) -> np.ndarray:
    variances = _variances(allocation, sensitivities)
    score = np.sqrt(
        np.sum(
            weights[:, None]
            * variances[:, None] ** (power - 1.0)
            * sensitivities**2,
            axis=0,
        )
        / costs
    )
    return budget * score / float(costs @ score)


def build_eirr_shared_margin_design_report() -> dict[str, object]:
    sensitivities = np.asarray(
        [
            [1.00, 0.30, 0.10, 0.55, 0.20],
            [0.15, 0.90, 0.45, 0.20, 0.35],
            [0.25, 0.20, 1.10, 0.40, 0.15],
            [0.60, 0.10, 0.25, 0.35, 0.95],
        ],
        dtype=np.float64,
    )
    weights = np.asarray([1.0, 0.7, 1.3, 0.9], dtype=np.float64)
    costs = np.asarray([1.0, 1.4, 0.8, 1.8, 1.1], dtype=np.float64)
    budget = 1000.0
    powers = [0.4, 0.75, 1.0, 1.5]
    rng = np.random.default_rng(25082026)

    optimization_rows: list[dict[str, object]] = []
    maximum_kkt_relative_error = 0.0
    minimum_directional_curvature = float("inf")
    minimum_random_objective_ratio = float("inf")
    for power in powers:
        allocation = _solve(
            sensitivities, weights, costs, budget, power
        )
        fixed_point = _kkt_fixed_point(
            allocation, sensitivities, weights, costs, budget, power
        )
        kkt_error = float(
            np.max(np.abs(fixed_point / allocation - 1.0))
        )
        maximum_kkt_relative_error = max(
            maximum_kkt_relative_error, kkt_error
        )
        optimum = _objective(allocation, sensitivities, weights, power)

        random_ratios: list[float] = []
        for _ in range(1000):
            shares = rng.dirichlet(np.ones(costs.size))
            candidate = budget * shares / costs
            random_ratios.append(
                _objective(candidate, sensitivities, weights, power) / optimum
            )
        minimum_random_objective_ratio = min(
            minimum_random_objective_ratio, min(random_ratios)
        )

        for _ in range(500):
            direction = rng.normal(size=costs.size)
            direction /= np.linalg.norm(direction)
            curvature = 0.0
            for weight, row in zip(weights, sensitivities, strict=True):
                b = row**2
                scalar = float(np.sum(b / allocation))
                first = float(np.sum(b * direction / allocation**2))
                second = float(
                    np.sum(b * direction**2 / allocation**3)
                )
                curvature += (
                    weight
                    * power
                    * scalar ** (power - 2.0)
                    * ((power - 1.0) * first**2 + 2.0 * scalar * second)
                )
            minimum_directional_curvature = min(
                minimum_directional_curvature, curvature
            )

        optimization_rows.append(
            {
                "power": power,
                "allocation": allocation.tolist(),
                "cost": float(costs @ allocation),
                "objective": optimum,
                "kkt_fixed_point_relative_error": kkt_error,
                "minimum_random_objective_ratio": min(random_ratios),
            }
        )

    common_gap_allocations: list[list[float]] = []
    for gap in [0.20, 0.10, 0.05]:
        common_gap_allocations.append(
            _solve(
                sensitivities / gap,
                weights,
                costs,
                budget,
                power=0.75,
            ).tolist()
        )
    common_gap_array = np.asarray(common_gap_allocations)
    common_gap_relative_spread = float(
        np.max(np.ptp(common_gap_array, axis=0) / common_gap_array[0])
    )

    heterogeneous = np.eye(2, dtype=np.float64) / np.asarray(
        [0.05, 0.20], dtype=np.float64
    )[:, None]
    heterogeneous_allocation = _solve(
        heterogeneous,
        np.ones(2),
        np.ones(2),
        budget=100.0,
        power=1.0,
    )

    checks = {
        "all_sampled_directional_curvatures_are_positive": bool(
            minimum_directional_curvature > 0.0
        ),
        "kkt_fixed_point_matches_convex_optimum": bool(
            maximum_kkt_relative_error < 2e-5
        ),
        "no_random_allocation_beats_optimum": bool(
            minimum_random_objective_ratio >= 1.0 - 1e-10
        ),
        "common_critical_factor_cancels": bool(
            common_gap_relative_spread < 2e-6
        ),
        "more_critical_query_receives_more_budget": bool(
            heterogeneous_allocation[0] > heterogeneous_allocation[1]
        ),
    }
    status = (
        "EIRR_SHARED_MARGIN_DESIGN_VERIFIED"
        if all(checks.values())
        else "EIRR_SHARED_MARGIN_DESIGN_VERIFICATION_FAILED"
    )
    return {
        "status": status,
        "optimization_rows": optimization_rows,
        "maximum_kkt_relative_error": maximum_kkt_relative_error,
        "minimum_directional_curvature": minimum_directional_curvature,
        "minimum_random_objective_ratio": minimum_random_objective_ratio,
        "common_critical_gap": {
            "allocations": common_gap_allocations,
            "maximum_relative_spread": common_gap_relative_spread,
        },
        "heterogeneous_critical_gaps": {
            "gaps": [0.05, 0.20],
            "allocation": heterogeneous_allocation.tolist(),
        },
        "checks": checks,
        "non_claim": (
            "This verifies the continuous convex program and critical scaling. "
            "It does not verify learned sensitivities, dependent replay, integer "
            "rounding, or an empirical Pareto gain."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_eirr_shared_margin_design_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
