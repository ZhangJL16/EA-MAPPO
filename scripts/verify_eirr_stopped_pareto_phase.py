#!/usr/bin/env python3
"""Verify the Gaussian stopped-margin powers in EIRR Theorem 24."""

from __future__ import annotations

import argparse
import json
from math import gamma, pi, sqrt
from pathlib import Path

import numpy as np


def _normal_absolute_moment(power: float) -> float:
    return float(
        2.0 ** (power / 2.0)
        * gamma((power + 1.0) / 2.0)
        / sqrt(pi)
    )


def _normal_expectation(function_values: np.ndarray) -> float:
    nodes, weights = np.polynomial.legendre.leggauss(600)
    grid = 4.0 * (nodes + 1.0)
    density = np.exp(-0.5 * grid**2) / sqrt(2.0 * pi)
    return float(8.0 * np.sum(weights * density * function_values(grid)))


def _gaussian_slice(
    margin_exponent: float,
    standard_error: float,
    radius: float = 1.0,
) -> tuple[float, float]:
    def disagreement(grid: np.ndarray) -> np.ndarray:
        scaled = np.minimum(standard_error * grid / radius, 1.0)
        return 0.5 * scaled**margin_exponent

    def boundary_loss(grid: np.ndarray) -> np.ndarray:
        clipped = np.minimum(standard_error * grid, radius)
        coefficient = margin_exponent / (
            2.0 * (margin_exponent + 1.0) * radius**margin_exponent
        )
        return coefficient * clipped ** (margin_exponent + 1.0)

    return _normal_expectation(disagreement), _normal_expectation(boundary_loss)


def build_eirr_stopped_pareto_phase_report() -> dict[str, object]:
    exponents = [0.5, 1.0, 2.0]
    standard_errors = np.asarray([0.08, 0.04, 0.02, 0.01])
    rows: list[dict[str, object]] = []
    disagreement_slope_errors: list[float] = []
    loss_slope_errors: list[float] = []
    leading_ratio_errors: list[float] = []

    for exponent in exponents:
        disagreements: list[float] = []
        losses: list[float] = []
        slice_rows: list[dict[str, float]] = []
        for standard_error in standard_errors:
            disagreement, loss = _gaussian_slice(
                exponent, float(standard_error)
            )
            disagreement_leading = (
                0.5
                * standard_error**exponent
                * _normal_absolute_moment(exponent)
            )
            loss_leading = (
                exponent
                / (2.0 * (exponent + 1.0))
                * standard_error ** (exponent + 1.0)
                * _normal_absolute_moment(exponent + 1.0)
            )
            disagreements.append(disagreement)
            losses.append(loss)
            disagreement_ratio = disagreement / disagreement_leading
            loss_ratio = loss / loss_leading
            leading_ratio_errors.extend(
                [abs(disagreement_ratio - 1.0), abs(loss_ratio - 1.0)]
            )
            slice_rows.append(
                {
                    "standard_error": float(standard_error),
                    "disagreement_probability": disagreement,
                    "boundary_weighted_loss": loss,
                    "disagreement_leading_ratio": disagreement_ratio,
                    "loss_leading_ratio": loss_ratio,
                }
            )

        disagreement_slope = float(
            np.polyfit(np.log(standard_errors), np.log(disagreements), 1)[0]
        )
        loss_slope = float(
            np.polyfit(np.log(standard_errors), np.log(losses), 1)[0]
        )
        disagreement_slope_errors.append(abs(disagreement_slope - exponent))
        loss_slope_errors.append(abs(loss_slope - (exponent + 1.0)))
        rows.append(
            {
                "margin_exponent": exponent,
                "disagreement_log_slope": disagreement_slope,
                "predicted_disagreement_slope": exponent,
                "loss_log_slope": loss_slope,
                "predicted_loss_slope": exponent + 1.0,
                "rows": slice_rows,
            }
        )

    variance_ratio_rows: list[dict[str, float]] = []
    for exponent in exponents:
        for omega in [0.02, 0.05, 0.10]:
            ratio = (1.0 + omega) / (1.0 - omega)
            variance_ratio_rows.append(
                {
                    "margin_exponent": exponent,
                    "omega": omega,
                    "adaptive_variance_ratio": ratio,
                    "pareto_radius_ratio": ratio ** (exponent / 2.0),
                    "boundary_loss_radius_ratio": ratio
                    ** ((exponent + 1.0) / 2.0),
                }
            )

    checks = {
        "disagreement_power_matches_margin_exponent": bool(
            max(disagreement_slope_errors) < 2e-4
        ),
        "loss_power_matches_margin_plus_one": bool(
            max(loss_slope_errors) < 2e-4
        ),
        "gaussian_slice_matches_exact_leading_constants": bool(
            max(leading_ratio_errors) < 2e-4
        ),
        "adaptive_variance_ratio_propagates_by_declared_powers": all(
            row["boundary_loss_radius_ratio"]
            >= row["pareto_radius_ratio"]
            >= 1.0
            for row in variance_ratio_rows
        ),
    }
    status = (
        "EIRR_STOPPED_PARETO_PHASE_VERIFIED"
        if all(checks.values())
        else "EIRR_STOPPED_PARETO_PHASE_VERIFICATION_FAILED"
    )
    return {
        "status": status,
        "gaussian_shift_slices": rows,
        "adaptive_design_transfer": variance_ratio_rows,
        "maximum_disagreement_slope_error": max(
            disagreement_slope_errors
        ),
        "maximum_loss_slope_error": max(loss_slope_errors),
        "maximum_leading_constant_relative_error": max(leading_ratio_errors),
        "checks": checks,
        "non_claim": (
            "The calculation verifies the exact regular Gaussian plug-in "
            "slice and variance-ratio algebra. It is not a universal minimax "
            "lower bound or empirical Pareto evidence."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_eirr_stopped_pareto_phase_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
