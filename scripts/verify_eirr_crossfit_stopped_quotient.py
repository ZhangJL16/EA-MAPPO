#!/usr/bin/env python3
"""Verify EIRR Theorem 32 rate and quotient-overlap algebra."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def local_shift_coefficient(
    source_prob: np.ndarray,
    target_prob: np.ndarray,
    cell_index: np.ndarray,
) -> float:
    """Return sum_A Q(A)/P(A) after pushing both laws onto cells."""
    source = np.asarray(source_prob, dtype=float)
    target = np.asarray(target_prob, dtype=float)
    cells = np.asarray(cell_index)
    if source.shape != target.shape or cells.shape != source.shape:
        raise ValueError("source, target, and cell arrays must have equal shape")
    if np.any(source < 0.0) or np.any(target < 0.0):
        raise ValueError("probabilities must be nonnegative")
    if not np.isclose(source.sum(), 1.0) or not np.isclose(target.sum(), 1.0):
        raise ValueError("source and target probabilities must each sum to one")
    coefficient = 0.0
    for cell in np.unique(cells):
        mask = cells == cell
        source_mass = float(source[mask].sum())
        target_mass = float(target[mask].sum())
        if target_mass == 0.0:
            continue
        if source_mass == 0.0:
            return float("inf")
        coefficient += target_mass / source_mass
    return float(coefficient)


def holder_rate_powers(
    *, smoothness: float, dimension: float, margin_exponent: float
) -> dict[str, float]:
    alpha = float(smoothness)
    dim = float(dimension)
    kappa = float(margin_exponent)
    if alpha <= 0.0 or dim <= 0.0 or kappa <= 0.0:
        raise ValueError("smoothness, dimension, and margin exponent must be positive")
    root_score = alpha / (2.0 * alpha + dim)
    mse = 2.0 * root_score
    return {
        "root_score_power": root_score,
        "mse_power": mse,
        "disagreement_power": mse * kappa / (kappa + 2.0),
        "boundary_loss_power": mse * (kappa + 1.0) / (kappa + 2.0),
        "joint_gap_power": (2.0 * alpha + dim) / alpha,
    }


def build_crossfit_stopped_quotient_report() -> dict[str, object]:
    checks: dict[str, bool] = {}
    overlap_rows: list[dict[str, float]] = []
    z_cells = 8
    for nuisance_cells in [1, 4, 16, 64]:
        atoms = z_cells * nuisance_cells
        source = np.full(atoms, 1.0 / atoms)
        target = np.zeros(atoms)
        raw_cell = np.arange(atoms)
        quotient_cell = np.repeat(np.arange(z_cells), nuisance_cells)
        for z in range(z_cells):
            target[z * nuisance_cells] = 1.0 / z_cells
        raw = local_shift_coefficient(source, target, raw_cell)
        quotient = local_shift_coefficient(source, target, quotient_cell)
        ratio = raw / quotient
        overlap_rows.append(
            {
                "nuisance_multiplicity": float(nuisance_cells),
                "raw_coefficient": raw,
                "quotient_coefficient": quotient,
                "raw_to_quotient_ratio": ratio,
            }
        )
        checks[f"raw_penalty_equals_nuisance_{nuisance_cells}"] = bool(
            abs(ratio - nuisance_cells) < 1e-12
        )

    quotient_powers = holder_rate_powers(
        smoothness=1.0, dimension=2.0, margin_exponent=1.0
    )
    raw_powers = holder_rate_powers(
        smoothness=1.0, dimension=8.0, margin_exponent=1.0
    )
    checks["quotient_mse_is_faster"] = bool(
        quotient_powers["mse_power"] > raw_powers["mse_power"]
    )
    checks["quotient_decision_is_faster"] = bool(
        quotient_powers["disagreement_power"]
        > raw_powers["disagreement_power"]
    )
    checks["quotient_boundary_loss_is_faster"] = bool(
        quotient_powers["boundary_loss_power"]
        > raw_powers["boundary_loss_power"]
    )
    checks["quotient_joint_gap_requirement_is_weaker"] = bool(
        quotient_powers["joint_gap_power"] < raw_powers["joint_gap_power"]
    )

    status = (
        "EIRR_CROSSFIT_STOPPED_QUOTIENT_VERIFIED"
        if all(checks.values())
        else "EIRR_CROSSFIT_STOPPED_QUOTIENT_VERIFICATION_FAILED"
    )
    return {
        "status": status,
        "overlap_rows": overlap_rows,
        "holder_rate_comparison": {
            "quotient_dimension_2": quotient_powers,
            "raw_dimension_8": raw_powers,
        },
        "checks": checks,
        "non_claim": (
            "The strict comparison is against a declared raw local model. A "
            "Doob/DICE/KROPE method supplied with or able to learn the same "
            "quotient may attain the quotient coefficient and rate as well."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_crossfit_stopped_quotient_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
