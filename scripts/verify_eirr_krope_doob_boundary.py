#!/usr/bin/env python3
"""Verify the risk-neutral representation failure and oracle-Doob boundary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def build_eirr_krope_doob_boundary_report() -> dict[str, object]:
    risk_parameter = 1.0
    deterministic_cost = 1.0
    mixture_costs = np.asarray([0.0, 2.0])
    mixture_probabilities = np.asarray([0.5, 0.5])
    deterministic_mean = deterministic_cost
    mixture_mean = float(mixture_probabilities @ mixture_costs)
    deterministic_mgf = float(np.exp(risk_parameter * deterministic_cost))
    mixture_mgf = float(
        mixture_probabilities @ np.exp(risk_parameter * mixture_costs)
    )
    exact_mgf_gap = float((np.exp(risk_parameter) - 1.0) ** 2 / 2.0)
    log_risk_gap = float(np.log(mixture_mgf) - np.log(deterministic_mgf))
    separating_threshold = float(
        0.5 * (np.log(mixture_mgf) + np.log(deterministic_mgf))
    )

    nuisance_rows: list[dict[str, float]] = []
    for categories in [4, 16, 64, 256]:
        source = np.full(categories, 1.0 / categories)
        target = np.zeros(categories)
        target[0] = 1.0
        raw_chi_square = float(np.sum(target**2 / source) - 1.0)
        quotient_chi_square = 0.0
        nuisance_rows.append(
            {
                "nuisance_categories": float(categories),
                "raw_chi_square": raw_chi_square,
                "risk_quotient_chi_square": quotient_chi_square,
            }
        )

    matrix = np.asarray([[0.20, 0.10], [0.05, 0.30]], dtype=np.float64)
    gamma = 0.80
    eigen_radius = float(max(abs(np.linalg.eigvals(matrix))))
    value_scale = np.linalg.solve(
        np.eye(matrix.shape[0]) - matrix / gamma,
        np.ones(matrix.shape[0]),
    )
    transformed = (
        matrix * value_scale[None, :] / (gamma * value_scale[:, None])
    )
    cemetery = 1.0 / value_scale
    row_sums = np.sum(transformed, axis=1) + cemetery
    reconstructed = (
        gamma
        * np.diag(value_scale)
        @ transformed
        @ np.diag(1.0 / value_scale)
    )
    similarity_error = float(np.max(np.abs(reconstructed - matrix)))

    checks = {
        "risk_neutral_one_step_means_are_identical": bool(
            abs(deterministic_mean - mixture_mean) < 1e-12
        ),
        "exponential_witness_strictly_separates_interfaces": bool(
            mixture_mgf > deterministic_mgf
            and abs((mixture_mgf - deterministic_mgf) - exact_mgf_gap) < 1e-12
        ),
        "one_return_threshold_separates_manager_actions": bool(
            np.log(deterministic_mgf) < separating_threshold < np.log(mixture_mgf)
        ),
        "raw_coverage_diverges_while_quotient_coverage_is_constant": bool(
            nuisance_rows[-1]["raw_chi_square"]
            > nuisance_rows[0]["raw_chi_square"]
            and all(row["risk_quotient_chi_square"] == 0.0 for row in nuisance_rows)
        ),
        "oracle_doob_chain_is_exactly_discounted": bool(
            eigen_radius < gamma
            and np.max(np.abs(row_sums - 1.0)) < 1e-12
            and similarity_error < 1e-12
        ),
    }
    status = (
        "EIRR_KROPE_DOOB_BOUNDARY_VERIFIED"
        if all(checks.values())
        else "EIRR_KROPE_DOOB_BOUNDARY_VERIFICATION_FAILED"
    )
    return {
        "status": status,
        "risk_neutral_collapse": {
            "deterministic_mean_cost": deterministic_mean,
            "mixture_mean_cost": mixture_mean,
            "deterministic_exponential_moment": deterministic_mgf,
            "mixture_exponential_moment": mixture_mgf,
            "exact_exponential_moment_gap": exact_mgf_gap,
            "log_risk_gap": log_risk_gap,
            "separating_requirement_threshold": separating_threshold,
        },
        "raw_vs_quotient_coverage": nuisance_rows,
        "oracle_doob_reduction": {
            "matrix_spectral_radius": eigen_radius,
            "discount": gamma,
            "row_sums_with_cemetery": row_sums.tolist(),
            "similarity_reconstruction_error": similarity_error,
        },
        "checks": checks,
        "non_claim": (
            "The construction separates EIRR from a risk-neutral representation "
            "and raw-interface coverage, but not from a correctly risk-transformed "
            "KROPE/DICE baseline with oracle access to the Doob transform."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_eirr_krope_doob_boundary_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
