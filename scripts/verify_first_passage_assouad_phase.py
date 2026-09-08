#!/usr/bin/env python3
"""Verify algebra for the distributed first-passage Assouad lower slice.

The script checks the exact scalar killed-chain inverse, critical scaling, KL
order, margin-cell feasibility, and stopped/Pareto exponents.  The existence of
the standard Holder-margin bump hypercube is a mathematical assumption checked
in the proof, not something a numerical script can establish.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def killed_requirement(
    *, cost_multiplier: float, termination_probability: float, risk_lambda: float
) -> float:
    multiplier = float(cost_multiplier)
    probability = float(termination_probability)
    lam = float(risk_lambda)
    gap = 1.0 - multiplier * (1.0 - probability)
    if multiplier <= 1.0 or lam <= 0.0:
        raise ValueError("cost multiplier must exceed one and lambda must be positive")
    if probability <= 0.0 or probability >= 1.0 or gap <= 0.0:
        raise ValueError("termination probability must define a proper killed risk chain")
    return float(np.log(multiplier * probability / gap) / lam)


def termination_for_requirement(
    *, cost_multiplier: float, requirement: float, risk_lambda: float
) -> float:
    multiplier = float(cost_multiplier)
    lam = float(risk_lambda)
    value = float(np.exp(lam * float(requirement)))
    if multiplier <= 1.0 or lam <= 0.0 or value <= 1.0:
        raise ValueError("inverse requires multiplier>1, lambda>0, and exp(lambda U)>1")
    return float(value * (multiplier - 1.0) / (multiplier * (value - 1.0)))


def _bernoulli_kl(p: float, q: float) -> float:
    return float(p * np.log(p / q) + (1.0 - p) * np.log((1.0 - p) / (1.0 - q)))


def build_first_passage_assouad_report() -> dict[str, object]:
    alpha = 1.0
    dimension = 2.0
    kappa = 1.0
    multiplier = 1.5
    lam = 0.7
    critical_beta = alpha / (2.0 * alpha + dimension)
    gap_beta = 0.5 * critical_beta
    sample_sizes = np.asarray([2**power for power in range(12, 23)], dtype=np.float64)

    rows: list[dict[str, float]] = []
    for sample_size in sample_sizes:
        bandwidth = sample_size ** (-1.0 / (2.0 * alpha + dimension))
        gap = sample_size ** (-gap_beta)
        score_amplitude = 0.05 * bandwidth**alpha / gap
        base_probability = (multiplier - 1.0 + gap) / multiplier
        base_requirement = killed_requirement(
            cost_multiplier=multiplier,
            termination_probability=base_probability,
            risk_lambda=lam,
        )
        probability_continue = termination_for_requirement(
            cost_multiplier=multiplier,
            requirement=base_requirement - score_amplitude,
            risk_lambda=lam,
        )
        probability_return = termination_for_requirement(
            cost_multiplier=multiplier,
            requirement=base_requirement + score_amplitude,
            risk_lambda=lam,
        )
        recovered_continue_score = base_requirement - killed_requirement(
            cost_multiplier=multiplier,
            termination_probability=probability_continue,
            risk_lambda=lam,
        )
        recovered_return_score = base_requirement - killed_requirement(
            cost_multiplier=multiplier,
            termination_probability=probability_return,
            risk_lambda=lam,
        )
        cell_mass = bandwidth**dimension
        symmetric_kl = max(
            _bernoulli_kl(probability_continue, probability_return),
            _bernoulli_kl(probability_return, probability_continue),
        )
        flipped_bit_kl = sample_size * cell_mass * symmetric_kl
        active_mass = score_amplitude**kappa
        active_cells = active_mass / cell_mass
        available_cells = 1.0 / cell_mass
        rows.append(
            {
                "sample_size": sample_size,
                "bandwidth": bandwidth,
                "transience_gap": gap,
                "score_amplitude": score_amplitude,
                "base_termination_probability": base_probability,
                "continue_termination_probability": probability_continue,
                "return_termination_probability": probability_return,
                "recovered_continue_score": recovered_continue_score,
                "recovered_return_score": recovered_return_score,
                "primitive_probability_separation": abs(
                    probability_continue - probability_return
                ),
                "separation_over_holder_bump": abs(
                    probability_continue - probability_return
                )
                / bandwidth**alpha,
                "separation_over_gap_score": abs(
                    probability_continue - probability_return
                )
                / (gap * score_amplitude),
                "flipped_bit_n_sample_kl": flipped_bit_kl,
                "active_margin_mass": active_mass,
                "active_cell_count_continuous": active_cells,
                "available_cell_count": available_cells,
                "disagreement_lower_scale": score_amplitude**kappa,
                "boundary_loss_lower_scale": score_amplitude ** (kappa + 1.0),
                "critical_information_scale": sample_size
                * gap ** ((2.0 * alpha + dimension) / alpha),
            }
        )

    def slope(field: str) -> float:
        values = np.asarray([row[field] for row in rows], dtype=np.float64)
        return float(np.polyfit(np.log(sample_sizes), np.log(values), 1)[0])

    expected_score_slope = -critical_beta + gap_beta
    checks = {
        "exact_inverse_recovers_both_scores": bool(
            max(
                abs(row["recovered_continue_score"] - row["score_amplitude"])
                for row in rows
            )
            < 1e-12
            and max(
                abs(row["recovered_return_score"] + row["score_amplitude"])
                for row in rows
            )
            < 1e-12
        ),
        "termination_probabilities_are_proper": bool(
            all(
                0.0 < row["return_termination_probability"]
                < row["continue_termination_probability"]
                < 1.0
                for row in rows
            )
        ),
        "primitive_separation_has_holder_order": bool(
            min(row["separation_over_holder_bump"] for row in rows) > 0.02
            and max(row["separation_over_holder_bump"] for row in rows) < 0.2
            and max(row["separation_over_holder_bump"] for row in rows)
            / min(row["separation_over_holder_bump"] for row in rows)
            < 2.0
        ),
        "flipped_bit_kl_is_constant_order": bool(
            min(row["flipped_bit_n_sample_kl"] for row in rows) > 0.004
            and max(row["flipped_bit_n_sample_kl"] for row in rows) < 0.02
            and max(row["flipped_bit_n_sample_kl"] for row in rows)
            / min(row["flipped_bit_n_sample_kl"] for row in rows)
            < 3.0
        ),
        "score_is_transience_amplified": bool(
            abs(slope("score_amplitude") - expected_score_slope) < 1e-12
        ),
        "margin_cell_packing_is_feasible": bool(
            all(
                1.0 <= row["active_cell_count_continuous"]
                <= row["available_cell_count"]
                for row in rows
            )
        ),
        "disagreement_power_matches_margin": bool(
            abs(slope("disagreement_lower_scale") - kappa * expected_score_slope)
            < 1e-12
        ),
        "boundary_power_matches_margin": bool(
            abs(
                slope("boundary_loss_lower_scale")
                - (kappa + 1.0) * expected_score_slope
            )
            < 1e-12
        ),
        "subcritical_gap_schedule_has_growing_information": bool(
            slope("critical_information_scale") > 0.0
        ),
    }
    status = (
        "FIRST_PASSAGE_ASSOUAD_PHASE_VERIFIED"
        if all(checks.values())
        else "FIRST_PASSAGE_ASSOUAD_PHASE_VERIFICATION_FAILED"
    )
    return {
        "status": status,
        "parameters": {
            "holder_alpha": alpha,
            "quotient_dimension": dimension,
            "margin_exponent": kappa,
            "cost_multiplier": multiplier,
            "risk_lambda": lam,
            "gap_schedule_exponent": gap_beta,
            "critical_gap_exponent": critical_beta,
            "compatibility_alpha_kappa_le_dimension": alpha * kappa <= dimension,
        },
        "fitted_log_slopes": {
            "score_amplitude": slope("score_amplitude"),
            "primitive_probability_separation": slope(
                "primitive_probability_separation"
            ),
            "flipped_bit_n_sample_kl": slope("flipped_bit_n_sample_kl"),
            "disagreement_lower_scale": slope("disagreement_lower_scale"),
            "boundary_loss_lower_scale": slope("boundary_loss_lower_scale"),
            "critical_information_scale": slope("critical_information_scale"),
        },
        "checks": checks,
        "rows": rows,
        "scope": (
            "exact chart, source equals target, standard strong-density "
            "Holder-margin Assouad slice embedded in scalar killed first passage"
        ),
        "non_claim": (
            "This verifier checks algebra and rates; it does not prove existence "
            "of the classical bump hypercube or arbitrary covariate-shift lower bounds."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_first_passage_assouad_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
