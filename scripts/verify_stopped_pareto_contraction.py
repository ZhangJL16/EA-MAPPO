#!/usr/bin/env python3
"""Verify the contracted stopped-L2-to-Pareto certificate.

This verifier checks the algebra in Step 35 of the derivation package.  It does
not simulate a mission and does not claim an end-to-end minimax lower bound.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def stopped_pareto_certificate(
    *,
    score_mse_bound: float,
    margin_constant: float,
    margin_exponent: float,
    margin_radius: float,
    throughput_range: float,
) -> dict[str, float | bool]:
    """Optimize the valid-radius p=2 stopped-margin bounds."""

    risk = float(score_mse_bound)
    constant = float(margin_constant)
    kappa = float(margin_exponent)
    radius = float(margin_radius)
    q_range = float(throughput_range)
    values = np.asarray([risk, constant, kappa, radius, q_range], dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("certificate inputs must be finite")
    if risk < 0.0:
        raise ValueError("score MSE bound must be nonnegative")
    if constant <= 0.0 or kappa <= 0.0 or radius <= 0.0:
        raise ValueError("margin parameters must be strictly positive")
    if q_range < 0.0:
        raise ValueError("throughput range must be nonnegative")

    if risk == 0.0:
        return {
            "score_mse_bound": 0.0,
            "disagreement_threshold": 0.0,
            "boundary_loss_threshold": 0.0,
            "disagreement_bound_before_probability_clipping": 0.0,
            "first_disagreement_probability_bound": 0.0,
            "boundary_loss_bound": 0.0,
            "stranding_deviation_bound": 0.0,
            "throughput_deviation_bound": 0.0,
            "disagreement_threshold_clipped_by_margin_radius": False,
            "boundary_threshold_clipped_by_margin_radius": False,
        }

    unconstrained_d = (2.0 * risk / (kappa * constant)) ** (1.0 / (kappa + 2.0))
    unconstrained_l = (risk / ((kappa + 1.0) * constant)) ** (
        1.0 / (kappa + 2.0)
    )
    threshold_d = min(unconstrained_d, radius)
    threshold_l = min(unconstrained_l, radius)
    raw_disagreement = constant * threshold_d**kappa + risk / threshold_d**2
    disagreement = min(1.0, raw_disagreement)
    boundary_loss = constant * threshold_l ** (kappa + 1.0) + risk / threshold_l
    return {
        "score_mse_bound": risk,
        "disagreement_threshold": float(threshold_d),
        "boundary_loss_threshold": float(threshold_l),
        "disagreement_bound_before_probability_clipping": float(raw_disagreement),
        "first_disagreement_probability_bound": float(disagreement),
        "boundary_loss_bound": float(boundary_loss),
        "stranding_deviation_bound": float(disagreement),
        "throughput_deviation_bound": float(q_range * disagreement),
        "disagreement_threshold_clipped_by_margin_radius": bool(
            unconstrained_d > radius
        ),
        "boundary_threshold_clipped_by_margin_radius": bool(unconstrained_l > radius),
    }


def dual_stopped_pareto_certificate(
    *,
    score_mse_bound: float,
    uniform_score_error_bound: float,
    margin_constant: float,
    margin_exponent: float,
    margin_radius: float,
    throughput_range: float,
) -> dict[str, float | str | bool | None]:
    """Take the tighter valid integrated-risk or uniform-error route."""

    integrated = stopped_pareto_certificate(
        score_mse_bound=score_mse_bound,
        margin_constant=margin_constant,
        margin_exponent=margin_exponent,
        margin_radius=margin_radius,
        throughput_range=throughput_range,
    )
    uniform_error = float(uniform_score_error_bound)
    if not np.isfinite(uniform_error) or uniform_error < 0.0:
        raise ValueError("uniform score error bound must be finite and nonnegative")
    radius = float(margin_radius)
    constant = float(margin_constant)
    kappa = float(margin_exponent)
    q_range = float(throughput_range)
    uniform_route_valid = bool(uniform_error <= radius)
    uniform_disagreement = (
        min(1.0, constant * uniform_error**kappa) if uniform_route_valid else None
    )
    uniform_boundary_loss = (
        constant * uniform_error ** (kappa + 1.0) if uniform_route_valid else None
    )
    integrated_disagreement = float(integrated["first_disagreement_probability_bound"])
    integrated_boundary_loss = float(integrated["boundary_loss_bound"])
    disagreement = (
        min(integrated_disagreement, uniform_disagreement)
        if uniform_disagreement is not None
        else integrated_disagreement
    )
    boundary_loss = (
        min(integrated_boundary_loss, uniform_boundary_loss)
        if uniform_boundary_loss is not None
        else integrated_boundary_loss
    )
    selected = (
        "uniform_active_cell"
        if uniform_disagreement is not None
        and uniform_disagreement < integrated_disagreement
        else "integrated_stopped_l2"
    )
    return {
        "selected_disagreement_route": selected,
        "uniform_route_valid": uniform_route_valid,
        "integrated_disagreement_bound": integrated_disagreement,
        "uniform_disagreement_bound": uniform_disagreement,
        "first_disagreement_probability_bound": disagreement,
        "integrated_boundary_loss_bound": integrated_boundary_loss,
        "uniform_boundary_loss_bound": uniform_boundary_loss,
        "boundary_loss_bound": boundary_loss,
        "stranding_deviation_bound": disagreement,
        "throughput_deviation_bound": q_range * disagreement,
    }


def _log_slope(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.polyfit(np.log(x), np.log(y), 1)[0])


def build_stopped_pareto_contraction_report() -> dict[str, object]:
    risks = np.geomspace(1e-12, 1e-6, num=9)
    rows: list[dict[str, object]] = []
    checks: dict[str, bool] = {}
    for kappa in (0.5, 1.0, 2.0):
        certificates = [
            stopped_pareto_certificate(
                score_mse_bound=float(risk),
                margin_constant=1.0,
                margin_exponent=kappa,
                margin_radius=1.0,
                throughput_range=7.0,
            )
            for risk in risks
        ]
        disagreement = np.asarray(
            [row["disagreement_bound_before_probability_clipping"] for row in certificates]
        )
        boundary_loss = np.asarray([row["boundary_loss_bound"] for row in certificates])
        disagreement_slope = _log_slope(risks, disagreement)
        boundary_slope = _log_slope(risks, boundary_loss)
        expected_disagreement = kappa / (kappa + 2.0)
        expected_boundary = (kappa + 1.0) / (kappa + 2.0)
        key = str(kappa).replace(".", "_")
        checks[f"integrated_disagreement_power_kappa_{key}"] = bool(
            abs(disagreement_slope - expected_disagreement) < 1e-12
        )
        checks[f"integrated_boundary_power_kappa_{key}"] = bool(
            abs(boundary_slope - expected_boundary) < 1e-12
        )
        checks[f"pareto_rectangle_kappa_{key}"] = bool(
            all(
                np.isclose(
                    row["throughput_deviation_bound"],
                    7.0 * row["stranding_deviation_bound"],
                )
                for row in certificates
            )
        )
        rows.append(
            {
                "margin_exponent": kappa,
                "fitted_integrated_disagreement_power": disagreement_slope,
                "expected_integrated_disagreement_power": expected_disagreement,
                "fitted_integrated_boundary_power": boundary_slope,
                "expected_integrated_boundary_power": expected_boundary,
                "uniform_error_disagreement_power_as_function_of_mse": kappa / 2.0,
                "integrated_and_uniform_regimes_are_distinct": bool(
                    not np.isclose(expected_disagreement, kappa / 2.0)
                ),
            }
        )

    clipped = stopped_pareto_certificate(
        score_mse_bound=4.0,
        margin_constant=1.0,
        margin_exponent=1.0,
        margin_radius=0.1,
        throughput_range=2.0,
    )
    checks["large_risk_uses_declared_margin_radius"] = bool(
        clipped["disagreement_threshold_clipped_by_margin_radius"]
        and clipped["boundary_threshold_clipped_by_margin_radius"]
        and np.isclose(clipped["disagreement_threshold"], 0.1)
        and np.isclose(clipped["boundary_loss_threshold"], 0.1)
    )
    checks["probability_bound_is_clipped_at_one"] = bool(
        clipped["first_disagreement_probability_bound"] == 1.0
    )

    dual_uniform = dual_stopped_pareto_certificate(
        score_mse_bound=1e-2,
        uniform_score_error_bound=1e-3,
        margin_constant=1.0,
        margin_exponent=1.0,
        margin_radius=0.1,
        throughput_range=5.0,
    )
    dual_integrated = dual_stopped_pareto_certificate(
        score_mse_bound=1e-8,
        uniform_score_error_bound=0.2,
        margin_constant=1.0,
        margin_exponent=1.0,
        margin_radius=0.1,
        throughput_range=5.0,
    )
    checks["dual_certificate_selects_tighter_uniform_route"] = bool(
        dual_uniform["selected_disagreement_route"] == "uniform_active_cell"
        and dual_uniform["uniform_route_valid"]
    )
    checks["dual_certificate_rejects_out_of_radius_uniform_route"] = bool(
        dual_integrated["selected_disagreement_route"] == "integrated_stopped_l2"
        and not dual_integrated["uniform_route_valid"]
    )

    status = (
        "STOPPED_PARETO_CONTRACTION_VERIFIED"
        if all(checks.values())
        else "STOPPED_PARETO_CONTRACTION_VERIFICATION_FAILED"
    )
    return {
        "status": status,
        "information_regime_rows": rows,
        "valid_radius_stress_case": clipped,
        "dual_certificate_examples": {
            "strong_active_cell_coverage": dual_uniform,
            "weak_active_cell_coverage": dual_integrated,
        },
        "checks": checks,
        "theory_status": "COHERENT_AFTER_REFRAMING_EXTRA_ASSUMPTION",
        "proved_scope": (
            "finite-sample stopped-L2 and target-active uniform upper routes "
            "composed with risk-to-decision localization and Pareto stability"
        ),
        "minimax_audit": (
            "the stopped-L2 plus Markov route is not end-to-end minimax under "
            "strong-density Holder structure; the uniform active-cell route "
            "recovers the classical passive boundary exponent up to logarithms"
        ),
        "open_lower_bound": (
            "the repaired Assouad first-passage family closes the exact-chart "
            "source-equals-target strong-coverage phase; arbitrary pushed "
            "overlap, learned-chart error, and nuisance-bearing covariate shift "
            "still lack a joint lower bound"
        ),
        "non_claim": (
            "This algebraic verifier does not establish empirical Pareto "
            "dominance or authorize a downstream formal experiment."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_stopped_pareto_contraction_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
