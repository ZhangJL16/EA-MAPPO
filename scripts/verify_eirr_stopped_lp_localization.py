#!/usr/bin/env python3
"""Verify sharp Lp-to-stopped-margin exponents in EIRR Theorem 31."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def stopped_lp_bounds(
    *, error_p_moment: float,
    moment_power: float,
    margin_exponent: float,
    margin_constant: float,
) -> tuple[float, float, float, float]:
    risk = float(error_p_moment)
    power = float(moment_power)
    kappa = float(margin_exponent)
    constant = float(margin_constant)
    if risk < 0.0 or not np.isfinite(risk):
        raise ValueError("error moment must be finite and nonnegative")
    if power <= 1.0 or kappa <= 0.0 or constant <= 0.0:
        raise ValueError("p>1, margin exponent, and margin constant are required")
    if risk == 0.0:
        return 0.0, 0.0, 0.0, 0.0
    disagreement_threshold = (
        power * risk / (kappa * constant)
    ) ** (1.0 / (power + kappa))
    disagreement_bound = (
        constant * disagreement_threshold**kappa
        + risk / disagreement_threshold**power
    )
    loss_threshold = (
        (power - 1.0) * risk / ((kappa + 1.0) * constant)
    ) ** (1.0 / (power + kappa))
    loss_bound = (
        constant * loss_threshold ** (kappa + 1.0)
        + risk / loss_threshold ** (power - 1.0)
    )
    return (
        float(disagreement_bound),
        float(loss_bound),
        float(disagreement_threshold),
        float(loss_threshold),
    )


def _log_slope(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.polyfit(np.log(x), np.log(y), 1)[0])


def build_eirr_stopped_lp_localization_report() -> dict[str, object]:
    power = 2.0
    thresholds = np.asarray([0.20, 0.10, 0.05, 0.025, 0.0125])
    rows: list[dict[str, object]] = []
    checks: dict[str, bool] = {}
    for kappa in [0.5, 1.0, 2.0]:
        moments = (
            2.0**power * kappa / (kappa + power)
        ) * thresholds ** (kappa + power)
        disagreements = thresholds**kappa
        boundary_losses = (
            kappa / (kappa + 1.0)
        ) * thresholds ** (kappa + 1.0)
        disagreement_slope = _log_slope(moments, disagreements)
        loss_slope = _log_slope(moments, boundary_losses)
        expected_disagreement_slope = kappa / (kappa + power)
        expected_loss_slope = (kappa + 1.0) / (kappa + power)

        upper_rows: list[dict[str, float]] = []
        for moment, disagreement, loss in zip(
            moments, disagreements, boundary_losses, strict=True
        ):
            disagreement_bound, loss_bound, d_threshold, l_threshold = (
                stopped_lp_bounds(
                    error_p_moment=float(moment),
                    moment_power=power,
                    margin_exponent=kappa,
                    margin_constant=1.0,
                )
            )
            upper_rows.append(
                {
                    "error_p_moment": float(moment),
                    "exact_disagreement": float(disagreement),
                    "disagreement_upper_bound": disagreement_bound,
                    "exact_boundary_loss": float(loss),
                    "boundary_loss_upper_bound": loss_bound,
                    "disagreement_split_threshold": d_threshold,
                    "loss_split_threshold": l_threshold,
                }
            )
        key = str(kappa).replace(".", "_")
        checks[f"disagreement_power_matches_kappa_{key}"] = bool(
            abs(disagreement_slope - expected_disagreement_slope) < 1e-12
        )
        checks[f"loss_power_matches_kappa_{key}"] = bool(
            abs(loss_slope - expected_loss_slope) < 1e-12
        )
        checks[f"split_bounds_cover_construction_kappa_{key}"] = bool(
            all(
                row["exact_disagreement"] <= row["disagreement_upper_bound"]
                and row["exact_boundary_loss"] <= row["boundary_loss_upper_bound"]
                for row in upper_rows
            )
        )
        rows.append(
            {
                "margin_exponent": kappa,
                "moment_power": power,
                "fitted_disagreement_power": disagreement_slope,
                "expected_disagreement_power": expected_disagreement_slope,
                "fitted_boundary_loss_power": loss_slope,
                "expected_boundary_loss_power": expected_loss_slope,
                "construction": upper_rows,
            }
        )

    status = (
        "EIRR_STOPPED_LP_LOCALIZATION_VERIFIED"
        if all(checks.values())
        else "EIRR_STOPPED_LP_LOCALIZATION_VERIFICATION_FAILED"
    )
    return {
        "status": status,
        "sharp_power_rows": rows,
        "checks": checks,
        "non_claim": (
            "The verifier checks the deterministic moment-to-decision transfer. "
            "An estimator still needs a separately valid stopped-occupation Lp "
            "risk bound; empirical Pareto improvement is not implied."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_eirr_stopped_lp_localization_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
