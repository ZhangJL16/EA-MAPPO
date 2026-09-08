#!/usr/bin/env python3
"""Verify that an unknown Doob transform is an exact reparameterization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def doob_transform(matrix: np.ndarray, discount: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix = np.asarray(matrix, dtype=np.float64)
    gamma = float(discount)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be nonempty and square")
    if np.any(matrix < 0.0) or not np.all(np.isfinite(matrix)):
        raise ValueError("matrix must be finite and nonnegative")
    spectral_radius = float(max(abs(np.linalg.eigvals(matrix))))
    if not spectral_radius < gamma < 1.0:
        raise ValueError("discount must lie strictly between the spectral radius and one")
    value_scale = np.linalg.solve(
        np.eye(matrix.shape[0]) - matrix / gamma,
        np.ones(matrix.shape[0]),
    )
    transformed = (
        matrix * value_scale[None, :] / (gamma * value_scale[:, None])
    )
    cemetery = 1.0 / value_scale
    return value_scale, transformed, cemetery


def inverse_doob_transform(
    *, value_scale: np.ndarray, transformed: np.ndarray, discount: float
) -> np.ndarray:
    scale = np.asarray(value_scale, dtype=np.float64)
    kernel = np.asarray(transformed, dtype=np.float64)
    gamma = float(discount)
    if scale.ndim != 1 or kernel.shape != (scale.size, scale.size):
        raise ValueError("scale and transformed kernel must align")
    if np.any(scale <= 0.0) or not np.all(np.isfinite(scale)):
        raise ValueError("scale must be finite and strictly positive")
    return gamma * scale[:, None] * kernel / scale[None, :]


def build_eirr_unknown_doob_reparameterization_report() -> dict[str, object]:
    rng = np.random.default_rng(29082026)
    matrix = rng.uniform(0.01, 0.12, size=(4, 4))
    matrix *= 0.55 / np.max(np.sum(matrix, axis=1))
    gamma = 0.90
    scale, transformed, cemetery = doob_transform(matrix, gamma)
    reconstructed = inverse_doob_transform(
        value_scale=scale, transformed=transformed, discount=gamma
    )

    terminal = rng.uniform(0.1, 0.5, size=matrix.shape[0])
    direct_value = np.linalg.solve(np.eye(matrix.shape[0]) - matrix, terminal)
    transformed_value = scale * np.linalg.solve(
        np.eye(matrix.shape[0]) - gamma * transformed,
        terminal / scale,
    )

    perturbation = rng.uniform(-0.002, 0.002, size=matrix.shape)
    learned_matrix = np.maximum(matrix + perturbation, 0.0)
    learned_scale, learned_transformed, _ = doob_transform(learned_matrix, gamma)
    direct_plugin = np.linalg.solve(
        np.eye(matrix.shape[0]) - learned_matrix, terminal
    )
    transformed_plugin = learned_scale * np.linalg.solve(
        np.eye(matrix.shape[0]) - gamma * learned_transformed,
        terminal / learned_scale,
    )

    scalar_rows: list[dict[str, float]] = []
    for transient_margin in [0.20, 0.10, 0.05, 0.025]:
        scalar_gamma = 1.0 - transient_margin / 2.0
        scalar_matrix = 1.0 - transient_margin
        slack = (scalar_gamma - scalar_matrix) / scalar_gamma
        finite_difference = 1e-6 * (scalar_gamma - scalar_matrix)
        scale_zero = 1.0 / slack
        scale_one = 1.0 / (slack - finite_difference / scalar_gamma)
        absolute_derivative = (scale_one - scale_zero) / finite_difference
        log_derivative = (np.log(scale_one) - np.log(scale_zero)) / finite_difference
        scalar_rows.append(
            {
                "transient_margin": transient_margin,
                "transform_discount": scalar_gamma,
                "transform_slack": slack,
                "normalized_absolute_derivative": float(
                    absolute_derivative * scalar_gamma * slack**2
                ),
                "normalized_relative_derivative": float(
                    log_derivative * scalar_gamma * slack
                ),
                "slack_to_transient_margin_ratio": float(
                    (scalar_gamma - scalar_matrix) / transient_margin
                ),
            }
        )

    inverse_error = float(np.max(np.abs(reconstructed - matrix)))
    row_sum_error = float(
        np.max(np.abs(np.sum(transformed, axis=1) + cemetery - 1.0))
    )
    value_identity_error = float(np.max(np.abs(direct_value - transformed_value)))
    plugin_identity_error = float(np.max(np.abs(direct_plugin - transformed_plugin)))
    checks = {
        "doob_map_has_exact_inverse": bool(inverse_error < 1e-12),
        "transformed_kernel_with_cemetery_is_markov": bool(row_sum_error < 1e-12),
        "direct_and_transformed_values_are_identical": bool(
            value_identity_error < 1e-12
        ),
        "direct_and_transformed_plugins_are_identical": bool(
            plugin_identity_error < 1e-12
        ),
        "scalar_absolute_condition_is_inverse_slack_squared": bool(
            max(
                abs(row["normalized_absolute_derivative"] - 1.0)
                for row in scalar_rows
            )
            < 2e-6
        ),
        "scalar_relative_condition_is_inverse_slack": bool(
            max(
                abs(row["normalized_relative_derivative"] - 1.0)
                for row in scalar_rows
            )
            < 2e-6
        ),
        "chosen_transform_slack_tracks_transience": bool(
            all(
                abs(row["slack_to_transient_margin_ratio"] - 0.5) < 1e-12
                for row in scalar_rows
            )
        ),
    }
    status = (
        "EIRR_UNKNOWN_DOOB_REPARAMETERIZATION_VERIFIED"
        if all(checks.values())
        else "EIRR_UNKNOWN_DOOB_REPARAMETERIZATION_VERIFICATION_FAILED"
    )
    return {
        "status": status,
        "finite_matrix_identity": {
            "inverse_error": inverse_error,
            "row_sum_error": row_sum_error,
            "value_identity_error": value_identity_error,
            "plugin_identity_error": plugin_identity_error,
        },
        "scalar_conditioning": scalar_rows,
        "checks": checks,
        "non_claim": (
            "The exact reparameterization does not prohibit useful transformed "
            "regularizers or computation. It proves only that the coordinate "
            "change itself cannot improve statistical information or minimax rate."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_eirr_unknown_doob_reparameterization_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
