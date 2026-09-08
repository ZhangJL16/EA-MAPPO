#!/usr/bin/env python3
"""Verify finite-interface EIRR transfer upper and lower bounds."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.energy_mc.eirr_transfer import (
    bernstein_interface_radii,
    compose_target_operator_radii,
    interface_transfer_minimax_lower_bound,
    resolvent_plugin_certificate,
)


def build_eirr_transfer_verification_report() -> dict[str, object]:
    variances = np.full((2, 2, 3), 0.2, dtype=np.float64)
    counts = np.full((2, 2), 1_000_000, dtype=np.int64)
    interface_radii = bernstein_interface_radii(
        sample_variances=variances,
        sample_counts=counts,
        outcome_upper_bound=4.0,
        failure_probability=0.05,
    )
    target_radii = compose_target_operator_radii(
        interface_cell_radii=interface_radii,
        target_executed_action_weights=np.asarray(
            [[0.25, 0.75], [0.60, 0.40]], dtype=np.float64
        ),
    )

    learned_matrix = np.asarray([[0.20, 0.10], [0.00, 0.30]], dtype=np.float64)
    learned_terminal = np.asarray([1.10, 1.20], dtype=np.float64)
    exact_matrix = learned_matrix + np.asarray(
        [[0.0005, -0.0004], [0.0003, -0.0002]], dtype=np.float64
    )
    exact_terminal = learned_terminal + np.asarray([0.0004, -0.0005])
    plugin = resolvent_plugin_certificate(
        learned_transient_operator=learned_matrix,
        learned_terminal_vector=learned_terminal,
        transient_operator_radius_inf=(
            target_radii.transient_operator_radius_inf
        ),
        terminal_vector_radius_inf=target_radii.terminal_vector_radius_inf,
    )
    exact_value = np.linalg.solve(np.eye(2) - exact_matrix, exact_terminal)
    learned_value = np.asarray(plugin.learned_value)
    exact_value_error = float(np.linalg.norm(exact_value - learned_value, ord=np.inf))

    high_resource = float(np.log(20.0))
    lower = interface_transfer_minimax_lower_bound(
        risk_parameter=1.0,
        high_resource=high_resource,
        source_interface_probability=0.10,
        number_of_source_transitions=50,
    )
    exponential_weight = float(np.exp(high_resource))
    rare = lower.rare_event_probability
    weighted_outcome_variance = float(
        rare * (1.0 - rare) * (exponential_weight - 1.0) ** 2
    )
    return {
        "status": "EIRR_TRANSFER_BOUNDS_VERIFIED",
        "interface_radius_max": float(np.max(interface_radii)),
        "target_operator_radius": target_radii.as_dict(),
        "plugin_certificate": plugin.as_dict(),
        "exact_value": exact_value.tolist(),
        "exact_value_error_inf": exact_value_error,
        "upper_bound_holds": bool(exact_value_error <= plugin.value_error_bound_inf),
        "lower_bound": lower.as_dict(),
        "constant_error_floor_9_over_32_holds": bool(
            lower.minimax_absolute_error_lower >= 9.0 / 32.0
        ),
        "decision_error_floor_3_over_8_holds": bool(
            lower.minimax_decision_error_lower >= 3.0 / 8.0
        ),
        "exponential_weight": exponential_weight,
        "rare_weighted_outcome_variance": weighted_outcome_variance,
        "variance_is_linear_order_in_exponential_weight": bool(
            weighted_outcome_variance <= exponential_weight
            and weighted_outcome_variance >= 0.5 * exponential_weight
        ),
        "non_claim": (
            "The verifier checks finite tabular concentration/perturbation and "
            "two-model algebra; it does not certify neural nuisance estimation."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_eirr_transfer_verification_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
