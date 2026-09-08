#!/usr/bin/env python3
"""Audit empirical risk-tilted path and pre-hitting occupation concentration.

This is a mechanism diagnostic, not a formal Gate and not an estimator of a
deployment guarantee.  It consumes completed trajectory files produced by the
frozen Energy collection stage and reports how exponential resource tilting
changes effective sample size and executed-interface statistics.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class TrajectoryRecord:
    trajectory_id: int
    file: str
    total_energy: float
    num_transitions: int
    total_safety_burden: float
    initial_distance: float
    goal_type: str
    distance_bucket: str


def _normalized_weights(log_weights: np.ndarray) -> np.ndarray:
    values = np.asarray(log_weights, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("log weights must be a finite nonempty vector")
    shifted = values - float(np.max(values))
    weights = np.exp(shifted)
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("risk weights could not be normalized")
    return weights / total


def _effective_sample_size(weights: np.ndarray) -> float:
    values = np.asarray(weights, dtype=np.float64)
    return float(1.0 / np.sum(np.square(values)))


def _top_fraction_mass(weights: np.ndarray, fraction: float) -> float:
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must lie in (0, 1]")
    count = max(1, int(math.ceil(float(weights.size) * fraction)))
    return float(np.sum(np.sort(weights)[-count:]))


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    left_values = np.asarray(left, dtype=np.float64)
    right_values = np.asarray(right, dtype=np.float64)
    if left_values.shape != right_values.shape or left_values.ndim != 1:
        raise ValueError("correlation inputs must be aligned vectors")
    if left_values.size < 2:
        return None
    if np.std(left_values) <= 0.0 or np.std(right_values) <= 0.0:
        return None
    value = float(np.corrcoef(left_values, right_values)[0, 1])
    return value if np.isfinite(value) else None


def load_manifest(trajectory_dir: Path) -> list[TrajectoryRecord]:
    manifest_path = trajectory_dir / "manifest.jsonl"
    records: list[TrajectoryRecord] = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        records.append(
            TrajectoryRecord(
                trajectory_id=int(item["trajectory_id"]),
                file=str(item["file"]),
                total_energy=float(item["total_realized_energy"]),
                num_transitions=int(item["num_transitions"]),
                total_safety_burden=float(item["total_safety_burden"]),
                initial_distance=float(item["initial_distance"]),
                goal_type=str(item["goal_type"]),
                distance_bucket=str(item["distance_bucket"]),
            )
        )
    if not records:
        raise ValueError(f"no trajectory records in {manifest_path}")
    return records


def analyze_trajectory_directory(
    trajectory_dir: Path,
    *,
    betas: Iterable[float],
) -> dict[str, object]:
    records = load_manifest(trajectory_dir)
    energies = np.asarray([item.total_energy for item in records], dtype=np.float64)
    horizons = np.asarray([item.num_transitions for item in records], dtype=np.float64)
    burdens = np.asarray(
        [item.total_safety_burden for item in records], dtype=np.float64
    )
    energy_scale = float(np.std(energies, ddof=1))
    if not np.isfinite(energy_scale) or energy_scale <= 0.0:
        raise ValueError("trajectory energy must have positive finite sample deviation")

    prefix_energy_parts: list[np.ndarray] = []
    step_energy_parts: list[np.ndarray] = []
    intervention_parts: list[np.ndarray] = []
    remaining_energy_parts: list[np.ndarray] = []
    trajectory_index_parts: list[np.ndarray] = []
    trajectory_mean_interventions: list[float] = []
    maximum_energy_sum_error = 0.0

    for index, record in enumerate(records):
        path = trajectory_dir / record.file
        with np.load(path, allow_pickle=False) as payload:
            step_energy = np.asarray(payload["step_energy"], dtype=np.float64)
            nominal = np.asarray(payload["nominal_actions"], dtype=np.float64)
            executed = np.asarray(payload["executed_actions"], dtype=np.float64)
            remaining = np.asarray(payload["energy_to_go"], dtype=np.float64)
        if not (
            step_energy.ndim == 1
            and nominal.shape == executed.shape == (step_energy.size, 3)
            and remaining.shape == step_energy.shape
            and step_energy.size == record.num_transitions
        ):
            raise ValueError(f"trajectory contract mismatch: {path}")
        maximum_energy_sum_error = max(
            maximum_energy_sum_error,
            abs(float(np.sum(step_energy)) - record.total_energy),
        )
        prefix = np.cumsum(step_energy) - step_energy
        prefix_energy_parts.append(prefix)
        step_energy_parts.append(step_energy)
        intervention_parts.append(np.linalg.norm(executed - nominal, axis=1))
        trajectory_mean_interventions.append(
            float(np.mean(np.linalg.norm(executed - nominal, axis=1)))
        )
        remaining_energy_parts.append(remaining)
        trajectory_index_parts.append(
            np.full(step_energy.size, index, dtype=np.int64)
        )

    prefix_energy = np.concatenate(prefix_energy_parts)
    step_energy = np.concatenate(step_energy_parts)
    intervention = np.concatenate(intervention_parts)
    remaining_energy = np.concatenate(remaining_energy_parts)
    trajectory_index = np.concatenate(trajectory_index_parts)
    mean_interventions = np.asarray(
        trajectory_mean_interventions, dtype=np.float64
    )
    num_visits = int(prefix_energy.size)

    horizon_design = np.column_stack([np.ones_like(horizons), horizons])
    energy_horizon_fit = horizon_design @ np.linalg.lstsq(
        horizon_design, energies, rcond=None
    )[0]
    intervention_horizon_fit = horizon_design @ np.linalg.lstsq(
        horizon_design, mean_interventions, rcond=None
    )[0]
    energy_residual = energies - energy_horizon_fit
    intervention_residual = mean_interventions - intervention_horizon_fit
    energy_total_sum_squares = float(np.sum(np.square(energies - np.mean(energies))))
    energy_horizon_r2 = 1.0 - float(np.sum(np.square(energy_residual))) / max(
        energy_total_sum_squares,
        np.finfo(np.float64).eps,
    )
    energy_per_step = energies / horizons
    burden_per_step = burdens / horizons

    matched_strata: list[dict[str, object]] = []
    sorted_indices = np.argsort(horizons)
    num_horizon_strata = min(3, max(1, len(records) // 2))
    horizon_labels = {
        1: ["all"],
        2: ["short", "long"],
        3: ["short", "medium", "long"],
    }[num_horizon_strata]
    for stratum_index, stratum_indices in enumerate(
        np.array_split(sorted_indices, num_horizon_strata)
    ):
        intervention_order = stratum_indices[
            np.argsort(mean_interventions[stratum_indices])
        ]
        split_index = int(math.ceil(intervention_order.size / 2.0))
        for label, selected in (
            ("low_intervention", intervention_order[:split_index]),
            ("high_intervention", intervention_order[split_index:]),
        ):
            if selected.size == 0:
                continue
            matched_strata.append(
                {
                    "horizon_stratum": horizon_labels[stratum_index],
                    "intervention_stratum": label,
                    "num_trajectories": int(selected.size),
                    "horizon_min": float(np.min(horizons[selected])),
                    "horizon_max": float(np.max(horizons[selected])),
                    "mean_horizon": float(np.mean(horizons[selected])),
                    "mean_intervention_norm": float(
                        np.mean(mean_interventions[selected])
                    ),
                    "mean_total_energy": float(np.mean(energies[selected])),
                    "mean_energy_per_step": float(np.mean(energy_per_step[selected])),
                    "mean_horizon_residual_energy": float(
                        np.mean(energy_residual[selected])
                    ),
                }
            )

    rows: list[dict[str, float]] = []
    for beta in betas:
        beta_value = float(beta)
        if not np.isfinite(beta_value) or beta_value < 0.0:
            raise ValueError("betas must be finite and nonnegative")
        risk_lambda = beta_value / energy_scale

        terminal_weights = _normalized_weights(risk_lambda * energies)
        visit_weights = _normalized_weights(risk_lambda * prefix_energy)
        per_trajectory_visit_mass = np.bincount(
            trajectory_index,
            weights=visit_weights,
            minlength=len(records),
        )

        rows.append(
            {
                "beta": beta_value,
                "lambda_per_energy_unit": risk_lambda,
                "terminal_ess": _effective_sample_size(terminal_weights),
                "terminal_ess_fraction": _effective_sample_size(terminal_weights)
                / len(records),
                "terminal_max_path_mass": float(np.max(terminal_weights)),
                "terminal_top_1pct_path_mass": _top_fraction_mass(
                    terminal_weights, 0.01
                ),
                "terminal_top_5pct_path_mass": _top_fraction_mass(
                    terminal_weights, 0.05
                ),
                "terminal_top_10pct_path_mass": _top_fraction_mass(
                    terminal_weights, 0.10
                ),
                "terminal_weighted_energy": float(
                    np.sum(terminal_weights * energies)
                ),
                "terminal_weighted_horizon": float(
                    np.sum(terminal_weights * horizons)
                ),
                "terminal_weighted_safety_burden": float(
                    np.sum(terminal_weights * burdens)
                ),
                "visit_ess": _effective_sample_size(visit_weights),
                "visit_ess_fraction": _effective_sample_size(visit_weights)
                / num_visits,
                "visit_top_1pct_path_mass": _top_fraction_mass(
                    per_trajectory_visit_mass, 0.01
                ),
                "visit_top_5pct_path_mass": _top_fraction_mass(
                    per_trajectory_visit_mass, 0.05
                ),
                "visit_top_10pct_path_mass": _top_fraction_mass(
                    per_trajectory_visit_mass, 0.10
                ),
                "visit_weighted_step_energy": float(
                    np.sum(visit_weights * step_energy)
                ),
                "visit_weighted_intervention_norm": float(
                    np.sum(visit_weights * intervention)
                ),
                "visit_weighted_remaining_energy": float(
                    np.sum(visit_weights * remaining_energy)
                ),
            }
        )

    return {
        "status": "EXPLORATORY_MECHANISM_AUDIT",
        "formal_gate_evidence": False,
        "interpretation": (
            "Empirical concentration audit for the risk-tilted path and killed "
            "pre-hitting occupation measures. It neither establishes exponential "
            "transience nor certifies deployment tail coverage."
        ),
        "trajectory_dir": str(trajectory_dir.resolve()),
        "num_trajectories": len(records),
        "num_executed_interface_visits": num_visits,
        "energy_scale_sample_std": energy_scale,
        "maximum_manifest_energy_sum_error": maximum_energy_sum_error,
        "ordinary_summary": {
            "energy_mean": float(np.mean(energies)),
            "energy_std": energy_scale,
            "energy_q50": float(np.quantile(energies, 0.50)),
            "energy_q90": float(np.quantile(energies, 0.90)),
            "energy_q95": float(np.quantile(energies, 0.95)),
            "energy_q99": float(np.quantile(energies, 0.99)),
            "energy_max": float(np.max(energies)),
            "horizon_mean": float(np.mean(horizons)),
            "horizon_q95": float(np.quantile(horizons, 0.95)),
            "energy_horizon_correlation": _safe_correlation(energies, horizons),
            "energy_safety_burden_correlation": _safe_correlation(
                energies, burdens
            ),
            "visit_mean_step_energy": float(np.mean(step_energy)),
            "visit_mean_intervention_norm": float(np.mean(intervention)),
            "visit_mean_remaining_energy": float(np.mean(remaining_energy)),
        },
        "horizon_adjusted_summary": {
            "linear_horizon_r_squared": energy_horizon_r2,
            "partial_energy_intervention_correlation_given_linear_horizon": (
                _safe_correlation(energy_residual, intervention_residual)
            ),
            "energy_rate_intervention_correlation": _safe_correlation(
                energy_per_step, mean_interventions
            ),
            "energy_rate_safety_burden_rate_correlation": _safe_correlation(
                energy_per_step, burden_per_step
            ),
            "matched_horizon_intervention_strata": matched_strata,
        },
        "risk_tilt_rows": rows,
    }


def render_markdown(report: dict[str, object]) -> str:
    ordinary = report["ordinary_summary"]
    adjusted = report["horizon_adjusted_summary"]
    rows = report["risk_tilt_rows"]
    lines = [
        "# R3 Exploratory Risk-Tilted Occupancy Audit",
        "",
        "> This is an exploratory mechanism audit, not formal Gate evidence.",
        "",
        f"- Complete trajectories: {report['num_trajectories']}",
        f"- Executed-interface visits: {report['num_executed_interface_visits']}",
        f"- Mean / q95 / max total energy: {ordinary['energy_mean']:.4f} / "
        f"{ordinary['energy_q95']:.4f} / {ordinary['energy_max']:.4f}",
        f"- Energy--horizon correlation: {ordinary['energy_horizon_correlation']}",
        f"- Energy--safety-burden correlation: "
        f"{ordinary['energy_safety_burden_correlation']}",
        f"- Linear horizon R-squared for total energy: "
        f"{adjusted['linear_horizon_r_squared']:.4f}",
        f"- Partial energy--intervention correlation given linear horizon: "
        f"{adjusted['partial_energy_intervention_correlation_given_linear_horizon']}",
        f"- Energy-rate--intervention correlation: "
        f"{adjusted['energy_rate_intervention_correlation']}",
        "",
        "Here `beta = lambda * sample_std(total_energy)`. ESS is effective sample "
        "size after normalizing the empirical exponential weights.",
        "",
        "| beta | lambda | terminal ESS frac. | terminal top-5% mass | visit ESS frac. | visit top-5% path mass | tilted energy | tilted horizon | tilted intervention |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {beta:.2f} | {lambda_per_energy_unit:.6f} | "
            "{terminal_ess_fraction:.4f} | {terminal_top_5pct_path_mass:.4f} | "
            "{visit_ess_fraction:.4f} | {visit_top_5pct_path_mass:.4f} | "
            "{terminal_weighted_energy:.4f} | {terminal_weighted_horizon:.2f} | "
            "{visit_weighted_intervention_norm:.4f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Horizon-matched intervention diagnostic",
            "",
            "Intervention is an intensity diagnostic inside one composition; it is "
            "not an interface-coverage label.",
            "",
            "| horizon | intervention | paths | mean horizon | mean intervention | mean energy/step | mean horizon-residual energy |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in adjusted["matched_horizon_intervention_strata"]:
        lines.append(
            "| {horizon_stratum} | {intervention_stratum} | {num_trajectories} | "
            "{mean_horizon:.2f} | {mean_intervention_norm:.4f} | "
            "{mean_energy_per_step:.6f} | {mean_horizon_residual_energy:.4f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation rule",
            "",
            "A sharp ESS collapse or large top-path mass supports the mechanism "
            "hypothesis that ordinary coverage is not the relevant tail coverage. "
            "It does not prove source-to-target non-identifiability, a finite-sample "
            "generalization bound, or an Oracle/Pareto improvement.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--betas",
        type=float,
        nargs="+",
        default=[0.0, 0.25, 0.5, 1.0, 2.0],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = analyze_trajectory_directory(args.trajectory_dir, betas=args.betas)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "risk_tilted_occupancy_audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "risk_tilted_occupancy_audit.md").write_text(
        render_markdown(report),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
