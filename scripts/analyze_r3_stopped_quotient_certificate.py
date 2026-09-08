#!/usr/bin/env python3
"""Compute exploratory Theorem 32/33 proxy quantities on the frozen R3 data."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def pushed_forward_overlap(
    source_labels: np.ndarray,
    target_labels: np.ndarray,
    target_weights: np.ndarray,
) -> float:
    """Compute sum_j q_j/p_j after pushforward to declared chart cells."""
    source = np.asarray(source_labels)
    target = np.asarray(target_labels)
    weights = np.asarray(target_weights, dtype=float)
    if target.shape != weights.shape:
        raise ValueError("target labels and weights must have equal shape")
    if source.ndim != 1 or target.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    if np.any(weights < 0.0) or not np.isfinite(weights).all() or weights.sum() <= 0.0:
        raise ValueError("target weights must be finite, nonnegative, and nonzero")
    weights = weights / weights.sum()
    source_count = Counter(source.tolist())
    target_mass: dict[Any, float] = defaultdict(float)
    for label, weight in zip(target.tolist(), weights.tolist(), strict=True):
        target_mass[label] += weight
    coefficient = 0.0
    for label, q_mass in target_mass.items():
        p_mass = source_count.get(label, 0) / len(source)
        if q_mass > 0.0 and p_mass == 0.0:
            return float("inf")
        coefficient += q_mass / p_mass
    return float(coefficient)


def partition_diameter_from_atoms(
    atom_means: dict[str, np.ndarray], atom_to_chart: dict[str, str]
) -> float:
    """Maximum L-infinity witness distance between atoms merged by a chart."""
    grouped: dict[str, list[np.ndarray]] = defaultdict(list)
    for atom, mean in atom_means.items():
        grouped[atom_to_chart[atom]].append(np.asarray(mean, dtype=float))
    diameter = 0.0
    for means in grouped.values():
        for left in means:
            for right in means:
                diameter = max(diameter, float(np.max(np.abs(left - right))))
    return diameter


def _stable_weights(total_energy: np.ndarray, beta: float, scale: float) -> np.ndarray:
    if beta == 0.0:
        return np.ones_like(total_energy, dtype=float)
    log_weight = beta * total_energy / scale
    log_weight -= np.max(log_weight)
    return np.exp(log_weight)


def _effective_sample_size(weights: np.ndarray) -> float:
    normalized = weights / weights.sum()
    return float(1.0 / np.sum(normalized**2))


def _load_rows(trajectory_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    manifest = trajectory_dir / "manifest.jsonl"
    for line in manifest.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        with np.load(trajectory_dir / item["file"], allow_pickle=False) as data:
            intervention = np.linalg.norm(
                data["executed_actions"] - data["nominal_actions"], axis=1
            )
        rows.append(
            {
                "trajectory_id": int(item["trajectory_id"]),
                "goal_type": str(item["goal_type"]),
                "horizon": int(item["num_transitions"]),
                "initial_distance": float(item["initial_distance"]),
                "total_energy": float(item["total_realized_energy"]),
                "intervention_mean": float(np.mean(intervention)),
            }
        )
    return sorted(rows, key=lambda row: row["trajectory_id"])


def _fit_witnesses(rows: list[dict[str, Any]], structure_end: int) -> np.ndarray:
    horizon = np.asarray([row["horizon"] for row in rows], dtype=float)
    distance = np.asarray([row["initial_distance"] for row in rows], dtype=float)
    charger = np.asarray([row["goal_type"] == "CHARGER" for row in rows], dtype=float)
    energy = np.asarray([row["total_energy"] for row in rows], dtype=float)
    design = np.column_stack([np.ones(len(rows)), horizon, distance, charger])
    coefficient, *_ = np.linalg.lstsq(
        design[:structure_end], energy[:structure_end], rcond=None
    )
    residual = energy - design @ coefficient
    energy_center = float(np.median(energy[:structure_end]))
    energy_scale = max(float(np.max(np.abs(energy[:structure_end] - energy_center))), 1e-12)
    residual_scale = max(float(np.max(np.abs(residual[:structure_end]))), 1e-12)
    energy_scaled = np.clip((energy - energy_center) / energy_scale, -1.0, 1.0)
    residual_scaled = np.clip(residual / residual_scale, -1.0, 1.0)
    exponential_residual = np.exp(residual_scaled - 1.0)
    return np.column_stack(
        [energy_scaled, residual_scaled, residual_scaled**2, exponential_residual]
    )


def _atom_label(row: dict[str, Any], h_cut: float, i_cut: float, with_goal: bool) -> str:
    parts = [f"h{int(row['horizon'] > h_cut)}", f"i{int(row['intervention_mean'] > i_cut)}"]
    if with_goal:
        parts.append("gC" if row["goal_type"] == "CHARGER" else "gT")
    return "_".join(parts)


def _chart_label(atom: str, chart: str) -> str:
    parts = atom.split("_")
    h = parts[0]
    i = parts[1]
    goal = parts[2] if len(parts) == 3 else None
    if chart == "raw":
        return atom
    if chart == "drop_goal":
        return f"{h}_{i}"
    if chart == "horizon_goal":
        return f"{h}_{goal}"
    if chart == "horizon":
        return h
    if chart == "intervention":
        return i
    if chart == "goal":
        return str(goal)
    if chart == "collapsed":
        return "all"
    raise ValueError(f"unknown chart {chart}")


def _family_report(
    *,
    rows: list[dict[str, Any]],
    witnesses: np.ndarray,
    structure_slice: slice,
    source_slice: slice,
    target_slice: slice,
    h_cut: float,
    i_cut: float,
    with_goal: bool,
    charts: list[str],
    betas: list[float],
    delta: float,
) -> dict[str, Any]:
    atoms = np.asarray(
        [_atom_label(row, h_cut, i_cut, with_goal) for row in rows], dtype=object
    )
    structure_atoms = atoms[structure_slice]
    structure_witnesses = witnesses[structure_slice]
    atom_counts = Counter(structure_atoms.tolist())
    atom_means = {
        atom: np.mean(structure_witnesses[structure_atoms == atom], axis=0)
        for atom in sorted(atom_counts)
    }
    n_atoms = len(atom_means)
    witness_dim = witnesses.shape[1]
    min_count = min(atom_counts.values())
    radius = float(
        np.sqrt(2.0 * np.log(2.0 * n_atoms * witness_dim / delta) / min_count)
    )
    maximum_diameter = 2.0
    source_atoms = atoms[source_slice]
    target_atoms = atoms[target_slice]
    source_n = len(source_atoms)
    target_energy = np.asarray(
        [row["total_energy"] for row in rows[target_slice]], dtype=float
    )
    energy_scale = float(np.std([row["total_energy"] for row in rows], ddof=1))

    chart_rows: list[dict[str, Any]] = []
    for chart in charts:
        mapping = {atom: _chart_label(atom, chart) for atom in atom_means}
        empirical_diameter = partition_diameter_from_atoms(atom_means, mapping)
        upper_diameter = min(maximum_diameter, empirical_diameter + 2.0 * radius)
        source_labels = np.asarray(
            [_chart_label(atom, chart) for atom in source_atoms], dtype=object
        )
        target_labels = np.asarray(
            [_chart_label(atom, chart) for atom in target_atoms], dtype=object
        )
        beta_rows: list[dict[str, Any]] = []
        for beta in betas:
            weights = _stable_weights(target_energy, beta, energy_scale)
            coefficient = pushed_forward_overlap(source_labels, target_labels, weights)
            active_cells = len(set(target_labels.tolist()))
            variance_proxy = (
                float("inf")
                if not np.isfinite(coefficient)
                else 12.0
                * np.log(4.0 * active_cells / delta)
                * coefficient
                / source_n
            )
            beta_rows.append(
                {
                    "beta": beta,
                    "target_ess": _effective_sample_size(weights),
                    "target_ess_fraction": _effective_sample_size(weights) / len(weights),
                    "pushed_forward_overlap": coefficient,
                    "theorem32_stochastic_proxy": variance_proxy,
                    "partial_objective_excluding_nuisance_and_transience": (
                        upper_diameter**2 + variance_proxy
                    ),
                    "target_active_cells": active_cells,
                }
            )
        chart_rows.append(
            {
                "chart": chart,
                "num_chart_cells_structure": len(set(mapping.values())),
                "empirical_witness_diameter": empirical_diameter,
                "upper_witness_diameter": upper_diameter,
                "risk_tilt_rows": beta_rows,
            }
        )
    return {
        "with_goal_in_base_atoms": with_goal,
        "num_base_atoms": n_atoms,
        "base_atom_counts": dict(sorted(atom_counts.items())),
        "minimum_structure_atom_count": min_count,
        "uniform_witness_radius": radius,
        "diameter_uncertainty_fraction": min(1.0, radius),
        "certificate_width_status": (
            "VACUOUS_AT_MAX_DIAMETER" if radius >= 1.0 else "WIDE_EXPLORATORY"
        ),
        "charts": chart_rows,
    }


def build_r3_stopped_quotient_proxy(trajectory_dir: Path) -> dict[str, Any]:
    rows = _load_rows(trajectory_dir)
    if len(rows) != 930:
        raise ValueError(f"expected 930 complete R3 trajectories, found {len(rows)}")
    structure_end = 310
    source_end = 620
    witnesses = _fit_witnesses(rows, structure_end)
    h_cut = float(np.median([row["horizon"] for row in rows[:structure_end]]))
    i_cut = float(
        np.median([row["intervention_mean"] for row in rows[:structure_end]])
    )
    common = dict(
        rows=rows,
        witnesses=witnesses,
        structure_slice=slice(0, structure_end),
        source_slice=slice(structure_end, source_end),
        target_slice=slice(source_end, len(rows)),
        h_cut=h_cut,
        i_cut=i_cut,
        betas=[0.0, 0.25, 0.5, 1.0],
        delta=0.05,
    )
    interface_family = _family_report(
        **common,
        with_goal=False,
        charts=["raw", "horizon", "intervention", "collapsed"],
    )
    goal_family = _family_report(
        **common,
        with_goal=True,
        charts=["raw", "drop_goal", "horizon_goal", "horizon", "goal", "collapsed"],
    )
    return {
        "status": "EXPLORATORY_RISK_TILTED_PROXY_ONLY",
        "formal_gate_evidence": False,
        "statistical_unit": "complete_trajectory",
        "num_trajectories": len(rows),
        "split": {
            "structure": [0, structure_end - 1],
            "ordinary_source": [structure_end, source_end - 1],
            "risk_tilted_target": [source_end, len(rows) - 1],
            "scheme": "three contiguous trajectory-id folds",
        },
        "structure_thresholds": {
            "horizon_median": h_cut,
            "mean_intervention_median": i_cut,
        },
        "witness_definition": [
            "clipped structure-normalized total Energy",
            "clipped residual from structure-fold Energy~horizon+initial-distance+goal-type",
            "squared clipped residual",
            "exp(clipped residual-1)",
        ],
        "target_law": (
            "terminal exponential Energy tilt within the target fold; this is "
            "not the oracle-shadow stopped-decision occupation law"
        ),
        "interface_family": interface_family,
        "goal_sensitivity_family": goal_family,
        "gate_verdict": {
            "probability_semantics": "MECHANISM_PROXY_ONLY",
            "oracle_decision_headroom": "NOT_EVALUABLE",
            "stranding_throughput_pareto": "NOT_EVALUABLE",
            "reason": (
                "R3 has no formal battery-cycle manager decisions or paired "
                "Oracle counterfactuals, and the chart certificate is wide or vacuous."
            ),
        },
        "non_claim": (
            "Post-trajectory horizons, residuals, and intervention summaries are "
            "used only to audit mechanism and coverage. They are not deployable "
            "features, independent visit samples, stopped-law certificates, "
            "Oracle headroom, or Pareto evidence."
        ),
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# R3 Stopped-Quotient Certificate Proxy",
        "",
        f"Status: `{report['status']}`",
        "",
        "The statistical unit is one complete trajectory. The target law is a",
        "terminal exponential-Energy tilt, not the ReturnManager stopped law.",
        "Consequently every number below is exploratory mechanism evidence.",
        "",
        "| Family | Base atoms | Minimum structure count | Witness radius | Width status |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for name, key in [
        ("horizon × intervention", "interface_family"),
        ("horizon × intervention × goal", "goal_sensitivity_family"),
    ]:
        family = report[key]
        lines.append(
            f"| {name} | {family['num_base_atoms']} | "
            f"{family['minimum_structure_atom_count']} | "
            f"{family['uniform_witness_radius']:.6f} | "
            f"`{family['certificate_width_status']}` |"
        )
    lines.extend(
        [
            "",
            "## Chart diagnostics",
            "",
            "| Family | Chart | Empirical diameter | Upper diameter | beta | Target ESS fraction | Pushed overlap | Partial objective |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for family_name, key in [
        ("interface", "interface_family"),
        ("goal sensitivity", "goal_sensitivity_family"),
    ]:
        for chart in report[key]["charts"]:
            for row in chart["risk_tilt_rows"]:
                lines.append(
                    f"| {family_name} | {chart['chart']} | "
                    f"{chart['empirical_witness_diameter']:.6f} | "
                    f"{chart['upper_witness_diameter']:.6f} | {row['beta']:.2f} | "
                    f"{row['target_ess_fraction']:.6f} | "
                    f"{row['pushed_forward_overlap']:.6f} | "
                    f"{row['partial_objective_excluding_nuisance_and_transience']:.6f} |"
                )
    lines.extend(
        [
            "",
            "## Gate verdict",
            "",
            "- Oracle Decision Headroom: `NOT_EVALUABLE`.",
            "- Stranding–throughput Pareto: `NOT_EVALUABLE`.",
            "- Theorem 32/33 certificate: descriptive proxy only; nuisance,",
            "  transience, stopped occupation, and manager decisions are absent.",
            "",
            report["non_claim"],
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_r3_stopped_quotient_proxy(args.trajectory_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "r3_stopped_quotient_proxy.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "r3_stopped_quotient_proxy.md").write_text(
        render_markdown(report), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
