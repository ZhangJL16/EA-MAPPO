from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from experiments.energy_mc.statistical_gate import probability_empirical_meets_target


ROOT = Path(__file__).resolve().parents[1]
DEVELOPMENT = ROOT / "artifacts/energy_risk_v5_development"


def load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def percentage(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def evidence_row(name: str, row: dict[str, object]) -> str:
    return (
        f"| {name} | {row['count']} | {row['successes']} | "
        f"{percentage(row['empirical_coverage'])} | "
        f"[{percentage(row['exact_interval_low'])}, "
        f"{percentage(row['exact_interval_high'])}] | "
        f"{float(row['lower_tail_pvalue_at_target']):.4g} |"
    )


def phase2_summary(payload: dict[str, object] | None) -> tuple[str, str]:
    if not payload:
        return "SKIPPED", "INCONCLUSIVE"
    baseline_tasks = []
    candidate_tasks = []
    baseline_exhaustion = []
    candidate_exhaustion = []
    baseline_unnecessary = []
    candidate_unnecessary = []
    rows = []
    for seed, result in sorted(payload.items()):
        baseline = result["baseline"]
        candidate = result["candidate"]
        baseline_tasks.append(float(baseline["tasks_per_1000_transitions"]))
        candidate_tasks.append(float(candidate["tasks_per_1000_transitions"]))
        baseline_exhaustion.append(int(baseline["energy_exhaustion_count"]))
        candidate_exhaustion.append(int(candidate["energy_exhaustion_count"]))
        baseline_unnecessary.append(float(baseline.get("unnecessary_return_rate") or 0.0))
        candidate_unnecessary.append(float(candidate.get("unnecessary_return_rate") or 0.0))
        rows.append(
            f"| {seed} | {baseline['total_delivery_tasks_completed']} | "
            f"{candidate['total_delivery_tasks_completed']} | "
            f"{baseline['energy_exhaustion_count']} | "
            f"{candidate['energy_exhaustion_count']} | "
            f"{float(baseline.get('unnecessary_return_rate') or 0.0):.3f} | "
            f"{float(candidate.get('unnecessary_return_rate') or 0.0):.3f} |"
        )
    safe = sum(candidate_exhaustion) <= sum(baseline_exhaustion)
    efficient = (
        mean(candidate_tasks) > mean(baseline_tasks)
        or mean(candidate_unnecessary) < mean(baseline_unnecessary)
    )
    conclusion = "YES" if safe and efficient else "NO MEASURABLE IMPROVEMENT"
    table = "\n".join(
        [
            "| Seed | Baseline Tasks | Candidate Tasks | Baseline Exhaustion | Candidate Exhaustion | Baseline Unnecessary | Candidate Unnecessary |",
            "|---|---:|---:|---:|---:|---:|---:|",
            *rows,
            "",
            f"Mean tasks/1000: baseline {mean(baseline_tasks):.4f}, candidate {mean(candidate_tasks):.4f}.",
        ]
    )
    return table, conclusion


def build(run: Path, output: Path) -> None:
    completed = load(run / "COMPLETED.json")
    prereg = load(run / "PREREGISTERED_ENERGY_RISK_V5.json")
    readiness = load(run / "FRESH_V5_READINESS.json")
    goal = load(run / "fresh_v5/goal_results.json")
    mission = load(run / "fresh_v5/mission_results.json")
    goal_evidence = load(run / "fresh_v5/goal_statistical_evidence.json")
    mission_evidence = load(run / "fresh_v5/mission_statistical_evidence.json")
    gate_audit = load(DEVELOPMENT / "statistical_gate_audit.json")
    forensics = load(DEVELOPMENT / "forensics/forensic_summary.json")
    point = load(DEVELOPMENT / "point_ablation_warmstart/POINT_MODEL_DECISION.json")
    ladder = load(DEVELOPMENT / "method_ladder/goal_method_results.json")
    mission_ladder = load(DEVELOPMENT / "method_ladder/mission_method_results.json")
    short = load(DEVELOPMENT / "short_rollout/selection_decision.json")
    switch = load(DEVELOPMENT / "switch_attribution.json")
    historical_switch = switch["methods"]["selected_adaptive"]
    phase2_table, efficiency = phase2_summary(completed.get("phase2_100k"))

    goal_groups = goal_evidence["groups"]
    mission_groups = mission_evidence["groups"]
    goal_rows = "\n".join(
        evidence_row(name, row) for name, row in goal_groups.items()
    )
    mission_rows = "\n".join(
        evidence_row(name, row) for name, row in mission_groups.items()
    )
    goal_worst_name = min(
        goal["by_primary_group"],
        key=lambda name: goal["by_primary_group"][name]["coverage"],
    )
    mission_worst_name = min(
        mission["by_primary_group"],
        key=lambda name: mission["by_primary_group"][name]["coverage"],
    )
    k2 = ladder["k2_suffix"]["0.975"]["metrics"]
    mission_selected = mission_ladder["0.99"]["metrics"]

    text = f"""# Energy Risk Root Cause and Final v5 Solution

## Executive Decision

- Fresh v5 status: **{readiness['fresh_v5_status']}**.
- Formal 500k Phase2: **NOT STARTED**.
- Frozen SAC and MC point estimator were not retrained.
- Selected Goal construction: K2 suffix-future-max positive residual model, compact decision-time context, direct Goal-type x distance Mondrian calibration at {percentage(prereg['goal']['conformal_construction_coverage'])}.
- Selected Mission construction: frozen heteroscedastic Laplace model plus direct distance Mondrian correction at {percentage(prereg['mission']['conformal_construction_coverage'])}.

## Statistical Gate Audit

Raw subgroup empirical coverage >=95% is **not a well-designed sole gate**. At n=231 and true coverage 95%, the probability of observing empirical coverage >=95% is only {probability_empirical_meets_target(231, true_coverage=0.95):.4f}. The v4 worst Goal group had 215/231 covered with exact 95% interval [{percentage(gate_audit['v4_goal_worst']['exact_interval_low'])}, {percentage(gate_audit['v4_goal_worst']['exact_interval_high'])}].

The preregistered v5 gate therefore combines:

1. direct finite-sample split-conformal construction inside each predefined physical group;
2. fresh empirical stress tests;
3. one-sided binomial undercoverage tests at p=0.95 with Holm family-wise correction.

Failure to reject undercoverage is not proof of arbitrary conditional coverage. The guarantee is predefined finite-group conditional coverage under within-group exchangeability, not arbitrary conditional coverage at every state x.

## Root Cause

The original 7D state is **partially insufficient for tail risk**, while remaining sufficient for the point estimate. Adding absolute position reduced point MAE from {point['original_mae']:.6f} to {point['best_mae']:.6f}, only {100.0 * point['relative_mae_improvement']:.2f}%, below the preregistered 10% materiality threshold. Therefore the original point model was retained.

The deployable missing information is compact position/boundary context and a target aligned with future worst underestimation. K2's predicted risk had Pearson/Spearman correlation {ladder['k2_correlation_with_suffix_max']['pearson']:.3f}/{ladder['k2_correlation_with_suffix_max']['spearman']:.3f} with suffix maximum underestimation, stronger than K1's {ladder['k1_correlation_with_suffix_max']['pearson']:.3f}/{ladder['k1_correlation_with_suffix_max']['spearman']:.3f}. On development data at 97.5% construction, K2 achieved overall {percentage(k2['overall']['coverage'])}, worst group {percentage(k2['worst_primary_coverage'])}, and mean width {k2['mean_bound_width']:.4f}.

Post-hoc path ratio, future boundary contact, future steps, and future acceleration were used only for diagnosis and were prohibited as model inputs. The top-100 forensics found long-distance and boundary-interaction concentration, but these realized-future labels are unavailable at decision time.

Short rollout context did not improve the worst group: H=10/25/50 worst coverage was {percentage(short['short_rollout_at_0.95']['h10']['worst_primary_coverage'])}/{percentage(short['short_rollout_at_0.95']['h25']['worst_primary_coverage'])}/{percentage(short['short_rollout_at_0.95']['h50']['worst_primary_coverage'])}. The deterministic full-rollout oracle was nearly exact (MAE {short['full_rollout_oracle']['mae']:.3g}) but required {short['full_rollout_oracle']['mean_latency_seconds']:.3f}s per state and is not the learned deployment estimator.

## Mission Tail

Task and return residuals were essentially uncorrelated (Pearson {mission_ladder['task_return_residual_dependence']['overall']['pearson']:.4f}, Spearman {mission_ladder['task_return_residual_dependence']['overall']['spearman']:.4f}); the long-distance failure was therefore treated as distance-specific Mission calibration rather than a correlated-component theorem. Development performance at the selected 99% construction was overall {percentage(mission_selected['overall']['coverage'])}, worst distance group {percentage(mission_selected['worst_primary_coverage'])}, mean width {mission_selected['mean_bound_width']:.4f}.

## Fresh v5 Goal Results

Point MAE: {goal['point_mae']:.6f}. Mean/P95 bound width: {goal['mean_bound_width']:.6f}/{goal['p95_bound_width']:.6f}. Overall whole-trajectory coverage: {percentage(goal['overall']['coverage'])}. Worst interaction: {goal_worst_name} at {percentage(goal['by_primary_group'][goal_worst_name]['coverage'])}.

| Group | n | Covered | Coverage | Exact 95% CI | One-sided p at 0.95 |
|---|---:|---:|---:|---:|---:|
{goal_rows}

Goal Holm stress gate: **{'PASS' if goal_evidence['holm_undercoverage_test']['passed'] else 'FAIL'}**.

## Fresh v5 Mission Results

Point MAE: {mission['point_mae']:.6f}. Mean/P95 bound width: {mission['mean_bound_width']:.6f}/{mission['p95_bound_width']:.6f}. Overall complete-Mission coverage: {percentage(mission['overall']['coverage'])}. Worst distance group: {mission_worst_name} at {percentage(mission['by_primary_group'][mission_worst_name]['coverage'])}.

| Group | n | Covered | Coverage | Exact 95% CI | One-sided p at 0.95 |
|---|---:|---:|---:|---:|---:|
{mission_rows}

Mission Holm stress gate: **{'PASS' if mission_evidence['holm_undercoverage_test']['passed'] else 'FAIL'}**.

## Safety-Efficiency and Reserve Attribution

Fresh Goal unnecessary-return proxy: {percentage(goal['unnecessary_return_proxy_rate'])}. Fresh Mission unnecessary-return proxy: {percentage(mission['unnecessary_return_proxy_rate'])}.

Historical 20k decisions were reserve dominated: {percentage(historical_switch['cause_fractions'].get('reserve', 0.0))} of switches were attributed to reserve, and reserve was roughly {historical_switch['mean_reserve_to_uncertainty_ratio']:.1f}x the uncertainty margin. Therefore a narrower uncertainty bound is not expected to improve throughput unless it changes a decision after the fixed reserve is applied.

## Paired 100k Phase2

{phase2_table}

Adaptive risk improves delivery efficiency: **{efficiency}**.

## Final Architecture and Scope

Future obstacle/CBF integration should keep the point network on motion state plus goal geometry and expose a compact safe-trajectory context interface to the risk model. Raw LiDAR should not be inserted into the Energy network without evidence. Energy MC data must be collected from the actually executed CBF-filtered trajectory.

The result supports marginal and predefined finite-group conditional statements under exchangeability. It does not support arbitrary conditional coverage, distribution-shift-free guarantees, or physical UAV safety without calibration.

## Recommended Final 500k Configuration

- Frozen SAC checkpoint: `{prereg['sac']['sha256']}`.
- Frozen 7D point checkpoint: `{prereg['point_model']['sha256']}`.
- Goal: K2 suffix risk + compact decision context + direct Mondrian, construction coverage {percentage(prereg['goal']['conformal_construction_coverage'])}.
- Mission: frozen Laplace + direct distance Mondrian, construction coverage {percentage(prereg['mission']['conformal_construction_coverage'])}.
- Safety target: 95%; reserve: 10% of calibrated capacity.
- Formal 500k Phase2 remains **NOT STARTED** pending explicit user approval.
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build final Energy Risk v5 report")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "docs/energy_risk_root_cause_and_solution_v5.md",
    )
    args = parser.parse_args()
    build(args.run_dir, args.output)


if __name__ == "__main__":
    main()
