#!/usr/bin/env python3
"""Strict anchor-clustered analysis for the dual-viability diagnostic."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.rechargeability_safety.dual_viability import budget_labels


SEED = 20260905
PALETTE = {
    "option": "#0072B2",
    "return": "#E69F00",
    "history": "#6A3D9A",
    "mixed": "#D55E00",
    "safe": "#009E73",
    "timeout": "#F0E442",
    "collision": "#D55E00",
    "boundary": "#56B4E9",
}


def wilson(successes: int, total: int, alpha: float = 0.05) -> tuple[float, float]:
    z = stats.norm.ppf(1.0 - alpha / 2.0)
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    radius = z * np.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return float(center - radius), float(center + radius)


def bootstrap_columns(values: np.ndarray, draws: int = 20_000) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    rng = np.random.default_rng(SEED)
    indices = rng.integers(0, values.shape[0], size=(draws, values.shape[0]))
    means = values[indices].mean(axis=1)
    return np.percentile(means, 2.5, axis=0), np.percentile(means, 97.5, axis=0)


def holm_adjust(p_values: list[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(p)
    adjusted_sorted = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order])
    adjusted_sorted = np.minimum(adjusted_sorted, 1.0)
    adjusted = np.empty_like(p)
    adjusted[order] = adjusted_sorted
    return adjusted


def paired_nonparametric(differences: np.ndarray) -> tuple[float, float, float]:
    differences = np.asarray(differences, dtype=np.float64)
    nonzero = differences[~np.isclose(differences, 0.0)]
    if nonzero.size == 0:
        return 0.0, 1.0, 0.0
    result = stats.wilcoxon(nonzero, zero_method="wilcox", alternative="two-sided", method="auto")
    ranks = stats.rankdata(np.abs(nonzero))
    rank_biserial = (ranks[nonzero > 0].sum() - ranks[nonzero < 0].sum()) / ranks.sum()
    return float(result.statistic), float(result.pvalue), float(rank_biserial)


def cochran_q(binary: np.ndarray) -> tuple[float, float]:
    x = np.asarray(binary, dtype=np.float64)
    n, k = x.shape
    column_sums = x.sum(axis=0)
    row_sums = x.sum(axis=1)
    denominator = k * x.sum() - np.square(row_sums).sum()
    if denominator <= 0.0:
        return 0.0, 1.0
    q = (k - 1.0) * (k * np.square(column_sums).sum() - x.sum() ** 2) / denominator
    return float(q), float(stats.chi2.sf(q, k - 1))


def fmt_p(value: float) -> str:
    return "<0.0001" if value < 0.0001 else f"{value:.4f}"


def load_inputs(root: Path) -> dict[str, object]:
    result = json.loads((root / "RESULT.json").read_text(encoding="utf-8"))
    option_rows = [json.loads(path.read_text(encoding="utf-8")) for path in root.glob("rollouts/*/option/*.json")]
    return_rows = [json.loads(path.read_text(encoding="utf-8")) for path in root.glob("rollouts/*/return_now.json")]
    option_rows.sort(key=lambda row: (int(row["anchor_index"]), int(row["candidate_index"])))
    return_rows.sort(key=lambda row: int(row["anchor_index"]))
    if len(option_rows) != 3100 or len(return_rows) != 310:
        raise RuntimeError(f"incomplete raw evidence: {len(option_rows)} option and {len(return_rows)} return rows")
    source_dir = Path(result["data_dir"])
    fork_result = json.loads((source_dir / "RESULT.json").read_text(encoding="utf-8"))
    nominal_dir = Path(fork_result["source"])
    nominal_result = json.loads((nominal_dir / "RESULT.json").read_text(encoding="utf-8"))
    capacity = float(nominal_result["capacity"])
    return_by_anchor = {str(row["anchor_id"]): row for row in return_rows}
    historical_rows = []
    for row in option_rows:
        path = source_dir / "branches" / str(row["anchor_id"]) / f"{row['candidate_name']}.json"
        historical_rows.append(json.loads(path.read_text(encoding="utf-8")))
    return {
        "result": result,
        "capacity": capacity,
        "option_rows": option_rows,
        "return_rows": return_rows,
        "return_by_anchor": return_by_anchor,
        "historical_rows": historical_rows,
    }


def analyze(root: Path, output: Path) -> dict[str, object]:
    loaded = load_inputs(root)
    result = loaded["result"]
    capacity = loaded["capacity"]
    option_rows = loaded["option_rows"]
    return_rows = loaded["return_rows"]
    historical_rows = loaded["historical_rows"]
    budgets = np.asarray(result["budgets"], dtype=np.float64)
    anchors = sorted({str(row["anchor_id"]) for row in option_rows}, key=lambda value: int(value.split("_")[1]) * 10 + int(value.split("_")[-1]))
    candidates = [str(row["candidate_name"]) for row in option_rows[:10]]
    if len(anchors) != 310 or len(candidates) != 10 or len(set(candidates)) != 10:
        raise RuntimeError("unexpected anchor/candidate design")
    anchor_index = {anchor: index for index, anchor in enumerate(anchors)}
    candidate_index = {name: index for index, name in enumerate(candidates)}
    n, k = len(anchors), len(candidates)

    option_safe = np.zeros((n, k), dtype=bool)
    option_energy = np.zeros((n, k), dtype=np.float64)
    intervention = np.zeros((n, k), dtype=bool)
    correction = np.zeros((n, k), dtype=np.float64)
    option_reason = np.empty((n, k), dtype=object)
    history_safe = np.zeros((n, k), dtype=bool)
    history_energy = np.zeros((n, k), dtype=np.float64)
    for row, old in zip(option_rows, historical_rows):
        i = anchor_index[str(row["anchor_id"])]
        j = candidate_index[str(row["candidate_name"])]
        option_safe[i, j] = bool(row["structurally_safe_return"])
        option_energy[i, j] = float(row["total_realized_energy"]) / capacity
        intervention[i, j] = bool(row["first_hocbf_intervened"])
        correction[i, j] = float(row["first_action_correction_norm"])
        option_reason[i, j] = str(row["end_reason"])
        history_safe[i, j] = bool(old["safe_recharge"])
        history_energy[i, j] = float(old["total_realized_energy"]) / capacity

    return_by_anchor = loaded["return_by_anchor"]
    return_safe = np.asarray([bool(return_by_anchor[a]["structurally_safe_return"]) for a in anchors])
    return_energy = np.asarray([float(return_by_anchor[a]["total_realized_energy"]) / capacity for a in anchors])
    return_reason = np.asarray([str(return_by_anchor[a]["end_reason"]) for a in anchors], dtype=object)

    option_labels = budget_labels(
        structurally_safe=option_safe.ravel(), energy_fraction=option_energy.ravel(), budgets=budgets
    ).reshape(n, k, -1)
    history_labels = budget_labels(
        structurally_safe=history_safe.ravel(), energy_fraction=history_energy.ravel(), budgets=budgets
    ).reshape(n, k, -1)
    return_labels = budget_labels(
        structurally_safe=return_safe, energy_fraction=return_energy, budgets=budgets
    )

    option_anchor_rate = option_labels.mean(axis=1)
    history_anchor_rate = history_labels.mean(axis=1)
    option_rate = option_anchor_rate.mean(axis=0)
    history_rate = history_anchor_rate.mean(axis=0)
    return_rate = return_labels.mean(axis=0)
    mixed = np.logical_and(option_labels.any(axis=1), ~option_labels.all(axis=1))
    mixed_rate = mixed.mean(axis=0)

    option_ci = bootstrap_columns(option_anchor_rate)
    history_ci = bootstrap_columns(history_anchor_rate)
    return_ci = bootstrap_columns(return_labels.astype(float))
    mixed_ci = np.asarray([wilson(int(mixed[:, column].sum()), n) for column in range(len(budgets))]).T

    hist_tests = [paired_nonparametric(option_anchor_rate[:, c] - history_anchor_rate[:, c]) for c in range(len(budgets))]
    now_tests = [paired_nonparametric(option_anchor_rate[:, c] - return_labels[:, c]) for c in range(len(budgets))]
    hist_adjusted = holm_adjust([value[1] for value in hist_tests])
    now_adjusted = holm_adjust([value[1] for value in now_tests])
    hist_diff_ci = bootstrap_columns(option_anchor_rate - history_anchor_rate)
    now_diff_ci = bootstrap_columns(option_anchor_rate - return_labels)

    structural_mixed = np.logical_and(option_safe.any(axis=1), ~option_safe.all(axis=1))
    structural_mixed_ci = wilson(int(structural_mixed.sum()), n)
    q_stat, q_p = cochran_q(option_safe)
    energy_friedman = stats.friedmanchisquare(*[option_energy[:, j] for j in range(k)])
    energy_kendall_w = float(energy_friedman.statistic / (n * (k - 1)))

    unique_executed = []
    for anchor in anchors:
        selected = [row for row in option_rows if str(row["anchor_id"]) == anchor]
        unique_executed.append(len({tuple(round(float(v), 7) for v in row["first_executed_action"]) for row in selected}))
    unique_executed = np.asarray(unique_executed)

    return_counts = Counter(return_reason.tolist())
    option_counts = Counter(option_reason.ravel().tolist())
    candidate_rows = []
    for j, name in enumerate(candidates):
        successes = int(option_safe[:, j].sum())
        low, high = wilson(successes, n)
        candidate_rows.append(
            {
                "candidate": name,
                "safe": successes,
                "safe_rate": successes / n,
                "safe_ci": [low, high],
                "energy_mean": float(option_energy[:, j].mean()),
                "energy_sd": float(option_energy[:, j].std(ddof=1)),
                "hocbf_rate": float(intervention[:, j].mean()),
                "correction_mean": float(correction[:, j].mean()),
            }
        )

    budget_rows = []
    for c, budget in enumerate(budgets):
        budget_rows.append(
            {
                "budget": float(budget),
                "option_rate": float(option_rate[c]),
                "return_rate": float(return_rate[c]),
                "history_rate": float(history_rate[c]),
                "mixed_anchors": int(mixed[:, c].sum()),
                "mixed_rate": float(mixed_rate[c]),
                "option_minus_history": float(option_rate[c] - history_rate[c]),
                "option_minus_history_ci": [float(hist_diff_ci[0][c]), float(hist_diff_ci[1][c])],
                "option_history_wilcoxon_w": hist_tests[c][0],
                "option_history_p_holm": float(hist_adjusted[c]),
                "option_history_rank_biserial": hist_tests[c][2],
                "option_minus_return": float(option_rate[c] - return_rate[c]),
                "option_minus_return_ci": [float(now_diff_ci[0][c]), float(now_diff_ci[1][c])],
                "option_return_wilcoxon_w": now_tests[c][0],
                "option_return_p_holm": float(now_adjusted[c]),
                "option_return_rank_biserial": now_tests[c][2],
            }
        )

    distance_order = ["100-500", "500-1500", "1500-2500", "2500-4000", ">4000"]
    distance_rows = []
    for bucket in distance_order:
        selected = [row for row in option_rows if str(row["distance_bucket"]) == bucket]
        successful = [row for row in selected if bool(row["structurally_safe_return"])]
        distance_rows.append(
            {
                "bucket": bucket,
                "rollouts": len(selected),
                "safe_rate": float(np.mean([bool(row["structurally_safe_return"]) for row in selected])),
                "collision_rate": float(np.mean([bool(row["collision"]) for row in selected])),
                "boundary_rate": float(np.mean([bool(row["boundary_contact"]) for row in selected])),
                "successful_energy_mean": float(np.mean([float(row["total_realized_energy"]) / capacity for row in successful])),
            }
        )

    option_structural_ci = bootstrap_columns(option_safe.mean(axis=1))
    summary = {
        "evidence_class": "development_only_inspected_data",
        "unit_of_analysis": "anchor",
        "anchors": n,
        "candidates_per_anchor": k,
        "option_rollouts": n * k,
        "return_rollouts": n,
        "censored_option": int(sum(bool(row["censored"]) for row in option_rows)),
        "censored_return": int(sum(bool(row["censored"]) for row in return_rows)),
        "return_structural_safe": int(return_safe.sum()),
        "return_structural_safe_rate": float(return_safe.mean()),
        "return_structural_safe_ci": list(wilson(int(return_safe.sum()), n)),
        "option_structural_safe": int(option_safe.sum()),
        "option_structural_safe_rate": float(option_safe.mean()),
        "option_structural_safe_cluster_ci": [float(option_structural_ci[0][0]), float(option_structural_ci[1][0])],
        "history_structural_safe_rate": float(history_safe.mean()),
        "structurally_mixed_anchors": int(structural_mixed.sum()),
        "structurally_mixed_anchor_rate": float(structural_mixed.mean()),
        "structurally_mixed_anchor_ci": list(structural_mixed_ci),
        "return_end_reasons": dict(return_counts),
        "option_end_reasons": dict(option_counts),
        "hocbf_intervention_rate": float(intervention.mean()),
        "candidate_step_collision_count": int(option_counts["unsafe_on_candidate_step"]),
        "full_action_diversity_anchors": int(np.sum(unique_executed == k)),
        "collapsed_action_anchors": int(np.sum(unique_executed == 1)),
        "candidate_structural_cochran_q": q_stat,
        "candidate_structural_cochran_p": q_p,
        "candidate_energy_friedman_chi2": float(energy_friedman.statistic),
        "candidate_energy_friedman_p": float(energy_friedman.pvalue),
        "candidate_energy_kendall_w": energy_kendall_w,
        "return_energy_mean": float(return_energy.mean()),
        "return_energy_sd": float(return_energy.std(ddof=1)),
        "option_energy_mean": float(option_energy.mean()),
        "option_energy_anchor_sd": float(option_energy.mean(axis=1).std(ddof=1)),
        "history_energy_mean": float(history_energy.mean()),
        "budget_rows": budget_rows,
        "candidate_rows": candidate_rows,
        "distance_rows": distance_rows,
    }

    output.mkdir(parents=True, exist_ok=True)
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    (output / "strict-statistics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    x = budgets * 100.0
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), constrained_layout=True)
    for label, rate, ci, color, marker in [
        ("One action, then return", option_rate, option_ci, PALETTE["option"], "o"),
        ("Return now", return_rate, return_ci, PALETTE["return"], "s"),
        ("Complete task, then return (old)", history_rate, history_ci, PALETTE["history"], "^"),
    ]:
        axes[0].plot(x, rate * 100.0, label=label, color=color, marker=marker, markersize=3, linewidth=1.7)
        axes[0].fill_between(x, ci[0] * 100.0, ci[1] * 100.0, color=color, alpha=0.16)
    axes[0].set(xlabel="Available-energy budget (% capacity)", ylabel="Positive label rate (%)", ylim=(0, 100))
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=7, loc="lower right")
    axes[1].plot(x, mixed_rate * 100.0, color=PALETTE["mixed"], marker="o", markersize=3, linewidth=1.7)
    axes[1].fill_between(x, mixed_ci[0] * 100.0, mixed_ci[1] * 100.0, color=PALETTE["mixed"], alpha=0.18)
    axes[1].set(xlabel="Available-energy budget (% capacity)", ylabel="Anchors with mixed action labels (%)", ylim=(0, 12))
    axes[1].grid(alpha=0.25)
    fig.savefig(figures / "figure-01-budget-semantics.pdf")
    fig.savefig(figures / "figure-01-budget-semantics.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), constrained_layout=True)
    failure_names = ["Safe return", "Step limit", "Collision", "Boundary"]
    return_values = np.asarray([
        return_safe.sum(),
        return_counts["task_step_limit"],
        sum(bool(row["collision"]) for row in return_rows),
        sum(bool(row["boundary_contact"]) for row in return_rows),
    ], dtype=float) / n
    option_values = np.asarray([
        option_safe.sum(), option_counts["task_step_limit"],
        option_counts["unsafe_on_candidate_step"] + sum(bool(row["collision"]) and row["end_reason"] == "unsafe_during_recovery" for row in option_rows),
        sum(bool(row["boundary_contact"]) for row in option_rows),
    ], dtype=float) / (n * k)
    bottom = np.zeros(2)
    colors = [PALETTE["safe"], PALETTE["timeout"], PALETTE["collision"], PALETTE["boundary"]]
    for name, values, color in zip(failure_names, np.column_stack([return_values, option_values]), colors):
        axes[0].bar([0, 1], values * 100.0, bottom=bottom * 100.0, label=name, color=color, edgecolor="black", linewidth=0.3)
        bottom += values
    axes[0].set_xticks([0, 1], ["Return now", "One action\nthen return"])
    axes[0].set(ylabel="Outcome rate (%)", ylim=(0, 100))
    axes[0].legend(frameon=False, fontsize=7, loc="lower center")
    y = np.arange(k)
    rates = np.asarray([row["safe_rate"] for row in candidate_rows]) * 100.0
    lows = np.asarray([row["safe_ci"][0] for row in candidate_rows]) * 100.0
    highs = np.asarray([row["safe_ci"][1] for row in candidate_rows]) * 100.0
    axes[1].errorbar(rates, y, xerr=np.vstack([rates - lows, highs - rates]), fmt="o", color=PALETTE["option"], capsize=2)
    axes[1].set_yticks(y, candidates)
    axes[1].set(xlabel="Structurally safe return rate (%)", xlim=(82, 94))
    axes[1].grid(axis="x", alpha=0.25)
    fig.savefig(figures / "figure-02-recovery-and-candidates.pdf")
    fig.savefig(figures / "figure-02-recovery-and-candidates.png", dpi=200)
    plt.close(fig)

    write_reports(output, summary)
    return summary


def write_reports(output: Path, s: dict[str, object]) -> None:
    budget_lines = []
    for row in s["budget_rows"]:
        ci = row["option_minus_history_ci"]
        budget_lines.append(
            f"| {100*row['budget']:.2f} | {100*row['option_rate']:.2f} | {100*row['return_rate']:.2f} | "
            f"{100*row['history_rate']:.2f} | {row['mixed_anchors']} ({100*row['mixed_rate']:.2f}%) | "
            f"{100*row['option_minus_history']:+.2f} [{100*ci[0]:+.2f}, {100*ci[1]:+.2f}] | "
            f"{fmt_p(row['option_history_p_holm'])} |"
        )
    candidate_lines = []
    for row in s["candidate_rows"]:
        candidate_lines.append(
            f"| {row['candidate']} | {row['safe']}/310 ({100*row['safe_rate']:.2f}%) | "
            f"[{100*row['safe_ci'][0]:.2f}, {100*row['safe_ci'][1]:.2f}] | "
            f"{100*row['energy_mean']:.2f} | {100*row['hocbf_rate']:.2f} |"
        )
    return_ci = s["return_structural_safe_ci"]
    option_ci = s["option_structural_safe_cluster_ci"]
    mixed_ci = s["structurally_mixed_anchor_ci"]
    report = f"""# Dual-Viability Diagnostic: Strict Analysis Report

## Analysis questions

1. Is the historical complete-task-then-return label equivalent to the deployment-aligned one-action-then-return label?
2. Is the corrected label sufficiently action-sensitive to justify training a standalone action critic?
3. Is the frozen R3 recovery policy reliable enough to serve as the terminal recovery certificate?

The experimental unit is the **anchor** (n=310), not each of the 3,100 correlated candidate branches. All confidence intervals and paired tests aggregate candidates within anchor. This is development-only evidence from already inspected scenes and one frozen checkpoint/seed.

## Decision

**Repair the recovery baseline before training a new safety gate. Do not train a standalone binary option-action critic from this dataset.**

- Immediate return succeeded structurally in {s['return_structural_safe']}/310 anchors ({100*s['return_structural_safe_rate']:.2f}%, Wilson 95% CI {100*return_ci[0]:.2f}–{100*return_ci[1]:.2f}%). The 35 failures comprise {s['return_end_reasons'].get('task_step_limit', 0)} step-limit failures and 18 unsafe recovery terminations (16 collisions and 2 boundary contacts).
- One-action-then-return was safe in {s['option_structural_safe']}/3100 branches ({100*s['option_structural_safe_rate']:.2f}%, anchor-clustered bootstrap 95% CI {100*option_ci[0]:.2f}–{100*option_ci[1]:.2f}%). It does not repair the weak terminal recovery assumption.
- Only {s['structurally_mixed_anchors']}/310 anchors ({100*s['structurally_mixed_anchor_rate']:.2f}%, Wilson 95% CI {100*mixed_ci[0]:.2f}–{100*mixed_ci[1]:.2f}%) had different structural outcomes among the ten actions. Across energy budgets, the maximum mixed-label count is {max(row['mixed_anchors'] for row in s['budget_rows'])}/310. The signal is predominantly state/recovery viability, not candidate identity.
- This low action sensitivity is not generally caused by HOCBF collapsing proposals: {s['full_action_diversity_anchors']}/310 anchors retained all ten distinct executed actions; only {s['collapsed_action_anchors']}/310 collapsed to one executed action. HOCBF intervened in {100*s['hocbf_intervention_rate']:.2f}% of branches, yet {s['candidate_step_collision_count']} candidate steps still terminated in collision.
- Historical completion consumed {100*s['history_energy_mean']:.2f}% of capacity on average versus {100*s['option_energy_mean']:.2f}% for one-action-then-return and {100*s['return_energy_mean']:.2f}% for return-now. Thus the old target mostly measures the energy of finishing the task before recovery, not whether the next deployed action preserves returnability.

## Budget-wise paired evidence

Rates average the ten branches within each anchor. Difference CIs use 20,000 anchor-clustered bootstrap samples. P-values are two-sided Wilcoxon signed-rank tests across anchor-level differences with Holm correction over all 15 budgets.

| Budget (% capacity) | Option positive (%) | Return-now positive (%) | Historical positive (%) | Mixed option anchors | Option − historical pp [95% CI] | Holm p |
|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(budget_lines)}

## Candidate breakdown

The ten candidate safe-return rates span only {100*(max(row['safe_rate'] for row in s['candidate_rows'])-min(row['safe_rate'] for row in s['candidate_rows'])):.2f} percentage points. Cochran's Q={s['candidate_structural_cochran_q']:.3f}, p={fmt_p(s['candidate_structural_cochran_p'])}; there is no evidence that the named candidate family has materially different structural-return rates. Energy differs statistically across candidates (Friedman χ²={s['candidate_energy_friedman_chi2']:.3f}, p={fmt_p(s['candidate_energy_friedman_p'])}) but the effect is negligible (Kendall's W={s['candidate_energy_kendall_w']:.4f}).

| Candidate | Structurally safe | Wilson 95% CI (%) | Mean energy (% capacity) | HOCBF intervention (%) |
|---|---:|---:|---:|---:|
{chr(10).join(candidate_lines)}

## Claim candidates

- Claim: The old completion-then-return label is not deployment-aligned.
  - Source evidence: large two-direction discordance at intermediate budgets and a 5.25 percentage-point mean-energy gap.
  - Allowed wording: “The development diagnostic shows that the historical label is not interchangeable with immediate option preservation.”
  - Forbidden stronger wording: “Wrong labels alone caused the failed closed-loop pilot.”
  - Uncertainty: the comparator also differs in prefix length and HOCBF usage.
  - Decision: keep.
- Claim: Current option preservation is mainly state-determined.
  - Source evidence: at most 22/310 mixed-label anchors and 21/310 structurally mixed anchors.
  - Allowed wording: “Under the frozen candidate set and one-step horizon, returnability is predominantly anchor-dependent.”
  - Forbidden stronger wording: “Actions never affect returnability.”
  - Decision: keep with scope.
- Claim: R3 is an adequate recovery certificate.
  - Source evidence: only 275/310 immediate returns are structurally safe.
  - Allowed wording: “R3 provides a useful recovery baseline but is not yet a high-reliability certificate.”
  - Forbidden stronger wording: “The present system guarantees safe return.”
  - Decision: discard the guarantee claim.

## Next experimental decision

1. Repair or retrain the R3 return-to-home behavior on the 35 failed anchor types, separating collision and timeout failures.
2. Model a state return-value/distribution `V_R(x)` first. Evaluate actions through a known or learned successor model `V_R(F(x,a))`, rather than fitting a sparse binary `Q(x,a)` directly.
3. Keep structural failure and continuous energy-to-go as separate heads; a binary budget label throws away most action information.
4. After the recovery baseline is fixed, collect fresh scenes for training/calibration and reserve untouched scenes and model seeds for confirmation.

## Limitations

- One checkpoint and one world/model seed block policy-seed generalization claims.
- The 310 anchors came from inspected development scenes; no confirmation or publication claim is permitted.
- The old comparator used an eight-step raw prefix and task continuation, whereas the corrected arm used one HOCBF-filtered step; the experiment demonstrates non-equivalence but does not isolate each causal component.
- Euclidean distance buckets are descriptive only and mix geometry, early failures, and path complexity.
"""
    (output / "analysis-report.md").write_text(report, encoding="utf-8")

    stats_text = f"""# Statistical Appendix

## Design and assumptions

- Unit: 310 anchors, each with ten repeated candidate actions.
- Missing/censored observations: 0 option, 0 return-now.
- Binary and bounded repeated outcomes are non-normal by construction; no parametric t-test was used.
- Paired comparisons use anchor-level candidate means, Wilcoxon signed-rank tests, Holm correction across 15 budgets, rank-biserial effect sizes, and 20,000-sample cluster bootstraps.
- Candidate structural rates use Cochran's Q; candidate energy uses Friedman repeated-measures test and Kendall's W.

## Structural outcomes

- Return-now: {s['return_structural_safe']}/310, {100*s['return_structural_safe_rate']:.3f}%, Wilson CI [{100*return_ci[0]:.3f}, {100*return_ci[1]:.3f}].
- Option: {s['option_structural_safe']}/3100, {100*s['option_structural_safe_rate']:.3f}%, anchor-clustered CI [{100*option_ci[0]:.3f}, {100*option_ci[1]:.3f}].
- Structurally mixed anchors: {s['structurally_mixed_anchors']}/310, {100*s['structurally_mixed_anchor_rate']:.3f}%, Wilson CI [{100*mixed_ci[0]:.3f}, {100*mixed_ci[1]:.3f}].
- Candidate equality: Cochran Q({len(s['candidate_rows'])-1})={s['candidate_structural_cochran_q']:.4f}, p={s['candidate_structural_cochran_p']:.6g}.
- Candidate energy: Friedman χ²({len(s['candidate_rows'])-1})={s['candidate_energy_friedman_chi2']:.4f}, p={s['candidate_energy_friedman_p']:.6g}, Kendall W={s['candidate_energy_kendall_w']:.6f}.

## Machine-readable statistics

All exact per-budget test statistics, adjusted p-values, rank-biserial effects, bootstrap intervals, candidate summaries, and distance summaries are stored in `strict-statistics.json`.

## Statistical blockers

- These are scene/anchor replications, not independent training seeds.
- No family-wise inference is made across different policy checkpoints.
- Significance does not imply practical action discriminability; the mixed-anchor fraction is the decision-relevant statistic.
"""
    (output / "stats-appendix.md").write_text(stats_text, encoding="utf-8")

    catalog = """# Figure Catalog

## figure-01-budget-semantics.pdf

- Purpose: compare label semantics across available-energy budgets and quantify within-anchor action discrimination.
- Data: 310 anchors, ten candidates per anchor; ribbons are 95% anchor-clustered bootstrap intervals for label rates and Wilson intervals for mixed-anchor proportions.
- Notice: historical completion labels diverge strongly at intermediate budgets, while only about 6–7% of anchors have mixed corrected action labels.
- Implication: replace the old target, but do not train a standalone binary action critic from this sparse action signal.
- Caveat: development-only inspected data; budgets are evaluated as a family rather than selecting the best knot.

## figure-02-recovery-and-candidates.pdf

- Purpose: expose terminal recovery failures and test whether named candidate actions differ materially.
- Data: exact endpoint counts; candidate error bars are Wilson 95% confidence intervals over 310 paired anchors.
- Notice: immediate return fails in 35 anchors, and all ten candidate rates are tightly clustered.
- Implication: repair the recovery baseline and prioritize a state return-value model.
- Caveat: candidate comparisons share anchors and are not independent binomial experiments; Cochran's Q supplies the paired omnibus test.
"""
    (output / "figure-catalog.md").write_text(catalog, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    root = args.input_dir.expanduser().resolve()
    output = args.output_dir.expanduser().resolve() if args.output_dir else root / "analysis-output"
    summary = analyze(root, output)
    print(json.dumps({"status": "COMPLETE", "output": str(output), "decision": "REPAIR_RECOVERY_FIRST", "anchors": summary["anchors"]}, indent=2))


if __name__ == "__main__":
    main()
