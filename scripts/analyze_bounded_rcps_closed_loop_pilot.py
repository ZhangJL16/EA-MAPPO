from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
from scipy import stats

from scripts.run_lidar_forked_action_data_gate import atomic_json, sha256_file


METHODS = ("ungated", "geometry_gate", "certified_meet_gate")
LABELS = {
    "ungated": "Ungated",
    "geometry_gate": "Geometry Gate",
    "certified_meet_gate": "Certified Meet",
}
COLORS = {
    "ungated": "#999999",
    "geometry_gate": "#56B4E9",
    "certified_meet_gate": "#E69F00",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    for row in rows[1:]:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def wilson(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    p = successes / n
    denominator = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denominator
    radius = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denominator
    return center - radius, center + radius


def holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=np.float64)
    running = 0.0
    count = len(p_values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * float(p_values[index]))
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted.tolist()


def paired_rank_biserial(values: np.ndarray) -> float:
    nonzero = np.asarray(values, dtype=np.float64)
    nonzero = nonzero[nonzero != 0.0]
    if nonzero.size == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(nonzero))
    positive = float(np.sum(ranks[nonzero > 0.0]))
    negative = float(np.sum(ranks[nonzero < 0.0]))
    return (positive - negative) / (positive + negative)


def validate(input_dir: Path) -> tuple[list[dict[str, object]], dict[str, object], dict[str, object]]:
    required = [
        input_dir / "RESULT.json",
        input_dir / "COMPLETED.json",
        input_dir / "RUNNING.json",
        input_dir / "CASES.json",
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    result = json.loads((input_dir / "RESULT.json").read_text(encoding="utf-8"))
    completed = json.loads((input_dir / "COMPLETED.json").read_text(encoding="utf-8"))
    manifest = json.loads((input_dir / "RUNNING.json").read_text(encoding="utf-8"))
    cases = json.loads((input_dir / "CASES.json").read_text(encoding="utf-8"))
    if result.get("status") != "DO_NOT_PROMOTE" or completed.get("status") != result.get("status"):
        raise RuntimeError("terminal status is missing or inconsistent")
    if int(result.get("num_cases", 0)) != 75 or int(result.get("num_rollouts", 0)) != 225:
        raise RuntimeError("pilot does not contain the frozen sample size")
    if result.get("cases_sha256") != sha256_file(input_dir / "CASES.json"):
        raise RuntimeError("case hash mismatch")
    if result.get("manifest_sha256") != sha256_file(input_dir / "RUNNING.json"):
        raise RuntimeError("manifest hash mismatch")
    rows = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((input_dir / "rollouts").glob("*/*.json"))
    ]
    lookup = {(int(row["case_index"]), str(row["method"])): row for row in rows}
    case_lookup = {int(row["case_index"]): row for row in cases["cases"]}
    if len(rows) != 225 or len(lookup) != 225 or set(case_lookup) != set(range(75)):
        raise RuntimeError("rollout records are incomplete or duplicated")
    for case_index in range(75):
        case = case_lookup[case_index]
        for method in METHODS:
            row = lookup[(case_index, method)]
            if row.get("protocol") != result.get("protocol") or bool(row.get("censored")):
                raise RuntimeError("rollout protocol mismatch or censoring detected")
            if row.get("obstacle_layout_sha256") != case.get("obstacle_layout_sha256"):
                raise RuntimeError("paired static-world hash mismatch")
            if int(row["world_seed"]) != int(case["world_seed"]):
                raise RuntimeError("paired world seed mismatch")
            if abs(float(row["budget_fraction"]) - float(case["budget_fraction"])) > 1e-12:
                raise RuntimeError("paired budget mismatch")
            numeric = [
                row["task_progress"],
                row["total_realized_energy"],
                row["total_steps"],
                row["hocbf_intervention_rate"],
            ]
            if not np.all(np.isfinite(numeric)):
                raise RuntimeError("non-finite terminal metric")
    return rows, result, manifest


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else input_dir / "analysis-output"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    rows, result, manifest = validate(input_dir)
    lookup = {(int(row["case_index"]), str(row["method"])): row for row in rows}

    summaries: list[dict[str, object]] = []
    for method in METHODS:
        selected = [lookup[(case_index, method)] for case_index in range(75)]
        record: dict[str, object] = {"method": method, "n_cases": 75}
        for metric in ("task_completed", "safe_return", "mission_success", "unsafe_terminal_event"):
            count = sum(bool(row[metric]) for row in selected)
            lower, upper = wilson(count, 75)
            record[f"{metric}_count"] = count
            record[f"{metric}_rate"] = count / 75
            record[f"{metric}_wilson95_lower"] = lower
            record[f"{metric}_wilson95_upper"] = upper
        for metric in ("task_progress", "total_realized_energy", "total_steps"):
            values = np.asarray([float(row[metric]) for row in selected])
            record[f"{metric}_mean"] = float(np.mean(values))
            record[f"{metric}_sd"] = float(np.std(values, ddof=1))
            record[f"{metric}_median"] = float(np.median(values))
            record[f"{metric}_iqr"] = float(np.quantile(values, 0.75) - np.quantile(values, 0.25))
        commits = [row["commit_step"] for row in selected]
        record["immediate_commit_count"] = sum(value == 0 for value in commits)
        record["delayed_commit_count"] = sum(value is not None and value > 0 for value in commits)
        record["no_commit_count"] = sum(value is None for value in commits)
        record["timeout_count"] = sum("step_limit" in str(row["end_reason"]) for row in selected)
        summaries.append(record)
    write_csv(output_dir / "summary-table.csv", summaries)

    test_rows: list[dict[str, object]] = []
    for baseline in ("geometry_gate", "ungated"):
        for metric in ("task_completed", "safe_return", "unsafe_terminal_event"):
            differences = np.asarray(
                [
                    float(lookup[(case_index, "certified_meet_gate")][metric])
                    - float(lookup[(case_index, baseline)][metric])
                    for case_index in range(75)
                ]
            )
            base_positive_meet_negative = int(np.sum(differences < 0.0))
            base_negative_meet_positive = int(np.sum(differences > 0.0))
            discordant = base_positive_meet_negative + base_negative_meet_positive
            p_value = (
                float(
                    stats.binomtest(
                        min(base_positive_meet_negative, base_negative_meet_positive),
                        discordant,
                        0.5,
                    ).pvalue
                )
                if discordant
                else 1.0
            )
            interval = result["paired_bootstrap"][f"certified_meet_minus_{baseline}"][metric]
            test_rows.append(
                {
                    "comparison": f"certified_meet_gate - {baseline}",
                    "metric": metric,
                    "direction": "higher_better" if metric != "unsafe_terminal_event" else "lower_better",
                    "unit": "paired_case",
                    "n": 75,
                    "mean_difference": float(np.mean(differences)),
                    "bootstrap95_lower": float(interval["lower_95"]),
                    "bootstrap95_upper": float(interval["upper_95"]),
                    "test": "exact_McNemar_binomial",
                    "statistic": (
                        f"base1_meet0={base_positive_meet_negative};"
                        f"base0_meet1={base_negative_meet_positive}"
                    ),
                    "effect_size": f"paired_risk_difference={np.mean(differences):.9f}",
                    "p_raw": p_value,
                }
            )
        metric = "task_progress"
        differences = np.asarray(
            [
                float(lookup[(case_index, "certified_meet_gate")][metric])
                - float(lookup[(case_index, baseline)][metric])
                for case_index in range(75)
            ]
        )
        shapiro = stats.shapiro(differences)
        wilcoxon = stats.wilcoxon(differences, zero_method="pratt", alternative="two-sided")
        interval = result["paired_bootstrap"][f"certified_meet_minus_{baseline}"][metric]
        test_rows.append(
            {
                "comparison": f"certified_meet_gate - {baseline}",
                "metric": metric,
                "direction": "higher_better",
                "unit": "paired_case",
                "n": 75,
                "mean_difference": float(np.mean(differences)),
                "bootstrap95_lower": float(interval["lower_95"]),
                "bootstrap95_upper": float(interval["upper_95"]),
                "test": "Wilcoxon_signed_rank_Pratt",
                "statistic": float(wilcoxon.statistic),
                "effect_size": f"paired_rank_biserial={paired_rank_biserial(differences):.9f}",
                "p_raw": float(wilcoxon.pvalue),
                "normality_check": f"Shapiro_W={shapiro.statistic:.9f};p={shapiro.pvalue:.9g}",
            }
        )
    adjusted = holm_adjust([float(row["p_raw"]) for row in test_rows])
    for row, value in zip(test_rows, adjusted, strict=True):
        row["p_holm_8"] = value
    write_csv(output_dir / "stats-table.csv", test_rows)

    discordant_rows = []
    for case_index in range(75):
        geometry = lookup[(case_index, "geometry_gate")]
        meet = lookup[(case_index, "certified_meet_gate")]
        if any(
            bool(geometry[metric]) != bool(meet[metric])
            for metric in ("task_completed", "safe_return", "unsafe_terminal_event")
        ):
            discordant_rows.append(
                {
                    "case_index": case_index,
                    "distance_bucket": meet["distance_bucket"],
                    "budget_fraction": meet["budget_fraction"],
                    "geometry_task": geometry["task_completed"],
                    "meet_task": meet["task_completed"],
                    "geometry_safe_return": geometry["safe_return"],
                    "meet_safe_return": meet["safe_return"],
                    "geometry_unsafe": geometry["unsafe_terminal_event"],
                    "meet_unsafe": meet["unsafe_terminal_event"],
                    "geometry_commit_step": geometry["commit_step"],
                    "meet_commit_step": meet["commit_step"],
                    "geometry_end_reason": geometry["end_reason"],
                    "meet_end_reason": meet["end_reason"],
                    "geometry_energy": geometry["total_realized_energy"],
                    "meet_energy": meet["total_realized_energy"],
                }
            )
    write_csv(output_dir / "discordant-cases.csv", discordant_rows)

    # Figure 1: absolute utility/safety outcomes with marginal Wilson intervals.
    metric_specs = (
        ("task_completed", "Task completion"),
        ("safe_return", "Safe return"),
        ("unsafe_terminal_event", "Unsafe event"),
    )
    x = np.arange(len(metric_specs), dtype=np.float64)
    width = 0.24
    fig, axis = plt.subplots(figsize=(7.0, 3.8), constrained_layout=True)
    for offset, method in enumerate(METHODS):
        summary = summaries[offset]
        values, low, high = [], [], []
        for metric, _ in metric_specs:
            value = float(summary[f"{metric}_rate"])
            values.append(value)
            low.append(value - float(summary[f"{metric}_wilson95_lower"]))
            high.append(float(summary[f"{metric}_wilson95_upper"]) - value)
        positions = x + (offset - 1) * width
        axis.bar(
            positions,
            values,
            width,
            color=COLORS[method],
            edgecolor="black",
            linewidth=0.6,
            label=LABELS[method],
            yerr=np.asarray([low, high]),
            capsize=2.5,
        )
        for position, value in zip(positions, values, strict=True):
            axis.text(position, min(1.04, value + 0.055), f"{value:.1%}", ha="center", va="bottom", fontsize=7)
    axis.set_xticks(x, [label for _, label in metric_specs])
    axis.set_ylabel("Case rate (Wilson 95% CI)")
    axis.set_ylim(0.0, 1.12)
    axis.grid(axis="y", alpha=0.25, linewidth=0.6)
    axis.legend(frameon=False, ncol=3, loc="upper center")
    fig.savefig(figures / "figure-01-utility-safety.pdf")
    plt.close(fig)

    # Figure 2: when the two Gates abandon the task.
    fig, axis = plt.subplots(figsize=(5.5, 3.6), constrained_layout=True)
    gate_methods = ("geometry_gate", "certified_meet_gate")
    categories = ("Immediate (step 0)", "Delayed", "No commitment")
    category_colors = ("#D55E00", "#F0E442", "#0072B2")
    bottom = np.zeros(2, dtype=np.float64)
    for category, color, field in zip(
        categories,
        category_colors,
        ("immediate_commit_count", "delayed_commit_count", "no_commit_count"),
        strict=True,
    ):
        values = np.asarray(
            [next(row for row in summaries if row["method"] == method)[field] / 75 for method in gate_methods]
        )
        axis.bar(
            np.arange(2), values, bottom=bottom, color=color, edgecolor="black", linewidth=0.6, label=category
        )
        for index, (value, base) in enumerate(zip(values, bottom, strict=True)):
            if value >= 0.05:
                axis.text(index, base + value / 2, f"{value:.1%}", ha="center", va="center", fontsize=8)
        bottom += values
    axis.set_xticks(np.arange(2), [LABELS[value] for value in gate_methods])
    axis.set_ylabel("Fraction of 75 paired cases")
    axis.set_ylim(0.0, 1.0)
    axis.legend(frameon=False, bbox_to_anchor=(1.02, 1.0), loc="upper left")
    axis.grid(axis="y", alpha=0.2, linewidth=0.6)
    fig.savefig(figures / "figure-02-commitment-timing.pdf")
    plt.close(fig)

    meet = next(row for row in summaries if row["method"] == "certified_meet_gate")
    geometry = next(row for row in summaries if row["method"] == "geometry_gate")
    ungated = next(row for row in summaries if row["method"] == "ungated")
    checks = result["checks"]
    unsafe_meet_geometry = next(
        row
        for row in test_rows
        if row["comparison"] == "certified_meet_gate - geometry_gate"
        and row["metric"] == "unsafe_terminal_event"
    )
    task_meet_geometry = next(
        row
        for row in test_rows
        if row["comparison"] == "certified_meet_gate - geometry_gate"
        and row["metric"] == "task_completed"
    )
    unsafe_meet_ungated = next(
        row
        for row in test_rows
        if row["comparison"] == "certified_meet_gate - ungated"
        and row["metric"] == "unsafe_terminal_event"
    )
    analysis_report = f"""# Bounded RCPS closed-loop pilot: strict analysis

## Analysis questions

1. Does Certified Meet preserve more task utility than the geometry Gate without reducing safety?
2. Do the Gates improve energy safety relative to ungated R3 navigation?
3. Does the offline forked-action certificate transfer to a useful adaptive closed-loop stopping rule?

The unit of analysis is one paired task/world/budget case (`n=75`), not a policy step. All three methods use the same single frozen R3 checkpoint; therefore this pilot measures case-level behavior for this checkpoint, not across-seed algorithmic generalization.

## Artifact validity

- 225/225 unique paired rollout records are present; none are censored.
- Case, world, budget, protocol, manifest, and obstacle-layout hashes agree.
- Terminal state is `{result['status']}` and 7/8 frozen promotion checks pass.
- Runtime was {float(result['wall_clock_seconds']) / 60.0:.2f} minutes; the maximum per-rollout Meet Gate p95 latency was {max(float(lookup[(i, 'certified_meet_gate')]['gate_latency_p95_seconds'] or 0.0) for i in range(75)) * 1000.0:.2f} ms, below the 200 ms policy period.
- `PROGRESS.json` retained the stale label `RUNNING` at 225/225, but the process is absent and matching `RESULT.json` plus `COMPLETED.json` are terminal. This is a non-material telemetry bug to fix before the next runner.

## Main findings

| Method | Task completion | Safe return | Unsafe event | Mean task progress | Immediate return |
|---|---:|---:|---:|---:|---:|
| Ungated | {int(ungated['task_completed_count'])}/75 ({float(ungated['task_completed_rate']):.1%}) | {int(ungated['safe_return_count'])}/75 ({float(ungated['safe_return_rate']):.1%}) | {int(ungated['unsafe_terminal_event_count'])}/75 ({float(ungated['unsafe_terminal_event_rate']):.1%}) | {float(ungated['task_progress_mean']):.3f} | 0/75 |
| Geometry Gate | {int(geometry['task_completed_count'])}/75 ({float(geometry['task_completed_rate']):.1%}) | {int(geometry['safe_return_count'])}/75 ({float(geometry['safe_return_rate']):.1%}) | {int(geometry['unsafe_terminal_event_count'])}/75 ({float(geometry['unsafe_terminal_event_rate']):.1%}) | {float(geometry['task_progress_mean']):.3f} | {int(geometry['immediate_commit_count'])}/75 |
| Certified Meet | {int(meet['task_completed_count'])}/75 ({float(meet['task_completed_rate']):.1%}) | {int(meet['safe_return_count'])}/75 ({float(meet['safe_return_rate']):.1%}) | {int(meet['unsafe_terminal_event_count'])}/75 ({float(meet['unsafe_terminal_event_rate']):.1%}) | {float(meet['task_progress_mean']):.3f} | {int(meet['immediate_commit_count'])}/75 |

Certified Meet gained 2/75 task completions over geometry, but the paired 95% bootstrap interval was [{float(task_meet_geometry['bootstrap95_lower']):.1%}, {float(task_meet_geometry['bootstrap95_upper']):.1%}] and exact McNemar `p={float(task_meet_geometry['p_raw']):.3f}`. It also added one unsafe energy-exhaustion case: +1/75, 95% interval [{float(unsafe_meet_geometry['bootstrap95_lower']):.1%}, {float(unsafe_meet_geometry['bootstrap95_upper']):.1%}], exact McNemar `p={float(unsafe_meet_geometry['p_raw']):.3f}`. Thus neither apparent difference supports Meet superiority over geometry.

Relative to ungated navigation, Meet reduced unsafe terminals from 10/75 to 2/75 (paired difference -8/75; 95% interval [{float(unsafe_meet_ungated['bootstrap95_lower']):.1%}, {float(unsafe_meet_ungated['bootstrap95_upper']):.1%}]). The exact paired test is `p={float(unsafe_meet_ungated['p_raw']):.4f}` before and `p={float(unsafe_meet_ungated['p_holm_8']):.4f}` after Holm correction across the eight declared exploratory comparisons. This safety observation comes with a collapse in task completion from 72/75 to 5/75.

All five Meet task completions occur below 1500 m (four in 100--500 m and one in 500--1500 m); it completes zero of the 45 cases at 1500 m or farther. The current rule therefore does not preserve useful long-range mission behavior.

## Mechanism diagnosis

The failure is not a simulator crash. It is a mismatch between the learned event and the deployed action rule, amplified by sequential reuse:

1. The offline label asks whether **the proposed task action, frozen task continuation, and later return** are feasible under a budget. A low score does not establish that **immediate return from the current state** is feasible.
2. Case 23 is outside the return-now energy set at a 0.20 budget: both Gates reject at step 0 and still exhaust during return.
3. Cases 42 and 57 reject at step 0 but the R3 return policy times out after 4000 steps; ungated task-then-return succeeds in both.
4. In case 72, Meet rejects at step 0 and exhausts after 3382 return steps, while geometry executes four task steps and then returns safely. This is a direct counterexample to `reject continuation => immediate return is safe`.
5. Only 19/75 Meet cases and 17/75 geometry cases pass the Gate at the initial state, versus 50.15% and 46.41% selective coverage on the independent forked-action confirmation distribution. These are different sampling distributions, so the gap is evidence of deployment shift rather than a like-for-like accuracy drop.
6. Pointwise thresholds are applied repeatedly with an absorbing stop. Meet commits immediately in 56/75 cases and eventually commits in 70/75. A pointwise scene-risk calibration does not control trajectory-level repeated stopping, so rare low scores compound into systematic early abandonment.

## Decision

`DO_NOT_PROMOTE` is the correct decision. The present Gate is useful evidence that learned resource feasibility can suppress energy exhaustion, but it is not a useful energy-sustainable navigation policy: most safety is purchased by abandoning the task, and return-now safety is not represented by the learned target.

The next method should separate two viability functions: `V_continue(s,a,B)` for task-then-return and `V_return(s,B)` for the backup return policy. Continue only when the first is safe; commit to return only when the second is safe. If neither is safe, choose a short recovery action that maximizes next-state return viability rather than blindly retargeting. The stopping rule also needs trajectory-level calibration (time-uniform risk control or a preregistered persistent/hysteretic trigger), not repeated use of a pointwise RCPS threshold.

## Claim Candidates

- Claim:
  - Source evidence: ER-20260905-closed-loop-pilot-01; 75 paired cases, 225 complete rollouts.
  - Allowed wording: "In a bounded single-checkpoint pilot, Certified Meet reduced unsafe terminal events relative to ungated execution, but task completion fell sharply because the Gate usually committed to return at or near mission start."
  - Forbidden stronger wording: "Certified Meet guarantees safe energy-sustainable navigation" or "Meet is superior to geometry."
  - Uncertainty: one frozen actor/checkpoint and one 75-case set; Holm-adjusted inference does not support all safety contrasts.
  - Next check: train/calibrate distinct continuation and return-now viability heads, then test a fresh paired pilot with an absolute task-utility Gate.
  - Decision: revise

- Claim:
  - Source evidence: ER-20260905-closed-loop-pilot-01, cases 23, 42, 57, and 72.
  - Allowed wording: "The pilot supplies counterexamples showing that low task-continuation feasibility does not imply immediate-return feasibility."
  - Forbidden stronger wording: "Immediate return always fails after rejection."
  - Uncertainty: mechanism is observed in a small number of deterministic cases and still needs a dedicated return-only counterfactual experiment.
  - Next check: paired return-now rollouts from every Gate decision state.
  - Decision: keep
"""
    (output_dir / "analysis-report.md").write_text(analysis_report, encoding="utf-8")

    stats_lines = [
        "# Statistical appendix",
        "",
        "## Design and assumptions",
        "",
        "The repeated-measure unit is the paired case (`n=75`). Policy steps are not treated as independent. Binary outcomes use exact McNemar/binomial tests on discordant pairs. Task-progress differences are zero-inflated and strongly non-normal (Shapiro tests reject normality), so the Wilcoxon signed-rank test with Pratt zero handling is used. All eight exploratory comparison p-values are Holm-corrected as one family. Bootstrap intervals are the frozen 10,000-draw paired case intervals stored by the runner.",
        "",
        "There is only one actor training seed/checkpoint. Case-level intervals quantify variation over the frozen task/world generator, not training-seed uncertainty.",
        "",
        "## Exact comparisons",
        "",
        "| Comparison | Metric | Mean paired difference | Paired bootstrap 95% CI | Test/statistic | Raw p | Holm p | Effect size |",
        "|---|---|---:|---:|---|---:|---:|---|",
    ]
    for row in test_rows:
        stats_lines.append(
            f"| {row['comparison']} | {row['metric']} | {float(row['mean_difference']):.6f} | "
            f"[{float(row['bootstrap95_lower']):.6f}, {float(row['bootstrap95_upper']):.6f}] | "
            f"{row['test']}: {row['statistic']} | {float(row['p_raw']):.6g} | "
            f"{float(row['p_holm_8']):.6g} | {row['effect_size']} |"
        )
    stats_lines.extend(
        [
            "",
            "## Frozen decision Gate",
            "",
            f"Seven of eight checks passed. The failed check was `meet_unsafe_not_worse_than_geometry` ({checks['meet_unsafe_not_worse_than_geometry']}). The noninferiority check passed only at its allowed boundary: Meet had exactly one fewer safe return than geometry (71 versus 72). Promotion remains false regardless of exploratory p-values.",
            "",
            "## Multiplicity and interpretation",
            "",
            "The preregistered promotion decision is threshold-based, not significance-based. Inferential tests here are diagnostic. No post-hoc p-value can reverse `DO_NOT_PROMOTE`. Mission success is omitted from the test family because it duplicates task completion for both gated arms in these data.",
        ]
    )
    (output_dir / "stats-appendix.md").write_text("\n".join(stats_lines) + "\n", encoding="utf-8")

    catalog = """# Figure catalog

## figure-01-utility-safety.pdf

- Purpose: compare the absolute utility and safety outcomes of the three paired arms.
- Data source: all 225 terminal rollout records; 75 paired cases per arm.
- Caption requirements: bars are marginal case rates; error bars are Wilson 95% intervals, not across-seed intervals; higher is better for task completion and safe return, lower is better for unsafe events.
- Key observation: both Gates reduce unsafe events, but task completion collapses from 96.0% ungated to 4.0% geometry and 6.7% Meet.
- Interpretation: the current safety gain is dominated by early mission abandonment, so it is not evidence of energy-sustainable task execution.
- Caveat: marginal Wilson intervals do not encode pairing; exact paired contrasts are in `stats-appendix.md`.

## figure-02-commitment-timing.pdf

- Purpose: identify when the Gate changes the mission rather than merely whether it changes it.
- Data source: `commit_step` for 75 cases in each gated arm.
- Caption requirements: immediate means step 0, delayed means step >0, and no commitment means task execution was never preempted.
- Key observation: geometry commits immediately in 58/75 cases and Meet in 56/75; only 3 and 5 cases, respectively, avoid early commitment.
- Interpretation: repeated pointwise thresholding creates a near-always-return controller and explains the utility collapse.
- Caveat: the plot diagnoses behavior but does not by itself distinguish distribution shift from sequential multiple-testing effects.
"""
    (output_dir / "figure-catalog.md").write_text(catalog, encoding="utf-8")
    atomic_json(
        output_dir / "ANALYSIS.json",
        {
            "status": "VALID_DO_NOT_PROMOTE",
            "evidence_id": "ER-20260905-closed-loop-pilot-01",
            "input_dir": input_dir,
            "input_result_sha256": sha256_file(input_dir / "RESULT.json"),
            "unit_of_analysis": "paired_case",
            "n_cases": 75,
            "n_rollouts": 225,
            "holm_family_size": len(test_rows),
            "failed_frozen_checks": [name for name, passed in checks.items() if not passed],
            "analysis_files": [
                "analysis-report.md",
                "stats-appendix.md",
                "figure-catalog.md",
                "summary-table.csv",
                "stats-table.csv",
                "discordant-cases.csv",
                "figures/figure-01-utility-safety.pdf",
                "figures/figure-02-commitment-timing.pdf",
            ],
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
