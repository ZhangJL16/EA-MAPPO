#!/usr/bin/env python3
"""Paired analysis of the immutable 500-task scratch SAC/PPO evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


BUCKETS = ["100-500", "500-1500", "1500-2500", "2500-4000", ">4000"]
COLORS = {"SAC": "#0072B2", "PPO": "#E69F00"}


def load_rows(root: Path, algorithm: str) -> list[dict[str, Any]]:
    path = root / algorithm.lower() / "evaluation_stratified.json"
    return list(json.loads(path.read_text(encoding="utf-8")))


def load_training(root: Path, algorithm: str) -> dict[str, Any]:
    path = root / algorithm.lower() / "training_records.json"
    return dict(json.loads(path.read_text(encoding="utf-8")))


def validate_pair(sac: list[dict[str, Any]], ppo: list[dict[str, Any]]) -> None:
    if len(sac) != 500 or len(ppo) != 500:
        raise ValueError("formal analysis requires exactly 500 rows per algorithm")
    for left, right in zip(sac, ppo, strict=True):
        keys = ("source_task_index", "seed", "distance_bucket", "straight_line_distance")
        if any(left[key] != right[key] for key in keys):
            raise ValueError(f"paired task mismatch at evaluation index {left['evaluation_index']}")
    for bucket in BUCKETS:
        if sum(row["distance_bucket"] == bucket for row in sac) != 100:
            raise ValueError(f"bucket {bucket} does not contain 100 tasks")


def paired_binary(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    left_only = int(np.sum(left & ~right))
    right_only = int(np.sum(~left & right))
    discordant = left_only + right_only
    pvalue = float(stats.binomtest(left_only, discordant, 0.5).pvalue) if discordant else 1.0
    return {
        "sac_success": int(left.sum()),
        "ppo_success": int(right.sum()),
        "sac_only": left_only,
        "ppo_only": right_only,
        "both": int(np.sum(left & right)),
        "neither": int(np.sum(~left & ~right)),
        "risk_difference_pp": float(100.0 * np.mean(left.astype(float) - right.astype(float))),
        "matched_odds_ratio": float(left_only / right_only) if right_only else None,
        "mcnemar_exact_p": pvalue,
    }


def holm(values: list[float]) -> list[float]:
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = (len(values) - rank) * values[index]
        running = max(running, candidate)
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def stratified_bootstrap(
    sac: list[dict[str, Any]],
    ppo: list[dict[str, Any]],
    value,
    *,
    replicates: int = 20_000,
    seed: int = 20260906,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    differences: list[np.ndarray] = []
    for bucket in BUCKETS:
        indices = np.asarray([i for i, row in enumerate(sac) if row["distance_bucket"] == bucket])
        per_task = np.asarray([value(sac[i]) - value(ppo[i]) for i in indices], dtype=float)
        draw = rng.integers(0, len(indices), size=(replicates, len(indices)))
        differences.append(per_task[draw].mean(axis=1))
    samples = np.stack(differences).mean(axis=0)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def quantiles(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)),
        "min": float(np.min(values)),
        "q25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "q75": float(np.quantile(values, 0.75)),
        "q90": float(np.quantile(values, 0.90)),
        "q95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
    }


def wilson(successes: int, n: int) -> tuple[float, float]:
    z = 1.959963984540054
    p = successes / n
    den = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / den
    half = z * np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / den
    return float(center - half), float(center + half)


def plot_outcomes(output: Path, sac: list[dict[str, Any]], ppo: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True)
    x = np.arange(len(BUCKETS))
    for axis, field, label in zip(
        axes, ("goal_reached", "safe_goal"), ("Goal arrival", "Zero-contact arrival"), strict=True
    ):
        for offset, (name, rows) in zip((-0.08, 0.08), (("SAC", sac), ("PPO", ppo)), strict=True):
            rates, lower, upper = [], [], []
            for bucket in BUCKETS:
                values = [bool(row[field]) for row in rows if row["distance_bucket"] == bucket]
                rate = np.mean(values)
                lo, hi = wilson(sum(values), len(values))
                rates.append(rate)
                lower.append(rate - lo)
                upper.append(hi - rate)
            axis.errorbar(
                x + offset,
                rates,
                yerr=np.asarray([lower, upper]),
                marker="o" if name == "SAC" else "s",
                linestyle="-" if name == "SAC" else "--",
                linewidth=1.7,
                markersize=4.5,
                capsize=2,
                color=COLORS[name],
                label=name,
            )
        axis.set_title(label)
        axis.set_xticks(x, ["0.1–0.5", "0.5–1.5", "1.5–2.5", "2.5–4", ">4"], rotation=25)
        axis.set_xlabel("Start–goal distance (km)")
        axis.grid(alpha=0.25)
        axis.set_ylim(0.0, 1.04)
    axes[0].set_ylabel("Task fraction")
    axes[1].legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(output / "fig1_distance_outcomes.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_contacts(output: Path, sac: list[dict[str, Any]], ppo: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    for name, rows in (("SAC", sac), ("PPO", ppo)):
        values = np.sort(np.asarray([row["collision_count"] for row in rows], dtype=float))
        axes[0].step(np.log10(1.0 + values), np.arange(1, len(values) + 1) / len(values), where="post",
                     color=COLORS[name], linewidth=1.8, label=name)
    axes[0].set_xlabel(r"$\log_{10}(1 + \mathrm{unified\ contacts})$")
    axes[0].set_ylabel("Empirical CDF")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)
    ppo_timeouts = [row for row in ppo if not row["goal_reached"]]
    groups = [[row["remaining_distance"] for row in ppo_timeouts if row["distance_bucket"] == bucket]
              for bucket in BUCKETS]
    axes[1].boxplot(groups, labels=["0.1–0.5", "0.5–1.5", "1.5–2.5", "2.5–4", ">4"],
                    showfliers=False)
    axes[1].set_xlabel("Start–goal distance (km)")
    axes[1].set_ylabel("PPO timeout remaining distance (m)")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "fig2_contact_and_timeout_distributions.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_ppo_training(output: Path, training: dict[str, Any]) -> None:
    updates = training["updates"]
    transitions = np.asarray([row["transitions"] for row in updates])
    metrics = [row["metrics"] for row in updates]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.65))
    series = (
        ("train/approx_kl", "Approximate KL", 0.02),
        ("train/explained_variance", "Explained variance", None),
        ("train/std", "Policy action std.", None),
    )
    for axis, (key, label, reference) in zip(axes, series, strict=True):
        values = np.asarray([row[key] for row in metrics], dtype=float)
        axis.plot(transitions / 1000.0, values, color=COLORS["PPO"], linewidth=1.2)
        if reference is not None:
            axis.axhline(reference, color="#000000", linestyle=":", linewidth=1.1, label="target KL")
            axis.legend(frameon=False, fontsize=7)
        axis.set_xlabel("Transitions (thousands)")
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "fig3_ppo_optimization_diagnostics.pdf", bbox_inches="tight")
    plt.close(fig)


def analyse(root: Path, output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    sac, ppo = load_rows(root, "SAC"), load_rows(root, "PPO")
    validate_pair(sac, ppo)
    result: dict[str, Any] = {"unit": "paired held-out task conditional on one trained seed"}
    for field in ("goal_reached", "safe_goal"):
        left = np.asarray([row[field] for row in sac], dtype=bool)
        right = np.asarray([row[field] for row in ppo], dtype=bool)
        item = paired_binary(left, right)
        lo, hi = stratified_bootstrap(sac, ppo, lambda row, f=field: float(row[f]))
        item["stratified_bootstrap_95ci_pp"] = [100.0 * lo, 100.0 * hi]
        bucket_items = []
        for bucket in BUCKETS:
            idx = [i for i, row in enumerate(sac) if row["distance_bucket"] == bucket]
            bucket_items.append(paired_binary(left[idx], right[idx]))
        adjusted = holm([item["mcnemar_exact_p"] for item in bucket_items])
        for bucket, bucket_item, p_adj in zip(BUCKETS, bucket_items, adjusted, strict=True):
            bucket_item["bucket"] = bucket
            bucket_item["holm_adjusted_p"] = p_adj
        item["by_bucket"] = bucket_items
        result[field] = item

    for field in ("collision_count", "steps", "energy"):
        left = np.asarray([row[field] for row in sac], dtype=float)
        right = np.asarray([row[field] for row in ppo], dtype=float)
        difference = left - right
        lo, hi = stratified_bootstrap(sac, ppo, lambda row, f=field: float(row[f]))
        wilcoxon = stats.wilcoxon(left, right, zero_method="pratt", alternative="two-sided")
        result[field] = {
            "sac": quantiles(left),
            "ppo": quantiles(right),
            "paired_difference_sac_minus_ppo": quantiles(difference),
            "stratified_bootstrap_mean_difference_95ci": [lo, hi],
            "wilcoxon_statistic": float(wilcoxon.statistic),
            "wilcoxon_p": float(wilcoxon.pvalue),
        }

    both_goal = np.asarray([a["goal_reached"] and b["goal_reached"] for a, b in zip(sac, ppo, strict=True)])
    sac_ratio = np.asarray([row["success_path_ratio"] for row, keep in zip(sac, both_goal, strict=True) if keep])
    ppo_ratio = np.asarray([row["success_path_ratio"] for row, keep in zip(ppo, both_goal, strict=True) if keep])
    ratio_test = stats.wilcoxon(sac_ratio, ppo_ratio, zero_method="pratt", alternative="two-sided")
    result["path_ratio_common_success"] = {
        "n": int(both_goal.sum()),
        "sac": quantiles(sac_ratio),
        "ppo": quantiles(ppo_ratio),
        "paired_difference_sac_minus_ppo": quantiles(sac_ratio - ppo_ratio),
        "wilcoxon_statistic": float(ratio_test.statistic),
        "wilcoxon_p": float(ratio_test.pvalue),
    }
    for name, rows in (("sac", sac), ("ppo", ppo)):
        total_steps = sum(int(row["steps"]) for row in rows)
        total_contacts = sum(int(row["collision_count"]) for row in rows)
        ordered_contacts = np.sort(
            np.asarray([int(row["collision_count"]) for row in rows], dtype=float)
        )[::-1]
        result.setdefault("contact_step_rate", {})[name] = {
            "contacts": total_contacts,
            "policy_steps": total_steps,
            "rate": total_contacts / total_steps,
            "tasks_with_contact": sum(int(row["collision_count"]) > 0 for row in rows),
            "goal_after_contact": sum(
                bool(row["goal_reached"]) and int(row["collision_count"]) > 0 for row in rows
            ),
            "timeout_after_contact": sum(
                not bool(row["goal_reached"]) and int(row["collision_count"]) > 0 for row in rows
            ),
            "clean_timeout": sum(
                not bool(row["goal_reached"]) and int(row["collision_count"]) == 0 for row in rows
            ),
            "top_5_task_contact_share": float(ordered_contacts[:5].sum() / total_contacts),
        }
    ppo_timeouts = [row for row in ppo if not row["goal_reached"]]
    result["ppo_timeouts"] = {
        "n": len(ppo_timeouts),
        "remaining_distance": quantiles(np.asarray([row["remaining_distance"] for row in ppo_timeouts])),
        "within_5m": sum(row["remaining_distance"] <= 5 for row in ppo_timeouts),
        "within_20m": sum(row["remaining_distance"] <= 20 for row in ppo_timeouts),
        "within_60m": sum(row["remaining_distance"] <= 60 for row in ppo_timeouts),
    }

    sac_train, ppo_train = load_training(root, "SAC"), load_training(root, "PPO")
    ppo_updates = ppo_train["updates"]
    cumulative = np.asarray([row["optimizer_steps"]["joint_updates"] for row in ppo_updates], dtype=int)
    increments = np.diff(np.concatenate(([0], cumulative)))
    effective_epochs = increments / 16.0
    metrics = [row["metrics"] for row in ppo_updates]
    result["training"] = {
        "sac": {
            "seconds": sac_train["training_seconds"],
            "actor_updates": sac_train["actor_updates"],
            "critic_updates": sac_train["critic_updates"],
            "episodes_consumed": len(sac_train["episodes"]),
            "final_entropy_coefficient": sac_train["updates"][-1]["metrics"]["train/ent_coef"],
        },
        "ppo": {
            "seconds": ppo_train["training_seconds"],
            "joint_updates": ppo_train["joint_updates"],
            "episodes_consumed": len(ppo_train["episodes"]),
            "effective_epochs_per_rollout": quantiles(effective_epochs),
            "blocks_below_10_epochs": int(np.sum(effective_epochs < 10.0)),
            "approx_kl": quantiles(np.asarray([row["train/approx_kl"] for row in metrics])),
            "clip_fraction": quantiles(np.asarray([row["train/clip_fraction"] for row in metrics])),
            "explained_variance": quantiles(np.asarray([row["train/explained_variance"] for row in metrics])),
            "action_std": quantiles(np.asarray([row["train/std"] for row in metrics])),
        },
    }
    (output / "statistics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    plot_outcomes(output, sac, ppo)
    plot_contacts(output, sac, ppo)
    plot_ppo_training(output, ppo_train)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("artifacts/recovery_sac_ppo_scratch_20260906_v1"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/recovery_sac_ppo_scratch_analysis_20260906_v1"))
    args = parser.parse_args()
    result = analyse(args.input, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
