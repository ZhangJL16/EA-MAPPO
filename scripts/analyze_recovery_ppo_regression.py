"""Strict analysis bundle for the completed recovery-PPO regression."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest, spearmanr, wilcoxon


def load_groups(path: Path) -> list[dict]:
    return [
        r for p in sorted(path.glob("cohort_*.json")) for r in json.loads(p.read_text())
    ]


def holm(rows: list[dict]) -> None:
    order = sorted(range(len(rows)), key=lambda i: rows[i]["p_raw"])
    running = 0.0
    for rank, index in enumerate(order):
        adjusted = min(1.0, (len(rows) - rank) * rows[index]["p_raw"])
        running = max(running, adjusted)
        rows[index]["p_holm"] = running


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trained", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    modes = ("deterministic", "stochastic")
    paired = []
    summaries = {}
    rng = np.random.default_rng(20260906)
    for mode in modes:
        old = sorted(
            load_groups(args.baseline / f"control_{mode}"), key=lambda r: r["seed"]
        )
        new = sorted(
            load_groups(args.trained / f"evaluation_0096_{mode}"),
            key=lambda r: r["seed"],
        )
        assert (
            [r["seed"] for r in old]
            == [r["seed"] for r in new]
            == list(range(593800001, 593800051))
        )
        summaries[mode] = {"baseline": {}, "trained": {}}
        for label, rows in (("baseline", old), ("trained", new)):
            summaries[mode][label] = {
                "goal": sum(r["goal_reached"] for r in rows),
                "safe": sum(r["safe_goal"] for r in rows),
                "timeout": sum(not r["goal_reached"] for r in rows),
                "collision_mean": float(np.mean([r["collision_count"] for r in rows])),
                "collision_median": float(
                    np.median([r["collision_count"] for r in rows])
                ),
            }
        for metric in ("goal_reached", "safe_goal"):
            a = np.array([r[metric] for r in old], int)
            b = np.array([r[metric] for r in new], int)
            gained = int(np.sum((b == 1) & (a == 0)))
            lost = int(np.sum((b == 0) & (a == 1)))
            paired.append(
                {
                    "mode": mode,
                    "metric": metric,
                    "difference": float((b - a).mean()),
                    "gained": gained,
                    "lost": lost,
                    "p_raw": float(binomtest(gained, gained + lost).pvalue),
                }
            )
        d = np.array([r["collision_count"] for r in new]) - np.array(
            [r["collision_count"] for r in old]
        )
        boot = d[rng.integers(0, 50, size=(20000, 50))].mean(1)
        summaries[mode]["collision_difference"] = {
            "mean": float(d.mean()),
            "median": float(np.median(d)),
            "bootstrap95": np.quantile(boot, [0.025, 0.975]).tolist(),
            "wilcoxon_p": cast(float, wilcoxon(d)[1]),
            "nonzero_pairs": int(np.sum(d != 0)),
        }
    holm(paired)
    updates = [
        json.loads(x)
        for x in (args.trained / "model/updates.jsonl").read_text().splitlines()
    ]
    episodes = [
        json.loads(x)
        for x in (args.trained / "model/training_episodes.jsonl")
        .read_text()
        .splitlines()
    ]
    blocks = []
    for start in range(65, 97, 8):
        er = [r for r in episodes if start <= r["cohort"] < start + 8]
        ur = [r for r in updates if start <= r["cohort"] < start + 8]
        blocks.append(
            {
                "start": start,
                "tasks": len(er),
                "goal": sum(r["goal_reached"] for r in er),
                "safe": sum(r["safe_goal"] for r in er),
                "mean_collision": float(np.mean([r["collision_count"] for r in er])),
                "mean_steps": float(np.mean([r["steps"] for r in er])),
                "gradient_steps": sum(r["gradient_steps"] for r in ur),
            }
        )
    correlation = spearmanr(
        [u["physical_steps"] for u in updates], [u["gradient_steps"] for u in updates]
    )
    rho = cast(float, correlation.statistic)
    p_s = cast(float, correlation.pvalue)
    replay = []
    for path in sorted(args.replay.glob("replay_*.json")):
        row = json.loads(path.read_text())
        d = row["diagnostic"]
        replay.append(
            {
                "cohort": row["cohort"],
                "mc_value_r2": d["all"]["mc_value_r2"],
                "gae_mc_correlation": d["all"]["gae_mc_correlation"],
                "gradient_cosine": d["actor_gae_mc_gradient_cosine"],
                "sign_disagreement": d["all"]["gae_mc_sign_disagreement"],
                "analytic_kl_mean": d["post_update"]["full_batch_analytic_kl_mean"],
                "analytic_kl_p95": d["post_update"]["full_batch_analytic_kl_p95"],
            }
        )
    analysis = {
        "summaries": summaries,
        "paired_binary": paired,
        "training_blocks": blocks,
        "step_update_spearman": {"rho": float(rho), "p": float(p_s)},
        "early_kl_stop_cohorts": sum(u["early_kl_stop"] for u in updates),
        "replay": replay,
    }
    (out / "summary.json").write_text(json.dumps(analysis, indent=2) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    for ax, mode in zip(axes, modes):
        x = np.arange(3)
        w = 0.35
        ax.bar(
            x - w / 2,
            [summaries[mode]["baseline"][k] for k in ("goal", "safe", "timeout")],
            w,
            label="Frozen control",
        )
        ax.bar(
            x + w / 2,
            [summaries[mode]["trained"][k] for k in ("goal", "safe", "timeout")],
            w,
            label="Recovery PPO",
        )
        ax.set_xticks(x, ["Goal", "Collision-free goal", "Timeout"])
        ax.set_title(mode.capitalize())
        ax.set_ylabel("Tasks (n=50)")
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(
            figures / f"figure-01-outcomes.{suffix}", dpi=220, bbox_inches="tight"
        )
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    x = [b["start"] for b in blocks]
    axes[0].plot(x, [b["goal"] for b in blocks], "o-", label="Goal")
    axes[0].plot(x, [b["safe"] for b in blocks], "s-", label="Collision-free goal")
    axes[0].set(xlabel="First cohort in 64-task block", ylabel="Tasks (n=64)")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.25)
    axes[1].plot(
        x, [b["mean_collision"] for b in blocks], "o-", label="Mean collision count"
    )
    ax2 = axes[1].twinx()
    ax2.plot(
        x,
        [b["gradient_steps"] for b in blocks],
        "s--",
        color="tab:red",
        label="Gradient updates",
    )
    axes[1].set(xlabel="First cohort in block", ylabel="Collision count / task")
    ax2.set_ylabel("Gradient updates / block", color="tab:red")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(
            figures / f"figure-02-training-blocks.{suffix}",
            dpi=220,
            bbox_inches="tight",
        )
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax = cast(Any, ax)
    x = np.arange(len(replay))
    w = 0.24
    ax.bar(x - w, [r["mc_value_r2"] for r in replay], w, label="MC value R²")
    ax.bar(x, [r["gae_mc_correlation"] for r in replay], w, label="GAE–MC corr.")
    ax.bar(
        x + w, [r["gradient_cosine"] for r in replay], w, label="Actor-gradient cosine"
    )
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x, [r["cohort"] for r in replay])
    ax.set(xlabel="Replayed cohort", ylabel="Diagnostic value")
    ax.legend(frameon=False, ncol=3)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(
            figures / f"figure-03-credit-diagnostics.{suffix}",
            dpi=220,
            bbox_inches="tight",
        )
    plt.close(fig)

    rows_md = "\n".join(
        f"| {m} | {v['baseline']['goal']} | {v['trained']['goal']} | {v['baseline']['safe']} | {v['trained']['safe']} | {v['baseline']['collision_mean']:.2f} | {v['trained']['collision_mean']:.2f} |"
        for m, v in summaries.items()
    )
    (
        out / "analysis-report.md"
    ).write_text(f"""# Recovery PPO regression: evidence audit

## Question

Does reward-only PPO continuation under the locked nonterminal-collision environment preserve or improve navigation and collision-free arrival relative to the identical frozen starting policy?

| Mode | Goal before | Goal after | Safe before | Safe after | Collisions before | Collisions after |
|---|---:|---:|---:|---:|---:|---:|
{rows_md}

## Findings

The continuation causes a large paired loss in goal and collision-free arrival in both action modes. Exact paired tests remain below .05 after Holm correction; this establishes regression on these 50 fixed development scenes, not across training seeds. Mean collision-count differences are heavy-tailed and their bootstrap intervals include zero, so the evidence does not establish a uniform collision-count increase.

All four replay cohorts match task outcomes, lengths, collision counts and raw returns. The replay therefore rules out silent record mismatch for the audited batches. Across cohorts65/81/89/96, actor GAE-vs-MC gradient cosine is {", ".join(f"{r["gradient_cosine"]:.3f}" for r in replay)}. Overall GAE–MC advantage correlations are {", ".join(f"{r["gae_mc_correlation"]:.3f}" for r in replay)}. This supports testing long-horizon credit estimation; it does not prove MC will improve policy learning.

The proposed simple explanation “longer batches mechanically cause more optimizer steps and therefore regression” is not supported by the cohort-level Spearman check (rho={rho:.3f}, p={p_s:.3g}). It remains true that only eight tasks form each on-policy batch and late batches are dominated by long timeouts, but causation is not established.

## Claim candidates

- **Keep:** Under one training fork and 50 fixed development scenes per mode, this recovery-PPO continuation substantially regressed goal attainment and collision-free arrival.
- **Keep, weak:** Replayed batches show poor alignment between short-trace GAE and complete-return actor signals, motivating a single-factor actor-return estimator test.
- **Discard:** PPO is intrinsically unable to learn recovery navigation.
- **Discard:** Collision counts significantly increased; paired uncertainty is too wide.
- **Discard:** Excess optimizer steps alone caused collapse.

## Limits and next check

This is one inherited checkpoint and one training RNG fork on already viewed scenes. GAE–MC comparisons use finite replay samples and MC has higher variance. Test actor MC advantage against a matched GAE control while preserving critic targets, optimizer, network, locked environment and reward. Do not promote either arm without new-seed and held-out evaluation.
""")
    table = "\n".join(
        f"| {r['mode']} | {r['metric']} | {r['difference']:.2f} | {r['gained']} | {r['lost']} | {r['p_raw']:.3g} | {r['p_holm']:.3g} |"
        for r in paired
    )
    (out / "stats-appendix.md").write_text(f"""# Statistical appendix

Unit: paired evaluation scene, n=50 per action mode. Binary outcomes use two-sided exact McNemar/binomial tests on discordant pairs. Four tests (two outcomes × two modes) use Holm correction.

| Mode | Metric | After-before | Gained | Lost | Raw p | Holm p |
|---|---|---:|---:|---:|---:|---:|
{table}

Collision counts are zero-inflated/heavy-tailed. Paired mean differences use 20,000 scene bootstrap resamples; Wilcoxon is supplementary. Deterministic: {summaries["deterministic"]["collision_difference"]}. Stochastic: {summaries["stochastic"]["collision_difference"]}.

Training contains32 cohorts,256 tasks and581,518 new environment steps. Gradient-update quartiles are {np.quantile([u["gradient_steps"] for u in updates], [0, 0.25, 0.5, 0.75, 1]).tolist()};29/32 cohorts stopped early by the KL rule. Physical-steps/update-count Spearman rho={rho:.4f}, p={p_s:.4f}. Scene-level inference does not capture training-seed variability.
""")
    (out / "figure-catalog.md").write_text("""# Figure catalog

## figure-01-outcomes
Purpose: paired-scene performance regression. Bars show exact task counts, not uncertainty. Notice both goal measures collapse while timeouts increase. This blocks continuation of the trained checkpoint.

## figure-02-training-blocks
Purpose: descriptive training evolution in four consecutive 64-task blocks. Blocks use different task seeds, so the plot is not a controlled checkpoint evaluation. Notice the last block has fewer arrivals and many more updates; do not infer causation from co-movement.

## figure-03-credit-diagnostics
Purpose: assess whether the bootstrapped GAE signal aligns with complete-return evidence on exactly replayed cohorts. Notice weak/unstable gradient alignment, including near zero at cohort81. This motivates, but does not validate, the MC actor test.
""")


if __name__ == "__main__":
    main()
