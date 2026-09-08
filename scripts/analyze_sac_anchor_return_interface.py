#!/usr/bin/env python3
"""Scene-clustered paired analysis of raw versus HOCBF SAC return rollouts."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


BUCKETS = ["100-500", "500-1500", "1500-2500", "2500-4000", ">4000"]
COLORS = {"raw": "#0072B2", "hocbf": "#E69F00"}


def load_pair(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw = list(json.loads((root / "raw" / "evaluation.json").read_text(encoding="utf-8")))
    hocbf = list(json.loads((root / "hocbf" / "evaluation.json").read_text(encoding="utf-8")))
    if len(raw) != 310 or len(hocbf) != 310:
        raise ValueError("complete diagnostic requires 310 rows per arm")
    hocbf_by_id = {row["anchor_id"]: row for row in hocbf}
    if len(hocbf_by_id) != len(hocbf):
        raise ValueError("duplicate HOCBF anchor ID")
    ordered = []
    for left in raw:
        right = hocbf_by_id.get(left["anchor_id"])
        if right is None:
            raise ValueError(f"missing paired anchor {left['anchor_id']}")
        for key in ("anchor_index", "scene_index", "source_transition_index", "distance_bucket"):
            if left[key] != right[key]:
                raise ValueError(f"paired provenance mismatch for {left['anchor_id']}: {key}")
        ordered.append(right)
    if len({row["scene_index"] for row in raw}) != 150:
        raise ValueError("expected 150 source-scene clusters")
    return raw, ordered


def q(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)),
        "q25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "q75": float(np.quantile(values, 0.75)),
        "q90": float(np.quantile(values, 0.90)),
        "q95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
    }


def scene_differences(
    raw: list[dict[str, Any]],
    hocbf: list[dict[str, Any]],
    value: Callable[[dict[str, Any]], float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grouped: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(raw):
        grouped[int(row["scene_index"])].append(index)
    scenes, differences, buckets = [], [], []
    for scene in sorted(grouped):
        indices = grouped[scene]
        scenes.append(scene)
        differences.append(
            np.mean([value(hocbf[i]) - value(raw[i]) for i in indices])
        )
        labels = {str(raw[i]["distance_bucket"]) for i in indices}
        if len(labels) != 1:
            raise ValueError("one source scene crosses distance buckets")
        buckets.append(BUCKETS.index(labels.pop()))
    return np.asarray(scenes), np.asarray(differences), np.asarray(buckets)


def stratified_scene_bootstrap(
    differences: np.ndarray,
    bucket_ids: np.ndarray,
    *,
    replicates: int = 50_000,
    seed: int = 20260906,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    sampled = []
    for bucket in range(len(BUCKETS)):
        values = differences[bucket_ids == bucket]
        draw = rng.integers(0, len(values), size=(replicates, len(values)))
        sampled.append(values[draw].mean(axis=1))
    means = np.stack(sampled).mean(axis=0)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def sign_flip_p(differences: np.ndarray, *, seed: int = 20260906, replicates: int = 100_000) -> float:
    nonzero = differences[differences != 0.0]
    if len(nonzero) == 0:
        return 1.0
    rng = np.random.default_rng(seed)
    observed = abs(float(np.mean(differences)))
    signs = rng.choice((-1.0, 1.0), size=(replicates, len(nonzero)))
    null = np.abs((signs * nonzero).sum(axis=1) / len(differences))
    return float((1 + np.sum(null >= observed)) / (replicates + 1))


def holm(values: list[float]) -> list[float]:
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(values) - rank) * values[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def paired_summary(
    raw: list[dict[str, Any]],
    hocbf: list[dict[str, Any]],
    value: Callable[[dict[str, Any]], float],
    *,
    scale: float = 1.0,
) -> dict[str, Any]:
    scenes, differences, bucket_ids = scene_differences(raw, hocbf, value)
    lo, hi = stratified_scene_bootstrap(differences, bucket_ids)
    by_bucket = []
    bucket_ps = []
    for index, bucket in enumerate(BUCKETS):
        selected = differences[bucket_ids == index]
        pvalue = sign_flip_p(selected, seed=20260906 + index)
        bucket_ps.append(pvalue)
        by_bucket.append(
            {
                "bucket": bucket,
                "scenes": len(selected),
                "mean_hocbf_minus_raw": float(scale * np.mean(selected)),
                "sign_flip_p": pvalue,
            }
        )
    for item, adjusted in zip(by_bucket, holm(bucket_ps), strict=True):
        item["holm_adjusted_p"] = adjusted
    return {
        "scenes": len(scenes),
        "scene_mean_hocbf_minus_raw": float(scale * np.mean(differences)),
        "stratified_scene_bootstrap_95ci": [scale * lo, scale * hi],
        "scene_sign_flip_p": sign_flip_p(differences),
        "scenes_hocbf_better": int(np.sum(differences > 0)),
        "scenes_raw_better": int(np.sum(differences < 0)),
        "scenes_tied": int(np.sum(differences == 0)),
        "by_bucket": by_bucket,
    }


def arm_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    contacts = np.asarray([row["collision_count"] for row in rows], dtype=float)
    steps = np.asarray([row["steps"] for row in rows], dtype=float)
    energy = np.asarray([row["energy"] for row in rows], dtype=float)
    ordered = np.sort(contacts)[::-1]
    return {
        "anchors": len(rows),
        "goal_rate": float(np.mean([row["goal_reached"] for row in rows])),
        "safe_goal_rate": float(np.mean([row["safe_goal"] for row in rows])),
        "timeout_rate": float(np.mean([not row["goal_reached"] for row in rows])),
        "contact_task_rate": float(np.mean(contacts > 0)),
        "contacts": q(contacts),
        "contact_step_rate": float(contacts.sum() / steps.sum()),
        "top_5_contact_share": float(ordered[:5].sum() / ordered.sum()) if ordered.sum() else 0.0,
        "steps": q(steps),
        "energy": q(energy),
        "hocbf_step_intervention_rate": float(
            np.sum([row["hocbf_step_interventions"] for row in rows]) / steps.sum()
        ),
        "contact_timeouts": sum(not row["goal_reached"] and row["collision_count"] > 0 for row in rows),
        "clean_timeouts": sum(not row["goal_reached"] and row["collision_count"] == 0 for row in rows),
    }


def paired_binary_table(raw: list[dict[str, Any]], hocbf: list[dict[str, Any]], field: str) -> dict[str, int]:
    left = np.asarray([row[field] for row in raw], dtype=bool)
    right = np.asarray([row[field] for row in hocbf], dtype=bool)
    return {
        "both": int(np.sum(left & right)),
        "raw_only": int(np.sum(left & ~right)),
        "hocbf_only": int(np.sum(~left & right)),
        "neither": int(np.sum(~left & ~right)),
    }


def plot_outcomes(output: Path, raw: list[dict[str, Any]], hocbf: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True)
    for axis, field, title in zip(
        axes, ("goal_reached", "safe_goal"), ("Charger arrival", "Zero-contact charger arrival"), strict=True
    ):
        for offset, (name, rows) in zip((-0.08, 0.08), (("raw", raw), ("hocbf", hocbf)), strict=True):
            rates = []
            for bucket in BUCKETS:
                selected = [row for row in rows if row["distance_bucket"] == bucket]
                rates.append(np.mean([row[field] for row in selected]))
            axis.plot(
                np.arange(5) + offset,
                rates,
                marker="o" if name == "raw" else "s",
                linestyle="-" if name == "raw" else "--",
                color=COLORS[name],
                linewidth=1.7,
                markersize=4.5,
                label="SAC raw" if name == "raw" else "SAC + HOCBF",
            )
        axis.set_title(title)
        axis.set_xticks(np.arange(5), ["0.1–0.5", "0.5–1.5", "1.5–2.5", "2.5–4", ">4"], rotation=25)
        axis.set_xlabel("Source task distance stratum (km)")
        axis.set_ylim(0.0, 1.04)
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Anchor fraction")
    axes[1].legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(output / "figures" / "figure-01-interface-outcomes.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_energy(output: Path, raw: list[dict[str, Any]], hocbf: list[dict[str, Any]]) -> None:
    raw_energy = np.asarray([row["energy"] for row in raw])
    hocbf_energy = np.asarray([row["energy"] for row in hocbf])
    categories = np.asarray(
        [
            "both safe"
            if a["safe_goal"] and b["safe_goal"]
            else "HOCBF rescue"
            if not a["safe_goal"] and b["safe_goal"]
            else "HOCBF loss"
            if a["safe_goal"] and not b["safe_goal"]
            else "neither safe"
            for a, b in zip(raw, hocbf, strict=True)
        ]
    )
    palette = {
        "both safe": "#999999",
        "HOCBF rescue": "#009E73",
        "HOCBF loss": "#D55E00",
        "neither safe": "#CC79A7",
    }
    fig, axis = plt.subplots(figsize=(3.6, 3.2))
    limit = max(raw_energy.max(), hocbf_energy.max())
    axis.plot([0, limit], [0, limit], color="black", linestyle=":", linewidth=1)
    for category in palette:
        selected = categories == category
        axis.scatter(raw_energy[selected], hocbf_energy[selected], s=14, alpha=0.65,
                     color=palette[category], label=f"{category} (n={selected.sum()})")
    axis.set_xlabel("Raw SAC energy to terminal")
    axis.set_ylabel("SAC + HOCBF energy to terminal")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(output / "figures" / "figure-02-paired-energy-shift.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_tail(output: Path, raw: list[dict[str, Any]], hocbf: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    for name, rows in (("raw", raw), ("hocbf", hocbf)):
        values = np.sort(np.asarray([row["collision_count"] for row in rows], dtype=float))
        axes[0].step(np.log10(1.0 + values), np.arange(1, len(values) + 1) / len(values), where="post",
                     color=COLORS[name], linewidth=1.7, label="SAC raw" if name == "raw" else "SAC + HOCBF")
    axes[0].set_xlabel(r"$\log_{10}(1 + \mathrm{unified\ contacts})$")
    axes[0].set_ylabel("Empirical CDF")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)
    failures = [row for row in hocbf if not row["goal_reached"]]
    contact = np.asarray([row["collision_count"] > 0 for row in failures])
    axes[1].scatter(
        [row["hocbf_step_interventions"] / row["steps"] for row in failures],
        [row["remaining_distance"] for row in failures],
        c=np.where(contact, "#D55E00", "#0072B2"),
        marker="o",
        s=24,
    )
    axes[1].set_xlabel("HOCBF intervention fraction")
    axes[1].set_ylabel("Timeout remaining distance (m)")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "figures" / "figure-03-contact-timeout-tail.pdf", bbox_inches="tight")
    plt.close(fig)


def analyse(root: Path, output: Path) -> dict[str, Any]:
    (output / "figures").mkdir(parents=True, exist_ok=True)
    raw, hocbf = load_pair(root)
    result: dict[str, Any] = {
        "unit": "150 paired source scenes; 310 anchors clustered within scenes",
        "raw": arm_summary(raw),
        "hocbf": arm_summary(hocbf),
        "anchor_contingency": {
            "goal_reached": paired_binary_table(raw, hocbf, "goal_reached"),
            "safe_goal": paired_binary_table(raw, hocbf, "safe_goal"),
        },
        "scene_paired": {
            "goal_rate_difference_pp": paired_summary(
                raw, hocbf, lambda row: float(row["goal_reached"]), scale=100.0
            ),
            "safe_goal_rate_difference_pp": paired_summary(
                raw, hocbf, lambda row: float(row["safe_goal"]), scale=100.0
            ),
            "energy_difference": paired_summary(raw, hocbf, lambda row: float(row["energy"])),
            "contact_count_difference": paired_summary(
                raw, hocbf, lambda row: float(row["collision_count"])
            ),
            "steps_difference": paired_summary(raw, hocbf, lambda row: float(row["steps"])),
        },
    }
    common = [i for i, (a, b) in enumerate(zip(raw, hocbf, strict=True)) if a["goal_reached"] and b["goal_reached"]]
    ratio_difference = np.asarray(
        [hocbf[i]["success_path_ratio"] - raw[i]["success_path_ratio"] for i in common]
    )
    result["common_goal_path_ratio"] = {
        "anchors": len(common),
        "raw": q(np.asarray([raw[i]["success_path_ratio"] for i in common])),
        "hocbf": q(np.asarray([hocbf[i]["success_path_ratio"] for i in common])),
        "difference_hocbf_minus_raw": q(ratio_difference),
        "wilcoxon_p_descriptive": float(stats.wilcoxon(ratio_difference, zero_method="pratt").pvalue),
    }
    failures = [row for row in hocbf if not row["goal_reached"]]
    result["hocbf_failures"] = {
        "anchors": len(failures),
        "contact_timeouts": sum(row["collision_count"] > 0 for row in failures),
        "clean_timeouts": sum(row["collision_count"] == 0 for row in failures),
        "remaining_distance": q(np.asarray([row["remaining_distance"] for row in failures])),
        "intervention_fraction": q(
            np.asarray([row["hocbf_step_interventions"] / row["steps"] for row in failures])
        ),
    }
    (output / "statistics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    plot_outcomes(output, raw, hocbf)
    plot_energy(output, raw, hocbf)
    plot_tail(output, raw, hocbf)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("artifacts/sac_anchor_return_interface_310_20260906_v1"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/sac_anchor_return_interface_analysis_20260906_v1/analysis-output"),
    )
    args = parser.parse_args()
    print(json.dumps(analyse(args.input, args.output), indent=2))


if __name__ == "__main__":
    main()
