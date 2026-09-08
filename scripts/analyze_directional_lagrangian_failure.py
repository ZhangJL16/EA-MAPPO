"""Read-only checkpoint replay and descriptive failure analysis; never train."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from collections import Counter
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "1"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import binomtest
from stable_baselines3.common.vec_env import SubprocVecEnv

from experiments.directional_navigation.cohort import CohortNavigation, terminal_gae
from experiments.directional_navigation.features import DirectionalLidarExtractor
from experiments.directional_navigation.lagrangian import TwoValuePolicy, collect_cohort
from scripts.run_directional_lagrangian_pair import state_hash
from scripts.train_directional_ppo import atomic_json

ARMS = ("lagrangian", "control")
OUTCOMES = ("safe_goal", "contact", "timeout")


def rows(path: Path) -> list[dict]:
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def wilson(k: int, n: int) -> list[float]:
    z = 1.959963984540054
    p, d = k / n, 1 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, float(center - half)), min(1.0, float(center + half))]


def summarize(source: Path, output: Path) -> dict:
    manifest = json.loads((source / "manifest.json").read_text())
    for name, digest in manifest["contract"]["source_hashes"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"original source changed: {name}")
    assert json.loads((source / "RESULT.json").read_text())["status"] == "COMPLETE"
    result: dict = {"training": {}, "evaluation": {}, "contrasts": []}
    updates = {}
    for arm in ARMS:
        episodes = rows(source / arm / "training_episodes.jsonl")
        assert len(episodes) == 512 and len({r["seed"] for r in episodes}) == 512
        assert sorted(r["seed"] for r in episodes) == list(range(483600001, 483600513))
        updates[arm] = rows(source / arm / "updates.jsonl")
        assert [u["cohort"] for u in updates[arm]] == list(range(1, 65))
        blocks = []
        for first in range(1, 65, 8):
            block = [r for r in episodes if first <= r["cohort"] < first + 8]
            blocks.append(
                {
                    "first_cohort": first,
                    "counts": dict(Counter(r["outcome"] for r in block)),
                    "mean_steps": float(np.mean([r["steps"] for r in block])),
                }
            )
        result["training"][arm] = {
            "counts": dict(Counter(r["outcome"] for r in episodes)),
            "blocks": blocks,
            "all_contact_cohorts": sum(
                u["task_contact_rate"] == 1 for u in updates[arm]
            ),
            "physical_steps": sum(r["steps"] for r in episodes),
            "gradient_steps": sum(u["gradient_steps"] for u in updates[arm]),
            "contact_types": dict(
                Counter(
                    "boundary" if r["boundary_contact"] else "obstacle"
                    for r in episodes
                    if r["outcome"] == "contact"
                )
            ),
        }
        result["evaluation"][arm] = {}
        for mode in ("deterministic", "stochastic"):
            rr = rows(source / arm / f"evaluation_0064_{mode}.jsonl")
            assert sorted(r["seed"] for r in rr) == list(range(593800001, 593800051))
            result["evaluation"][arm][mode] = {
                k: {
                    "count": sum(r["outcome"] == k for r in rr),
                    "wilson95": wilson(sum(r["outcome"] == k for r in rr), len(rr)),
                }
                for k in OUTCOMES
            }
    # One declared family: two outcomes x two execution modes, paired by scene.
    # These conditional developer-scene tests do not estimate training-seed variability.
    rng = np.random.default_rng(983500001)
    for mode in ("deterministic", "stochastic"):
        pair = [
            [
                r
                for r in sorted(
                    rows(source / arm / f"evaluation_0064_{mode}.jsonl"),
                    key=lambda r: r["seed"],
                )
            ]
            for arm in ARMS
        ]
        for outcome in ("safe_goal", "contact"):
            a, b = [
                np.array([r["outcome"] == outcome for r in rr], dtype=int)
                for rr in pair
            ]
            difference = a - b
            plus, minus = int(np.sum(difference == 1)), int(np.sum(difference == -1))
            p = binomtest(plus, plus + minus).pvalue if plus + minus else 1.0
            boot = difference[rng.integers(0, 50, size=(20000, 50))].mean(axis=1)
            result["contrasts"].append(
                {
                    "mode": mode,
                    "outcome": outcome,
                    "difference_lag_minus_control": float(difference.mean()),
                    "paired_bootstrap95": np.quantile(boot, [0.025, 0.975]).tolist(),
                    "discordant_lag_only": plus,
                    "discordant_control_only": minus,
                    "exact_mcnemar_p": float(p),
                }
            )
    ordered = sorted(result["contrasts"], key=lambda r: r["exact_mcnemar_p"])
    previous = 0.0
    for i, record in enumerate(ordered):
        previous = max(
            previous, min(1.0, (len(ordered) - i) * record["exact_mcnemar_p"])
        )
        record["holm_p"] = previous
    atomic_json(output / "summary.json", result)
    figure_dir = output / "figures"
    figure_dir.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), sharey=True)
    for ax, mode in zip(axes, ("deterministic", "stochastic")):
        for j, (arm, color) in enumerate(zip(ARMS, ("#D55E00", "#0072B2"))):
            values = result["evaluation"][arm][mode]
            mean = np.array([values[k]["count"] / 50 for k in OUTCOMES])
            ci = np.array([values[k]["wilson95"] for k in OUTCOMES]).T
            ax.bar(
                np.arange(3) + (j - 0.5) * 0.36,
                mean,
                0.36,
                label=arm,
                color=color,
                hatch="//" if j == 0 else None,
            )
            ax.errorbar(
                np.arange(3) + (j - 0.5) * 0.36,
                mean,
                yerr=np.maximum(0, np.array([mean - ci[0], ci[1] - mean])),
                fmt="none",
                color="black",
                capsize=3,
            )
        ax.set_xticks(range(3), ["Safe goal", "Contact", "Timeout"])
        ax.set_title(mode)
        ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("Task proportion (Wilson 95% CI)")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_dir / "01-outcomes.pdf")
    fig.savefig(figure_dir / "01-outcomes.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(9, 6))
    for arm, color, style in zip(ARMS, ("#D55E00", "#0072B2"), ("-", "--")):
        blocks = result["training"][arm]["blocks"]
        for outcome, ax in zip(("safe_goal", "contact"), axes[0]):
            ax.plot(
                [b["first_cohort"] + 7 for b in blocks],
                [b["counts"].get(outcome, 0) / 64 for b in blocks],
                style,
                color=color,
                marker="o",
                label=arm,
            )
            ax.set_title(outcome + " (64-task blocks)")
            ax.set_ylim(0, 1.02)
        for key, ax in zip(
            ("multiplier", "cost_value_outside_unit_interval_before"), axes[1]
        ):
            ax.plot(
                [u["cohort"] for u in updates[arm]],
                [u[key] for u in updates[arm]],
                style,
                color=color,
                label=arm,
            )
            ax.set_title(
                key.replace(
                    "cost_value_outside_unit_interval_before", "Cost V outside [0,1]"
                )
            )
        for ax in axes.flat:
            ax.set_xlabel("Completed policy updates")
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "02-training.pdf")
    fig.savefig(figure_dir / "02-training.png", dpi=180)
    plt.close(fig)
    return result


def gradient_diagnostic(
    policy: TwoValuePolicy, data: dict, multiplier: float, episodes: list[dict]
) -> dict:
    vc = data["rc"] - data["ac"]
    alternative = []
    offset = 0
    for row in sorted(episodes, key=lambda r: r["seed"]):
        length = row["steps"]
        cost = np.zeros(length, dtype=np.float32)
        cost[-1] = float(row["outcome"] == "contact")
        alternative.append(terminal_gae(cost, vc[offset : offset + length], 0.95)[0])
        offset += length
    assert offset == len(vc)
    gae95 = np.concatenate(alternative)
    indices = np.random.default_rng(763500001).choice(
        len(vc), min(len(vc), 4096), replace=False
    )
    obs = torch.as_tensor(data["obs"][indices], device=policy.device)
    actions = torch.as_tensor(data["actions"][indices], device=policy.device)
    policy.set_training_mode(False)
    _, _, _, logp, _ = policy.evaluate_both(obs, actions)
    parameters = list(policy.parameters())
    grads = []
    for values in (data["ar"], data["ac"], gae95):
        adv = torch.as_tensor(values[indices], device=policy.device)
        loss = -(logp * (adv - adv.mean())).mean()
        gradient = torch.autograd.grad(
            loss, parameters, retain_graph=True, allow_unused=True
        )
        grads.append(
            torch.cat([g.flatten() for g in gradient if g is not None]).detach()
        )
    gr, gc, g95 = grads
    norm_r, norm_c = gr.norm().item(), gc.norm().item()
    std_r, std_c = float(np.std(data["ar"])), float(np.std(data["ac"]))
    variance = float(np.var(data["rc"]))
    return {
        "reward_adv_std": std_r,
        "cost_mc_adv_std": std_c,
        "cost_gae95_adv_std": float(np.std(gae95)),
        "lambda_cost_to_reward_std_ratio": multiplier * std_c / max(std_r, 1e-12),
        "actor_gradient_samples": len(indices),
        "reward_gradient_norm": norm_r,
        "cost_gradient_norm": norm_c,
        "lambda_cost_to_reward_gradient_ratio": multiplier
        * norm_c
        / max(norm_r, 1e-12),
        "reward_cost_gradient_cosine": float(
            torch.nn.functional.cosine_similarity(gr, gc, dim=0)
        ),
        "cost_mc_gae95_gradient_cosine": float(
            torch.nn.functional.cosine_similarity(gc, g95, dim=0)
        ),
        "cost_mc_target_variance": variance,
        "cost_mc_mse": float(np.mean((vc - data["rc"]) ** 2)),
        "cost_mc_r2": float(1 - np.mean((vc - data["rc"]) ** 2) / variance)
        if variance > 1e-9
        else None,
        "cost_value_range": [float(vc.min()), float(vc.max())],
        "outside_unit_interval": float(np.mean((vc < 0) | (vc > 1))),
    }


def replay(source: Path, output: Path, cohorts: list[int]) -> None:
    vec = SubprocVecEnv(
        [partial(CohortNavigation, horizon=4000, obstacles=24) for _ in range(8)],
        start_method="forkserver",
    )
    try:
        policy = TwoValuePolicy(
            vec.observation_space,
            vec.action_space,
            lambda _: 3e-4,
            features_extractor_class=DirectionalLidarExtractor,
            features_extractor_kwargs={"remaining_time": True},
            share_features_extractor=False,
            net_arch={"pi": [256, 256], "vf": [256, 256]},
            log_std_init=float(np.log(0.5)),
        ).to("cuda")
        for arm in ARMS:
            original_episodes = rows(source / arm / "training_episodes.jsonl")
            updates = rows(source / arm / "updates.jsonl")
            for cohort in cohorts:
                target = output / f"replay_{arm}_{cohort:04d}.json"
                if target.exists():
                    continue
                checkpoint = source / arm / f"checkpoint_{cohort - 1:04d}"
                payload = torch.load(
                    checkpoint / "state.pt", map_location="cuda", weights_only=False
                )
                policy.load_state_dict(payload["model"])
                before_hash = state_hash(policy)
                random.setstate(payload["python_rng"])
                np.random.set_state(payload["numpy_rng"])
                torch.set_rng_state(payload["torch_rng"].cpu())
                torch.cuda.set_rng_state_all([s.cpu() for s in payload["cuda_rng"]])
                seeds = list(
                    range(483600001 + (cohort - 1) * 8, 483600001 + cohort * 8)
                )
                data, episodes, audit = collect_cohort(policy, vec, seeds)
                expected = sorted(
                    [r for r in original_episodes if r["cohort"] == cohort],
                    key=lambda r: r["seed"],
                )
                actual = sorted(episodes, key=lambda r: r["seed"])
                matches = all(
                    (a["seed"], a["outcome"], a["steps"])
                    == (b["seed"], b["outcome"], b["steps"])
                    for a, b in zip(actual, expected)
                )
                if not matches:
                    raise AssertionError(
                        f"replay diverged from original training {arm} {cohort}"
                    )
                diagnostic = gradient_diagnostic(
                    policy, data, updates[cohort - 1]["multiplier"], episodes
                )
                assert before_hash == state_hash(policy), (
                    "diagnostics changed frozen weights"
                )
                result = {
                    "arm": arm,
                    "cohort": cohort,
                    "replay_matches": matches,
                    "multiplier": updates[cohort - 1]["multiplier"],
                    "episodes": episodes,
                    "collection": audit,
                    **diagnostic,
                }
                atomic_json(target, result)
                print(
                    json.dumps(
                        {
                            k: v
                            for k, v in result.items()
                            if k not in ("episodes", "collection")
                        }
                    ),
                    flush=True,
                )
    finally:
        vec.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replay-cohorts", type=int, nargs="*", default=[])
    args = parser.parse_args()
    if args.source.resolve() == args.output.resolve():
        raise ValueError("analysis must not overwrite source")
    if any(not 1 <= c <= 64 for c in args.replay_cohorts):
        raise ValueError("cohorts must be 1..64")
    args.output.mkdir(exist_ok=True, parents=True)
    torch.set_num_threads(1)
    summarize(args.source, args.output)
    if args.replay_cohorts:
        replay(args.source, args.output, args.replay_cohorts)


if __name__ == "__main__":
    main()
