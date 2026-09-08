"""Frozen training-batch replay: critic/advantage and policy-update diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from functools import partial
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "1"

import numpy as np
import torch
from stable_baselines3.common.vec_env import SubprocVecEnv

from experiments.directional_navigation.features import DirectionalLidarExtractor
from experiments.directional_navigation.lagrangian import TwoValuePolicy, collect_cohort
from experiments.directional_navigation.recovery import RecoveryCohort
from scripts.run_directional_cost_trace_pair import restore
from scripts.run_directional_lagrangian_pair import state_hash
from scripts.train_directional_ppo import atomic_json


class RewardTrace(RecoveryCohort):
    def reset(self, **kwargs):
        result = super().reset(**kwargs)
        if not self.parked:
            self.audit_rewards = []
        return result

    def step(self, action):
        parked = self.parked
        result = super().step(action)
        if not parked:
            self.audit_rewards.append(float(result[1]))
            if result[2]:
                episode = result[4]["navigation_episode"]
                if not isinstance(episode, dict):
                    raise TypeError("terminal navigation episode must be a dictionary")
                episode["audit_rewards"] = self.audit_rewards.copy()
        return result


def advantage_diagnostic(policy, updated, data, rows, seeds):
    by_seed = {r["seed"]: r for r in rows}
    mc, labels, immediate = [], [], []
    for seed in seeds:
        row = by_seed[seed]
        rewards = np.array(row["audit_rewards"])
        mc.append(np.cumsum(rewards[::-1])[::-1])
        immediate.extend(rewards)
        labels.extend(["goal" if row["goal_reached"] else "timeout"] * len(rewards))
    returns = np.concatenate(mc)
    value = (data["rr"] - data["ar"]).astype(float)
    gae, mc_adv = data["ar"].astype(float), returns - value
    labels = np.array(labels)

    def stats(mask):
        a, b, v, r = gae[mask], mc_adv[mask], value[mask], returns[mask]
        return {
            "steps": len(a),
            "value_mean": float(v.mean()),
            "mc_return_mean": float(r.mean()),
            "mc_value_mse": float(np.mean((v - r) ** 2)),
            "mc_value_r2": float(1 - np.mean((v - r) ** 2) / max(np.var(r), 1e-12)),
            "gae_mean": float(a.mean()),
            "mc_adv_mean": float(b.mean()),
            "gae_std": float(a.std()),
            "mc_adv_std": float(b.std()),
            "gae_mc_sign_disagreement": float(np.mean((a > 0) != (b > 0))),
            "gae_mc_correlation": float(np.corrcoef(a, b)[0, 1])
            if min(a.std(), b.std()) > 0
            else None,
        }

    result: dict[str, Any] = {"all": stats(np.ones(len(gae), bool))}
    for label in ("goal", "timeout"):
        if np.any(labels == label):
            result[label] = stats(labels == label)
    # Same registered sample of at most2048states, fixed; no model update.
    idx = np.linspace(0, len(gae) - 1, min(2048, len(gae)), dtype=int)
    obs = torch.as_tensor(data["obs"][idx], device=policy.device)
    actions = torch.as_tensor(data["actions"][idx], device=policy.device)
    params = (
        list(policy.pi_features_extractor.parameters())
        + list(policy.mlp_extractor.policy_net.parameters())
        + list(policy.action_net.parameters())
        + [policy.log_std]
    )
    vectors = []
    for signal in (gae, mc_adv):
        adv = torch.as_tensor(signal[idx], device=policy.device, dtype=torch.float32)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        _, _, _, lp, _ = policy.evaluate_both(obs, actions)
        grad = torch.autograd.grad(-(lp * adv).mean(), params, allow_unused=True)
        vectors.append(torch.cat([g.flatten() for g in grad if g is not None]))
    result["actor_gae_mc_gradient_cosine"] = float(
        torch.nn.functional.cosine_similarity(vectors[0], vectors[1], dim=0)
    )
    result["actor_gradient_samples"] = len(idx)
    # Analytic Gaussian KL on the ENTIRE original rollout, before vs after update.
    kl, lp_delta, displacement = [], [], []
    with torch.no_grad():
        for start in range(0, len(gae), 1024):
            obs = torch.as_tensor(
                data["obs"][start : start + 1024], device=policy.device
            )
            actions = torch.as_tensor(
                data["actions"][start : start + 1024], device=policy.device
            )
            old = policy.get_distribution(obs).distribution
            new = updated.get_distribution(obs).distribution
            kl.extend(
                torch.distributions.kl_divergence(old, new).sum(-1).cpu().tolist()
            )
            lp_delta.extend(
                (new.log_prob(actions) - old.log_prob(actions)).sum(-1).cpu().tolist()
            )
            displacement.extend(
                torch.linalg.vector_norm(new.mean - old.mean, dim=-1).cpu().tolist()
            )
    kl = np.array(kl)
    ratio = np.exp(np.array(lp_delta))
    result["post_update"] = {
        "full_batch_analytic_kl_mean": float(kl.mean()),
        "full_batch_analytic_kl_p95": float(np.quantile(kl, 0.95)),
        "sampled_action_clip_fraction": float(np.mean((ratio < 0.8) | (ratio > 1.2))),
        "mean_action_displacement": float(np.mean(displacement)),
        "gae_surrogate_change": float(np.mean((ratio - 1) * (gae - gae.mean()))),
        "mc_surrogate_change": float(np.mean((ratio - 1) * (mc_adv - mc_adv.mean()))),
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_dir
    root.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.source / "manifest.json").read_text())["contract"]
    for p, h in manifest["source_hashes"].items():
        if hashlib.sha256((ROOT / p).read_bytes()).hexdigest() != h:
            raise ValueError(f"source changed:{p}")
    torch.set_num_threads(1)
    vec = SubprocVecEnv(
        [partial(RewardTrace) for _ in range(8)], start_method="forkserver"
    )
    try:

        def make():
            return TwoValuePolicy(
                vec.observation_space,
                vec.action_space,
                lambda _: 3e-4,
                features_extractor_class=DirectionalLidarExtractor,
                features_extractor_kwargs={"remaining_time": True},
                share_features_extractor=False,
                net_arch={"pi": [256, 256], "vf": [256, 256]},
                log_std_init=float(np.log(0.5)),
            ).to("cuda")

        policy, updated = make(), make()
        for cohort in (65, 81, 89, 96):
            target = root / f"replay_{cohort:04d}.json"
            if target.exists():
                continue
            cp = args.source / "model" / f"checkpoint_{cohort:04d}"
            expected = json.loads((cp / "metadata.json").read_text())
            restore(updated, cp)
            restore(policy, args.source / "model" / f"checkpoint_{cohort - 1:04d}")
            before = state_hash(policy)
            seeds = list(range(483600001 + (cohort - 1) * 8, 483600001 + cohort * 8))
            data, rows, _ = collect_cohort(policy, vec, seeds)
            old = {r["seed"]: r for r in expected["last_episodes"]}
            for r in rows:
                for key in ("steps", "collision_count", "goal_reached", "safe_goal"):
                    if r[key] != old[r["seed"]][key]:
                        raise AssertionError(f"replay mismatch {cohort} {key}")
                if abs(r["raw_return"] - old[r["seed"]]["raw_return"]) > 1e-6:
                    raise AssertionError("reward mismatch")
            diagnostic = advantage_diagnostic(policy, updated, data, rows, seeds)
            if state_hash(policy) != before:
                raise AssertionError("replay changed weights")
            for r in rows:
                r.pop("audit_rewards")
                r.pop("outcome")
            result = {
                "cohort": cohort,
                "replay_matches": True,
                "weights_unchanged": True,
                "diagnostic": diagnostic,
                "training_update": expected["last_update"],
                "episodes": rows,
            }
            atomic_json(target, result)
            print(json.dumps({"cohort": cohort, "diagnostic": diagnostic}), flush=True)
        atomic_json(
            root / "REPLAY_RESULT.json",
            {
                "status": "COMPLETE",
                "cohorts": [65, 81, 89, 96],
                "matched_tasks": 32,
                "weights_unchanged": True,
            },
        )
    finally:
        vec.close()


if __name__ == "__main__":
    main()
