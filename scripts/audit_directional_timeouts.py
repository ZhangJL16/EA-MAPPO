"""Frozen evaluation replay with full-resolution trajectory summaries, no training."""

from __future__ import annotations

import argparse
import json
import os
import sys
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
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv

from experiments.directional_navigation.cohort import CohortNavigation
from experiments.directional_navigation.features import DirectionalLidarExtractor
from experiments.directional_navigation.lagrangian import TwoValuePolicy, collect_cohort
from scripts.analyze_directional_lagrangian_failure import rows, summarize
from scripts.run_directional_lagrangian_pair import state_hash
from scripts.train_directional_ppo import atomic_json


def trajectory_metrics(trace: np.ndarray, goal_radius: float) -> dict:
    """Columns: step,x,y,z,distance,speed,raw_return,clearance,velocity_reward."""
    if trace.ndim != 2 or trace.shape[1] != 9 or len(trace) < 2:
        raise ValueError("full trajectory with nine columns required")
    tail = trace[-min(501, len(trace)) :]
    travel = np.linalg.norm(np.diff(trace[:, 1:4], axis=0), axis=1)
    tail_travel = np.linalg.norm(np.diff(tail[:, 1:4], axis=0), axis=1)
    distance = trace[:, 4]
    tail_net = float(tail[0, 4] - tail[-1, 4])
    tail_path = float(tail_travel.sum())
    metrics = {
        "min_distance": float(distance.min()),
        "first_within_60_step": next((int(t[0]) for t in trace if t[4] <= 60), None),
        "steps_within_60": int(np.sum(distance[1:] <= 60)),
        "path_length": float(travel.sum()),
        "path_over_initial_distance": float(travel.sum() / max(distance[0], 1e-8)),
        "tail_steps": len(tail) - 1,
        "tail_mean_speed": float(tail[1:, 5].mean()),
        "tail_low_speed_fraction_lt1": float(np.mean(tail[1:, 5] < 1.0)),
        "tail_net_progress": tail_net,
        "tail_path": tail_path,
        "tail_distance_span": float(np.ptp(tail[:, 4])),
        "tail_position_span": np.ptp(tail[:, 1:4], axis=0).tolist(),
        "tail_raw_reward": float(tail[-1, 6] - tail[0, 6]),
        "tail_min_clearance": float(tail[:, 7].min()),
        "tail_mean_clearance": float(tail[:, 7].mean()),
        "min_physical_clearance": float(trace[:, 7].min()),
        "entered_goal_ball_at_policy_sample": bool(np.any(distance <= goal_radius)),
        "tail_positive_radial_variation": float(
            np.maximum(0, -np.diff(tail[:, 4])).sum()
        ),
        "tail_mean_velocity_reward": float(tail[1:, 8].mean()),
    }
    # Descriptive flags, not changed task success/physics thresholds.
    metrics["tail_stall_flag"] = metrics["tail_low_speed_fraction_lt1"] >= 0.8
    metrics["tail_inefficient_motion_flag"] = (
        tail_path > 100 and tail_net / tail_path < 0.1
    )
    return metrics


class TracedCohort(CohortNavigation):
    def reset(self, *, seed=None, options=None):
        result = super().reset(seed=seed, options=options)
        if not self.parked:
            self.trace = [self.snapshot()]
        return result

    def snapshot(self) -> list[float]:
        pos = np.asarray(self.base.agent.pos, dtype=float)
        vel = np.asarray(self.base.agent.vel, dtype=float)
        delta = self.base.active_goal - pos
        distance = float(np.linalg.norm(delta))
        bounds = np.array([self.base.length, self.base.width, self.base.height])
        clearance = (
            min(float(np.min(pos)), float(np.min(bounds - pos))) - self.base.safe_radius
        )
        for obstacle in self.base.obstacles:
            clearance = min(
                clearance,
                float(np.linalg.norm(pos[:2] - obstacle.pos))
                - obstacle.radius
                - self.base.safe_radius,
            )
        velocity_reward = self.base.velocity_reward_weight * max(
            0.0,
            float(np.dot(vel, delta / max(distance, 1e-8)))
            / self.base.horizontal_v_max,
        )
        return [
            float(self.steps),
            *pos.tolist(),
            distance,
            float(np.linalg.norm(vel)),
            self.raw_return,
            clearance,
            velocity_reward,
        ]

    def step(self, action):
        parked = self.parked
        result = super().step(action)
        if not parked:
            self.trace.append(self.snapshot())
        if result[2]:
            trace = np.asarray(self.trace)
            row = result[4]["navigation_episode"]
            if not isinstance(row, dict):
                raise TypeError("terminal navigation episode must be a dictionary")
            row["trajectory_audit"] = trajectory_metrics(
                trace, self.base.goal_tolerance
            )
            # Full-resolution metrics above; plots retain one sample per20steps plus endpoint.
            row["trajectory_samples"] = trace[
                np.unique(np.r_[np.arange(0, len(trace), 20), len(trace) - 1])
            ].tolist()
            row["goal"] = self.base.active_goal.tolist()
            row["obstacles"] = self.base.static_obstacle_layout()
            if row["outcome"] == "timeout":
                predicted = (
                    trace[0, 4]
                    - trace[-1, 4]
                    + trace[1:, 8].sum()
                    - self.base.time_penalty * self.steps
                )
                row["timeout_reward_identity_error"] = float(
                    self.raw_return - predicted
                )
        return result


def plot_timeouts(output: Path) -> None:
    for mode in ("deterministic", "stochastic"):
        rr = [
            r
            for r in rows(output / f"trace_lagrangian_{mode}.jsonl")
            if r["outcome"] == "timeout"
        ]
        fig, axes = plt.subplots(3, 4, figsize=(12, 8), squeeze=False)
        for ax, row in zip(axes.flat, rr):
            trace = np.asarray(row["trajectory_samples"])
            ax.plot(trace[:, 0] * 0.2, trace[:, 4], color="#0072B2", lw=1.4)
            ax.axhline(5, color="#D55E00", ls="--", lw=1)
            ax.set_title(str(row["seed"]), fontsize=9)
            ax.set_xlabel("Time (s)", fontsize=8)
            ax.set_ylabel("Goal distance (m)", fontsize=8)
            ax.tick_params(labelsize=8)
            ax.set_ylim(bottom=0)
        for ax in list(axes.flat)[len(rr) :]:
            ax.set_visible(False)
        fig.tight_layout()
        fig.savefig(output / "figures" / f"03-timeouts-{mode}.pdf")
        fig.savefig(output / "figures" / f"03-timeouts-{mode}.png", dpi=150)
        plt.close(fig)
    rr = [
        r
        for r in rows(output / "trace_lagrangian_deterministic.jsonl")
        if r["outcome"] == "timeout"
    ]
    fig, axes = plt.subplots(3, 4, figsize=(12, 8))
    for ax, row in zip(axes.flat, rr):
        trace = np.asarray(row["trajectory_samples"])
        goal = np.asarray(row["goal"])
        ax.plot(trace[:, 1], trace[:, 2], color="#0072B2")
        ax.scatter(*goal[:2], marker="*", color="#D55E00", s=55)
        ax.scatter(*trace[-1, 1:3], marker="x", color="black", s=30)
        for obstacle in row["obstacles"]:
            ax.add_patch(
                plt.Circle(
                    obstacle["position"], obstacle["radius"], color="gray", alpha=0.25
                )
            )
        ax.set_xlim(
            min(trace[:, 1].min(), goal[0]) - 100, max(trace[:, 1].max(), goal[0]) + 100
        )
        ax.set_ylim(
            min(trace[:, 2].min(), goal[1]) - 100, max(trace[:, 2].max(), goal[1]) + 100
        )
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(str(row["seed"]), fontsize=9)
        ax.tick_params(labelsize=7)
    for ax in list(axes.flat)[len(rr) :]:
        ax.set_visible(False)
    fig.tight_layout()
    fig.savefig(output / "figures" / "04-timeout-paths.pdf")
    fig.savefig(output / "figures" / "04-timeout-paths.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summarize(args.source, args.output)
    torch.set_num_threads(1)
    vec = SubprocVecEnv(
        [partial(TracedCohort) for _ in range(8)], start_method="forkserver"
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
        for arm in ("lagrangian", "control"):
            payload = torch.load(
                args.source / arm / "checkpoint_0064/state.pt",
                map_location="cuda",
                weights_only=False,
            )
            policy.load_state_dict(payload["model"])
            original_hash = state_hash(policy)
            for mode in ("deterministic", "stochastic"):
                target = args.output / f"trace_{arm}_{mode}.jsonl"
                recorded = rows(target) if target.exists() else []
                seen = {r["seed"] for r in recorded}
                original = {
                    r["seed"]: r
                    for r in rows(args.source / arm / f"evaluation_0064_{mode}.jsonl")
                }
                for start in range(0, 50, 8):
                    seeds = list(
                        range(593800001 + start, 593800001 + min(50, start + 8))
                    )
                    if all(s in seen for s in seeds):
                        continue
                    if any(s in seen for s in seeds):
                        raise ValueError("partial saved cohort: use a fresh output")
                    set_random_seed(693800001 + seeds[0], using_cuda=True)
                    _, replay, _ = collect_cohort(
                        policy,
                        vec,
                        seeds,
                        deterministic=mode == "deterministic",
                        training=False,
                    )
                    for row in replay:
                        expected = original[row["seed"]]
                        assert (row["outcome"], row["steps"]) == (
                            expected["outcome"],
                            expected["steps"],
                        )
                        assert abs(row["raw_return"] - expected["raw_return"]) < 1e-6
                    recorded.extend(replay)
                    # Atomic whole-cohort writes allow safe replay resume.
                    temporary = target.with_suffix(".tmp")
                    temporary.write_text(
                        "".join(json.dumps(r) + "\n" for r in recorded)
                    )
                    temporary.replace(target)
                    print(
                        json.dumps(
                            {
                                "arm": arm,
                                "mode": mode,
                                "completed": len(recorded),
                                "matched": True,
                            }
                        ),
                        flush=True,
                    )
                assert state_hash(policy) == original_hash
    finally:
        vec.close()
    plot_timeouts(args.output)
    records = {
        f"{arm}_{mode}": [
            {k: v for k, v in r.items() if k not in {"trajectory_samples", "obstacles"}}
            for r in rows(args.output / f"trace_{arm}_{mode}.jsonl")
            if r["outcome"] == "timeout"
        ]
        for arm in ("lagrangian", "control")
        for mode in ("deterministic", "stochastic")
    }
    atomic_json(args.output / "TIMEOUTS.json", records)
    atomic_json(
        args.output / "REPLAY_RESULT.json",
        {
            "status": "COMPLETE",
            "matched_tasks": 200,
            "weights_unchanged": True,
            "independent_new_evaluation": False,
        },
    )


if __name__ == "__main__":
    main()
