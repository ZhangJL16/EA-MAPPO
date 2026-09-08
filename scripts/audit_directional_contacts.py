"""Replay frozen development tasks, add event details, never refit a policy."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

from experiments.directional_navigation.cohort import CohortNavigation
from scripts.train_directional_ppo import atomic_json, evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    model = PPO.load(args.source / "checkpoint_000262144/model.zip", device="cuda")
    vec = SubprocVecEnv(
        [partial(CohortNavigation) for _ in range(8)], start_method="forkserver"
    )
    started = time.perf_counter()
    try:
        summary = evaluate(model, vec, args.output_dir, 50)
    finally:
        vec.close()
    original = {
        r["seed"]: r
        for r in map(
            json.loads,
            (args.source / "evaluation_000262144.jsonl").read_text().splitlines(),
        )
    }
    rows = list(
        map(
            json.loads,
            (args.output_dir / "evaluation_000262144.jsonl").read_text().splitlines(),
        )
    )
    checks = [
        r["outcome"] == original[r["seed"]]["outcome"]
        and r["steps"] == original[r["seed"]]["steps"]
        for r in rows
    ]
    terminal_types = Counter(
        "both"
        if r["obstacle_collision"] and r["boundary_contact"]
        else "obstacle"
        if r["obstacle_collision"]
        else "boundary"
        for r in rows
        if r["outcome"] == "contact"
    )
    atomic_json(
        args.output_dir / "RESULT.json",
        {
            "evaluation": summary,
            "outcome_and_step_match": all(checks),
            "mismatch_count": len(checks) - sum(checks),
            "contact_types": dict(terminal_types),
            "wall_seconds": time.perf_counter() - started,
            "note": "Replay event characterization; not independent new performance evidence or causal attribution.",
        },
    )
    print(json.dumps({"matched": all(checks), "contacts": dict(terminal_types)}))


if __name__ == "__main__":
    main()
