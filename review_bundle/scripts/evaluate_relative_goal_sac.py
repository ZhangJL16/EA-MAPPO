from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import sys

from stable_baselines3 import SAC

from experiments.navigation_scale import evaluate_relative_goal_policy
from experiments.new_route.provenance import sha256_file, utc_now, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen 7D relative-goal SAC zero-shot evaluation")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--world-sizes", type=float, nargs="+", default=[4.0, 8.0, 16.0])
    parser.add_argument("--sorties-per-bin", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=800)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite zero-shot evaluation: {output}")
    output.mkdir(parents=True)
    checkpoint = Path(args.checkpoint).resolve()
    config = vars(args) | {
        "status": "RUNNING",
        "exact_command": shlex.join([sys.executable, *sys.argv]),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "navigation_policy_frozen": True,
        "optimizer_updates": 0,
        "replay_buffer_updates": 0,
        "started_at": utc_now(),
    }
    write_json(output / "config.json", config)
    write_json(output / "RUNNING.json", {"status": "RUNNING", "started_at": config["started_at"]})
    model = SAC.load(checkpoint, device=args.device)
    results = evaluate_relative_goal_policy(
        model,
        world_sizes=args.world_sizes,
        sorties_per_bin=args.sorties_per_bin,
        max_steps=args.max_steps,
        seed=args.seed + 70_000_000,
    ) | {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": config["checkpoint_sha256"],
        "completed_at": utc_now(),
    }
    write_json(output / "results.json", results)
    write_json(output / "COMPLETED.json", {"status": "COMPLETED", "completed_at": results["completed_at"]})


if __name__ == "__main__":
    main()
