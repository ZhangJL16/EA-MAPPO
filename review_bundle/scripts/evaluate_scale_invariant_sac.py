from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import SAC

from experiments.navigation_scale import DEFAULT_EVALUATION_SCALES, evaluate_policy
from experiments.new_route.provenance import sha256_file, utc_now, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate scale-invariant SAC on training and held-out map sizes")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--scales", type=float, nargs="+", default=list(DEFAULT_EVALUATION_SCALES))
    parser.add_argument("--sorties-per-distance-bin", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=800)
    parser.add_argument("--seed", type=int, default=100_000)
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint).resolve()
    model = SAC.load(checkpoint, device=args.device)
    results = evaluate_policy(
        model,
        scales=args.scales,
        sorties_per_distance_bin=args.sorties_per_distance_bin,
        max_steps=args.max_steps,
        seed=args.seed,
    )
    results |= {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "completed_at": utc_now(),
    }
    write_json(args.output, results)


if __name__ == "__main__":
    main()
