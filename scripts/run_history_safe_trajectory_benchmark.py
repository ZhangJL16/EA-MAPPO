from __future__ import annotations

import argparse
import json

from experiments.history_safe_trajectory import BenchmarkConfig, run_benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--initial-states", type=int, default=100_000)
    parser.add_argument("--proposals-per-state", type=int, default=1_000)
    parser.add_argument("--exact-reference-states", type=int, default=50)
    parser.add_argument("--hard-states", type=int, default=5)
    parser.add_argument("--hard-proposals-per-state", type=int, default=10_000)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--coarse-keep", type=int, default=20)
    parser.add_argument("--chunk-states", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_benchmark(
        BenchmarkConfig(
            seed=args.seed,
            initial_states=args.initial_states,
            proposals_per_state=args.proposals_per_state,
            exact_reference_states=args.exact_reference_states,
            hard_states=args.hard_states,
            hard_proposals_per_state=args.hard_proposals_per_state,
            horizon=args.horizon,
            coarse_keep=args.coarse_keep,
            chunk_states=args.chunk_states,
            output_dir=args.output_dir,
        )
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
