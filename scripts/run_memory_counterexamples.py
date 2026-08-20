from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.memory_safety.counterexamples import CounterexampleConfig, run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--random-states", type=int, default=100_000)
    parser.add_argument("--abrupt-states", type=int, default=100_000)
    parser.add_argument("--tracking-trials-per-count", type=int, default=100)
    parser.add_argument("--tracking-frames", type=int, default=20)
    args = parser.parse_args()
    summary = run(
        CounterexampleConfig(
            seed=args.seed,
            random_states=args.random_states,
            abrupt_states=args.abrupt_states,
            tracking_trials_per_count=args.tracking_trials_per_count,
            tracking_frames=args.tracking_frames,
            output_dir=args.output_dir,
        )
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
