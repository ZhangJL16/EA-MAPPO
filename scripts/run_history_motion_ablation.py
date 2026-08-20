from __future__ import annotations

import argparse
import json

from experiments.history_safe_trajectory.history_ablation import (
    HistoryAblationConfig,
    run_history_ablation,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cases", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260820)
    args = parser.parse_args()
    summary = run_history_ablation(
        HistoryAblationConfig(
            output_dir=args.output_dir,
            cases=args.cases,
            seed=args.seed,
        )
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
