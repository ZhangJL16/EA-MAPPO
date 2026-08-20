from __future__ import annotations

import argparse
import json

from experiments.history_safe_trajectory.adaptive_tube_ablation import (
    AdaptiveTubeAblationConfig,
    run_adaptive_tube_ablation,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cases-per-regime", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260819)
    args = parser.parse_args()
    summary = run_adaptive_tube_ablation(
        AdaptiveTubeAblationConfig(
            output_dir=args.output_dir,
            cases_per_regime=args.cases_per_regime,
            seed=args.seed,
        )
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
