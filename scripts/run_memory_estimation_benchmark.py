from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.memory_safety.estimation_benchmark import (
    EstimationBenchmarkConfig,
    run,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--train-trajectories", type=int, default=3000)
    parser.add_argument("--validation-trajectories", type=int, default=1000)
    parser.add_argument("--test-trajectories", type=int, default=1000)
    args = parser.parse_args()
    summary = run(
        EstimationBenchmarkConfig(
            seed=args.seed,
            epochs=args.epochs,
            train_trajectories=args.train_trajectories,
            validation_trajectories=args.validation_trajectories,
            test_trajectories=args.test_trajectories,
            output_dir=args.output_dir,
        )
    )
    compact = {
        "device": summary["device"],
        "ranking_by_joint_mae": summary["ranking_by_joint_mae"],
        "metrics": {
            name: values["metrics"]
            for name, values in summary["methods"].items()
        },
    }
    print(json.dumps(compact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
