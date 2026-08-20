from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.memory_safety.hidden_size_ablation import (
    HiddenSizeAblationConfig,
    run,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--source-artifact",
        default="artifacts/memory_estimation_5k_20260819_v2",
    )
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--epochs", type=int, default=25)
    args = parser.parse_args()
    summary = run(
        HiddenSizeAblationConfig(
            seed=args.seed,
            source_artifact=args.source_artifact,
            epochs=args.epochs,
            output_dir=args.output_dir,
        )
    )
    compact = {
        name: {
            "joint_mae": row["metrics"]["joint_mae"],
            "p99_ms": row["latency"]["p99_ms"],
            "parameters": row["parameter_count"],
        }
        for name, row in summary["methods"].items()
    }
    print(json.dumps(compact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
