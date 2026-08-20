from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.memory_safety.adaptation_analysis import AdaptationConfig, run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-dir",
        default="artifacts/memory_estimation_5k_20260819_v2",
    )
    parser.add_argument("--output-path")
    parser.add_argument("--velocity-l2-threshold", type=float, default=1.0)
    args = parser.parse_args()
    output = args.output_path or str(Path(args.artifact_dir) / "adaptation_summary.json")
    summary = run(
        AdaptationConfig(
            artifact_dir=args.artifact_dir,
            output_path=output,
            velocity_l2_threshold=args.velocity_l2_threshold,
        )
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
