from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.memory_safety.closed_loop import ClosedLoopConfig, run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--scenarios", type=int, default=12)
    parser.add_argument("--max-steps", type=int, default=800)
    parser.add_argument("--sac-checkpoint")
    parser.add_argument("--estimation-artifact")
    args = parser.parse_args()
    defaults = ClosedLoopConfig()
    summary = run(
        ClosedLoopConfig(
            seed=args.seed,
            scenarios=args.scenarios,
            max_steps=args.max_steps,
            sac_checkpoint=args.sac_checkpoint or defaults.sac_checkpoint,
            estimation_artifact=args.estimation_artifact or defaults.estimation_artifact,
            output_dir=args.output_dir,
        )
    )
    print(json.dumps(summary["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
