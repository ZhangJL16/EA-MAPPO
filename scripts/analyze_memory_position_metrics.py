from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.memory_safety.posthoc_position_metrics import run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-artifact", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    result = run(Path(args.source_artifact), Path(args.output_dir))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
