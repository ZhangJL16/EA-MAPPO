#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", default="artifacts/new_route/e1_energy")
    parser.add_argument("--output", default="artifacts/new_route/e1_energy_aggregate.json")
    args = parser.parse_args()
    results = []
    for path in sorted(Path(args.root).glob("seed*/results.json")):
        if not (path.parent / "COMPLETED.json").exists():
            continue
        results.append(json.loads(path.read_text(encoding="utf-8")))
    if len(results) < 3:
        raise RuntimeError("at least three completed seeds are required for evaluation")
    Path(args.output).write_text(json.dumps({"seed_count": len(results), "results": results}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
