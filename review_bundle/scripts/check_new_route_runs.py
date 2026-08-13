#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", default="artifacts/new_route")
    args = parser.parse_args()
    root = Path(args.root)
    rows = []
    for config in sorted(root.rglob("config.json")):
        run = config.parent
        marker = next((name for name in ("COMPLETED.json", "FAILED.json", "RUNNING.json") if (run / name).exists()), "MISSING")
        payload = json.loads((run / marker).read_text(encoding="utf-8")) if marker != "MISSING" else {}
        rows.append({"run": str(run), "marker": marker, "status": payload.get("status", "UNKNOWN")})
    print(json.dumps(rows, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
