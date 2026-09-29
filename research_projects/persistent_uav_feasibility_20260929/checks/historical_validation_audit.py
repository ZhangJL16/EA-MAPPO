"""Read-only audit of historical full-window B4/B5 validation records.

This script does not run a UAV simulator and does not select a deployable policy.
"""

from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
EVIDENCE = ROOT / "persistent_uav_throughput_v1" / "evidence"
SOURCES = {
    "B4": EVIDENCE / "arm_validation_20260920" / "results.json",
    "B5": EVIDENCE / "b5_validation_complete_20260920" / "results.json",
}


def audit():
    report = {
        "interpretation": (
            "historical validation/tuning only; no independent policy certification, "
            "no new UAV simulation"
        ),
        "sources": {},
        "methods": {},
    }
    for method, path in SOURCES.items():
        payload = path.read_bytes()
        rows = json.loads(payload)
        report["sources"][method] = {
            "path": str(path),
            "sha256": sha256(payload).hexdigest(),
            "rows": len(rows),
        }
        expected_rows = 1080 if method == "B4" else 270
        assert len(rows) == expected_rows
        cells = defaultdict(list)
        for row in rows:
            assert row["done"] and abs(
                row["simulation_time"] - row["evaluation_horizon"]
            ) <= 1e-6
            assert row["depletion"] == (row["failure"] == "energy_depletion")
            assert row["navigation_failure"] == (row["failure"] == "navigation_failure")
            cells[(row["regime"], row["threshold"])].append(row)
        assert len(cells) == (108 if method == "B4" else 27)
        cell_rows = []
        for (regime, threshold), cases in sorted(cells.items()):
            assert len(cases) == 10 and len({x["seed"] for x in cases}) == 10
            cell_rows.append({
                "regime": regime,
                "threshold": threshold,
                "n": len(cases),
                "depletion_count": sum(x["depletion"] for x in cases),
                "navigation_failure_count": sum(x["navigation_failure"] for x in cases),
                "mean_completed": sum(x["completed"] for x in cases) / len(cases),
            })
        report["methods"][method] = {
            "total_depletions": sum(x["depletion_count"] for x in cell_rows),
            "total_navigation_failures": sum(
                x["navigation_failure_count"] for x in cell_rows
            ),
            "minimum_cell_depletions": min(x["depletion_count"] for x in cell_rows),
            "zero_depletion_cells": sum(x["depletion_count"] == 0 for x in cell_rows),
            "cells": cell_rows,
        }
    return report


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2, ensure_ascii=False, sort_keys=True))
