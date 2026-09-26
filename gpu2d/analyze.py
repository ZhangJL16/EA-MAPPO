"""Descriptive paired report for GPU horizon and exact CPU comparator."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CPU_MATRIX = ROOT / "artifacts" / "dual_constraint_2d_static_matrix_v1_20260926"


def _row(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def analyze(output: Path) -> dict:
    maps = []
    for map_id in range(40, 56):
        h4 = _row(output / "test" / "gpu_mpc_h4" / f"map_{map_id:03d}" / "summary.json")
        h8 = _row(output / "test" / "gpu_mpc_h8" / f"map_{map_id:03d}" / "summary.json")
        cpu = _row(CPU_MATRIX / "test" / "mpc_h2" / f"map_{map_id:03d}" / "summary.json")
        if h4 and h8:
            maps.append({"map_id": map_id, "gpu_h4": h4, "gpu_h8": h8, "cpu_h2": cpu})
    result = {
        "protocol": "gpu_ranked_mpc_descriptive_pairs_v0",
        "paired_gpu_test_maps": [row["map_id"] for row in maps],
        "complete_frozen_test": len(maps) == 16,
        "gpu_h8_minus_h4_completed_mean": (
            float(np.mean([row["gpu_h8"]["completed_targets"] -
                           row["gpu_h4"]["completed_targets"] for row in maps]))
            if maps else None
        ),
        "gpu_h4_completed_mean": (float(np.mean([
            row["gpu_h4"]["completed_targets"] for row in maps
        ])) if maps else None),
        "gpu_h8_completed_mean": (float(np.mean([
            row["gpu_h8"]["completed_targets"] for row in maps
        ])) if maps else None),
        "gpu_h4_search_wall_seconds_total": sum(
            row["gpu_h4"]["gpu_wall_seconds"] +
            row["gpu_h4"]["exact_wall_seconds"] for row in maps
        ),
        "gpu_h8_search_wall_seconds_total": sum(
            row["gpu_h8"]["gpu_wall_seconds"] +
            row["gpu_h8"]["exact_wall_seconds"] for row in maps
        ),
        "cpu_h2_available_paired_maps": [row["map_id"] for row in maps if row["cpu_h2"]],
        "interpretation_limit": (
            "Incomplete-map means are exploratory and can reflect runtime selection. "
            "GPU proposals are approximate; only the unchanged CPU shield certifies "
            "executed blocks.  A full comparison requires all 16 frozen test maps."
        ),
    }
    (output / "PAIR_REPORT.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result
