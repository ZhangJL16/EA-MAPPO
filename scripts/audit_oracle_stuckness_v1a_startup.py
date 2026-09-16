#!/usr/bin/env python3
"""Integrity-only v1a startup audit; never reads tasks/hour."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.validate_oracle_stuckness_v1a_contract import validate


def audit(contract_path: Path, raw: Path, split: str) -> dict:
    integrity = validate(contract_path)
    contract = json.loads(contract_path.read_text())
    worlds = [w for w in contract["worlds"] if w["split"] == split]
    if split == "DEV":
        worlds = worlds[:1]
    duration = 120.0 if split == "SMOKE_DEBUG" else 7200.0
    errors = list(integrity["errors"])
    probe_events = timestamp_records = 0
    wall = []
    for world in worlds:
        identities = set()
        directory = raw / f"world_{world['world_index']:03d}_{world['world_seed']}"
        for method in range(5):
            path = directory / f"M{method}.json"
            if not path.exists():
                errors.append(f"missing {path}")
                continue
            row = json.loads(path.read_text())
            if not row.get("completed") or row.get("schema_version") != "oracle-stuckness-method-result-v1a":
                errors.append(f"invalid/incomplete {path}")
            if row.get("configured_denominator_seconds") != duration:
                errors.append(f"denominator mismatch {path}")
            if row.get("tasks_completed") != len(row.get("task_completion_timestamps", [])):
                errors.append(f"timestamp/count mismatch {path}")
            if any(float(t) > duration for t in row.get("task_completion_timestamps", [])):
                errors.append(f"post-horizon timestamp retained {path}")
            timestamp_records += int("task_completion_timestamps" in row)
            identities.add(row.get("physical_world_identity"))
            wall.append(float(row.get("wall_seconds", 0)))
            if method == 4:
                for event in row.get("events", []):
                    if "state_rng_sha256" in event:
                        probe_events += 1
                        if event["state_rng_sha256"] != event["restored_state_rng_sha256"]:
                            errors.append(f"probe restore mismatch {path}")
        if identities != {world["physical_world_identity"]}:
            errors.append(f"pairing mismatch world {world['world_index']}")
    return {"schema_version": "oracle-stuckness-v1a-startup-audit-v1",
            "healthy": not errors, "errors": errors, "split": split,
            "world_grids_expected": len(worlds), "method_records_expected": 5 * len(worlds),
            "timestamp_records_checked": timestamp_records,
            "probe_events_checked": probe_events,
            "runtime_wall_seconds_min": min(wall) if wall else None,
            "runtime_wall_seconds_max": max(wall) if wall else None,
            "confirm_access_count": integrity["confirm_access_count"],
            "scientific_outcomes_inspected": False, "formal_gate_analysis_executed": False}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--split", choices=("SMOKE_DEBUG", "DEV"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.contract, args.raw, args.split)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["healthy"] else 1)


if __name__ == "__main__":
    main()
