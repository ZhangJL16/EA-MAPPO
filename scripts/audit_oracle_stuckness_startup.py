#!/usr/bin/env python3
"""Integrity-only startup audit. Deliberately omits scientific outcomes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.validate_oracle_stuckness_contract import validate


def audit(contract_path: Path, raw: Path, split: str) -> dict[str, Any]:
    integrity = validate(contract_path)
    contract = json.loads(contract_path.read_text())
    worlds = [row for row in contract["worlds"] if row["split"] == split]
    expected_duration = 120.0 if split == "SMOKE_DEBUG" else 7200.0
    if split == "DEV":
        worlds = worlds[:1]
    errors = list(integrity["errors"])
    wall_seconds: list[float] = []
    probe_events = 0
    for world in worlds:
        directory = raw / f"world_{int(world['world_index']):03d}_{int(world['world_seed'])}"
        identities = set()
        for method in range(5):
            path = directory / f"M{method}.json"
            if not path.exists():
                errors.append(f"missing method result: {path}")
                continue
            row = json.loads(path.read_text())
            identities.add(row.get("physical_world_identity"))
            if not row.get("completed"):
                errors.append(f"incomplete method result: {path}")
            if row.get("method") != method:
                errors.append(f"method mismatch: {path}")
            if float(row.get("configured_denominator_seconds", -1)) != expected_duration:
                errors.append(f"denominator mismatch: {path}")
            wall_seconds.append(float(row.get("wall_seconds", 0.0)))
            if method == 4:
                for event in row.get("events", []):
                    if "state_rng_sha256" in event:
                        probe_events += 1
                        if event["state_rng_sha256"] != event["restored_state_rng_sha256"]:
                            errors.append(f"probe restoration leakage: {path}")
        if identities != {world["physical_world_identity"]}:
            errors.append(f"pairing/identity mismatch for world {world['world_index']}")
    return {
        "schema_version": "oracle-stuckness-startup-audit-v1",
        "healthy": not errors,
        "errors": errors,
        "split": split,
        "world_grids_expected": len(worlds),
        "method_records_expected": 5 * len(worlds),
        "probe_events_checked": probe_events,
        "runtime_wall_seconds_min": min(wall_seconds) if wall_seconds else None,
        "runtime_wall_seconds_max": max(wall_seconds) if wall_seconds else None,
        "confirm_access_count": integrity["confirm_access_count"],
        "scientific_outcomes_inspected": False,
        "formal_gate_analysis_executed": False,
    }


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
