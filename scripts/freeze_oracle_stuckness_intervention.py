#!/usr/bin/env python3
"""One-shot freezer for fresh smoke/DEV identities and frozen live sources."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.directional_navigation.oracle_stuckness_intervention import METHODS, OracleStucknessEnv
from scripts.validate_oracle_stuckness_contract import file_hash, object_hash, validate

SOURCES = (
    "docs/ORACLE_PRODUCTIVE_STUCKNESS_INTERVENTION_PLAN_20260916.md",
    "experiments/directional_navigation/oracle_stuckness_intervention.py",
    "experiments/directional_navigation/threshold_stress.py",
    "experiments/directional_navigation/battery_sortie.py",
    "experiments/directional_navigation/standard_baselines.py",
    "scripts/run_oracle_stuckness_intervention.py",
    "scripts/analyze_oracle_stuckness_dev.py",
    "scripts/audit_oracle_stuckness_startup.py",
    "scripts/validate_oracle_stuckness_contract.py",
    "scripts/freeze_oracle_stuckness_intervention.py",
    "scripts/run_dvoi_h_collection.py",
    "tests/test_oracle_stuckness_intervention.py",
    "envs/UAVEnergyDeliverySAC.py",
)


def world(seed: int, index: int, split: str) -> dict:
    job = dict(job_id=index * 5, world_seed=seed, head_seed=0, method=0,
               method_name=METHODS[0], threshold=.4, obstacles=48, soc=1.)
    env = OracleStucknessEnv(job)
    try:
        env.reset()
        record = env.physical_world_record()
        return {
            "world_index": index,
            "world_seed": seed,
            "split": split,
            "physical_world_identity": object_hash(record),
            "physical_world_record": record,
        }
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=1_138_000_001)
    args = parser.parse_args()
    root = args.artifact_root.resolve()
    if root.exists() and any(root.iterdir()):
        raise SystemExit("refusing to overwrite nonempty artifact root")
    root.mkdir(parents=True, exist_ok=True)
    source_registry = {rel: file_hash(ROOT / rel) for rel in SOURCES}
    worlds = [world(args.seed_start, 0, "SMOKE_DEBUG")]
    worlds.extend(world(args.seed_start + i, i, "DEV") for i in range(1, 25))
    preaccess = {
        "contract_id": "oracle-productive-stuckness-20260916-v1",
        "frozen_at_unix": time.time(),
        "duration_seconds": {"SMOKE_DEBUG": 120.0, "DEV": 7200.0},
        "method_order": list(range(5)),
        "formal_world_count": 24,
        "smoke_world_count": 1,
        "automatic_analysis": False,
        "training_authorized": False,
        "startup_stop_after_dev_worlds": 1,
    }
    preaccess["record_hash"] = object_hash(preaccess)
    contract = {
        "schema_version": "oracle-stuckness-contract-v1",
        "confirm_allocated": False,
        "methods": [{"method": m, "name": METHODS[m]} for m in range(5)],
        "worlds": worlds,
        "source_registry": source_registry,
        "preaccess": preaccess,
        "access_events": [],
    }
    (root / "contract.json").write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    receipt = {
        "schema_version": "oracle-stuckness-local-freeze-receipt-v1",
        "preaccess_record_hash": preaccess["record_hash"],
        "source_registry_hash": object_hash(source_registry),
        "world_registry_hash": object_hash(worlds),
        "note": "Local content-addressed freeze receipt; no external transparency-log claim.",
    }
    (root / "freeze_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    frozen = root / "frozen_sources"
    for rel in SOURCES:
        destination = frozen / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, destination)
    result = validate(root / "contract.json")
    (root / "freeze_validation.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if not result["valid"]:
        raise SystemExit("freeze validation failed: " + "; ".join(result["errors"]))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
