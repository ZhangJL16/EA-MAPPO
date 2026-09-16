#!/usr/bin/env python3
"""Freeze the prospective v1a amendment while preserving the v1 lineage."""
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

from scripts.validate_oracle_stuckness_v1a_contract import file_hash, object_hash, validate

SOURCES = (
    "docs/ORACLE_PRODUCTIVE_STUCKNESS_V1A_AMENDMENT_20260916.md",
    "experiments/directional_navigation/oracle_stuckness_intervention.py",
    "experiments/directional_navigation/threshold_stress.py",
    "experiments/directional_navigation/battery_sortie.py",
    "experiments/directional_navigation/standard_baselines.py",
    "scripts/run_dvoi_h_collection.py",
    "scripts/run_oracle_stuckness_intervention.py",
    "scripts/run_oracle_stuckness_intervention_v1a.py",
    "scripts/analyze_oracle_stuckness_v1a.py",
    "scripts/validate_oracle_stuckness_v1a_contract.py",
    "scripts/freeze_oracle_stuckness_v1a.py",
    "scripts/audit_oracle_stuckness_v1a_startup.py",
    "tests/test_oracle_stuckness_v1a.py",
    "envs/UAVEnergyDeliverySAC.py",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.artifact_root.resolve()
    if root.exists() and any(root.iterdir()):
        raise SystemExit("refusing to overwrite nonempty v1a artifact root")
    root.mkdir(parents=True, exist_ok=True)
    parent_root = ROOT / "artifacts/oracle_stuckness_intervention_20260916"
    parent_contract_path = parent_root / "contract.json"
    smoke_audit_path = parent_root / "smoke_startup_audit.json"
    dev_audit_path = parent_root / "dev_startup_audit.json"
    parent = json.loads(parent_contract_path.read_text())
    amendment = {
        "amendment_id": "oracle-productive-stuckness-20260916-v1a",
        "frozen_at_unix": time.time(),
        "prospective_status": "no scientific outcome inspected; no Gate analysis run",
        "v1_outputs": "integrity/startup only; excluded from formal v1a analysis",
        "changes": [
            "task-completion timestamps and common-alive survival accounting",
            "four simultaneous M4-M0/M1/M2/M3 max-t contrasts",
            "resumable configurable DEV stop-after-worlds execution path",
        ],
        "unchanged": [
            "M0-M4 decisions", "worlds and seeds", "SOC40", "4000 and 256 thresholds",
            "7200-second horizon", "collision semantics", "navigator", "plant physics",
        ],
        "current_execution_authorization": "one smoke plus one first-DEV startup grid only",
        "formal_analysis_authorized": False,
        "remaining_23_dev_authorized": False,
    }
    amendment["record_hash"] = object_hash(amendment)
    source_registry = {rel: file_hash(ROOT / rel) for rel in SOURCES}
    contract = {
        "schema_version": "oracle-stuckness-contract-v1a",
        "confirm_allocated": False,
        "parent": {
            "contract_path": str(parent_contract_path.relative_to(ROOT)),
            "contract_sha256": file_hash(parent_contract_path),
            "smoke_audit_path": str(smoke_audit_path.relative_to(ROOT)),
            "smoke_audit_sha256": file_hash(smoke_audit_path),
            "dev_audit_path": str(dev_audit_path.relative_to(ROOT)),
            "dev_audit_sha256": file_hash(dev_audit_path),
        },
        "methods": parent["methods"], "worlds": parent["worlds"],
        "source_registry": source_registry, "amendment": amendment,
        "access_events": [],
    }
    (root / "contract.json").write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    receipt = {
        "schema_version": "oracle-stuckness-v1a-local-freeze-receipt-v1",
        "amendment_record_hash": amendment["record_hash"],
        "source_registry_hash": object_hash(source_registry),
        "world_registry_hash": object_hash(contract["worlds"]),
        "parent_contract_sha256": contract["parent"]["contract_sha256"],
        "note": "Local content-addressed amendment receipt; no external log claim.",
    }
    (root / "freeze_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    for rel in SOURCES:
        destination = root / "frozen_sources" / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, destination)
    result = validate(root / "contract.json")
    (root / "freeze_validation.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if not result["valid"]:
        raise SystemExit("v1a freeze invalid: " + "; ".join(result["errors"]))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
