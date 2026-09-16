#!/usr/bin/env python3
"""Freeze and validate the PSPS-v1 execution-only 4x8 -> 3x8 overlay."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_conditional_history_collection import atomic_json
from scripts.validate_identification_contract import file_hash, object_hash, record_hash
from scripts.validate_psps_v1_contract import validate_contract

SCHEMA = "psps-v1-execution-amendment-v1"
SNAPSHOT_SCHEMA = "psps-v1-pre-amendment-evidence-snapshot-v1"
OLD_EXECUTION = {"world_processes": 4, "workers_per_world": 8, "max_worlds_per_child": 1}
NEW_EXECUTION = {"world_processes": 3, "workers_per_world": 8, "max_worlds_per_child": 1}
IMMUTABLE_AREAS = ("records", "staged", "partials", "incidents")


def immutable_inventory(raw: Path) -> dict[str, str]:
    paths: list[Path] = []
    for area in IMMUTABLE_AREAS:
        directory = raw / area
        if directory.exists():
            paths.extend(path for path in directory.rglob("*") if path.is_file())
    return {path.relative_to(raw).as_posix(): file_hash(path) for path in sorted(paths)}


def scientific_spec_hash(contract: dict[str, Any]) -> str:
    spec = dict(contract["pre_h"]["static_spec"])
    spec.pop("execution")
    return object_hash(spec)


def validate_amendment(
    amendment_path: Path, contract_path: Path, receipt_path: Path, raw: Path
) -> dict[str, Any]:
    errors: list[str] = []
    contract = json.loads(contract_path.read_text())
    receipt = json.loads(receipt_path.read_text())
    base = validate_contract(contract, contract_path.parent, receipt)
    if not base["PSPS_DEV_READY"]:
        errors.append("base PSPS-v1 contract is not DEV-ready")
    amendment = json.loads(amendment_path.read_text())
    if amendment.get("schema_version") != SCHEMA or amendment.get("record_sha256") != record_hash(amendment):
        errors.append("execution amendment schema/hash mismatch")
    if amendment.get("base_preaccess_record_sha256") != contract["pre_h"]["record_sha256"]:
        errors.append("execution amendment has wrong base preaccess record")
    if amendment.get("base_freeze_receipt_sha256") != file_hash(receipt_path):
        errors.append("execution amendment has wrong base receipt")
    if amendment.get("scientific_spec_without_execution_sha256") != scientific_spec_hash(contract):
        errors.append("scientific specification changed")
    if amendment.get("old_execution") != OLD_EXECUTION or amendment.get("new_execution") != NEW_EXECUTION:
        errors.append("execution amendment is not exactly 4x8 -> 3x8")
    if amendment.get("automatic_analysis") or amendment.get("automatic_confirmation") or amendment.get("automatic_training"):
        errors.append("automatic downstream action was enabled")

    manifest = raw / "manifest.json"
    if amendment.get("original_manifest_sha256") != file_hash(manifest):
        errors.append("original scientific manifest changed")
    snapshot_path = amendment_path.parent / amendment.get("evidence_snapshot", "missing")
    if amendment.get("evidence_snapshot_sha256") != file_hash(snapshot_path):
        errors.append("evidence snapshot hash mismatch")
        snapshot: dict[str, Any] = {}
    else:
        snapshot = json.loads(snapshot_path.read_text())
    if snapshot.get("schema_version") != SNAPSHOT_SCHEMA:
        errors.append("evidence snapshot schema mismatch")
    else:
        for relative, expected in snapshot.get("immutable_files", {}).items():
            path = raw / relative
            if not path.is_file() or file_hash(path) != expected:
                errors.append(f"pre-amendment evidence changed: {relative}")
        latest = json.loads((raw / "latest.json").read_text())
        prefix = snapshot.get("completed_prefix", [])
        if latest.get("completed", [])[: len(prefix)] != prefix:
            errors.append("ordered completed prefix changed")

    events = contract["access_registry"]["events"]
    event_count = amendment.get("access_event_count_at_amendment")
    if not isinstance(event_count, int) or event_count < 1 or len(events) < event_count:
        errors.append("access-chain amendment boundary unavailable")
    elif events[event_count - 1].get("event_hash") != amendment.get("access_head_at_amendment"):
        errors.append("access-chain amendment boundary changed")

    sources = amendment.get("source_hashes", {})
    expected_sources = {
        "original_collector": ROOT / "scripts/run_psps_v1_collection.py",
        "execution_overlay_collector": ROOT / "scripts/run_psps_v1_collection_3x8.py",
        "amendment_tool": ROOT / "scripts/amend_psps_v1_parallelism.py",
    }
    for name, path in expected_sources.items():
        if sources.get(name) != file_hash(path):
            errors.append(f"execution amendment source changed: {name}")
    return {
        "schema_version": "psps-v1-execution-amendment-validation-v1",
        "valid": not errors,
        "errors": errors,
        "world_processes": 3,
        "workers_per_world": 8,
        "preserved_file_count": len(snapshot.get("immutable_files", {})),
        "snapshot_generation": snapshot.get("generation"),
        "psps_confirm_accessed": base.get("psps_confirm_accessed"),
        "old_pai_confirm_accessed": base.get("old_pai_confirm_accessed"),
        "method_train_authorized": False,
    }


def freeze(args: argparse.Namespace) -> None:
    contract_path = args.contract.resolve()
    receipt_path = args.receipt.resolve()
    raw = args.raw_dir.resolve()
    output = args.output_dir.resolve()
    contract = json.loads(contract_path.read_text())
    receipt = json.loads(receipt_path.read_text())
    base = validate_contract(contract, contract_path.parent, receipt)
    if not base["PSPS_DEV_READY"]:
        raise ValueError("base PSPS-v1 contract is not DEV-ready")
    status = json.loads((raw / "status.json").read_text())
    if status.get("status") != "PAUSED_RESUMABLE" or status.get("active") != 0:
        raise ValueError("collector is not at a resumable inactive boundary")
    if not (raw / "PAUSE").exists():
        raise ValueError("pause marker is required while freezing amendment")
    manifest = json.loads((raw / "manifest.json").read_text())
    if {key: manifest.get(key) for key in ("world_processes", "workers_per_world", "max_worlds_per_child")} != OLD_EXECUTION:
        raise ValueError("original execution manifest is not 4x8")
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)

    latest = json.loads((raw / "latest.json").read_text())
    inventory = immutable_inventory(raw)
    snapshot = {
        "schema_version": SNAPSHOT_SCHEMA,
        "generation": latest["generation"],
        "completed_prefix": latest["completed"],
        "latest_records": latest["records"],
        "total_branch_policy_steps": latest["total_branch_policy_steps"],
        "staged_count": len(list((raw / "staged").glob("*.json"))),
        "partial_json_count": len(list((raw / "partials").rglob("replicate_*.json"))),
        "immutable_files": inventory,
        "immutable_inventory_sha256": object_hash(inventory),
    }
    snapshot_path = output / "pre_amendment_evidence_snapshot.json"
    atomic_json(snapshot_path, snapshot)
    registry = contract["access_registry"]
    amendment = {
        "schema_version": SCHEMA,
        "base_preaccess_record_sha256": contract["pre_h"]["record_sha256"],
        "base_contract_sha256_at_amendment": file_hash(contract_path),
        "base_freeze_receipt_sha256": file_hash(receipt_path),
        "access_event_count_at_amendment": registry["event_count"],
        "access_head_at_amendment": registry["head_event_hash"],
        "scientific_spec_without_execution_sha256": scientific_spec_hash(contract),
        "original_manifest_sha256": file_hash(raw / "manifest.json"),
        "evidence_snapshot": snapshot_path.name,
        "evidence_snapshot_sha256": file_hash(snapshot_path),
        "old_execution": OLD_EXECUTION,
        "new_execution": NEW_EXECUTION,
        "scope": "EXECUTION_PARALLELISM_ONLY_NO_SCIENTIFIC_CHANGE",
        "source_hashes": {
            "original_collector": file_hash(ROOT / "scripts/run_psps_v1_collection.py"),
            "execution_overlay_collector": file_hash(ROOT / "scripts/run_psps_v1_collection_3x8.py"),
            "amendment_tool": file_hash(Path(__file__).resolve()),
        },
        "automatic_analysis": False,
        "automatic_confirmation": False,
        "automatic_training": False,
        "method_train_authorized": False,
    }
    amendment["record_sha256"] = record_hash(amendment)
    amendment_path = output / "amendment.json"
    atomic_json(amendment_path, amendment)
    result = validate_amendment(amendment_path, contract_path, receipt_path, raw)
    if not result["valid"]:
        raise ValueError("; ".join(result["errors"]))
    print(json.dumps({"amendment": str(amendment_path), "record_sha256": amendment["record_sha256"], **result}, indent=2, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--contract", type=Path, required=True)
    common.add_argument("--receipt", type=Path, required=True)
    common.add_argument("--raw-dir", type=Path, required=True)
    freeze_parser = subparsers.add_parser("freeze", parents=[common])
    freeze_parser.add_argument("--output-dir", type=Path, required=True)
    validate_parser = subparsers.add_parser("validate", parents=[common])
    validate_parser.add_argument("--amendment", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        freeze(args)
        return 0
    result = validate_amendment(args.amendment.resolve(), args.contract.resolve(), args.receipt.resolve(), args.raw_dir.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
