#!/usr/bin/env python3
"""Integrity and no-leakage checks for the oracle-stuckness experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def object_hash(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _identity_values(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if "identity" in key.lower() and isinstance(item, str) and item.startswith("sha256:"):
                found.add(item)
            found.update(_identity_values(item))
    elif isinstance(value, list):
        for item in value:
            found.update(_identity_values(item))
    return found


def _confirm_access_count(value: Any) -> int:
    count = 0
    if isinstance(value, dict):
        split = str(value.get("split", value.get("assigned_split", ""))).upper()
        if "CONFIRM" in split and any(k in value for k in ("accessed_at", "event_hash", "access_type")):
            count += 1
        for item in value.values():
            count += _confirm_access_count(item)
    elif isinstance(value, list):
        for item in value:
            count += _confirm_access_count(item)
    return count


def validate(contract_path: Path, *, scan_prior: bool = True) -> dict[str, Any]:
    contract_path = contract_path.resolve()
    contract = json.loads(contract_path.read_text())
    errors: list[str] = []
    if contract.get("schema_version") != "oracle-stuckness-contract-v1":
        errors.append("schema_version mismatch")
    if contract.get("confirm_allocated") is not False:
        errors.append("CONFIRM must not be allocated")
    methods = contract.get("methods", [])
    if [row.get("method") for row in methods] != list(range(5)):
        errors.append("methods M0--M4 are not frozen exactly once")
    worlds = contract.get("worlds", [])
    identities = [row.get("physical_world_identity") for row in worlds]
    if len(worlds) != 25 or len(set(identities)) != len(identities):
        errors.append("expected 25 unique worlds (1 smoke + 24 DEV)")
    if sum(row.get("split") == "SMOKE_DEBUG" for row in worlds) != 1:
        errors.append("expected exactly one SMOKE_DEBUG world")
    if sum(row.get("split") == "DEV" for row in worlds) != 24:
        errors.append("expected exactly 24 DEV worlds")
    for rel, expected in contract.get("source_registry", {}).items():
        path = ROOT / rel
        if not path.exists() or file_hash(path) != expected:
            errors.append(f"source hash mismatch: {rel}")
    preaccess = contract.get("preaccess", {})
    if preaccess.get("record_hash") != object_hash({k: v for k, v in preaccess.items() if k != "record_hash"}):
        errors.append("preaccess record hash mismatch")
    receipt_path = contract_path.parent / "freeze_receipt.json"
    if not receipt_path.exists():
        errors.append("freeze_receipt.json missing")
    else:
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("preaccess_record_hash") != preaccess.get("record_hash"):
            errors.append("freeze receipt does not bind the preaccess record")
        if receipt.get("source_registry_hash") != object_hash(contract.get("source_registry", {})):
            errors.append("freeze receipt source registry hash mismatch")
    overlap: dict[str, list[str]] = {}
    confirm_access = 0
    if scan_prior:
        current = set(x for x in identities if isinstance(x, str))
        for path in (ROOT / "artifacts").glob("**/contract.json"):
            if path.resolve() == contract_path:
                continue
            try:
                other = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            common = sorted(current & _identity_values(other))
            if common:
                overlap[str(path.relative_to(ROOT))] = common
            confirm_access += _confirm_access_count(other)
        if overlap:
            errors.append(f"physical identity overlap with retained lineages: {sorted(overlap)}")
    confirm_access += _confirm_access_count(contract)
    if confirm_access:
        errors.append(f"CONFIRM access count must be zero, observed {confirm_access}")
    return {
        "valid": not errors,
        "errors": errors,
        "world_count": len(worlds),
        "identity_overlap": overlap,
        "confirm_access_count": confirm_access,
        "contract_path": str(contract_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("contract", type=Path)
    parser.add_argument("--no-prior-scan", action="store_true")
    args = parser.parse_args()
    result = validate(args.contract, scan_prior=not args.no_prior_scan)
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["valid"] else 1)


if __name__ == "__main__":
    main()
