#!/usr/bin/env python3
"""Validate prospective v1a amendment bindings, sources, identities, and access."""
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
    out: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if "identity" in key.lower() and isinstance(item, str) and item.startswith("sha256:"):
                out.add(item)
            out |= _identity_values(item)
    elif isinstance(value, list):
        for item in value:
            out |= _identity_values(item)
    return out


def _confirm_access(value: Any) -> int:
    if isinstance(value, dict):
        here = int("CONFIRM" in str(value.get("split", "")).upper()
                   and any(k in value for k in ("access_type", "accessed_at_unix", "event_hash")))
        return here + sum(_confirm_access(v) for v in value.values())
    if isinstance(value, list):
        return sum(_confirm_access(v) for v in value)
    return 0


def validate(path: Path, *, scan_prior: bool = True) -> dict[str, Any]:
    path = path.resolve()
    contract = json.loads(path.read_text())
    errors: list[str] = []
    if contract.get("schema_version") != "oracle-stuckness-contract-v1a":
        errors.append("schema mismatch")
    parent_path = ROOT / contract.get("parent", {}).get("contract_path", "")
    if not parent_path.exists() or file_hash(parent_path) != contract.get("parent", {}).get("contract_sha256"):
        errors.append("parent v1 contract binding mismatch")
        parent = {}
    else:
        parent = json.loads(parent_path.read_text())
    for key in ("smoke_audit_path", "dev_audit_path"):
        bound = contract.get("parent", {}).get(key)
        expected = contract.get("parent", {}).get(key.replace("_path", "_sha256"))
        if not bound or not (ROOT / bound).exists() or file_hash(ROOT / bound) != expected:
            errors.append(f"parent audit binding mismatch: {key}")
    if parent and contract.get("worlds") != parent.get("worlds"):
        errors.append("v1a worlds differ from v1")
    if parent and contract.get("methods") != parent.get("methods"):
        errors.append("v1a methods differ from v1")
    if contract.get("confirm_allocated") is not False:
        errors.append("CONFIRM allocated")
    identities = [w.get("physical_world_identity") for w in contract.get("worlds", [])]
    if len(identities) != 25 or len(set(identities)) != 25:
        errors.append("v1a requires the same 25 unique identities")
    for rel, expected in contract.get("source_registry", {}).items():
        if not (ROOT / rel).exists() or file_hash(ROOT / rel) != expected:
            errors.append(f"source hash mismatch: {rel}")
    amendment = contract.get("amendment", {})
    if amendment.get("record_hash") != object_hash({k: v for k, v in amendment.items() if k != "record_hash"}):
        errors.append("amendment record hash mismatch")
    receipt_path = path.parent / "freeze_receipt.json"
    if not receipt_path.exists():
        errors.append("freeze receipt missing")
    else:
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("amendment_record_hash") != amendment.get("record_hash"):
            errors.append("receipt amendment binding mismatch")
        if receipt.get("source_registry_hash") != object_hash(contract.get("source_registry", {})):
            errors.append("receipt source binding mismatch")
    overlap: dict[str, list[str]] = {}
    confirm = _confirm_access(contract)
    if scan_prior:
        current = set(identities)
        allowed_parent = parent_path.resolve() if parent_path.exists() else None
        for other_path in (ROOT / "artifacts").glob("**/contract.json"):
            if other_path.resolve() in {path, allowed_parent}:
                continue
            try:
                other = json.loads(other_path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            common = sorted(current & _identity_values(other))
            if common:
                overlap[str(other_path.relative_to(ROOT))] = common
            confirm += _confirm_access(other)
    if overlap:
        errors.append(f"identity overlap outside intentional v1 parent: {sorted(overlap)}")
    if confirm:
        errors.append(f"CONFIRM access count is {confirm}")
    return {"valid": not errors, "errors": errors, "world_count": len(identities),
            "intentional_parent_overlap_count": len(set(identities) & set(
                w.get("physical_world_identity") for w in parent.get("worlds", []))),
            "unapproved_identity_overlap": overlap, "confirm_access_count": confirm,
            "contract_path": str(path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("contract", type=Path)
    args = parser.parse_args()
    result = validate(args.contract)
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["valid"] else 1)


if __name__ == "__main__":
    main()
