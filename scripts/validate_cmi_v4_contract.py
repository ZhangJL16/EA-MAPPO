#!/usr/bin/env python3
"""Validate the fresh CMI-v4 execution-replay contract and access prefix."""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.validate_cmi_v3_contract import expected_pre_h_without_hash as v3_template
from scripts.validate_identification_contract import (
    access_event_hash,
    file_hash,
    object_hash,
    record_hash,
    seal_access_registry,
    validate_access_anchor,
    validate_content_registry,
    validate_registry,
)


SCHEMA = "conditional-history-mc-q-contract-v4"
PRE_H_SCHEMA = "conditional-history-mc-q-preaccess-v4"
SPLIT_SIZES = {"SMOKE_DEBUG": 2, "CMI_V3_DEV": 96, "CMI_V3_CONFIRM": 128}
SEED_RANGES = {
    "SMOKE_DEBUG": {"first": 1_135_000_227, "last": 1_135_000_228},
    "CMI_V3_DEV": {"first": 1_135_000_229, "last": 1_135_000_324},
    "CMI_V3_CONFIRM": {"first": 1_135_000_325, "last": 1_135_000_452},
}
LIVE_SOURCE_PATHS = {
    "protocol_v3": ROOT / "docs/CONDITIONAL_HISTORY_MC_Q_PROTOCOL_20260912_V3.md",
    "execution_addendum_v4": ROOT / "docs/CMI_V4_PARALLEL_EXECUTION_ADDENDUM_20260913.md",
    "freezer_v4": ROOT / "scripts/freeze_cmi_v4.py",
    "collector_v4": ROOT / "scripts/run_cmi_v4_parallel_collection.py",
    "validator_v4": ROOT / "scripts/validate_cmi_v4_contract.py",
    "environment": ROOT / "experiments/directional_navigation/conditional_model_access.py",
    "collector_common": ROOT / "scripts/run_conditional_history_collection.py",
    "models_v3": ROOT / "scripts/cmi_v3_models.py",
    "dev_analyzer_v3": ROOT / "scripts/analyze_cmi_v3_dev.py",
    "confirmation_freezer_v3": ROOT / "scripts/freeze_cmi_v3_confirmation.py",
    "confirmation_collector_v3": ROOT / "scripts/run_cmi_v3_confirmation.py",
    "confirmation_analyzer_v3": ROOT / "scripts/analyze_cmi_v3_confirmation.py",
    "dvoi_environment": ROOT / "experiments/directional_navigation/dvoi_h_branch.py",
    "frozen_navigator": ROOT / "scripts/run_dvoi_h_collection.py",
    "locked_recovery": ROOT / "experiments/directional_navigation/recovery.py",
    "collision_protocol": ROOT / "docs/LOCKED_COLLISION_RECOVERY_PROTOCOL.md",
}


def expected_pre_h_without_hash() -> dict[str, Any]:
    value = copy.deepcopy(v3_template())
    value["schema_version"] = PRE_H_SCHEMA
    value["world_seed_ranges"] = SEED_RANGES
    value.pop("superseded_zero_access_freeze_hash")
    value.update(
        {
            "execution_version": "WORLD_PROCESS_COORDINATOR_V4",
            "parent_v3_contract_hash": None,
            "parent_v3_checkpoint_hash": None,
            "acceleration_analysis_v2_hash": None,
            "execution_addendum_hash": None,
            "execution_rule": {
                "world_processes": 4,
                "workers_per_world": 8,
                "max_worlds_per_child": 1,
                "single_access_and_commit_writer": True,
                "ordered_contiguous_commits": True,
                "durable_out_of_order_tickets": True,
                "formal_smoke_data_used": False,
                "minimum_measured_speedup": 2.0,
                "measured_speedup": 2.105830182070052,
            },
        }
    )
    return value


def freeze_only_contract(contract: dict[str, Any]) -> dict[str, Any]:
    frozen = copy.deepcopy(contract)
    events = frozen["access_registry"]["events"]
    if len(events) < 1 or events[0].get("event_type") != "FREEZE":
        raise ValueError("CMI-v4 registry has no genesis freeze")
    frozen["access_registry"]["events"] = [copy.deepcopy(events[0])]
    seal_access_registry(frozen)
    return frozen


def validate_contract(
    contract: dict[str, Any], base_dir: Path, receipt: dict[str, Any] | None = None
) -> dict[str, Any]:
    errors: list[str] = []
    registered, content_errors = validate_content_registry(contract, base_dir)
    events, registry_errors = validate_registry(contract, registered)
    errors.extend(content_errors)
    errors.extend(registry_errors)
    if contract.get("schema_version") != SCHEMA:
        errors.append("invalid CMI-v4 contract schema")
    if contract.get("status") != "FROZEN_CMI_V4_DEV":
        errors.append("CMI-v4 contract is not frozen for DEV")
    pre = contract.get("pre_h")
    template = expected_pre_h_without_hash()
    if not isinstance(pre, dict) or set(pre) != set(template) | {"record_sha256"}:
        errors.append("CMI-v4 preaccess record has an invalid exact schema")
        pre = {}
    else:
        if pre.get("record_sha256") != record_hash(pre):
            errors.append("CMI-v4 preaccess record hash mismatch")
        for key, expected in template.items():
            if expected is not None and pre.get(key) != expected:
                errors.append(f"CMI-v4 frozen field changed: {key}")
        if pre.get("execution_addendum_hash") != pre.get("source_hashes", {}).get(
            "execution_addendum_v4"
        ):
            errors.append("CMI-v4 execution addendum binding changed")
        source_hashes = pre.get("source_hashes")
        if not isinstance(source_hashes, dict) or set(source_hashes) != set(
            LIVE_SOURCE_PATHS
        ):
            errors.append("CMI-v4 source-hash set is incomplete")
            source_hashes = {}
        for name, source in LIVE_SOURCE_PATHS.items():
            digest = source_hashes.get(name)
            entry = contract.get("content_registry", {}).get(digest, {})
            if not isinstance(entry, dict) or set(entry) != {"path"}:
                errors.append(f"CMI-v4 frozen source is not path-backed: {name}")
            elif file_hash(base_dir / entry["path"]) != digest:
                errors.append(f"CMI-v4 frozen source hash mismatch: {name}")
            if source.exists() and digest is not None and file_hash(source) != digest:
                errors.append(f"live CMI-v4 source differs from frozen source: {name}")

        splits = pre.get("master_splits", {})
        manifests = pre.get("world_manifest_hashes", {})
        identities: list[str] = []
        for split, size in SPLIT_SIZES.items():
            assigned = splits.get(split)
            if not isinstance(assigned, list) or len(assigned) != size:
                errors.append(f"CMI-v4 {split} size differs from {size}")
                continue
            identities.extend(assigned)
            entry = contract.get("content_registry", {}).get(manifests.get(split), {})
            rows = entry.get("inline") if isinstance(entry, dict) else None
            expected_seeds = list(
                range(SEED_RANGES[split]["first"], SEED_RANGES[split]["last"] + 1)
            )
            if not isinstance(rows, list) or [row.get("world_identity") for row in rows] != assigned:
                errors.append(f"CMI-v4 {split} manifest is not ordered/content-backed")
                continue
            if [row.get("world_seed") for row in rows] != expected_seeds:
                errors.append(f"CMI-v4 {split} seed allocation changed")
            for row in rows:
                identity, seed = row["world_identity"], row["world_seed"]
                path = base_dir / "worlds" / f"{identity.removeprefix('sha256:')}.json"
                try:
                    world = json.loads(path.read_text())
                except (OSError, json.JSONDecodeError):
                    errors.append(f"CMI-v4 world record unreadable: {identity}")
                    continue
                if object_hash(world) != identity or world.get("world_seed") != seed:
                    errors.append(f"CMI-v4 world identity/seed mismatch: {identity}")
                registry = contract.get("world_registry", {}).get(identity)
                if not isinstance(registry, dict) or registry.get("source_split") != split:
                    errors.append(f"CMI-v4 world registry mismatch: {identity}")
        if len(identities) != len(set(identities)):
            errors.append("CMI-v4 physical-world splits overlap")
        if set(contract.get("world_registry", {})) != set(identities):
            errors.append("CMI-v4 world registry differs from split union")
        old_worlds: set[str] = set()
        for old_root in (
            ROOT / "artifacts/conditional_history_information_20260912_v2",
            ROOT / "artifacts/conditional_history_mc_q_20260912_v3",
        ):
            try:
                old_contract = json.loads((old_root / "contract.json").read_text())
            except (OSError, json.JSONDecodeError):
                errors.append(f"CMI-v4 parent contract unavailable: {old_root.name}")
            else:
                old_worlds.update(old_contract.get("world_registry", {}))
        if old_worlds.intersection(identities):
            errors.append("CMI-v4 worlds overlap v2/v3 physical worlds")
        v3_root = ROOT / "artifacts/conditional_history_mc_q_20260912_v3"
        if pre.get("parent_v3_contract_hash") != file_hash(v3_root / "contract.json"):
            errors.append("CMI-v4 parent v3 contract hash changed")
        if pre.get("parent_v3_checkpoint_hash") != file_hash(
            v3_root / "cmi_v3_dev_raw_v1/latest.json"
        ):
            errors.append("CMI-v4 parent v3 checkpoint hash changed")
        acceleration = ROOT / "artifacts/cmi_v3_acceleration_smoke_20260913/analysis_v2.json"
        if pre.get("acceleration_analysis_v2_hash") != file_hash(acceleration):
            errors.append("CMI-v4 acceleration decision hash changed")
        else:
            decision = json.loads(acceleration.read_text())
            if (
                decision.get("decision") != "ELIGIBLE_FOR_FRESH_ACCELERATED_FREEZE"
                or decision.get("selected_processes") != 4
                or decision.get("selected_max_tasks_per_child") != 1
                or decision.get("selected_speedup", 0) < 2.0
                or not decision.get("all_exactly_equivalent")
            ):
                errors.append("CMI-v4 acceleration decision is not eligible")

    if events:
        first = events[0]
        if (
            first.get("event_type") != "FREEZE"
            or first.get("stage") != "CMI_V4_DEV"
            or first.get("record_hash") != pre.get("record_sha256")
        ):
            errors.append("CMI-v4 access chain has an invalid freeze genesis")
    accesses = [event for event in events if event.get("event_type") == "ACCESS"]
    observed = [
        event.get("world_identity")
        for event in accesses
        if event.get("split") == "CMI_V3_DEV"
    ]
    assigned = pre.get("master_splits", {}).get("CMI_V3_DEV", [])
    if observed != assigned[: len(observed)] or len(observed) != len(set(observed)):
        errors.append("CMI-v4 DEV access is not an ordered unique prefix")
    if any(event.get("split") == "CMI_V3_CONFIRM" for event in accesses):
        errors.append("CMI-v4 confirmation access is forbidden in DEV")

    receipt_errors: list[str] = []
    if receipt is None:
        receipt_errors.append("CMI-v4 freeze-time external receipt is required")
    else:
        try:
            frozen = freeze_only_contract(contract)
        except ValueError as error:
            receipt_errors.append(str(error))
        else:
            receipt_errors.extend(validate_access_anchor(frozen, receipt, registered))
    return {
        "schema_version": "conditional-history-mc-q-validation-v4",
        "valid": not errors,
        "errors": errors,
        "receipt_errors": receipt_errors,
        "CMI_V4_DEV_READY": not errors and not receipt_errors,
        "CMI_V4_CONFIRM_READY": False,
        "cmi_v4_dev_accessed": len(observed),
        "cmi_v4_confirm_accessed": len(
            [event for event in accesses if event.get("split") == "CMI_V3_CONFIRM"]
        ),
    }


def append_access_event(
    contract_path: Path, split: str, identity: str, run_id: str
) -> None:
    from scripts.freeze_dvoi_pre_h import atomic_json

    contract = json.loads(contract_path.read_text())
    events = contract["access_registry"]["events"]
    matching = [
        event
        for event in events
        if event.get("event_type") == "ACCESS"
        and event.get("split") == split
        and event.get("world_identity") == identity
    ]
    if matching:
        if len(matching) != 1 or matching[0].get("run_id") != run_id:
            raise ValueError("conflicting existing CMI-v4 access event")
        return
    assigned = contract["pre_h"]["master_splits"][split]
    prior = [
        event["world_identity"]
        for event in events
        if event.get("event_type") == "ACCESS" and event.get("split") == split
    ]
    if prior != assigned[: len(prior)] or identity != assigned[len(prior)]:
        raise ValueError("refusing out-of-order CMI-v4 access")
    artifact = {
        "schema_version": "conditional-history-mc-q-world-access-v4",
        "run_id": run_id,
        "split": split,
        "world_identity": identity,
    }
    artifact_hash = object_hash(artifact)
    contract["content_registry"][artifact_hash] = {"inline": artifact}
    event = {
        "sequence": int(events[-1]["sequence"]) + 1,
        "event_id": f"access-cmi_v4_dev-{len(prior):04d}",
        "event_type": "ACCESS",
        "split": split,
        "world_identity": identity,
        "run_id": run_id,
        "access_artifact_hash": artifact_hash,
        "previous_event_hash": contract["access_registry"]["head_event_hash"],
    }
    event["event_hash"] = access_event_hash(event)
    events.append(event)
    contract["access_registry"]["event_count"] = len(events)
    contract["access_registry"]["head_event_hash"] = event["event_hash"]
    atomic_json(contract_path, contract)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("contract", type=Path)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    contract_path = args.contract.resolve()
    contract = json.loads(contract_path.read_text())
    receipt = json.loads(args.receipt.resolve().read_text()) if args.receipt else None
    result = validate_contract(contract, contract_path.parent, receipt)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["CMI_V4_DEV_READY"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
