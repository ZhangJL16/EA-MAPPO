#!/usr/bin/env python3
"""Validate the standalone conditional-history experiment freeze and access prefix."""
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


SCHEMA = "conditional-history-information-contract-v1"
PRE_H_SCHEMA = "conditional-history-preaccess-freeze-v1"
SPLIT_SIZES = {"SMOKE_DEBUG": 2, "CMI_DEV": 64, "CMI_CONFIRM": 96}
ANCHOR_STEPS = [256, 768]
LIVE_SOURCE_PATHS = {
    "protocol": ROOT / "docs/CONDITIONAL_HISTORY_INFORMATION_PROTOCOL_20260912.md",
    "environment": ROOT / "experiments/directional_navigation/conditional_model_access.py",
    "collector": ROOT / "scripts/run_conditional_history_collection.py",
    "validator": ROOT / "scripts/validate_conditional_history_contract.py",
    "models": ROOT / "scripts/conditional_history_models.py",
    "dev_analyzer": ROOT / "scripts/analyze_conditional_history_dev.py",
    "confirmation_analyzer": ROOT / "scripts/analyze_conditional_history_confirmation.py",
    "dvoi_environment": ROOT / "experiments/directional_navigation/dvoi_h_branch.py",
    "frozen_navigator": ROOT / "scripts/run_dvoi_h_collection.py",
    "locked_recovery": ROOT / "experiments/directional_navigation/recovery.py",
    "collision_protocol": ROOT / "docs/LOCKED_COLLISION_RECOVERY_PROTOCOL.md",
}


def freeze_only_contract(contract: dict[str, Any]) -> dict[str, Any]:
    frozen = copy.deepcopy(contract)
    events = frozen["access_registry"]["events"]
    freeze_events = [event for event in events if event.get("event_type") == "FREEZE"]
    if len(freeze_events) != 1 or events[0].get("event_type") != "FREEZE":
        raise ValueError("CMI registry must start with exactly one freeze event")
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
        errors.append("invalid CMI contract schema")
    if contract.get("status") != "FROZEN_CMI_DEV":
        errors.append("CMI contract is not frozen for development")
    pre = contract.get("pre_h")
    required = {
        "schema_version", "contract_id", "supersession_artifact_hash",
        "parent_dvoi_terminal_record_hash",
        "parent_dvoi_result_hash", "protocol_hash", "master_splits",
        "world_manifest_hashes", "world_seed_ranges", "anchor_steps",
        "history_length", "history_stride", "branch_guard", "disturbance_law",
        "resamples", "utility", "legal_information_sets", "model_rule",
        "inference_rule", "decision_rule", "source_hashes",
        "anchor_service_contract_hash", "anchor_verification_policy_hash",
        "record_sha256",
    }
    if not isinstance(pre, dict) or set(pre) != required:
        errors.append("CMI preaccess record does not have the exact frozen schema")
        pre = {}
    else:
        if pre.get("schema_version") != PRE_H_SCHEMA:
            errors.append("invalid CMI preaccess schema version")
        if pre.get("contract_id") != "conditional-history-information-20260912-v2":
            errors.append("CMI contract identity changed")
        supersession_entry = contract.get("content_registry", {}).get(
            pre.get("supersession_artifact_hash"), {}
        )
        supersession = supersession_entry.get("inline") if isinstance(supersession_entry, dict) else None
        if (
            not isinstance(supersession, dict)
            or set(supersession) != {
                "schema_version", "superseded_contract_id",
                "superseded_preaccess_record_hash", "superseded_access_head",
                "formal_access_events", "excluded_smoke_worlds_completed", "reason",
                "preserved_file_hashes",
            }
            or supersession.get("schema_version") != "conditional-history-supersession-v1"
            or supersession.get("superseded_contract_id") != "conditional-history-information-20260912-v1"
            or supersession.get("superseded_preaccess_record_hash")
            != "sha256:cb8d806f2cc3da58aeb96f5cd8f8474cff6d9b07b9f8f6de388bbdc8c3281e6e"
            or supersession.get("superseded_access_head")
            != "sha256:b1370325d300609d3a8a5e7e6fdde1c8ef4931e0ff697bc01523f704e26714cf"
            or supersession.get("formal_access_events") != 0
            or supersession.get("excluded_smoke_worlds_completed") != 1
            or supersession.get("reason")
            != "JSON tuple/list manifest mismatch blocked resume before second smoke-world access"
        ):
            errors.append("CMI v1 supersession boundary is invalid or not content-backed")
        elif (
            not isinstance(supersession.get("preserved_file_hashes"), dict)
            or set(supersession["preserved_file_hashes"]) != {
                "contract", "freeze_receipt", "smoke_manifest", "smoke_latest", "smoke_status"
            }
            or any(value not in registered for value in supersession["preserved_file_hashes"].values())
        ):
            errors.append("CMI v1 supersession files are incomplete or not content-backed")
        if pre.get("record_sha256") != record_hash(pre):
            errors.append("CMI preaccess record hash mismatch")
        source_hashes = pre.get("source_hashes")
        if not isinstance(source_hashes, dict) or set(source_hashes) != set(LIVE_SOURCE_PATHS):
            errors.append("CMI source-hash set is incomplete or changed")
            source_hashes = {}
        if pre.get("protocol_hash") != source_hashes.get("protocol"):
            errors.append("CMI protocol binding changed")
        if pre.get("anchor_steps") != ANCHOR_STEPS:
            errors.append("CMI anchor steps changed")
        if pre.get("history_length") != 64 or pre.get("history_stride") != 4:
            errors.append("CMI legal-history shape changed")
        if pre.get("branch_guard") != 12_000:
            errors.append("CMI branch guard changed")
        if pre.get("resamples") != {
            "SMOKE_DEBUG": {"evaluation": 1, "oracle_selection": 0},
            "CMI_DEV": {"evaluation": 8, "oracle_selection": 0},
            "CMI_CONFIRM": {"evaluation": 16, "oracle_selection": 16},
        }:
            errors.append("CMI resampling counts changed")
        expected_disturbance = {
            "enabled_during_history": False,
            "enabled_only_after_anchor": True,
            "kind": "POST_FILTER_NORMALIZED_ACTION_AR1",
            "sigma": 0.04,
            "rho": 0.95,
            "clip_sigma": 3.0,
            "policy_step_seconds": 0.2,
            "common_random_numbers_across_R_C": True,
        }
        if pre.get("disturbance_law") != expected_disturbance:
            errors.append("CMI disturbance law changed")
        if pre.get("utility") != {
            "formula": "task_increment - 2*operational_failure - 0.25*I[unified_collision_count>0]",
            "collision_counting": "ONE_UNIFIED_COUNT_PER_POLICY_STEP",
            "contact_terminates": False,
        }:
            errors.append("CMI utility/collision rule changed")
        if pre.get("legal_information_sets") != {
            "latest": "final frame only",
            "full": "ordered 64-frame legal history ending in latest",
            "allowed": [
                "LiDAR", "known_charger_relative_observation",
                "distance_to_charger_derived_from_known_charger_relative_observation",
                "battery", "task_progress", "task_clock", "policy_step",
                "legal_action_response_telemetry",
            ],
            "forbidden": [
                "obstacle_centers", "obstacle_radii", "full_map", "world_seed",
                "snapshot", "disturbance_seed", "future_outcomes",
            ],
        }:
            errors.append("CMI legal information-set contract changed")
        if pre.get("model_rule") != {
            "families": ["CENTERED_GRU_RESIDUAL", "CENTERED_TCN_RESIDUAL"],
            "training_seeds": [2_026_091_211, 2_026_091_212, 2_026_091_213],
            "world_folds": 5,
            "ensemble_weights": "UNIFORM_FIXED",
            "action_order": ["R", "C"],
            "exact_tie_action": "R",
            "raw_legal_frame_dimension": 4128,
            "negative_claim_requires_adequacy": True,
        }:
            errors.append("CMI model-selection rule changed")
        if pre.get("inference_rule") != {
            "independent_unit": "PHYSICAL_WORLD",
            "method": "WHOLE_WORLD_MULTIPLIER_BOOTSTRAP_MAX_T",
            "replicates": 20_000,
            "seed": 2_026_091_214,
            "simultaneous_confidence": 0.95,
            "stopping": "FIXED_N_96_WORLDS",
            "repeated_looks": False,
        }:
            errors.append("CMI inference/stopping rule changed")
        if pre.get("decision_rule") != {
            "delta_history": 0.05,
            "epsilon_history": 0.025,
            "delta_privileged": 0.05,
            "branches": [
                "HISTORY_INFORMATION_PRESENT", "OBSERVATION_OR_BENCHMARK_LIMITED",
                "TERMINATE_BENCHMARK_PROBING", "INCONCLUSIVE",
            ],
            "method_train_authorized": False,
        }:
            errors.append("CMI decision table changed")
        splits = pre.get("master_splits")
        manifests = pre.get("world_manifest_hashes")
        if not isinstance(splits, dict) or set(splits) != set(SPLIT_SIZES):
            errors.append("CMI master splits are invalid")
            splits = {}
        if not isinstance(manifests, dict) or set(manifests) != set(SPLIT_SIZES):
            errors.append("CMI manifest hashes are invalid")
            manifests = {}
        expected_ranges = {
            "SMOKE_DEBUG": {"first": 1_134_000_001, "last": 1_134_000_002},
            "CMI_DEV": {"first": 1_134_000_003, "last": 1_134_000_066},
            "CMI_CONFIRM": {"first": 1_134_000_067, "last": 1_134_000_162},
        }
        if pre.get("world_seed_ranges") != expected_ranges:
            errors.append("CMI world seed ranges changed")
        identities: list[str] = []
        observed_seeds: list[int] = []
        for split, size in SPLIT_SIZES.items():
            assigned = splits.get(split)
            if not isinstance(assigned, list) or len(assigned) != size:
                errors.append(f"CMI {split} size differs from {size}")
                continue
            identities.extend(assigned)
            manifest = contract.get("content_registry", {}).get(manifests.get(split), {})
            rows = manifest.get("inline") if isinstance(manifest, dict) else None
            if not isinstance(rows, list) or [row.get("world_identity") for row in rows] != assigned:
                errors.append(f"CMI {split} manifest is not content-backed or ordered")
                continue
            expected_seeds = list(range(expected_ranges[split]["first"], expected_ranges[split]["last"] + 1))
            if [row.get("world_seed") for row in rows] != expected_seeds:
                errors.append(f"CMI {split} seed sequence differs from the frozen allocation")
            for row in rows:
                path = base_dir / "worlds" / f"{row['world_identity'].removeprefix('sha256:')}.json"
                try:
                    world = json.loads(path.read_text())
                except (OSError, json.JSONDecodeError):
                    errors.append(f"CMI world record is unreadable: {path.name}")
                    continue
                if object_hash(world) != row["world_identity"] or row.get("world_seed") != world.get("world_seed"):
                    errors.append(f"CMI world identity/seed mismatch: {path.name}")
                    continue
                observed_seeds.append(int(row["world_seed"]))
                world_entry = contract.get("world_registry", {}).get(row["world_identity"])
                if not isinstance(world_entry, dict) or set(world_entry) != {
                    "physical_world_hash", "world_seed_or_parameters_hash", "source_split"
                }:
                    errors.append(f"CMI world registry entry is invalid: {path.name}")
                    continue
                seed_entry = contract.get("content_registry", {}).get(
                    world_entry["world_seed_or_parameters_hash"], {}
                )
                if (
                    world_entry["physical_world_hash"] != row["world_identity"]
                    or world_entry["source_split"] != split
                    or not isinstance(seed_entry, dict)
                    or seed_entry.get("inline") != {
                        "schema_version": "dvoi-world-seed-v1", "world_seed": row["world_seed"]
                    }
                ):
                    errors.append(f"CMI world registry binding is invalid: {path.name}")
        if len(identities) != len(set(identities)):
            errors.append("CMI physical-world splits overlap")
        if isinstance(contract.get("world_registry"), dict) and set(contract["world_registry"]) != set(identities):
            errors.append("CMI world registry keys differ from the frozen split union")
        for name, digest in source_hashes.items():
            entry = contract.get("content_registry", {}).get(digest)
            if not isinstance(entry, dict) or set(entry) != {"path"}:
                errors.append(f"CMI source {name} is not path-backed")
            elif file_hash(base_dir / entry["path"]) != digest:
                errors.append(f"CMI source {name} hash mismatch")
            if name in LIVE_SOURCE_PATHS and file_hash(LIVE_SOURCE_PATHS[name]) != digest:
                errors.append(f"live CMI source {name} differs from its frozen copy")
        parent_path = ROOT / "artifacts/dvoi_h_gate_20260911_v3/identification_contract.json"
        try:
            parent = json.loads(parent_path.read_text())
        except (OSError, json.JSONDecodeError):
            errors.append("parent DVOI terminal contract is unavailable")
        else:
            if pre.get("parent_dvoi_terminal_record_hash") != parent.get("p_terminal", {}).get("record_sha256"):
                errors.append("CMI parent terminal binding is invalid")
            if pre.get("parent_dvoi_result_hash") != parent.get("p_terminal", {}).get("p_confirmation_result_artifact_hash"):
                errors.append("CMI parent P-result binding is invalid")
            old_seeds = {
                int(entry["inline"]["world_seed"])
                for entry in parent.get("content_registry", {}).values()
                if isinstance(entry, dict)
                and isinstance(entry.get("inline"), dict)
                and set(entry["inline"]) == {"schema_version", "world_seed"}
            }
            if old_seeds.intersection(observed_seeds):
                errors.append("CMI world seeds overlap the parent DVOI registry")

    if events:
        freeze = events[0]
        if (
            freeze.get("event_type") != "FREEZE"
            or freeze.get("stage") != "CMI_DEV"
            or freeze.get("record_hash") != pre.get("record_sha256")
        ):
            errors.append("CMI access chain does not begin at the frozen preaccess record")
    accesses = [event for event in events if event.get("event_type") == "ACCESS"]
    if any(event.get("split") == "SMOKE_DEBUG" for event in accesses):
        errors.append("SMOKE_DEBUG must remain outside the formal access chain")
    for split in ("CMI_DEV", "CMI_CONFIRM"):
        observed = [event.get("world_identity") for event in accesses if event.get("split") == split]
        assigned = pre.get("master_splits", {}).get(split, [])
        if observed != assigned[: len(observed)] or len(observed) != len(set(observed)):
            errors.append(f"CMI {split} access is not an ordered unique prefix")
    if any(event.get("split") == "CMI_CONFIRM" for event in accesses):
        errors.append("CMI_CONFIRM access is forbidden before a confirmation child is frozen")

    receipt_errors: list[str] = []
    if receipt is None:
        receipt_errors.append("CMI freeze-time external receipt is required")
    else:
        try:
            frozen = freeze_only_contract(contract)
        except ValueError as error:
            receipt_errors.append(str(error))
        else:
            receipt_errors.extend(validate_access_anchor(frozen, receipt, registered))
    return {
        "schema_version": "conditional-history-contract-validation-v1",
        "valid": not errors,
        "errors": errors,
        "receipt_errors": receipt_errors,
        "CMI_DEV_READY": not errors and not receipt_errors,
        "CMI_CONFIRM_READY": False,
        "cmi_dev_accessed": len([event for event in accesses if event.get("split") == "CMI_DEV"]),
        "cmi_confirm_accessed": len([event for event in accesses if event.get("split") == "CMI_CONFIRM"]),
    }


def append_access_event(contract_path: Path, split: str, identity: str, run_id: str) -> None:
    from scripts.freeze_dvoi_pre_h import atomic_json

    contract = json.loads(contract_path.read_text())
    events = contract["access_registry"]["events"]
    matching = [
        event for event in events
        if event.get("event_type") == "ACCESS"
        and event.get("split") == split
        and event.get("world_identity") == identity
    ]
    if matching:
        if len(matching) != 1 or matching[0].get("run_id") != run_id:
            raise ValueError("existing CMI access event conflicts with this run")
        return
    assigned = contract["pre_h"]["master_splits"][split]
    prior = [event["world_identity"] for event in events if event.get("event_type") == "ACCESS" and event.get("split") == split]
    if prior != assigned[: len(prior)] or identity != assigned[len(prior)]:
        raise ValueError("refusing out-of-order CMI access")
    artifact = {
        "schema_version": "conditional-history-world-access-v1",
        "run_id": run_id,
        "split": split,
        "world_identity": identity,
    }
    artifact_hash = object_hash(artifact)
    contract["content_registry"][artifact_hash] = {"inline": artifact}
    event = {
        "sequence": int(events[-1]["sequence"]) + 1,
        "event_id": f"access-{split.lower()}-{len(prior):04d}",
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
    receipt = json.loads(args.receipt.read_text()) if args.receipt else None
    result = validate_contract(contract, contract_path.parent, receipt)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["CMI_DEV_READY"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
