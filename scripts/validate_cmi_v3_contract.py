#!/usr/bin/env python3
"""Validate the independently frozen CMI-v3 contract and ordered access prefix."""
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


SCHEMA = "conditional-history-mc-q-contract-v3"
PRE_H_SCHEMA = "conditional-history-mc-q-preaccess-v3"
SPLIT_SIZES = {"SMOKE_DEBUG": 2, "CMI_V3_DEV": 96, "CMI_V3_CONFIRM": 128}
SEED_RANGES = {
    "SMOKE_DEBUG": {"first": 1_135_000_001, "last": 1_135_000_002},
    "CMI_V3_DEV": {"first": 1_135_000_003, "last": 1_135_000_098},
    "CMI_V3_CONFIRM": {"first": 1_135_000_099, "last": 1_135_000_226},
}
LIVE_SOURCE_PATHS = {
    "protocol": ROOT / "docs/CONDITIONAL_HISTORY_MC_Q_PROTOCOL_20260912_V3.md",
    "freezer": ROOT / "scripts/freeze_cmi_v3.py",
    "environment": ROOT / "experiments/directional_navigation/conditional_model_access.py",
    "collector_v3": ROOT / "scripts/run_cmi_v3_collection.py",
    "collector_common": ROOT / "scripts/run_conditional_history_collection.py",
    "validator_v3": ROOT / "scripts/validate_cmi_v3_contract.py",
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


def freeze_only_contract(contract: dict[str, Any]) -> dict[str, Any]:
    frozen = copy.deepcopy(contract)
    events = frozen["access_registry"]["events"]
    if len(events) < 1 or events[0].get("event_type") != "FREEZE":
        raise ValueError("CMI-v3 registry has no genesis freeze")
    frozen["access_registry"]["events"] = [copy.deepcopy(events[0])]
    seal_access_registry(frozen)
    return frozen


def expected_pre_h_without_hash() -> dict[str, Any]:
    return {
        "schema_version": PRE_H_SCHEMA,
        "contract_id": "conditional-history-mc-q-20260912-v3",
        "superseded_zero_access_freeze_hash": None,
        "parent_v2_contract_hash": None,
        "parent_v2_dev_analysis_hash": None,
        "parent_v2_adequacy_hash": None,
        "protocol_hash": None,
        "master_splits": None,
        "world_manifest_hashes": None,
        "world_seed_ranges": SEED_RANGES,
        "anchor_steps": [256, 768],
        "history_length": 64,
        "history_stride": 4,
        "branch_guard": 12_000,
        "disturbance_law": {
            "enabled_during_history": False,
            "enabled_only_after_anchor": True,
            "kind": "POST_FILTER_NORMALIZED_ACTION_AR1",
            "sigma": 0.04,
            "rho": 0.95,
            "clip_sigma": 3.0,
            "policy_step_seconds": 0.2,
            "common_random_numbers_across_R_C": True,
        },
        "resamples_per_anchor_action": {
            "SMOKE_DEBUG": 1,
            "CMI_V3_DEV": 64,
            "CMI_V3_CONFIRM": 64,
        },
        "utility": {
            "formula": "task_increment - 2*operational_failure - 0.25*I[unified_collision_count>0]",
            "collision_counting": "ONE_UNIFIED_COUNT_PER_POLICY_STEP",
            "contact_terminates": False,
        },
        "legal_information_sets": {
            "latest": "final legal frame only",
            "full": "ordered 64-frame legal history ending in latest",
            "allowed": [
                "LiDAR", "known_charger_relative_observation",
                "distance_to_charger_derived_from_known_charger_relative_observation",
                "battery", "task_progress", "task_clock", "policy_step",
                "legal_action_response_telemetry", "unified_previous_contact",
            ],
            "forbidden": [
                "obstacle_centers", "obstacle_radii", "full_map", "world_seed",
                "snapshot", "disturbance_seed", "future_outcomes",
            ],
        },
        "feature_rule": {
            "lidar_pool": "8_VERTICAL_X_8_HORIZONTAL_MEAN_MIN_HIT_FRACTION",
            "latest_dimension": 416,
            "full_dimension": 2912,
            "history_summaries": [
                "LATEST", "MEAN", "STD", "MIN", "MAX",
                "ENDPOINT_DIFFERENCE", "LATE16_MINUS_EARLY16_MEAN",
            ],
            "simulator_geometry_features": False,
        },
        "model_rule": {
            "families": ["LINEAR_RIDGE", "RFF_RIDGE"],
            "outer_world_folds": 6,
            "inner_world_folds": 4,
            "ridge_grid": [0.1, 1.0, 10.0, 100.0],
            "rff_dimension": 256,
            "rff_seeds": [2_026_091_221, 2_026_091_222, 2_026_091_223],
            "family_ensemble_weights": "UNIFORM_FIXED",
            "action_order": ["R", "C"],
            "exact_tie_action": "R",
            "constant_q_fit": "OUTER_TRAIN_WORLDS_ONLY",
            "v1_v2_outcomes_forbidden": True,
        },
        "adequacy_rule": {
            "all_checks_required": True,
            "bootstrap": {
                "unit": "PHYSICAL_WORLD", "replicates": 20_000,
                "seed": 2_026_091_224, "simultaneous_confidence": 0.95,
                "method": "MULTIPLIER_MAX_T", "repeated_looks": False,
            },
            "minimum_relative_rmse_reduction": 0.05,
            "strict_simultaneous_mse_improvement_lcb_above": 0.0,
            "mc_se_median_max": 0.10,
            "mc_se_p95_max": 0.20,
            "minimum_contributing_worlds": 90,
            "minimum_anchors_per_step": 85,
            "maximum_family_rmse_spread": 0.05,
            "maximum_family_action_disagreement": 0.20,
            "minimum_heldout_coverage_90": 0.85,
            "model_mean_halfwidth_not_above_constant": True,
            "support_pca_components_max": 32,
            "support_minimum_positive_singular_values": 8,
            "support_radius_quantile": 0.99,
            "support_radius_multiplier": 1.15,
            "failure_status": "INCONCLUSIVE_ESTIMATOR_INADEQUATE",
        },
        "confirmation_rule": {
            "requires_all_dev_adequacy_checks": True,
            "requires_hash_linked_child": True,
            "requires_new_external_receipt": True,
            "support_minimum_overall_fraction": 0.95,
            "support_minimum_each_anchor_step_fraction": 0.90,
            "bootstrap_replicates": 20_000,
            "bootstrap_seed": 2_026_091_225,
            "delta_history": 0.05,
            "epsilon_history": 0.025,
            "method_train_authorized": False,
        },
        "source_hashes": None,
        "anchor_service_contract_hash": None,
        "anchor_verification_policy_hash": None,
    }


def validate_contract(
    contract: dict[str, Any], base_dir: Path, receipt: dict[str, Any] | None = None
) -> dict[str, Any]:
    errors: list[str] = []
    registered, content_errors = validate_content_registry(contract, base_dir)
    events, registry_errors = validate_registry(contract, registered)
    errors.extend(content_errors)
    errors.extend(registry_errors)
    if contract.get("schema_version") != SCHEMA:
        errors.append("invalid CMI-v3 contract schema")
    if contract.get("status") != "FROZEN_CMI_V3_DEV":
        errors.append("CMI-v3 contract is not frozen for DEV")
    pre = contract.get("pre_h")
    template = expected_pre_h_without_hash()
    required = set(template) | {"record_sha256"}
    if not isinstance(pre, dict) or set(pre) != required:
        errors.append("CMI-v3 preaccess record has an invalid exact schema")
        pre = {}
    else:
        if pre.get("record_sha256") != record_hash(pre):
            errors.append("CMI-v3 preaccess record hash mismatch")
        for key, expected in template.items():
            if expected is not None and pre.get(key) != expected:
                errors.append(f"CMI-v3 frozen field changed: {key}")
        if pre.get("protocol_hash") != pre.get("source_hashes", {}).get("protocol"):
            errors.append("CMI-v3 protocol/source binding changed")

        supersession_hash = pre.get("superseded_zero_access_freeze_hash")
        supersession_entry = contract.get("content_registry", {}).get(supersession_hash, {})
        supersession = supersession_entry.get("inline") if isinstance(supersession_entry, dict) else None
        if (
            not isinstance(supersession, dict)
            or set(supersession) != {
                "schema_version", "superseded_preaccess_record_hash",
                "superseded_access_head", "formal_access_events",
                "smoke_outcome_worlds_accessed", "reason", "preserved_file_hashes",
            }
            or supersession.get("schema_version") != "cmi-v3-zero-access-supersession-v1"
            or supersession.get("superseded_preaccess_record_hash")
            != "sha256:abf640c250b14f6c5fb076186e0e7229d2dea31506a6ebc9596192acf9bd8b35"
            or supersession.get("superseded_access_head")
            != "sha256:103a793400d42d25b392515c0c0e0eb92421c87ece8327576c98b75bec10a9e7"
            or supersession.get("formal_access_events") != 0
            or supersession.get("smoke_outcome_worlds_accessed") != 0
            or supersession.get("reason")
            != "wrapper import failed before argument parsing because repository root was absent from sys.path"
        ):
            errors.append("CMI-v3 invalid1 supersession boundary is not content-backed")
        elif (
            not isinstance(supersession.get("preserved_file_hashes"), dict)
            or set(supersession["preserved_file_hashes"]) != {"contract", "freeze_receipt", "invalid_record"}
            or any(digest not in registered for digest in supersession["preserved_file_hashes"].values())
        ):
            errors.append("CMI-v3 invalid1 preserved files are incomplete")

        source_hashes = pre.get("source_hashes")
        if not isinstance(source_hashes, dict) or set(source_hashes) != set(LIVE_SOURCE_PATHS):
            errors.append("CMI-v3 source-hash set is incomplete")
            source_hashes = {}
        for name, source in LIVE_SOURCE_PATHS.items():
            digest = source_hashes.get(name)
            entry = contract.get("content_registry", {}).get(digest, {})
            if not isinstance(entry, dict) or set(entry) != {"path"}:
                errors.append(f"CMI-v3 frozen source is not path-backed: {name}")
            elif file_hash(base_dir / entry["path"]) != digest:
                errors.append(f"CMI-v3 frozen source hash mismatch: {name}")
            if source.exists() and digest is not None and file_hash(source) != digest:
                errors.append(f"live CMI-v3 source differs from frozen source: {name}")

        splits = pre.get("master_splits")
        manifests = pre.get("world_manifest_hashes")
        if not isinstance(splits, dict) or set(splits) != set(SPLIT_SIZES):
            errors.append("CMI-v3 split set is invalid")
            splits = {}
        if not isinstance(manifests, dict) or set(manifests) != set(SPLIT_SIZES):
            errors.append("CMI-v3 manifest hash set is invalid")
            manifests = {}
        identities: list[str] = []
        seeds: list[int] = []
        for split, size in SPLIT_SIZES.items():
            assigned = splits.get(split)
            if not isinstance(assigned, list) or len(assigned) != size:
                errors.append(f"CMI-v3 {split} size differs from {size}")
                continue
            identities.extend(assigned)
            entry = contract.get("content_registry", {}).get(manifests.get(split), {})
            rows = entry.get("inline") if isinstance(entry, dict) else None
            expected_seeds = list(range(SEED_RANGES[split]["first"], SEED_RANGES[split]["last"] + 1))
            if not isinstance(rows, list) or [row.get("world_identity") for row in rows] != assigned:
                errors.append(f"CMI-v3 {split} manifest is not ordered/content-backed")
                continue
            if [row.get("world_seed") for row in rows] != expected_seeds:
                errors.append(f"CMI-v3 {split} seed allocation changed")
            for row in rows:
                identity, seed = row["world_identity"], row["world_seed"]
                seeds.append(seed)
                path = base_dir / "worlds" / f"{identity.removeprefix('sha256:')}.json"
                try:
                    world = json.loads(path.read_text())
                except (OSError, json.JSONDecodeError):
                    errors.append(f"CMI-v3 world record unreadable: {identity}")
                    continue
                if object_hash(world) != identity or world.get("world_seed") != seed:
                    errors.append(f"CMI-v3 world identity/seed mismatch: {identity}")
                world_entry = contract.get("world_registry", {}).get(identity)
                if not isinstance(world_entry, dict) or world_entry.get("source_split") != split:
                    errors.append(f"CMI-v3 world registry mismatch: {identity}")
        if len(identities) != len(set(identities)):
            errors.append("CMI-v3 physical-world splits overlap")
        if set(contract.get("world_registry", {})) != set(identities):
            errors.append("CMI-v3 world registry differs from split union")

        v2_root = ROOT / "artifacts/conditional_history_information_20260912_v2"
        try:
            v2_contract = json.loads((v2_root / "contract.json").read_text())
        except (OSError, json.JSONDecodeError):
            errors.append("CMI-v2 provenance contract unavailable")
        else:
            if pre.get("parent_v2_contract_hash") != file_hash(v2_root / "contract.json"):
                errors.append("CMI-v3 parent v2 contract hash changed")
            if pre.get("parent_v2_dev_analysis_hash") != file_hash(v2_root / "cmi_dev_analysis/analysis.json"):
                errors.append("CMI-v3 parent v2 analysis hash changed")
            if pre.get("parent_v2_adequacy_hash") != file_hash(v2_root / "cmi_dev_analysis/adequacy.json"):
                errors.append("CMI-v3 parent v2 adequacy hash changed")
            old = set(v2_contract.get("world_registry", {}))
            if old.intersection(identities):
                errors.append("CMI-v3 worlds overlap CMI-v2 worlds")
            old_accesses = [
                event for event in v2_contract.get("access_registry", {}).get("events", [])
                if event.get("event_type") == "ACCESS"
            ]
            if len([event for event in old_accesses if event.get("split") == "CMI_DEV"]) != 64:
                errors.append("CMI-v2 provenance boundary is not the audited terminal")
            if any(event.get("split") == "CMI_CONFIRM" for event in old_accesses):
                errors.append("CMI-v2 confirmation provenance unexpectedly opened")

    if events:
        first = events[0]
        if (
            first.get("event_type") != "FREEZE"
            or first.get("stage") != "CMI_V3_DEV"
            or first.get("record_hash") != pre.get("record_sha256")
        ):
            errors.append("CMI-v3 access chain has an invalid freeze genesis")
    accesses = [event for event in events if event.get("event_type") == "ACCESS"]
    for split in ("CMI_V3_DEV", "CMI_V3_CONFIRM"):
        observed = [event.get("world_identity") for event in accesses if event.get("split") == split]
        assigned = pre.get("master_splits", {}).get(split, [])
        if observed != assigned[:len(observed)] or len(observed) != len(set(observed)):
            errors.append(f"CMI-v3 {split} access is not an ordered unique prefix")
    if any(event.get("split") == "CMI_V3_CONFIRM" for event in accesses):
        errors.append("CMI-v3 confirmation access is forbidden in the DEV parent contract")

    receipt_errors: list[str] = []
    if receipt is None:
        receipt_errors.append("CMI-v3 freeze-time external receipt is required")
    else:
        try:
            frozen = freeze_only_contract(contract)
        except ValueError as error:
            receipt_errors.append(str(error))
        else:
            receipt_errors.extend(validate_access_anchor(frozen, receipt, registered))
    return {
        "schema_version": "conditional-history-mc-q-validation-v3",
        "valid": not errors,
        "errors": errors,
        "receipt_errors": receipt_errors,
        "CMI_V3_DEV_READY": not errors and not receipt_errors,
        "CMI_V3_CONFIRM_READY": False,
        "cmi_v3_dev_accessed": len([e for e in accesses if e.get("split") == "CMI_V3_DEV"]),
        "cmi_v3_confirm_accessed": len([e for e in accesses if e.get("split") == "CMI_V3_CONFIRM"]),
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
            raise ValueError("conflicting existing CMI-v3 access event")
        return
    assigned = contract["pre_h"]["master_splits"][split]
    prior = [
        event["world_identity"] for event in events
        if event.get("event_type") == "ACCESS" and event.get("split") == split
    ]
    if prior != assigned[:len(prior)] or identity != assigned[len(prior)]:
        raise ValueError("refusing out-of-order CMI-v3 access")
    artifact = {
        "schema_version": "conditional-history-mc-q-world-access-v3",
        "run_id": run_id, "split": split, "world_identity": identity,
    }
    artifact_hash = object_hash(artifact)
    contract["content_registry"][artifact_hash] = {"inline": artifact}
    event = {
        "sequence": int(events[-1]["sequence"]) + 1,
        "event_id": f"access-{split.lower()}-{len(prior):04d}",
        "event_type": "ACCESS", "split": split,
        "world_identity": identity, "run_id": run_id,
        "access_artifact_hash": artifact_hash,
        "previous_event_hash": contract["access_registry"]["head_event_hash"],
    }
    event["event_hash"] = access_event_hash(event)
    events.append(event)
    contract["access_registry"]["event_count"] = len(events)
    contract["access_registry"]["head_event_hash"] = event["event_hash"]
    atomic_json(contract_path, contract)


def confirmation_freeze_only(contract: dict[str, Any]) -> dict[str, Any]:
    frozen = copy.deepcopy(contract)
    events = frozen["access_registry"]["events"]
    kept = [
        copy.deepcopy(event) for event in events
        if not (
            event.get("event_type") == "ACCESS"
            and event.get("split") == "CMI_V3_CONFIRM"
        )
    ]
    frozen["access_registry"]["events"] = kept
    seal_access_registry(frozen)
    return frozen


def validate_confirmation_contract(
    contract: dict[str, Any],
    base_dir: Path,
    dev_analysis: Path,
    receipt: dict[str, Any] | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    registered, content_errors = validate_content_registry(contract, base_dir)
    events, registry_errors = validate_registry(contract, registered)
    errors.extend(content_errors)
    errors.extend(registry_errors)
    if contract.get("schema_version") != SCHEMA or contract.get("status") != "FROZEN_CMI_V3_CONFIRM":
        errors.append("invalid CMI-v3 confirmation child status/schema")
    pre = contract.get("pre_h", {})
    if pre.get("record_sha256") != record_hash(pre):
        errors.append("CMI-v3 confirmation inherited preaccess hash mismatch")
    for name, source in LIVE_SOURCE_PATHS.items():
        digest = pre.get("source_hashes", {}).get(name)
        entry = contract.get("content_registry", {}).get(digest, {})
        if not isinstance(entry, dict) or set(entry) != {"path"}:
            errors.append(f"CMI-v3 confirmation source is not frozen: {name}")
        elif file_hash(base_dir / entry["path"]) != digest:
            errors.append(f"CMI-v3 confirmation source hash mismatch: {name}")
        if source.exists() and digest is not None and file_hash(source) != digest:
            errors.append(f"live CMI-v3 confirmation source changed: {name}")
    child = contract.get("confirmation_child")
    required = {
        "schema_version", "parent_preaccess_hash", "parent_dev_access_head",
        "dev_analysis_hash", "dev_artifact_hashes", "confirm_world_manifest_hash",
        "adequacy_passed", "method_train_authorized", "record_sha256",
    }
    if not isinstance(child, dict) or set(child) != required:
        errors.append("CMI-v3 confirmation child has invalid exact schema")
        child = {}
    else:
        if child.get("schema_version") != "conditional-history-mc-q-confirm-child-v3":
            errors.append("invalid CMI-v3 confirmation child schema")
        if child.get("record_sha256") != record_hash(child):
            errors.append("CMI-v3 confirmation child hash mismatch")
        if child.get("parent_preaccess_hash") != pre.get("record_sha256"):
            errors.append("CMI-v3 confirmation parent preaccess binding changed")
        if child.get("confirm_world_manifest_hash") != pre.get("world_manifest_hashes", {}).get("CMI_V3_CONFIRM"):
            errors.append("CMI-v3 confirmation world manifest binding changed")
        if child.get("adequacy_passed") is not True or child.get("method_train_authorized") is not False:
            errors.append("CMI-v3 confirmation child has invalid gate/training semantics")
        try:
            adequacy = json.loads((dev_analysis / "adequacy.json").read_text())
            analysis = json.loads((dev_analysis / "analysis.json").read_text())
        except (OSError, json.JSONDecodeError):
            errors.append("CMI-v3 DEV analysis is unavailable")
        else:
            if not adequacy.get("passed") or not analysis.get("adequacy_passed"):
                errors.append("CMI-v3 DEV adequacy did not pass")
            if child.get("dev_analysis_hash") != file_hash(dev_analysis / "analysis.json"):
                errors.append("CMI-v3 DEV analysis hash binding changed")
            hashes = child.get("dev_artifact_hashes")
            if not isinstance(hashes, dict) or hashes != analysis.get("artifact_hashes"):
                errors.append("CMI-v3 DEV artifact hash set changed")
            elif any(file_hash(dev_analysis / name) != digest for name, digest in hashes.items()):
                errors.append("CMI-v3 DEV artifact bytes changed")
    dev_accesses = [e for e in events if e.get("event_type") == "ACCESS" and e.get("split") == "CMI_V3_DEV"]
    confirm_accesses = [e for e in events if e.get("event_type") == "ACCESS" and e.get("split") == "CMI_V3_CONFIRM"]
    confirm_freezes = [e for e in events if e.get("event_type") == "FREEZE" and e.get("stage") == "CMI_V3_CONFIRM"]
    if len(dev_accesses) != SPLIT_SIZES["CMI_V3_DEV"]:
        errors.append("CMI-v3 confirmation child does not inherit complete DEV access")
    if len(confirm_freezes) != 1 or events.index(confirm_freezes[0]) != 1 + len(dev_accesses):
        errors.append("CMI-v3 confirmation freeze is missing or misplaced")
    elif confirm_freezes[0].get("record_hash") != child.get("record_sha256"):
        errors.append("CMI-v3 confirmation freeze record hash changed")
    elif confirm_freezes[0].get("previous_event_hash") != child.get("parent_dev_access_head"):
        errors.append("CMI-v3 confirmation freeze does not bind the DEV head")
    assigned = pre.get("master_splits", {}).get("CMI_V3_CONFIRM", [])
    observed = [e.get("world_identity") for e in confirm_accesses]
    if observed != assigned[:len(observed)] or len(observed) != len(set(observed)):
        errors.append("CMI-v3 confirmation access is not an ordered unique prefix")

    receipt_errors: list[str] = []
    if receipt is None:
        receipt_errors.append("CMI-v3 confirmation external receipt is required")
    else:
        receipt_errors.extend(
            validate_access_anchor(confirmation_freeze_only(contract), receipt, registered)
        )
    return {
        "schema_version": "conditional-history-mc-q-confirm-validation-v3",
        "valid": not errors,
        "errors": errors,
        "receipt_errors": receipt_errors,
        "CMI_V3_CONFIRM_READY": not errors and not receipt_errors,
        "cmi_v3_dev_accessed": len(dev_accesses),
        "cmi_v3_confirm_accessed": len(confirm_accesses),
    }


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
    return 0 if result["CMI_V3_DEV_READY"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
