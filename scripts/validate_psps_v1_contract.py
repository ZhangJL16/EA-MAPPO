#!/usr/bin/env python3
"""Validate the independent PSPS-v1 DEV contract and ordered access chain."""
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
    access_event_hash, file_hash, object_hash, record_hash, seal_access_registry,
    validate_access_anchor, validate_content_registry, validate_registry,
)

SCHEMA = "psps-v1-contract-v1"
PRE_SCHEMA = "psps-v1-preaccess-v1"
SPLIT_SIZES = {"SMOKE_DEBUG": 2, "PSPS_DEV": 96, "PSPS_CONFIRM": 128}
SEED_RANGES = {
    "SMOKE_DEBUG": {"first": 1_137_000_001, "last": 1_137_000_002},
    "PSPS_DEV": {"first": 1_137_000_003, "last": 1_137_000_098},
    "PSPS_CONFIRM": {"first": 1_137_000_099, "last": 1_137_000_226},
}
LIVE_SOURCE_PATHS = {
    "protocol": ROOT / "docs/PSPS_V1_PROSPECTIVE_PROTOCOL_20260915.md",
    "features": ROOT / "scripts/psps_v1_features.py",
    "freezer": ROOT / "scripts/freeze_psps_v1.py",
    "validator": ROOT / "scripts/validate_psps_v1_contract.py",
    "collector": ROOT / "scripts/run_psps_v1_collection.py",
    "dev_analyzer": ROOT / "scripts/analyze_psps_v1_dev.py",
    "confirmation_freezer": ROOT / "scripts/freeze_psps_v1_confirmation.py",
    "collector_common": ROOT / "scripts/run_conditional_history_collection.py",
    "coordinator_reference": ROOT / "scripts/run_cmi_v4_parallel_collection.py",
    "environment": ROOT / "experiments/directional_navigation/conditional_model_access.py",
    "dvoi_environment": ROOT / "experiments/directional_navigation/dvoi_h_branch.py",
    "frozen_navigator": ROOT / "scripts/run_dvoi_h_collection.py",
    "locked_recovery": ROOT / "experiments/directional_navigation/recovery.py",
    "collision_protocol": ROOT / "docs/LOCKED_COLLISION_RECOVERY_PROTOCOL.md",
}


def static_spec() -> dict[str, Any]:
    return {
        "contract_id": "progress-stall-predictive-state-20260915-v1",
        "world_seed_ranges": SEED_RANGES,
        "anchor_steps": [256, 768], "history_length": 64, "history_stride": 4,
        "branch_guard": 12_000,
        "resamples_per_anchor_action": {"SMOKE_DEBUG": 1, "PSPS_DEV": 64, "PSPS_CONFIRM": 64},
        "action_order": ["R", "C"], "replicate_halves": "PARITY_32_32_SWAP_AVERAGE",
        "disturbance": {"kind": "POST_FILTER_NORMALIZED_ACTION_AR1", "sigma": 0.04, "rho": 0.95, "clip_sigma": 3.0, "common_random_numbers_across_R_C": True},
        "utility": "task_increment - 2*operational_failure - 0.25*I[unified_collision_count>0]",
        "gate_order": ["O", "D"], "strict_failure_stops": True,
        "gate_o_target": "I[C_TASK_INCREMENT_AT_LEAST_ONE_BEFORE_BRANCH_GUARD]",
        "feature_channels": {"latest_dimension": 10, "psps_dimension": 20, "generic_history_encoder": False, "simulator_geometry": False},
        "estimator": {"family": "BINOMIAL_LOGISTIC_RIDGE", "outer_world_folds": 6, "inner_world_folds": 4, "alpha_grid": [0.1, 1.0, 10.0, 100.0], "selection_loss": "BINARY_LOG_LOSS", "probability_clip": [1e-6, 0.999999]},
        "gate_o": {"primary_contrasts": ["LATEST_BRIER_MINUS_PSPS", "CONSTANT_BRIER_MINUS_PSPS"], "simultaneous_lcb_strictly_above": 0.0, "minimum_relative_brier_reduction": 0.05, "completion_prevalence_range_each_discovery_half": [0.05, 0.95], "minimum_worlds": 90, "minimum_anchors_each_step": 85, "minimum_positive_singular_values": 8},
        "gate_d": {"interpreted_only_if_gate_o_passes": True, "threshold_grid": [i / 20 for i in range(1, 20)], "threshold_tie": "LARGER_MORE_CONSERVATIVE", "primary_contrasts": ["PSPS_MINUS_LATEST_UTILITY", "PSPS_MINUS_BEST_CONSTANT_UTILITY"], "simultaneous_lcb_strictly_above": 0.0, "minimum_point_gain": 0.05, "minimum_psps_latest_disagreement": 0.05, "minimum_each_action_fraction": 0.05, "minimum_each_action_worlds": 5},
        "bootstrap": {"method": "WHOLE_WORLD_MAX_T", "draws": 20_000, "gate_o_seed": 2_026_091_511, "gate_d_seed": 2_026_091_512, "confidence": 0.95, "repeated_looks": False},
        "confirmation": {"requires_gate_o_and_gate_d": True, "requires_hash_linked_child": True, "requires_external_anchor": True, "refit": False, "support_overall_min": 0.95, "support_each_step_min": 0.90, "support_pca_components_max": 20, "support_radius_quantile": 0.99, "support_radius_multiplier": 1.15},
        "old_pai_confirm_access_forbidden_forever": True,
        "automatic_analysis": False, "automatic_confirmation": False,
        "method_train_authorized": False,
        "execution": {"world_processes": 4, "workers_per_world": 8, "max_worlds_per_child": 1, "ordered_single_writer": True, "resumable": True},
    }


def freeze_only_contract(contract: dict[str, Any]) -> dict[str, Any]:
    frozen = copy.deepcopy(contract)
    events = frozen["access_registry"]["events"]
    if not events or events[0].get("event_type") != "FREEZE":
        raise ValueError("PSPS-v1 has no freeze genesis")
    frozen["access_registry"]["events"] = [copy.deepcopy(events[0])]
    seal_access_registry(frozen)
    return frozen


def validate_contract(contract: dict[str, Any], base_dir: Path, receipt: dict[str, Any] | None = None) -> dict[str, Any]:
    errors: list[str] = []
    registered, content_errors = validate_content_registry(contract, base_dir)
    events, registry_errors = validate_registry(contract, registered)
    errors.extend(content_errors + registry_errors)
    if contract.get("schema_version") != SCHEMA or contract.get("status") != "FROZEN_PSPS_V1_DEV":
        errors.append("PSPS-v1 schema/status is not frozen DEV")
    pre = contract.get("pre_h", {})
    required = {"schema_version", "record_sha256", "static_spec", "master_splits", "world_manifest_hashes", "source_hashes", "old_pai_boundary", "anchor_service_contract_hash", "anchor_verification_policy_hash"}
    if not isinstance(pre, dict) or set(pre) != required:
        errors.append("PSPS-v1 preaccess exact schema changed")
    else:
        if pre.get("schema_version") != PRE_SCHEMA or pre.get("record_sha256") != record_hash(pre):
            errors.append("PSPS-v1 preaccess hash/schema mismatch")
        if pre.get("static_spec") != static_spec():
            errors.append("PSPS-v1 prospective specification changed")
        hashes = pre.get("source_hashes", {})
        if set(hashes) != set(LIVE_SOURCE_PATHS):
            errors.append("PSPS-v1 source set changed")
        for name, live in LIVE_SOURCE_PATHS.items():
            digest = hashes.get(name)
            entry = contract.get("content_registry", {}).get(digest, {})
            if not isinstance(entry, dict) or set(entry) != {"path"} or file_hash(base_dir / entry.get("path", "missing")) != digest:
                errors.append(f"PSPS-v1 frozen source invalid: {name}")
            if live.exists() and digest and file_hash(live) != digest:
                errors.append(f"PSPS-v1 live source differs: {name}")
        splits, manifests = pre.get("master_splits", {}), pre.get("world_manifest_hashes", {})
        identities = []
        for split, size in SPLIT_SIZES.items():
            assigned = splits.get(split, [])
            entry = contract.get("content_registry", {}).get(manifests.get(split), {})
            rows = entry.get("inline") if isinstance(entry, dict) else None
            seeds = list(range(SEED_RANGES[split]["first"], SEED_RANGES[split]["last"] + 1))
            if len(assigned) != size or not isinstance(rows, list) or [r.get("world_identity") for r in rows] != assigned or [r.get("world_seed") for r in rows] != seeds:
                errors.append(f"PSPS-v1 {split} manifest changed")
                continue
            identities.extend(assigned)
            for row in rows:
                identity = row["world_identity"]
                path = base_dir / "worlds" / f"{identity.removeprefix('sha256:')}.json"
                try:
                    world = json.loads(path.read_text())
                except (OSError, json.JSONDecodeError):
                    errors.append(f"PSPS-v1 world unreadable: {identity}")
                    continue
                if object_hash(world) != identity or world.get("world_seed") != row["world_seed"]:
                    errors.append(f"PSPS-v1 world identity mismatch: {identity}")
                if contract.get("world_registry", {}).get(identity, {}).get("source_split") != split:
                    errors.append(f"PSPS-v1 world registry mismatch: {identity}")
        if len(identities) != len(set(identities)) or set(identities) != set(contract.get("world_registry", {})):
            errors.append("PSPS-v1 split overlap or registry mismatch")
        pai_path = ROOT / "artifacts/paired_advantage_identification_20260914_v3/contract.json"
        pai = json.loads(pai_path.read_text())
        pai_confirm = sum(e.get("event_type") == "ACCESS" and e.get("split") == "CMI_V3_CONFIRM" for e in pai["access_registry"]["events"])
        boundary = pre.get("old_pai_boundary", {})
        if boundary != {"contract_sha256": file_hash(pai_path), "confirm_accesses": 0, "permanently_forbidden": True} or pai_confirm != 0:
            errors.append("old PAI CONFIRM boundary changed or was accessed")
        prior = set()
        for path in (ROOT / "artifacts").glob("*/contract.json"):
            if path.resolve() == (base_dir / "contract.json").resolve():
                continue
            try:
                prior.update(json.loads(path.read_text()).get("world_registry", {}))
            except (OSError, json.JSONDecodeError):
                pass
        if prior.intersection(identities):
            errors.append("PSPS-v1 worlds overlap a prior registered lineage")
    if events:
        if events[0].get("event_type") != "FREEZE" or events[0].get("stage") != "PSPS_DEV" or events[0].get("record_hash") != pre.get("record_sha256"):
            errors.append("PSPS-v1 freeze genesis invalid")
    accesses = [e for e in events if e.get("event_type") == "ACCESS"]
    observed = [e.get("world_identity") for e in accesses if e.get("split") == "PSPS_DEV"]
    assigned = pre.get("master_splits", {}).get("PSPS_DEV", [])
    if observed != assigned[:len(observed)] or len(observed) != len(set(observed)):
        errors.append("PSPS DEV access is not an ordered unique prefix")
    confirm_count = sum(e.get("split") == "PSPS_CONFIRM" for e in accesses)
    if confirm_count:
        errors.append("PSPS CONFIRM access is forbidden under the DEV contract")
    receipt_errors = []
    if receipt is None:
        receipt_errors.append("PSPS-v1 external freeze receipt required")
    else:
        try:
            frozen = freeze_only_contract(contract)
        except ValueError as error:
            receipt_errors.append(str(error))
        else:
            receipt_errors.extend(validate_access_anchor(frozen, receipt, registered))
    return {"schema_version": "psps-v1-validation-v1", "valid": not errors, "errors": errors, "receipt_errors": receipt_errors, "PSPS_DEV_READY": not errors and not receipt_errors, "PSPS_CONFIRM_READY": False, "psps_dev_accessed": len(observed), "psps_confirm_accessed": confirm_count, "old_pai_confirm_accessed": 0 if not errors else None, "method_train_authorized": False}


def append_access_event(contract_path: Path, split: str, identity: str, run_id: str) -> None:
    from scripts.run_conditional_history_collection import atomic_json
    if split != "PSPS_DEV":
        raise ValueError("DEV contract may append only PSPS_DEV access")
    contract = json.loads(contract_path.read_text())
    events = contract["access_registry"]["events"]
    prior = [e["world_identity"] for e in events if e.get("event_type") == "ACCESS" and e.get("split") == split]
    assigned = contract["pre_h"]["master_splits"][split]
    if identity in prior:
        return
    if prior != assigned[:len(prior)] or identity != assigned[len(prior)]:
        raise ValueError("refusing out-of-order PSPS DEV access")
    artifact = {"schema_version": "psps-v1-world-access-v1", "run_id": run_id, "split": split, "world_identity": identity}
    artifact_hash = object_hash(artifact)
    contract["content_registry"][artifact_hash] = {"inline": artifact}
    event = {"sequence": events[-1]["sequence"] + 1, "event_id": f"access-psps-dev-{len(prior):04d}", "event_type": "ACCESS", "split": split, "world_identity": identity, "run_id": run_id, "access_artifact_hash": artifact_hash, "previous_event_hash": contract["access_registry"]["head_event_hash"]}
    event["event_hash"] = access_event_hash(event)
    events.append(event)
    contract["access_registry"]["event_count"] = len(events)
    contract["access_registry"]["head_event_hash"] = event["event_hash"]
    atomic_json(contract_path, contract)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("contract", type=Path); parser.add_argument("--receipt", type=Path)
    args = parser.parse_args(); path = args.contract.resolve()
    receipt = json.loads(args.receipt.resolve().read_text()) if args.receipt else None
    result = validate_contract(json.loads(path.read_text()), path.parent, receipt)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["PSPS_DEV_READY"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
