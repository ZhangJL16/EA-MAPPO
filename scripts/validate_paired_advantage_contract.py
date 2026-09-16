#!/usr/bin/env python3
"""Validate the fresh paired-advantage DEV contract and access prefix."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.validate_cmi_v4_contract as base
from scripts.validate_identification_contract import file_hash

SCHEMA = "paired-advantage-identification-contract-v3"
PRE_H_SCHEMA = "paired-advantage-identification-preaccess-v3"
SPLIT_SIZES = {"SMOKE_DEBUG": 2, "CMI_V3_DEV": 96, "CMI_V3_CONFIRM": 128}
SEED_RANGES = {
    "SMOKE_DEBUG": {"first": 1_136_000_453, "last": 1_136_000_454},
    "CMI_V3_DEV": {"first": 1_136_000_455, "last": 1_136_000_550},
    "CMI_V3_CONFIRM": {"first": 1_136_000_551, "last": 1_136_000_678},
}
LIVE_SOURCE_PATHS = {
    "protocol_v3": ROOT / "docs/PAIRED_ADVANTAGE_IDENTIFICATION_PROTOCOL_20260914.md",
    "execution_addendum_v4": ROOT / "docs/PAIRED_ADVANTAGE_IDENTIFICATION_CONTRACT_TEMPLATE.json",
    "freezer_v4": ROOT / "scripts/freeze_paired_advantage_experiment.py",
    "collector_v4": ROOT / "scripts/run_paired_advantage_collection.py",
    "validator_v4": ROOT / "scripts/validate_paired_advantage_contract.py",
    "environment": ROOT / "experiments/directional_navigation/conditional_model_access.py",
    "collector_common": ROOT / "scripts/run_conditional_history_collection.py",
    "models_v3": ROOT / "scripts/cmi_v3_models.py",
    "dev_analyzer_v3": ROOT / "scripts/analyze_paired_advantage_dev.py",
    "dvoi_environment": ROOT / "experiments/directional_navigation/dvoi_h_branch.py",
    "frozen_navigator": ROOT / "scripts/run_dvoi_h_collection.py",
    "locked_recovery": ROOT / "experiments/directional_navigation/recovery.py",
    "collision_protocol": ROOT / "docs/LOCKED_COLLISION_RECOVERY_PROTOCOL.md",
}
_base_template = base.expected_pre_h_without_hash


def expected_pre_h_without_hash():
    value = _base_template()
    value["schema_version"] = PRE_H_SCHEMA
    value["contract_id"] = "paired-advantage-identification-20260914-v3"
    value["world_seed_ranges"] = SEED_RANGES
    value["stage_aliases"] = {
        "CMI_V3_DEV": "PAI_DEV", "CMI_V3_CONFIRM": "PAI_CONFIRM"
    }
    value["adequacy_rule"] = json.loads(
        (ROOT / "docs/PAIRED_ADVANTAGE_IDENTIFICATION_CONTRACT_TEMPLATE.json").read_text()
    )
    value["adequacy_rule"].pop("status")
    value["confirmation_rule"] = {
        "requires_gate_h_and_gate_i": True,
        "requires_hash_linked_child": True,
        "requires_new_external_receipt": True,
        "method_train_authorized": False,
    }
    value["model_rule"]["target"] = "PAIRED_U_C_MINUS_U_R"
    value["model_rule"]["no_model_complexity_increase"] = True
    return value


def configure_base() -> None:
    base.SCHEMA = SCHEMA
    base.PRE_H_SCHEMA = PRE_H_SCHEMA
    base.SPLIT_SIZES = SPLIT_SIZES
    base.SEED_RANGES = SEED_RANGES
    base.LIVE_SOURCE_PATHS = LIVE_SOURCE_PATHS
    base.expected_pre_h_without_hash = expected_pre_h_without_hash


def validate_contract(contract, base_dir: Path, receipt=None):
    configure_base()
    result = base.validate_contract(contract, base_dir, receipt)
    errors = result["errors"]
    identities = set(contract.get("world_registry", {}))
    prior_paths = (
        ROOT / "artifacts/conditional_history_information_20260912_v2/contract.json",
        ROOT / "artifacts/conditional_history_mc_q_20260912_v3/contract.json",
        ROOT / "artifacts/conditional_history_mc_q_20260913_v4/contract.json",
        ROOT / "artifacts/paired_advantage_identification_20260914_v1/contract.json",
        ROOT / "artifacts/paired_advantage_identification_20260914_v2/contract.json",
    )
    for path in prior_paths:
        prior = json.loads(path.read_text())
        if identities.intersection(prior.get("world_registry", {})):
            errors.append(f"paired-advantage worlds overlap {path.parent.name}")
    pre = contract.get("pre_h", {})
    if pre.get("protocol_hash") != pre.get("source_hashes", {}).get("protocol_v3"):
        errors.append("paired-advantage protocol binding changed")
    result.update({
        "schema_version": "paired-advantage-identification-validation-v1",
        "valid": not errors,
        "PAI_DEV_READY": not errors and not result["receipt_errors"],
        "PAI_CONFIRM_READY": False,
    })
    result["CMI_V4_DEV_READY"] = result["PAI_DEV_READY"]
    return result


append_access_event = base.append_access_event


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("contract", type=Path)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    contract_path = args.contract.resolve()
    receipt = json.loads(args.receipt.resolve().read_text()) if args.receipt else None
    result = validate_contract(json.loads(contract_path.read_text()), contract_path.parent, receipt)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["PAI_DEV_READY"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
