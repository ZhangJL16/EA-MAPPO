#!/usr/bin/env python3
"""Materialize a CMI-v3 confirmation child only after every DEV adequacy gate passes."""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.freeze_dvoi_pre_h import atomic_json
from scripts.validate_cmi_v3_contract import SPLIT_SIZES, validate_contract
from scripts.validate_identification_contract import (
    access_event_hash, file_hash, record_hash, seal_access_registry,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-contract", type=Path, required=True)
    parser.add_argument("--parent-receipt", type=Path, required=True)
    parser.add_argument("--dev-analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    parent_path = args.parent_contract.resolve()
    parent = json.loads(parent_path.read_text())
    parent_receipt = json.loads(args.parent_receipt.resolve().read_text())
    validation = validate_contract(parent, parent_path.parent, parent_receipt)
    if not validation["CMI_V3_DEV_READY"] or validation["cmi_v3_dev_accessed"] != SPLIT_SIZES["CMI_V3_DEV"]:
        raise ValueError("CMI-v3 parent is not complete and valid for confirmation freeze")
    dev = args.dev_analysis.resolve()
    adequacy = json.loads((dev / "adequacy.json").read_text())
    analysis = json.loads((dev / "analysis.json").read_text())
    if adequacy.get("passed") is not True or analysis.get("adequacy_passed") is not True:
        raise ValueError("CMI-v3 estimator adequacy failed; confirmation remains forbidden")
    hashes = analysis.get("artifact_hashes")
    if not isinstance(hashes, dict) or any(file_hash(dev / name) != digest for name, digest in hashes.items()):
        raise ValueError("CMI-v3 DEV artifact binding is invalid")
    child_contract = copy.deepcopy(parent)
    parent_head = child_contract["access_registry"]["head_event_hash"]
    child = {
        "schema_version": "conditional-history-mc-q-confirm-child-v3",
        "parent_preaccess_hash": parent["pre_h"]["record_sha256"],
        "parent_dev_access_head": parent_head,
        "dev_analysis_hash": file_hash(dev / "analysis.json"),
        "dev_artifact_hashes": hashes,
        "confirm_world_manifest_hash": parent["pre_h"]["world_manifest_hashes"]["CMI_V3_CONFIRM"],
        "adequacy_passed": True,
        "method_train_authorized": False,
    }
    child["record_sha256"] = record_hash(child)
    child_contract["content_registry"][child["record_sha256"]] = {"inline": child}
    child_contract["confirmation_child"] = child
    child_contract["status"] = "FROZEN_CMI_V3_CONFIRM"
    event = {
        "sequence": int(child_contract["access_registry"]["events"][-1]["sequence"]) + 1,
        "event_id": "freeze-cmi-v3-confirm-v1",
        "event_type": "FREEZE",
        "stage": "CMI_V3_CONFIRM",
        "record_hash": child["record_sha256"],
        "previous_event_hash": parent_head,
    }
    event["event_hash"] = access_event_hash(event)
    child_contract["access_registry"]["events"].append(event)
    seal_access_registry(child_contract)
    atomic_json(args.output.resolve(), child_contract)
    print(json.dumps({
        "confirmation_contract": str(args.output.resolve()),
        "record_sha256": child["record_sha256"],
        "head_event_hash": child_contract["access_registry"]["head_event_hash"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
