#!/usr/bin/env python3
"""Freeze fresh PSPS-v1 worlds, sources, rules, and access genesis."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.directional_navigation.conditional_model_access import ConditionalModelAccessBranch
from scripts.run_conditional_history_collection import atomic_json
from scripts.validate_identification_contract import file_hash, object_hash, record_hash, seal_access_registry
from scripts.validate_psps_v1_contract import LIVE_SOURCE_PATHS, PRE_SCHEMA, SCHEMA, SEED_RANGES, SPLIT_SIZES, static_spec


def register_inline(contract: dict[str, Any], value: Any) -> str:
    digest = object_hash(value); contract["content_registry"][digest] = {"inline": value}; return digest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(); output = args.output_dir.resolve()
    if output.exists(): raise FileExistsError(output)
    output.mkdir(parents=True); (output / "worlds").mkdir(); (output / "frozen_sources").mkdir()
    contract: dict[str, Any] = {"schema_version": SCHEMA, "status": "FROZEN_PSPS_V1_DEV", "content_registry": {}, "world_registry": {}, "access_registry": {"mode": "HASH_CHAINED_APPEND_ONLY_V1", "chain_genesis": "sha256:" + "0" * 64, "events": [], "event_count": 0, "head_event_hash": "sha256:" + "0" * 64}}

    pai_path = ROOT / "artifacts/paired_advantage_identification_20260914_v3/contract.json"
    pai = json.loads(pai_path.read_text()); old_pre = pai["pre_h"]
    service_hash = old_pre["anchor_service_contract_hash"]; policy_hash = old_pre["anchor_verification_policy_hash"]
    service = pai["content_registry"][service_hash]["inline"]
    for digest in {service_hash, policy_hash, service["rekor_public_key_hash"], service["anchor_signer_public_key_hash"]}:
        contract["content_registry"][digest] = pai["content_registry"][digest]

    source_hashes = {}
    for name, source in LIVE_SOURCE_PATHS.items():
        if not source.exists(): raise FileNotFoundError(source)
        destination = output / "frozen_sources" / f"{name}{source.suffix}"
        shutil.copy2(source, destination); digest = file_hash(destination)
        contract["content_registry"][digest] = {"path": destination.relative_to(output).as_posix()}; source_hashes[name] = digest

    prior_worlds = set()
    for path in (ROOT / "artifacts").glob("*/contract.json"):
        try: prior_worlds.update(json.loads(path.read_text()).get("world_registry", {}))
        except (OSError, json.JSONDecodeError): pass
    manifests = {split: [] for split in SPLIT_SIZES}
    environment = ConditionalModelAccessBranch(obstacles=48, guard=12_000)
    try:
        for split, bounds in SEED_RANGES.items():
            seeds = list(range(bounds["first"], bounds["last"] + 1))
            if len(seeds) != SPLIT_SIZES[split]: raise ValueError(f"seed count mismatch: {split}")
            for seed in seeds:
                environment.reset_world(seed); world = environment.physical_world_record(); identity = object_hash(world)
                if identity in prior_worlds: raise ValueError("PSPS world overlaps prior lineage")
                path = output / "worlds" / f"{identity.removeprefix('sha256:')}.json"; atomic_json(path, world)
                rendered = file_hash(path)
                if rendered != identity: contract["content_registry"][rendered] = {"path": path.relative_to(output).as_posix()}
                contract["content_registry"][identity] = {"inline": world}
                seed_hash = register_inline(contract, {"schema_version": "psps-v1-world-seed-v1", "world_seed": seed})
                contract["world_registry"][identity] = {"physical_world_hash": identity, "world_seed_or_parameters_hash": seed_hash, "source_split": split}
                manifests[split].append({"world_identity": identity, "world_seed": seed})
    finally: environment.close()
    identities = [r["world_identity"] for rows in manifests.values() for r in rows]
    if len(identities) != len(set(identities)): raise ValueError("duplicate PSPS physical worlds")
    manifest_hashes = {split: register_inline(contract, rows) for split, rows in manifests.items()}
    pre = {"schema_version": PRE_SCHEMA, "static_spec": static_spec(), "master_splits": {split: [r["world_identity"] for r in rows] for split, rows in manifests.items()}, "world_manifest_hashes": manifest_hashes, "source_hashes": source_hashes, "old_pai_boundary": {"contract_sha256": file_hash(pai_path), "confirm_accesses": 0, "permanently_forbidden": True}, "anchor_service_contract_hash": service_hash, "anchor_verification_policy_hash": policy_hash}
    pre["record_sha256"] = record_hash(pre); contract["pre_h"] = pre
    contract["access_registry"]["events"] = [{"sequence": 1, "event_id": "freeze-psps-v1-dev", "event_type": "FREEZE", "stage": "PSPS_DEV", "record_hash": pre["record_sha256"]}]
    seal_access_registry(contract); atomic_json(output / "contract.json", contract)
    print(json.dumps({"contract": str(output / 'contract.json'), "record_sha256": pre["record_sha256"], "head_event_hash": contract["access_registry"]["head_event_hash"], "split_sizes": SPLIT_SIZES}, sort_keys=True))
    return 0


if __name__ == "__main__": raise SystemExit(main())
