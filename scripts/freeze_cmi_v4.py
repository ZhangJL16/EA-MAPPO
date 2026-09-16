#!/usr/bin/env python3
"""Freeze fresh CMI-v4 worlds and the execution-only parallel replay."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.directional_navigation.conditional_model_access import ConditionalModelAccessBranch
from scripts.freeze_dvoi_pre_h import atomic_json
from scripts.validate_cmi_v4_contract import (
    LIVE_SOURCE_PATHS,
    PRE_H_SCHEMA,
    SCHEMA,
    SEED_RANGES,
    SPLIT_SIZES,
    expected_pre_h_without_hash,
)
from scripts.validate_identification_contract import (
    file_hash,
    object_hash,
    record_hash,
    seal_access_registry,
)


def register_inline(contract: dict[str, Any], value: Any) -> str:
    digest = object_hash(value)
    contract["content_registry"][digest] = {"inline": value}
    return digest


def service_inactive() -> bool:
    completed = subprocess.run(
        ["systemctl", "--user", "is-active", "cmi-history-mcq-dev-v3-resume2.service"],
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() in {"inactive", "failed", "unknown"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)

    v3_root = ROOT / "artifacts/conditional_history_mc_q_20260912_v3"
    v3_raw = v3_root / "cmi_v3_dev_raw_v1"
    if not (v3_raw / "PAUSE").exists() or not service_inactive():
        raise RuntimeError("CMI-v3 must be paused and inactive before the v4 freeze")
    if (v3_raw / "ERROR.json").exists():
        raise RuntimeError("CMI-v3 has an unresolved ERROR artifact")
    v3_contract_path = v3_root / "contract.json"
    v3_contract = json.loads(v3_contract_path.read_text())
    v3_latest_path = v3_raw / "latest.json"
    v3_latest = json.loads(v3_latest_path.read_text())
    if v3_latest.get("generation") != 31:
        raise RuntimeError("CMI-v3 preserved checkpoint is not generation 31")
    v3_accesses = [
        event
        for event in v3_contract["access_registry"]["events"]
        if event.get("event_type") == "ACCESS"
    ]
    if (
        len([event for event in v3_accesses if event.get("split") == "CMI_V3_DEV"]) != 32
        or any(event.get("split") == "CMI_V3_CONFIRM" for event in v3_accesses)
    ):
        raise RuntimeError("CMI-v3 access boundary differs from DEV=32, CONFIRM=0")

    acceleration_path = (
        ROOT / "artifacts/cmi_v3_acceleration_smoke_20260913/analysis_v2.json"
    )
    acceleration = json.loads(acceleration_path.read_text())
    if (
        acceleration.get("decision") != "ELIGIBLE_FOR_FRESH_ACCELERATED_FREEZE"
        or acceleration.get("selected_processes") != 4
        or acceleration.get("selected_max_tasks_per_child") != 1
        or acceleration.get("selected_speedup", 0) < 2.0
        or not acceleration.get("all_exactly_equivalent")
    ):
        raise RuntimeError("eligible exact-equivalence acceleration decision is absent")

    output.mkdir(parents=True)
    (output / "worlds").mkdir()
    (output / "frozen_sources").mkdir()
    contract: dict[str, Any] = {
        "schema_version": SCHEMA,
        "status": "FROZEN_CMI_V4_DEV",
        "content_registry": {},
        "world_registry": {},
        "access_registry": {
            "mode": "SHA256_CANONICAL_EVENT_CHAIN_V1",
            "chain_genesis": "sha256:" + "0" * 64,
            "events": [],
            "event_count": 0,
            "head_event_hash": "sha256:" + "0" * 64,
        },
    }

    old_pre = v3_contract["pre_h"]
    service_hash = old_pre["anchor_service_contract_hash"]
    policy_hash = old_pre["anchor_verification_policy_hash"]
    service = v3_contract["content_registry"][service_hash]["inline"]
    for digest in {
        service_hash,
        policy_hash,
        service["rekor_public_key_hash"],
        service["anchor_signer_public_key_hash"],
    }:
        contract["content_registry"][digest] = v3_contract["content_registry"][digest]

    source_hashes: dict[str, str] = {}
    for name, source in LIVE_SOURCE_PATHS.items():
        if not source.exists():
            raise FileNotFoundError(f"CMI-v4 source missing before freeze: {source}")
        destination = output / "frozen_sources" / f"{name}{source.suffix}"
        shutil.copy2(source, destination)
        digest = file_hash(destination)
        contract["content_registry"][digest] = {
            "path": destination.relative_to(output).as_posix()
        }
        source_hashes[name] = digest

    old_worlds: set[str] = set()
    for old_root in (
        ROOT / "artifacts/conditional_history_information_20260912_v2",
        v3_root,
        ROOT / "artifacts/conditional_history_mc_q_20260913_v4",
        ROOT / "artifacts/paired_advantage_identification_20260914_v1",
        ROOT / "artifacts/paired_advantage_identification_20260914_v2",
    ):
        old_worlds.update(json.loads((old_root / "contract.json").read_text())["world_registry"])
    manifests: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLIT_SIZES}
    environment = ConditionalModelAccessBranch(obstacles=48, guard=12_000)
    try:
        for split, bounds in SEED_RANGES.items():
            seeds = list(range(bounds["first"], bounds["last"] + 1))
            if len(seeds) != SPLIT_SIZES[split]:
                raise ValueError(f"CMI-v4 seed count mismatch for {split}")
            for seed in seeds:
                environment.reset_world(seed)
                world = environment.physical_world_record()
                identity = object_hash(world)
                if identity in old_worlds:
                    raise ValueError("CMI-v4 physical world overlaps v2/v3")
                path = output / "worlds" / f"{identity.removeprefix('sha256:')}.json"
                atomic_json(path, world)
                rendered_hash = file_hash(path)
                if rendered_hash != identity:
                    contract["content_registry"][rendered_hash] = {
                        "path": path.relative_to(output).as_posix()
                    }
                contract["content_registry"][identity] = {"inline": world}
                seed_hash = register_inline(
                    contract,
                    {"schema_version": "dvoi-world-seed-v1", "world_seed": seed},
                )
                contract["world_registry"][identity] = {
                    "physical_world_hash": identity,
                    "world_seed_or_parameters_hash": seed_hash,
                    "source_split": split,
                }
                manifests[split].append(
                    {"world_identity": identity, "world_seed": seed}
                )
    finally:
        environment.close()
    identities = [row["world_identity"] for rows in manifests.values() for row in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("CMI-v4 generated duplicate physical worlds")
    manifest_hashes = {
        split: register_inline(contract, rows) for split, rows in manifests.items()
    }

    pre = expected_pre_h_without_hash()
    pre.update(
        {
            "schema_version": PRE_H_SCHEMA,
            "parent_v3_contract_hash": file_hash(v3_contract_path),
            "parent_v3_checkpoint_hash": file_hash(v3_latest_path),
            "acceleration_analysis_v2_hash": file_hash(acceleration_path),
            "execution_addendum_hash": source_hashes["execution_addendum_v4"],
            "protocol_hash": source_hashes["protocol_v3"],
            "master_splits": {
                split: [row["world_identity"] for row in rows]
                for split, rows in manifests.items()
            },
            "world_manifest_hashes": manifest_hashes,
            "source_hashes": source_hashes,
            "anchor_service_contract_hash": service_hash,
            "anchor_verification_policy_hash": policy_hash,
        }
    )
    pre["record_sha256"] = record_hash(pre)
    contract["pre_h"] = pre
    contract["access_registry"]["events"] = [
        {
            "sequence": 1,
            "event_id": "freeze-cmi-v4-dev-v1",
            "event_type": "FREEZE",
            "stage": "CMI_V4_DEV",
            "record_hash": pre["record_sha256"],
        }
    ]
    seal_access_registry(contract)
    atomic_json(output / "contract.json", contract)
    print(
        json.dumps(
            {
                "contract": str(output / "contract.json"),
                "record_sha256": pre["record_sha256"],
                "head_event_hash": contract["access_registry"]["head_event_hash"],
                "split_sizes": SPLIT_SIZES,
                "execution_rule": pre["execution_rule"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
