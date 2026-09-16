#!/usr/bin/env python3
"""Resumable CMI-v3 confirmation collector gated by the anchored DEV child."""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
import traceback
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_conditional_history_collection import atomic_json, collect_world, write_record
from scripts.run_dvoi_h_collection import FrozenNavigator, read_world_seed
from scripts.validate_cmi_v3_contract import (
    append_access_event, validate_confirmation_contract,
)


STOP = False


def request_stop(*_: object) -> None:
    global STOP
    STOP = True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirmation-contract", type=Path, required=True)
    parser.add_argument("--confirmation-receipt", type=Path, required=True)
    parser.add_argument("--dev-analysis", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-worlds", type=int, default=0)
    args = parser.parse_args()
    if args.workers <= 0:
        raise ValueError("CMI-v3 confirmation worker count must be positive")
    contract_path = args.confirmation_contract.resolve()
    contract = json.loads(contract_path.read_text())
    receipt = json.loads(args.confirmation_receipt.resolve().read_text())
    validation = validate_confirmation_contract(
        contract, contract_path.parent, args.dev_analysis.resolve(), receipt
    )
    if not validation["CMI_V3_CONFIRM_READY"]:
        raise ValueError(
            "CMI_V3_CONFIRM_READY failed: "
            + "; ".join(validation["errors"] + validation["receipt_errors"])
        )
    pre = contract["pre_h"]
    split = "CMI_V3_CONFIRM"
    identities = pre["master_splits"][split]
    replicates = pre["resamples_per_anchor_action"][split]
    output = args.output_dir.resolve()
    manifest = {
        "schema_version": "conditional-history-mc-q-confirm-collection-v3",
        "run_id": f"{pre['contract_id']}:{split}:RAW_V1",
        "confirmation_record_hash": contract["confirmation_child"]["record_sha256"],
        "split": split,
        "world_identities": identities,
        "anchor_steps": pre["anchor_steps"],
        "evaluation_resamples_per_anchor_action": replicates,
        "workers": args.workers,
        "device": args.device,
        "automatic_analysis": False,
        "automatic_training": False,
    }
    if args.resume:
        if json.loads((output / "manifest.json").read_text()) != manifest:
            raise ValueError("CMI-v3 confirmation resume manifest mismatch")
        latest = json.loads((output / "latest.json").read_text())
        completed, records = list(latest["completed"]), list(latest["records"])
        generation = int(latest["generation"])
        total_steps, elapsed_before = int(latest["total_branch_policy_steps"]), float(latest["seconds"])
    else:
        if output.exists():
            raise FileExistsError(output)
        output.mkdir(parents=True)
        atomic_json(output / "manifest.json", manifest)
        completed, records, generation, total_steps, elapsed_before = [], [], 0, 0, 0.0
        atomic_json(output / "latest.json", {
            "schema_version": "conditional-history-mc-q-confirm-checkpoint-v3",
            "generation": 0, "completed": [], "target": len(identities),
            "records": [], "total_branch_policy_steps": 0, "seconds": 0.0,
        })
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    torch.set_num_threads(1)
    navigator = FrozenNavigator(args.device)
    started = time.monotonic()

    def pause_requested() -> bool:
        return STOP or (output / "PAUSE").exists()

    atomic_json(output / "status.json", {
        "status": "RUNNING", "pid": os.getpid(), "completed": len(completed),
        "target": len(identities), "generation": generation,
        "automatic_analysis": False, "automatic_training": False,
    })
    try:
        for identity in identities:
            if identity in completed:
                continue
            append_access_event(contract_path, split, identity, manifest["run_id"])
            seed = read_world_seed(json.loads(contract_path.read_text()), identity)
            meta, arrays = collect_world(
                output=output, navigator=navigator, world_seed=seed, identity=identity,
                split=split, contract_id=pre["contract_id"], replicates=replicates,
                workers=args.workers, obstacles=48, guard=pre["branch_guard"],
                pause_requested=pause_requested,
            )
            if meta is None or arrays is None:
                break
            meta.update({
                "schema_version": "conditional-history-mc-q-confirm-world-result-v3",
                "role": "CMI_V3_CONFIRM_RAW_NO_EARLY_ANALYSIS",
                "mc_q_resamples_per_anchor_action": replicates,
                "automatic_confirmation_analysis": False,
            })
            record = write_record(output, meta, arrays)
            completed.append(identity)
            records.append(record)
            generation += 1
            total_steps += int(record["total_branch_policy_steps"])
            elapsed = elapsed_before + time.monotonic() - started
            atomic_json(output / "latest.json", {
                "schema_version": "conditional-history-mc-q-confirm-checkpoint-v3",
                "generation": generation, "completed": completed,
                "target": len(identities), "records": records,
                "total_branch_policy_steps": total_steps, "seconds": elapsed,
            })
            atomic_json(output / "status.json", {
                "status": "RUNNING", "pid": os.getpid(), "completed": len(completed),
                "target": len(identities), "generation": generation,
                "total_branch_policy_steps": total_steps, "seconds": elapsed,
                "automatic_analysis": False, "automatic_training": False,
            })
            print(f"checkpoint {generation}: {len(completed)}/{len(identities)} worlds", flush=True)
            if pause_requested() or (args.stop_after_worlds and len(completed) >= args.stop_after_worlds):
                break
        complete = len(completed) == len(identities)
        status = json.loads((output / "status.json").read_text())
        status["status"] = "COMPLETE_AWAITING_CMI_V3_CONFIRM_ANALYSIS" if complete else "PAUSED_RESUMABLE"
        atomic_json(output / "status.json", status)
        if complete:
            atomic_json(output / "results.json", {
                "schema_version": "conditional-history-mc-q-confirm-results-v3",
                "records": records, "completed": len(completed), "target": len(identities),
                "total_branch_policy_steps": total_steps,
                "automatic_analysis": False, "automatic_training": False,
            })
        return 0
    except Exception:
        atomic_json(output / "ERROR.json", {
            "error": traceback.format_exc(), "completed": len(completed),
            "target": len(identities), "automatic_analysis": False,
            "automatic_training": False,
        })
        raise


if __name__ == "__main__":
    raise SystemExit(main())
