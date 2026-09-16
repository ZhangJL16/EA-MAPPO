#!/usr/bin/env python3
"""Run fresh paired-advantage DEV with the recyclable four-world coordinator."""
from __future__ import annotations

import sys
import os
import time
import traceback
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_cmi_v4_parallel_collection as base
from scripts.run_conditional_history_collection import collect_world as legacy_collect_world
from scripts.run_conditional_history_collection import atomic_json, write_record
from scripts.validate_paired_advantage_contract import append_access_event, validate_contract


def collect_world(*args, **kwargs):
    meta, arrays = legacy_collect_world(*args, **kwargs)
    if meta is not None:
        for anchor in meta["anchors"]:
            for row in anchor.get("resamples", []):
                outcome = row["outcome"]
                outcome["collision_free_arrival"] = outcome["collision_count"] == 0
    return meta, arrays


def collect_task(task, *, output: str, split: str, contract_id: str, replicates: int, guard: int):
    output_path = Path(output)
    started = time.monotonic()
    try:
        if base._NAVIGATOR is None:
            raise RuntimeError("paired-advantage worker was not initialized")
        meta, arrays = collect_world(
            output=output_path,
            navigator=base._NAVIGATOR,
            world_seed=task.world_seed,
            identity=task.identity,
            split=split,
            contract_id=contract_id,
            replicates=replicates,
            workers=base.WORKERS_PER_WORLD,
            obstacles=48,
            guard=guard,
            pause_requested=lambda: (output_path / "PAUSE").exists(),
        )
        if meta is None or arrays is None:
            return None
        meta.update({
            "schema_version": "conditional-history-mc-q-world-result-v3",
            "role": "PAI_DEV_RAW_PAIRED_ADVANTAGE_NO_VERDICT",
            "execution_version": "WORLD_PROCESS_COORDINATOR_V4",
            "mc_q_resamples_per_anchor_action": replicates,
            "v1_v2_outcomes_used": False,
            "automatic_confirmation": False,
        })
        record = write_record(output_path, meta, arrays)
        ticket = {
            "schema_version": "cmi-v4-world-ticket-v1",
            "task": asdict(task), "record": record, "worker_pid": os.getpid(),
            "wall_seconds": time.monotonic() - started,
            "role": "CMI_V4_FORMAL_RAW_WORLD",
        }
        atomic_json(base.ticket_path(output_path, task.index), ticket)
        return ticket
    except Exception:
        atomic_json(output_path / "worker_errors" / f"{task.index:04d}.json", {
            "schema_version": "paired-advantage-worker-error-v1",
            "task": asdict(task), "traceback": traceback.format_exc(),
        })
        raise


def main() -> int:
    base.collect_world = collect_world
    base.collect_task = collect_task
    base.validate_contract = validate_contract
    base.append_access_event = append_access_event
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
