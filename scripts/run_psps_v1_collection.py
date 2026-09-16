#!/usr/bin/env python3
"""Resumable four-world PSPS-v1 SMOKE/DEV collector; never opens CONFIRM."""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import signal
import sys
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

import scripts.run_cmi_v4_parallel_collection as coordinator
from scripts.run_cmi_v4_parallel_collection import WorldTask, checkpoint, ticket_path, validate_ticket
from scripts.run_conditional_history_collection import atomic_json, collect_world as common_collect_world, write_record
from scripts.run_dvoi_h_collection import FrozenNavigator, read_world_seed
from scripts.validate_psps_v1_contract import append_access_event, validate_contract

WORLD_PROCESSES = 4; WORKERS_PER_WORLD = 8; STOP = False; NAVIGATOR = None


def request_stop(*_: object) -> None:
    global STOP; STOP = True


def worker_init(device: str) -> None:
    global NAVIGATOR
    torch.set_num_threads(1); NAVIGATOR = FrozenNavigator(device)


def collect_task(task: WorldTask, *, output: str, split: str, contract_id: str, replicates: int, guard: int):
    output_path = Path(output); started = time.monotonic()
    try:
        if NAVIGATOR is None: raise RuntimeError("PSPS worker navigator unavailable")
        meta, arrays = common_collect_world(output=output_path, navigator=NAVIGATOR, world_seed=task.world_seed, identity=task.identity, split=split, contract_id=contract_id, replicates=replicates, workers=WORKERS_PER_WORLD, obstacles=48, guard=guard, pause_requested=lambda: (output_path / "PAUSE").exists())
        if meta is None or arrays is None: return None
        for anchor in meta["anchors"]:
            for row in anchor.get("resamples", []):
                row["outcome"]["collision_free_arrival"] = row["outcome"]["collision_count"] == 0
        meta.update({"schema_version": "psps-v1-world-result-v1", "role": "PSPS_V1_RAW_NO_GATE_VERDICT", "execution_version": "WORLD_PROCESS_COORDINATOR_V4", "paired_resamples_per_anchor_action": replicates, "old_pai_outcomes_used": False, "automatic_analysis": False, "automatic_confirmation": False, "automatic_training": False})
        record = write_record(output_path, meta, arrays)
        ticket = {"schema_version": "cmi-v4-world-ticket-v1", "task": asdict(task), "record": record, "worker_pid": os.getpid(), "wall_seconds": time.monotonic() - started, "role": "CMI_V4_FORMAL_RAW_WORLD"}
        atomic_json(ticket_path(output_path, task.index), ticket); return ticket
    except Exception:
        atomic_json(output_path / "worker_errors" / f"{task.index:04d}.json", {"schema_version": "psps-v1-worker-error-v1", "task": asdict(task), "traceback": traceback.format_exc()}); raise


def write_status(output: Path, state: str, records: list, tasks: list, active: int, accessed: int, seconds: float) -> None:
    staged = sum(validate_ticket(output, task) is not None for task in tasks)
    atomic_json(output / "status.json", {"status": state, "pid": os.getpid(), "completed": len(records), "target": len(tasks), "generation": len(records), "staged": staged, "active": active, "accessed": accessed, "seconds": seconds, "world_processes": WORLD_PROCESSES, "workers_per_world": WORKERS_PER_WORLD, "max_worlds_per_child": 1, "execution_version": "WORLD_PROCESS_COORDINATOR_V4", "old_pai_outcomes_used": False, "automatic_analysis": False, "automatic_confirmation": False, "automatic_training": False})


def run(args: argparse.Namespace) -> int:
    contract_path = args.contract.resolve(); contract = json.loads(contract_path.read_text()); receipt = json.loads(args.freeze_receipt.resolve().read_text())
    validation = validate_contract(contract, contract_path.parent, receipt)
    if not validation["PSPS_DEV_READY"]: raise ValueError("PSPS_DEV_READY failed: " + "; ".join(validation["errors"] + validation["receipt_errors"]))
    if args.split not in {"SMOKE_DEBUG", "PSPS_DEV"}: raise ValueError("DEV collector cannot open CONFIRM")
    pre = contract["pre_h"]; spec = pre["static_spec"]; identities = list(pre["master_splits"][args.split])
    tasks = [WorldTask(i, identity, read_world_seed(contract, identity)) for i, identity in enumerate(identities)]
    replicates = int(spec["resamples_per_anchor_action"][args.split]); output = args.output_dir.resolve(); run_id = f"{spec['contract_id']}:{args.split}:RAW_V1"
    manifest = {"schema_version": "psps-v1-collection-v1", "run_id": run_id, "contract_record_hash": pre["record_sha256"], "split": args.split, "world_identities": identities, "anchor_steps": spec["anchor_steps"], "history_length": spec["history_length"], "history_stride": spec["history_stride"], "paired_resamples_per_anchor_action": replicates, "world_processes": 4, "workers_per_world": 8, "max_worlds_per_child": 1, "device": args.device, "old_pai_outcomes_used": False, "automatic_analysis": False, "automatic_confirmation": False, "automatic_training": False}
    if args.resume:
        if json.loads((output / "manifest.json").read_text()) != manifest: raise ValueError("PSPS resume manifest mismatch")
        latest = json.loads((output / "latest.json").read_text()); records = list(latest["records"]); elapsed_before = float(latest["seconds"])
        if [r["world_identity"] for r in records] != identities[:len(records)]: raise ValueError("PSPS checkpoint is not ordered prefix")
    else:
        if output.exists(): raise FileExistsError(output)
        output.mkdir(parents=True); atomic_json(output / "manifest.json", manifest); records=[]; elapsed_before=0.0; records=checkpoint(output,tasks,records,0.0)
    signal.signal(signal.SIGTERM, request_stop); signal.signal(signal.SIGINT, request_stop); started=time.monotonic()
    elapsed=lambda: elapsed_before + time.monotonic()-started
    def access_count():
        if args.split == "SMOKE_DEBUG": return len(tasks)
        current=json.loads(contract_path.read_text()); return sum(e.get("event_type")=="ACCESS" and e.get("split")==args.split for e in current["access_registry"]["events"])
    for task in tasks: validate_ticket(output, task)
    records=checkpoint(output,tasks,records,elapsed()); futures={}; active_indices=set(); context=mp.get_context("spawn")
    write_status(output,"RUNNING",records,tasks,0,access_count(),elapsed())
    try:
        with ProcessPoolExecutor(max_workers=WORLD_PROCESSES, mp_context=context, initializer=worker_init, initargs=(args.device,), max_tasks_per_child=1) as pool:
            while len(records)<len(tasks):
                records=checkpoint(output,tasks,records,elapsed())
                if args.stop_after_worlds and len(records)>=args.stop_after_worlds: break
                if not STOP and not (output/"PAUSE").exists():
                    for task in tasks:
                        if len(futures)>=WORLD_PROCESSES: break
                        if task.index<len(records) or task.index in active_indices or validate_ticket(output,task) is not None: continue
                        accessed=access_count()
                        if task.index>accessed: break
                        if task.index==accessed and args.split=="PSPS_DEV": append_access_event(contract_path,args.split,task.identity,run_id)
                        futures[pool.submit(collect_task,task,output=str(output),split=args.split,contract_id=spec["contract_id"],replicates=replicates,guard=spec["branch_guard"])]=task; active_indices.add(task.index)
                write_status(output,"RUNNING",records,tasks,len(futures),access_count(),elapsed())
                if not futures: break
                done,_=wait(futures,return_when=FIRST_COMPLETED)
                for future in done:
                    task=futures.pop(future); active_indices.remove(task.index); future.result(); records=checkpoint(output,tasks,records,elapsed()); print(f"finished task {task.index}; committed {len(records)}/{len(tasks)}",flush=True)
        records=checkpoint(output,tasks,records,elapsed()); complete=len(records)==len(tasks)
        state=("COMPLETE_AWAITING_PSPS_DEV_ANALYSIS" if args.split=="PSPS_DEV" else "COMPLETE_SMOKE_DEBUG") if complete else "PAUSED_RESUMABLE"
        write_status(output,state,records,tasks,0,access_count(),elapsed())
        if complete: atomic_json(output/"results.json", {"schema_version":"psps-v1-run-results-v1","records":records,"completed":len(records),"target":len(tasks),"total_branch_policy_steps":sum(r["total_branch_policy_steps"] for r in records),"role":"PSPS_V1_RAW_NO_GATE_VERDICT","automatic_analysis":False,"automatic_confirmation":False,"automatic_training":False})
        return 0
    except Exception:
        atomic_json(output/"ERROR.json",{"error":traceback.format_exc(),"completed":len(records),"target":len(tasks),"automatic_analysis":False,"automatic_confirmation":False,"automatic_training":False}); raise


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--contract",type=Path,required=True); parser.add_argument("--split",choices=("SMOKE_DEBUG","PSPS_DEV"),required=True); parser.add_argument("--freeze-receipt",type=Path,required=True); parser.add_argument("--output-dir",type=Path,required=True); parser.add_argument("--device",default="cuda"); parser.add_argument("--resume",action="store_true"); parser.add_argument("--stop-after-worlds",type=int,default=0)
    return run(parser.parse_args())


if __name__=="__main__": raise SystemExit(main())
