#!/usr/bin/env python3
"""Resumable four-process CMI-v4 world collector.

Only execution scheduling differs from CMI-v3.  The parent is the sole writer
of formal access events and ordered checkpoints; each spawned child executes
one complete physical world with the unchanged within-world collector.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import signal
import sys
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_conditional_history_collection import atomic_json, collect_world, write_record
from scripts.run_dvoi_h_collection import FrozenNavigator, read_world_seed
from scripts.validate_cmi_v4_contract import append_access_event, validate_contract
from scripts.validate_identification_contract import file_hash


WORLD_PROCESSES = 4
WORKERS_PER_WORLD = 8
MAX_WORLDS_PER_CHILD = 1
STOP = False
_NAVIGATOR: FrozenNavigator | None = None


@dataclass(frozen=True)
class WorldTask:
    index: int
    identity: str
    world_seed: int


def request_stop(*_: object) -> None:
    global STOP
    STOP = True


def worker_init(device: str) -> None:
    global _NAVIGATOR
    torch.set_num_threads(1)
    _NAVIGATOR = FrozenNavigator(device)


def ticket_path(output: Path, index: int) -> Path:
    return output / "staged" / f"{index:04d}.json"


def validate_ticket(output: Path, task: WorldTask) -> dict[str, Any] | None:
    path = ticket_path(output, task.index)
    if not path.exists():
        return None
    ticket = json.loads(path.read_text())
    if set(ticket) != {
        "schema_version", "task", "record", "worker_pid", "wall_seconds", "role"
    } or ticket.get("schema_version") != "cmi-v4-world-ticket-v1":
        raise ValueError(f"invalid CMI-v4 staged ticket: {path}")
    if ticket.get("role") != "CMI_V4_FORMAL_RAW_WORLD":
        raise ValueError(f"invalid CMI-v4 ticket role: {path}")
    if ticket.get("task") != asdict(task):
        raise ValueError(f"CMI-v4 ticket task mismatch: {path}")
    record = ticket.get("record")
    if not isinstance(record, dict) or record.get("world_identity") != task.identity:
        raise ValueError(f"CMI-v4 ticket record mismatch: {path}")
    for kind in ("json", "npz"):
        artifact = output / "records" / str(record[kind])
        if file_hash(artifact).removeprefix("sha256:") != record[f"{kind}_sha256"]:
            raise ValueError(f"CMI-v4 ticket {kind} hash mismatch: {path}")
    return ticket


def collect_task(
    task: WorldTask,
    *,
    output: str,
    split: str,
    contract_id: str,
    replicates: int,
    guard: int,
) -> dict[str, Any] | None:
    output_path = Path(output)
    started = time.monotonic()
    try:
        if _NAVIGATOR is None:
            raise RuntimeError("CMI-v4 worker was not initialized")
        meta, arrays = collect_world(
            output=output_path,
            navigator=_NAVIGATOR,
            world_seed=task.world_seed,
            identity=task.identity,
            split=split,
            contract_id=contract_id,
            replicates=replicates,
            workers=WORKERS_PER_WORLD,
            obstacles=48,
            guard=guard,
            pause_requested=lambda: (output_path / "PAUSE").exists(),
        )
        if meta is None or arrays is None:
            return None
        meta.update(
            {
                # Kept compatible with the frozen v3 scientific analyzer.
                "schema_version": "conditional-history-mc-q-world-result-v3",
                "role": "CMI_V4_DEV_RAW_MC_Q_NO_INFORMATION_VERDICT",
                "execution_version": "WORLD_PROCESS_COORDINATOR_V4",
                "mc_q_resamples_per_anchor_action": replicates,
                "v1_v2_outcomes_used": False,
                "automatic_confirmation": False,
            }
        )
        record = write_record(output_path, meta, arrays)
        ticket = {
            "schema_version": "cmi-v4-world-ticket-v1",
            "task": asdict(task),
            "record": record,
            "worker_pid": os.getpid(),
            "wall_seconds": time.monotonic() - started,
            "role": "CMI_V4_FORMAL_RAW_WORLD",
        }
        atomic_json(ticket_path(output_path, task.index), ticket)
        return ticket
    except Exception:
        atomic_json(
            output_path / "worker_errors" / f"{task.index:04d}.json",
            {
                "schema_version": "cmi-v4-worker-error-v1",
                "task": asdict(task),
                "traceback": traceback.format_exc(),
            },
        )
        raise


def checkpoint(
    output: Path,
    tasks: list[WorldTask],
    records: list[dict[str, Any]],
    elapsed: float,
) -> list[dict[str, Any]]:
    committed = list(records)
    while len(committed) < len(tasks):
        ticket = validate_ticket(output, tasks[len(committed)])
        if ticket is None:
            break
        committed.append(ticket["record"])
    atomic_json(
        output / "latest.json",
        {
            "schema_version": "conditional-history-mc-q-checkpoint-v3",
            "generation": len(committed),
            "completed": [row["world_identity"] for row in committed],
            "target": len(tasks),
            "records": committed,
            "total_branch_policy_steps": int(
                sum(row["total_branch_policy_steps"] for row in committed)
            ),
            "seconds": elapsed,
            "execution_version": "WORLD_PROCESS_COORDINATOR_V4",
        },
    )
    return committed


def write_status(
    output: Path,
    *,
    state: str,
    committed: int,
    target: int,
    staged: int,
    active: int,
    accessed: int,
    seconds: float,
) -> None:
    atomic_json(
        output / "status.json",
        {
            "status": state,
            "pid": os.getpid(),
            "completed": committed,
            "target": target,
            "generation": committed,
            "staged": staged,
            "active": active,
            "accessed": accessed,
            "seconds": seconds,
            "world_processes": WORLD_PROCESSES,
            "workers_per_world": WORKERS_PER_WORLD,
            "max_worlds_per_child": MAX_WORLDS_PER_CHILD,
            "execution_version": "WORLD_PROCESS_COORDINATOR_V4",
            "v1_v2_outcomes_used": False,
            "automatic_analysis": False,
            "automatic_confirmation": False,
            "automatic_training": False,
        },
    )


def run_collection(args: argparse.Namespace) -> int:
    contract_path = args.contract.resolve()
    contract = json.loads(contract_path.read_text())
    receipt = json.loads(args.freeze_receipt.resolve().read_text())
    validation = validate_contract(contract, contract_path.parent, receipt)
    if not validation["CMI_V4_DEV_READY"]:
        raise ValueError(
            "CMI_V4_DEV_READY failed: "
            + "; ".join(validation["errors"] + validation["receipt_errors"])
        )
    pre = contract["pre_h"]
    identities = list(pre["master_splits"][args.split])
    tasks = [
        WorldTask(index, identity, read_world_seed(contract, identity))
        for index, identity in enumerate(identities)
    ]
    replicates = int(pre["resamples_per_anchor_action"][args.split])
    output = args.output_dir.resolve()
    run_id = f"{pre['contract_id']}:{args.split}:V4_PARALLEL_RAW_V1"
    manifest = {
        "schema_version": "conditional-history-mc-q-collection-v4",
        "run_id": run_id,
        "contract_record_hash": pre["record_sha256"],
        "split": args.split,
        "world_identities": identities,
        "anchor_steps": list(pre["anchor_steps"]),
        "history_length": pre["history_length"],
        "history_stride": pre["history_stride"],
        "evaluation_resamples_per_anchor_action": replicates,
        "world_processes": WORLD_PROCESSES,
        "workers_per_world": WORKERS_PER_WORLD,
        "max_worlds_per_child": MAX_WORLDS_PER_CHILD,
        "device": args.device,
        "v1_v2_outcomes_used": False,
        "automatic_analysis": False,
        "automatic_confirmation": False,
        "automatic_training": False,
    }
    if args.resume:
        if json.loads((output / "manifest.json").read_text()) != manifest:
            raise ValueError("CMI-v4 resume manifest mismatch")
        latest = json.loads((output / "latest.json").read_text())
        records = list(latest["records"])
        elapsed_before = float(latest["seconds"])
        if [row["world_identity"] for row in records] != identities[: len(records)]:
            raise ValueError("CMI-v4 committed checkpoint is not an ordered prefix")
    else:
        if output.exists():
            raise FileExistsError(output)
        output.mkdir(parents=True)
        atomic_json(output / "manifest.json", manifest)
        records, elapsed_before = [], 0.0
        records = checkpoint(output, tasks, records, elapsed_before)

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    started = time.monotonic()

    def elapsed() -> float:
        return elapsed_before + time.monotonic() - started

    def pause_requested() -> bool:
        return STOP or (output / "PAUSE").exists()

    # Validate every durable ticket before scheduling.  Tickets outside the
    # committed prefix are intentionally retained and committed only in order.
    for task in tasks:
        validate_ticket(output, task)
    records = checkpoint(output, tasks, records, elapsed())
    futures: dict[Future[dict[str, Any] | None], WorldTask] = {}
    active_indices: set[int] = set()
    context = mp.get_context("spawn")

    def access_count() -> int:
        if args.split == "SMOKE_DEBUG":
            return len(tasks)
        current = json.loads(contract_path.read_text())
        return len(
            [
                event
                for event in current["access_registry"]["events"]
                if event.get("event_type") == "ACCESS" and event.get("split") == args.split
            ]
        )

    def staged_count() -> int:
        return sum(validate_ticket(output, task) is not None for task in tasks)

    write_status(
        output, state="RUNNING", committed=len(records), target=len(tasks),
        staged=staged_count(), active=0, accessed=access_count(), seconds=elapsed(),
    )
    try:
        with ProcessPoolExecutor(
            max_workers=WORLD_PROCESSES,
            mp_context=context,
            initializer=worker_init,
            initargs=(args.device,),
            max_tasks_per_child=MAX_WORLDS_PER_CHILD,
        ) as pool:
            while len(records) < len(tasks):
                records = checkpoint(output, tasks, records, elapsed())
                if args.stop_after_worlds > 0 and len(records) >= args.stop_after_worlds:
                    break
                if not pause_requested():
                    for task in tasks:
                        if len(futures) >= WORLD_PROCESSES:
                            break
                        if task.index < len(records) or task.index in active_indices:
                            continue
                        if validate_ticket(output, task) is not None:
                            continue
                        accessed = access_count()
                        if task.index > accessed:
                            break
                        if task.index == accessed:
                            if args.split == "CMI_V3_DEV":
                                append_access_event(contract_path, args.split, task.identity, run_id)
                            # SMOKE_DEBUG has no formal access event.
                        futures[
                            pool.submit(
                                collect_task,
                                task,
                                output=str(output),
                                split=args.split,
                                contract_id=pre["contract_id"],
                                replicates=replicates,
                                guard=int(pre["branch_guard"]),
                            )
                        ] = task
                        active_indices.add(task.index)
                write_status(
                    output, state="RUNNING", committed=len(records), target=len(tasks),
                    staged=staged_count(), active=len(futures), accessed=access_count(),
                    seconds=elapsed(),
                )
                if not futures:
                    break
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    task = futures.pop(future)
                    active_indices.remove(task.index)
                    future.result()
                    records = checkpoint(output, tasks, records, elapsed())
                    print(
                        f"finished task {task.index}; committed {len(records)}/{len(tasks)}",
                        flush=True,
                    )

        records = checkpoint(output, tasks, records, elapsed())
        complete = len(records) == len(tasks)
        if complete and args.split == "CMI_V3_DEV":
            state = "COMPLETE_AWAITING_CMI_V3_DEV_ANALYSIS"
        elif complete:
            state = "COMPLETE_SMOKE_DEBUG"
        else:
            state = "PAUSED_RESUMABLE"
        write_status(
            output, state=state, committed=len(records), target=len(tasks),
            staged=staged_count(), active=0, accessed=access_count(), seconds=elapsed(),
        )
        if complete:
            atomic_json(
                output / "results.json",
                {
                    "schema_version": "conditional-history-mc-q-run-results-v3",
                    "records": records,
                    "completed": len(records),
                    "target": len(tasks),
                    "total_branch_policy_steps": int(
                        sum(row["total_branch_policy_steps"] for row in records)
                    ),
                    "role": "CMI_V4_DEV_RAW_MC_Q_NO_INFORMATION_VERDICT",
                    "execution_version": "WORLD_PROCESS_COORDINATOR_V4",
                    "v1_v2_outcomes_used": False,
                    "automatic_analysis": False,
                    "automatic_confirmation": False,
                    "automatic_training": False,
                },
            )
        return 0
    except Exception:
        atomic_json(
            output / "ERROR.json",
            {
                "error": traceback.format_exc(),
                "completed": len(records),
                "target": len(tasks),
                "automatic_analysis": False,
                "automatic_confirmation": False,
                "automatic_training": False,
            },
        )
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--split", choices=("SMOKE_DEBUG", "CMI_V3_DEV"), required=True)
    parser.add_argument("--freeze-receipt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-worlds", type=int, default=0)
    return run_collection(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
