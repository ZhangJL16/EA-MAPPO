#!/usr/bin/env python3
"""Resumable raw CMI-DEV collection with paired downstream model-access resampling."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import sys
import tempfile
import time
import traceback
from collections import deque
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.directional_navigation.conditional_model_access import ConditionalModelAccessBranch
from scripts.run_dvoi_h_collection import (
    ANCHOR_STEPS,
    HISTORY_LENGTH,
    HISTORY_STRIDE,
    REGISTERED_BRANCH_TERMINATIONS,
    FrozenNavigator,
    arrays_from_histories,
    eligible,
    history_row,
    read_world_seed,
    select_history,
)
from scripts.validate_conditional_history_contract import append_access_event, validate_contract
from scripts.validate_identification_contract import file_hash, object_hash


STOP = False


def request_stop(*_: object) -> None:
    global STOP
    STOP = True


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def resample_seed(contract_id: str, split: str, identity: str, anchor_step: int, block: str, replicate: int) -> int:
    payload = f"{contract_id}|{split}|{identity}|{anchor_step}|{block}|{replicate}|cmi-resample-v1"
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big") & ((1 << 63) - 1)


def partial_path(root: Path, identity: str, anchor_step: int, action: str, replicate: int) -> Path:
    stem = identity.removeprefix("sha256:")
    return root / "partials" / stem / f"anchor_{anchor_step}" / action / f"replicate_{replicate:03d}.json"


def validate_resample_row(
    row: dict[str, Any], *, identity: str, anchor_step: int, action: str,
    replicate: int, disturbance_seed: int, guard: int,
) -> None:
    expected_fields = {
        "schema_version", "world_identity", "anchor_step", "block", "replicate",
        "action", "disturbance_seed", "policy_steps", "outcome",
        "execution_error_rms_norm", "execution_error_max_norm",
    }
    if not isinstance(row, dict) or set(row) != expected_fields:
        raise ValueError("partial CMI trajectory has an invalid exact schema")
    expected = {
        "schema_version": "conditional-history-resample-v1",
        "world_identity": identity,
        "anchor_step": anchor_step,
        "block": "evaluation",
        "replicate": replicate,
        "action": action,
        "disturbance_seed": disturbance_seed,
    }
    if any(row.get(name) != value for name, value in expected.items()):
        raise ValueError("partial CMI trajectory differs from the frozen key")
    steps = row.get("policy_steps")
    if not isinstance(steps, int) or isinstance(steps, bool) or not 1 <= steps <= guard:
        raise ValueError("partial CMI trajectory has invalid policy-step accounting")
    outcome = row.get("outcome")
    outcome_fields = {
        "termination", "returned", "collision_count", "collision_free_arrival",
        "task_increment", "initial_energy", "remaining_energy", "operational_failure",
        "utility",
    }
    if not isinstance(outcome, dict) or set(outcome) != outcome_fields:
        raise ValueError("partial CMI outcome has an invalid exact schema")
    if outcome.get("termination") not in REGISTERED_BRANCH_TERMINATIONS:
        raise ValueError("partial CMI outcome has an unregistered terminal")
    returned = outcome.get("returned")
    failure = outcome.get("operational_failure")
    collisions = outcome.get("collision_count")
    task_increment = outcome.get("task_increment")
    if not isinstance(returned, bool) or not isinstance(failure, bool) or failure == returned:
        raise ValueError("partial CMI outcome has inconsistent return/failure accounting")
    if not isinstance(collisions, int) or isinstance(collisions, bool) or not 0 <= collisions <= steps:
        raise ValueError("partial CMI outcome has invalid unified collision accounting")
    if not isinstance(task_increment, int) or isinstance(task_increment, bool) or task_increment < 0:
        raise ValueError("partial CMI outcome has invalid task increment")
    if outcome.get("collision_free_arrival") is not (returned and collisions == 0):
        raise ValueError("partial CMI collision-free arrival is not derived from unified collisions")
    expected_utility = float(task_increment - 2.0 * failure - 0.25 * (collisions > 0))
    if outcome.get("utility") != expected_utility:
        raise ValueError("partial CMI outcome utility differs from the frozen rule")
    numeric = (
        outcome.get("initial_energy"), outcome.get("remaining_energy"),
        row.get("execution_error_rms_norm"), row.get("execution_error_max_norm"),
    )
    if any(not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) for value in numeric):
        raise ValueError("partial CMI trajectory contains non-finite numeric evidence")
    disturbance_bound = math.sqrt(3.0) * 0.12 + 1e-6
    if not 0 <= row["execution_error_rms_norm"] <= disturbance_bound or not 0 <= row["execution_error_max_norm"] <= disturbance_bound:
        raise ValueError("partial CMI trajectory exceeds the frozen disturbance bound")


def run_jobs(
    *,
    root: Path,
    identity: str,
    split: str,
    contract_id: str,
    anchor_step: int,
    snapshot: bytes,
    navigator: FrozenNavigator,
    worker_count: int,
    replicates: int,
    guard: int,
    obstacles: int,
    pause_requested: Callable[[], bool],
) -> tuple[list[dict[str, Any]], bool]:
    jobs = [(action, replicate) for replicate in range(replicates) for action in ("R", "C")]
    completed: dict[tuple[str, int], dict[str, Any]] = {}
    for action, replicate in jobs:
        path = partial_path(root, identity, anchor_step, action, replicate)
        if path.exists():
            row = json.loads(path.read_text())
            expected = resample_seed(contract_id, split, identity, anchor_step, "evaluation", replicate)
            validate_resample_row(
                row, identity=identity, anchor_step=anchor_step, action=action,
                replicate=replicate, disturbance_seed=expected, guard=guard,
            )
            completed[(action, replicate)] = row
    pending = deque(job for job in jobs if job not in completed)
    if not pending:
        return [completed[job] for job in jobs], True
    workers = [ConditionalModelAccessBranch(obstacles=obstacles, guard=guard) for _ in range(min(worker_count, len(pending)))]
    active: dict[int, tuple[str, int, int, dict[str, np.ndarray], int]] = {}

    def assign(indices: list[int]) -> None:
        for index in indices:
            if not pending or pause_requested():
                active.pop(index, None)
                continue
            action, replicate = pending.popleft()
            seed = resample_seed(contract_id, split, identity, anchor_step, "evaluation", replicate)
            env = workers[index]
            env.restore(snapshot)
            env.arm_resample(seed)
            env.begin_branch(action)
            active[index] = (action, replicate, seed, env.observation(), 0)

    try:
        assign(list(range(len(workers))))
        while active:
            indices = sorted(active)
            selected = np.stack(
                [
                    active[index][3]["return"]
                    if workers[index].base.mode.name == "CHARGER_COMMITTED"
                    else active[index][3]["nav"]
                    for index in indices
                ]
            )
            actions, _ = navigator.model.predict(selected, deterministic=True)
            finished: list[int] = []
            for position, index in enumerate(indices):
                action_name, replicate, seed, _, steps = active[index]
                observation, terminal, done = workers[index].step_policy(actions[position])
                steps += 1
                active[index] = (action_name, replicate, seed, observation, steps)
                if not done:
                    continue
                termination = str(terminal["termination"])
                if termination not in REGISTERED_BRANCH_TERMINATIONS:
                    raise RuntimeError(f"unregistered CMI branch terminal: {termination}")
                outcome = workers[index].branch_outcome(terminal)
                row = {
                    "schema_version": "conditional-history-resample-v1",
                    "world_identity": identity,
                    "anchor_step": anchor_step,
                    "block": "evaluation",
                    "replicate": replicate,
                    "action": action_name,
                    "disturbance_seed": seed,
                    "policy_steps": steps,
                    "outcome": outcome,
                    **workers[index].disturbance_summary(steps),
                }
                validate_resample_row(
                    row, identity=identity, anchor_step=anchor_step,
                    action=action_name, replicate=replicate,
                    disturbance_seed=seed, guard=guard,
                )
                atomic_json(partial_path(root, identity, anchor_step, action_name, replicate), row)
                completed[(action_name, replicate)] = row
                finished.append(index)
            assign(finished)
        complete = len(completed) == len(jobs)
        return [completed[job] for job in jobs if job in completed], complete
    finally:
        for worker in workers:
            worker.close()


def collect_world(
    *,
    output: Path,
    navigator: FrozenNavigator,
    world_seed: int,
    identity: str,
    split: str,
    contract_id: str,
    replicates: int,
    workers: int,
    obstacles: int,
    guard: int,
    pause_requested: Callable[[], bool],
) -> tuple[dict[str, Any] | None, dict[str, np.ndarray] | None]:
    env = ConditionalModelAccessBranch(obstacles=obstacles, guard=guard)
    histories: list[list[dict[str, Any]]] = []
    anchors: list[dict[str, Any]] = []
    try:
        for anchor_step in ANCHOR_STEPS:
            observation = env.reset_world(world_seed, soc=1.0)
            if object_hash(env.physical_world_record()) != identity:
                raise RuntimeError("generated CMI physical world differs from frozen identity")
            ring: deque[dict[str, Any]] = deque(maxlen=HISTORY_LENGTH * HISTORY_STRIDE + 1)
            ring.append(history_row(env, observation, previous=None))
            early: dict[str, Any] | None = None
            for _ in range(anchor_step):
                action = navigator.goal_action(observation, returning=False)
                observation, transition, done = env.step_policy(action)
                ring.append(history_row(env, observation, previous=transition))
                if done:
                    early = transition
                    break
            if early is not None:
                anchors.append({"anchor_step": anchor_step, "included": False, "eligibility": early["termination"]})
                continue
            ok, reason = eligible(env, observation)
            if not ok:
                anchors.append({"anchor_step": anchor_step, "included": False, "eligibility": reason})
                continue
            history = select_history(ring)
            snapshot = env.snapshot()
            rows, complete = run_jobs(
                root=output,
                identity=identity,
                split=split,
                contract_id=contract_id,
                anchor_step=anchor_step,
                snapshot=snapshot,
                navigator=navigator,
                worker_count=workers,
                replicates=replicates,
                guard=guard,
                obstacles=obstacles,
                pause_requested=pause_requested,
            )
            if not complete:
                return None, None
            anchors.append(
                {
                    "anchor_step": anchor_step,
                    "included": True,
                    "eligibility": "ELIGIBLE",
                    "history_index": len(histories),
                    "snapshot_sha256": hashlib.sha256(snapshot).hexdigest(),
                    "battery": float(observation["battery"][0]),
                    "distance_to_charger": float(history[-1]["distance_to_charger"]),
                    "task_progress": int(env.base.tasks_completed),
                    "resamples": rows,
                }
            )
            histories.append(history)
            if pause_requested():
                return None, None
        if not histories:
            raise RuntimeError("CMI world supplied no eligible legal histories")
        arrays = arrays_from_histories(histories)
        meta = {
            "schema_version": "conditional-history-world-result-v1",
            "role": "CMI_DEVELOPMENT_RAW_NO_INFORMATION_VERDICT",
            "world_identity": identity,
            "world_seed": world_seed,
            "anchors": anchors,
            "eligible_anchor_count": len(histories),
            "evaluation_resamples_per_action": replicates,
            "total_branch_policy_steps": int(sum(row["policy_steps"] for anchor in anchors if anchor.get("included") for row in anchor["resamples"])),
            "history_disturbance_enabled": False,
            "simulator_geometry_actor_input": False,
            "collision_statistics": "ONE_UNIFIED_COUNT_PER_POLICY_STEP",
            "automatic_analysis": False,
            "automatic_training": False,
        }
        return meta, arrays
    finally:
        env.close()


def write_record(output: Path, meta: dict[str, Any], arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    records = output / "records"
    records.mkdir(parents=True, exist_ok=True)
    stem = meta["world_identity"].removeprefix("sha256:")
    temporary = records / f"{stem}.tmp.npz"
    npz = records / f"{stem}.npz"
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, npz)
    saved = dict(meta)
    saved["npz"] = npz.name
    saved["npz_sha256"] = file_hash(npz).removeprefix("sha256:")
    json_path = records / f"{stem}.json"
    atomic_json(json_path, saved)
    return {
        "world_identity": saved["world_identity"],
        "json": json_path.name,
        "json_sha256": file_hash(json_path).removeprefix("sha256:"),
        "npz": npz.name,
        "npz_sha256": saved["npz_sha256"],
        "eligible_anchor_count": saved["eligible_anchor_count"],
        "total_branch_policy_steps": saved["total_branch_policy_steps"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--split", choices=("SMOKE_DEBUG", "CMI_DEV"), required=True)
    parser.add_argument("--freeze-receipt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-worlds", type=int, default=0)
    args = parser.parse_args()
    if args.workers <= 0:
        raise ValueError("CMI worker count must be positive")
    contract_path = args.contract.resolve()
    contract = json.loads(contract_path.read_text())
    receipt = json.loads(args.freeze_receipt.resolve().read_text())
    validation = validate_contract(contract, contract_path.parent, receipt)
    if not validation["CMI_DEV_READY"]:
        raise ValueError("CMI_DEV_READY failed: " + "; ".join(validation["errors"] + validation["receipt_errors"]))
    pre = contract["pre_h"]
    identities = pre["master_splits"][args.split]
    replicates = pre["resamples"][args.split]["evaluation"]
    output = args.output_dir.resolve()
    manifest = {
        "schema_version": "conditional-history-collection-run-v1",
        "run_id": f"{pre['contract_id']}:{args.split}:RAW_V1",
        "contract_record_hash": pre["record_sha256"],
        "split": args.split,
        "world_identities": identities,
        "anchor_steps": list(ANCHOR_STEPS),
        "history_length": HISTORY_LENGTH,
        "history_stride": HISTORY_STRIDE,
        "evaluation_resamples_per_action": replicates,
        "workers": args.workers,
        "device": args.device,
        "automatic_analysis": False,
        "automatic_training": False,
    }
    manifest_path = output / "manifest.json"
    if args.resume:
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("CMI resume manifest mismatch")
        latest = json.loads((output / "latest.json").read_text())
        completed = list(latest["completed"])
        records = list(latest["records"])
        generation = int(latest["generation"])
        total_steps = int(latest["total_branch_policy_steps"])
        elapsed_before = float(latest["seconds"])
    else:
        if output.exists():
            raise FileExistsError(output)
        output.mkdir(parents=True)
        atomic_json(manifest_path, manifest)
        completed, records, generation, total_steps, elapsed_before = [], [], 0, 0, 0.0
        atomic_json(output / "latest.json", {
            "schema_version": "conditional-history-checkpoint-v1",
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
            if args.split == "CMI_DEV":
                append_access_event(contract_path, args.split, identity, manifest["run_id"])
            seed = read_world_seed(json.loads(contract_path.read_text()), identity)
            meta, arrays = collect_world(
                output=output, navigator=navigator, world_seed=seed, identity=identity,
                split=args.split, contract_id=pre["contract_id"], replicates=replicates,
                workers=args.workers, obstacles=48, guard=pre["branch_guard"],
                pause_requested=pause_requested,
            )
            if meta is None or arrays is None:
                break
            record = write_record(output, meta, arrays)
            completed.append(identity)
            records.append(record)
            generation += 1
            total_steps += int(record["total_branch_policy_steps"])
            elapsed = elapsed_before + time.monotonic() - started
            latest = {
                "schema_version": "conditional-history-checkpoint-v1",
                "generation": generation, "completed": completed, "target": len(identities),
                "records": records, "total_branch_policy_steps": total_steps,
                "seconds": elapsed,
            }
            atomic_json(output / "latest.json", latest)
            atomic_json(output / "status.json", {
                "status": "RUNNING", "pid": os.getpid(), "completed": len(completed),
                "target": len(identities), "generation": generation,
                "total_branch_policy_steps": total_steps, "seconds": elapsed,
                "automatic_analysis": False, "automatic_training": False,
            })
            print(f"checkpoint {generation}: {len(completed)}/{len(identities)} worlds, branch_steps={record['total_branch_policy_steps']}", flush=True)
            if pause_requested() or (args.stop_after_worlds > 0 and len(completed) >= args.stop_after_worlds):
                break
        complete = len(completed) == len(identities)
        state = "COMPLETE_AWAITING_CMI_DEV_ANALYSIS" if complete else "PAUSED_RESUMABLE"
        status = json.loads((output / "status.json").read_text())
        status["status"] = state
        atomic_json(output / "status.json", status)
        if complete:
            atomic_json(output / "results.json", {
                "schema_version": "conditional-history-run-results-v1",
                "records": records, "completed": len(completed), "target": len(identities),
                "total_branch_policy_steps": total_steps,
                "role": "CMI_DEVELOPMENT_RAW_NO_INFORMATION_VERDICT",
                "automatic_analysis": False, "automatic_training": False,
            })
        return 0
    except Exception:
        atomic_json(output / "ERROR.json", {
            "error": traceback.format_exc(), "completed": len(completed), "target": len(identities),
            "automatic_analysis": False, "automatic_training": False,
        })
        raise


if __name__ == "__main__":
    raise SystemExit(main())
