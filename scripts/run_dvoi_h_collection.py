#!/usr/bin/env python3
"""Run the fresh, resumable Gate-H paired C/R collection.

The pre-H contract assigns every physical world before this runner is called.
For each assigned world, the runner records an ACCESS event before reset,
verifies the generated physical realization, samples the two registered anchor
times using legal pre-decision information only, and executes paired immediate-R
and one-task-then-R branches from the same snapshot.  It performs no fitting,
analysis, verdict generation, training, or automatic promotion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import sys
import tempfile
import time
import traceback
from collections import deque
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from stable_baselines3 import SAC

from experiments.directional_navigation.battery_sortie import CAPACITY
from experiments.directional_navigation.dvoi_h_branch import DVOIHBranch
from review_bundle.safety.switching.commitment import SortieMode
from scripts.validate_identification_contract import (
    access_event_hash,
    file_hash,
    object_hash,
    record_hash,
    seal_access_registry,
    validate_contract,
    validate_content_registry,
    validate_registry,
)


ANCHOR_STEPS = (256, 768)
HISTORY_LENGTH = 64
HISTORY_STRIDE = 4
MIN_BATTERY_FRACTION = 0.25
REGISTERED_BRANCH_TERMINATIONS = frozenset(
    {"returned", "energy_exhausted", "return_deadline", "branch_guard"}
)
STOP = False


def request_stop(*_: object) -> None:
    global STOP
    STOP = True


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


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


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FrozenNavigator:
    def __init__(self, device: str):
        repair_path = ROOT / "artifacts/new_navigation_energy_global_scale_repair_20260908_v1/repair.json"
        repair = json.loads(repair_path.read_text())
        if sha256_file(Path(repair["head_checkpoint"])) != repair["head_sha256"]:
            raise RuntimeError("frozen energy-head checkpoint hash mismatch")
        if sha256_file(Path(repair["features"])) != repair["features_sha256"]:
            raise RuntimeError("frozen feature-cache hash mismatch")
        state = torch.load(repair["head_checkpoint"], weights_only=False, map_location="cpu")
        cache = torch.load(repair["features"], weights_only=False, map_location="cpu")
        if state["contract"] != cache["contract"]:
            raise RuntimeError("frozen navigation/feature contracts differ")
        model_path = Path(state["contract"]["model"])
        if sha256_file(model_path) != state["contract"]["model_sha256"]:
            raise RuntimeError("frozen navigation checkpoint hash mismatch")
        self.model = SAC.load(model_path, device=device)
        self.model.policy.set_training_mode(False)

    def goal_action(self, observation: dict[str, np.ndarray], *, returning: bool) -> np.ndarray:
        selected = observation["return"] if returning else observation["nav"]
        action, _ = self.model.predict(selected, deterministic=True)
        return np.asarray(action, np.float32)


def history_row(
    env: DVOIHBranch,
    observation: dict[str, np.ndarray],
    *,
    previous: dict[str, Any] | None,
) -> dict[str, Any]:
    base = env.base
    zeros = np.zeros(3, np.float32)
    return {
        "step": int(env.total_steps),
        "nav_observation": np.asarray(observation["nav"], np.float32),
        "return_observation": np.asarray(observation["return"], np.float32),
        "battery": float(observation["battery"][0]),
        "distance_to_charger": float(np.linalg.norm(base.agent.pos - base.charger_position)),
        "task_progress": int(base.tasks_completed),
        "task_clock": int(base.steps_in_current_task),
        "previous_nominal_action": zeros if previous is None else previous["nominal_action"],
        "previous_executed_action": zeros if previous is None else previous["executed_action"],
        "previous_realized_acceleration": zeros if previous is None else previous["realized_acceleration"],
        "previous_action_valid": previous is not None,
        "previous_contact": False if previous is None else bool(previous["contact"]),
    }


def select_history(rows: deque[dict[str, Any]]) -> list[dict[str, Any]]:
    values = list(rows)
    selected = values[-1 :: -HISTORY_STRIDE][:HISTORY_LENGTH][::-1]
    if len(selected) != HISTORY_LENGTH:
        raise RuntimeError(f"registered history length unavailable: {len(selected)}")
    return selected


def eligible(env: DVOIHBranch, observation: dict[str, np.ndarray]) -> tuple[bool, str]:
    if float(observation["at_home"][0]) > 0.5:
        return False, "AT_CHARGER"
    if float(observation["battery"][0]) < MIN_BATTERY_FRACTION * CAPACITY:
        return False, "BATTERY_BELOW_REGISTERED_MINIMUM"
    if env.base.mode is not SortieMode.TASK:
        return False, "NOT_TASK_MODE"
    return True, "ELIGIBLE"


def run_branch(
    env: DVOIHBranch,
    navigator: FrozenNavigator,
    snapshot: bytes,
    mode: str,
) -> tuple[dict[str, Any], int]:
    env.restore(snapshot)
    env.begin_branch(mode)
    observation = env.observation()
    terminal: dict[str, Any] | None = None
    steps = 0
    for _ in range(env.guard + 1):
        returning = env.base.mode is SortieMode.CHARGER_COMMITTED
        action = navigator.goal_action(observation, returning=returning)
        observation, terminal, done = env.step_policy(action)
        steps += 1
        if done:
            break
    if terminal is None or not done:
        raise RuntimeError(f"branch {mode} did not terminate under registered guard")
    termination = str(terminal["termination"])
    if termination not in REGISTERED_BRANCH_TERMINATIONS:
        raise RuntimeError(
            f"branch {mode} reached unregistered terminal reason: {termination}"
        )
    return env.branch_outcome(terminal), steps


def arrays_from_histories(histories: list[list[dict[str, Any]]]) -> dict[str, np.ndarray]:
    fields = (
        "nav_observation",
        "return_observation",
        "battery",
        "distance_to_charger",
        "task_progress",
        "task_clock",
        "step",
        "previous_nominal_action",
        "previous_executed_action",
        "previous_realized_acceleration",
        "previous_action_valid",
        "previous_contact",
    )
    arrays: dict[str, np.ndarray] = {}
    for field in fields:
        arrays[field] = np.asarray([[row[field] for row in history] for history in histories])
    return arrays


def collect_world(
    navigator: FrozenNavigator,
    *,
    world_seed: int,
    expected_world_identity: str,
    obstacles: int,
    guard: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    env = DVOIHBranch(obstacles=obstacles, guard=guard)
    histories: list[list[dict[str, Any]]] = []
    anchors: list[dict[str, Any]] = []
    physical_identity: str | None = None
    total_branch_steps = 0
    try:
        for anchor_step in ANCHOR_STEPS:
            observation = env.reset_world(world_seed, soc=1.0)
            observed_identity = object_hash(env.physical_world_record())
            if physical_identity is None:
                physical_identity = observed_identity
            if observed_identity != physical_identity or observed_identity != expected_world_identity:
                raise RuntimeError("generated physical world differs from the blind master identity")
            ring: deque[dict[str, Any]] = deque(maxlen=HISTORY_LENGTH * HISTORY_STRIDE + 1)
            ring.append(history_row(env, observation, previous=None))
            terminal_early = None
            for _ in range(anchor_step):
                action = navigator.goal_action(observation, returning=False)
                observation, transition, done = env.step_policy(action)
                ring.append(history_row(env, observation, previous=transition))
                if done:
                    terminal_early = transition["termination"]
                    break
            ok, reason = eligible(env, observation) if terminal_early is None else (False, terminal_early)
            if not ok:
                anchors.append(
                    {
                        "anchor_step": anchor_step,
                        "eligibility": reason,
                        "included": False,
                    }
                )
                continue
            history = select_history(ring)
            snapshot = env.snapshot()
            anchor_meta = {
                "anchor_step": anchor_step,
                "eligibility": reason,
                "included": True,
                "battery": float(observation["battery"][0]),
                "distance_to_charger": float(history[-1]["distance_to_charger"]),
                "task_progress": int(env.base.tasks_completed),
                "history_index": len(histories),
            }
            r_outcome, r_steps = run_branch(env, navigator, snapshot, "R")
            c_outcome, c_steps = run_branch(env, navigator, snapshot, "C")
            total_branch_steps += r_steps + c_steps
            anchor_meta["R"] = r_outcome
            anchor_meta["C"] = c_outcome
            anchor_meta["paired_snapshot_sha256"] = hashlib.sha256(snapshot).hexdigest()
            anchors.append(anchor_meta)
            histories.append(history)
        if not histories:
            raise RuntimeError("world produced no eligible registered anchors")
        meta = {
            "schema_version": "dvoi-h-world-result-v1",
            "world_identity": expected_world_identity,
            "world_seed": world_seed,
            "anchors": anchors,
            "eligible_anchor_count": len(histories),
            "total_branch_policy_steps": total_branch_steps,
            "collision_statistics": "ONE_UNIFIED_COUNT_PER_POLICY_STEP",
            "automatic_analysis": False,
            "automatic_training": False,
        }
        return meta, arrays_from_histories(histories)
    finally:
        env.close()


def read_world_seed(contract: dict[str, Any], identity: str) -> int:
    world = contract["world_registry"][identity]
    digest = world["world_seed_or_parameters_hash"]
    entry = contract["content_registry"][digest]
    value = entry.get("inline")
    if not isinstance(value, dict) or set(value) != {"schema_version", "world_seed"}:
        raise ValueError("world seed artifact has an invalid schema")
    return int(value["world_seed"])


def append_access_event(contract_path: Path, split: str, identity: str, run_id: str) -> str:
    contract = json.loads(contract_path.read_text())
    registered, content_errors = validate_content_registry(contract, contract_path.parent)
    _, chain_errors = validate_registry(contract, registered)
    if content_errors or chain_errors:
        raise ValueError(
            "refusing to append to an invalid contract: "
            + "; ".join(content_errors + chain_errors)
        )
    pre_h = contract["pre_h"]
    if pre_h.get("record_sha256") != record_hash(pre_h):
        raise ValueError("pre-H record hash is invalid")
    if identity not in pre_h["master_splits"].get(split, []):
        raise ValueError(f"world is not assigned to {split}")
    artifact = {
        "schema_version": "dvoi-world-access-v1",
        "run_id": run_id,
        "split": split,
        "world_identity": identity,
    }
    artifact_hash = object_hash(artifact)
    contract["content_registry"][artifact_hash] = {"inline": artifact}
    event_id = f"access-{split.lower()}-{identity.removeprefix('sha256:')[:24]}"
    existing = [event for event in contract["access_registry"]["events"] if event.get("event_id") == event_id]
    if existing:
        event = existing[0]
        if (
            event.get("event_type") != "ACCESS"
            or event.get("split") != split
            or event.get("world_identity") != identity
            or event.get("access_artifact_hash") != artifact_hash
        ):
            raise ValueError("existing deterministic ACCESS event differs")
        return artifact_hash
    sequence = max(
        (int(event["sequence"]) for event in contract["access_registry"]["events"]),
        default=0,
    ) + 1
    contract["access_registry"]["events"].append(
        {
            "sequence": sequence,
            "event_id": event_id,
            "event_type": "ACCESS",
            "split": split,
            "world_identity": identity,
            "access_artifact_hash": artifact_hash,
        }
    )
    seal_access_registry(contract)
    atomic_json(contract_path, contract)
    return artifact_hash


def write_world_record(output_dir: Path, meta: dict[str, Any], arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    records = output_dir / "records"
    records.mkdir(parents=True, exist_ok=True)
    stem = meta["world_identity"].removeprefix("sha256:")
    npz_path = records / f"{stem}.npz"
    temp_path = records / f"{stem}.tmp.npz"
    np.savez_compressed(temp_path, **arrays)
    os.replace(temp_path, npz_path)
    meta = dict(meta)
    meta["npz"] = npz_path.name
    meta["npz_sha256"] = file_hash(npz_path).removeprefix("sha256:")
    json_path = records / f"{stem}.json"
    atomic_json(json_path, meta)
    return {
        "world_identity": meta["world_identity"],
        "json": json_path.name,
        "json_sha256": file_hash(json_path).removeprefix("sha256:"),
        "npz": npz_path.name,
        "npz_sha256": meta["npz_sha256"],
        "eligible_anchor_count": meta["eligible_anchor_count"],
        "total_branch_policy_steps": meta["total_branch_policy_steps"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--split", choices=("SMOKE_DEBUG", "H_DEV", "H_CONFIRM"), required=True)
    parser.add_argument("--access-anchor-receipt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--obstacles", type=int, default=48)
    parser.add_argument("--guard", type=int, default=12_000)
    parser.add_argument("--stop-after-worlds", type=int, default=0)
    args = parser.parse_args()

    contract_path = args.contract.resolve()
    contract = json.loads(contract_path.read_text())
    receipt = json.loads(args.access_anchor_receipt.resolve().read_text())
    required_stage = "H_CONFIRM_READY" if args.split == "H_CONFIRM" else "PRE_H_READY"
    readiness = validate_contract(contract, contract_path.parent, receipt)
    if not readiness["readiness"][required_stage]:
        raise ValueError(
            f"refusing {args.split} access because {required_stage} failed: "
            + "; ".join(readiness["errors"][required_stage])
        )
    identities = list(contract["pre_h"]["master_splits"][args.split])
    run_id = f"{contract['pre_h']['contract_id']}:{args.split}"
    manifest = {
        "schema_version": "dvoi-h-collection-run-v1",
        "run_id": run_id,
        "pre_h_record_hash": contract["pre_h"]["record_sha256"],
        "split": args.split,
        "split_world_manifest_hash": contract["pre_h"][{
            "SMOKE_DEBUG": "smoke_debug_world_manifest_hash",
            "H_DEV": "h_dev_world_manifest_hash",
            "H_CONFIRM": "h_confirm_world_manifest_hash",
        }[args.split]],
        "world_identities": identities,
        "anchor_steps": list(ANCHOR_STEPS),
        "history_length": HISTORY_LENGTH,
        "history_stride": HISTORY_STRIDE,
        "minimum_battery_fraction": MIN_BATTERY_FRACTION,
        "obstacles": args.obstacles,
        "guard": args.guard,
        "device": args.device,
        "source_hashes": {
            "runner": file_hash(Path(__file__)),
            "environment": file_hash(ROOT / "experiments/directional_navigation/dvoi_h_branch.py"),
            "locked_recovery": file_hash(ROOT / "experiments/directional_navigation/recovery.py"),
            "collision_protocol": file_hash(ROOT / "docs/LOCKED_COLLISION_RECOVERY_PROTOCOL.md"),
        },
        "automatic_analysis": False,
        "automatic_training": False,
    }
    output_dir = args.output_dir.resolve()
    manifest_path = output_dir / "manifest.json"
    if args.resume:
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("resume manifest mismatch")
        checkpoint = json.loads((output_dir / "latest.json").read_text())
        completed = list(checkpoint["completed"])
        record_index = list(checkpoint["records"])
        total_steps = int(checkpoint["total_branch_policy_steps"])
        elapsed_before = float(checkpoint["seconds"])
        generation = int(checkpoint["generation"])
    else:
        if output_dir.exists():
            raise FileExistsError(f"output directory already exists: {output_dir}")
        output_dir.mkdir(parents=True)
        atomic_json(manifest_path, manifest)
        completed, record_index, total_steps, elapsed_before, generation = [], [], 0, 0.0, 0

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    torch.set_num_threads(1)
    navigator = FrozenNavigator(args.device)
    started = time.monotonic()
    atomic_json(
        output_dir / "status.json",
        {
            "status": "RUNNING",
            "pid": os.getpid(),
            "completed": len(completed),
            "target": len(identities),
            "automatic_analysis": False,
            "automatic_training": False,
        },
    )
    try:
        for identity in identities:
            if identity in completed:
                continue
            append_access_event(contract_path, args.split, identity, run_id)
            current = json.loads(contract_path.read_text())
            seed = read_world_seed(current, identity)
            meta, arrays = collect_world(
                navigator,
                world_seed=seed,
                expected_world_identity=identity,
                obstacles=args.obstacles,
                guard=args.guard,
            )
            record = write_world_record(output_dir, meta, arrays)
            completed.append(identity)
            record_index.append(record)
            total_steps += int(record["total_branch_policy_steps"])
            generation += 1
            elapsed = elapsed_before + time.monotonic() - started
            latest = {
                "schema_version": "dvoi-h-checkpoint-v1",
                "generation": generation,
                "completed": completed,
                "target": len(identities),
                "records": record_index,
                "total_branch_policy_steps": total_steps,
                "seconds": elapsed,
            }
            atomic_json(output_dir / "latest.json", latest)
            atomic_json(
                output_dir / "status.json",
                {
                    "status": "RUNNING",
                    "pid": os.getpid(),
                    "completed": len(completed),
                    "target": len(identities),
                    "generation": generation,
                    "total_branch_policy_steps": total_steps,
                    "seconds": elapsed,
                    "automatic_analysis": False,
                    "automatic_training": False,
                },
            )
            print(
                f"checkpoint {generation}: {len(completed)}/{len(identities)} "
                f"worlds, anchors={record['eligible_anchor_count']}, "
                f"branch_steps={record['total_branch_policy_steps']}",
                flush=True,
            )
            if STOP or (output_dir / "PAUSE").exists() or (
                args.stop_after_worlds > 0 and len(completed) >= args.stop_after_worlds
            ):
                break
        complete = len(completed) == len(identities)
        state = "COMPLETE_AWAITING_ANALYSIS" if complete else "PAUSED"
        status = json.loads((output_dir / "status.json").read_text())
        status["status"] = state
        atomic_json(output_dir / "status.json", status)
        if complete:
            atomic_json(
                output_dir / "results.json",
                {
                    "schema_version": "dvoi-h-run-results-v1",
                    "records": record_index,
                    "completed": len(completed),
                    "target": len(identities),
                    "total_branch_policy_steps": total_steps,
                    "automatic_analysis": False,
                    "automatic_training": False,
                },
            )
        return 0
    except Exception:
        atomic_json(
            output_dir / "ERROR.json",
            {"error": traceback.format_exc(), "completed": len(completed), "target": len(identities)},
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
