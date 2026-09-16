#!/usr/bin/env python3
"""Resumable runner for the frozen M0--M4 productive-stuckness grid."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from experiments.directional_navigation.battery_sortie import CAPACITY
from experiments.directional_navigation.oracle_stuckness_intervention import (
    FrozenIntervention, METHODS, OracleStucknessEnv, SOC_THRESHOLD,
)
from review_bundle.safety.switching.commitment import SortieMode
from scripts.run_dvoi_h_collection import FrozenNavigator
from scripts.validate_oracle_stuckness_contract import object_hash, validate

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


def navigator_action(navigator: FrozenNavigator, observation: dict[str, np.ndarray], returning: bool) -> np.ndarray:
    return navigator.goal_action(observation, returning=returning)


def run_method(
    navigator: FrozenNavigator,
    world: dict[str, Any],
    method: int,
    duration_seconds: float,
) -> dict[str, Any]:
    job = {
        "job_id": int(world["world_index"]) * 5 + method,
        "world_seed": int(world["world_seed"]),
        "head_seed": 0,
        "method": method,
        "method_name": METHODS[method],
        "threshold": SOC_THRESHOLD,
        "obstacles": 48,
        "soc": 1.0,
    }
    env = OracleStucknessEnv(job)
    started = time.time()
    try:
        observation, _ = env.reset()
        observed_identity = object_hash(env.physical_world_record())
        if observed_identity != world["physical_world_identity"]:
            raise RuntimeError("generated physical world does not match frozen identity")
        policy = FrozenIntervention(method)
        events: list[dict[str, Any]] = []
        attempt_seconds = 0.0
        attempt_steps = 0
        productive_dwell = 0.0
        unproductive_dwell = 0.0
        long_attempt_count = 0
        long_attempt_seconds = 0.0
        return_travel_seconds = 0.0
        zero_task_returns = 0
        recharges_before = 0
        previous_tasks = int(env.base.tasks_completed)
        previous_mode = env.base.mode
        while float(env.base.simulation_time) < duration_seconds and not env.exhausted and not STOP:
            was_task = env.base.mode is SortieMode.TASK
            was_returning = env.base.mode is SortieMode.CHARGER_COMMITTED
            time_before = float(env.base.simulation_time)
            commit, event = policy.decide(
                env, lambda obs, returning: navigator_action(navigator, obs, returning)
            )
            if event is not None:
                events.append({"policy_step": int(env.total_steps), "simulation_seconds": time_before, **event})
            # A high-level intervention closes the abandoned attempt before the
            # next physical step. That next step belongs to a new task (M3) or
            # return travel (M0/M1/M2/M4), never to task dwell.
            if event is not None and event["kind"] in {
                "replan", "oracle_return", "timeout", "no_progress_return", "soc40_return"
            }:
                unproductive_dwell += attempt_seconds
                if attempt_steps > 4000:
                    long_attempt_count += 1
                    long_attempt_seconds += attempt_seconds
                attempt_seconds = 0.0
                attempt_steps = 0
                observation = env.observation()
            returning = env.base.mode is SortieMode.CHARGER_COMMITTED or commit
            nav = navigator_action(navigator, observation, returning)
            action = np.concatenate((nav, [1.0 if commit else 0.0])).astype(np.float32)
            observation, _, _, _, info = env.step(action)
            dt = float(env.base.simulation_time) - time_before
            if was_task and not commit:
                attempt_seconds += dt
                attempt_steps += 1
            if was_returning or commit:
                return_travel_seconds += dt
            task_gain = int(info.get("task_gain", 0))
            if task_gain:
                productive_dwell += attempt_seconds
                if attempt_steps > 4000:
                    long_attempt_count += 1
                    long_attempt_seconds += attempt_seconds
                attempt_seconds = 0.0
                attempt_steps = 0
            if bool(info.get("recharge_event", False)):
                zero_task_returns += int(int(env.base.tasks_completed) == previous_tasks)
                previous_tasks = int(env.base.tasks_completed)
                recharges_before += 1
            previous_mode = env.base.mode
        if attempt_steps:
            unproductive_dwell += attempt_seconds
            if attempt_steps > 4000:
                long_attempt_count += 1
                long_attempt_seconds += attempt_seconds
        tasks = int(env.base.tasks_completed)
        denominator = float(duration_seconds)
        return {
            "schema_version": "oracle-stuckness-method-result-v1",
            "world_index": int(world["world_index"]),
            "world_seed": int(world["world_seed"]),
            "physical_world_identity": observed_identity,
            "split": world["split"],
            "method": method,
            "method_name": METHODS[method],
            "experimental_action_outside_cr": method == 3,
            "oracle_scope": "completion-within-4000 privileged oracle" if method == 4 else None,
            "configured_denominator_seconds": denominator,
            "simulated_seconds": float(env.base.simulation_time),
            "tasks_completed": tasks,
            "tasks_per_hour": 3600.0 * tasks / denominator,
            "recharges": int(env.recharges),
            "zero_task_returns": int(zero_task_returns),
            "energy_exhausted": bool(env.exhausted),
            "time_alive_seconds": min(float(env.base.simulation_time), denominator),
            "final_energy": float(env.base.agent.energy),
            "collision_count": int(env.contacts),
            "return_travel_seconds": float(return_travel_seconds),
            "productive_task_dwell_seconds": float(productive_dwell),
            "unproductive_task_dwell_seconds": float(unproductive_dwell),
            "attempts_over_4000": int(long_attempt_count),
            "attempts_over_4000_seconds": float(long_attempt_seconds),
            "intervention_counts": policy.counts,
            "events": events,
            "completed": not STOP and (float(env.base.simulation_time) >= duration_seconds or env.exhausted),
            "wall_seconds": time.time() - started,
            "diagnostic": env.diagnostic(),
        }
    finally:
        env.close()


def append_access(contract_path: Path, world: dict[str, Any], split: str) -> None:
    contract = json.loads(contract_path.read_text())
    events = contract.setdefault("access_events", [])
    if any(int(e["world_seed"]) == int(world["world_seed"]) and e["split"] == split for e in events):
        return
    event = {
        "split": split,
        "world_seed": int(world["world_seed"]),
        "physical_world_identity": world["physical_world_identity"],
        "access_type": "excluded_smoke" if split == "SMOKE_DEBUG" else "formal_startup_grid",
        "accessed_at_unix": time.time(),
        "previous_event_hash": events[-1]["event_hash"] if events else "GENESIS",
    }
    event["event_hash"] = object_hash(event)
    events.append(event)
    atomic_json(contract_path, contract)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("SMOKE_DEBUG", "DEV"), required=True)
    parser.add_argument("--duration-seconds", type=float, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--stop-after-worlds", type=int, required=True)
    args = parser.parse_args()
    checked = validate(args.contract)
    if not checked["valid"]:
        raise SystemExit("contract validation failed: " + "; ".join(checked["errors"]))
    contract = json.loads(args.contract.read_text())
    registered_duration = 120.0 if args.split == "SMOKE_DEBUG" else 7200.0
    if args.duration_seconds != registered_duration:
        raise SystemExit("duration differs from frozen split duration")
    if args.split == "DEV" and args.stop_after_worlds != 1:
        raise SystemExit("current authorization permits exactly one DEV startup world")
    worlds = [row for row in contract["worlds"] if row["split"] == args.split]
    navigator = FrozenNavigator(args.device)
    completed_worlds = 0
    for world in worlds:
        if completed_worlds >= args.stop_after_worlds or STOP:
            break
        append_access(args.contract, world, args.split)
        world_dir = args.output / f"world_{int(world['world_index']):03d}_{int(world['world_seed'])}"
        for method in range(5):
            destination = world_dir / f"M{method}.json"
            if destination.exists() and json.loads(destination.read_text()).get("completed"):
                continue
            result = run_method(navigator, world, method, args.duration_seconds)
            atomic_json(destination, result)
            if not result["completed"]:
                return
        if all((world_dir / f"M{m}.json").exists() for m in range(5)):
            completed_worlds += 1
    atomic_json(args.output / "startup_status.json", {
        "split": args.split,
        "completed_worlds_this_invocation": completed_worlds,
        "stop_after_worlds": args.stop_after_worlds,
        "stopped": bool(STOP),
        "automatic_analysis": False,
        "confirm_access": 0,
    })


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    main()
