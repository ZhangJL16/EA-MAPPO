#!/usr/bin/env python3
"""v1a resumable runner: v1 interventions plus task-completion timestamps."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from experiments.directional_navigation.oracle_stuckness_intervention import (
    FrozenIntervention, METHODS, OracleStucknessEnv, SOC_THRESHOLD,
)
from review_bundle.safety.switching.commitment import SortieMode
from scripts.run_dvoi_h_collection import FrozenNavigator
from scripts.run_oracle_stuckness_intervention import atomic_json, navigator_action
from scripts.validate_oracle_stuckness_v1a_contract import object_hash, validate

STOP = False


def request_stop(*_: object) -> None:
    global STOP
    STOP = True


def run_method(navigator: FrozenNavigator, world: dict[str, Any], method: int,
               duration_seconds: float) -> dict[str, Any]:
    job = {"job_id": int(world["world_index"]) * 5 + method,
           "world_seed": int(world["world_seed"]), "head_seed": 0,
           "method": method, "method_name": METHODS[method],
           "threshold": SOC_THRESHOLD, "obstacles": 48, "soc": 1.0}
    env = OracleStucknessEnv(job)
    started = time.time()
    try:
        observation, _ = env.reset()
        identity = object_hash(env.physical_world_record())
        if identity != world["physical_world_identity"]:
            raise RuntimeError("generated physical world does not match v1a registry")
        policy = FrozenIntervention(method)
        events: list[dict[str, Any]] = []
        completion_timestamps: list[float] = []
        attempt_seconds = 0.0
        attempt_steps = 0
        productive_dwell = unproductive_dwell = 0.0
        long_attempt_count = 0
        long_attempt_seconds = return_travel_seconds = 0.0
        zero_task_returns = 0
        tasks_at_last_recharge = 0
        while float(env.base.simulation_time) < duration_seconds and not env.exhausted and not STOP:
            was_task = env.base.mode is SortieMode.TASK
            was_returning = env.base.mode is SortieMode.CHARGER_COMMITTED
            time_before = float(env.base.simulation_time)
            commit, event = policy.decide(
                env, lambda obs, returning: navigator_action(navigator, obs, returning)
            )
            if event is not None:
                events.append({"policy_step": int(env.total_steps),
                               "simulation_seconds": time_before, **event})
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
            now = float(env.base.simulation_time)
            dt = now - time_before
            if was_task and not commit:
                attempt_seconds += dt
                attempt_steps += 1
            if was_returning or commit:
                return_travel_seconds += dt
            gain = int(info.get("task_gain", 0))
            if gain:
                completion_timestamps.extend([now] * gain)
                productive_dwell += attempt_seconds
                if attempt_steps > 4000:
                    long_attempt_count += 1
                    long_attempt_seconds += attempt_seconds
                attempt_seconds = 0.0
                attempt_steps = 0
            if bool(info.get("recharge_event", False)):
                zero_task_returns += int(int(env.base.tasks_completed) == tasks_at_last_recharge)
                tasks_at_last_recharge = int(env.base.tasks_completed)
        if attempt_steps:
            unproductive_dwell += attempt_seconds
            if attempt_steps > 4000:
                long_attempt_count += 1
                long_attempt_seconds += attempt_seconds
        horizon_completions = [t for t in completion_timestamps if t <= duration_seconds]
        return {
            "schema_version": "oracle-stuckness-method-result-v1a",
            "parent_lineage": "oracle-productive-stuckness-20260916-v1",
            "world_index": int(world["world_index"]),
            "world_seed": int(world["world_seed"]),
            "physical_world_identity": identity,
            "split": world["split"], "method": method, "method_name": METHODS[method],
            "experimental_action_outside_cr": method == 3,
            "oracle_scope": "completion-within-4000 privileged oracle" if method == 4 else None,
            "configured_denominator_seconds": float(duration_seconds),
            "simulated_seconds": float(env.base.simulation_time),
            "time_alive_seconds": min(float(env.base.simulation_time), float(duration_seconds)),
            "task_completion_timestamps": horizon_completions,
            "tasks_completed": len(horizon_completions),
            "tasks_per_hour": 3600.0 * len(horizon_completions) / float(duration_seconds),
            "raw_poststep_tasks_completed": int(env.base.tasks_completed),
            "recharges": int(env.recharges), "zero_task_returns": int(zero_task_returns),
            "energy_exhausted": bool(env.exhausted), "final_energy": float(env.base.agent.energy),
            "collision_count": int(env.contacts),
            "return_travel_seconds": float(return_travel_seconds),
            "productive_task_dwell_seconds": float(productive_dwell),
            "unproductive_task_dwell_seconds": float(unproductive_dwell),
            "attempts_over_4000": int(long_attempt_count),
            "attempts_over_4000_seconds": float(long_attempt_seconds),
            "intervention_counts": policy.counts, "events": events,
            "completed": not STOP and (float(env.base.simulation_time) >= duration_seconds or env.exhausted),
            "wall_seconds": time.time() - started,
            "diagnostic": env.diagnostic(),
        }
    finally:
        env.close()


def append_access(path: Path, world: dict[str, Any], split: str) -> None:
    contract = json.loads(path.read_text())
    events = contract.setdefault("access_events", [])
    if any(e["world_seed"] == world["world_seed"] and e["split"] == split for e in events):
        return
    event = {"split": split, "world_seed": world["world_seed"],
             "physical_world_identity": world["physical_world_identity"],
             "access_type": "v1a_excluded_smoke" if split == "SMOKE_DEBUG" else "v1a_dev",
             "accessed_at_unix": time.time(),
             "previous_event_hash": events[-1]["event_hash"] if events else "GENESIS"}
    event["event_hash"] = object_hash(event)
    events.append(event)
    atomic_json(path, contract)


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
        raise SystemExit("v1a contract validation failed: " + "; ".join(checked["errors"]))
    expected = 120.0 if args.split == "SMOKE_DEBUG" else 7200.0
    if args.duration_seconds != expected:
        raise SystemExit("duration differs from frozen v1a duration")
    maximum = 1 if args.split == "SMOKE_DEBUG" else 24
    if not 1 <= args.stop_after_worlds <= maximum:
        raise SystemExit("stop-after-worlds outside registered split")
    contract = json.loads(args.contract.read_text())
    worlds = [w for w in contract["worlds"] if w["split"] == args.split]
    navigator = FrozenNavigator(args.device)
    complete = 0
    for world in worlds:
        if complete >= args.stop_after_worlds or STOP:
            break
        append_access(args.contract, world, args.split)
        directory = args.output / f"world_{world['world_index']:03d}_{world['world_seed']}"
        for method in range(5):
            destination = directory / f"M{method}.json"
            if destination.exists() and json.loads(destination.read_text()).get("completed"):
                continue
            result = run_method(navigator, world, method, args.duration_seconds)
            atomic_json(destination, result)
            if not result["completed"]:
                return
        complete += 1
    atomic_json(args.output / "startup_status.json", {
        "split": args.split, "completed_worlds_this_invocation": complete,
        "stop_after_worlds": args.stop_after_worlds, "stopped": bool(STOP),
        "automatic_analysis": False, "confirm_access": 0,
    })


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    main()
