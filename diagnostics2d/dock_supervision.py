"""Evaluate a frozen PPO with only its dock departure failures repaired.

This is a causal diagnostic, not a new learned policy or a fair main-table
method. The supervisor uses the same full-map reference round-trip estimate
available to the route baseline and leaves every in-flight action unchanged.
It records both the PPO proposal and the executed action for audit.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import pickle
from time import perf_counter
import traceback

from stable_baselines3 import PPO

from dual_constraint_2d.calibration import DEFAULT_OUTPUT as CALIBRATION_OUTPUT
from dual_constraint_2d.evaluate_matrix import ROOT, SOURCE_FILES
from dual_constraint_2d.gym_adapter import DualConstraintGym


def supervised_action(env, proposed: int) -> int:
    """Charge just enough when a docked PPO proposal cannot cover the task.

    The reference estimate is a scheduling rule, not an executable safety
    certificate; the unchanged shield remains responsible for execution.
    """
    if env.mode != "docked":
        return proposed
    capacity = env.charger.capacity
    if proposed in (10, 11, 12):
        fraction = (0.5, 0.75, 1.0)[proposed - 10]
        if fraction * capacity > env.energy + 1e-8:
            return proposed
    needed = min(capacity, env.target.reference_round_trip_energy + 0.05 * capacity)
    if env.energy + 1e-8 >= needed:
        return proposed
    for action_id, fraction in ((10, 0.5), (11, 0.75), (12, 1.0)):
        if fraction * capacity + 1e-8 >= needed:
            return action_id
    raise AssertionError("full charge must cover the capped reference need")


def _atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def _atomic_json(path: Path, value: dict) -> None:
    _atomic_bytes(path, (json.dumps(value, sort_keys=True, indent=2) + "\n").encode())


def _manifest(map_id: int, model_path: Path) -> dict:
    files = (*SOURCE_FILES, "diagnostics2d/dock_supervision.py")
    return {
        "protocol": "ppo_dock_charge_diagnostic_v2",
        "map_id": map_id,
        "model_path": str(model_path.resolve()),
        "model_sha256": sha256(model_path.read_bytes()).hexdigest(),
        "calibration_sha256": sha256(
            (CALIBRATION_OUTPUT / "calibration.json").read_bytes()
        ).hexdigest(),
        "source_sha256": {
            name: sha256((ROOT / name).read_bytes()).hexdigest() for name in files
        },
    }


def _summary(wrapper: DualConstraintGym, events: list[dict]) -> dict:
    env = wrapper.env
    assert env is not None
    return {
        "algorithm": "ppo_dock_supervised_diagnostic",
        "map_id": env.case.map_id,
        "done": env.done,
        "policy_decisions": len(events),
        "simulated_seconds": env.time_s,
        "completed_targets": env.completed_targets,
        "collision_count": env.state.collision_count,
        "actual_depletion": env.failure_reason == "depletion",
        "failure_reason": env.failure_reason,
        "charge_events": env.charge_events,
        "return_takeovers": env.return_takeovers,
        "energy_filter_events": env.energy_filter_events,
        "departure_rejections": sum(row["event"] == "departure_rejected" for row in events),
        "dock_decisions": sum(row["mode_before"] == "docked" for row in events),
        "proposed_dock_charge_actions": sum(
            row["mode_before"] == "docked" and row["proposed_action_id"] >= 10
            for row in events
        ),
        "executed_dock_charge_actions": sum(
            row["mode_before"] == "docked" and row["executed_action_id"] >= 10
            for row in events
        ),
        "supervisor_overrides": sum(
            row["proposed_action_id"] != row["executed_action_id"] for row in events
        ),
    }


def _save(output: Path, wrapper: DualConstraintGym, events: list[dict]) -> None:
    _atomic_bytes(output / "checkpoint.pkl", pickle.dumps(
        {"wrapper": wrapper, "events": events}, protocol=pickle.HIGHEST_PROTOCOL,
    ))
    _atomic_bytes(output / "events.jsonl", (
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in events)
    ).encode())
    summary = _summary(wrapper, events)
    _atomic_json(output / "status.json", summary)
    if summary["done"]:
        _atomic_json(output / "summary.json", summary)


def evaluate(output: Path, *, map_id: int, model_path: Path,
             max_new_decisions: int | None = None,
             checkpoint_every: int = 20) -> dict:
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    if checkpoint_every <= 0 or (max_new_decisions is not None and max_new_decisions <= 0):
        raise ValueError("invalid decision or checkpoint limit")
    output.mkdir(parents=True, exist_ok=True)
    manifest = _manifest(map_id, model_path)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise RuntimeError("diagnostic manifest or source changed")
    else:
        _atomic_json(manifest_path, manifest)
    checkpoint = output / "checkpoint.pkl"
    if checkpoint.exists():
        saved = pickle.loads(checkpoint.read_bytes())
        wrapper, events = saved["wrapper"], saved["events"]
        if wrapper.env.case.map_id != map_id or len(events) != wrapper.policy_decisions:
            raise RuntimeError("diagnostic checkpoint does not match request")
    else:
        calibration = json.loads((CALIBRATION_OUTPUT / "calibration.json").read_text())
        wrapper = DualConstraintGym(
            (map_id,), calibration["capacity_synthetic_energy"],
            calibration["full_charge_seconds"], shielded=True,
        )
        wrapper.reset(seed=map_id, options={"map_id": map_id})
        events = []
    model = PPO.load(str(model_path), device="cpu")
    began = perf_counter()
    new_count = 0
    try:
        while not wrapper.env.done and (
            max_new_decisions is None or new_count < max_new_decisions
        ):
            observation = wrapper.adapter.features(wrapper.env.observe())
            action, _ = model.predict(observation, deterministic=True)
            proposed = int(action)
            mode_before = wrapper.env.mode
            executed = supervised_action(wrapper.env, proposed)
            _, reward, terminated, truncated, info = wrapper.step(executed)
            events.append({
                "decision": len(events), "mode_before": mode_before,
                "proposed_action_id": proposed, "executed_action_id": executed,
                "reward_shaped": reward, "terminated": terminated,
                "truncated": truncated, **info,
            })
            new_count += 1
            if new_count % checkpoint_every == 0 or wrapper.env.done:
                _save(output, wrapper, events)
        _save(output, wrapper, events)
    except Exception:
        _atomic_json(output / "error.json", {
            "saved_decisions": len(events), "traceback": traceback.format_exc(),
        })
        raise
    return {**_summary(wrapper, events), "new_decisions": new_count,
            "wall_seconds_this_call": perf_counter() - began}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--map-id", type=int, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--max-new-decisions", type=int)
    parser.add_argument("--checkpoint-every", type=int, default=20)
    args = parser.parse_args()
    print(json.dumps(evaluate(
        args.output, map_id=args.map_id, model_path=args.model_path,
        max_new_decisions=args.max_new_decisions,
        checkpoint_every=args.checkpoint_every,
    ), sort_keys=True))


if __name__ == "__main__":
    main()
