"""Return-now versus complete-current-task-then-return on frozen real states."""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
import pickle

import torch
from stable_baselines3 import PPO

from dual_constraint_2d.calibration import DEFAULT_OUTPUT as CALIBRATION_OUTPUT
from dual_constraint_2d.gym_adapter import DualConstraintGym
from learning2d.mode_policy import ModeMaskedPolicy  # noqa: F401
from learning2d.shield_feedback_gym import intervention_penalty
from .branch_return_timing_cpu import _result, _rollout
from .collect_shield_feedback_cpu import evaluation_dir, read_json, training_dir
from .freeze_return_success_roots import OUTPUT as ROOTS_PATH, ROOT
from .freeze_return_timing_roots import archived_actions, digest


OUTPUT = ROOT / "evidence" / "return_success_control_cpu_20260927" / "branches"


def task_then_return(env: DualConstraintGym, model: PPO, actions: list[int],
                     start: int, completion: int) -> dict:
    before_time, before_energy = env.env.time_s, env.env.energy
    before_completed = env.env.completed_targets
    reward = extra = 0.0
    infos = []
    for index in range(start, completion + 1):
        observation = env.adapter.features(env.env.observe())
        predicted, _ = model.predict(observation, deterministic=True)
        if int(predicted) != actions[index]:
            raise ValueError(f"historical task action differs at {index}")
        _, value, _, _, info = env.step(actions[index])
        added, _, _ = intervention_penalty(info, feedback=True)
        reward += value
        extra += added
        infos.append(info)
        if index < completion and (env.env.done or env.env.mode != "flight"
                                   or env.env.completed_targets != before_completed):
            raise ValueError("historical task leg changed before first completion")
    completion_elapsed_s = env.env.time_s - before_time
    if (env.env.completed_targets != before_completed + 1
            or env.env.failure_reason is not None):
        raise ValueError("frozen first task completion did not replay")
    if not env.env.done:
        _, value, _, _, info = env.step(9)
        added, _, _ = intervention_penalty(info, feedback=True)
        reward += value
        extra += added
        infos.append(info)
    result = _result(env, before_time, before_energy, before_completed,
                     reward, extra, infos, len(infos))
    result["first_task_completion_elapsed_s"] = completion_elapsed_s
    return result


def one_parent(roots: list[dict], roots_sha256: str) -> list[dict]:
    torch.set_num_threads(1)
    roots = sorted(roots, key=lambda root: root["decision"])
    first = roots[0]
    if any(root["parent_id"] != first["parent_id"] for root in roots):
        raise ValueError("multiple historical parents in one worker job")
    train = training_dir(first["panel"], first["seed"], "control")
    status = read_json(train / "status.json")
    model_path = train / status["latest_model"]
    archive = evaluation_dir(first["panel"], first["seed"], "control", "shielded",
                             first["map_id"])
    event_path, summary_path = archive / "events.jsonl", archive / "summary.json"
    model_hash, event_hash, summary_hash = (digest(model_path), digest(event_path),
                                           digest(summary_path))
    for root in roots:
        if (model_hash, event_hash, summary_hash) != (
                root["model_sha256"], root["events_sha256"], root["summary_sha256"]):
            raise ValueError(f"frozen input changed: {root['id']}")
    actions = archived_actions(event_path)
    calibration = read_json(CALIBRATION_OUTPUT / "calibration.json")
    model = PPO.load(str(model_path), device="cpu")
    env = DualConstraintGym(
        (first["map_id"],), calibration["capacity_synthetic_energy"],
        calibration["full_charge_seconds"], shielded=True,
    )
    by_decision = {root["decision"]: root for root in roots}
    if len(by_decision) != len(roots):
        raise ValueError("duplicate historical decision in parent")
    outputs = []
    try:
        observation, _ = env.reset(seed=first["map_id"],
                                   options={"map_id": first["map_id"]})
        for index in range(max(by_decision) + 1):
            predicted, _ = model.predict(observation, deterministic=True)
            if int(predicted) != actions[index]:
                raise ValueError(f"historical prefix action differs: {first['parent_id']}/{index}")
            root = by_decision.get(index)
            if root is not None:
                if env.env.mode != "flight" or actions[index] != root["archived_action"]:
                    raise ValueError(f"frozen flight root differs: {root['id']}")
                state = {
                    "time_s": env.env.time_s, "energy": env.env.energy,
                    "capacity": env.env.charger.capacity,
                    "completed_targets": env.env.completed_targets,
                    "route_distance_to_target": float(observation[14]) * 2 * env.env.case.config.side_m,
                    "route_distance_to_station": float(observation[15]) * 2 * env.env.case.config.side_m,
                }
                parent_hash = sha256(pickle.dumps(env, protocol=pickle.HIGHEST_PROTOCOL)).hexdigest()
                immediate = _rollout(deepcopy(env), [9], repeat_four=False)
                try:
                    completed = task_then_return(deepcopy(env), model, actions, index,
                                                 root["first_completion_decision"])
                except ValueError as exc:
                    raise ValueError(f"{root['id']}: {exc}") from exc
                if sha256(pickle.dumps(env, protocol=pickle.HIGHEST_PROTOCOL)).hexdigest() != parent_hash:
                    raise ValueError(f"branching mutated historical parent: {root['id']}")
                outputs.append({
                    "protocol": "return_success_control_cpu_v1",
                    "roots_sha256": roots_sha256,
                    "root": root, "state": state,
                    "branches": {"return_now": immediate,
                                 "task_then_return": completed},
                    "parent_pickle_sha256": parent_hash,
                })
            if index < max(by_decision):
                observation, _, _, _, _ = env.step(actions[index])
                if env.env.done:
                    raise ValueError("historical prefix ended before a grouped root")
        return outputs
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-id", help="Only replay one frozen source trajectory")
    args = parser.parse_args()
    manifest = read_json(ROOTS_PATH)
    if manifest["protocol_sha256"] != digest(ROOT / "RETURN_SUCCESS_CONTROL_PROTOCOL_20260927.md"):
        raise ValueError("frozen success-control protocol changed")
    roots_hash = digest(ROOTS_PATH)
    grouped = defaultdict(list)
    for root in manifest["roots"]:
        grouped[root["parent_id"]].append(root)
    if args.parent_id:
        if args.parent_id not in grouped:
            raise ValueError("unknown parent ID")
        grouped = {args.parent_id: grouped[args.parent_id]}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    pending = {}
    for parent_id, roots in grouped.items():
        if any(not (OUTPUT / f"{root['id']}.json").exists() for root in roots):
            pending[parent_id] = roots
    with ProcessPoolExecutor(max_workers=min(16, max(1, len(pending)))) as pool:
        futures = {pool.submit(one_parent, roots, roots_hash): parent_id
                   for parent_id, roots in pending.items()}
        for future in as_completed(futures):
            parent_id = futures[future]
            for result in future.result():
                path = OUTPUT / f"{result['root']['id']}.json"
                if path.exists():
                    if read_json(path) != result:
                        raise ValueError(f"resumed branch differs from frozen result: {path}")
                else:
                    temp = path.with_suffix(".json.tmp")
                    temp.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                                    encoding="utf-8")
                    os.replace(temp, path)
                    print(f"verified {result['root']['id']}", flush=True)
            print(f"parent verified {parent_id}", flush=True)
    if (not args.parent_id
            and len(list(OUTPUT.glob("*.json"))) != manifest["root_count"]):
        raise ValueError("incomplete successful-task negative-control matrix")


if __name__ == "__main__":
    main()
