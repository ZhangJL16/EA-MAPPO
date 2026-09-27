"""Replay frozen takeover roots with a simple route-to-target alternative."""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from hashlib import sha256
import json
import os
import pickle

import torch
from stable_baselines3 import PPO

from dual_constraint_2d.calibration import DEFAULT_OUTPUT as CALIBRATION_OUTPUT
from dual_constraint_2d.gym_adapter import DualConstraintGym
from learning2d.mode_policy import ModeMaskedPolicy  # noqa: F401
from learning2d.shield_feedback_gym import intervention_penalty
from .branch_return_timing_cpu import _result
from .collect_shield_feedback_cpu import evaluation_dir, read_json, training_dir
from .freeze_return_route4_roots import OUTPUT as ROOTS_PATH, WINDOW_BRANCHES
from .freeze_return_timing_roots import ROOT, archived_actions, digest


OUTPUT = ROOT / "evidence" / "return_route4_alternative_cpu_20260927" / "branches"


def route_four_then_return(env: DualConstraintGym) -> dict:
    before_time, before_energy = env.env.time_s, env.env.energy
    before_completed = env.env.completed_targets
    reward = extra = 0.0
    infos = []
    while (not env.env.done and env.env.mode == "flight"
           and env.env.completed_targets == before_completed):
        _, value, _, _, info = env.step(4)
        added, _, _ = intervention_penalty(info, feedback=True)
        reward += value
        extra += added
        infos.append(info)
        if len(infos) > 3000:
            raise RuntimeError("route-4 task attempt exceeded decision budget")
    task_completion_elapsed_s = (
        env.env.time_s - before_time
        if env.env.completed_targets > before_completed else None
    )
    if task_completion_elapsed_s is not None and not env.env.done:
        _, value, _, _, info = env.step(9)
        added, _, _ = intervention_penalty(info, feedback=True)
        reward += value
        extra += added
        infos.append(info)
    result = _result(env, before_time, before_energy, before_completed,
                     reward, extra, infos, len(infos))
    result["task_completion_elapsed_s"] = task_completion_elapsed_s
    return result


def original_window_root(root: dict) -> dict:
    source = {key: value for key, value in root.items()
              if key not in ("window_root_id", "window_branch_sha256")}
    source["id"] = root["window_root_id"]
    return source


def one_parent(roots: list[dict], roots_sha256: str) -> list[dict]:
    torch.set_num_threads(1)
    roots = sorted(roots, key=lambda root: root["decision"])
    first = roots[0]
    if any(root["parent_first_takeover_id"] != first["parent_first_takeover_id"]
           for root in roots):
        raise ValueError("multiple historical parents in route-4 worker job")
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
        raise ValueError("duplicate historical root decision")
    outputs = []
    try:
        observation, _ = env.reset(seed=first["map_id"],
                                   options={"map_id": first["map_id"]})
        for index in range(max(by_decision) + 1):
            predicted, _ = model.predict(observation, deterministic=True)
            if int(predicted) != actions[index]:
                raise ValueError(f"historical prefix action differs: {first['parent_first_takeover_id']}/{index}")
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
                prior_path = WINDOW_BRANCHES / f"{root['window_root_id']}.json"
                if digest(prior_path) != root["window_branch_sha256"]:
                    raise ValueError(f"prior window branch changed: {prior_path}")
                prior = read_json(prior_path)
                parent_hash = sha256(pickle.dumps(env, protocol=pickle.HIGHEST_PROTOCOL)).hexdigest()
                if (prior["root"] != original_window_root(root)
                        or prior["state"] != state
                        or prior["parent_pickle_sha256"] != parent_hash):
                    raise ValueError(f"historical route-4 state differs from prior clone: {root['id']}")
                route = route_four_then_return(deepcopy(env))
                if sha256(pickle.dumps(env, protocol=pickle.HIGHEST_PROTOCOL)).hexdigest() != parent_hash:
                    raise ValueError(f"route-4 branch mutated historical parent: {root['id']}")
                outputs.append({
                    "protocol": "return_route4_alternative_cpu_v1",
                    "roots_sha256": roots_sha256,
                    "root": root, "state": state,
                    "route4_branch": route,
                    "parent_pickle_sha256": parent_hash,
                })
            if index < max(by_decision):
                observation, _, _, _, _ = env.step(actions[index])
                if env.env.done:
                    raise ValueError("historical prefix ended before grouped root")
        return outputs
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-id", help="Only replay one frozen source trajectory")
    args = parser.parse_args()
    manifest = read_json(ROOTS_PATH)
    if manifest["protocol_sha256"] != digest(ROOT / "RETURN_ROUTE4_ALTERNATIVE_PROTOCOL_20260927.md"):
        raise ValueError("frozen route-4 protocol changed")
    roots_hash = digest(ROOTS_PATH)
    grouped = defaultdict(list)
    for root in manifest["roots"]:
        grouped[root["parent_first_takeover_id"]].append(root)
    if args.parent_id:
        if args.parent_id not in grouped:
            raise ValueError("unknown parent ID")
        grouped = {args.parent_id: grouped[args.parent_id]}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    pending = {parent: roots for parent, roots in grouped.items()
               if any(not (OUTPUT / f"{root['id']}.json").exists() for root in roots)}
    with ProcessPoolExecutor(max_workers=min(16, max(1, len(pending)))) as pool:
        futures = {pool.submit(one_parent, roots, roots_hash): parent
                   for parent, roots in pending.items()}
        for future in as_completed(futures):
            parent = futures[future]
            for result in future.result():
                path = OUTPUT / f"{result['root']['id']}.json"
                if path.exists():
                    if read_json(path) != result:
                        raise ValueError(f"resumed route-4 branch differs: {path}")
                else:
                    temp = path.with_suffix(".json.tmp")
                    temp.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                                    encoding="utf-8")
                    os.replace(temp, path)
                    print(f"verified {result['root']['id']}", flush=True)
            print(f"parent verified {parent}", flush=True)
    if (not args.parent_id
            and len(list(OUTPUT.glob("*.json"))) != manifest["root_count"]):
        raise ValueError("incomplete route-4 alternative matrix")


if __name__ == "__main__":
    main()
