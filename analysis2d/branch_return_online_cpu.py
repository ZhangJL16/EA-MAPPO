"""Branch observable SOC crossing states through three fixed return choices."""

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
from .branch_return_route4_cpu import route_four_then_return
from .branch_return_timing_cpu import _result, _rollout
from .collect_shield_feedback_cpu import evaluation_dir, read_json, training_dir
from .freeze_return_online_roots import OUTPUT as ROOTS_PATH
from .freeze_return_timing_roots import ROOT, archived_actions, digest


OUTPUT = ROOTS_PATH.parent / "branches"


def archived_ppo_then_return(env: DualConstraintGym, model: PPO,
                             actions: list[int], start: int) -> dict:
    before_time, before_energy = env.env.time_s, env.env.energy
    before_completed = env.env.completed_targets
    reward = extra = 0.0
    infos = []
    index = start
    while (not env.env.done and env.env.mode == "flight"
           and env.env.completed_targets == before_completed):
        if index >= len(actions):
            raise ValueError("historical PPO branch exceeded archived actions")
        observation = env.adapter.features(env.env.observe())
        predicted, _ = model.predict(observation, deterministic=True)
        if int(predicted) != actions[index]:
            raise ValueError(f"historical PPO action differs at {index}")
        _, value, _, _, info = env.step(actions[index])
        added, _, _ = intervention_penalty(info, feedback=True)
        reward += value
        extra += added
        infos.append(info)
        index += 1
        if len(infos) > 3000:
            raise RuntimeError("PPO task attempt exceeded decision budget")
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
    actual_hashes = (digest(model_path), digest(event_path), digest(summary_path))
    for root in roots:
        if actual_hashes != (root["model_sha256"], root["events_sha256"],
                             root["summary_sha256"]):
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
                raise ValueError(f"historical prefix action differs: {first['parent_id']}/{index}")
            root = by_decision.get(index)
            if root is not None:
                soc = env.env.energy / env.env.charger.capacity
                if (env.env.mode != "flight" or env.env.completed_targets != 0
                        or env.env.charge_events != 0
                        or actions[index] != root["archived_action"]
                        or abs(soc - root["archived_soc"]) > 1e-6):
                    raise ValueError(f"online-state root replay differs: {root['id']}")
                state = {
                    "time_s": env.env.time_s, "energy": env.env.energy,
                    "capacity": env.env.charger.capacity, "soc": soc,
                    "completed_targets": env.env.completed_targets,
                    "route_distance_to_target": float(observation[14]) * 2 * env.env.case.config.side_m,
                    "route_distance_to_station": float(observation[15]) * 2 * env.env.case.config.side_m,
                }
                parent_hash = sha256(pickle.dumps(env, protocol=pickle.HIGHEST_PROTOCOL)).hexdigest()
                immediate = _rollout(deepcopy(env), [9], repeat_four=False)
                route4 = route_four_then_return(deepcopy(env))
                ppo = archived_ppo_then_return(deepcopy(env), model, actions, index)
                if sha256(pickle.dumps(env, protocol=pickle.HIGHEST_PROTOCOL)).hexdigest() != parent_hash:
                    raise ValueError(f"branch mutated historical parent: {root['id']}")
                outputs.append({
                    "protocol": "return_online_state_branch_cpu_v1",
                    "roots_sha256": roots_sha256, "root": root, "state": state,
                    "parent_pickle_sha256": parent_hash,
                    "branches": {"return_now": immediate,
                                 "route4_then_return": route4,
                                 "ppo_then_return": ppo},
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
    parser.add_argument("--parent-id", help="Run only one frozen parent trajectory")
    args = parser.parse_args()
    manifest = read_json(ROOTS_PATH)
    if manifest["protocol_sha256"] != digest(ROOT / "RETURN_ONLINE_STATE_BRANCH_PROTOCOL_20260927.md"):
        raise ValueError("frozen online-state protocol changed")
    roots_hash = digest(ROOTS_PATH)
    grouped = defaultdict(list)
    for root in manifest["roots"]:
        grouped[root["parent_id"]].append(root)
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
                        raise ValueError(f"resumed branch differs: {path}")
                else:
                    temp = path.with_suffix(".json.tmp")
                    temp.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                                    encoding="utf-8")
                    os.replace(temp, path)
                    print(f"verified {result['root']['id']}", flush=True)
            print(f"parent verified {parent}", flush=True)
    if (not args.parent_id
            and len(list(OUTPUT.glob("*.json"))) != manifest["root_count"]):
        raise ValueError("incomplete online-state branch matrix")


if __name__ == "__main__":
    main()
