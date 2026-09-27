"""Return now versus archived wait-to-takeover from fixed earlier states."""

from __future__ import annotations

import argparse
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
from .freeze_return_timing_roots import archived_actions, digest
from .freeze_return_window_roots import OUTPUT as ROOTS_PATH


REPO = Path(__file__).resolve().parents[1]
OUTPUT = REPO / "evidence" / "return_timing_window_cpu_20260927" / "branches"


def _wait_to_takeover(env: DualConstraintGym, model: PPO,
                      actions: list[int], start: int, end: int) -> dict:
    before_time, before_energy = env.env.time_s, env.env.energy
    before_completed = env.env.completed_targets
    reward = extra = 0.0
    infos = []
    for index in range(start, end + 1):
        observation = env.adapter.features(env.env.observe())
        predicted, _ = model.predict(observation, deterministic=True)
        if int(predicted) != actions[index]:
            raise ValueError(f"historical continuation action differs at {index}")
        _, value, _, _, info = env.step(actions[index])
        added, _, _ = intervention_penalty(info, feedback=True)
        reward += value
        extra += added
        infos.append(info)
        if index < end and (env.env.done or env.env.mode != "flight"):
            raise ValueError("historical flight segment ended before takeover")
    result = _result(env, before_time, before_energy, before_completed,
                     reward, extra, infos, len(infos))
    if result["return_takeovers"] < 1 or result["task_completed"] != 0:
        raise ValueError(
            "selected same-task historical continuation changed: "
            f"start={start}, end={end}, "
            f"return_takeovers={result['return_takeovers']}, "
            f"task_completed={result['task_completed']}, "
            f"post_mode={result['post_mode']}, "
            f"failure_reason={result['failure_reason']}"
        )
    return result


def one_root(root: dict, roots_sha256: str) -> dict:
    torch.set_num_threads(1)
    train = training_dir(root["panel"], root["seed"], "control")
    status = read_json(train / "status.json")
    model_path = train / status["latest_model"]
    archive = evaluation_dir(root["panel"], root["seed"], "control", "shielded",
                             root["map_id"])
    event_path, summary_path = archive / "events.jsonl", archive / "summary.json"
    if (digest(model_path) != root["model_sha256"]
            or digest(event_path) != root["events_sha256"]
            or digest(summary_path) != root["summary_sha256"]):
        raise ValueError(f"frozen window input changed: {root['id']}")
    actions = archived_actions(event_path)
    calibration = read_json(CALIBRATION_OUTPUT / "calibration.json")
    model = PPO.load(str(model_path), device="cpu")
    env = DualConstraintGym(
        (root["map_id"],), calibration["capacity_synthetic_energy"],
        calibration["full_charge_seconds"], shielded=True,
    )
    try:
        observation, _ = env.reset(seed=root["map_id"], options={"map_id": root["map_id"]})
        for index in range(root["decision"]):
            predicted, _ = model.predict(observation, deterministic=True)
            if int(predicted) != actions[index]:
                raise ValueError(f"historical prefix action differs: {root['id']}/{index}")
            observation, _, _, _, _ = env.step(actions[index])
            if env.env.done:
                raise ValueError("historical prefix ended before fixed-lag state")
        if env.env.mode != "flight" or actions[root["decision"]] != root["archived_action"]:
            raise ValueError("frozen root differs from actual flight decision")
        state = {
            "time_s": env.env.time_s, "energy": env.env.energy,
            "capacity": env.env.charger.capacity,
            "completed_targets": env.env.completed_targets,
            "route_distance_to_target": float(observation[14]) * 2 * env.env.case.config.side_m,
            "route_distance_to_station": float(observation[15]) * 2 * env.env.case.config.side_m,
        }
        parent_hash = sha256(pickle.dumps(env, protocol=pickle.HIGHEST_PROTOCOL)).hexdigest()
        returned = _rollout(deepcopy(env), [9], repeat_four=False)
        waited = _wait_to_takeover(deepcopy(env), model, actions, root["decision"],
                                   root["first_takeover_decision"])
        if sha256(pickle.dumps(env, protocol=pickle.HIGHEST_PROTOCOL)).hexdigest() != parent_hash:
            raise ValueError("branching mutated historical parent")
        return {
            "protocol": "return_timing_window_cpu_v1",
            "roots_sha256": roots_sha256,
            "root": root, "state": state,
            "branches": {"return_now": returned,
                         "wait_until_archived_takeover": waited},
            "parent_pickle_sha256": parent_hash,
        }
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-id")
    args = parser.parse_args()
    manifest = read_json(ROOTS_PATH)
    if manifest["protocol_sha256"] != digest(REPO / "RETURN_TIMING_WINDOW_PROTOCOL_20260927.md"):
        raise ValueError("frozen return-window protocol changed")
    roots_hash = digest(ROOTS_PATH)
    jobs = manifest["roots"]
    if args.root_id:
        jobs = [root for root in jobs if root["id"] == args.root_id]
        if len(jobs) != 1:
            raise ValueError("root ID is absent or not unique")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    pending = []
    for root in jobs:
        path = OUTPUT / f"{root['id']}.json"
        if path.exists():
            prior = read_json(path)
            if prior["roots_sha256"] != roots_hash or prior["root"] != root:
                raise ValueError(f"existing branch differs from frozen root: {path}")
        else:
            pending.append(root)
    with ProcessPoolExecutor(max_workers=min(16, max(1, len(pending)))) as pool:
        futures = {pool.submit(one_root, root, roots_hash): root for root in pending}
        for future in as_completed(futures):
            result = future.result()
            path = OUTPUT / f"{result['root']['id']}.json"
            temp = path.with_suffix(".json.tmp")
            temp.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            os.replace(temp, path)
            print(f"verified {result['root']['id']}", flush=True)
    if not args.root_id and len(list(OUTPUT.glob("*.json"))) != manifest["root_count"]:
        raise ValueError("incomplete return-window branch matrix")


if __name__ == "__main__":
    main()
