"""Exact-state counterfactual return-timing branches on archived 2D maps."""

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
from .collect_shield_feedback_cpu import evaluation_dir, read_json, training_dir
from .freeze_return_timing_roots import OUTPUT as ROOTS_PATH, archived_actions, digest


REPO = Path(__file__).resolve().parents[1]
OUTPUT = REPO / "evidence" / "return_timing_branch_cpu_20260927" / "branches"


def _result(env: DualConstraintGym, before_time: float, before_energy: float,
            before_completed: int, reward: float, extra: float,
            infos: list[dict], policy_decisions: int) -> dict:
    events = [row.get("event") for info in infos for row in info["plant_trace"]]
    return {
        "policy_decisions": policy_decisions,
        "elapsed_s": env.env.time_s - before_time,
        "energy_spent": before_energy - env.env.energy,
        "post_energy": env.env.energy,
        "post_mode": env.env.mode,
        "post_time_s": env.env.time_s,
        "task_completed": env.env.completed_targets - before_completed,
        "base_shaped_reward": reward,
        "feedback_shaped_reward": reward + extra,
        "extra_feedback_reward": extra,
        "return_takeovers": events.count("return_takeover"),
        "rejected_departures": events.count("departure_rejected"),
        "charge_events": env.env.charge_events,
        "failure_reason": env.env.failure_reason,
        "horizon_censored": env.env.done and env.env.failure_reason is None,
        "events": events,
    }


def _rollout(env: DualConstraintGym, actions: list[int], *, repeat_four: bool) -> dict:
    before_time = env.env.time_s
    before_energy = env.env.energy
    before_completed = env.env.completed_targets
    reward = extra = 0.0
    infos = []
    decisions = 0
    for action in actions:
        _, value, _, _, info = env.step(action)
        added, _, _ = intervention_penalty(info, feedback=True)
        reward += value
        extra += added
        infos.append(info)
        decisions += 1
    if repeat_four:
        while (not env.env.done and env.env.mode == "flight"
               and env.env.completed_targets == before_completed):
            _, value, _, _, info = env.step(4)
            added, _, _ = intervention_penalty(info, feedback=True)
            reward += value
            extra += added
            infos.append(info)
            decisions += 1
            if decisions > 3000:
                raise RuntimeError("route-4 continuation exceeded decision budget")
    return _result(env, before_time, before_energy, before_completed,
                   reward, extra, infos, decisions)


def one_root(root: dict, roots_sha256: str) -> dict:
    torch.set_num_threads(1)
    panel, seed, condition = root["panel"], root["seed"], root["source_condition"]
    train = training_dir(panel, seed, condition)
    status = read_json(train / "status.json")
    model_path = train / status["latest_model"]
    archive = evaluation_dir(panel, seed, condition, "shielded", root["map_id"])
    event_path, summary_path = archive / "events.jsonl", archive / "summary.json"
    if (digest(model_path) != root["model_sha256"]
            or digest(event_path) != root["events_sha256"]
            or digest(summary_path) != root["summary_sha256"]):
        raise ValueError(f"frozen input changed: {root['id']}")
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
                raise ValueError(f"historical policy action differs: {root['id']}/{index}")
            observation, _, _, _, _ = env.step(actions[index])
            if env.env.done:
                raise ValueError(f"historical prefix ended before root: {root['id']}")
        predicted, _ = model.predict(observation, deterministic=True)
        if (int(predicted) != root["archived_action"]
                or actions[root["decision"]] != root["archived_action"]
                or env.env.mode != "flight"):
            raise ValueError(f"selected root mismatches archived flight decision: {root['id']}")
        baseline = {
            "time_s": env.env.time_s, "energy": env.env.energy,
            "capacity": env.env.charger.capacity,
            "completed_targets": env.env.completed_targets,
            "position_xy": list(env.env.observe()["position_xy"]),
            "target_xy": list(env.env.observe()["target_xy"]),
            "station_xy": list(env.env.observe()["station_xy"]),
            "route_distance_to_target": float(observation[14]) * 2 * env.env.case.config.side_m,
            "route_distance_to_station": float(observation[15]) * 2 * env.env.case.config.side_m,
        }
        parent_hash = sha256(pickle.dumps(env, protocol=pickle.HIGHEST_PROTOCOL)).hexdigest()
        if root["kind"] == "first_takeover":
            original = _rollout(deepcopy(env), [root["archived_action"]], repeat_four=False)
            voluntary = _rollout(deepcopy(env), [9], repeat_four=False)
            if original["return_takeovers"] < 1:
                raise ValueError(f"selected action no longer causes takeover: {root['id']}")
            branches = {"original_action": original, "voluntary_return": voluntary}
        elif root["kind"] == "early_voluntary":
            voluntary = _rollout(deepcopy(env), [9], repeat_four=False)
            route = _rollout(deepcopy(env), [4], repeat_four=True)
            branches = {"voluntary_return": voluntary, "route_four_to_task_or_dock": route}
        else:
            raise ValueError("unknown root kind")
        if sha256(pickle.dumps(env, protocol=pickle.HIGHEST_PROTOCOL)).hexdigest() != parent_hash:
            raise ValueError(f"branching mutated historical parent: {root['id']}")
        return {
            "protocol": "return_timing_real_state_branch_cpu_v1",
            "roots_sha256": roots_sha256,
            "root": root, "state": baseline, "branches": branches,
            "parent_pickle_sha256": parent_hash,
        }
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-id", help="Run one frozen root as a startup smoke")
    args = parser.parse_args()
    payload = read_json(ROOTS_PATH)
    if payload["protocol_sha256"] != digest(REPO / "RETURN_TIMING_BRANCH_PROTOCOL_20260927.md"):
        raise ValueError("frozen protocol changed")
    roots_hash = digest(ROOTS_PATH)
    jobs = payload["roots"]
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
                raise ValueError(f"existing branch differs from frozen inputs: {path}")
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
    if not args.root_id and len(list(OUTPUT.glob("*.json"))) != payload["root_count"]:
        raise ValueError("incomplete return-timing branch matrix")


if __name__ == "__main__":
    main()
