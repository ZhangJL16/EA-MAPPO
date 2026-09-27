"""Execute frozen return-window roots by shared historical trajectory.

This is an execution optimization only: each root gets the same pre-decision
simulator state and the same two branches as branch_return_window_cpu.py.
"""

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
from .audit_return_window_roots import audit
from .branch_return_timing_cpu import _rollout
from .branch_return_window_cpu import OUTPUT, REPO, _wait_to_takeover
from .collect_shield_feedback_cpu import evaluation_dir, read_json, training_dir
from .freeze_return_timing_roots import archived_actions, digest
from .freeze_return_window_roots import OUTPUT as ROOTS_PATH


def one_parent(roots: list[dict], roots_sha256: str) -> list[dict]:
    """Replay once through one map, branching at every requested decision."""
    torch.set_num_threads(1)
    roots = sorted(roots, key=lambda root: root["decision"])
    first = roots[0]
    parent_id = first["parent_first_takeover_id"]
    if any(root["parent_first_takeover_id"] != parent_id for root in roots):
        raise ValueError("multiple historical parents in grouped job")
    train = training_dir(first["panel"], first["seed"], "control")
    status = read_json(train / "status.json")
    model_path = train / status["latest_model"]
    archive = evaluation_dir(first["panel"], first["seed"], "control", "shielded",
                             first["map_id"])
    event_path, summary_path = archive / "events.jsonl", archive / "summary.json"
    for root in roots:
        if (digest(model_path) != root["model_sha256"]
                or digest(event_path) != root["events_sha256"]
                or digest(summary_path) != root["summary_sha256"]):
            raise ValueError(f"frozen window input changed: {root['id']}")
    actions = archived_actions(event_path)
    calibration = read_json(CALIBRATION_OUTPUT / "calibration.json")
    model = PPO.load(str(model_path), device="cpu")
    env = DualConstraintGym(
        (first["map_id"],), calibration["capacity_synthetic_energy"],
        calibration["full_charge_seconds"], shielded=True,
    )
    by_decision = {root["decision"]: root for root in roots}
    if len(by_decision) != len(roots):
        raise ValueError("duplicate decision in grouped job")
    outputs = []
    try:
        observation, _ = env.reset(seed=first["map_id"],
                                   options={"map_id": first["map_id"]})
        for index in range(max(by_decision) + 1):
            predicted, _ = model.predict(observation, deterministic=True)
            if int(predicted) != actions[index]:
                raise ValueError(f"historical prefix action differs: {parent_id}/{index}")
            root = by_decision.get(index)
            if root is not None:
                if env.env.mode != "flight" or actions[index] != root["archived_action"]:
                    raise ValueError(f"frozen root differs from actual flight decision: {root['id']}")
                state = {
                    "time_s": env.env.time_s, "energy": env.env.energy,
                    "capacity": env.env.charger.capacity,
                    "completed_targets": env.env.completed_targets,
                    "route_distance_to_target": float(observation[14]) * 2 * env.env.case.config.side_m,
                    "route_distance_to_station": float(observation[15]) * 2 * env.env.case.config.side_m,
                }
                parent_hash = sha256(pickle.dumps(env, protocol=pickle.HIGHEST_PROTOCOL)).hexdigest()
                returned = _rollout(deepcopy(env), [9], repeat_four=False)
                try:
                    waited = _wait_to_takeover(deepcopy(env), model, actions, index,
                                               root["first_takeover_decision"])
                except ValueError as exc:
                    raise ValueError(f"{root['id']}: {exc}") from exc
                if sha256(pickle.dumps(env, protocol=pickle.HIGHEST_PROTOCOL)).hexdigest() != parent_hash:
                    raise ValueError(f"branching mutated historical parent: {root['id']}")
                outputs.append({
                    "protocol": "return_timing_window_cpu_v1",
                    "roots_sha256": roots_sha256,
                    "root": root, "state": state,
                    "branches": {"return_now": returned,
                                 "wait_until_archived_takeover": waited},
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
    parser.add_argument("--parent-id", help="Only replay this frozen historical parent")
    parser.add_argument("--verify-only", action="store_true",
                        help="Compare results to existing files without writing")
    args = parser.parse_args()
    manifest = read_json(ROOTS_PATH)
    if manifest["protocol_sha256"] != digest(REPO / "RETURN_TIMING_WINDOW_PROTOCOL_20260927.md"):
        raise ValueError("frozen return-window protocol changed")
    roots_hash = digest(ROOTS_PATH)
    eligibility = audit()
    valid_ids = set(eligibility["valid_root_ids"])
    grouped = defaultdict(list)
    for root in manifest["roots"]:
        if root["id"] in valid_ids:
            grouped[root["parent_first_takeover_id"]].append(root)
    if args.parent_id:
        grouped = {args.parent_id: grouped[args.parent_id]}
        if not grouped[args.parent_id]:
            raise ValueError("unknown parent ID")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    pending = {}
    for parent_id, roots in grouped.items():
        if args.verify_only or any(not (OUTPUT / f"{root['id']}.json").exists() for root in roots):
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
                        raise ValueError(f"grouped replay differs from individual replay: {path}")
                elif args.verify_only:
                    raise ValueError(f"verify-only root has no existing branch: {path}")
                else:
                    temp = path.with_suffix(".json.tmp")
                    temp.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                                    encoding="utf-8")
                    os.replace(temp, path)
                    print(f"verified {result['root']['id']}", flush=True)
            print(f"parent verified {parent_id}", flush=True)
    if (not args.parent_id and not args.verify_only
            and len(list(OUTPUT.glob("*.json"))) != eligibility["valid_roots"]):
        raise ValueError("incomplete return-window branch matrix")


if __name__ == "__main__":
    main()
