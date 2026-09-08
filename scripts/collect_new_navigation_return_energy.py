#!/usr/bin/env python3
"""Exactly budgeted frozen-navigation energy collection; full-state resume."""
from __future__ import annotations

import argparse
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv

from experiments.directional_navigation.return_energy_data import ReturnEnergyRecovery, scene_split, suffix_labels
from experiments.directional_navigation.standard_baselines import atomic_json
from scripts.evaluate_hocbf_correction_fixed500 import validated_checkpoint
from scripts.resume_recovery_sac_ppo_stratified import file_hash

STOP = False


def request_stop(_signum, _frame) -> None:
    global STOP
    STOP = True


def empty_episode() -> dict:
    return {"observations": [], "indices": [], "energy": [], "contact": []}


def policy_hash(model: SAC) -> str:
    digest = hashlib.sha256()
    for key, value in model.policy.state_dict().items():
        digest.update(key.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def commit_episode(root: Path, data: dict, episode: dict) -> dict:
    seed = int(episode["seed"])
    split = scene_split(seed)
    energy, safe = suffix_labels(data["energy"], data["contact"], episode["goal_reached"])
    indices = np.asarray(data["indices"], dtype=np.int32)
    obs = np.asarray(data["observations"], dtype=np.float32)
    if (obs.shape != (len(indices), 2056) or len(energy) != episode["steps"]
            or int(np.sum(data["contact"])) != episode["collision_count"]
            or not np.isclose(np.sum(data["energy"]), episode["energy"], atol=1e-8)):
        raise ValueError("episode accounting mismatch")
    directory = root / "episodes" / split
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"scene_{seed}.npz"
    # A hard crash may leave an episode beyond the committed checkpoint cursor.
    # Such deterministic replays MUST match, never silently overwrite evidence.
    arrays = {"observations": obs, "indices": indices, "energy_to_end": energy[indices],
              "safe_return": safe[indices], "step_energy": np.asarray(data["energy"], np.float64),
              "step_contact": np.asarray(data["contact"], np.uint8)}
    if destination.exists():
        with np.load(destination) as saved:
            if any(not np.array_equal(saved[k], v) for k, v in arrays.items()):
                raise ValueError("replayed episode differs from existing evidence")
    else:
        temporary = destination.with_suffix(".npz.tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(temporary, destination)
    record = {**episode, "scene_split": split, "samples": len(indices),
              "labels_complete": True, "goal_role": "actual_known_charging_station"}
    record_path = destination.with_suffix(".json")
    if record_path.exists():
        if json.loads(record_path.read_text()) != record:
            raise ValueError("replayed episode metadata differs from existing evidence")
    else:
        atomic_json(record_path, record)
    return {"seed": seed, "split": split, "samples": len(indices),
            "path": str(destination.relative_to(root)), "steps": len(energy),
            "safe_goal": bool(episode["safe_goal"]), "goal_reached": bool(episode["goal_reached"])}


def save_state(root: Path, vec, obs, pending, rows, steps, contract, seconds) -> None:
    directory = root / f"checkpoint_{steps:09d}"
    directory.mkdir(exist_ok=True)
    state = {"observations": obs, "pending": pending, "rows": rows, "steps": steps,
             "workers": vec.env_method("snapshot"), "seconds": seconds, "contract": contract}
    temporary = directory / "state.pt.tmp"
    torch.save(state, temporary)
    os.replace(temporary, directory / "state.pt")
    atomic_json(directory / "metadata.json", {"steps": steps, "rows": len(rows),
                "state_size": (directory / "state.pt").stat().st_size})
    atomic_json(root / "latest.json", {"checkpoint": directory.name})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=500000)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--checkpoint-steps", type=int, default=8192)
    parser.add_argument("--sample-stride", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1040000001)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if (min(args.steps, args.num_envs, args.checkpoint_steps, args.sample_stride) <= 0
            or args.steps % args.num_envs or args.checkpoint_steps % args.num_envs):
        raise ValueError("positive aligned collection budgets required")
    root, source = args.output_dir.resolve(), args.source.resolve()
    if root == source or source in root.parents:
        raise ValueError("new energy output must be separate from navigation")
    checkpoint, training = validated_checkpoint(source, 131072)
    paths = set(training["source_hashes"]) | {
        "scripts/collect_new_navigation_return_energy.py",
        "experiments/directional_navigation/return_energy_data.py",
        "scripts/evaluate_hocbf_correction_fixed500.py",
        "scripts/evaluate_deployable_observation_v2.py",
        "scripts/resume_recovery_sac_ppo_stratified.py"}
    horizon = 7 if args.smoke else training["arguments"]["horizon"]
    contract = {"protocol": "new_navigation_return_energy_v1", "formal": not args.smoke,
                "model": str(checkpoint / "model.zip"), "model_sha256": file_hash(checkpoint / "model.zip"),
                "source_hashes": {p: file_hash(ROOT / p) for p in sorted(paths)},
                "num_envs": args.num_envs, "sample_stride": args.sample_stride,
                "seed": args.seed, "horizon": horizon, "obstacles": training["arguments"]["obstacles"],
                "device": args.device, "checkpoint_steps": args.checkpoint_steps,
                "observation_dim": 2056, "hocbf": True, "navigation_updates": 0,
                "initial_velocity": "uniform([-8,-8,-2],[8,8,2])m/s",
                "collection_domain": "fresh legal start to actual charger; not mission switching",
                "battery_kill": False, "physical_disturbance_added": False,
                "budget_counts": "all real environment transitions; not optimizer updates",
                "labels": "future-contact suffix safe arrival; unfinished tails censored",
                "collision_contract": training["collision_contract"]}
    if args.resume:
        if json.loads((root / "manifest.json").read_text())["contract"] != contract:
            raise ValueError("energy resume contract mismatch")
    else:
        root.mkdir(parents=True, exist_ok=False)
        atomic_json(root / "manifest.json", {"contract": contract, "planned_steps": args.steps,
                    "created_unix": time.time()})
    if (root / "PAUSE").exists():
        raise ValueError("PAUSE marker still present")
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    torch.set_num_threads(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    vec = None
    try:
        model = SAC.load(checkpoint / "model.zip", device=args.device)
        model.policy.set_training_mode(False)
        for parameter in model.policy.parameters():
            parameter.requires_grad_(False)
        before_hash = policy_hash(model)
        vec = SubprocVecEnv([partial(ReturnEnergyRecovery, seed_start=args.seed + i,
                       seed_stride=args.num_envs, horizon=horizon,
                       obstacles=contract["obstacles"], hocbf=True)
                       for i in range(args.num_envs)], start_method="forkserver")
        steps, elapsed, rows = 0, 0., []
        pending = [empty_episode() for _ in range(args.num_envs)]
        if args.resume:
            pointer = json.loads((root / "latest.json").read_text())["checkpoint"]
            if Path(pointer).name != pointer:
                raise ValueError("invalid checkpoint pointer")
            directory = root / pointer
            metadata = json.loads((directory / "metadata.json").read_text())
            if (directory / "state.pt").stat().st_size != metadata["state_size"]:
                raise ValueError("incomplete energy checkpoint")
            saved = torch.load(directory / "state.pt", map_location="cpu", weights_only=False)
            if saved["contract"] != contract:
                raise ValueError("energy checkpoint contract mismatch")
            for i, state in enumerate(saved["workers"]):
                vec.env_method("restore", state, indices=[i])
            obs, pending, rows = saved["observations"], saved["pending"], saved["rows"]
            steps, elapsed = saved["steps"], saved["seconds"]
        else:
            obs = vec.reset()
        if steps > args.steps:
            raise ValueError("requested budget precedes restored checkpoint")
        started = time.monotonic()
        committed = steps
        atomic_json(root / "status.json", {"status": "COLLECTING", "pid": os.getpid(),
                    "steps": steps, "target": args.steps, "completed_episodes": len(rows)})
        while steps < args.steps and not STOP and not (root / "PAUSE").exists():
            if obs.shape != (args.num_envs, 2056) or not np.isfinite(obs).all():
                raise ValueError("invalid actor observation")
            for i, data in enumerate(pending):
                index = len(data["energy"])
                if index % args.sample_stride == 0:
                    data["observations"].append(obs[i].copy())
                    data["indices"].append(index)
            actions, _ = model.predict(obs, deterministic=True)
            if not np.isfinite(actions).all():
                raise ValueError("nonfinite frozen action")
            obs, _, dones, infos = vec.step(actions)
            steps += args.num_envs
            for i, info in enumerate(infos):
                energy = float(info["realized_energy"])
                if not np.isfinite(energy) or energy < 0:
                    raise ValueError("invalid measured energy")
                pending[i]["energy"].append(energy)
                pending[i]["contact"].append(int(info["cost"]))
                if dones[i]:
                    episode = dict(info["navigation_episode"])
                    episode.pop("outcome", None)
                    rows.append(commit_episode(root, pending[i], episode))
                    pending[i] = empty_episode()
            if steps % args.checkpoint_steps == 0 or steps == args.steps:
                seconds = elapsed + time.monotonic() - started
                save_state(root, vec, obs, pending, rows, steps, contract, seconds)
                committed = steps
                atomic_json(root / "status.json", {"status": "COLLECTING", "pid": os.getpid(),
                            "steps": steps, "target": args.steps, "completed_episodes": len(rows),
                            "samples": sum(r["samples"] for r in rows), "seconds": seconds,
                            "checkpoint_steps": committed, "navigation_updates": 0})
        if committed != steps:
            save_state(root, vec, obs, pending, rows, steps, contract,
                       elapsed + time.monotonic() - started)
        if policy_hash(model) != before_hash:
            raise RuntimeError("frozen navigation weights changed")
        censored = sum(len(d["energy"]) for d in pending)
        if sum(r["steps"] for r in rows) + censored != steps:
            raise RuntimeError("completed/censored transition budget mismatch")
        atomic_json(root / "dataset_index.json", {"episodes": rows,
                    "completed_steps": steps - censored, "censored_steps": censored,
                    "steps": steps, "policy_hash_before": before_hash,
                    "policy_hash_after": policy_hash(model)})
        atomic_json(root / "status.json", {
            "status": "COLLECTION_COMPLETE_AWAITING_ANALYSIS" if steps == args.steps else "PAUSED",
            "pid": os.getpid(), "steps": steps, "target": args.steps,
            "completed_episodes": len(rows), "censored_steps": censored,
            "seconds": elapsed + time.monotonic() - started, "navigation_updates": 0,
            "automatic_model_fit": False, "automatic_mission_experiment": False})
    except Exception:
        atomic_json(root / "ERROR.json", {"error": traceback.format_exc()})
        raise
    finally:
        if vec is not None:
            vec.close()


if __name__ == "__main__":
    main()
