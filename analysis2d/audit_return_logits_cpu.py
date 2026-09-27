"""Paired return-action probabilities on exact archived control trajectories.

Each frozen control checkpoint is replayed on validation maps 32–39. At every
flight decision, both final policies are queried on the *same* observation.
The feedback model is evaluated counterfactually; only control drives the
plant. This is inference-only and leaves the holdout split untouched.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from hashlib import sha256
import json
import os
from pathlib import Path
import re

import numpy as np
import torch
from stable_baselines3 import PPO

from dual_constraint_2d.calibration import DEFAULT_OUTPUT as CALIBRATION_OUTPUT
from dual_constraint_2d.gym_adapter import DualConstraintGym
from learning2d.mode_policy import ModeMaskedPolicy  # noqa: F401
from .collect_shield_feedback_cpu import (MAPS, PANELS, SEEDS, action_counts,
                                           evaluation_dir, read_json, training_dir)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "evidence" / "shield_feedback_cpu_20260927" / "logit_audit"
ACTION_ID = re.compile(rb'^\{"action_id": (\d+),')


def _prob9(model: PPO, observation: np.ndarray) -> float:
    with torch.no_grad():
        tensor = torch.as_tensor(observation, dtype=torch.float32,
                                 device=model.device).unsqueeze(0)
        return float(model.policy.get_distribution(tensor).distribution.probs[0, 9])


def _archived_actions(path: Path) -> list[int]:
    result = []
    with path.open("rb") as stream:
        for line in stream:
            match = ACTION_ID.match(line)
            if match is None:
                raise ValueError(f"malformed archived event in {path}")
            result.append(int(match.group(1)))
    return result


def one_map(panel: str, seed: int, map_id: int) -> dict:
    torch.set_num_threads(1)
    paths = {}
    models = {}
    for condition in ("control", "feedback"):
        train = training_dir(panel, seed, condition)
        status = read_json(train / "status.json")
        if not status["complete"] or status["timesteps"] != 25600:
            raise ValueError("final training checkpoint required")
        paths[condition] = train / status["latest_model"]
        models[condition] = PPO.load(str(paths[condition]), device="cpu")
    archive = evaluation_dir(panel, seed, "control", "shielded", map_id)
    expected = _archived_actions(archive / "events.jsonl")
    expected_summary = read_json(archive / "summary.json")
    calibration = read_json(CALIBRATION_OUTPUT / "calibration.json")
    env = DualConstraintGym((map_id,), calibration["capacity_synthetic_energy"],
                            calibration["full_charge_seconds"], shielded=True)
    observations = []
    try:
        observation, _ = env.reset(seed=map_id, options={"map_id": map_id})
        for decision, expected_action in enumerate(expected):
            control_action, _ = models["control"].predict(observation, deterministic=True)
            control_action = int(control_action)
            if control_action != expected_action:
                raise ValueError(f"replay action differs at {panel}/{seed}/{map_id}/{decision}")
            if observation[11] > 0.5:
                feedback_action, _ = models["feedback"].predict(observation, deterministic=True)
                row = {
                    "decision": decision, "time_s": float(env.env.time_s),
                    "energy_fraction": float(observation[8]),
                    "control_action": control_action,
                    "feedback_action": int(feedback_action),
                    "p_return_control": _prob9(models["control"], observation),
                    "p_return_feedback": _prob9(models["feedback"], observation),
                }
            else:
                row = None
            observation, _, _, _, info = env.step(control_action)
            if row is not None:
                row["immediate_takeover"] = any(
                    item.get("event") == "return_takeover" for item in info["plant_trace"]
                )
                observations.append(row)
        if (not env.env.done or env.env.completed_targets != expected_summary["completed_targets"]
                or env.env.return_takeovers != expected_summary["return_takeovers"]
                or env.env.charge_events != expected_summary["charge_events"]
                or abs(env.env.time_s - expected_summary["simulated_seconds"]) > 1e-8
                or len(expected) != expected_summary["policy_decisions"]):
            raise ValueError(f"replay summary differs at {panel}/{seed}/{map_id}")
    finally:
        env.close()
    return {
        "panel": panel, "seed": seed, "map_id": map_id,
        "control_model_sha256": sha256(paths["control"].read_bytes()).hexdigest(),
        "feedback_model_sha256": sha256(paths["feedback"].read_bytes()).hexdigest(),
        "archived_control_action_counts": action_counts(archive / "events.jsonl"),
        "archived_control_summary_sha256": sha256((archive / "summary.json").read_bytes()).hexdigest(),
        "flight_observations": observations,
    }


def main() -> None:
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = "1"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    jobs = [(panel, seed, map_id) for panel, seeds in SEEDS.items()
            for seed in seeds for map_id in MAPS]
    with ProcessPoolExecutor(max_workers=16) as pool:
        futures = {pool.submit(one_map, *job): job for job in jobs}
        for future in as_completed(futures):
            result = future.result()
            path = OUTPUT / f"{result['panel']}_seed_{result['seed']}_map_{result['map_id']:03d}.json"
            temp = path.with_suffix(".json.tmp")
            temp.write_text(json.dumps(result, sort_keys=True) + "\n", encoding="utf-8")
            os.replace(temp, path)
            print(f"verified {result['panel']} seed {result['seed']} map {result['map_id']}", flush=True)
    if len(list(OUTPUT.glob("*.json"))) != len(jobs):
        raise ValueError("missing or extra logit audit job")


if __name__ == "__main__":
    main()
