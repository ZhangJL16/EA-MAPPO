"""Paired, atomic, resumable interface audit before any R3-RACT fine-tuning."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
import torch
from stable_baselines3.common.vec_env import SubprocVecEnv

from envs.UAVEnergyDeliverySAC import UAVEnergyDeliverySACEnv, SACTrainingPhase
from experiments.jacobian_energy_bridge.sac import JacobianBridgeSAC
from experiments.r3_collision.core import correction_distance
from scripts.run_jacobian_safety_energy_1m import parse_args as base_args
from scripts.train_uav_energy_delivery_sac import (
    _configure_navigation_worker_threads,
    environment_kwargs_from_args,
    generate_stratified_navigation_tasks,
)

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/r3_sac_accelerated_reproduction_seed0_500k_formal500_20260902_v1"


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def environment_contract() -> dict:
    config = json.loads((BASE / "config.json").read_text())
    args = base_args(config["exact_command"][1:])
    return environment_kwargs_from_args(args, phase=SACTrainingPhase.NAVIGATION)


class AuditEnv(UAVEnergyDeliverySACEnv):
    def __init__(self, kwargs: dict, jobs: list[dict]):
        super().__init__(**kwargs)
        self.jobs = jobs
        self.job_index = -1
        self.idle = False
        self.correction_integral = 0.0
        self.measurements: list[float] = []

    def reset(self, *, seed=None, options=None):
        self.job_index += 1
        if self.job_index >= len(self.jobs):
            self.idle = True
            return np.zeros(self.observation_space.shape, np.float32), {}
        job = self.jobs[self.job_index]
        self.cbf_enabled = job["arm"] != "off"
        self.hocbf_sampled_data_robust = job["arm"] == "robust"
        if self.safety_filter is not None:
            from dataclasses import replace
            self.safety_filter.config = replace(
                self.safety_filter.config, sampled_data_robust=job["arm"] == "robust"
            )
        observation, info = super().reset(seed=job["world_seed"], options=job["options"])
        if self.safety_filter is None or self.safety_filter.config.sampled_data_robust != self.hocbf_sampled_data_robust:
            raise AssertionError("audit arm does not match active filter")
        self.initial_observation_hash = hashlib.sha256(observation.tobytes()).hexdigest()
        self.correction_integral = 0.0
        self.measurements = []
        self.counts = dict(intervention=0, emergency=0, fallback=0, invalid_slack=0, physical_change=0)
        self.obstacle_hash = hashlib.sha256(
            json.dumps(self.static_obstacle_layout(), sort_keys=True, default=lambda x: x.tolist()).encode()
        ).hexdigest()
        return observation, {}

    def _safety_filtered_action(self, nominal_action):
        executed, diagnostics = super()._safety_filtered_action(nominal_action)
        if hasattr(self, "measurements"):
            d = correction_distance(
                self._normalized_action_to_acceleration(nominal_action),
                self._normalized_action_to_acceleration(executed),
                self.horizontal_a_max, self.vertical_a_max,
            )
            self.measurements.append(d)
        return executed, diagnostics

    def step(self, action):
        if self.idle:
            return np.zeros(self.observation_space.shape, np.float32), 0.0, True, False, {"idle": True}
        self.measurements = []
        obs, reward, terminated, truncated, info = super().step(action)
        if len(self.measurements) != int(info["physics_substeps"]):
            raise AssertionError("correction samples do not match executed physics substeps")
        integral = float(np.square(self.measurements).sum() * self.physics_dt)
        self.correction_integral += integral
        self.counts["intervention"] += int(info["hocbf_intervened"])
        self.counts["emergency"] += int(info["hocbf_emergency_brake"])
        self.counts["fallback"] += int(info["hocbf_fallback_used"])
        self.counts["physical_change"] += int(any(d > 1e-7 for d in self.measurements))
        self.counts["invalid_slack"] += int(any(
            row.get("minimum_executed_slack") is not None
            and row["minimum_executed_slack"] < -1e-6
            for row in info["hocbf_substep_diagnostics"]
        ))
        contact = bool(info["obstacle_collision"] or info["boundary_contact"])
        final = bool(terminated or truncated or contact)
        light = {"idle": False}
        if final:
            job = self.jobs[self.job_index]
            light["record"] = {
                "arm": job["arm"], "case": job["case"], "world_seed": job["world_seed"],
                "distance_bucket": job["distance_bucket"], "obstacle_hash": self.obstacle_hash,
                "initial_observation_hash": self.initial_observation_hash,
                "sampled_data_robust": self.safety_filter.config.sampled_data_robust,
                "safe_goal": bool(info["is_success"] and not contact),
                "contact": contact, "obstacle_contact": bool(info["obstacle_collision"]),
                "boundary_contact": bool(info["boundary_contact"]),
                "end_reason": "first_contact" if contact else info["end_reason"],
                "steps": self.current_step, "path_length": self._current_goal_path_length,
                "path_ratio": self._current_goal_path_length / job["distance"],
                "correction_integral": self.correction_integral,
                "old_emergency_kill_would_zero_goal": bool(info["is_success"] and self.counts["emergency"]),
                **self.counts,
            }
        return obs, reward, bool(terminated or contact), bool(truncated and not contact), light


def assess(records: list[dict]) -> dict:
    by_arm = {arm: sorted([r for r in records if r["arm"] == arm], key=lambda r: r["case"])
              for arm in ("ordinary", "robust")}
    old, new = by_arm["ordinary"], by_arm["robust"]
    if not old or len(old) != len(new):
        raise ValueError("incomplete paired audit")
    for a, b in zip(old, new):
        if (a["case"] != b["case"] or a["obstacle_hash"] != b["obstacle_hash"]
                or a["initial_observation_hash"] != b["initial_observation_hash"]):
            raise AssertionError("paired scene mismatch")
        if a["sampled_data_robust"] or not b["sampled_data_robust"]:
            raise AssertionError("paired filter configuration mismatch")
    summaries = {}
    for arm, rows in by_arm.items():
        steps = sum(r["steps"] for r in rows)
        safe = [r for r in rows if r["safe_goal"]]
        summaries[arm] = {
            "tasks": len(rows), "safe_goals": len(safe),
            "safe_goal_rate": len(safe) / len(rows),
            "contacts": sum(r["contact"] for r in rows),
            "timeouts": sum(not r["contact"] and not r["safe_goal"] for r in rows),
            "successful_mean_path_ratio": float(np.mean([r["path_ratio"] for r in safe])) if safe else None,
            **{key + "_step_rate": sum(r[key] for r in rows) / steps
               for key in ("intervention", "physical_change", "emergency", "fallback", "invalid_slack")},
            "safe_goals_with_emergency": sum(r["old_emergency_kill_would_zero_goal"] for r in rows),
        }
    gain = sum(b["safe_goal"] - a["safe_goal"] for a, b in zip(old, new))
    # Engineering feasibility gate, not a significance test or safety proof.
    checks = {
        "at_most_one_lost_safe_goal": gain >= -1,
        "contact_not_worse": summaries["robust"]["contacts"] <= summaries["ordinary"]["contacts"],
        "fallback_not_increased_over_five_points": (
            summaries["robust"]["fallback_step_rate"] <= summaries["ordinary"]["fallback_step_rate"] + .05
        ),
        "at_least_half_safe_goals": summaries["robust"]["safe_goal_rate"] >= .5,
    }
    safe_integrals = [r["correction_integral"] for r in new if r["safe_goal"] and r["correction_integral"] > 1e-12]
    # Half weight per typical complete successful trajectory, not every second.
    kappa = float(np.log(2) / np.median(safe_integrals)) if safe_integrals else 0.0
    success_lengths = [r["steps"] for r in new if r["safe_goal"]]
    gamma = float(.5 ** (1 / max(float(np.median(success_lengths)), 1))) if success_lengths else None
    checks["nonzero_continuous_signal"] = kappa > 0
    return {
        "decision": "READY_FOR_MATCHED_TRAINING" if all(checks.values()) else "STOP_BEFORE_TRAINING",
        "evidence": "DEVELOPMENT_INTERFACE_FEASIBILITY_NOT_CONFIRMATION",
        "checks": checks, "summary": summaries, "paired_safe_goal_count_difference": gain,
        "kappa": kappa, "gamma": gamma,
        "positive_hazard_safe_trajectory_weights": [float(np.exp(-kappa * z)) for z in safe_integrals],
    }


def run(args) -> None:
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = BASE / "phase1_navigation/checkpoint_transition_500000.zip"
    kwargs = environment_contract()
    if args.smoke_steps:
        kwargs["phase1_episode_max_policy_steps"] = args.smoke_steps
        kwargs["max_steps_per_task"] = args.smoke_steps
    tasks = generate_stratified_navigation_tasks(num_tasks=args.tasks, seed=args.task_seed)
    protocol = {
        "version": 2, "tasks": args.tasks, "task_seed": args.task_seed,
        "world_seed": args.world_seed, "checkpoint_sha256": sha256(checkpoint),
        "smoke_steps": args.smoke_steps, "first_contact_terminal": True,
        "gate": "gain>=-1; contact_not_worse; fallback_delta<=.05; safe_rate>=.5; nonzero_signal",
        "source_hashes": {p: sha256(ROOT / p) for p in (
            "scripts/run_r3_collision_interface_audit.py", "experiments/r3_collision/core.py",
            "envs/UAVEnergyDeliverySAC.py", "review_bundle/safety/collision/filter.py",
            "review_bundle/safety/collision/hocbf.py",
        )},
    }
    manifest = output / "protocol.json"
    if manifest.exists() and json.loads(manifest.read_text()) != protocol:
        raise RuntimeError("resume protocol/source mismatch; choose a fresh directory")
    atomic_json(manifest, protocol)
    jobs = []
    for i, task in enumerate(tasks):
        for arm in ("ordinary", "robust"):
            if (output / "records" / f"{i:03d}_{arm}.json").exists():
                continue
            jobs.append(dict(case=i, arm=arm, world_seed=args.world_seed+i,
                             distance=task.straight_line_distance, distance_bucket=task.distance_bucket,
                             options=dict(start_position=task.start_position,
                                          start_velocity=task.initial_velocity, task_point=task.goal_position)))
    started = time.monotonic()
    if jobs:
        workers = min(args.workers, len(jobs))
        factories = []
        for worker in range(workers):
            local_jobs = jobs[worker::workers]
            def factory(local_jobs=local_jobs):
                _configure_navigation_worker_threads()
                return AuditEnv(kwargs, local_jobs)
            factories.append(factory)
        model = JacobianBridgeSAC.load(checkpoint, device=args.device)
        model.policy.set_training_mode(False)
        vec = SubprocVecEnv(factories, start_method="forkserver")
        try:
            obs = vec.reset()
            remaining = len(jobs)
            ticks = 0
            while remaining:
                actions, _ = model.predict(obs, deterministic=True)
                obs, _, _, infos = vec.step(actions)
                ticks += 1
                for info in infos:
                    if "record" in info:
                        row = info["record"]
                        atomic_json(output / "records" / f"{row['case']:03d}_{row['arm']}.json", row)
                        remaining -= 1
                if ticks % 25 == 0 or remaining == 0:
                    status = dict(status="RUNNING" if remaining else "COLLECTION_COMPLETE",
                                  completed=2*args.tasks-remaining, total=2*args.tasks,
                                  wall_seconds=time.monotonic()-started, vector_steps=ticks,
                                  device=str(model.device), workers=workers, pid=os.getpid())
                    atomic_json(output / "PROGRESS.json", status)
                    if ticks == 25:
                        print(json.dumps(status), flush=True)
                if (output / "PAUSE_REQUEST").exists():
                    atomic_json(output / "PAUSED.json", {"completed": 2*args.tasks-remaining,
                                "resume": "same command; completed atomic episodes retained"})
                    return
        finally:
            vec.close()
    rows = [json.loads(p.read_text()) for p in sorted((output / "records").glob("*.json"))]
    result = assess(rows)
    if args.smoke_steps:
        result["decision"] = "SMOKE_ONLY"
    result["elapsed_this_invocation_seconds"] = time.monotonic()-started
    atomic_json(output / "RESULT.json", result)
    atomic_json(output / "COMPLETED.json", {"decision": result["decision"], "records": len(rows)})
    atomic_json(output / "PROGRESS.json", {"status": "COMPLETED", "completed": len(rows), "total": 2*args.tasks})
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--tasks", type=int, default=50)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--task-seed", type=int, default=510001)
    parser.add_argument("--world-seed", type=int, default=520001)
    parser.add_argument("--smoke-steps", type=int, default=0)
    args = parser.parse_args()
    if args.workers < 1 or args.tasks < 5 or args.tasks % 5 or args.smoke_steps < 0:
        parser.error("positive workers, tasks a positive multiple of 5, and nonnegative smoke steps required")
    _configure_navigation_worker_threads()
    try:
        run(args)
    except BaseException:
        atomic_json(Path(args.output) / "FAILED.json", {"traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
