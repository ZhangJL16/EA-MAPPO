from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import argparse
import json
import signal
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from stable_baselines3 import SAC

from envs.UAVEnergyDeliverySAC import SACTrainingPhase
from experiments.forked_action_safety.core import analytic_clearance
from experiments.rechargeability_safety.closed_loop import (
    geometry_action_features,
    temperature_scale,
)
from experiments.rechargeability_safety.core import MonotoneBudgetCritic
from experiments.uav_energy_parallel import (
    ParallelUAVEnvPool,
    WorkerReset,
    WorkerRetarget,
    WorkerStep,
)
from scripts.evaluate_certified_meet_fusion import paired_bootstrap
from scripts.evaluate_jseb_checkpoints import reconstruct_environment_args
from scripts.run_lidar_forked_action_data_gate import atomic_json, sha256_file, sha256_json
from scripts.train_uav_energy_delivery_sac import (
    DISTANCE_BUCKETS,
    NavigationTask,
    environment_from_args,
    environment_kwargs_from_args,
    generate_stratified_navigation_tasks,
)


PROTOCOL = "BOUNDED_RCPS_CLOSED_LOOP_PILOT_V1"
METHODS = ("ungated", "geometry_gate", "certified_meet_gate")
BUDGET_LEVELS = (0.20, 0.30, 0.40, 0.50, 0.60)
TASK_SEED = 830_001
WORLD_SEED = 840_001
BUDGET_SEED = 850_001
BOOTSTRAP_SEED = 860_001
_STOP_REQUESTED = False


def request_stop(_signum: int, _frame: object) -> None:
    global _STOP_REQUESTED
    _STOP_REQUESTED = True


@dataclass
class ActiveRollout:
    case: dict[str, object]
    method: str
    observation: np.ndarray
    position: np.ndarray
    velocity: np.ndarray
    current_step: int
    leg: int = 0
    total_steps: int = 0
    task_steps: int = 0
    return_steps: int = 0
    total_energy: float = 0.0
    task_completed: bool = False
    return_success: bool = False
    early_commit: bool = False
    commit_step: int | None = None
    commit_budget_fraction: float | None = None
    commit_score: float | None = None
    task_progress: float = 0.0
    collision: bool = False
    boundary_contact: bool = False
    energy_exhausted: bool = False
    censored: bool = False
    end_reason: str = ""
    hocbf_interventions: int = 0
    gate_decisions: int = 0
    gate_accepts: int = 0
    gate_score_sum: float = 0.0
    gate_score_min: float = float("inf")
    gate_score_max: float = float("-inf")
    gate_latencies: list[float] = field(default_factory=list)

    @property
    def case_index(self) -> int:
        return int(self.case["case_index"])

    @property
    def budget_fraction(self) -> float:
        return float(self.case["budget_fraction"])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact",
        type=Path,
        default=Path("artifacts/jseb_navigation_repair_r3r4_uv_seed0_20260828_193904/R3"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "artifacts/jseb_navigation_repair_r3r4_uv_seed0_20260828_193904/"
            "R3/phase1_navigation/checkpoint_transition_500000.zip"
        ),
    )
    parser.add_argument(
        "--freeze-dir",
        type=Path,
        default=Path("artifacts/certified_meet_fusion_freeze_150scenes_20260904_v2"),
    )
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        default=Path("artifacts/rcps_meet_calibration_450scenes_20260904_v1"),
    )
    parser.add_argument(
        "--confirmation-dir",
        type=Path,
        default=Path("artifacts/rcps_meet_confirmation_150scenes_20260904_v1"),
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("docs/BOUNDED_RCPS_CLOSED_LOOP_PILOT_PROTOCOL_V1.md"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument("--num-cases", type=int, default=75)
    parser.add_argument("--task-seed", type=int, default=TASK_SEED)
    parser.add_argument("--world-seed", type=int, default=WORLD_SEED)
    parser.add_argument("--budget-seed", type=int, default=BUDGET_SEED)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--max-policy-steps-per-leg", type=int, default=4000)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-max-total-steps", type=int, default=12)
    args = parser.parse_args(argv)
    if args.smoke:
        args.num_cases = 5
        args.num_workers = min(args.num_workers, 5)
    if args.num_cases <= 0 or args.num_cases % len(DISTANCE_BUCKETS):
        parser.error("--num-cases must be a positive multiple of five")
    if min(
        args.num_workers,
        args.max_policy_steps_per_leg,
        args.bootstrap_draws,
        args.torch_threads,
    ) <= 0:
        parser.error("worker, horizon, bootstrap, and thread counts must be positive")
    if args.smoke and args.smoke_max_total_steps <= 0:
        parser.error("--smoke-max-total-steps must be positive")
    if not args.smoke and (
        args.num_cases != 75
        or args.task_seed != TASK_SEED
        or args.world_seed != WORLD_SEED
        or args.budget_seed != BUDGET_SEED
        or args.max_policy_steps_per_leg != 4000
    ):
        parser.error(
            "formal pilot requires the frozen 75-case seeds and 4000-step horizon"
        )
    return args


def resolve_paths(args: argparse.Namespace) -> None:
    for name in (
        "artifact",
        "checkpoint",
        "freeze_dir",
        "calibration_dir",
        "confirmation_dir",
        "protocol",
        "output_dir",
    ):
        setattr(args, name, getattr(args, name).expanduser().resolve())


def validate_upstream(args: argparse.Namespace) -> tuple[dict[str, object], dict[str, object]]:
    required = (
        args.artifact / "config.json",
        args.checkpoint,
        args.freeze_dir / "frozen_meet.pt",
        args.freeze_dir / "RESULT.json",
        args.calibration_dir / "RESULT.json",
        args.confirmation_dir / "RESULT.json",
        args.protocol,
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    freeze_result = json.loads((args.freeze_dir / "RESULT.json").read_text(encoding="utf-8"))
    calibration = json.loads((args.calibration_dir / "RESULT.json").read_text(encoding="utf-8"))
    confirmation = json.loads((args.confirmation_dir / "RESULT.json").read_text(encoding="utf-8"))
    model_hash = sha256_file(args.freeze_dir / "frozen_meet.pt")
    checkpoint_hash = sha256_file(args.checkpoint)
    if calibration.get("status") != "FROZEN_FOR_RCPS_CONFIRMATION" or not calibration.get("promotable"):
        raise RuntimeError("RCPS calibration did not authorize confirmation")
    if confirmation.get("status") != "PROMOTE_TO_BOUNDED_CLOSED_LOOP_PILOT" or not confirmation.get("promotable"):
        raise RuntimeError("independent confirmation did not authorize this pilot")
    if calibration.get("frozen_checkpoint_sha256") != model_hash:
        raise RuntimeError("calibration critic hash mismatch")
    if confirmation.get("frozen_checkpoint_sha256") != model_hash:
        raise RuntimeError("confirmation critic hash mismatch")
    expected_checkpoint_hash = calibration.get("dataset", {}).get("checkpoint_sha256")
    if expected_checkpoint_hash != checkpoint_hash:
        raise RuntimeError("R3 checkpoint differs from the calibrated critic source")
    reports = calibration.get("reports", {})
    thresholds = {
        "geometry_gate": float(reports["geometry_action"]["threshold"]),
        "certified_meet_gate": float(reports["certified_meet"]["threshold"]),
    }
    metadata = {
        "checkpoint_sha256": checkpoint_hash,
        "frozen_checkpoint_sha256": model_hash,
        "calibration_result_sha256": sha256_file(args.calibration_dir / "RESULT.json"),
        "confirmation_result_sha256": sha256_file(args.confirmation_dir / "RESULT.json"),
        "protocol_sha256": sha256_file(args.protocol),
        "thresholds": thresholds,
        "capacity": float(calibration["dataset"]["capacity"]),
    }
    return metadata, calibration


def budget_assignments(tasks: list[NavigationTask], *, seed: int) -> list[float]:
    if len(tasks) % len(DISTANCE_BUCKETS):
        raise ValueError("task count is not distance-stratified")
    assignments = [float("nan")] * len(tasks)
    rng = np.random.default_rng(seed)
    for bucket, _, _ in DISTANCE_BUCKETS:
        indices = [index for index, task in enumerate(tasks) if task.distance_bucket == bucket]
        if len(indices) % len(BUDGET_LEVELS):
            # Smoke runs use one case per bucket; fix it to the middle supported budget.
            if len(indices) == 1:
                assignments[indices[0]] = 0.40
                continue
            raise ValueError("each distance bucket must balance the five budget levels")
        values = np.repeat(np.asarray(BUDGET_LEVELS, dtype=np.float64), len(indices) // 5)
        values = rng.permutation(values)
        for index, value in zip(indices, values, strict=True):
            assignments[index] = float(value)
    if not np.all(np.isfinite(assignments)):
        raise RuntimeError("budget assignment is incomplete")
    return assignments


def make_cases(
    args: argparse.Namespace,
    probe,
) -> dict[str, object]:
    tasks = generate_stratified_navigation_tasks(num_tasks=args.num_cases, seed=args.task_seed)
    budgets = budget_assignments(tasks, seed=args.budget_seed)
    cases: list[dict[str, object]] = []
    for index, (task, budget) in enumerate(zip(tasks, budgets, strict=True)):
        probe.reset(
            seed=args.world_seed + index,
            options={
                "start_position": task.start_position,
                "start_velocity": task.initial_velocity,
                "task_point": task.goal_position,
            },
        )
        layout = probe.static_obstacle_layout()
        cases.append(
            {
                "case_index": index,
                "task": task.as_dict(),
                "distance_bucket": task.distance_bucket,
                "budget_fraction": budget,
                "world_seed": args.world_seed + index,
                "obstacle_layout": layout,
                "obstacle_layout_sha256": sha256_json(layout),
            }
        )
    return {
        "protocol": PROTOCOL,
        "num_cases": args.num_cases,
        "task_seed": args.task_seed,
        "world_seed": args.world_seed,
        "budget_seed": args.budget_seed,
        "budget_levels": list(BUDGET_LEVELS),
        "charger_position": np.asarray(probe.charger_position, dtype=np.float32),
        "cases": cases,
    }


def load_or_make_cases(args: argparse.Namespace, probe) -> dict[str, object]:
    path = args.output_dir / "CASES.json"
    if args.resume:
        if not path.is_file():
            raise FileNotFoundError("resume requested but CASES.json is absent")
        document = json.loads(path.read_text(encoding="utf-8"))
        expected = (args.num_cases, args.task_seed, args.world_seed, args.budget_seed)
        actual = tuple(int(document[key]) for key in ("num_cases", "task_seed", "world_seed", "budget_seed"))
        if actual != expected or document.get("protocol") != PROTOCOL:
            raise RuntimeError("resume arguments do not match frozen cases")
        return document
    document = make_cases(args, probe)
    atomic_json(path, document)
    return document


def load_critics(args: argparse.Namespace, policy: SAC, device: torch.device):
    frozen = torch.load(
        args.freeze_dir / "frozen_meet.pt", map_location=device, weights_only=False
    )
    geometry_dim = int(frozen["geometry_dim"])
    embedding_dim = int(frozen["embedding_dim"])
    geometry = MonotoneBudgetCritic(geometry_dim, hidden_dim=96).to(device)
    direct = MonotoneBudgetCritic(geometry_dim + embedding_dim, hidden_dim=96).to(device)
    geometry.load_state_dict(frozen["geometry_state_dict"])
    direct.load_state_dict(frozen["direct_state_dict"])
    geometry.eval()
    direct.eval()
    encoder = policy.policy.actor.features_extractor.to(device).eval()
    return frozen, geometry, direct, encoder


def rollout_path(output: Path, case_index: int, method: str) -> Path:
    return output / "rollouts" / method / f"case_{case_index:03d}.json"


def progress_from_position(job: ActiveRollout, position: np.ndarray) -> float:
    task = job.case["task"]
    goal = np.asarray(task["goal_position"], dtype=np.float32)
    initial = float(task["straight_line_distance"])
    remaining = float(np.linalg.norm(goal - np.asarray(position, dtype=np.float32)))
    return float(np.clip(1.0 - remaining / max(initial, 1e-8), 0.0, 1.0))


def finish_row(job: ActiveRollout, capacity: float) -> dict[str, object]:
    unsafe = bool(job.collision or job.boundary_contact or job.energy_exhausted)
    latency = np.asarray(job.gate_latencies, dtype=np.float64)
    row = {
        "protocol": PROTOCOL,
        "case_index": job.case_index,
        "method": job.method,
        "distance_bucket": job.case["distance_bucket"],
        "world_seed": job.case["world_seed"],
        "obstacle_layout_sha256": job.case["obstacle_layout_sha256"],
        "budget_fraction": job.budget_fraction,
        "initial_available_energy": job.budget_fraction * capacity,
        "total_realized_energy": job.total_energy,
        "final_remaining_budget_fraction": job.budget_fraction - job.total_energy / capacity,
        "total_steps": job.total_steps,
        "task_steps": job.task_steps,
        "return_steps": job.return_steps,
        "task_completed": job.task_completed,
        "safe_return": bool(job.return_success and not unsafe),
        "mission_success": bool(job.task_completed and job.return_success and not unsafe),
        "early_commit": job.early_commit,
        "commit_step": job.commit_step,
        "commit_budget_fraction": job.commit_budget_fraction,
        "commit_score": job.commit_score,
        "task_progress": 1.0 if job.task_completed else job.task_progress,
        "collision": job.collision,
        "boundary_contact": job.boundary_contact,
        "energy_exhausted": job.energy_exhausted,
        "unsafe_terminal_event": unsafe,
        "censored": job.censored,
        "end_reason": job.end_reason,
        "hocbf_intervention_steps": job.hocbf_interventions,
        "hocbf_intervention_rate": job.hocbf_interventions / max(job.total_steps, 1),
        "gate_decisions": job.gate_decisions,
        "gate_accepts": job.gate_accepts,
        "gate_acceptance_rate": (
            job.gate_accepts / job.gate_decisions if job.gate_decisions else None
        ),
        "gate_score_mean": (
            job.gate_score_sum / job.gate_decisions if job.gate_decisions else None
        ),
        "gate_score_min": job.gate_score_min if job.gate_decisions else None,
        "gate_score_max": job.gate_score_max if job.gate_decisions else None,
        "gate_latency_count": int(latency.size),
        "gate_latency_p50_seconds": float(np.quantile(latency, 0.50)) if latency.size else None,
        "gate_latency_p95_seconds": float(np.quantile(latency, 0.95)) if latency.size else None,
        "gate_latency_max_seconds": float(np.max(latency)) if latency.size else None,
        "final_position": job.position,
        "final_velocity": job.velocity,
    }
    atomic_json(rollout_path(Path(job.case["output_dir"]), job.case_index, job.method), row)
    return row


def score_gates(
    jobs: list[ActiveRollout],
    nominal_actions: np.ndarray,
    *,
    frozen: dict[str, object],
    geometry_model: MonotoneBudgetCritic,
    direct_model: MonotoneBudgetCritic,
    encoder,
    probe,
    charger: np.ndarray,
    capacity: float,
    maximum_steps: int,
    device: torch.device,
) -> np.ndarray:
    if not jobs:
        return np.empty(0, dtype=np.float32)
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        started = time.perf_counter()
    contexts = []
    observations = []
    budgets = []
    for job, nominal in zip(jobs, nominal_actions, strict=True):
        task = job.case["task"]
        layout = list(job.case["obstacle_layout"])
        clearance = float(
            analytic_clearance(
                np.asarray(job.position, dtype=np.float64)[None, :],
                layout,
                safe_radius=float(probe.safe_radius),
            )[0]
        )
        contexts.append(
            geometry_action_features(
                position=job.position,
                velocity=job.velocity,
                task_goal=np.asarray(task["goal_position"], dtype=np.float32),
                charger_goal=charger,
                active_goal=np.asarray(task["goal_position"], dtype=np.float32),
                obstacle_layout=layout,
                nearest_clearance=clearance,
                leg=0,
                elapsed_policy_steps=min(job.current_step, maximum_steps - 1),
                maximum_steps=maximum_steps,
                nominal_action=nominal,
                proposed_action=nominal,
                world_extent=np.asarray([probe.length, probe.width, probe.height], dtype=np.float32),
                d_max=float(probe.d_max),
                horizontal_v_max=float(probe.horizontal_v_max),
                vertical_v_max=float(probe.vertical_v_max),
                lidar_max_range=float(probe.lidar_max_range),
            )
        )
        observations.append(job.observation)
        budgets.append(job.budget_fraction - job.total_energy / capacity)
    geometry_raw = np.asarray(contexts, dtype=np.float32)
    geometry_norm = (geometry_raw - np.asarray(frozen["geometry_mean"])) / np.asarray(
        frozen["geometry_scale"]
    )
    with torch.no_grad():
        observation_tensor = torch.as_tensor(np.asarray(observations), dtype=torch.float32, device=device)
        embedding = encoder(observation_tensor)
        embedding_norm = (
            embedding - torch.as_tensor(frozen["embedding_mean"], device=device)
        ) / torch.as_tensor(frozen["embedding_scale"], device=device)
        geometry_tensor = torch.as_tensor(geometry_norm, dtype=torch.float32, device=device)
        budget_tensor = torch.as_tensor(budgets, dtype=torch.float32, device=device)
        geometry_probability = geometry_model(geometry_tensor, budget_tensor).cpu().numpy()
        direct_probability = direct_model(
            torch.cat([geometry_tensor, embedding_norm], dim=1), budget_tensor
        ).cpu().numpy()
    geometry_probability = temperature_scale(
        geometry_probability, float(frozen["geometry_temperature"])
    )
    direct_probability = temperature_scale(
        direct_probability, float(frozen["direct_temperature"])
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    per_item_latency = (time.perf_counter() - started) / len(jobs)
    scores = np.asarray(
        [
            geometry_probability[index]
            if job.method == "geometry_gate"
            else min(geometry_probability[index], direct_probability[index])
            for index, job in enumerate(jobs)
        ],
        dtype=np.float32,
    )
    for job, score in zip(jobs, scores, strict=True):
        value = float(score)
        job.gate_decisions += 1
        job.gate_score_sum += value
        job.gate_score_min = min(job.gate_score_min, value)
        job.gate_score_max = max(job.gate_score_max, value)
        job.gate_latencies.append(per_item_latency)
    return scores


def collect_rollouts(
    args: argparse.Namespace,
    cases_document: dict[str, object],
    policy: SAC,
    probe,
    environment_kwargs: dict[str, object],
    frozen: dict[str, object],
    geometry_model: MonotoneBudgetCritic,
    direct_model: MonotoneBudgetCritic,
    encoder,
    metadata: dict[str, object],
    device: torch.device,
) -> tuple[list[dict[str, object]], bool]:
    completed: list[dict[str, object]] = []
    pending: deque[tuple[dict[str, object], str]] = deque()
    for raw_case in cases_document["cases"]:
        case = dict(raw_case)
        case["output_dir"] = str(args.output_dir)
        for method in METHODS:
            path = rollout_path(args.output_dir, int(case["case_index"]), method)
            if args.resume and path.is_file():
                completed.append(json.loads(path.read_text(encoding="utf-8")))
            else:
                pending.append((case, method))
    total = len(completed) + len(pending)
    if not pending:
        return completed, False
    charger = np.asarray(cases_document["charger_position"], dtype=np.float32)
    capacity = float(metadata["capacity"])
    thresholds = dict(metadata["thresholds"])
    workers = min(args.num_workers, len(pending))
    paused = False
    with ParallelUAVEnvPool(environment_kwargs, num_workers=workers) as pool:
        active: dict[int, ActiveRollout] = {}

        def assign(worker_ids: list[int]) -> None:
            requests = []
            assignments: list[tuple[int, dict[str, object], str]] = []
            for worker_id in worker_ids:
                if not pending or _STOP_REQUESTED:
                    break
                case, method = pending.popleft()
                task = case["task"]
                requests.append(
                    WorkerReset(
                        worker_id=worker_id,
                        seed=int(case["world_seed"]),
                        options={
                            "start_position": np.asarray(task["start_position"], dtype=np.float32),
                            "start_velocity": np.asarray(task["initial_velocity"], dtype=np.float32),
                            "task_point": np.asarray(task["goal_position"], dtype=np.float32),
                            "static_obstacles": list(case["obstacle_layout"]),
                        },
                    )
                )
                assignments.append((worker_id, case, method))
            if not requests:
                return
            resets = pool.reset_many(requests)
            for worker_id, case, method in assignments:
                result = resets[worker_id]
                active[worker_id] = ActiveRollout(
                    case=case,
                    method=method,
                    observation=result.observation,
                    position=result.position,
                    velocity=result.velocity,
                    current_step=result.current_step,
                )

        assign(list(range(workers)))
        while active:
            worker_ids = sorted(active)
            observations = np.asarray([active[index].observation for index in worker_ids])
            nominal_actions, _ = policy.predict(observations, deterministic=True)
            nominal_actions = np.asarray(nominal_actions, dtype=np.float32)
            gated_positions = [
                offset
                for offset, worker_id in enumerate(worker_ids)
                if active[worker_id].leg == 0 and active[worker_id].method != "ungated"
            ]
            if gated_positions:
                gated_jobs = [active[worker_ids[offset]] for offset in gated_positions]
                gate_scores = score_gates(
                    gated_jobs,
                    nominal_actions[gated_positions],
                    frozen=frozen,
                    geometry_model=geometry_model,
                    direct_model=direct_model,
                    encoder=encoder,
                    probe=probe,
                    charger=charger,
                    capacity=capacity,
                    maximum_steps=args.max_policy_steps_per_leg,
                    device=device,
                )
                retargets = []
                for position, score in zip(gated_positions, gate_scores, strict=True):
                    worker_id = worker_ids[position]
                    job = active[worker_id]
                    threshold = float(thresholds[job.method])
                    if float(score) >= threshold:
                        job.gate_accepts += 1
                        continue
                    job.early_commit = True
                    job.commit_step = job.total_steps
                    job.commit_budget_fraction = job.budget_fraction - job.total_energy / capacity
                    job.commit_score = float(score)
                    job.task_progress = progress_from_position(job, job.position)
                    job.leg = 1
                    retargets.append(
                        WorkerRetarget(
                            worker_id=worker_id,
                            seed=int(job.case["world_seed"]),
                            start_position=job.position,
                            start_velocity=job.velocity,
                            goal_position=charger,
                        )
                    )
                if retargets:
                    resets = pool.retarget_many(retargets)
                    for request in retargets:
                        job = active[request.worker_id]
                        result = resets[request.worker_id]
                        job.observation = result.observation
                        job.position = result.position
                        job.velocity = result.velocity
                        job.current_step = result.current_step
                    observations = np.asarray([active[index].observation for index in worker_ids])
                    nominal_actions, _ = policy.predict(observations, deterministic=True)
                    nominal_actions = np.asarray(nominal_actions, dtype=np.float32)
            results = pool.step_many(worker_ids, nominal_actions)
            finished: list[int] = []
            successful_task_retargets = []
            for worker_id in worker_ids:
                job = active[worker_id]
                result: WorkerStep = results[worker_id]
                energy = float(result.info.get("realized_energy_cost") or 0.0)
                job.total_energy += energy
                job.total_steps += 1
                if job.leg == 0:
                    job.task_steps += 1
                    job.task_progress = progress_from_position(job, result.position)
                else:
                    job.return_steps += 1
                job.position = result.position
                job.velocity = result.velocity
                job.observation = result.observation
                job.current_step = result.current_step
                job.hocbf_interventions += int(bool(result.info.get("hocbf_intervened", False)))
                job.collision = job.collision or bool(result.info.get("obstacle_collision", False))
                job.boundary_contact = job.boundary_contact or bool(result.info.get("boundary_contact", False))
                job.energy_exhausted = job.total_energy > job.budget_fraction * capacity + 1e-9
                if job.collision or job.boundary_contact or job.energy_exhausted:
                    job.end_reason = (
                        "energy_exhausted"
                        if job.energy_exhausted
                        else "obstacle_collision"
                        if job.collision
                        else "boundary_contact"
                    )
                    completed.append(finish_row(job, capacity))
                    finished.append(worker_id)
                    continue
                if result.terminated or result.truncated:
                    success = bool(result.info.get("is_success", False))
                    if job.leg == 0 and success:
                        job.task_completed = True
                        job.task_progress = 1.0
                        job.leg = 1
                        successful_task_retargets.append(
                            WorkerRetarget(
                                worker_id=worker_id,
                                seed=int(job.case["world_seed"]),
                                start_position=result.position,
                                start_velocity=np.zeros(3, dtype=np.float32),
                                goal_position=charger,
                            )
                        )
                    else:
                        job.return_success = bool(job.leg == 1 and success)
                        job.end_reason = str(result.info.get("end_reason") or "environment_terminal")
                        completed.append(finish_row(job, capacity))
                        finished.append(worker_id)
                        continue
                if args.smoke and job.total_steps >= args.smoke_max_total_steps:
                    job.censored = True
                    job.end_reason = "smoke_step_limit"
                    completed.append(finish_row(job, capacity))
                    finished.append(worker_id)
            if successful_task_retargets:
                resets = pool.retarget_many(successful_task_retargets)
                for request in successful_task_retargets:
                    if request.worker_id in finished:
                        continue
                    job = active[request.worker_id]
                    result = resets[request.worker_id]
                    job.observation = result.observation
                    job.position = result.position
                    job.velocity = result.velocity
                    job.current_step = result.current_step
            for worker_id in finished:
                del active[worker_id]
            if finished and not _STOP_REQUESTED:
                assign(finished)
            atomic_json(
                args.output_dir / "PROGRESS.json",
                {
                    "status": "STOP_REQUESTED" if _STOP_REQUESTED else "RUNNING",
                    "completed_rollouts": len(completed),
                    "total_rollouts": total,
                    "active_rollouts": len(active),
                    "pending_rollouts": len(pending),
                    "updated_unix": time.time(),
                },
            )
            if _STOP_REQUESTED:
                paused = True
                break
    return completed, paused


def summarize_rows(
    rows: list[dict[str, object]], *, draws: int, seed: int
) -> tuple[dict[str, object], dict[str, object], bool]:
    by_key = {(int(row["case_index"]), str(row["method"])): row for row in rows}
    case_ids = sorted({int(row["case_index"]) for row in rows})
    if len(by_key) != len(case_ids) * len(METHODS):
        raise RuntimeError("terminal records do not form complete paired cases")

    def block(selected: list[dict[str, object]]) -> dict[str, object]:
        bool_fields = (
            "task_completed",
            "safe_return",
            "mission_success",
            "early_commit",
            "unsafe_terminal_event",
            "collision",
            "boundary_contact",
            "energy_exhausted",
        )
        numeric_fields = (
            "task_progress",
            "total_realized_energy",
            "total_steps",
            "hocbf_intervention_rate",
        )
        report = {name: float(np.mean([bool(row[name]) for row in selected])) for name in bool_fields}
        report.update({name: float(np.mean([float(row[name]) for row in selected])) for name in numeric_fields})
        return report

    reports: dict[str, object] = {}
    for method in METHODS:
        selected = [by_key[(case_id, method)] for case_id in case_ids]
        reports[method] = {
            "overall": block(selected),
            "by_distance_bucket": {
                bucket: block([row for row in selected if row["distance_bucket"] == bucket])
                for bucket, _, _ in DISTANCE_BUCKETS
            },
            "by_budget_fraction": {
                f"{budget:.2f}": block(
                    [row for row in selected if abs(float(row["budget_fraction"]) - budget) < 1e-9]
                )
                for budget in sorted({float(row["budget_fraction"]) for row in selected})
            },
        }
    comparisons: dict[str, object] = {}
    metric_names = (
        "task_completed",
        "safe_return",
        "mission_success",
        "unsafe_terminal_event",
        "task_progress",
        "total_realized_energy",
    )
    for baseline in ("geometry_gate", "ungated"):
        comparison = {}
        for offset, metric in enumerate(metric_names):
            values = np.asarray(
                [
                    float(by_key[(case_id, "certified_meet_gate")][metric])
                    - float(by_key[(case_id, baseline)][metric])
                    for case_id in case_ids
                ]
            )
            comparison[metric] = paired_bootstrap(values, seed=seed + 20 * offset, draws=draws)
        comparisons[f"certified_meet_minus_{baseline}"] = comparison
    meet = reports["certified_meet_gate"]["overall"]
    geometry = reports["geometry_gate"]["overall"]
    ungated = reports["ungated"]["overall"]
    meet_rows = [by_key[(case_id, "certified_meet_gate")] for case_id in case_ids]
    maximum_p95 = max(
        float(row["gate_latency_p95_seconds"] or 0.0) for row in meet_rows
    )
    checks = {
        "all_225_paired_rollouts_complete": len(rows) == 225 and len(case_ids) == 75,
        "meet_intervention_nondegenerate": 0.05 <= float(meet["early_commit"]) <= 0.95,
        "meet_gate_p95_below_policy_period": maximum_p95 < 0.2,
        "meet_safe_return_noninferior_within_one_case": (
            float(meet["safe_return"]) >= float(geometry["safe_return"]) - 1.0 / 75.0 - 1e-12
        ),
        "meet_unsafe_not_worse_than_geometry": (
            float(meet["unsafe_terminal_event"]) <= float(geometry["unsafe_terminal_event"]) + 1e-12
        ),
        "meet_task_completion_not_below_geometry": (
            float(meet["task_completed"]) >= float(geometry["task_completed"]) - 1e-12
        ),
        "meet_utility_improves_over_geometry": (
            float(meet["task_completed"]) >= float(geometry["task_completed"]) + 1.0 / 75.0 - 1e-12
            or float(meet["task_progress"]) >= float(geometry["task_progress"]) + 0.01 - 1e-12
        ),
        "meet_safety_improves_over_ungated": (
            float(meet["safe_return"]) >= float(ungated["safe_return"]) + 1.0 / 75.0 - 1e-12
            or float(meet["unsafe_terminal_event"])
            <= float(ungated["unsafe_terminal_event"]) - 1.0 / 75.0 + 1e-12
        ),
    }
    return reports, {"paired_bootstrap": comparisons, "checks": checks}, bool(all(checks.values()))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    resolve_paths(args)
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.resume:
        raise FileExistsError(f"output directory is not fresh: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata, _calibration = validate_upstream(args)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device.type == "cuda":
        torch.cuda.set_device(0 if device.index is None else device.index)
    torch.set_num_threads(args.torch_threads)
    environment_args, reconstruction = reconstruct_environment_args(
        args.artifact, device=args.device, seed=args.world_seed
    )
    environment_args.phase1_episode_max_steps = args.max_policy_steps_per_leg
    probe = environment_from_args(environment_args, phase=SACTrainingPhase.NAVIGATION)
    policy = SAC.load(args.checkpoint, env=probe, device=args.device, print_system_info=False)
    frozen, geometry_model, direct_model, encoder = load_critics(args, policy, device)
    cases_document = load_or_make_cases(args, probe)
    environment_kwargs = environment_kwargs_from_args(
        environment_args, phase=SACTrainingPhase.NAVIGATION
    )
    manifest = {
        "status": "RUNNING",
        "protocol": PROTOCOL,
        "formal_evidence": False,
        "smoke": args.smoke,
        "artifact": args.artifact,
        "checkpoint": args.checkpoint,
        "freeze_dir": args.freeze_dir,
        "calibration_dir": args.calibration_dir,
        "confirmation_dir": args.confirmation_dir,
        "protocol_document": args.protocol,
        **metadata,
        "num_cases": args.num_cases,
        "num_rollouts": args.num_cases * len(METHODS),
        "methods": list(METHODS),
        "task_seed": args.task_seed,
        "world_seed": args.world_seed,
        "budget_seed": args.budget_seed,
        "bootstrap_seed": args.bootstrap_seed,
        "num_workers": args.num_workers,
        "maximum_policy_steps_per_leg": args.max_policy_steps_per_leg,
        "cases_sha256": sha256_file(args.output_dir / "CASES.json"),
        "environment_reconstruction": reconstruction,
        "exact_command": [sys.executable, *sys.argv],
        "started_unix": time.time(),
    }
    if args.resume and (args.output_dir / "RUNNING.json").is_file():
        previous = json.loads((args.output_dir / "RUNNING.json").read_text(encoding="utf-8"))
        for key in (
            "checkpoint_sha256",
            "frozen_checkpoint_sha256",
            "calibration_result_sha256",
            "confirmation_result_sha256",
            "protocol_sha256",
            "cases_sha256",
        ):
            if previous.get(key) != manifest.get(key):
                raise RuntimeError(f"resume provenance mismatch: {key}")
        manifest["original_started_unix"] = previous.get(
            "original_started_unix", previous.get("started_unix")
        )
    atomic_json(args.output_dir / "RUNNING.json", manifest)
    (args.output_dir / "PAUSED.json").unlink(missing_ok=True)
    (args.output_dir / "FAILED.json").unlink(missing_ok=True)
    started = time.time()
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    try:
        rows, paused = collect_rollouts(
            args,
            cases_document,
            policy,
            probe,
            environment_kwargs,
            frozen,
            geometry_model,
            direct_model,
            encoder,
            metadata,
            device,
        )
        if paused:
            atomic_json(
                args.output_dir / "PAUSED.json",
                {
                    "status": "PAUSED_RESUMABLE",
                    "completed_rollouts": len(rows),
                    "total_rollouts": args.num_cases * len(METHODS),
                    "wall_clock_seconds_this_invocation": time.time() - started,
                },
            )
            return 130
        if args.smoke:
            result = {
                "status": "SMOKE_COMPLETED_NOT_EVIDENCE",
                "promotable": False,
                "protocol": PROTOCOL,
                "num_rollouts": len(rows),
                "all_censored": all(bool(row["censored"]) for row in rows),
                "manifest_sha256": sha256_file(args.output_dir / "RUNNING.json"),
                "wall_clock_seconds": time.time() - started,
            }
        else:
            reports, evaluation, promotable = summarize_rows(
                rows, draws=args.bootstrap_draws, seed=args.bootstrap_seed
            )
            result = {
                "status": "PROMOTE_TO_FORMAL_CLOSED_LOOP_EVALUATION" if promotable else "DO_NOT_PROMOTE",
                "promotable": promotable,
                "protocol": PROTOCOL,
                "formal_evidence": False,
                "num_cases": args.num_cases,
                "num_rollouts": len(rows),
                "reports": reports,
                **evaluation,
                "manifest_sha256": sha256_file(args.output_dir / "RUNNING.json"),
                "cases_sha256": sha256_file(args.output_dir / "CASES.json"),
                "wall_clock_seconds": time.time() - started,
            }
        atomic_json(args.output_dir / "RESULT.json", result)
        atomic_json(
            args.output_dir / "COMPLETED.json",
            {"status": result["status"], "finished_unix": time.time()},
        )
        return 0
    except BaseException as error:
        atomic_json(
            args.output_dir / "FAILED.json",
            {
                "status": "FAILED_RESUMABLE",
                "error_type": type(error).__name__,
                "error": str(error),
                "wall_clock_seconds_this_invocation": time.time() - started,
            },
        )
        raise
    finally:
        probe.close()


if __name__ == "__main__":
    raise SystemExit(main())
