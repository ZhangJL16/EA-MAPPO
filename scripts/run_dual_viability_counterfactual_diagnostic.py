"""Collect return-now and one-executed-action-then-return counterfactuals."""

from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import argparse
import json
import signal
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from stable_baselines3 import SAC

from envs.UAVEnergyDeliverySAC import SACTrainingPhase
from experiments.rechargeability_safety.dual_viability import (
    budget_labels,
    make_job_keys,
    paired_semantic_summary,
)
from experiments.uav_energy_parallel import (
    ParallelUAVEnvPool,
    WorkerForkReset,
    WorkerInPlaceRetarget,
)
from scripts.evaluate_jseb_checkpoints import reconstruct_environment_args
from scripts.run_lidar_forked_action_data_gate import atomic_json, observation_sha256, sha256_file
from scripts.train_uav_energy_delivery_sac import environment_from_args, environment_kwargs_from_args


PROTOCOL = "DUAL_VIABILITY_COUNTERFACTUAL_DIAGNOSTIC_V1"
_STOP_REQUESTED = False


def request_stop(_signum: int, _frame: object) -> None:
    global _STOP_REQUESTED
    _STOP_REQUESTED = True


def json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


@dataclass(frozen=True)
class Job:
    anchor: dict[str, object]
    arm: str
    candidate_index: int | None = None
    candidate_name: str | None = None
    candidate_action: np.ndarray | None = None

    @property
    def anchor_id(self) -> str:
        return str(self.anchor["anchor_id"])


@dataclass
class Active:
    job: Job
    observation: np.ndarray
    position: np.ndarray
    velocity: np.ndarray
    phase: str
    total_energy: float = 0.0
    total_steps: int = 0
    recovery_steps: int = 0
    first_executed_action: np.ndarray | None = None
    first_hocbf_intervened: bool | None = None
    collision: bool = False
    boundary_contact: bool = False
    return_success: bool = False
    censored: bool = False
    end_reason: str = ""


def result_path(output: Path, job: Job) -> Path:
    base = output / "rollouts" / job.anchor_id
    if job.arm == "return_now":
        return base / "return_now.json"
    return base / "option" / f"{job.candidate_name}.json"


def jobs_from_anchors(anchors: list[dict[str, object]]) -> list[Job]:
    # Validate deterministic key construction in the pure module first.
    expected_keys = make_job_keys(anchors)
    jobs: list[Job] = []
    for anchor in anchors:
        if int(anchor["leg"]) != 0:
            continue
        jobs.append(Job(anchor=anchor, arm="return_now"))
        for index, (name, action) in enumerate(
            zip(anchor["candidate_names"], anchor["candidate_actions"], strict=True)
        ):
            jobs.append(
                Job(
                    anchor=anchor,
                    arm="option",
                    candidate_index=index,
                    candidate_name=str(name),
                    candidate_action=np.asarray(action, dtype=np.float32),
                )
            )
    actual_keys = [
        (job.anchor_id, "return_now" if job.arm == "return_now" else f"option/{job.candidate_name}")
        for job in jobs
    ]
    if actual_keys != expected_keys:
        raise RuntimeError("job construction is not deterministic")
    return jobs


def finish_row(output: Path, active: Active) -> dict[str, object]:
    job = active.job
    structurally_safe = bool(
        active.return_success
        and not active.collision
        and not active.boundary_contact
        and not active.censored
    )
    correction = None
    if job.candidate_action is not None and active.first_executed_action is not None:
        correction = float(np.linalg.norm(active.first_executed_action - job.candidate_action))
    row = {
        "protocol": PROTOCOL,
        "arm": job.arm,
        "anchor_id": job.anchor_id,
        "anchor_index": int(job.anchor["anchor_index"]),
        "scene_index": int(job.anchor["scene_index"]),
        "distance_bucket": str(job.anchor["distance_bucket"]),
        "source_transition_index": int(job.anchor["source_transition_index"]),
        "candidate_index": job.candidate_index,
        "candidate_name": job.candidate_name,
        "candidate_action": job.candidate_action,
        "first_executed_action": active.first_executed_action,
        "first_hocbf_intervened": active.first_hocbf_intervened,
        "first_action_correction_norm": correction,
        "candidate_steps": int(job.arm == "option"),
        "recovery_steps": active.recovery_steps,
        "total_steps": active.total_steps,
        "total_realized_energy": active.total_energy,
        "return_success": active.return_success,
        "structurally_safe_return": structurally_safe,
        "collision": active.collision,
        "boundary_contact": active.boundary_contact,
        "censored": active.censored,
        "end_reason": active.end_reason,
        "anchor_observation_sha256": str(job.anchor["anchor_observation_sha256"]),
        "obstacle_layout_sha256": str(job.anchor["obstacle_layout_sha256"]),
    }
    atomic_json(result_path(output, job), row)
    return row


def load_anchor_document(data_dir: Path) -> tuple[list[dict[str, object]], dict[str, object], dict[str, object]]:
    data_result = json.loads((data_dir / "RESULT.json").read_text(encoding="utf-8"))
    source_result = json.loads((Path(data_result["source"]) / "RESULT.json").read_text(encoding="utf-8"))
    document = json.loads((data_dir / "anchors.json").read_text(encoding="utf-8"))
    anchors = [dict(item) for item in document["anchors"]]
    with np.load(data_dir / "anchors.npz") as arrays:
        observations = np.asarray(arrays["observation"], dtype=np.float32)
    if observations.shape[0] != len(anchors):
        raise RuntimeError("anchor metadata/array length mismatch")
    for index, anchor in enumerate(anchors):
        if int(anchor["anchor_index"]) != index:
            # Canonical order is authoritative; old metadata indices are not.
            anchor["anchor_index"] = index
        if observation_sha256(observations[index]) != str(anchor["anchor_observation_sha256"]):
            raise RuntimeError(f"anchor hash mismatch: {anchor['anchor_id']}")
    return anchors, data_result, source_result


def collect(
    *,
    output: Path,
    jobs: list[Job],
    policy: SAC,
    environment_kwargs: dict[str, object],
    world_seed: int,
    workers: int,
    resume: bool,
    smoke_step_cap: int | None,
) -> tuple[list[dict[str, object]], bool]:
    pending: deque[Job] = deque()
    completed: list[dict[str, object]] = []
    for job in jobs:
        path = result_path(output, job)
        if resume and path.is_file():
            completed.append(json.loads(path.read_text(encoding="utf-8")))
        else:
            pending.append(job)
    total = len(completed) + len(pending)
    if not pending:
        return completed, False
    paused = False
    with ParallelUAVEnvPool(environment_kwargs, num_workers=min(workers, len(pending))) as pool:
        active: dict[int, Active] = {}

        def assign(worker_ids: list[int]) -> None:
            requests = []
            assignments: list[tuple[int, Job]] = []
            for worker_id in worker_ids:
                if not pending or _STOP_REQUESTED:
                    break
                job = pending.popleft()
                anchor = job.anchor
                requests.append(
                    WorkerForkReset(
                        worker_id=worker_id,
                        seed=world_seed + int(anchor["scene_index"]),
                        validation_start_position=np.asarray(anchor["validation_start_position"], dtype=np.float32),
                        anchor_position=np.asarray(anchor["position"], dtype=np.float32),
                        anchor_velocity=np.asarray(anchor["velocity"], dtype=np.float32),
                        goal_position=np.asarray(anchor["active_goal"], dtype=np.float32),
                        static_obstacles=list(anchor["obstacle_layout"]),
                        elapsed_policy_steps=int(anchor["elapsed_policy_steps"]),
                    )
                )
                assignments.append((worker_id, job))
            if not requests:
                return
            resets = pool.fork_reset_many(requests)
            retargets = []
            for worker_id, job in assignments:
                reset = resets[worker_id]
                if observation_sha256(reset.observation) != str(job.anchor["anchor_observation_sha256"]):
                    raise RuntimeError(f"fork observation mismatch: {job.anchor_id}")
                active[worker_id] = Active(
                    job=job,
                    observation=reset.observation,
                    position=reset.position,
                    velocity=reset.velocity,
                    phase="candidate" if job.arm == "option" else "recovery",
                )
                if job.arm == "return_now":
                    retargets.append(
                        WorkerInPlaceRetarget(
                            worker_id=worker_id,
                            goal_position=np.asarray(job.anchor["charger_goal"], dtype=np.float32),
                        )
                    )
            if retargets:
                before = {
                    request.worker_id: (
                        active[request.worker_id].position.copy(),
                        active[request.worker_id].velocity.copy(),
                    )
                    for request in retargets
                }
                retargeted = pool.retarget_in_place_many(retargets)
                for request in retargets:
                    state = active[request.worker_id]
                    reset = retargeted[request.worker_id]
                    if not np.array_equal(reset.position, before[request.worker_id][0]) or not np.array_equal(
                        reset.velocity, before[request.worker_id][1]
                    ):
                        raise RuntimeError("in-place return retarget changed physical state")
                    state.observation = reset.observation
                    state.position = reset.position
                    state.velocity = reset.velocity

        assign(list(range(pool.num_workers)))
        while active:
            worker_ids = sorted(active)
            actions = np.empty((len(worker_ids), 3), dtype=np.float32)
            recovery_offsets = []
            recovery_observations = []
            for offset, worker_id in enumerate(worker_ids):
                state = active[worker_id]
                if state.phase == "candidate":
                    actions[offset] = state.job.candidate_action
                else:
                    recovery_offsets.append(offset)
                    recovery_observations.append(state.observation)
            if recovery_offsets:
                predicted, _ = policy.predict(
                    np.asarray(recovery_observations, dtype=np.float32), deterministic=True
                )
                actions[np.asarray(recovery_offsets)] = np.asarray(predicted, dtype=np.float32)
            results = pool.step_many(worker_ids, actions, disable_cbf=False)
            retargets = []
            finished: list[int] = []
            for worker_id in worker_ids:
                state = active[worker_id]
                result = results[worker_id]
                energy = float(result.info.get("realized_energy_cost") or 0.0)
                state.total_energy += energy
                state.total_steps += 1
                state.position = result.position
                state.velocity = result.velocity
                state.observation = result.observation
                state.collision |= bool(result.info.get("obstacle_collision", False))
                state.boundary_contact |= bool(result.info.get("boundary_contact", False))
                if state.phase == "candidate":
                    state.first_executed_action = np.asarray(result.info["executed_action"], dtype=np.float32)
                    state.first_hocbf_intervened = bool(result.info.get("hocbf_intervened", False))
                    if state.collision or state.boundary_contact:
                        state.end_reason = "unsafe_on_candidate_step"
                        completed.append(finish_row(output, state))
                        finished.append(worker_id)
                    else:
                        state.phase = "recovery"
                        retargets.append(
                            WorkerInPlaceRetarget(
                                worker_id=worker_id,
                                goal_position=np.asarray(state.job.anchor["charger_goal"], dtype=np.float32),
                            )
                        )
                    continue
                state.recovery_steps += 1
                if state.collision or state.boundary_contact:
                    state.end_reason = "unsafe_during_recovery"
                    completed.append(finish_row(output, state))
                    finished.append(worker_id)
                    continue
                if smoke_step_cap is not None and state.total_steps >= smoke_step_cap:
                    state.censored = True
                    state.end_reason = "smoke_step_cap"
                    completed.append(finish_row(output, state))
                    finished.append(worker_id)
                    continue
                if result.terminated or result.truncated:
                    state.return_success = bool(result.info.get("is_success", False))
                    state.end_reason = str(result.info.get("end_reason") or "environment_terminal")
                    completed.append(finish_row(output, state))
                    finished.append(worker_id)
            if retargets:
                before = {
                    request.worker_id: (
                        active[request.worker_id].position.copy(),
                        active[request.worker_id].velocity.copy(),
                    )
                    for request in retargets
                }
                retargeted = pool.retarget_in_place_many(retargets)
                for request in retargets:
                    if request.worker_id in finished:
                        continue
                    state = active[request.worker_id]
                    reset = retargeted[request.worker_id]
                    if not np.array_equal(reset.position, before[request.worker_id][0]) or not np.array_equal(
                        reset.velocity, before[request.worker_id][1]
                    ):
                        raise RuntimeError("in-place option retarget changed physical state")
                    state.observation = reset.observation
                    state.position = reset.position
                    state.velocity = reset.velocity
            for worker_id in finished:
                del active[worker_id]
            if _STOP_REQUESTED:
                paused = True
                break
            if finished:
                assign(finished)
            atomic_json(
                output / "PROGRESS.json",
                {
                    "status": "RUNNING",
                    "completed_rollouts": len(completed),
                    "total_rollouts": total,
                    "active_rollouts": len(active),
                    "pending_rollouts": len(pending),
                    "updated_unix": time.time(),
                },
            )
    return completed, paused


def analyze(
    *,
    rows: list[dict[str, object]],
    data_dir: Path,
    capacity: float,
    budgets: np.ndarray,
) -> dict[str, object]:
    return_rows = {str(row["anchor_id"]): row for row in rows if row["arm"] == "return_now"}
    option_rows = sorted(
        (row for row in rows if row["arm"] == "option"),
        key=lambda row: (int(row["anchor_index"]), int(row["candidate_index"])),
    )
    anchor_ids = []
    candidate_names = []
    now_safe = []
    now_energy = []
    option_safe = []
    option_energy = []
    completion_safe = []
    completion_energy = []
    bucket_rows: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in option_rows:
        anchor_id = str(row["anchor_id"])
        candidate_name = str(row["candidate_name"])
        now = return_rows[anchor_id]
        old_path = data_dir / "branches" / anchor_id / f"{candidate_name}.json"
        old = json.loads(old_path.read_text(encoding="utf-8"))
        anchor_ids.append(anchor_id)
        candidate_names.append(candidate_name)
        now_safe.append(bool(now["structurally_safe_return"]))
        now_energy.append(float(now["total_realized_energy"]) / capacity)
        option_safe.append(bool(row["structurally_safe_return"]))
        option_energy.append(float(row["total_realized_energy"]) / capacity)
        completion_safe.append(bool(old["safe_recharge"]))
        completion_energy.append(float(old["total_realized_energy"]) / capacity)
        bucket_rows[str(row["distance_bucket"])].append(row)
    now_labels = budget_labels(
        structurally_safe=np.asarray(now_safe), energy_fraction=np.asarray(now_energy), budgets=budgets
    )
    option_labels = budget_labels(
        structurally_safe=np.asarray(option_safe), energy_fraction=np.asarray(option_energy), budgets=budgets
    )
    completion_labels = budget_labels(
        structurally_safe=np.asarray(completion_safe),
        energy_fraction=np.asarray(completion_energy),
        budgets=budgets,
    )
    semantic = paired_semantic_summary(
        anchor_ids=np.asarray(anchor_ids),
        candidate_names=np.asarray(candidate_names),
        return_now=now_labels,
        option=option_labels,
        completion=completion_labels,
        budgets=budgets,
    )
    correction = np.asarray(
        [float(row["first_action_correction_norm"]) for row in option_rows], dtype=np.float64
    )
    interventions = np.asarray(
        [bool(row["first_hocbf_intervened"]) for row in option_rows], dtype=np.bool_
    )
    by_bucket = {}
    for bucket, selected in bucket_rows.items():
        by_bucket[bucket] = {
            "rollouts": len(selected),
            "structurally_safe_rate": float(np.mean([bool(row["structurally_safe_return"]) for row in selected])),
            "collision_rate": float(np.mean([bool(row["collision"]) for row in selected])),
            "boundary_rate": float(np.mean([bool(row["boundary_contact"]) for row in selected])),
            "censor_rate": float(np.mean([bool(row["censored"]) for row in selected])),
            "mean_energy_fraction": float(np.mean([float(row["total_realized_energy"]) / capacity for row in selected])),
        }
    return {
        "budgets": budgets,
        "paired_budget_reports": semantic,
        "return_now": {
            "anchors": len(return_rows),
            "structurally_safe_rate": float(np.mean(now_safe)),
            "mean_energy_fraction": float(np.mean(now_energy)),
        },
        "option": {
            "rollouts": len(option_rows),
            "structurally_safe_rate": float(np.mean(option_safe)),
            "mean_energy_fraction": float(np.mean(option_energy)),
            "hocbf_intervention_rate": float(np.mean(interventions)),
            "action_changed_rate_1e-6": float(np.mean(correction > 1e-6)),
            "mean_action_correction_norm": float(np.mean(correction)),
        },
        "completion_historical": {
            "rollouts": len(option_rows),
            "structurally_safe_rate": float(np.mean(completion_safe)),
            "mean_energy_fraction": float(np.mean(completion_energy)),
        },
        "by_distance_bucket": by_bucket,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("artifacts/rcps_meet_confirmation_fork_150scenes_20260904_v1"),
    )
    parser.add_argument(
        "--freeze-dir",
        type=Path,
        default=Path("artifacts/certified_meet_fusion_freeze_150scenes_20260904_v2"),
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("docs/DUAL_VIABILITY_COUNTERFACTUAL_DIAGNOSTIC_PROTOCOL_V1.md"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-anchors", type=int, default=2)
    parser.add_argument("--smoke-step-cap", type=int, default=12)
    args = parser.parse_args(argv)
    if min(args.num_workers, args.smoke_anchors, args.smoke_step_cap) <= 0:
        parser.error("worker and smoke counts must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    for name in ("data_dir", "freeze_dir", "protocol", "output_dir"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    for path in (
        args.data_dir / "RESULT.json",
        args.data_dir / "anchors.json",
        args.data_dir / "anchors.npz",
        args.freeze_dir / "frozen_meet.pt",
        args.protocol,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.resume:
        raise FileExistsError(f"output directory is not fresh: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.resume:
        for name in ("FAILED.json", "PAUSED.json", "COMPLETED.json"):
            (args.output_dir / name).unlink(missing_ok=True)
    anchors, data_result, source_result = load_anchor_document(args.data_dir)
    task_anchors = [anchor for anchor in anchors if int(anchor["leg"]) == 0]
    if args.smoke:
        task_anchors = task_anchors[: args.smoke_anchors]
    elif len(task_anchors) != 310:
        raise RuntimeError(f"formal diagnostic requires 310 task anchors, got {len(task_anchors)}")
    jobs = jobs_from_anchors(task_anchors)
    expected = len(task_anchors) * 11
    if len(jobs) != expected:
        raise RuntimeError("expected one return-now plus ten option jobs per anchor")
    if data_result.get("checkpoint_sha256") != source_result.get("checkpoint_sha256"):
        raise RuntimeError("source checkpoint hashes disagree")
    environment_args, reconstruction = reconstruct_environment_args(
        Path(source_result["artifact"]), device=args.device, seed=int(source_result["world_seed"])
    )
    horizon = int(source_result["max_policy_steps_per_leg"])
    environment_args.phase1_episode_max_policy_steps = horizon
    environment_args.phase1_max_steps = horizon
    probe = environment_from_args(environment_args, phase=SACTrainingPhase.NAVIGATION)
    policy = SAC.load(
        Path(source_result["checkpoint"]), env=probe, device=args.device, print_system_info=False
    )
    kwargs = environment_kwargs_from_args(environment_args, phase=SACTrainingPhase.NAVIGATION)
    kwargs["max_steps_per_task"] = horizon
    kwargs["phase1_episode_max_policy_steps"] = horizon
    frozen = torch.load(args.freeze_dir / "frozen_meet.pt", map_location="cpu", weights_only=False)
    budgets = np.asarray(frozen["budgets"], dtype=np.float64)
    manifest = {
        "status": "RUNNING",
        "protocol": PROTOCOL,
        "formal_evidence": False,
        "development_only": True,
        "data_dir": args.data_dir,
        "data_result_sha256": sha256_file(args.data_dir / "RESULT.json"),
        "anchors_json_sha256": sha256_file(args.data_dir / "anchors.json"),
        "anchors_npz_sha256": sha256_file(args.data_dir / "anchors.npz"),
        "protocol_document": args.protocol,
        "protocol_sha256": sha256_file(args.protocol),
        "checkpoint": source_result["checkpoint"],
        "checkpoint_sha256": source_result["checkpoint_sha256"],
        "freeze_checkpoint_sha256": sha256_file(args.freeze_dir / "frozen_meet.pt"),
        "world_seed": source_result["world_seed"],
        "task_anchors": len(task_anchors),
        "return_now_rollouts": len(task_anchors),
        "option_rollouts": len(task_anchors) * 10,
        "total_rollouts": expected,
        "num_workers": min(args.num_workers, expected),
        "recovery_horizon": horizon,
        "candidate_execution_steps": 1,
        "candidate_hocbf_enabled": True,
        "retarget_preserves_velocity": True,
        "budgets": budgets,
        "environment_reconstruction": reconstruction,
        "smoke": args.smoke,
        "started_unix": time.time(),
        "exact_command": [sys.executable, *sys.argv],
    }
    atomic_json(args.output_dir / "RUNNING.json", json_value(manifest))
    started = time.time()
    try:
        rows, paused = collect(
            output=args.output_dir,
            jobs=jobs,
            policy=policy,
            environment_kwargs=kwargs,
            world_seed=int(source_result["world_seed"]),
            workers=args.num_workers,
            resume=args.resume,
            smoke_step_cap=args.smoke_step_cap if args.smoke else None,
        )
        probe.close()
        if paused:
            atomic_json(
                args.output_dir / "PAUSED.json",
                {
                    **manifest,
                    "status": "PAUSED",
                    "completed_rollouts": len(rows),
                    "resume_command": [sys.executable, *sys.argv, "--resume"],
                    "paused_unix": time.time(),
                },
            )
            (args.output_dir / "RUNNING.json").unlink(missing_ok=True)
            return 130
        if len(rows) != expected:
            raise RuntimeError(f"incomplete diagnostic: {len(rows)}/{expected}")
        analysis = None if args.smoke else analyze(
            rows=rows,
            data_dir=args.data_dir,
            capacity=float(source_result["capacity"]),
            budgets=budgets,
        )
        if analysis is not None:
            atomic_json(args.output_dir / "analysis.json", analysis)
        status = "SMOKE_COMPLETE_NOT_EVIDENCE" if args.smoke else "DIAGNOSTIC_COMPLETE"
        final = {
            **manifest,
            "status": status,
            "completed_rollouts": len(rows),
            "analysis": analysis,
            "finished_unix": time.time(),
            "wall_clock_seconds": time.time() - started,
        }
        atomic_json(args.output_dir / "RESULT.json", final)
        atomic_json(args.output_dir / "COMPLETED.json", {"status": status, "result": args.output_dir / "RESULT.json"})
        atomic_json(
            args.output_dir / "PROGRESS.json",
            {
                "status": "COMPLETE",
                "completed_rollouts": len(rows),
                "total_rollouts": expected,
                "active_rollouts": 0,
                "pending_rollouts": 0,
                "updated_unix": time.time(),
            },
        )
        (args.output_dir / "RUNNING.json").unlink(missing_ok=True)
        return 0
    except Exception as error:
        try:
            probe.close()
        except Exception:
            pass
        atomic_json(
            args.output_dir / "FAILED.json",
            {"type": type(error).__name__, "message": str(error), "failed_unix": time.time()},
        )
        raise


if __name__ == "__main__":
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    raise SystemExit(main())
