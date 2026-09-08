from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Apply native-thread limits before NumPy/Torch imports so simulator workers do
# not replicate BLAS/OpenMP pools. Neural-network updates remain on CUDA.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["BLIS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.UAVEnergyDeliverySAC import SACTrainingPhase, UAVEnergyDeliverySACEnv
from experiments.r7_cppo_pid import (
    CPPOPIDConfig,
    PIDLagrangian,
    PIDLagrangianConfig,
    R7ActorCritic,
    R7ModelConfig,
    RolloutBatch,
    cppo_pid_update,
)
from experiments.uav_energy_parallel import ParallelUAVEnvPool, WorkerReset
from review_bundle.envs.navigation.telemetry_cost import TelemetryCostConfig
from scripts.train_uav_energy_delivery_sac import (
    navigation_energy_gate_passed,
    navigation_safety_gate_passed,
)


PROTOCOL = "R7_R3_STRUCTURED_CPPO_PID_NAVIGATION_V1"
DEFAULT_TASKS = (
    ROOT
    / "artifacts/jseb_navigation_repair_r3r4_uv_seed0_20260828_193904"
    / "R3/eval_navigation_tasks.json"
)
DISTANCE_BUCKETS = ("100-500", "500-1500", "1500-2500", "2500-4000", ">4000")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(json_value(payload), stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def append_jsonl(path: Path, payload: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(json_value(payload), sort_keys=True) + "\n")
        stream.flush()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def git_status() -> list[str]:
    return subprocess.check_output(
        ["git", "status", "--short"], cwd=ROOT, text=True
    ).splitlines()


def environment_kwargs(args: argparse.Namespace) -> dict[str, object]:
    return {
        "phase": SACTrainingPhase.NAVIGATION,
        "minimum_task_distance": args.minimum_task_distance,
        "xy_sampling_margin": 100.0,
        "task_z_min": 20.0,
        "task_z_max": 380.0,
        "max_steps_per_task": args.episode_max_steps,
        "phase1_episode_max_policy_steps": args.episode_max_steps,
        "phase2_episode_limit": 20_000,
        "boundary_penalty": 0.0,
        "obstacle_collision_penalty": 0.0,
        "safety_intervention_penalty": 0.0,
        "telemetry_cost_config": TelemetryCostConfig(),
        "lidar_enabled": True,
        "lidar_max_range": args.lidar_range,
        "lidar_horizontal_sectors": args.lidar_horizontal_sectors,
        "lidar_vertical_sectors": args.lidar_vertical_sectors,
        "num_obstacles": args.num_obstacles,
        "obstacle_radius_min": args.obstacle_radius_min,
        "obstacle_radius_max": args.obstacle_radius_max,
        "obstacle_sampling_margin": 20.0,
        "cbf_enabled": True,
        "hocbf_k1": args.hocbf_k1,
        "hocbf_k2": args.hocbf_k2,
        "hocbf_uncertainty_margin": args.hocbf_uncertainty_margin,
        "hocbf_top_k": args.hocbf_top_k,
        "hocbf_sampled_data_robust": args.hocbf_sampled_data_robust,
        "projection_geometry_enabled": False,
    }


def _squared_correction_cost(info: dict[str, object]) -> float:
    substeps = info.get("hocbf_substep_diagnostics")
    if isinstance(substeps, list) and substeps:
        squared = []
        for row in substeps:
            if not isinstance(row, dict):
                continue
            value = float(row.get("intervention_norm", 0.0))
            if not np.isfinite(value) or value < 0.0:
                raise FloatingPointError("invalid HOCBF substep intervention norm")
            squared.append(value * value)
        if squared:
            return float(np.mean(squared))
    nominal = np.asarray(info.get("nominal_action", np.zeros(3)), dtype=np.float64)
    executed = np.asarray(info.get("executed_action", nominal), dtype=np.float64)
    if nominal.shape != (3,) or executed.shape != (3,) or not (
        np.all(np.isfinite(nominal)) and np.all(np.isfinite(executed))
    ):
        raise FloatingPointError("invalid nominal/executed action for correction cost")
    return float(np.sum((nominal - executed) ** 2))


def _step_cost(
    info: dict[str, object], collision_weight: float
) -> tuple[float, bool, bool, float]:
    collision = bool(info.get("obstacle_collision", False)) or bool(
        info.get("boundary_contact", False)
    )
    intervened = bool(info.get("hocbf_intervened", False)) or float(
        info.get("hocbf_intervention_norm", 0.0)
    ) > 1e-6
    correction_cost = _squared_correction_cost(info)
    cost = correction_cost + collision_weight * float(collision)
    return cost, intervened, collision, correction_cost


class EpisodeAccumulator:
    def __init__(self, num_envs: int) -> None:
        self.rewards = np.zeros(num_envs, dtype=np.float64)
        self.costs = np.zeros(num_envs, dtype=np.float64)
        self.steps = np.zeros(num_envs, dtype=np.int64)
        self.interventions = np.zeros(num_envs, dtype=np.int64)
        self.collisions = np.zeros(num_envs, dtype=np.int64)
        self.completed: list[dict[str, object]] = []

    def observe(
        self,
        environment: int,
        reward: float,
        cost: float,
        intervened: bool,
        collision: bool,
        info: dict[str, object],
        done: bool,
    ) -> dict[str, object] | None:
        self.rewards[environment] += reward
        self.costs[environment] += cost
        self.steps[environment] += 1
        self.interventions[environment] += int(intervened)
        self.collisions[environment] += int(collision)
        if not done:
            return None
        steps = int(self.steps[environment])
        row: dict[str, object] = {
            "reward": float(self.rewards[environment]),
            "cost": float(self.costs[environment]),
            "policy_steps": steps,
            "success": bool(info.get("is_success", False)),
            "end_reason": info.get("end_reason"),
            "intervention_step_rate": float(
                self.interventions[environment] / max(steps, 1)
            ),
            "collision_steps": int(self.collisions[environment]),
        }
        self.completed.append(row)
        self.rewards[environment] = 0.0
        self.costs[environment] = 0.0
        self.steps[environment] = 0
        self.interventions[environment] = 0
        self.collisions[environment] = 0
        return row

    def summary(self, maximum: int = 100) -> dict[str, object]:
        rows = self.completed[-maximum:]
        if not rows:
            return {
                "completed_episodes": 0,
                "window_episodes": 0,
                "successes": 0,
                "success_rate": None,
                "mean_episode_reward": None,
                "mean_episode_cost": None,
                "mean_episode_steps": None,
                "mean_intervention_step_rate": None,
                "collision_steps": 0,
            }
        return {
            "completed_episodes": len(self.completed),
            "window_episodes": len(rows),
            "successes": int(sum(bool(row["success"]) for row in rows)),
            "success_rate": float(np.mean([bool(row["success"]) for row in rows])),
            "mean_episode_reward": float(np.mean([float(row["reward"]) for row in rows])),
            "mean_episode_cost": float(np.mean([float(row["cost"]) for row in rows])),
            "mean_episode_steps": float(np.mean([int(row["policy_steps"]) for row in rows])),
            "mean_intervention_step_rate": float(
                np.mean([float(row["intervention_step_rate"]) for row in rows])
            ),
            "collision_steps": int(sum(int(row["collision_steps"]) for row in rows)),
        }


def collect_rollout(
    *,
    pool: ParallelUAVEnvPool,
    model: R7ActorCritic,
    device: torch.device,
    observations: np.ndarray,
    rollout_steps: int,
    seed_cursor: int,
    episodes: EpisodeAccumulator,
    collision_weight: float,
) -> tuple[RolloutBatch, np.ndarray, int, dict[str, float], list[dict[str, object]]]:
    num_envs = pool.num_workers
    buffers: dict[str, list[np.ndarray]] = {
        key: []
        for key in (
            "observations",
            "latents",
            "old_log_probs",
            "values",
            "rewards",
            "costs",
            "dones",
        )
    }
    worker_ids = list(range(num_envs))
    completed_rows: list[dict[str, object]] = []
    action_norm_sum = 0.0
    deterministic_action_norm_sum = 0.0
    intervention_steps = 0
    collision_steps = 0
    correction_cost_sum = 0.0
    started = time.perf_counter()
    for _ in range(rollout_steps):
        buffers["observations"].append(observations.copy())
        observation_tensor = torch.as_tensor(observations, dtype=torch.float32, device=device)
        with torch.no_grad():
            actions_t, latents_t, log_probs_t, _entropy_t, values_t = model.act(
                observation_tensor
            )
            deterministic_actions_t, *_unused = model.act(
                observation_tensor, deterministic=True
            )
        actions = actions_t.detach().cpu().numpy().astype(np.float32)
        if not np.all(np.isfinite(actions)) or np.any(np.abs(actions) > 1.000001):
            raise FloatingPointError("R7 actor produced invalid bounded actions")
        results = pool.step_many(worker_ids, actions)
        next_observations = np.stack([results[index].observation for index in worker_ids])
        rewards = np.asarray([results[index].reward for index in worker_ids], dtype=np.float32)
        dones = np.asarray(
            [results[index].terminated or results[index].truncated for index in worker_ids],
            dtype=bool,
        )
        costs = np.zeros(num_envs, dtype=np.float32)
        for environment in worker_ids:
            result = results[environment]
            cost, intervened, collision, correction_cost = _step_cost(
                result.info, collision_weight
            )
            costs[environment] = cost
            intervention_steps += int(intervened)
            collision_steps += int(collision)
            correction_cost_sum += correction_cost
            completed = episodes.observe(
                environment,
                float(rewards[environment]),
                cost,
                intervened,
                collision,
                result.info,
                bool(dones[environment]),
            )
            if completed is not None:
                completed_rows.append(completed)
        buffers["latents"].append(latents_t.detach().cpu().numpy())
        buffers["old_log_probs"].append(log_probs_t.detach().cpu().numpy())
        buffers["values"].append(values_t.detach().cpu().numpy())
        buffers["rewards"].append(rewards)
        buffers["costs"].append(costs)
        buffers["dones"].append(dones.astype(np.float32))
        action_norm_sum += float(torch.linalg.vector_norm(actions_t, dim=-1).sum().cpu())
        deterministic_action_norm_sum += float(
            torch.linalg.vector_norm(deterministic_actions_t, dim=-1).sum().cpu()
        )
        observations = next_observations
        done_workers = np.flatnonzero(dones).tolist()
        if done_workers:
            requests = []
            for worker_id in done_workers:
                seed_cursor += 1
                requests.append(WorkerReset(worker_id=worker_id, seed=seed_cursor))
            resets = pool.reset_many(requests)
            for worker_id in done_workers:
                observations[worker_id] = resets[worker_id].observation
    with torch.no_grad():
        last_values = model.values(
            torch.as_tensor(observations, dtype=torch.float32, device=device)
        ).cpu()
    tensors = {
        key: torch.as_tensor(np.stack(value), dtype=torch.float32)
        for key, value in buffers.items()
    }
    rollout = RolloutBatch(
        observations=tensors["observations"],
        latents=tensors["latents"],
        old_log_probs=tensors["old_log_probs"],
        values=tensors["values"],
        rewards=tensors["rewards"],
        costs=tensors["costs"],
        dones=tensors["dones"],
        last_values=last_values,
    )
    transitions = rollout_steps * num_envs
    elapsed = time.perf_counter() - started
    diagnostics = {
        "collection_seconds": elapsed,
        "collection_transitions_per_second": transitions / max(elapsed, 1e-9),
        "mean_sampled_action_norm": action_norm_sum / transitions,
        "mean_deterministic_action_norm": deterministic_action_norm_sum / transitions,
        "mean_action_standard_deviation": float(model.log_std.exp().mean().detach().cpu()),
        "mean_step_reward": float(rollout.rewards.mean()),
        "mean_step_cost": float(rollout.costs.mean()),
        "mean_squared_action_correction": correction_cost_sum / transitions,
        "intervention_step_rate": intervention_steps / transitions,
        "collision_step_rate": collision_steps / transitions,
    }
    return rollout, observations, seed_cursor, diagnostics, completed_rows


def save_checkpoint(
    path: Path,
    *,
    model: R7ActorCritic,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    pid: PIDLagrangian,
    model_config: R7ModelConfig,
    algorithm_config: CPPOPIDConfig,
    global_transitions: int,
    update_index: int,
    seed_cursor: int,
    rng: np.random.Generator,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(handle)
    try:
        torch.save(
            {
                "protocol": PROTOCOL,
                "model_config": model_config.as_dict(),
                "algorithm_config": algorithm_config.as_dict(),
                "model_state": model.state_dict(),
                "actor_optimizer_state": actor_optimizer.state_dict(),
                "critic_optimizer_state": critic_optimizer.state_dict(),
                "pid_state": pid.state_dict(),
                "global_transitions": global_transitions,
                "update_index": update_index,
                "seed_cursor": seed_cursor,
                "numpy_generator_state": rng.bit_generator.state,
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state_all": (
                    torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
                ),
            },
            temporary_name,
        )
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def load_evaluation_tasks(
    path: Path,
    count: int,
    allowed_buckets: tuple[str, ...],
) -> tuple[list[dict[str, object]], int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    by_bucket: dict[str, list[dict[str, object]]] = {
        bucket: [] for bucket in allowed_buckets
    }
    for source_index, task in enumerate(payload["tasks"]):
        bucket = str(task["distance_bucket"])
        if bucket not in by_bucket:
            continue
        row = dict(task)
        row["source_task_index"] = source_index
        by_bucket[bucket].append(row)
    tasks: list[dict[str, object]] = []
    for round_index in range(count):
        bucket = allowed_buckets[round_index % len(allowed_buckets)]
        bucket_index = round_index // len(allowed_buckets)
        if bucket_index >= len(by_bucket[bucket]):
            raise ValueError(
                f"only {len(by_bucket[bucket])} evaluation tasks match bucket {bucket}"
            )
        tasks.append(by_bucket[bucket][bucket_index])
    if len(tasks) != count:
        raise ValueError(f"only {len(tasks)} evaluation tasks match requested buckets")
    return tasks, int(payload["seed"])


def load_evaluation_tasks_by_source_indices(
    path: Path,
    source_indices: list[int],
    allowed_buckets: tuple[str, ...],
) -> tuple[list[dict[str, object]], int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    source_tasks = payload["tasks"]
    if len(set(source_indices)) != len(source_indices):
        raise ValueError("evaluation source task indices must be unique")
    tasks: list[dict[str, object]] = []
    for source_index in source_indices:
        if not 0 <= source_index < len(source_tasks):
            raise IndexError(f"evaluation source task index out of range: {source_index}")
        row = dict(source_tasks[source_index])
        if str(row["distance_bucket"]) not in allowed_buckets:
            raise ValueError("selected evaluation task lies outside requested buckets")
        row["source_task_index"] = source_index
        tasks.append(row)
    return tasks, int(payload["seed"])


def evaluate(
    *,
    model: R7ActorCritic,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, object]:
    allowed_buckets = tuple(args.evaluation_distance_buckets)
    source_indices = getattr(args, "evaluation_source_indices", None)
    if source_indices is None:
        tasks, base_seed = load_evaluation_tasks(
            Path(args.evaluation_tasks), args.eval_tasks, allowed_buckets
        )
    else:
        tasks, base_seed = load_evaluation_tasks_by_source_indices(
            Path(args.evaluation_tasks), list(source_indices), allowed_buckets
        )
        if len(tasks) != args.eval_tasks:
            raise ValueError("evaluation task count disagrees with selected indices")
    worker_count = min(args.evaluation_num_envs, len(tasks))
    records: list[dict[str, object]] = []
    next_task = 0
    assignments: dict[int, tuple[int, dict[str, object]]] = {}
    observations = np.zeros((worker_count, model.observation_dim), dtype=np.float32)
    rewards = np.zeros(worker_count, dtype=np.float64)
    steps = np.zeros(worker_count, dtype=np.int64)
    interventions = np.zeros(worker_count, dtype=np.int64)
    collisions = np.zeros(worker_count, dtype=np.int64)
    obstacle_collisions = np.zeros(worker_count, dtype=np.int64)
    boundary_contacts = np.zeros(worker_count, dtype=np.int64)
    started = time.perf_counter()
    with ParallelUAVEnvPool(environment_kwargs(args), num_workers=worker_count) as pool:
        requests = []
        for worker_id in range(worker_count):
            task_index = next_task
            task = tasks[task_index]
            next_task += 1
            assignments[worker_id] = (task_index, task)
            requests.append(
                WorkerReset(
                    worker_id=worker_id,
                    seed=base_seed + int(task["source_task_index"]),
                    options={
                        "start_position": task["start_position"],
                        "start_velocity": task["initial_velocity"],
                        "task_point": task["goal_position"],
                    },
                )
            )
        resets = pool.reset_many(requests)
        for worker_id in range(worker_count):
            observations[worker_id] = resets[worker_id].observation
        while assignments:
            active = sorted(assignments)
            with torch.no_grad():
                action_tensor, *_ = model.act(
                    torch.as_tensor(
                        observations[active], dtype=torch.float32, device=device
                    ),
                    deterministic=True,
                )
            results = pool.step_many(
                active, action_tensor.detach().cpu().numpy().astype(np.float32)
            )
            for local_index, worker_id in enumerate(active):
                del local_index
                result = results[worker_id]
                info = result.info
                rewards[worker_id] += result.reward
                steps[worker_id] += 1
                interventions[worker_id] += int(
                    bool(info.get("hocbf_intervened", False))
                )
                obstacle_collision = bool(info.get("obstacle_collision", False))
                boundary_contact = bool(info.get("boundary_contact", False))
                obstacle_collisions[worker_id] += int(obstacle_collision)
                boundary_contacts[worker_id] += int(boundary_contact)
                collisions[worker_id] += int(obstacle_collision or boundary_contact)
                observations[worker_id] = result.observation
                done = result.terminated or result.truncated
                if not done:
                    continue
                task_index, task = assignments.pop(worker_id)
                straight = float(task["straight_line_distance"])
                count = int(steps[worker_id])
                records.append(
                    {
                        "evaluation_index": task_index,
                        "source_task_index": int(task["source_task_index"]),
                        "distance_bucket": task["distance_bucket"],
                        "straight_line_distance": straight,
                        "path_length": float(result.current_goal_path_length),
                        "path_ratio": float(result.current_goal_path_length / straight),
                        "success": bool(info.get("is_success", False)),
                        "end_reason": info.get("end_reason"),
                        "policy_steps": count,
                        "reward": float(rewards[worker_id]),
                        "hocbf_intervention_rate": float(
                            interventions[worker_id] / max(count, 1)
                        ),
                        "obstacle_collision_steps": int(obstacle_collisions[worker_id]),
                        "boundary_contact_steps": int(boundary_contacts[worker_id]),
                        "collision_steps": int(collisions[worker_id]),
                    }
                )
                if next_task < len(tasks):
                    replacement_index = next_task
                    replacement = tasks[replacement_index]
                    next_task += 1
                    assignments[worker_id] = (replacement_index, replacement)
                    reset = pool.reset_many(
                        [
                            WorkerReset(
                                worker_id=worker_id,
                                seed=base_seed + int(replacement["source_task_index"]),
                                options={
                                    "start_position": replacement["start_position"],
                                    "start_velocity": replacement["initial_velocity"],
                                    "task_point": replacement["goal_position"],
                                },
                            )
                        ]
                    )[worker_id]
                    observations[worker_id] = reset.observation
                    rewards[worker_id] = 0.0
                    steps[worker_id] = 0
                    interventions[worker_id] = 0
                    collisions[worker_id] = 0
                    obstacle_collisions[worker_id] = 0
                    boundary_contacts[worker_id] = 0
    records.sort(key=lambda row: int(row["evaluation_index"]))
    bucket_success: dict[str, float | None] = {}
    for bucket in DISTANCE_BUCKETS:
        rows = [row for row in records if row["distance_bucket"] == bucket]
        bucket_success[bucket] = (
            None if not rows else float(np.mean([bool(row["success"]) for row in rows]))
        )
    total_steps = sum(int(row["policy_steps"]) for row in records)
    summary: dict[str, object] = {
        "protocol": PROTOCOL,
        "development_subset": (
            set(allowed_buckets) != set(DISTANCE_BUCKETS) or len(records) != 500
        ),
        "evaluation_distance_buckets": list(allowed_buckets),
        "task_source": str(Path(args.evaluation_tasks).resolve()),
        "task_source_sha256": file_sha256(Path(args.evaluation_tasks)),
        "num_tasks": len(records),
        "overall_success_rate": float(np.mean([bool(row["success"]) for row in records])),
        "distance_bucket_success": bucket_success,
        "mean_path_ratio": float(np.mean([float(row["path_ratio"]) for row in records])),
        "mean_policy_steps": float(np.mean([int(row["policy_steps"]) for row in records])),
        "hocbf_intervention_step_rate": float(
            sum(float(row["hocbf_intervention_rate"]) * int(row["policy_steps"]) for row in records)
            / max(total_steps, 1)
        ),
        "obstacle_collision_steps": int(
            sum(int(row["obstacle_collision_steps"]) for row in records)
        ),
        "boundary_contact_steps": int(
            sum(int(row["boundary_contact_steps"]) for row in records)
        ),
        "boundary_contact_step_rate": float(
            sum(int(row["boundary_contact_steps"]) for row in records)
            / max(total_steps, 1)
        ),
        "evaluation_env_transitions": total_steps,
        "wall_clock_seconds": time.perf_counter() - started,
        "records": records,
    }
    full_protocol = not bool(summary["development_subset"]) and len(records) == 500
    summary["formal_navigation_energy_gate_passed"] = (
        navigation_energy_gate_passed(summary) if full_protocol else None
    )
    summary["formal_navigation_safety_gate_passed"] = (
        navigation_safety_gate_passed(summary) if full_protocol else None
    )
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="R7 R3-structured CPPO-PID navigation")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume-checkpoint")
    parser.add_argument("--seed", type=int, default=7001)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--transition-budget", type=int, default=32_768)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--episode-max-steps", type=int, default=4000)
    parser.add_argument("--checkpoint-freq", type=int, default=16_384)
    parser.add_argument("--actor-learning-rate", type=float, default=3e-4)
    parser.add_argument("--critic-learning-rate", type=float, default=3e-4)
    parser.add_argument("--rollout-seed-base", type=int, default=710_000)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--initial-action-std", type=float, default=0.5)
    parser.add_argument("--update-epochs", type=int, default=10)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--entropy-coefficient", type=float, default=0.0)
    parser.add_argument("--target-kl", type=float, default=0.02)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--cost-gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--cost-gae-lambda", type=float, default=0.95)
    parser.add_argument("--cost-limit", type=float, default=300.0)
    parser.add_argument("--collision-cost-weight", type=float, default=25.0)
    parser.add_argument("--pid-kp", type=float, default=0.1)
    parser.add_argument("--pid-ki", type=float, default=0.01)
    parser.add_argument("--pid-kd", type=float, default=0.01)
    parser.add_argument("--lidar-range", type=float, default=100.0)
    parser.add_argument("--lidar-horizontal-sectors", type=int, default=128)
    parser.add_argument("--lidar-vertical-sectors", type=int, default=8)
    parser.add_argument("--num-obstacles", type=int, default=24)
    parser.add_argument("--obstacle-radius-min", type=float, default=50.0)
    parser.add_argument("--obstacle-radius-max", type=float, default=120.0)
    parser.add_argument("--minimum-task-distance", type=float, default=100.0)
    parser.add_argument("--hocbf-k1", type=float, default=1.0)
    parser.add_argument("--hocbf-k2", type=float, default=1.0)
    parser.add_argument("--hocbf-uncertainty-margin", type=float, default=1.0)
    parser.add_argument("--hocbf-top-k", type=int)
    parser.add_argument(
        "--hocbf-sampled-data-robust",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--evaluation-tasks", default=str(DEFAULT_TASKS))
    parser.add_argument("--eval-tasks", type=int, default=10)
    parser.add_argument("--evaluation-num-envs", type=int, default=6)
    parser.add_argument(
        "--evaluation-distance-buckets",
        nargs="+",
        choices=DISTANCE_BUCKETS,
        default=["100-500"],
    )
    parser.add_argument("--skip-final-evaluation", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    if args.smoke:
        args.transition_budget = 4096
        args.rollout_steps = 64
        args.episode_max_steps = 512
        args.checkpoint_freq = 2048
        args.update_epochs = 2
        args.minibatch_size = 128
        args.eval_tasks = 4
        args.evaluation_num_envs = min(args.evaluation_num_envs, 4)
    counts = (
        args.num_envs,
        args.transition_budget,
        args.rollout_steps,
        args.episode_max_steps,
        args.checkpoint_freq,
        args.hidden_dim,
        args.update_epochs,
        args.minibatch_size,
        args.eval_tasks,
        args.evaluation_num_envs,
    )
    if any(value <= 0 for value in counts):
        parser.error("all count and dimension arguments must be positive")
    transitions_per_rollout = args.num_envs * args.rollout_steps
    if args.transition_budget % transitions_per_rollout != 0:
        parser.error("transition-budget must be divisible by num-envs * rollout-steps")
    if args.checkpoint_freq % transitions_per_rollout != 0:
        parser.error("checkpoint-freq must be divisible by num-envs * rollout-steps")
    if args.collision_cost_weight <= 0.0 or args.cost_limit < 0.0:
        parser.error("collision cost weight must be positive and cost limit nonnegative")
    if not args.skip_final_evaluation and not Path(args.evaluation_tasks).is_file():
        parser.error("evaluation task file does not exist")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    if args.resume_checkpoint and not Path(args.resume_checkpoint).is_file():
        parser.error("resume checkpoint does not exist")
    return args


def train(args: argparse.Namespace) -> dict[str, object]:
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "checkpoints").mkdir()
    (output / "evaluation").mkdir()
    device = torch.device(args.device)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    model_config = R7ModelConfig(
        horizontal_sectors=args.lidar_horizontal_sectors,
        vertical_sectors=args.lidar_vertical_sectors,
        hidden_dim=args.hidden_dim,
        initial_action_std=args.initial_action_std,
    )
    algorithm_config = CPPOPIDConfig(
        gamma=args.gamma,
        cost_gamma=args.cost_gamma,
        gae_lambda=args.gae_lambda,
        cost_gae_lambda=args.cost_gae_lambda,
        clip_ratio=args.clip_ratio,
        entropy_coefficient=args.entropy_coefficient,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        target_kl=args.target_kl,
    )
    pid_config = PIDLagrangianConfig(
        kp=args.pid_kp,
        ki=args.pid_ki,
        kd=args.pid_kd,
        cost_limit=args.cost_limit,
    )
    model = R7ActorCritic(model_config).to(device)
    actor_optimizer = torch.optim.Adam(
        model.actor_parameters(), lr=args.actor_learning_rate, eps=1e-5
    )
    critic_optimizer = torch.optim.Adam(
        model.critic_parameters(), lr=args.critic_learning_rate, eps=1e-5
    )
    pid = PIDLagrangian(pid_config)
    resume_payload: dict[str, object] | None = None
    resume_checkpoint: Path | None = None
    if args.resume_checkpoint:
        resume_checkpoint = Path(args.resume_checkpoint).resolve()
        resume_payload = torch.load(
            resume_checkpoint, map_location=device, weights_only=False
        )
        if resume_payload.get("protocol") != PROTOCOL:
            raise ValueError("resume checkpoint protocol mismatch")
        if resume_payload.get("model_config") != model_config.as_dict():
            raise ValueError("resume checkpoint model configuration mismatch")
        if resume_payload.get("algorithm_config") != algorithm_config.as_dict():
            raise ValueError("resume checkpoint algorithm configuration mismatch")
        model.load_state_dict(resume_payload["model_state"])
        actor_optimizer.load_state_dict(resume_payload["actor_optimizer_state"])
        critic_optimizer.load_state_dict(resume_payload["critic_optimizer_state"])
        pid.load_state_dict(resume_payload["pid_state"])
    source_paths = [
        ROOT / "experiments/jacobian_energy_bridge/features.py",
        ROOT / "experiments/r7_cppo_pid/model.py",
        ROOT / "experiments/r7_cppo_pid/algorithm.py",
        ROOT / "experiments/uav_energy_parallel.py",
        ROOT / "envs/UAVEnergyDeliverySAC.py",
        ROOT / "review_bundle/safety/collision/filter.py",
        ROOT / "review_bundle/safety/collision/hocbf.py",
        ROOT / "review_bundle/safety/collision/_qp_native.c",
        ROOT / "review_bundle/safety/collision/_qp_native.so",
        Path(__file__).resolve(),
    ]
    manifest: dict[str, object] = {
        "status": "RUNNING",
        "protocol": PROTOCOL,
        "evidence_label": "NON_SCIENTIFIC_SMOKE" if args.smoke else "R7_DEVELOPMENT_GATE",
        "started_at": utc_now(),
        "command": [sys.executable, *sys.argv],
        "args": vars(args),
        "git_sha": git_sha(),
        "git_status": git_status(),
        "source_sha256": {
            str(path.relative_to(ROOT)): file_sha256(path) for path in source_paths
        },
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "model": model.architecture_audit(),
        "algorithm": algorithm_config.as_dict(),
        "pid_lagrangian": pid_config.as_dict(),
        "resume": (
            None
            if resume_checkpoint is None
            else {
                "checkpoint": str(resume_checkpoint),
                "checkpoint_sha256": file_sha256(resume_checkpoint),
                "from_global_transitions": int(resume_payload["global_transitions"]),
            }
        ),
        "swap_contract": {
            "structured_lidar_extractor_reused_from_r3": True,
            "navigation_environment_reward_unchanged": True,
            "hocbf_final_execution_layer": True,
            "algorithm": "CPPOPID",
            "reward_value_heads": 1,
            "scalar_cost_value_heads": 1,
            "route_planner": False,
            "waypoint_controller": False,
            "r6_recurrent_set_attention": False,
            "r6_safety_distillation": False,
        },
        "cost_contract": {
            "per_step_cost": "mean_substep_squared_normalized_action_correction + collision_weight * executed_contact_indicator",
            "collision_weight": args.collision_cost_weight,
            "positive_cost_limit": args.cost_limit,
            "zero_budget": False,
            "sampled_data_hocbf": args.hocbf_sampled_data_robust,
        },
    }
    atomic_json(output / "RUNNING.json", manifest)
    atomic_json(output / "config.json", manifest)
    probe = UAVEnergyDeliverySACEnv(**environment_kwargs(args))
    observation, _ = probe.reset(seed=args.rollout_seed_base)
    if observation.shape != (model.observation_dim,):
        raise RuntimeError("R7 model/environment observation dimensions disagree")
    probe.close()
    episodes = EpisodeAccumulator(args.num_envs)
    recent_episode_costs: deque[float] = deque(maxlen=100)
    rng = np.random.default_rng(args.seed)
    global_transitions = (
        0 if resume_payload is None else int(resume_payload["global_transitions"])
    )
    update_index = 0 if resume_payload is None else int(resume_payload["update_index"])
    updates_this_run = 0
    seed_cursor = (
        args.rollout_seed_base + args.num_envs - 1
        if resume_payload is None
        else int(resume_payload["seed_cursor"])
    )
    if global_transitions >= args.transition_budget:
        raise ValueError("resume checkpoint already meets or exceeds transition budget")
    if global_transitions % (args.num_envs * args.rollout_steps) != 0:
        raise ValueError("resume checkpoint is not aligned with the rollout schedule")
    if resume_payload is not None:
        if "numpy_generator_state" in resume_payload:
            rng.bit_generator.state = resume_payload["numpy_generator_state"]
        if "torch_rng_state" in resume_payload:
            torch.set_rng_state(resume_payload["torch_rng_state"].cpu())
        if device.type == "cuda" and resume_payload.get("cuda_rng_state_all") is not None:
            torch.cuda.set_rng_state_all(
                [state.cpu() for state in resume_payload["cuda_rng_state_all"]]
            )
    next_checkpoint = (
        global_transitions // args.checkpoint_freq + 1
    ) * args.checkpoint_freq
    training_started = time.perf_counter()
    final_checkpoint = output / "checkpoints" / f"checkpoint_transition_{args.transition_budget:09d}.pt"
    try:
        with ParallelUAVEnvPool(environment_kwargs(args), num_workers=args.num_envs) as pool:
            resets = pool.reset_many(
                [
                    WorkerReset(worker_id=index, seed=args.rollout_seed_base + index)
                    for index in range(args.num_envs)
                ]
            )
            observations = np.stack(
                [resets[index].observation for index in range(args.num_envs)]
            )
            while global_transitions < args.transition_budget:
                rollout, observations, seed_cursor, collection, completed = collect_rollout(
                    pool=pool,
                    model=model,
                    device=device,
                    observations=observations,
                    rollout_steps=args.rollout_steps,
                    seed_cursor=seed_cursor,
                    episodes=episodes,
                    collision_weight=args.collision_cost_weight,
                )
                episode_cost_mean = None
                if completed:
                    for completed_row in completed:
                        recent_episode_costs.append(float(completed_row["cost"]))
                    episode_cost_mean = float(np.mean(recent_episode_costs))
                update_started = time.perf_counter()
                update_metrics = cppo_pid_update(
                    model,
                    actor_optimizer,
                    critic_optimizer,
                    rollout,
                    pid,
                    algorithm_config,
                    device=device,
                    rng=rng,
                    episode_cost_mean=episode_cost_mean,
                )
                update_seconds = time.perf_counter() - update_started
                update_index += 1
                updates_this_run += 1
                global_transitions += args.rollout_steps * args.num_envs
                row: dict[str, object] = {
                    "update": update_index,
                    "run_update": updates_this_run,
                    "global_env_transitions": global_transitions,
                    "wall_clock_seconds": time.perf_counter() - training_started,
                    "update_seconds": update_seconds,
                    "actor_device": str(next(model.actor_parameters()).device),
                    **collection,
                    **update_metrics,
                    **{
                        f"recent_{key}": value
                        for key, value in episodes.summary().items()
                    },
                }
                append_jsonl(output / "training_metrics.jsonl", row)
                if updates_this_run == 1:
                    checks = {
                        "finite_actor_loss": np.isfinite(row["actor_loss"]),
                        "finite_critic_loss": np.isfinite(row["critic_loss"]),
                        "positive_collection_throughput": row[
                            "collection_transitions_per_second"
                        ]
                        > 0.0,
                        "bounded_nonzero_actions": 0.0
                        < row["mean_sampled_action_norm"]
                        <= np.sqrt(3.0) + 1e-6,
                        "separate_actor_critic_features": True,
                        "gpu_actor": next(model.actor_parameters()).device.type == "cuda",
                    }
                    health = {
                        "status": "HEALTHY" if all(checks.values()) else "UNHEALTHY",
                        "checked_at": utc_now(),
                        "checks": checks,
                        "first_update": row,
                    }
                    atomic_json(output / "HEALTH.json", health)
                    if health["status"] != "HEALTHY":
                        raise RuntimeError("R7 first-update health Gate failed")
                if global_transitions >= next_checkpoint or global_transitions == args.transition_budget:
                    checkpoint = output / "checkpoints" / f"checkpoint_transition_{global_transitions:09d}.pt"
                    save_checkpoint(
                        checkpoint,
                        model=model,
                        actor_optimizer=actor_optimizer,
                        critic_optimizer=critic_optimizer,
                        pid=pid,
                        model_config=model_config,
                        algorithm_config=algorithm_config,
                        global_transitions=global_transitions,
                        update_index=update_index,
                        seed_cursor=seed_cursor,
                        rng=rng,
                    )
                    next_checkpoint += args.checkpoint_freq
        evaluation = None
        if not args.skip_final_evaluation:
            evaluation = evaluate(model=model, args=args, device=device)
            atomic_json(output / "evaluation" / "final_navigation.json", evaluation)
        training_summary = episodes.summary(maximum=max(len(episodes.completed), 1))
        deterministic_development_success = (
            None if evaluation is None else float(evaluation["overall_success_rate"])
        )
        development_gate = {
            "formal_evidence": False,
            "training_has_real_goal_success": int(training_summary["successes"]) > 0,
            "deterministic_evaluation_has_real_goal_success": (
                None
                if deterministic_development_success is None
                else deterministic_development_success > 0.0
            ),
            "deterministic_evaluation_collision_free": (
                None
                if evaluation is None
                else int(evaluation["obstacle_collision_steps"]) == 0
                and int(evaluation["boundary_contact_steps"]) == 0
            ),
        }
        comparable_checks = [
            bool(development_gate["training_has_real_goal_success"])
        ]
        if evaluation is not None:
            comparable_checks.extend(
                [
                    bool(development_gate["deterministic_evaluation_has_real_goal_success"]),
                    bool(development_gate["deterministic_evaluation_collision_free"]),
                ]
            )
        development_gate["passed"] = all(comparable_checks)
        atomic_json(output / "DEVELOPMENT_GATE.json", development_gate)
        completed = {
            **manifest,
            "status": "COMPLETED",
            "completed_at": utc_now(),
            "actual_training_transitions": global_transitions,
            "exact_budget_match": global_transitions == args.transition_budget,
            "ppo_updates": update_index,
            "ppo_updates_this_run": updates_this_run,
            "training_wall_clock_seconds": time.perf_counter() - training_started,
            "final_checkpoint": str(final_checkpoint),
            "final_checkpoint_sha256": file_sha256(final_checkpoint),
            "final_pid_state": pid.state_dict(),
            "training_episode_summary": training_summary,
            "development_gate": development_gate,
            "final_navigation": (
                None
                if evaluation is None
                else {key: value for key, value in evaluation.items() if key != "records"}
            ),
        }
        atomic_json(output / "COMPLETED.json", completed)
        (output / "RUNNING.json").unlink(missing_ok=True)
        return completed
    except KeyboardInterrupt:
        interrupted_checkpoint = output / "checkpoints" / f"checkpoint_interrupted_{global_transitions:09d}.pt"
        save_checkpoint(
            interrupted_checkpoint,
            model=model,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            pid=pid,
            model_config=model_config,
            algorithm_config=algorithm_config,
            global_transitions=global_transitions,
            update_index=update_index,
            seed_cursor=seed_cursor,
            rng=rng,
        )
        atomic_json(
            output / "INTERRUPTED.json",
            {
                "status": "INTERRUPTED",
                "interrupted_at": utc_now(),
                "global_env_transitions": global_transitions,
                "checkpoint": str(interrupted_checkpoint),
            },
        )
        (output / "RUNNING.json").unlink(missing_ok=True)
        raise
    except BaseException as error:
        atomic_json(
            output / "FAILED.json",
            {
                "status": "FAILED",
                "failed_at": utc_now(),
                "global_env_transitions": global_transitions,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        (output / "RUNNING.json").unlink(missing_ok=True)
        raise


def main() -> int:
    # Background shells commonly start jobs with SIGINT ignored. Restore an
    # explicit Python handler so operational pauses still reach the checkpoint
    # path in train() instead of requiring a destructive termination.
    signal.signal(signal.SIGINT, signal.default_int_handler)
    args = parse_args()
    result = train(args)
    print(
        json.dumps(
            {
                key: value
                for key, value in result.items()
                if key not in {"git_status", "source_sha256"}
            },
            indent=2,
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
