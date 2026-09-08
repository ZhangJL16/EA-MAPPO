from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# These variables are set before NumPy/Torch are imported so spawned simulator
# workers cannot each create a full BLAS/OpenMP thread pool. CUDA remains the
# owner of neural-network parallelism in the parent process.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.UAVEnergyDeliverySAC import SACTrainingPhase, UAVEnergyDeliverySACEnv
from experiments.r6_learned_navigation import (
    ConstraintState,
    PPOConfig,
    R6ModelConfig,
    RecurrentSetActorCritic,
    RolloutBatch,
    ppo_update,
)
from experiments.uav_energy_parallel import ParallelUAVEnvPool, WorkerReset
from review_bundle.envs.navigation.telemetry_cost import TelemetryCostConfig


PROTOCOL = "R6_RECURRENT_CONSTRAINED_PPO_V1"
DEFAULT_R3_TASKS = (
    ROOT
    / "artifacts/jseb_navigation_repair_r3r4_uv_seed0_20260828_193904"
    / "R3/eval_navigation_tasks.json"
)


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
    path.parent.mkdir(parents=True, exist_ok=True)
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
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def git_status() -> list[str]:
    result = subprocess.check_output(
        ["git", "status", "--short"], cwd=ROOT, text=True
    )
    return [line for line in result.splitlines() if line]


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
        "hocbf_sampled_data_robust": False,
        "projection_geometry_enabled": False,
    }


def save_checkpoint(
    path: Path,
    *,
    model: RecurrentSetActorCritic,
    optimizer: torch.optim.Optimizer,
    constraints: ConstraintState,
    model_config: R6ModelConfig,
    ppo_config: PPOConfig,
    global_transitions: int,
    update_index: int,
    episode_seed_cursor: int,
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
                "ppo_config": ppo_config.as_dict(),
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "constraints": constraints.as_dict(),
                "global_transitions": int(global_transitions),
                "update_index": int(update_index),
                "episode_seed_cursor": int(episode_seed_cursor),
            },
            temporary_name,
        )
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _tensor(value: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(value, dtype=torch.float32, device=device)


def _feedback_from_info(info: dict[str, object]) -> np.ndarray:
    executed = np.asarray(info.get("executed_action"), dtype=np.float32)
    if executed.shape != (3,) or not np.all(np.isfinite(executed)):
        raise RuntimeError("worker did not expose a finite executed action")
    intervention = float(info.get("hocbf_intervention_norm", np.nan))
    progress = float(info.get("progress", np.nan))
    if not np.isfinite(intervention) or intervention < 0.0 or not np.isfinite(progress):
        raise RuntimeError("worker did not expose finite intervention/progress feedback")
    normalized_progress = float(np.clip(progress / 4.0, -1.0, 1.0))
    return np.concatenate(
        (executed, np.asarray([intervention, normalized_progress], dtype=np.float32))
    ).astype(np.float32)


class EpisodeAccumulator:
    def __init__(self, num_envs: int) -> None:
        self.rewards = np.zeros(num_envs, dtype=np.float64)
        self.steps = np.zeros(num_envs, dtype=np.int64)
        self.interventions = np.zeros(num_envs, dtype=np.int64)
        self.intervention_norms = np.zeros(num_envs, dtype=np.float64)
        self.collisions = np.zeros(num_envs, dtype=np.int64)
        self.completed: list[dict[str, object]] = []

    def observe(
        self,
        environment: int,
        reward: float,
        info: dict[str, object],
        done: bool,
    ) -> None:
        self.rewards[environment] += reward
        self.steps[environment] += 1
        self.interventions[environment] += int(bool(info.get("hocbf_intervened", False)))
        self.intervention_norms[environment] += float(
            info.get("hocbf_intervention_norm", 0.0)
        )
        self.collisions[environment] += int(
            bool(info.get("obstacle_collision", False))
            or bool(info.get("boundary_contact", False))
        )
        if not done:
            return
        steps = int(self.steps[environment])
        self.completed.append(
            {
                "reward": float(self.rewards[environment]),
                "policy_steps": steps,
                "success": bool(info.get("is_success", False)),
                "end_reason": info.get("end_reason"),
                "intervention_step_rate": float(
                    self.interventions[environment] / max(steps, 1)
                ),
                "mean_intervention_norm": float(
                    self.intervention_norms[environment] / max(steps, 1)
                ),
                "collision_steps": int(self.collisions[environment]),
            }
        )
        self.rewards[environment] = 0.0
        self.steps[environment] = 0
        self.interventions[environment] = 0
        self.intervention_norms[environment] = 0.0
        self.collisions[environment] = 0

    def recent_summary(self, maximum: int = 100) -> dict[str, object]:
        rows = self.completed[-maximum:]
        if not rows:
            return {
                "completed_episodes": 0,
                "success_rate": None,
                "mean_episode_reward": None,
                "mean_episode_steps": None,
                "mean_episode_intervention_rate": None,
                "mean_episode_intervention_norm": None,
                "collision_step_rate": None,
            }
        total_steps = sum(int(row["policy_steps"]) for row in rows)
        return {
            "completed_episodes": len(self.completed),
            "window_episodes": len(rows),
            "success_rate": float(np.mean([bool(row["success"]) for row in rows])),
            "mean_episode_reward": float(np.mean([float(row["reward"]) for row in rows])),
            "mean_episode_steps": float(
                np.mean([int(row["policy_steps"]) for row in rows])
            ),
            "mean_episode_intervention_rate": float(
                np.mean([float(row["intervention_step_rate"]) for row in rows])
            ),
            "mean_episode_intervention_norm": float(
                np.mean([float(row["mean_intervention_norm"]) for row in rows])
            ),
            "collision_step_rate": float(
                sum(int(row["collision_steps"]) for row in rows) / max(total_steps, 1)
            ),
        }


def collect_rollout(
    *,
    pool: ParallelUAVEnvPool,
    model: RecurrentSetActorCritic,
    device: torch.device,
    observations: np.ndarray,
    feedback: np.ndarray,
    hidden: torch.Tensor,
    episode_starts: np.ndarray,
    rollout_steps: int,
    seed_cursor: int,
    episodes: EpisodeAccumulator,
) -> tuple[RolloutBatch, np.ndarray, np.ndarray, torch.Tensor, np.ndarray, int, dict[str, float]]:
    num_envs = pool.num_workers
    buffers: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "observations",
            "feedback",
            "hiddens",
            "episode_starts",
            "actions",
            "executed_actions",
            "old_log_probs",
            "values",
            "rewards",
            "collision_costs",
            "intervention_costs",
            "dones",
        )
    }
    action_minimum = np.inf
    action_maximum = -np.inf
    started = time.perf_counter()
    worker_ids = list(range(num_envs))
    for _ in range(rollout_steps):
        buffers["observations"].append(observations.copy())
        buffers["feedback"].append(feedback.copy())
        buffers["hiddens"].append(hidden.detach().cpu().numpy().copy())
        buffers["episode_starts"].append(episode_starts.astype(np.float32).copy())
        with torch.no_grad():
            actions_t, log_probs_t, _entropy_t, values_t, next_hidden = model.step(
                _tensor(observations, device),
                _tensor(feedback, device),
                hidden,
                _tensor(episode_starts.astype(np.float32), device),
            )
        actions = actions_t.detach().cpu().numpy().astype(np.float32)
        if not np.all(np.isfinite(actions)) or np.any(actions < -1.000001) or np.any(actions > 1.000001):
            raise FloatingPointError("R6 actor produced invalid actions")
        action_minimum = min(action_minimum, float(actions.min()))
        action_maximum = max(action_maximum, float(actions.max()))
        results = pool.step_many(worker_ids, actions)
        next_observations = np.stack(
            [results[index].observation for index in worker_ids]
        )
        rewards = np.asarray([results[index].reward for index in worker_ids], dtype=np.float32)
        dones = np.asarray(
            [
                results[index].terminated or results[index].truncated
                for index in worker_ids
            ],
            dtype=bool,
        )
        infos = [results[index].info for index in worker_ids]
        executed_actions = np.stack(
            [np.asarray(info["executed_action"], dtype=np.float32) for info in infos]
        )
        collision_costs = np.asarray(
            [
                float(
                    bool(info.get("obstacle_collision", False))
                    or bool(info.get("boundary_contact", False))
                )
                for info in infos
            ],
            dtype=np.float32,
        )
        intervention_costs = np.asarray(
            [float(info["hocbf_intervention_norm"]) for info in infos],
            dtype=np.float32,
        )
        next_feedback = np.stack([_feedback_from_info(info) for info in infos])
        for environment, info in enumerate(infos):
            episodes.observe(environment, float(rewards[environment]), info, bool(dones[environment]))
        buffers["actions"].append(actions)
        buffers["executed_actions"].append(executed_actions)
        buffers["old_log_probs"].append(log_probs_t.detach().cpu().numpy())
        buffers["values"].append(values_t.detach().cpu().numpy())
        buffers["rewards"].append(rewards)
        buffers["collision_costs"].append(collision_costs)
        buffers["intervention_costs"].append(intervention_costs)
        buffers["dones"].append(dones.astype(np.float32))
        hidden = next_hidden.detach()
        observations = next_observations
        feedback = next_feedback
        episode_starts = dones.copy()
        done_workers = np.flatnonzero(dones).tolist()
        if done_workers:
            reset_requests = []
            for worker_id in done_workers:
                seed_cursor += 1
                reset_requests.append(
                    WorkerReset(worker_id=worker_id, seed=seed_cursor)
                )
            reset_results = pool.reset_many(reset_requests)
            for worker_id in done_workers:
                observations[worker_id] = reset_results[worker_id].observation
                feedback[worker_id] = 0.0
                hidden[worker_id] = 0.0
    with torch.no_grad():
        _action, _log_prob, _entropy, last_values, _next_hidden = model.step(
            _tensor(observations, device),
            _tensor(feedback, device),
            hidden,
            _tensor(episode_starts.astype(np.float32), device),
            deterministic=True,
        )
    tensors = {
        key: torch.as_tensor(np.stack(value), dtype=torch.float32)
        for key, value in buffers.items()
    }
    rollout = RolloutBatch(
        observations=tensors["observations"],
        feedback=tensors["feedback"],
        hiddens=tensors["hiddens"],
        episode_starts=tensors["episode_starts"],
        actions=tensors["actions"],
        executed_actions=tensors["executed_actions"],
        old_log_probs=tensors["old_log_probs"],
        values=tensors["values"],
        rewards=tensors["rewards"],
        collision_costs=tensors["collision_costs"],
        intervention_costs=tensors["intervention_costs"],
        dones=tensors["dones"],
        last_values=last_values.detach().cpu(),
    )
    elapsed = time.perf_counter() - started
    diagnostics = {
        "collection_seconds": elapsed,
        "collection_transitions_per_second": float(
            rollout_steps * num_envs / max(elapsed, 1e-9)
        ),
        "sampled_action_minimum": float(action_minimum),
        "sampled_action_maximum": float(action_maximum),
        "mean_step_reward": float(tensors["rewards"].mean()),
        "mean_progress_feedback": float(tensors["feedback"][..., 4].mean()),
        "intervention_step_rate": float(
            (tensors["intervention_costs"] > 1e-6).float().mean()
        ),
        "emitted_collision_step_rate": float(tensors["collision_costs"].mean()),
    }
    return (
        rollout,
        observations,
        feedback,
        hidden,
        episode_starts,
        seed_cursor,
        diagnostics,
    )


def _load_evaluation_tasks(path: Path, count: int) -> tuple[list[dict[str, object]], int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    tasks = list(payload["tasks"])
    buckets: dict[str, list[dict[str, object]]] = {}
    for index, task in enumerate(tasks):
        row = dict(task)
        row["source_task_index"] = index
        buckets.setdefault(str(row["distance_bucket"]), []).append(row)
    bucket_names = list(buckets)
    selected: list[dict[str, object]] = []
    cursor = {name: 0 for name in bucket_names}
    while len(selected) < count:
        made_progress = False
        for name in bucket_names:
            index = cursor[name]
            if index < len(buckets[name]) and len(selected) < count:
                selected.append(buckets[name][index])
                cursor[name] += 1
                made_progress = True
        if not made_progress:
            break
    if len(selected) != count:
        raise ValueError(f"evaluation task file contains only {len(selected)} selectable tasks")
    return selected, int(payload["seed"])


def evaluate(
    *,
    model: RecurrentSetActorCritic,
    model_config: R6ModelConfig,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, object]:
    tasks, base_seed = _load_evaluation_tasks(Path(args.evaluation_tasks), args.eval_tasks)
    worker_count = min(args.evaluation_num_envs, len(tasks))
    records: list[dict[str, object]] = []
    next_task = 0
    assignments: dict[int, tuple[int, dict[str, object]]] = {}
    observations = np.zeros((worker_count, model_config.observation_dim), dtype=np.float32)
    feedback = np.zeros((worker_count, model_config.feedback_dim), dtype=np.float32)
    episode_starts = np.ones(worker_count, dtype=bool)
    hidden = model.initial_hidden(worker_count, device=device)
    rewards = np.zeros(worker_count, dtype=np.float64)
    steps = np.zeros(worker_count, dtype=np.int64)
    intervention_steps = np.zeros(worker_count, dtype=np.int64)
    intervention_norms = np.zeros(worker_count, dtype=np.float64)
    collisions = np.zeros(worker_count, dtype=np.int64)
    started = time.perf_counter()
    with ParallelUAVEnvPool(
        environment_kwargs(args), num_workers=worker_count
    ) as pool:
        initial_requests: list[WorkerReset] = []
        for worker_id in range(worker_count):
            task_index = next_task
            task = tasks[task_index]
            next_task += 1
            assignments[worker_id] = (task_index, task)
            initial_requests.append(
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
        reset_results = pool.reset_many(initial_requests)
        for worker_id in range(worker_count):
            observations[worker_id] = reset_results[worker_id].observation
        while assignments:
            active = sorted(assignments)
            index = torch.as_tensor(active, dtype=torch.long, device=device)
            with torch.no_grad():
                actions_t, _logp, _entropy, _values, next_hidden = model.step(
                    _tensor(observations[active], device),
                    _tensor(feedback[active], device),
                    hidden.index_select(0, index),
                    _tensor(episode_starts[active].astype(np.float32), device),
                    deterministic=True,
                )
            actions = actions_t.detach().cpu().numpy().astype(np.float32)
            results = pool.step_many(active, actions)
            for local_index, worker_id in enumerate(active):
                result = results[worker_id]
                info = result.info
                rewards[worker_id] += result.reward
                steps[worker_id] += 1
                intervention_steps[worker_id] += int(bool(info.get("hocbf_intervened", False)))
                intervention_norms[worker_id] += float(info.get("hocbf_intervention_norm", 0.0))
                collisions[worker_id] += int(
                    bool(info.get("obstacle_collision", False))
                    or bool(info.get("boundary_contact", False))
                )
                observations[worker_id] = result.observation
                feedback[worker_id] = _feedback_from_info(info)
                episode_starts[worker_id] = False
                hidden[worker_id] = next_hidden[local_index]
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
                        "intervention_step_rate": float(
                            intervention_steps[worker_id] / max(count, 1)
                        ),
                        "mean_intervention_norm": float(
                            intervention_norms[worker_id] / max(count, 1)
                        ),
                        "collision_steps": int(collisions[worker_id]),
                    }
                )
                if next_task >= len(tasks):
                    continue
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
                feedback[worker_id] = 0.0
                hidden[worker_id] = 0.0
                episode_starts[worker_id] = True
                rewards[worker_id] = 0.0
                steps[worker_id] = 0
                intervention_steps[worker_id] = 0
                intervention_norms[worker_id] = 0.0
                collisions[worker_id] = 0
    records.sort(key=lambda row: int(row["evaluation_index"]))
    bucket_success: dict[str, float] = {}
    for bucket in sorted({str(row["distance_bucket"]) for row in records}):
        bucket_rows = [row for row in records if row["distance_bucket"] == bucket]
        bucket_success[bucket] = float(
            np.mean([bool(row["success"]) for row in bucket_rows])
        )
    total_steps = sum(int(row["policy_steps"]) for row in records)
    return {
        "protocol": PROTOCOL,
        "task_source": str(Path(args.evaluation_tasks).resolve()),
        "task_source_sha256": file_sha256(Path(args.evaluation_tasks)),
        "num_tasks": len(records),
        "overall_success_rate": float(np.mean([bool(row["success"]) for row in records])),
        "distance_bucket_success": bucket_success,
        "mean_path_ratio": float(np.mean([float(row["path_ratio"]) for row in records])),
        "mean_policy_steps": float(np.mean([int(row["policy_steps"]) for row in records])),
        "hocbf_intervention_step_rate": float(
            sum(float(row["intervention_step_rate"]) * int(row["policy_steps"]) for row in records)
            / max(total_steps, 1)
        ),
        "mean_intervention_norm": float(
            sum(float(row["mean_intervention_norm"]) * int(row["policy_steps"]) for row in records)
            / max(total_steps, 1)
        ),
        "collision_steps": int(sum(int(row["collision_steps"]) for row in records)),
        "evaluation_env_transitions": total_steps,
        "wall_clock_seconds": time.perf_counter() - started,
        "records": records,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="R6 recurrent constrained-PPO learned navigation"
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=6001)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--transition-budget", type=int, default=262_144)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--episode-max-steps", type=int, default=4000)
    parser.add_argument("--checkpoint-freq", type=int, default=32_768)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--rollout-seed-base", type=int, default=610_000)
    parser.add_argument("--horizontal-hidden", type=int, default=128)
    parser.add_argument("--recurrent-hidden", type=int, default=128)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--minibatch-sequences", type=int, default=4)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--entropy-coefficient", type=float, default=0.002)
    parser.add_argument("--value-coefficient", type=float, default=0.5)
    parser.add_argument("--distillation-coefficient", type=float, default=0.2)
    parser.add_argument("--target-kl", type=float, default=0.03)
    parser.add_argument("--dual-learning-rate", type=float, default=0.05)
    parser.add_argument("--maximum-multiplier", type=float, default=20.0)
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
    parser.add_argument("--evaluation-tasks", default=str(DEFAULT_R3_TASKS))
    parser.add_argument("--eval-tasks", type=int, default=100)
    parser.add_argument("--evaluation-num-envs", type=int, default=6)
    parser.add_argument("--skip-final-evaluation", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    if args.smoke:
        args.transition_budget = 4096
        args.rollout_steps = 64
        args.checkpoint_freq = 2048
        args.sequence_length = 32
        args.minibatch_sequences = 2
        args.update_epochs = 2
        args.eval_tasks = 10
        args.evaluation_num_envs = min(args.evaluation_num_envs, 2)
    positive_names = (
        "num_envs",
        "transition_budget",
        "rollout_steps",
        "episode_max_steps",
        "checkpoint_freq",
        "horizontal_hidden",
        "recurrent_hidden",
        "attention_heads",
        "update_epochs",
        "sequence_length",
        "minibatch_sequences",
        "lidar_horizontal_sectors",
        "lidar_vertical_sectors",
        "num_obstacles",
        "eval_tasks",
        "evaluation_num_envs",
    )
    if any(getattr(args, name) <= 0 for name in positive_names):
        parser.error("all count and dimension arguments must be positive")
    transitions_per_rollout = args.num_envs * args.rollout_steps
    if args.transition_budget % transitions_per_rollout != 0:
        parser.error("transition-budget must be divisible by num-envs * rollout-steps")
    if args.rollout_steps % args.sequence_length != 0:
        parser.error("rollout-steps must be divisible by sequence-length")
    if args.checkpoint_freq % transitions_per_rollout != 0:
        parser.error("checkpoint-freq must be divisible by num-envs * rollout-steps")
    if not Path(args.evaluation_tasks).is_file() and not args.skip_final_evaluation:
        parser.error("evaluation task file does not exist")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA was requested but torch.cuda.is_available() is false")
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
    model_config = R6ModelConfig(
        horizontal_sectors=args.lidar_horizontal_sectors,
        vertical_sectors=args.lidar_vertical_sectors,
        core_dim=args.horizontal_hidden,
        ray_dim=args.horizontal_hidden,
        attention_heads=args.attention_heads,
        recurrent_dim=args.recurrent_hidden,
    )
    ppo_config = PPOConfig(
        clip_ratio=args.clip_ratio,
        entropy_coefficient=args.entropy_coefficient,
        value_coefficient=args.value_coefficient,
        distillation_coefficient=args.distillation_coefficient,
        update_epochs=args.update_epochs,
        sequence_length=args.sequence_length,
        minibatch_sequences=args.minibatch_sequences,
        target_kl=args.target_kl,
    )
    constraints = ConstraintState(
        collision_budget=0.0,
        intervention_budget=0.0,
        dual_learning_rate=args.dual_learning_rate,
        maximum_multiplier=args.maximum_multiplier,
    )
    model = RecurrentSetActorCritic(model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, eps=1e-5)
    source_paths = [
        ROOT / "experiments/r6_learned_navigation/model.py",
        ROOT / "experiments/r6_learned_navigation/ppo.py",
        ROOT / "experiments/uav_energy_parallel.py",
        ROOT / "envs/UAVEnergyDeliverySAC.py",
        Path(__file__).resolve(),
    ]
    manifest: dict[str, object] = {
        "status": "RUNNING",
        "protocol": PROTOCOL,
        "evidence_label": (
            "NON_SCIENTIFIC_SMOKE" if args.smoke else "R6_DEVELOPMENT_CANDIDATE"
        ),
        "started_at": utc_now(),
        "command": [sys.executable, *sys.argv],
        "args": vars(args),
        "git_sha": git_sha(),
        "git_status": git_status(),
        "source_sha256": {str(path.relative_to(ROOT)): file_sha256(path) for path in source_paths},
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "model": model.architecture_audit(),
        "ppo": ppo_config.as_dict(),
        "constraints": constraints.as_dict(),
        "training_contract": {
            "nominal_action_likelihood": "sampled_actor_action",
            "transition_action": "environment_reported_executed_action",
            "safety_teacher": "continuous_intervention_weighted_distillation",
            "explicit_route_planner": False,
            "waypoint_controller": False,
            "heuristic_navigation_action": False,
            "gpu_actor_and_optimizer": device.type == "cuda",
            "worker_threads_per_environment": 1,
        },
    }
    atomic_json(output / "RUNNING.json", manifest)
    atomic_json(output / "config.json", manifest)
    environment = UAVEnergyDeliverySACEnv(**environment_kwargs(args))
    observation, _info = environment.reset(seed=args.rollout_seed_base)
    if observation.shape != (model_config.observation_dim,):
        raise RuntimeError("R6 model and environment observation dimensions disagree")
    environment.close()
    episodes = EpisodeAccumulator(args.num_envs)
    rng = np.random.default_rng(args.seed)
    global_transitions = 0
    update_index = 0
    seed_cursor = args.rollout_seed_base + args.num_envs - 1
    next_checkpoint = args.checkpoint_freq
    training_started = time.perf_counter()
    final_checkpoint = output / "checkpoints" / f"checkpoint_transition_{args.transition_budget:09d}.pt"
    try:
        with ParallelUAVEnvPool(
            environment_kwargs(args), num_workers=args.num_envs
        ) as pool:
            resets = pool.reset_many(
                [
                    WorkerReset(
                        worker_id=index,
                        seed=args.rollout_seed_base + index,
                    )
                    for index in range(args.num_envs)
                ]
            )
            observations = np.stack(
                [resets[index].observation for index in range(args.num_envs)]
            )
            feedback = np.zeros(
                (args.num_envs, model_config.feedback_dim), dtype=np.float32
            )
            hidden = model.initial_hidden(args.num_envs, device=device)
            episode_starts = np.ones(args.num_envs, dtype=bool)
            while global_transitions < args.transition_budget:
                (
                    rollout,
                    observations,
                    feedback,
                    hidden,
                    episode_starts,
                    seed_cursor,
                    collection_metrics,
                ) = collect_rollout(
                    pool=pool,
                    model=model,
                    device=device,
                    observations=observations,
                    feedback=feedback,
                    hidden=hidden,
                    episode_starts=episode_starts,
                    rollout_steps=args.rollout_steps,
                    seed_cursor=seed_cursor,
                    episodes=episodes,
                )
                update_started = time.perf_counter()
                update_metrics = ppo_update(
                    model,
                    optimizer,
                    rollout,
                    constraints,
                    ppo_config,
                    device=device,
                    rng=rng,
                )
                update_seconds = time.perf_counter() - update_started
                update_index += 1
                global_transitions += args.rollout_steps * args.num_envs
                row: dict[str, object] = {
                    "update": update_index,
                    "global_env_transitions": global_transitions,
                    "wall_clock_seconds": time.perf_counter() - training_started,
                    "update_seconds": update_seconds,
                    "optimizer_device": str(next(model.parameters()).device),
                    **collection_metrics,
                    **update_metrics,
                    **{f"recent_{key}": value for key, value in episodes.recent_summary().items()},
                }
                append_jsonl(output / "training_metrics.jsonl", row)
                if update_index == 1:
                    health = {
                        "status": "HEALTHY",
                        "checked_at": utc_now(),
                        "first_update": row,
                        "checks": {
                            "finite_losses": all(
                                np.isfinite(float(update_metrics[name]))
                                for name in (
                                    "loss",
                                    "policy_loss",
                                    "value_loss",
                                    "gradient_norm",
                                )
                            ),
                            "action_bounds_respected": bool(
                                collection_metrics["sampled_action_minimum"] >= -1.000001
                                and collection_metrics["sampled_action_maximum"] <= 1.000001
                            ),
                            "executed_action_feedback_present": True,
                            "positive_collection_throughput": bool(
                                collection_metrics["collection_transitions_per_second"] > 0.0
                            ),
                            "gpu_optimizer": next(model.parameters()).device.type == "cuda",
                        },
                    }
                    if not all(health["checks"].values()):
                        health["status"] = "UNHEALTHY"
                    atomic_json(output / "HEALTH.json", health)
                    if health["status"] != "HEALTHY":
                        raise RuntimeError("R6 first-update health Gate failed")
                if global_transitions >= next_checkpoint or global_transitions == args.transition_budget:
                    checkpoint = output / "checkpoints" / f"checkpoint_transition_{global_transitions:09d}.pt"
                    save_checkpoint(
                        checkpoint,
                        model=model,
                        optimizer=optimizer,
                        constraints=constraints,
                        model_config=model_config,
                        ppo_config=ppo_config,
                        global_transitions=global_transitions,
                        update_index=update_index,
                        episode_seed_cursor=seed_cursor,
                    )
                    next_checkpoint += args.checkpoint_freq
        evaluation = None
        if not args.skip_final_evaluation:
            evaluation = evaluate(
                model=model,
                model_config=model_config,
                args=args,
                device=device,
            )
            atomic_json(output / "evaluation" / "final_navigation.json", evaluation)
        completed = {
            **manifest,
            "status": "COMPLETED",
            "completed_at": utc_now(),
            "actual_training_transitions": global_transitions,
            "exact_budget_match": global_transitions == args.transition_budget,
            "ppo_updates": update_index,
            "training_wall_clock_seconds": time.perf_counter() - training_started,
            "final_checkpoint": str(final_checkpoint),
            "final_checkpoint_sha256": file_sha256(final_checkpoint),
            "final_constraints": constraints.as_dict(),
            "training_episode_summary": episodes.recent_summary(maximum=max(len(episodes.completed), 1)),
            "final_navigation": None if evaluation is None else {
                key: value for key, value in evaluation.items() if key != "records"
            },
        }
        atomic_json(output / "COMPLETED.json", completed)
        (output / "RUNNING.json").unlink(missing_ok=True)
        return completed
    except KeyboardInterrupt:
        interrupted_checkpoint = output / "checkpoints" / f"checkpoint_interrupted_{global_transitions:09d}.pt"
        save_checkpoint(
            interrupted_checkpoint,
            model=model,
            optimizer=optimizer,
            constraints=constraints,
            model_config=model_config,
            ppo_config=ppo_config,
            global_transitions=global_transitions,
            update_index=update_index,
            episode_seed_cursor=seed_cursor,
        )
        atomic_json(
            output / "INTERRUPTED.json",
            {
                "status": "INTERRUPTED",
                "protocol": PROTOCOL,
                "interrupted_at": utc_now(),
                "global_env_transitions": global_transitions,
                "checkpoint": str(interrupted_checkpoint),
                "reason": "SIGINT_or_keyboard_interrupt",
            },
        )
        (output / "RUNNING.json").unlink(missing_ok=True)
        raise
    except BaseException as error:
        atomic_json(
            output / "FAILED.json",
            {
                "status": "FAILED",
                "protocol": PROTOCOL,
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
    args = parse_args()
    result = train(args)
    print(json.dumps({key: value for key, value in result.items() if key != "git_status"}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
