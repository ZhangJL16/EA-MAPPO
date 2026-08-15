from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import gymnasium
import matplotlib
import numpy as np
import stable_baselines3
import torch
from PIL import Image
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv

from envs.UAVEnergyDeliverySAC import (
    ENERGY_GAMMA,
    ENERGY_QUANTILES,
    ENERGY_UNIT,
    GoalConditionedQuantileTDEnergyEstimator,
    SACTrainingPhase,
    UAVEnergyDeliverySACEnv,
)
from review_bundle.envs.navigation.telemetry_cost import TelemetryCostConfig


matplotlib.use("Agg")
from matplotlib import pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
FORMAL_PHASE1_TRANSITION_BUDGET = 500_000
FORMAL_PHASE2_TRANSITION_BUDGET = 500_000
DISTANCE_BUCKETS = (
    ("100-500", 100.0, 500.0),
    ("500-1500", 500.0, 1500.0),
    ("1500-2500", 1500.0, 2500.0),
    ("2500-4000", 2500.0, 4000.0),
    (">4000", 4000.0, float("inf")),
)


class DeterministicPolicy(Protocol):
    def predict(self, observation: np.ndarray, deterministic: bool = True): ...


@dataclass(frozen=True)
class NavigationTask:
    start_position: np.ndarray
    goal_position: np.ndarray
    initial_velocity: np.ndarray
    straight_line_distance: float
    distance_bucket: str

    def as_dict(self) -> dict[str, object]:
        return {
            "start_position": self.start_position.tolist(),
            "goal_position": self.goal_position.tolist(),
            "initial_velocity": self.initial_velocity.tolist(),
            "straight_line_distance": self.straight_line_distance,
            "distance_bucket": self.distance_bucket,
        }


class HeuristicGoalPolicy:
    def predict(self, observation: np.ndarray, deterministic: bool = True):
        del deterministic
        values = np.asarray(observation, dtype=np.float32)
        if values.ndim == 1:
            velocity = values[:3]
            direction = values[3:6]
            action = np.clip(1.5 * direction - 0.7 * velocity, -1.0, 1.0).astype(np.float32)
            return action, None
        velocity = values[:, :3]
        direction = values[:, 3:6]
        actions = np.clip(1.5 * direction - 0.7 * velocity, -1.0, 1.0).astype(np.float32)
        return actions, None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def git_clean() -> bool:
    return not subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    return value


def write_json(path: Path, payload: dict[str, object] | list[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_value(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(json_value(payload), sort_keys=True) + "\n")


def append_csv(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    normalized = {key: json_value(value) for key, value in payload.items()}
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(normalized))
        if not exists:
            writer.writeheader()
        writer.writerow(normalized)


def telemetry_config_from_args(args: argparse.Namespace) -> TelemetryCostConfig:
    return TelemetryCostConfig(
        base_power=args.base_power,
        velocity_coefficients=np.asarray(args.velocity_coefficients, dtype=np.float64),
        acceleration_coefficients=np.asarray(args.acceleration_coefficients, dtype=np.float64),
        compute_power=args.compute_power,
        communication_power=args.communication_power,
        simulation_error=args.simulation_error,
        flight_energy_multiplier=args.flight_energy_multiplier,
    )


def environment_from_args(
    args: argparse.Namespace,
    *,
    phase: SACTrainingPhase,
    render_mode: str | None = None,
    battery_capacity: float | None = None,
) -> UAVEnergyDeliverySACEnv:
    environment = UAVEnergyDeliverySACEnv(
        phase=phase,
        render_mode=render_mode,
        minimum_task_distance=args.minimum_task_distance,
        xy_sampling_margin=args.xy_sampling_margin,
        task_z_min=args.task_z_min,
        task_z_max=args.task_z_max,
        max_steps_per_task=args.phase1_episode_max_steps,
        phase1_episode_max_policy_steps=args.phase1_episode_max_steps,
        phase2_episode_limit=args.phase2_episode_max_steps,
        operational_energy_capacity=None,
        energy_reserve_fraction=args.energy_reserve_fraction,
        telemetry_cost_config=telemetry_config_from_args(args),
        render_vertical_exaggeration=args.render_vertical_exaggeration,
    )
    if battery_capacity is not None:
        environment.configure_calibrated_battery(
            battery_capacity,
            reserve_fraction=args.energy_reserve_fraction,
            source="phase1_frozen_policy_calibration",
        )
    return environment


def single_action_provider(policy: DeterministicPolicy):
    def provider(observation: np.ndarray) -> np.ndarray:
        action, _ = policy.predict(observation, deterministic=True)
        return np.asarray(action, dtype=np.float32)

    return provider


def batch_action_provider(policy: DeterministicPolicy):
    def provider(observations: np.ndarray) -> np.ndarray:
        actions, _ = policy.predict(observations, deterministic=True)
        return np.asarray(actions, dtype=np.float32)

    return provider


def _bucket_for_distance(distance: float) -> str:
    for name, lower, upper in DISTANCE_BUCKETS:
        if lower <= distance < upper:
            return name
    raise ValueError(f"distance {distance} is outside configured buckets")


def generate_stratified_navigation_tasks(
    *,
    num_tasks: int,
    seed: int,
    length: float = 4000.0,
    width: float = 4000.0,
    height: float = 400.0,
    xy_margin: float = 100.0,
    z_min: float = 20.0,
    z_max: float = 380.0,
) -> list[NavigationTask]:
    if num_tasks <= 0 or num_tasks % len(DISTANCE_BUCKETS) != 0:
        raise ValueError("stratified task count must be a positive multiple of five")
    rng = np.random.default_rng(seed)
    tasks: list[NavigationTask] = []
    per_bucket = num_tasks // len(DISTANCE_BUCKETS)
    for bucket_name, lower, upper in DISTANCE_BUCKETS:
        accepted = 0
        attempts = 0
        while accepted < per_bucket:
            attempts += 1
            if attempts > 500_000:
                raise RuntimeError(f"could not sample enough tasks for bucket {bucket_name}")
            start = np.array(
                [
                    rng.uniform(xy_margin, length - xy_margin),
                    rng.uniform(xy_margin, width - xy_margin),
                    rng.uniform(z_min, z_max),
                ],
                dtype=np.float32,
            )
            goal = np.array(
                [
                    rng.uniform(xy_margin, length - xy_margin),
                    rng.uniform(xy_margin, width - xy_margin),
                    rng.uniform(z_min, z_max),
                ],
                dtype=np.float32,
            )
            distance = float(np.linalg.norm(goal - start))
            if not lower <= distance < upper:
                continue
            tasks.append(
                NavigationTask(
                    start,
                    goal,
                    np.zeros(3, dtype=np.float32),
                    distance,
                    bucket_name,
                )
            )
            accepted += 1
    return tasks


def save_navigation_tasks(path: Path, tasks: list[NavigationTask], *, seed: int, role: str) -> None:
    write_json(
        path,
        {
            "role": role,
            "seed": seed,
            "num_tasks": len(tasks),
            "distance_buckets": [row[:3] for row in DISTANCE_BUCKETS],
            "tasks": [task.as_dict() for task in tasks],
        },
    )


def load_navigation_tasks(path: Path) -> list[NavigationTask]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        NavigationTask(
            np.asarray(row["start_position"], dtype=np.float32),
            np.asarray(row["goal_position"], dtype=np.float32),
            np.asarray(row["initial_velocity"], dtype=np.float32),
            float(row["straight_line_distance"]),
            str(row["distance_bucket"]),
        )
        for row in payload["tasks"]
    ]


def evaluate_navigation_tasks(
    policy: DeterministicPolicy,
    args: argparse.Namespace,
    tasks: list[NavigationTask],
    *,
    global_env_transitions: int,
    output_path: Path | None = None,
) -> dict[str, object]:
    environment = environment_from_args(args, phase=SACTrainingPhase.NAVIGATION)
    records: list[dict[str, object]] = []
    evaluation_transitions = 0
    for task_index, task in enumerate(tasks):
        observation, _ = environment.reset(
            seed=args.eval_task_seed + task_index,
            options={
                "start_position": task.start_position,
                "start_velocity": task.initial_velocity,
                "task_point": task.goal_position,
            },
        )
        boundary_contacts = 0
        reward_total = 0.0
        while True:
            action, _ = policy.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = environment.step(action)
            evaluation_transitions += 1
            reward_total += float(reward)
            boundary_contacts += int(info["boundary_contact"])
            if terminated or truncated:
                success = bool(info["is_success"])
                actual_path = float(environment._current_goal_path_length)
                records.append(
                    {
                        "task_index": task_index,
                        "distance_bucket": task.distance_bucket,
                        "straight_line_distance": task.straight_line_distance,
                        "success": success,
                        "policy_steps": int(environment.current_step),
                        "path_length": actual_path,
                        "path_ratio": actual_path / max(task.straight_line_distance, 1e-8),
                        "reward": reward_total,
                        "boundary_contacts": boundary_contacts,
                        "end_reason": info["end_reason"],
                    }
                )
                break
    bucket_success = {}
    for bucket_name, _, _ in DISTANCE_BUCKETS:
        subset = [row for row in records if row["distance_bucket"] == bucket_name]
        bucket_success[bucket_name] = (
            None if not subset else float(np.mean([row["success"] for row in subset]))
        )
    summary = {
        "global_env_transitions": int(global_env_transitions),
        "evaluation_env_transitions": int(evaluation_transitions),
        "num_tasks": len(records),
        "overall_success_rate": float(np.mean([row["success"] for row in records])),
        "distance_bucket_success": bucket_success,
        "mean_steps_per_task": float(np.mean([row["policy_steps"] for row in records])),
        "mean_path_ratio": float(np.mean([row["path_ratio"] for row in records])),
        "boundary_contact_rate": float(
            sum(int(row["boundary_contacts"]) for row in records) / max(evaluation_transitions, 1)
        ),
        "mean_reward": float(np.mean([row["reward"] for row in records])),
        "records": records,
    }
    if output_path is not None:
        write_json(output_path, summary)
    environment.close()
    return summary


def navigation_gate_passed(summary: dict[str, object]) -> bool:
    bucket_success = summary["distance_bucket_success"]
    if any(value is None for value in bucket_success.values()):
        return False
    return bool(
        summary["overall_success_rate"] >= 0.98
        and min(float(value) for value in bucket_success.values()) >= 0.95
        and summary["mean_path_ratio"] <= 1.10
        and summary["boundary_contact_rate"] < 0.01
    )


def make_navigation_vec_env(args: argparse.Namespace) -> DummyVecEnv:
    factories = []
    for env_index in range(args.num_envs):
        environment_seed = args.seed + env_index

        def factory(seed: int = environment_seed):
            environment = environment_from_args(args, phase=SACTrainingPhase.NAVIGATION)
            environment.reset(seed=seed)
            environment.action_space.seed(seed)
            return environment

        factories.append(factory)
    vector_environment = DummyVecEnv(factories)
    vector_environment.seed(args.seed)
    return vector_environment


class NavigationBudgetCallback(BaseCallback):
    def __init__(
        self,
        *,
        args: argparse.Namespace,
        output: Path,
        eval_tasks: list[NavigationTask],
    ) -> None:
        super().__init__(verbose=0)
        self.args = args
        self.output = output
        self.eval_tasks = eval_tasks
        self.num_envs = int(args.num_envs)
        self.vector_env_steps = 0
        self.completed_training_episodes = 0
        self.successful_training_episodes = 0
        self.failed_training_episodes = 0
        self.evaluation_env_transitions = 0
        self.gif_evaluation_env_transitions = 0
        self.first_convergence_transition: int | None = None
        self.stable_convergence_transition: int | None = None
        self._gate_streak = 0
        self._next_eval = int(args.eval_freq_transitions)
        self._next_checkpoint = int(args.checkpoint_freq_transitions)
        self._next_gif = int(args.gif_freq_transitions)
        self._episode_rewards = deque(maxlen=1000)
        self._episode_reward_by_env = np.zeros(self.num_envs, dtype=np.float64)
        self._completed_episode_steps = deque(maxlen=1000)
        self._completed_path_ratios = deque(maxlen=1000)
        self._interval_rewards: list[float] = []
        self._interval_infos: list[dict[str, object]] = []

    def _on_step(self) -> bool:
        self.vector_env_steps += 1
        if self.num_timesteps != self.vector_env_steps * self.num_envs:
            raise RuntimeError("SB3 num_timesteps does not match environment-transition accounting")
        rewards = np.asarray(self.locals["rewards"], dtype=np.float64)
        dones = np.asarray(self.locals["dones"], dtype=bool)
        infos = self.locals["infos"]
        self._episode_reward_by_env += rewards
        self._interval_rewards.extend(float(value) for value in rewards)
        self._interval_infos.extend(dict(info) for info in infos)
        for env_index, done in enumerate(dones):
            if not done:
                continue
            self.completed_training_episodes += 1
            success = bool(infos[env_index].get("is_success", False))
            self.successful_training_episodes += int(success)
            self.failed_training_episodes += int(not success)
            self._episode_rewards.append(float(self._episode_reward_by_env[env_index]))
            self._completed_episode_steps.append(int(infos[env_index].get("episode_policy_steps", 0)))
            self._completed_path_ratios.append(float(infos[env_index].get("goal_path_ratio", 0.0)))
            self._episode_reward_by_env[env_index] = 0.0
        if self.num_timesteps >= self._next_checkpoint:
            checkpoint = self.output / "phase1_navigation" / f"checkpoint_transition_{self._next_checkpoint:06d}"
            self.model.save(checkpoint)
            self._next_checkpoint += self.args.checkpoint_freq_transitions
        if self.num_timesteps >= self._next_eval:
            evaluation = evaluate_navigation_tasks(
                self.model,
                self.args,
                self.eval_tasks,
                global_env_transitions=self.num_timesteps,
                output_path=self.output / "eval" / f"eval_transition_{self._next_eval:06d}.json",
            )
            self.evaluation_env_transitions += int(evaluation["evaluation_env_transitions"])
            if navigation_gate_passed(evaluation):
                self._gate_streak += 1
                if self.first_convergence_transition is None:
                    self.first_convergence_transition = self.num_timesteps
                if self._gate_streak >= 3 and self.stable_convergence_transition is None:
                    self.stable_convergence_transition = self.num_timesteps
            else:
                self._gate_streak = 0
            self._next_eval += self.args.eval_freq_transitions
        if self.eval_tasks and self.num_timesteps >= self._next_gif:
            gif_transitions = generate_eval_gif(
                self.model,
                self.args,
                self.eval_tasks[0],
                self.output / "gifs" / f"eval_transition_{self._next_gif:06d}.gif",
            )
            self.evaluation_env_transitions += gif_transitions
            self.gif_evaluation_env_transitions += gif_transitions
            self._next_gif += self.args.gif_freq_transitions
        if self.num_timesteps % self.args.log_freq_transitions == 0:
            component_names = (
                "progress_reward_component",
                "velocity_reward_component",
                "task_completion_reward_component",
                "time_penalty_component",
                "boundary_penalty_component",
            )
            row = {
                "global_env_transitions": int(self.num_timesteps),
                "vector_env_steps": int(self.vector_env_steps),
                "num_envs": self.num_envs,
                "completed_training_episodes": self.completed_training_episodes,
                "successful_training_episodes": self.successful_training_episodes,
                "failed_training_episodes": self.failed_training_episodes,
                "partial_episode_at_training_stop": False,
                "mean_step_reward": float(np.mean(self._interval_rewards)),
                "tasks_completed_in_interval": int(
                    sum(bool(info["task_completed_now"]) for info in self._interval_infos)
                ),
                "tasks_per_1000_steps": float(
                    1000.0
                    * sum(bool(info["task_completed_now"]) for info in self._interval_infos)
                    / max(len(self._interval_infos), 1)
                ),
                "mean_speed": float(np.mean([info["speed"] for info in self._interval_infos])),
                "mean_horizontal_speed": float(
                    np.mean([info["horizontal_speed"] for info in self._interval_infos])
                ),
                "mean_vertical_speed": float(
                    np.mean([info["vertical_speed"] for info in self._interval_infos])
                ),
                "boundary_contact_count": int(
                    sum(bool(info["boundary_contact"]) for info in self._interval_infos)
                ),
                "boundary_contact_rate": float(
                    np.mean([bool(info["boundary_contact"]) for info in self._interval_infos])
                ),
                "task_stuck_count": int(
                    sum(bool(info["task_stuck"]) for info in self._interval_infos)
                ),
                "mean_steps_per_completed_task": (
                    None
                    if not self._completed_episode_steps
                    else float(np.mean(self._completed_episode_steps))
                ),
                "mean_path_ratio": (
                    None
                    if not self._completed_path_ratios
                    else float(np.mean(self._completed_path_ratios))
                ),
                "rolling_mean_reward_100": (
                    None if not self._episode_rewards else float(np.mean(list(self._episode_rewards)[-100:]))
                ),
                "rolling_mean_reward_1000": (
                    None if not self._episode_rewards else float(np.mean(self._episode_rewards))
                ),
            }
            for component_name in component_names:
                row[component_name] = float(
                    np.mean(
                        [
                            info["reward_components"][component_name]
                            for info in self._interval_infos
                        ]
                    )
                )
            append_jsonl(self.output / "training_curve.jsonl", row)
            append_csv(self.output / "training_curve.csv", row)
            self._interval_rewards = []
            self._interval_infos = []
        return True

    def audit(self) -> dict[str, object]:
        return {
            "global_env_transitions": int(self.num_timesteps),
            "vector_env_steps": int(self.vector_env_steps),
            "num_envs": self.num_envs,
            "completed_training_episodes": self.completed_training_episodes,
            "successful_training_episodes": self.successful_training_episodes,
            "failed_training_episodes": self.failed_training_episodes,
            "partial_episodes_at_budget_stop": self.num_envs,
            "training_stop_reason": "transition_budget_reached",
            "evaluation_env_transitions": self.evaluation_env_transitions,
            "gif_evaluation_env_transitions": self.gif_evaluation_env_transitions,
            "first_convergence_transition": self.first_convergence_transition,
            "stable_convergence_transition": self.stable_convergence_transition,
        }


def train_navigation_fixed_budget(
    args: argparse.Namespace,
    output: Path,
    eval_tasks: list[NavigationTask],
) -> tuple[SAC, dict[str, object], Path]:
    if args.phase1_transition_budget % args.num_envs != 0:
        raise ValueError("phase1_transition_budget must be exactly divisible by num_envs")
    vector_environment = make_navigation_vec_env(args)
    model = SAC(
        "MlpPolicy",
        vector_environment,
        seed=args.seed,
        device=args.device,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        tau=args.tau,
        gamma=args.gamma,
        train_freq=(1, "step"),
        verbose=1,
        tensorboard_log=str(output / "tensorboard"),
    )
    callback = NavigationBudgetCallback(args=args, output=output, eval_tasks=eval_tasks)
    model.learn(
        total_timesteps=args.phase1_transition_budget,
        callback=callback,
        reset_num_timesteps=True,
        progress_bar=False,
    )
    if model.num_timesteps != args.phase1_transition_budget:
        raise RuntimeError("Phase 1A did not stop at the exact transition budget")
    final_checkpoint = output / "phase1_navigation" / f"checkpoint_transition_{args.phase1_transition_budget:06d}"
    model.save(final_checkpoint)
    audit = callback.audit()
    audit.update(
        {
            "requested_transition_budget": args.phase1_transition_budget,
            "actual_training_transitions": int(model.num_timesteps),
            "exact_budget_match": model.num_timesteps == args.phase1_transition_budget,
            "finish_last_episode_after_budget": False,
            "final_checkpoint": str(final_checkpoint.with_suffix(".zip")),
        }
    )
    write_json(output / "phase1_navigation" / "summary.json", audit)
    vector_environment.close()
    return model, audit, final_checkpoint.with_suffix(".zip")


def freeze_navigation_and_start_td(
    model: SAC,
    estimator: GoalConditionedQuantileTDEnergyEstimator,
) -> dict[str, object]:
    freeze_navigation_policy(model)
    estimator.clear_replay()
    estimator.set_next_action_provider(batch_action_provider(model))
    return {
        "sac_frozen": all(not parameter.requires_grad for parameter in model.policy.parameters()),
        "td_replay_size": len(estimator.replay),
        "td_update_count": estimator.update_count,
    }


def freeze_navigation_policy(model: SAC) -> None:
    for parameter in model.policy.parameters():
        parameter.requires_grad_(False)
    model.policy.set_training_mode(False)


def calculate_battery_capacities(mean_power: float, target_minutes: float) -> dict[str, float]:
    power = float(mean_power)
    minutes = float(target_minutes)
    if not np.isfinite(power) or power <= 0.0:
        raise ValueError("mean_power must be finite and positive")
    if not np.isfinite(minutes) or minutes <= 0.0:
        raise ValueError("target endurance must be finite and positive")
    capacities = {f"capacity_for_{duration}min": power * duration * 60.0 for duration in (20, 30, 40)}
    capacities["calibrated_battery_capacity"] = power * minutes * 60.0
    return capacities


def _policy_parameter_hash(policy: DeterministicPolicy) -> str | None:
    if not isinstance(policy, SAC):
        return None
    digest = hashlib.sha256()
    for parameter in policy.policy.parameters():
        digest.update(parameter.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def run_battery_calibration(
    policy: DeterministicPolicy,
    args: argparse.Namespace,
    tasks: list[NavigationTask],
    output: Path,
) -> dict[str, object]:
    if isinstance(policy, SAC) and any(parameter.requires_grad for parameter in policy.policy.parameters()):
        raise RuntimeError("battery calibration requires a frozen SAC policy")
    environment = environment_from_args(args, phase=SACTrainingPhase.NAVIGATION)
    before_hash = _policy_parameter_hash(policy)
    task_records: list[dict[str, object]] = []
    full_policy_step_energies: list[float] = []
    calibration_transitions = 0
    for task_index, task in enumerate(tasks):
        observation, _ = environment.reset(
            seed=args.battery_calibration_seed + task_index,
            options={
                "start_position": task.start_position,
                "start_velocity": task.initial_velocity,
                "task_point": task.goal_position,
            },
        )
        total_energy = 0.0
        total_time = 0.0
        path_length = 0.0
        physics_steps = 0
        speeds: list[float] = []
        while True:
            previous_position = environment.agent.pos.copy()
            action, _ = policy.predict(observation, deterministic=True)
            observation, _, terminated, truncated, info = environment.step(action)
            calibration_transitions += 1
            total_energy += float(info["realized_energy_cost"])
            total_time += float(info["transition_dt"])
            physics_steps += int(info["physics_substeps"])
            path_length += float(np.linalg.norm(environment.agent.pos - previous_position))
            speeds.append(float(np.linalg.norm(environment.agent.vel)))
            if np.isclose(info["transition_dt"], environment.policy_dt):
                full_policy_step_energies.append(float(info["realized_energy_cost"]))
            if terminated or truncated:
                success = bool(info["is_success"])
                task_records.append(
                    {
                        **task.as_dict(),
                        "success": success,
                        "actual_path_length": path_length,
                        "simulation_flight_time": total_time,
                        "policy_steps": int(environment.current_step),
                        "physics_steps": physics_steps,
                        "total_realized_energy": total_energy,
                        "mean_realized_power": total_energy / max(total_time, 1e-12),
                        "mean_speed": float(np.mean(speeds)),
                        "max_speed": float(np.max(speeds)),
                        "end_reason": info["end_reason"],
                    }
                )
                break
    after_hash = _policy_parameter_hash(policy)
    successful = [row for row in task_records if row["success"]]
    if not successful:
        raise RuntimeError("battery calibration has no successful frozen-policy trajectories")
    total_flight_time = float(sum(row["simulation_flight_time"] for row in successful))
    total_energy = float(sum(row["total_realized_energy"] for row in successful))
    mean_power = total_energy / total_flight_time
    task_powers = np.asarray([row["mean_realized_power"] for row in successful], dtype=np.float64)
    capacities = calculate_battery_capacities(mean_power, args.target_nominal_endurance_minutes)
    summary = {
        "stage": "battery_calibration",
        "energy_source": "TelemetryCostModel.realized_cost",
        "energy_unit": ENERGY_UNIT,
        "power_unit": f"{ENERGY_UNIT}/simulation_second",
        "num_tasks": len(task_records),
        "num_successful_tasks": len(successful),
        "num_failed_tasks": len(task_records) - len(successful),
        "battery_calibration_env_transitions": calibration_transitions,
        "total_flight_time": total_flight_time,
        "total_realized_energy": total_energy,
        "mean_power": mean_power,
        "median_task_mean_power": float(np.median(task_powers)),
        "std_task_mean_power": float(np.std(task_powers)),
        "p50_task_mean_power": float(np.quantile(task_powers, 0.50)),
        "p75_task_mean_power": float(np.quantile(task_powers, 0.75)),
        "p90_task_mean_power": float(np.quantile(task_powers, 0.90)),
        "p95_task_mean_power": float(np.quantile(task_powers, 0.95)),
        "mean_energy_per_full_policy_step": float(np.mean(full_policy_step_energies)),
        "mean_task_energy": float(np.mean([row["total_realized_energy"] for row in successful])),
        "mean_task_distance": float(np.mean([row["straight_line_distance"] for row in successful])),
        "mean_actual_path_length": float(np.mean([row["actual_path_length"] for row in successful])),
        "mean_path_ratio": float(
            np.mean(
                [row["actual_path_length"] / row["straight_line_distance"] for row in successful]
            )
        ),
        "target_nominal_endurance_minutes": args.target_nominal_endurance_minutes,
        **capacities,
        "equivalent_nominal_steps_20min": int(round(20.0 * 60.0 / environment.policy_dt)),
        "equivalent_nominal_steps_30min": int(round(30.0 * 60.0 / environment.policy_dt)),
        "equivalent_nominal_steps_40min": int(round(40.0 * 60.0 / environment.policy_dt)),
        "sac_deterministic": True,
        "sac_training": False,
        "td_training": False,
        "td_replay_writes": 0,
        "policy_parameter_hash_before": before_hash,
        "policy_parameter_hash_after": after_hash,
        "policy_unchanged": before_hash == after_hash,
        "tasks": task_records,
    }
    write_json(output / "battery_calibration.json", summary)
    write_json(output / "battery_calibration_tasks.json", task_records)
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.hist(task_powers, bins=min(20, max(5, len(task_powers))), color="#4472c4")
    axis.set_xlabel(f"Mean power ({ENERGY_UNIT}/s)")
    axis.set_ylabel("Tasks")
    figure.tight_layout()
    figure.savefig(output / "battery_calibration_power_distribution.png", dpi=160)
    plt.close(figure)
    print("\n=== BATTERY CALIBRATION ===")
    print(f"Calibration tasks: {len(task_records)}")
    print(f"Total flight time: {total_flight_time:.3f} s")
    print(f"Mean realized power: {mean_power:.6f} {ENERGY_UNIT}/s")
    for duration in (20, 30, 40):
        print(f"{duration}-min capacity: {capacities[f'capacity_for_{duration}min']:.6f}")
    print(f"Selected nominal endurance: {args.target_nominal_endurance_minutes:g} min")
    print(f"Selected battery capacity: {capacities['calibrated_battery_capacity']:.6f}")
    print(
        "Equivalent nominal policy steps: "
        f"{args.target_nominal_endurance_minutes * 60.0 / environment.policy_dt:.0f}"
    )
    environment.close()
    return summary


def run_battery_validation(
    policy: DeterministicPolicy,
    args: argparse.Namespace,
    *,
    battery_capacity: float,
    output: Path,
) -> dict[str, object]:
    records: list[dict[str, object]] = []
    validation_transitions = 0
    for run_index in range(args.battery_validation_runs):
        environment = environment_from_args(
            args,
            phase=SACTrainingPhase.NAVIGATION,
            battery_capacity=battery_capacity,
        )
        environment.enable_battery_validation()
        observation, _ = environment.reset(seed=args.battery_validation_seed + run_index)
        distance_flown = 0.0
        while True:
            previous_position = environment.agent.pos.copy()
            action, _ = policy.predict(observation, deterministic=True)
            observation, _, terminated, truncated, info = environment.step(action)
            validation_transitions += 1
            distance_flown += float(np.linalg.norm(environment.agent.pos - previous_position))
            if terminated or truncated:
                records.append(
                    {
                        "run_index": run_index,
                        "actual_depletion_time": float(environment.simulation_time),
                        "actual_policy_steps_to_depletion": int(environment.current_step),
                        "tasks_before_depletion": int(environment.tasks_completed),
                        "distance_before_depletion": distance_flown,
                        "energy_exhausted": bool(terminated and info["end_reason"] == "energy_exhausted"),
                        "truncated": bool(truncated),
                        "end_reason": info["end_reason"],
                    }
                )
                environment.close()
                break
    depletion_times = np.asarray([row["actual_depletion_time"] for row in records], dtype=np.float64)
    expected_seconds = args.target_nominal_endurance_minutes * 60.0
    observed_mean = float(np.mean(depletion_times))
    relative_error = (observed_mean - expected_seconds) / expected_seconds
    all_depleted = all(row["energy_exhausted"] for row in records)
    calibration_valid = bool(all_depleted and abs(relative_error) <= 0.20)
    summary = {
        "stage": "battery_endurance_validation",
        "engineering_calibration_tolerance_fraction": 0.20,
        "target_nominal_endurance_minutes": args.target_nominal_endurance_minutes,
        "target_nominal_endurance_seconds": expected_seconds,
        "battery_capacity": battery_capacity,
        "battery_validation_runs": len(records),
        "battery_validation_env_transitions": validation_transitions,
        "mean_depletion_time": observed_mean,
        "median_depletion_time": float(np.median(depletion_times)),
        "std_depletion_time": float(np.std(depletion_times)),
        "p10_depletion_time": float(np.quantile(depletion_times, 0.10)),
        "p25_depletion_time": float(np.quantile(depletion_times, 0.25)),
        "p50_depletion_time": float(np.quantile(depletion_times, 0.50)),
        "p75_depletion_time": float(np.quantile(depletion_times, 0.75)),
        "p90_depletion_time": float(np.quantile(depletion_times, 0.90)),
        "mean_policy_steps_to_depletion": float(
            np.mean([row["actual_policy_steps_to_depletion"] for row in records])
        ),
        "mean_tasks_before_depletion": float(np.mean([row["tasks_before_depletion"] for row in records])),
        "mean_distance_before_depletion": float(
            np.mean([row["distance_before_depletion"] for row in records])
        ),
        "relative_endurance_error": relative_error,
        "all_runs_depleted": all_depleted,
        "battery_calibration_valid": calibration_valid,
        "records": records,
    }
    write_json(output / "battery_validation.json", summary)
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.hist(depletion_times / 60.0, bins=min(20, max(5, len(depletion_times))), color="#70ad47")
    axis.axvline(args.target_nominal_endurance_minutes, color="black", linestyle="--")
    axis.set_xlabel("Depletion time (simulation minutes)")
    axis.set_ylabel("Runs")
    figure.tight_layout()
    figure.savefig(output / "battery_validation_depletion_time.png", dpi=160)
    plt.close(figure)
    print("\n=== BATTERY VALIDATION ===")
    print(f"Target nominal endurance: {expected_seconds:.3f} s")
    print(f"Observed mean depletion time: {observed_mean:.3f} s")
    print(f"Observed median depletion time: {np.median(depletion_times):.3f} s")
    print(
        "Mean policy transitions to depletion: "
        f"{summary['mean_policy_steps_to_depletion']:.2f}"
    )
    print(f"Mean tasks before depletion: {summary['mean_tasks_before_depletion']:.2f}")
    print(f"Mean distance before depletion: {summary['mean_distance_before_depletion']:.2f} m")
    print(f"Relative endurance error: {relative_error:.3%}")
    print(f"Calibration sanity: {'PASS' if calibration_valid else 'FAIL'}")
    return summary


def run_td_pretraining(
    policy: DeterministicPolicy,
    estimator: GoalConditionedQuantileTDEnergyEstimator,
    args: argparse.Namespace,
    *,
    transition_budget: int,
    output: Path,
) -> dict[str, object]:
    environment = environment_from_args(args, phase=SACTrainingPhase.TD_PRETRAINING)
    environment.bind_energy_learning(
        energy_estimator=estimator,
        goal_action_provider=single_action_provider(policy),
        training_enabled=True,
    )
    observation, _ = environment.reset(seed=args.td_collection_seed)
    transitions = 0
    successful_segments = 0
    cumulative_energy = 0.0
    diagnostic_rows: list[dict[str, object]] = []
    goal_evaluations: list[dict[str, object]] = []
    while transitions < transition_budget:
        action, _ = policy.predict(observation, deterministic=True)
        observation, _, terminated, truncated, info = environment.step(action)
        transitions += 1
        cumulative_energy += float(info["realized_energy_cost"])
        if transitions % args.log_freq_transitions == 0 or terminated or truncated:
            completed = info["completed_goal_evaluation"]
            if completed is not None:
                goal_evaluations.append(dict(completed))
            row = {
                "transition": transitions,
                "cumulative_energy": cumulative_energy,
                "step_energy": float(info["realized_energy_cost"]),
                "td_loss": info["td_loss"],
                "td_update_count": info["td_update_count"],
                "td_replay_size": info["td_replay_size"],
                "td_mae": None if completed is None else completed.get("td_mae"),
                "td_rmse": None if completed is None else completed.get("td_rmse"),
                "q50_coverage": None if completed is None else completed.get("q50_coverage"),
                "q90_coverage": None if completed is None else completed.get("q90_coverage"),
                "q95_coverage": None if completed is None else completed.get("q95_coverage"),
                "q99_coverage": None if completed is None else completed.get("q99_coverage"),
                "Q50": info["Q50"],
                "Q90": info["Q90"],
                "Q95": info["Q95"],
                "Q99": info["Q99"],
            }
            diagnostic_rows.append(row)
            append_jsonl(output / "phase1_td" / "training_curve.jsonl", row)
        if terminated or truncated:
            successful_segments += int(info["is_success"])
            observation, _ = environment.reset(seed=args.td_collection_seed + transitions)
    estimator.save(output / "phase1_td" / "td_energy_phase1.pt")
    summary = {
        "transition_budget": transition_budget,
        "actual_transitions": transitions,
        "successful_goal_segments": successful_segments,
        "td_replay_size": len(estimator.replay),
        "td_trainable_replay_size": len(estimator.trainable_replay),
        "td_update_count": estimator.update_count,
        "td_last_loss": estimator.last_loss,
        "sac_frozen": True,
        "collection_environment_recreated": True,
        "td_evaluation": aggregate_goal_evaluations(goal_evaluations),
    }
    write_json(output / "phase1_td" / "summary.json", summary)
    _plot_td_diagnostics(diagnostic_rows, output)
    environment.close()
    return summary


def aggregate_goal_evaluations(records: list[dict[str, object]]) -> dict[str, object]:
    metric_names = (
        "td_mae",
        "td_rmse",
        "td_bias",
        "td_underestimation_rate",
        "td_overestimation_rate",
        "td_mean_underestimation_magnitude",
        "td_relative_error",
        "q50_coverage",
        "q90_coverage",
        "q95_coverage",
        "q99_coverage",
    )

    def aggregate(rows: list[dict[str, object]]) -> dict[str, object]:
        if not rows:
            return {"num_completed_goals": 0}
        return {
            "num_completed_goals": len(rows),
            **{
                metric: float(np.mean([float(row[metric]) for row in rows]))
                for metric in metric_names
            },
        }

    return {
        "overall": aggregate(records),
        "by_initial_goal_distance": {
            bucket_name: aggregate(
                [row for row in records if row["distance_bucket"] == bucket_name]
            )
            for bucket_name, _, _ in DISTANCE_BUCKETS
        },
    }


def _plot_td_diagnostics(rows: list[dict[str, object]], output: Path) -> None:
    if not rows:
        return
    transitions = np.asarray([row["transition"] for row in rows])

    def finite_series(key: str) -> tuple[np.ndarray, np.ndarray]:
        selected = [(row["transition"], row[key]) for row in rows if row[key] is not None]
        if not selected:
            return np.asarray([]), np.asarray([])
        return np.asarray([row[0] for row in selected]), np.asarray([row[1] for row in selected])

    figure, axis = plt.subplots(figsize=(8, 4))
    axis.plot(transitions, [row["step_energy"] for row in rows], alpha=0.8)
    axis.set_xlabel("TD collection transitions")
    axis.set_ylabel(f"Realized energy ({ENERGY_UNIT})")
    figure.tight_layout()
    figure.savefig(output / "energy_curve.png", dpi=160)
    plt.close(figure)
    for filename, keys, ylabel in (
        ("td_loss_curve.png", ("td_loss",), "TD loss"),
        ("td_error_curve.png", ("td_mae", "td_rmse"), "Energy error"),
        (
            "quantile_coverage_curve.png",
            ("q50_coverage", "q90_coverage", "q95_coverage", "q99_coverage"),
            "Empirical coverage",
        ),
    ):
        figure, axis = plt.subplots(figsize=(8, 4))
        for key in keys:
            selected_steps, values = finite_series(key)
            if values.size:
                axis.plot(selected_steps, values, label=key)
        axis.set_xlabel("TD collection transitions")
        axis.set_ylabel(ylabel)
        if axis.lines:
            axis.legend()
        figure.tight_layout()
        figure.savefig(output / filename, dpi=160)
        plt.close(figure)


class ConstantQuantileEstimator:
    def __init__(self, value: float) -> None:
        self.value = float(value)
        self.replay: deque[object] = deque()
        self.update_count = 0

    def predict_quantiles(self, energy_state: np.ndarray, action: np.ndarray) -> np.ndarray:
        del energy_state, action
        return np.full(4, self.value, dtype=np.float64)

    def predict(self, energy_state: np.ndarray, action: np.ndarray, quantile: float = 0.95) -> float:
        del energy_state, action, quantile
        return self.value

    def observe_transition(self, *args, **kwargs):
        self.replay.append((args, kwargs))
        return None


def run_phase2_state_machine_smoke(
    args: argparse.Namespace,
    *,
    battery_capacity: float,
    output: Path,
) -> dict[str, object]:
    policy = HeuristicGoalPolicy()
    environment = environment_from_args(
        args,
        phase=SACTrainingPhase.NAVIGATION,
        battery_capacity=battery_capacity,
    )
    estimator = ConstantQuantileEstimator(max(0.1, battery_capacity * 0.40))
    environment.bind_energy_learning(
        energy_estimator=estimator,
        goal_action_provider=single_action_provider(policy),
        training_enabled=False,
    )
    environment.enable_phase_two()
    observation, _ = environment.reset(seed=args.seed + 50_000)
    charger = environment.charger_position

    def nearby_task(reference_position: np.ndarray) -> np.ndarray:
        del reference_position
        candidate = charger + np.array([120.0, 0.0, 0.0], dtype=np.float32)
        if candidate[0] > environment.length - environment.xy_sampling_margin:
            candidate = charger - np.array([120.0, 0.0, 0.0], dtype=np.float32)
        return candidate

    environment._sample_task_point = nearby_task
    environment.current_task_point = nearby_task(environment.agent.pos)
    environment.agent.goal = environment.current_task_point.copy()
    environment._start_goal_trajectory(environment.current_task_point)
    observation = environment.sac_observation_for_goal(environment.current_task_point)
    successful_cycles = 0
    saw_task = False
    saw_commit = False
    charger_nonterminal = True
    max_steps = min(args.phase2_episode_max_steps, 20_000)
    for _ in range(max_steps):
        action, _ = policy.predict(observation, deterministic=True)
        observation, _, terminated, truncated, info = environment.step(action)
        saw_task = saw_task or bool(info["task_completed_now"])
        saw_commit = saw_commit or bool(info["switched_now"])
        if info["charger_reached_now"]:
            successful_cycles += 1
            charger_nonterminal = charger_nonterminal and not terminated
        if successful_cycles >= 2 or terminated or truncated:
            break
    result = {
        "task_observed": saw_task,
        "commitment_observed": saw_commit,
        "successful_battery_cycles": successful_cycles,
        "charger_reached_nonterminal": charger_nonterminal,
        "post_cycle_mode": environment.mode.value,
        "post_cycle_energy": float(environment.agent.energy),
        "battery_capacity": battery_capacity,
        "same_gym_episode": successful_cycles >= 2,
    }
    for record in environment.battery_cycle_records:
        append_jsonl(output / "battery_cycles.jsonl", record)
    for event in environment.switching_events:
        append_jsonl(output / "switching_events.jsonl", event)
    _plot_phase2_energy_smoke(environment, output)
    write_json(output / "phase2_state_machine_smoke.json", result)
    environment.close()
    return result


def _plot_phase2_energy_smoke(environment: UAVEnergyDeliverySACEnv, output: Path) -> None:
    rows = environment.get_trajectory_log()
    if not rows:
        return
    steps = np.asarray([row["step"] for row in rows], dtype=np.int64)
    remaining = np.asarray([row["remaining_energy"] for row in rows], dtype=np.float64)
    energy = np.asarray([row["realized_energy_cost"] for row in rows], dtype=np.float64)
    duration = np.asarray([row["transition_dt"] for row in rows], dtype=np.float64)
    power = energy / np.maximum(duration, 1e-12)
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.plot(steps, remaining)
    axis.set_xlabel("Policy transition")
    axis.set_ylabel(f"Remaining battery ({ENERGY_UNIT})")
    figure.tight_layout()
    figure.savefig(output / "battery_remaining_curve.png", dpi=160)
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.plot(steps, energy)
    axis.set_xlabel("Policy transition")
    axis.set_ylabel(f"Energy per policy step ({ENERGY_UNIT})")
    figure.tight_layout()
    figure.savefig(output / "energy_per_policy_step_curve.png", dpi=160)
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.plot(steps, power)
    axis.set_xlabel("Policy transition")
    axis.set_ylabel(f"Mean interval power ({ENERGY_UNIT}/s)")
    figure.tight_layout()
    figure.savefig(output / "energy_power_curve.png", dpi=160)
    plt.close(figure)
    cycle_records = environment.battery_cycle_records
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.bar(
        [record["battery_cycle_id"] for record in cycle_records],
        [record["total_energy_used"] for record in cycle_records],
    )
    axis.set_xlabel("Battery cycle")
    axis.set_ylabel(f"Cycle energy ({ENERGY_UNIT})")
    figure.tight_layout()
    figure.savefig(output / "battery_cycle_energy_curve.png", dpi=160)
    plt.close(figure)


def run_energy_exhaustion_smoke(args: argparse.Namespace, output: Path) -> dict[str, object]:
    policy = HeuristicGoalPolicy()
    test_capacity = 0.02
    environment = environment_from_args(
        args,
        phase=SACTrainingPhase.NAVIGATION,
        battery_capacity=test_capacity,
    )
    environment.enable_battery_validation()
    observation, _ = environment.reset(seed=args.seed + 60_000)
    result: dict[str, object] = {}
    for _ in range(100):
        action, _ = policy.predict(observation, deterministic=True)
        observation, _, terminated, truncated, info = environment.step(action)
        if terminated or truncated:
            result = {
                "terminated": terminated,
                "truncated": truncated,
                "end_reason": info["end_reason"],
                "remaining_energy": info["remaining_energy"],
                "policy_steps": environment.current_step,
                "battery_cycle_record": info["battery_cycle_record"],
            }
            if info["battery_cycle_record"] is not None:
                append_jsonl(output / "battery_cycles.jsonl", info["battery_cycle_record"])
            break
    write_json(output / "energy_exhaustion_smoke.json", result)
    environment.close()
    return result


def run_phase2_fixed_budget(
    policy: DeterministicPolicy,
    estimator: GoalConditionedQuantileTDEnergyEstimator,
    args: argparse.Namespace,
    *,
    battery_capacity: float,
    output: Path,
) -> dict[str, object]:
    environment = environment_from_args(
        args,
        phase=SACTrainingPhase.NAVIGATION,
        battery_capacity=battery_capacity,
    )
    environment.bind_energy_learning(
        energy_estimator=estimator,
        goal_action_provider=single_action_provider(policy),
        training_enabled=True,
    )
    environment.enable_phase_two()
    observation, _ = environment.reset(seed=args.seed + 200_000)
    transitions = 0
    episodes = 0
    exhaustions = 0
    emergency_truncations = 0
    completed_cycles = 0
    rows: list[dict[str, object]] = []
    while transitions < args.phase2_transition_budget:
        action, _ = policy.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = environment.step(action)
        transitions += 1
        if info["switched_now"] and environment.switching_events:
            append_jsonl(output / "switching_events.jsonl", environment.switching_events[-1])
        if info["battery_cycle_record"] is not None:
            completed_cycles += 1
            append_jsonl(output / "battery_cycles.jsonl", info["battery_cycle_record"])
        if transitions % args.log_freq_transitions == 0 or terminated or truncated:
            row = {
                "global_env_transitions": transitions,
                "phase": int(SACTrainingPhase.ENERGY_MANAGED),
                "reward": float(reward),
                "tasks_completed": info["tasks_completed"],
                "transition_dt": info["transition_dt"],
                "physics_substeps": info["physics_substeps"],
                "step_realized_energy": info["realized_energy_cost"],
                "interval_mean_power": info["interval_mean_power"],
                "remaining_energy": info["remaining_energy"],
                "remaining_energy_fraction": info["remaining_energy_fraction"],
                "battery_capacity": info["battery_capacity"],
                "battery_cycle_id": info["battery_cycle_id"],
                "tasks_in_cycle": info["tasks_in_current_battery_cycle"],
                "mode": info["mode"],
                "Q50": info["Q50"],
                "Q90": info["Q90"],
                "Q95": info["Q95"],
                "Q99": info["Q99"],
                "E_task_95": info["E_task_95"],
                "E_return_after_task_95": info["E_return_after_task_95"],
                "E_mission_95": info["E_mission_95"],
                "E_return_now_95": info["E_return_now_95"],
                "energy_reserve": info["energy_reserve"],
                "energy_reserve_fraction": info["energy_reserve_fraction"],
                "td_loss": info["td_loss"],
                "td_update_count": info["td_update_count"],
                "td_replay_size": info["td_replay_size"],
            }
            rows.append(row)
            append_jsonl(output / "phase2" / "training_curve.jsonl", row)
            append_csv(output / "phase2" / "training_curve.csv", row)
        if transitions % args.checkpoint_freq_transitions == 0:
            estimator.save(output / "phase2" / f"td_checkpoint_transition_{transitions:06d}.pt")
        if terminated or truncated:
            episodes += 1
            exhaustions += int(terminated and info["end_reason"] == "energy_exhausted")
            emergency_truncations += int(truncated)
            observation, _ = environment.reset(seed=args.seed + 200_000 + transitions)
    estimator.save(output / "phase2" / "td_energy_final.pt")
    _plot_phase2_training_rows(rows, output / "phase2")
    summary = {
        "requested_transition_budget": args.phase2_transition_budget,
        "actual_training_transitions": transitions,
        "exact_budget_match": transitions == args.phase2_transition_budget,
        "training_stop_reason": "transition_budget_reached",
        "episodes": episodes,
        "completed_battery_cycles": completed_cycles,
        "energy_exhaustions": exhaustions,
        "emergency_truncations": emergency_truncations,
        "battery_capacity": battery_capacity,
        "battery_capacity_source": "phase1_frozen_policy_calibration",
        "energy_reserve_fraction": args.energy_reserve_fraction,
        "energy_reserve_absolute": battery_capacity * args.energy_reserve_fraction,
        "sac_frozen": True,
        "td_update_count": estimator.update_count,
        "td_replay_size": len(estimator.replay),
    }
    write_json(output / "phase2" / "summary.json", summary)
    environment.close()
    return summary


def _plot_phase2_training_rows(rows: list[dict[str, object]], output: Path) -> None:
    if not rows:
        return
    transitions = [row["global_env_transitions"] for row in rows]
    plots = (
        ("battery_remaining_curve.png", "remaining_energy_fraction", "Remaining battery fraction"),
        ("energy_per_policy_step_curve.png", "step_realized_energy", f"Energy ({ENERGY_UNIT})"),
        ("energy_power_curve.png", "interval_mean_power", f"Power ({ENERGY_UNIT}/s)"),
    )
    for filename, key, label in plots:
        figure, axis = plt.subplots(figsize=(8, 4))
        axis.plot(transitions, [row[key] for row in rows])
        axis.set_xlabel("Phase 2 environment transitions")
        axis.set_ylabel(label)
        figure.tight_layout()
        figure.savefig(output / filename, dpi=160)
        plt.close(figure)


def generate_navigation_curves(output: Path) -> None:
    path = output / "training_curve.jsonl"
    if not path.exists() or not path.read_text(encoding="utf-8").strip():
        return
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    transitions = [row["global_env_transitions"] for row in rows]
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.plot(transitions, [row["rolling_mean_reward_100"] for row in rows])
    axis.set_xlabel("Training environment transitions")
    axis.set_ylabel("Rolling episode reward")
    figure.tight_layout()
    figure.savefig(output / "reward_curve.png", dpi=160)
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.plot(transitions, [row["successful_training_episodes"] for row in rows])
    axis.set_xlabel("Training environment transitions")
    axis.set_ylabel("Successful episodes")
    figure.tight_layout()
    figure.savefig(output / "tasks_curve.png", dpi=160)
    plt.close(figure)


def generate_eval_gif(
    policy: DeterministicPolicy,
    args: argparse.Namespace,
    task: NavigationTask,
    output_path: Path,
) -> int:
    environment = environment_from_args(args, phase=SACTrainingPhase.NAVIGATION, render_mode="rgb_array")
    observation, _ = environment.reset(
        seed=args.gif_seed,
        options={
            "start_position": task.start_position,
            "start_velocity": task.initial_velocity,
            "task_point": task.goal_position,
        },
    )
    frames: list[Image.Image] = []
    transitions = 0
    while True:
        action, _ = policy.predict(observation, deterministic=True)
        observation, _, terminated, truncated, _ = environment.step(action)
        transitions += 1
        if transitions == 1 or transitions % args.gif_frame_skip == 0 or terminated or truncated:
            frames.append(Image.fromarray(environment.render(view="xy")).convert("RGB"))
        if terminated or truncated:
            break
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(output_path, save_all=True, append_images=frames[1:], duration=100, loop=0)
    environment.close()
    return transitions


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UAV Energy Delivery V3 fixed-transition protocol")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--phase1-transition-budget", type=int, default=FORMAL_PHASE1_TRANSITION_BUDGET)
    parser.add_argument("--phase1b-transition-budget", type=int, default=500_000)
    parser.add_argument("--phase2-transition-budget", type=int, default=FORMAL_PHASE2_TRANSITION_BUDGET)
    parser.add_argument("--phase1-episode-max-steps", type=int, default=4000)
    parser.add_argument("--phase2-episode-max-steps", type=int, default=20_000)
    parser.add_argument("--eval-freq-transitions", type=int, default=50_000)
    parser.add_argument("--checkpoint-freq-transitions", type=int, default=50_000)
    parser.add_argument("--log-freq-transitions", type=int, default=1000)
    parser.add_argument("--gif-freq-transitions", type=int, default=100_000)
    parser.add_argument("--eval-navigation-tasks", type=int, default=500)
    parser.add_argument("--eval-task-seed", type=int, default=70_001)
    parser.add_argument("--battery-calibration-tasks", type=int, default=500)
    parser.add_argument("--battery-calibration-seed", type=int, default=80_001)
    parser.add_argument("--battery-validation-runs", type=int, default=100)
    parser.add_argument("--battery-validation-seed", type=int, default=90_001)
    parser.add_argument("--target-nominal-endurance-minutes", type=float, default=30.0)
    parser.add_argument("--energy-reserve-fraction", type=float, default=0.10)
    parser.add_argument("--allow-failed-battery-calibration", action="store_true")
    parser.add_argument("--td-collection-seed", type=int, default=100_001)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--validation", action="store_true")
    parser.add_argument("--run-td-pretraining", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-phase2", action="store_true")
    parser.add_argument("--smoke-use-heuristic-policy", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--minimum-task-distance", type=float, default=100.0)
    parser.add_argument("--xy-sampling-margin", type=float, default=100.0)
    parser.add_argument("--task-z-min", type=float, default=20.0)
    parser.add_argument("--task-z-max", type=float, default=380.0)
    parser.add_argument("--render-vertical-exaggeration", type=float, default=4.0)
    parser.add_argument("--gif-seed", type=int, default=110_001)
    parser.add_argument("--gif-frame-skip", type=int, default=20)
    parser.add_argument("--base-power", type=float, default=0.05)
    parser.add_argument("--velocity-coefficients", type=float, nargs=3, default=[0.005, 0.005, 0.005])
    parser.add_argument("--acceleration-coefficients", type=float, nargs=3, default=[0.005, 0.005, 0.005])
    parser.add_argument("--compute-power", type=float, default=0.005)
    parser.add_argument("--communication-power", type=float, default=0.005)
    parser.add_argument("--simulation-error", type=float, default=0.0)
    parser.add_argument("--flight-energy-multiplier", type=float, default=1.0)
    parser.add_argument("--energy-learning-rate", type=float, default=3e-4)
    parser.add_argument("--energy-learning-starts", type=int, default=512)
    parser.add_argument("--energy-batch-size", type=int, default=128)
    parser.add_argument("--energy-replay-capacity", type=int, default=100_000)
    parser.add_argument("--energy-target-tau", type=float, default=0.01)
    parser.add_argument("--gamma-energy", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--buffer-size", type=int, default=1_000_000)
    parser.add_argument("--learning-starts", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args(argv)
    if args.smoke and args.validation:
        parser.error("--smoke and --validation are mutually exclusive")
    if args.smoke:
        args.phase1_transition_budget = 8000
        args.phase1b_transition_budget = 2000
        args.eval_freq_transitions = 4000
        args.checkpoint_freq_transitions = 4000
        args.log_freq_transitions = 800
        args.gif_freq_transitions = 8000
        args.eval_navigation_tasks = 10
        args.battery_calibration_tasks = 20
        args.battery_validation_runs = 5
        args.learning_starts = min(args.learning_starts, 256)
        args.batch_size = min(args.batch_size, 128)
        args.energy_learning_starts = min(args.energy_learning_starts, 64)
        args.energy_batch_size = min(args.energy_batch_size, 64)
        args.target_nominal_endurance_minutes = min(args.target_nominal_endurance_minutes, 2.0)
    elif args.validation:
        args.phase1_transition_budget = 50_000
        args.phase1b_transition_budget = 5000
        args.eval_freq_transitions = 25_000
        args.checkpoint_freq_transitions = 25_000
        args.gif_freq_transitions = 50_000
        args.eval_navigation_tasks = 20
        args.battery_calibration_tasks = 20
        args.battery_validation_runs = 5
        args.target_nominal_endurance_minutes = min(args.target_nominal_endurance_minutes, 2.0)
    if args.num_envs <= 0:
        parser.error("num-envs must be positive")
    if args.phase1_transition_budget % args.num_envs != 0:
        parser.error("phase1-transition-budget must be exactly divisible by num-envs")
    for frequency_name in (
        "eval_freq_transitions",
        "checkpoint_freq_transitions",
        "log_freq_transitions",
        "gif_freq_transitions",
    ):
        if getattr(args, frequency_name) <= 0 or getattr(args, frequency_name) % args.num_envs != 0:
            parser.error(f"{frequency_name.replace('_', '-')} must be positive and divisible by num-envs")
    for name in ("eval_navigation_tasks", "battery_calibration_tasks"):
        if getattr(args, name) % 5 != 0:
            parser.error(f"{name.replace('_', '-')} must be divisible by five")
    if len({args.seed, args.eval_task_seed, args.battery_calibration_seed, args.battery_validation_seed}) != 4:
        parser.error("training, evaluation, calibration, and battery-validation seeds must be distinct")
    if args.gamma_energy != ENERGY_GAMMA:
        parser.error("gamma-energy must equal 1.0")
    if not 0.0 <= args.energy_reserve_fraction < 1.0:
        parser.error("energy-reserve-fraction must lie in [0, 1)")
    return args


def train(args: argparse.Namespace) -> dict[str, object]:
    if not args.allow_dirty and not (args.smoke or args.validation) and not git_clean():
        raise RuntimeError("formal training requires a clean worktree")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    for directory in ("phase1_navigation", "battery_calibration", "phase1_td", "phase2", "eval", "gifs"):
        (output / directory).mkdir()
    for filename in ("battery_cycles.jsonl", "switching_events.jsonl"):
        (output / filename).touch()
    contract_environment = environment_from_args(args, phase=SACTrainingPhase.NAVIGATION)
    check_env(contract_environment, warn=True, skip_render_check=True)
    contract_environment.close()
    eval_tasks = generate_stratified_navigation_tasks(
        num_tasks=args.eval_navigation_tasks,
        seed=args.eval_task_seed,
    )
    save_navigation_tasks(output / "eval_navigation_tasks.json", eval_tasks, seed=args.eval_task_seed, role="evaluation")
    calibration_tasks = generate_stratified_navigation_tasks(
        num_tasks=args.battery_calibration_tasks,
        seed=args.battery_calibration_seed,
    )
    save_navigation_tasks(
        output / "battery_calibration_tasks.json",
        calibration_tasks,
        seed=args.battery_calibration_seed,
        role="battery_calibration",
    )
    config = {
        "protocol": "UAV_ENERGY_DELIVERY_V3_FIXED_TRANSITION_WITH_BATTERY_CALIBRATION",
        "status": "RUNNING",
        "started_at": utc_now(),
        "pid": os.getpid(),
        "git_sha": git_sha(),
        "git_clean_at_start": git_clean(),
        "exact_argv": os.sys.argv,
        "phase1": {
            "training_unit": "environment_transition",
            "transition_budget": args.phase1_transition_budget,
            "num_envs": args.num_envs,
            "vector_env_steps": args.phase1_transition_budget // args.num_envs,
            "episode_semantics": "single_random_goal",
            "episode_max_policy_steps": args.phase1_episode_max_steps,
            "finish_last_episode_after_budget": False,
            "convergence_gate_role": "diagnostic_only_no_early_stop",
        },
        "battery_calibration": {
            "tasks": args.battery_calibration_tasks,
            "seed": args.battery_calibration_seed,
            "target_nominal_endurance_minutes": args.target_nominal_endurance_minutes,
            "capacity_source": (
                "technical_heuristic_realized_telemetry_power"
                if args.smoke or args.validation
                else "phase1_frozen_policy_realized_telemetry_power"
            ),
            "technical_policy_override": bool(args.smoke or args.validation),
        },
        "phase1b": {
            "transition_budget": args.phase1b_transition_budget,
            "sac_frozen": True,
            "td_replay_excludes_phase1a": True,
        },
        "phase2": {
            "transition_budget": args.phase2_transition_budget,
            "episode_emergency_guard": args.phase2_episode_max_steps,
            "continuous_delivery": True,
            "run_requested": args.run_phase2,
        },
        "environment": {
            "map_m": [4000.0, 4000.0, 400.0],
            "policy_dt_s": 0.2,
            "physics_dt_s": 0.05,
            "velocity_limits_mps": [20.0, 20.0, 5.0],
            "acceleration_limits_mps2": [5.0, 5.0, 3.0],
            "telemetry_cost_config": telemetry_config_from_args(args).__dict__,
        },
        "packages": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "gymnasium": gymnasium.__version__,
            "stable_baselines3": stable_baselines3.__version__,
        },
    }
    write_json(output / "config.json", config)
    write_json(output / "RUNNING.json", {"status": "RUNNING", "pid": os.getpid(), "started_at": config["started_at"]})
    try:
        trained_model, navigation_audit, checkpoint = train_navigation_fixed_budget(args, output, eval_tasks)
        generate_navigation_curves(output)
        del trained_model
        model = SAC.load(checkpoint, device=args.device)
        freeze_navigation_policy(model)
        navigation_audit["downstream_checkpoint_reloaded"] = True
        navigation_audit["downstream_checkpoint_sha256"] = file_sha256(checkpoint)
        navigation_audit["downstream_policy_frozen_before_calibration"] = all(
            not parameter.requires_grad for parameter in model.policy.parameters()
        )
        write_json(output / "phase1_navigation" / "summary.json", navigation_audit)
        calibration_policy: DeterministicPolicy = (
            HeuristicGoalPolicy()
            if (args.smoke or args.validation) and args.smoke_use_heuristic_policy
            else model
        )
        calibration = run_battery_calibration(
            calibration_policy,
            args,
            calibration_tasks,
            output / "battery_calibration",
        )
        capacity = float(calibration["calibrated_battery_capacity"])
        battery_validation = run_battery_validation(
            calibration_policy,
            args,
            battery_capacity=capacity,
            output=output / "battery_calibration",
        )
        if not battery_validation["battery_calibration_valid"] and not (
            args.allow_failed_battery_calibration or args.smoke or args.validation
        ):
            raise RuntimeError(
                "battery calibration sanity gate failed; pass --allow-failed-battery-calibration to override"
            )
        estimator = GoalConditionedQuantileTDEnergyEstimator(
            learning_rate=args.energy_learning_rate,
            batch_size=args.energy_batch_size,
            replay_capacity=args.energy_replay_capacity,
            learning_starts=args.energy_learning_starts,
            target_tau=args.energy_target_tau,
            gamma_energy=args.gamma_energy,
            seed=args.seed,
            device=str(model.device),
        )
        phase_boundary = freeze_navigation_and_start_td(model, estimator)
        write_json(
            output / "phase1_td" / "phase_boundary.json",
            {
                **phase_boundary,
                "source_checkpoint": str(checkpoint),
                "source_checkpoint_sha256": file_sha256(checkpoint),
                "source_transition": args.phase1_transition_budget,
                "calibrated_battery_capacity": capacity,
            },
        )
        td_summary = None
        td_policy = (
            calibration_policy
            if (args.smoke or args.validation) and args.smoke_use_heuristic_policy
            else model
        )
        estimator.set_next_action_provider(batch_action_provider(td_policy))
        if args.run_td_pretraining:
            td_summary = run_td_pretraining(
                td_policy,
                estimator,
                args,
                transition_budget=args.phase1b_transition_budget,
                output=output,
            )
        state_machine = run_phase2_state_machine_smoke(
            args,
            battery_capacity=capacity,
            output=output,
        )
        exhaustion = run_energy_exhaustion_smoke(args, output)
        phase2_summary = None
        if args.run_phase2:
            phase2_summary = run_phase2_fixed_budget(
                model,
                estimator,
                args,
                battery_capacity=capacity,
                output=output,
            )
        completed = {
            "status": "COMPLETED",
            "completed_at": utc_now(),
            "requested_transition_budget": args.phase1_transition_budget,
            "actual_training_transitions": navigation_audit["actual_training_transitions"],
            "exact_budget_match": navigation_audit["exact_budget_match"],
            "training_stop_reason": "transition_budget_reached",
            "navigation": navigation_audit,
            "battery_calibration": {
                key: value for key, value in calibration.items() if key != "tasks"
            },
            "battery_validation": {
                key: value for key, value in battery_validation.items() if key != "records"
            },
            "phase1b": td_summary,
            "phase2_state_machine_smoke": state_machine,
            "energy_exhaustion_smoke": exhaustion,
            "gif_evaluation_transitions": navigation_audit["gif_evaluation_env_transitions"],
            "formal_phase2_started": bool(args.run_phase2),
            "phase2": phase2_summary,
        }
        write_json(output / "COMPLETED.json", completed)
        (output / "RUNNING.json").unlink(missing_ok=True)
        return completed
    except Exception as error:
        write_json(
            output / "FAILED.json",
            {"status": "FAILED", "failed_at": utc_now(), "error_type": type(error).__name__, "error": str(error)},
        )
        raise


if __name__ == "__main__":
    train(parse_args())
