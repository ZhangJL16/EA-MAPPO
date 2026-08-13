#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import traceback

import gymnasium
import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.navigation import GOAL_DISTANCE_BINS, ScaleInvariantNavigationEnv
from experiments.navigation_scale import DEFAULT_EVALUATION_SCALES, evaluate_policy
from experiments.new_route.provenance import append_jsonl, code_hash, git_sha, sha256_file, utc_now, write_json


def _git_is_clean() -> bool:
    return not subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=normal"], cwd=ROOT, text=True
    ).strip()


def _environment(args: argparse.Namespace) -> ScaleInvariantNavigationEnv:
    return ScaleInvariantNavigationEnv(
        world_size_min=args.world_size_min,
        world_size_max=args.world_size_max,
        max_episode_steps=args.max_episode_steps,
    )


class ScaleTrainingCallback(BaseCallback):
    def __init__(self, args: argparse.Namespace, output: Path) -> None:
        super().__init__(verbose=0)
        self.args = args
        self.output = output
        self.completed_goals = 0
        self.boundary_contacts = 0
        self.episode_count = 0
        self.distance_bin_counts: dict[str, int] = {}
        self.world_widths: list[float] = []

    def assert_finite(self) -> None:
        for module_name in ("actor", "critic", "critic_target"):
            module = getattr(self.model, module_name)
            if any(not torch.isfinite(parameter).all() for parameter in module.parameters()):
                raise FloatingPointError(f"nonfinite SAC {module_name} parameters")
        for name, value in self.model.logger.name_to_value.items():
            if name.startswith("train/") and isinstance(value, (int, float, np.number)) and not np.isfinite(value):
                raise FloatingPointError(f"nonfinite SAC metric {name}={value}")

    def snapshot(self) -> dict:
        train_metrics = {
            name: float(value)
            for name, value in self.model.logger.name_to_value.items()
            if name.startswith("train/") and isinstance(value, (int, float, np.number))
        }
        return {
            "training_step": self.num_timesteps,
            "completed_goals": self.completed_goals,
            "episodes": self.episode_count,
            "boundary_contacts": self.boundary_contacts,
            "distance_bin_counts": dict(self.distance_bin_counts),
            "world_width_min_observed": min(self.world_widths) if self.world_widths else None,
            "world_width_max_observed": max(self.world_widths) if self.world_widths else None,
            "sb3_train_metrics": train_metrics,
        }

    def _save_checkpoint(self) -> Path:
        path = self.output / f"checkpoint_step_{self.num_timesteps:07d}"
        self.model.save(path)
        return path.with_suffix(".zip")

    def _evaluate(self) -> None:
        torch_state = torch.random.get_rng_state()
        numpy_state = np.random.get_state()
        cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        try:
            results = evaluate_policy(
                self.model,
                scales=DEFAULT_EVALUATION_SCALES,
                sorties_per_distance_bin=self.args.eval_sorties_per_distance_bin,
                max_steps=self.args.eval_max_steps,
                seed=self.args.eval_seed + self.num_timesteps,
            )
        finally:
            torch.random.set_rng_state(torch_state)
            np.random.set_state(numpy_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)
        payload = results | {"training_step": self.num_timesteps, "evaluated_at": utc_now()}
        write_json(self.output / f"evaluation_step_{self.num_timesteps:07d}.json", payload)
        append_jsonl(
            self.output / "evaluation_curve.jsonl",
            {"training_step": self.num_timesteps, "overall": results["overall"], "by_scale": results["by_scale"]},
        )

    def _on_step(self) -> bool:
        info = self.locals["infos"][0]
        rewards = np.asarray(self.locals["rewards"], dtype=np.float64)
        observations = np.asarray(self.locals["new_obs"], dtype=np.float64)
        if not np.all(np.isfinite(rewards)) or not np.all(np.isfinite(observations)):
            raise FloatingPointError("nonfinite rollout reward or observation")
        self.completed_goals += int(info["task_completed_now"])
        self.boundary_contacts += int(info["boundary_collision"])
        if bool(self.locals["dones"][0]):
            self.episode_count += 1
            label = str(info["distance_bin"])
            self.distance_bin_counts[label] = self.distance_bin_counts.get(label, 0) + 1
            self.world_widths.append(float(np.asarray(info["world_size"])[0]))
        if self.num_timesteps % self.args.log_interval == 0:
            self.assert_finite()
            append_jsonl(self.output / "learning_curve.jsonl", self.snapshot())
        if self.num_timesteps % self.args.checkpoint_freq == 0:
            self.assert_finite()
            self._save_checkpoint()
        if self.num_timesteps % self.args.eval_freq == 0:
            self.assert_finite()
            self._evaluate()
        return True


def train(args: argparse.Namespace) -> None:
    if not args.allow_dirty and not _git_is_clean():
        raise RuntimeError("formal scale-invariant SAC training requires a clean git working tree")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    unexpected = [path for path in output.iterdir() if path.name not in {"train.log", "train.pid"}]
    if unexpected:
        raise FileExistsError(f"refusing to overwrite nonempty training directory: {unexpected}")
    environment = _environment(args)
    check_env(environment, warn=True, skip_render_check=True)
    environment.reset(seed=args.seed)
    monitored = Monitor(environment)
    torch.set_num_threads(args.torch_threads)
    sb3_version = importlib.metadata.version("stable-baselines3")
    config = vars(args) | {
        "experiment": "SCALE_INVARIANT_GOAL_CONDITIONED_SAC",
        "status": "RUNNING",
        "pid": os.getpid(),
        "started_at": utc_now(),
        "exact_command": shlex.join([sys.executable, *sys.argv]),
        "git_commit_sha": git_sha(ROOT),
        "git_clean_at_start": _git_is_clean(),
        "code_hash": code_hash(ROOT),
        "python": platform.python_version(),
        "package_versions": {
            "stable_baselines3": sb3_version,
            "torch": torch.__version__,
            "gymnasium": gymnasium.__version__,
            "numpy": np.__version__,
            "tensorboard": importlib.metadata.version("tensorboard"),
        },
        "observation_definition": environment.observation_definition,
        "observation_dimension": int(environment.observation_space.shape[0]),
        "action_dimension": int(environment.action_space.shape[0]),
        "environment_definition": {
            "world_size_xy_distribution": f"Uniform({args.world_size_min}, {args.world_size_max}) per episode",
            "world_height_m": environment.world_height,
            "obstacles": False,
            "v_max": environment.config.v_max.tolist(),
            "a_max": environment.config.a_max.tolist(),
            "dt": environment.config.dt,
            "lidar_range_m": environment.config.lidar_range,
            "body_radius_m": environment.config.body_radius,
            "goal_radius_m": environment.goal_radius,
            "distance_bins_m": [
                {"label": label, "lower": lower, "upper": None if not np.isfinite(upper) else upper}
                for label, lower, upper in GOAL_DISTANCE_BINS
            ],
            "initial_velocity_fraction": environment.initial_velocity_fraction,
            "finite_battery_termination": False,
            "station_semantics": False,
        },
        "reward_definition": {
            "physical_progress": "distance_before_m - distance_after_m",
            "progress_scale": environment.reward_config.distance_potential_scale,
            "velocity_toward_goal_weight": environment.reward_config.velocity_toward_goal_weight,
            "time_cost": environment.reward_config.time_cost,
            "goal_completion_reward": environment.reward_config.task_completion_reward,
            "boundary_collision_penalty": environment.reward_config.collision_penalty,
            "energy_cost_weight": 0.0,
            "control_penalty": 0.0,
        },
        "evaluation_status": "EVALUATION_NOT_RUN_TRAINING_IN_PROGRESS",
    }
    write_json(output / "config.json", config)
    write_json(output / "RUNNING.json", {"status": "RUNNING", "pid": os.getpid(), "started_at": config["started_at"]})
    model = SAC(
        "MlpPolicy",
        monitored,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        tau=args.tau,
        gamma=args.gamma,
        train_freq=(1, "step"),
        gradient_steps=1,
        ent_coef="auto",
        target_entropy="auto",
        tensorboard_log=str(output / "tensorboard"),
        seed=args.seed,
        device=args.device,
        verbose=1,
    )
    config |= {
        "model_device": str(model.device),
        "actor_parameter_devices": sorted({str(parameter.device) for parameter in model.actor.parameters()}),
        "critic_parameter_devices": sorted({str(parameter.device) for parameter in model.critic.parameters()}),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    write_json(output / "config.json", config)
    print("DEVICE_CHECK " + json.dumps({key: config[key] for key in ("model_device", "actor_parameter_devices", "critic_parameter_devices")}), flush=True)
    model.set_logger(configure(str(output / "sb3_logger"), ["stdout", "csv", "json", "tensorboard"]))
    callback = ScaleTrainingCallback(args, output)
    try:
        model.learn(total_timesteps=args.total_timesteps, callback=callback, log_interval=1, progress_bar=False)
        callback.assert_finite()
        final_path = output / f"checkpoint_step_{args.total_timesteps:07d}_final"
        model.save(final_path)
        summary = callback.snapshot() | {
            "status": "COMPLETED",
            "completed_at": utc_now(),
            "final_checkpoint": str(final_path.with_suffix(".zip")),
            "final_checkpoint_sha256": sha256_file(final_path.with_suffix(".zip")),
        }
        write_json(output / "summary.json", summary)
        write_json(output / "COMPLETED.json", summary)
        write_json(output / "RUNNING.json", {"status": "COMPLETED", "pid": os.getpid(), "completed_at": summary["completed_at"]})
    except Exception as error:
        failure = {
            "status": "IMPLEMENTATION_FAILURE",
            "failed_at": utc_now(),
            "training_step": model.num_timesteps,
            "error": repr(error),
            "traceback": traceback.format_exc(),
        }
        write_json(output / "FAILED.json", failure)
        write_json(output / "RUNNING.json", failure)
        raise
    finally:
        monitored.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train scale-invariant goal-conditioned SB3 SAC")
    parser.add_argument("--total-timesteps", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--eval-freq", type=int, default=100_000)
    parser.add_argument("--checkpoint-freq", type=int, default=50_000)
    parser.add_argument("--world-size-min", type=float, default=4.0)
    parser.add_argument("--world-size-max", type=float, default=16.0)
    parser.add_argument("--max-episode-steps", type=int, default=800)
    parser.add_argument("--eval-sorties-per-distance-bin", type=int, default=2)
    parser.add_argument("--eval-max-steps", type=int, default=800)
    parser.add_argument("--eval-seed", type=int, default=90_000_000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--buffer-size", type=int, default=1_000_000)
    parser.add_argument("--learning-starts", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--log-interval", type=int, default=1_000)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--allow-dirty", action="store_true", help="smoke-only override")
    args = parser.parse_args()
    if args.total_timesteps <= 0 or not 0 <= args.learning_starts < args.total_timesteps:
        parser.error("training and learning-start budgets are inconsistent")
    if args.eval_freq <= 0 or args.checkpoint_freq <= 0 or args.log_interval <= 0:
        parser.error("logging, evaluation, and checkpoint frequencies must be positive")
    if args.eval_freq > args.total_timesteps:
        args.eval_freq = args.total_timesteps
    if args.checkpoint_freq > args.total_timesteps:
        args.checkpoint_freq = args.total_timesteps
    return args


if __name__ == "__main__":
    train(parse_args())
