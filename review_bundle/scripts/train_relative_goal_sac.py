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

from envs.navigation import D_NEAR, RelativeGoalNavigationEnv
from experiments.navigation_scale import evaluate_relative_goal_policy
from experiments.new_route.provenance import append_jsonl, code_hash, git_sha, sha256_file, utc_now, write_json


def _git_is_clean() -> bool:
    return not subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=normal"], cwd=ROOT, text=True
    ).strip()


class RelativeGoalTrainingCallback(BaseCallback):
    def __init__(self, args: argparse.Namespace, output: Path) -> None:
        super().__init__(verbose=0)
        self.args = args
        self.output = output
        self.completed_goals = 0
        self.episode_count = 0
        self.near_episodes = 0
        self.far_episodes = 0
        self.boundary_contacts = 0
        self.velocity_saturations = 0
        self.far_steps = 0
        self.far_positive_progress_steps = 0
        self.far_progress_sum = 0.0
        self.action_alignment_sum = 0.0
        self.negative_progress_steps = 0
        self.overshoot_events = 0
        self.maximum_negative_progress_run = 0

    def assert_finite(self) -> None:
        for module_name in ("actor", "critic", "critic_target"):
            if any(not torch.isfinite(parameter).all() for parameter in getattr(self.model, module_name).parameters()):
                raise FloatingPointError(f"nonfinite SAC {module_name} parameters")
        for name, value in self.model.logger.name_to_value.items():
            if name.startswith("train/") and isinstance(value, (int, float, np.number)) and not np.isfinite(value):
                raise FloatingPointError(f"nonfinite SAC metric {name}={value}")

    def snapshot(self) -> dict:
        metrics = {
            name: float(value)
            for name, value in self.model.logger.name_to_value.items()
            if name.startswith("train/") and isinstance(value, (int, float, np.number))
        }
        return {
            "training_step": self.num_timesteps,
            "episodes": self.episode_count,
            "completed_goals": self.completed_goals,
            "near_episodes": self.near_episodes,
            "far_episodes": self.far_episodes,
            "near_fraction": self.near_episodes / max(self.episode_count, 1),
            "boundary_contact_rate": self.boundary_contacts / max(self.num_timesteps, 1),
            "velocity_saturation_rate": self.velocity_saturations / max(self.num_timesteps, 1),
            "mean_action_goal_alignment": self.action_alignment_sum / max(self.num_timesteps, 1),
            "far_positive_progress_step_rate": self.far_positive_progress_steps / max(self.far_steps, 1),
            "far_mean_progress_per_step": self.far_progress_sum / max(self.far_steps, 1),
            "negative_progress_step_rate": self.negative_progress_steps / max(self.num_timesteps, 1),
            "overshoot_event_rate": self.overshoot_events / max(self.num_timesteps, 1),
            "maximum_consecutive_negative_progress_steps": self.maximum_negative_progress_run,
            "sb3_train_metrics": metrics,
        }

    def _save_checkpoint(self) -> None:
        self.model.save(self.output / f"checkpoint_step_{self.num_timesteps:07d}")

    def _source_evaluation(self) -> None:
        torch_state = torch.random.get_rng_state()
        numpy_state = np.random.get_state()
        cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        try:
            evaluation = evaluate_relative_goal_policy(
                self.model,
                world_sizes=(4.0,),
                sorties_per_bin=self.args.eval_sorties_per_bin,
                max_steps=self.args.eval_max_steps,
                seed=self.args.eval_seed + self.num_timesteps,
            )
        finally:
            torch.random.set_rng_state(torch_state)
            np.random.set_state(numpy_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)
        payload = evaluation | {
            "training_step": self.num_timesteps,
            "evaluation_scope": "4X4_SOURCE_ONLY_NO_MODEL_SELECTION",
            "evaluated_at": utc_now(),
        }
        write_json(self.output / f"source_evaluation_step_{self.num_timesteps:07d}.json", payload)
        append_jsonl(
            self.output / "source_evaluation_curve.jsonl",
            {"training_step": self.num_timesteps, "overall": evaluation["overall"]},
        )

    def _on_step(self) -> bool:
        info = self.locals["infos"][0]
        rewards = np.asarray(self.locals["rewards"], dtype=np.float64)
        observations = np.asarray(self.locals["new_obs"], dtype=np.float64)
        if not np.all(np.isfinite(rewards)) or not np.all(np.isfinite(observations)):
            raise FloatingPointError("nonfinite relative-goal rollout")
        self.completed_goals += int(info["task_completed_now"])
        self.boundary_contacts += int(info["boundary_collision"])
        self.velocity_saturations += int(info["velocity_saturated"])
        self.action_alignment_sum += float(info["action_goal_alignment"])
        self.negative_progress_steps += int(info["negative_progress_step"])
        self.overshoot_events += int(info["overshoot_event"])
        self.maximum_negative_progress_run = max(
            self.maximum_negative_progress_run,
            int(info["consecutive_negative_progress_steps"]),
        )
        if info["far_state"]:
            self.far_steps += 1
            self.far_positive_progress_steps += int(float(info["goal_progress"]) > 0.0)
            self.far_progress_sum += float(info["goal_progress"])
        if bool(self.locals["dones"][0]):
            self.episode_count += 1
            self.near_episodes += int(info["distance_group"] == "NEAR")
            self.far_episodes += int(info["distance_group"] == "FAR")
        if self.num_timesteps % self.args.log_interval == 0:
            self.assert_finite()
            append_jsonl(self.output / "learning_curve.jsonl", self.snapshot())
        if self.num_timesteps % self.args.checkpoint_freq == 0:
            self.assert_finite()
            self._save_checkpoint()
        if self.num_timesteps % self.args.eval_freq == 0:
            self.assert_finite()
            self._source_evaluation()
        return True


def train(args: argparse.Namespace) -> None:
    if args.world_size != 4.0:
        raise ValueError("formal relative-goal SAC training is restricted to 4x4")
    if args.near_distance_max != D_NEAR:
        raise ValueError("formal relative-goal SAC fixes D_NEAR=2.0 m")
    if not args.allow_dirty and not _git_is_clean():
        raise RuntimeError("formal training requires a clean git working tree")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    unexpected = [path for path in output.iterdir() if path.name not in {"train.log", "train.pid", "pipeline_command.sh"}]
    if unexpected:
        raise FileExistsError(f"refusing to overwrite training directory: {unexpected}")
    environment = RelativeGoalNavigationEnv(
        world_size_xy=args.world_size,
        max_episode_steps=args.max_episode_steps,
        near_probability=args.near_probability,
    )
    check_env(environment, warn=True, skip_render_check=True)
    environment.reset(seed=args.seed)
    monitored = Monitor(environment)
    torch.set_num_threads(args.torch_threads)
    config = vars(args) | {
        "protocol": "4X4_RELATIVE_GOAL_SAC",
        "status": "RUNNING",
        "pid": os.getpid(),
        "started_at": utc_now(),
        "exact_command": shlex.join([sys.executable, *sys.argv]),
        "git_commit_sha": git_sha(ROOT),
        "git_clean_at_start": _git_is_clean(),
        "code_hash": code_hash(ROOT),
        "python": platform.python_version(),
        "package_versions": {
            "stable_baselines3": importlib.metadata.version("stable-baselines3"),
            "torch": torch.__version__,
            "gymnasium": gymnasium.__version__,
            "numpy": np.__version__,
            "tensorboard": importlib.metadata.version("tensorboard"),
        },
        "observation_definition": environment.observation_definition,
        "environment_definition": {
            "training_world_size": [4.0, 4.0, 2.0],
            "world_size_randomization": False,
            "obstacles": False,
            "finite_battery_termination": False,
            "collision_module": False,
            "energy_module": False,
            "near_interval_m": list(environment.effective_distance_intervals()["NEAR"]),
            "far_interval_m": list(environment.effective_distance_intervals()["FAR"]),
            "near_probability": args.near_probability,
            "v_max": environment.config.v_max.tolist(),
            "a_max": environment.config.a_max.tolist(),
            "dt": environment.config.dt,
            "body_radius": environment.config.body_radius,
            "goal_radius": environment.goal_radius,
        },
        "reward_definition": {
            "physical_progress_weight": environment.reward_config.distance_potential_scale,
            "velocity_toward_goal_weight": environment.reward_config.velocity_toward_goal_weight,
            "time_cost": environment.reward_config.time_cost,
            "completion_reward": environment.reward_config.task_completion_reward,
            "boundary_penalty": environment.reward_config.collision_penalty,
            "energy_cost_weight": 0.0,
        },
        "periodic_evaluation_scope": "4X4_SOURCE_ONLY",
        "target_scale_model_selection_forbidden": True,
        "post_training_zero_shot_status": "NOT_STARTED",
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
    callback = RelativeGoalTrainingCallback(args, output)
    try:
        model.learn(total_timesteps=args.total_timesteps, callback=callback, log_interval=1, progress_bar=False)
        callback.assert_finite()
        final_checkpoint = output / f"checkpoint_step_{args.total_timesteps:07d}_final"
        model.save(final_checkpoint)
        summary = callback.snapshot() | {
            "status": "COMPLETED",
            "completed_at": utc_now(),
            "final_checkpoint": str(final_checkpoint.with_suffix(".zip")),
            "final_checkpoint_sha256": sha256_file(final_checkpoint.with_suffix(".zip")),
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
    parser = argparse.ArgumentParser(description="Train 4x4-only LiDAR-free relative-goal SAC")
    parser.add_argument("--total-timesteps", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--world-size", type=float, default=4.0)
    parser.add_argument("--near-distance-max", type=float, default=2.0)
    parser.add_argument("--near-probability", type=float, default=0.5)
    parser.add_argument("--max-episode-steps", type=int, default=800)
    parser.add_argument("--eval-freq", type=int, default=100_000)
    parser.add_argument("--checkpoint-freq", type=int, default=50_000)
    parser.add_argument("--eval-sorties-per-bin", type=int, default=5)
    parser.add_argument("--eval-max-steps", type=int, default=800)
    parser.add_argument("--eval-seed", type=int, default=80_000_000)
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
    if args.eval_freq > args.total_timesteps:
        args.eval_freq = args.total_timesteps
    if args.checkpoint_freq > args.total_timesteps:
        args.checkpoint_freq = args.total_timesteps
    return args


if __name__ == "__main__":
    train(parse_args())
