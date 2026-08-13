#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from stable_baselines3 import SAC

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.navigation import NavigationEnv


@dataclass
class EvaluationTrace:
    model_seed: int
    evaluation_seed: int
    positions: list[np.ndarray]
    goals: list[np.ndarray]
    collisions: list[bool]
    tasks_completed: list[int]
    cumulative_return: list[float]
    final_metrics: dict[str, Any]


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def evaluate_checkpoint(
    checkpoint: Path,
    model_seed: int,
    evaluation_seed: int,
    steps: int,
    *,
    record_trace: bool,
) -> EvaluationTrace:
    model = SAC.load(checkpoint, device="cpu")
    environment = NavigationEnv(max_episode_steps=steps)
    observation, _ = environment.reset(seed=evaluation_seed)
    positions = [environment.state.position.copy()]
    goals = [environment.goal.copy()]
    collisions = [False]
    tasks_completed = [0]
    cumulative_return = [0.0]
    total_return = 0.0
    goal_progress = 0.0
    velocity_toward_goal = 0.0
    action_abs_sum = np.zeros(3, dtype=np.float64)
    action_squared_sum = np.zeros(3, dtype=np.float64)
    final_info: dict[str, Any] | None = None

    for _ in range(steps):
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = environment.step(action)
        if terminated:
            raise RuntimeError("persistent navigation evaluation terminated unexpectedly")
        if not np.isfinite(reward) or not np.all(np.isfinite(action)) or not np.all(np.isfinite(observation)):
            raise FloatingPointError("nonfinite value during trained-model evaluation")
        total_return += float(reward)
        goal_progress += float(info["goal_progress"])
        velocity_toward_goal += float(info["velocity_toward_goal"])
        action_array = np.asarray(action, dtype=np.float64)
        action_abs_sum += np.abs(action_array)
        action_squared_sum += action_array * action_array
        final_info = info
        if record_trace:
            positions.append(environment.state.position.copy())
            goals.append(environment.goal.copy())
            collisions.append(bool(info["collision"]))
            tasks_completed.append(int(info["tasks_completed"]))
            cumulative_return.append(total_return)
        if truncated:
            break

    if final_info is None:
        raise RuntimeError("evaluation produced no transition")
    actual_steps = environment.episode_step
    final_metrics = {
        "model_seed": model_seed,
        "evaluation_seed": evaluation_seed,
        "steps": actual_steps,
        "tasks_completed": environment.tasks_completed,
        "tasks_per_1000_steps": 1000.0 * environment.tasks_completed / max(1, actual_steps),
        "goal_success": environment.tasks_completed > 0,
        "episode_return": total_return,
        "mean_goal_progress": goal_progress / max(1, actual_steps),
        "minimum_goal_distance": environment.minimum_goal_distance,
        "mean_goal_distance": environment.goal_distance_sum / max(1, environment.goal_distance_samples),
        "mean_velocity_toward_goal": velocity_toward_goal / max(1, actual_steps),
        "collision_count": environment.collision_count,
        "collision_rate": environment.collision_count / max(1, actual_steps),
        "energy_usage": environment.cumulative_energy_usage,
        "mean_abs_action": action_abs_sum / max(1, actual_steps),
        "rms_action": np.sqrt(action_squared_sum / max(1, actual_steps)),
        "finite_values": True,
        "unexpected_termination": False,
    }
    environment.close()
    return EvaluationTrace(
        model_seed,
        evaluation_seed,
        positions,
        goals,
        collisions,
        tasks_completed,
        cumulative_return,
        final_metrics,
    )


def aggregate_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total_steps = sum(int(row["steps"]) for row in rows)
    total_tasks = sum(int(row["tasks_completed"]) for row in rows)
    total_collisions = sum(int(row["collision_count"]) for row in rows)
    return {
        "evaluation_streams": len(rows),
        "total_steps": total_steps,
        "tasks_completed": total_tasks,
        "tasks_per_1000_steps": 1000.0 * total_tasks / max(1, total_steps),
        "stream_success_rate": float(np.mean([row["goal_success"] for row in rows])),
        "mean_episode_return": float(np.mean([row["episode_return"] for row in rows])),
        "mean_goal_progress": float(np.mean([row["mean_goal_progress"] for row in rows])),
        "mean_minimum_goal_distance": float(np.mean([row["minimum_goal_distance"] for row in rows])),
        "mean_goal_distance": float(np.mean([row["mean_goal_distance"] for row in rows])),
        "mean_velocity_toward_goal": float(np.mean([row["mean_velocity_toward_goal"] for row in rows])),
        "collision_count": total_collisions,
        "collision_rate": total_collisions / max(1, total_steps),
        "mean_energy_usage": float(np.mean([row["energy_usage"] for row in rows])),
        "mean_abs_action": np.mean([row["mean_abs_action"] for row in rows], axis=0),
        "rms_action": np.mean([row["rms_action"] for row in rows], axis=0),
        "all_finite": all(bool(row["finite_values"]) for row in rows),
        "unexpected_terminations": sum(bool(row["unexpected_termination"]) for row in rows),
    }


def _world_to_panel(position: np.ndarray, panel_left: int, panel_top: int, plot_size: int) -> tuple[int, int]:
    x = panel_left + int(round(float(position[0]) / 4.0 * plot_size))
    y = panel_top + plot_size - int(round(float(position[1]) / 4.0 * plot_size))
    return x, y


def render_comparison_gif(
    traces: list[EvaluationTrace],
    output_path: Path,
    preview_path: Path,
    frame_stride: int,
    frame_duration_ms: int,
) -> None:
    panel_width = 360
    panel_height = 430
    plot_size = 320
    plot_left_margin = 20
    plot_top = 68
    width = panel_width * len(traces)
    height = panel_height
    font = ImageFont.load_default()
    frame_indices = list(range(0, len(traces[0].positions), frame_stride))
    if frame_indices[-1] != len(traces[0].positions) - 1:
        frame_indices.append(len(traces[0].positions) - 1)
    frames: list[Image.Image] = []

    for frame_index in frame_indices:
        image = Image.new("RGB", (width, height), (245, 247, 250))
        draw = ImageDraw.Draw(image)
        for panel_index, trace in enumerate(traces):
            panel_left = panel_index * panel_width
            plot_left = panel_left + plot_left_margin
            draw.rectangle(
                (plot_left, plot_top, plot_left + plot_size, plot_top + plot_size),
                fill=(255, 255, 255),
                outline=(40, 46, 54),
                width=2,
            )
            station = _world_to_panel(np.array([0.4, 0.5, 1.0]), plot_left, plot_top, plot_size)
            draw.ellipse((station[0] - 6, station[1] - 6, station[0] + 6, station[1] + 6), fill=(43, 108, 176))
            draw.text((station[0] + 8, station[1] - 6), "station", fill=(43, 108, 176), font=font)

            trail_start = max(0, frame_index - 400)
            trail = [
                _world_to_panel(position, plot_left, plot_top, plot_size)
                for position in trace.positions[trail_start : frame_index + 1]
            ]
            if len(trail) >= 2:
                draw.line(trail, fill=(89, 134, 188), width=2)
            goal = _world_to_panel(trace.goals[frame_index], plot_left, plot_top, plot_size)
            draw.ellipse((goal[0] - 7, goal[1] - 7, goal[0] + 7, goal[1] + 7), outline=(224, 120, 48), width=3)
            draw.line((goal[0] - 10, goal[1], goal[0] + 10, goal[1]), fill=(224, 120, 48), width=2)
            draw.line((goal[0], goal[1] - 10, goal[0], goal[1] + 10), fill=(224, 120, 48), width=2)
            uav = _world_to_panel(trace.positions[frame_index], plot_left, plot_top, plot_size)
            color = (196, 55, 55) if trace.collisions[frame_index] else (36, 150, 90)
            draw.ellipse((uav[0] - 6, uav[1] - 6, uav[0] + 6, uav[1] + 6), fill=color, outline=(20, 20, 20))

            collision_count = sum(trace.collisions[: frame_index + 1])
            draw.text((panel_left + 14, 10), f"SAC training seed {trace.model_seed}", fill=(20, 25, 32), font=font)
            draw.text(
                (panel_left + 14, 28),
                f"held-out seed {trace.evaluation_seed} | step {frame_index:4d}",
                fill=(20, 25, 32),
                font=font,
            )
            draw.text(
                (panel_left + 14, 44),
                f"tasks {trace.tasks_completed[frame_index]} | collisions {collision_count} | return {trace.cumulative_return[frame_index]:.1f}",
                fill=(20, 25, 32),
                font=font,
            )
            draw.text(
                (panel_left + 20, 397),
                f"z={trace.positions[frame_index][2]:.2f}, goal_z={trace.goals[frame_index][2]:.2f}",
                fill=(70, 76, 85),
                font=font,
            )
        frames.append(image.quantize(colors=128, method=Image.Quantize.MEDIANCUT))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=frame_duration_ms,
        loop=0,
        optimize=False,
        disposal=2,
    )
    frames[-1].convert("RGB").save(preview_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate final SB3 SAC checkpoints and render a trajectory GIF")
    parser.add_argument("--artifact-root", type=Path, default=ROOT / "artifacts" / "phase1_sb3_sac_200k")
    parser.add_argument("--model-seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--evaluation-seeds", type=int, nargs="+", default=[100, 101, 102, 103, 104])
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--gif-seed", type=int, default=100)
    parser.add_argument("--frame-stride", type=int, default=20)
    parser.add_argument("--frame-duration-ms", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "phase1_sb3_sac_200k" / "model_test")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.gif_seed not in args.evaluation_seeds:
        raise ValueError("gif-seed must be one of evaluation-seeds")
    rows: list[dict[str, Any]] = []
    gif_traces: list[EvaluationTrace] = []
    checkpoints: dict[int, str] = {}
    for model_seed in args.model_seeds:
        checkpoint = args.artifact_root / f"seed{model_seed}" / "checkpoint_step_200000.zip"
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        checkpoints[model_seed] = str(checkpoint)
        for evaluation_seed in args.evaluation_seeds:
            trace = evaluate_checkpoint(
                checkpoint,
                model_seed,
                evaluation_seed,
                args.steps,
                record_trace=evaluation_seed == args.gif_seed,
            )
            rows.append(trace.final_metrics)
            if evaluation_seed == args.gif_seed:
                gif_traces.append(trace)

    per_model = {
        str(model_seed): aggregate_results([row for row in rows if row["model_seed"] == model_seed])
        for model_seed in args.model_seeds
    }
    payload = {
        "protocol": {
            "checkpoints": checkpoints,
            "deterministic_actions": True,
            "evaluation_seeds": args.evaluation_seeds,
            "steps_per_stream": args.steps,
            "gif_seed": args.gif_seed,
        },
        "per_stream": rows,
        "per_model": per_model,
        "overall": aggregate_results(rows),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "test_results.json"
    result_path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    gif_path = args.output_dir / f"heldout_seed{args.gif_seed}_comparison.gif"
    preview_path = args.output_dir / f"heldout_seed{args.gif_seed}_final_frame.png"
    render_comparison_gif(gif_traces, gif_path, preview_path, args.frame_stride, args.frame_duration_ms)
    print(json.dumps(_jsonable(payload["per_model"]), indent=2, sort_keys=True))
    print(f"results={result_path}")
    print(f"gif={gif_path}")
    print(f"preview={preview_path}")


if __name__ == "__main__":
    main()
