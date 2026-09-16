#!/usr/bin/env python3
"""Replay one registered PSPS-v1 R branch and render its full trajectory GIF."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.directional_navigation.conditional_model_access import ConditionalModelAccessBranch
from review_bundle.safety.switching.commitment import SortieMode
from scripts.run_conditional_history_collection import atomic_json
from scripts.run_dvoi_h_collection import FrozenNavigator
from scripts.validate_identification_contract import file_hash, object_hash

WORLD_IDENTITY = "sha256:a51b837e7ee2cf234f3c499eefd284cd4aa225bf315d70660d2f3644acd6407c"
WORLD_SEED = 1_137_000_014
ANCHOR_STEP = 768
REPLICATE = 27


def state_row(env: ConditionalModelAccessBranch, event: str = "") -> dict[str, Any]:
    base = env.base
    return {
        "step": int(env.total_steps),
        "position": np.asarray(base.agent.pos, np.float32).copy(),
        "goal": np.asarray(base.active_goal, np.float32).copy(),
        "energy": float(base.agent.energy),
        "mode": base.mode.value,
        "tasks_completed": int(base.tasks_completed),
        "collision_count": int(env.collision_count),
        "distance_to_charger": float(np.linalg.norm(base.agent.pos - base.charger_position)),
        "event": event,
    }


def select_expected(record: dict[str, Any]) -> dict[str, Any]:
    if record.get("world_identity") != WORLD_IDENTITY or record.get("world_seed") != WORLD_SEED:
        raise ValueError("selected record identity/seed mismatch")
    anchor = next(
        row for row in record["anchors"]
        if row.get("included") and row.get("anchor_step") == ANCHOR_STEP
    )
    return next(
        row for row in anchor["resamples"]
        if row["action"] == "R" and row["replicate"] == REPLICATE
    )


def replay(record_path: Path, contract_path: Path, device: str) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    contract = json.loads(contract_path.read_text())
    accessed = {
        event.get("world_identity") for event in contract["access_registry"]["events"]
        if event.get("event_type") == "ACCESS" and event.get("split") == "PSPS_DEV"
    }
    if WORLD_IDENTITY not in accessed:
        raise ValueError("visualization world was not already accessed in PSPS_DEV")
    record = json.loads(record_path.read_text())
    expected = select_expected(record)
    navigator = FrozenNavigator(device)
    env = ConditionalModelAccessBranch(obstacles=48, guard=12_000)
    trace: list[dict[str, Any]] = []
    try:
        observation = env.reset_world(WORLD_SEED, soc=1.0)
        world = env.physical_world_record()
        if object_hash(world) != WORLD_IDENTITY:
            raise RuntimeError("replay physical world differs from registered identity")
        trace.append(state_row(env, "reset_at_charger"))
        previous_tasks = int(env.base.tasks_completed)
        for _ in range(ANCHOR_STEP):
            action = navigator.goal_action(observation, returning=False)
            observation, transition, done = env.step_policy(action)
            event = ""
            if int(env.base.tasks_completed) > previous_tasks:
                event = "task_completed"
                previous_tasks = int(env.base.tasks_completed)
            if transition["contact"]:
                event = "collision_repair" if not event else event + "+collision_repair"
            trace.append(state_row(env, event))
            if done:
                raise RuntimeError(f"prefix terminated before anchor: {transition['termination']}")

        env.arm_resample(int(expected["disturbance_seed"]))
        env.begin_branch("R")
        trace[-1]["event"] = "return_commit_mid_task"
        terminal: dict[str, Any] | None = None
        branch_steps = 0
        for _ in range(env.guard + 1):
            returning = env.base.mode is SortieMode.CHARGER_COMMITTED
            action = navigator.goal_action(observation, returning=returning)
            observation, terminal, done = env.step_policy(action)
            branch_steps += 1
            event = "collision_repair" if terminal["contact"] else ""
            if done:
                event = terminal["termination"]
            trace.append(state_row(env, event))
            if done:
                break
        if terminal is None or not done:
            raise RuntimeError("replay branch did not terminate")
        observed = env.branch_outcome(terminal)
        # PSPS-v1's user-locked statistic is collision_count == 0, independent
        # of return status. This successful replay agrees with both definitions.
        observed["collision_free_arrival"] = observed["collision_count"] == 0
        critical = ("termination", "returned", "collision_count", "task_increment", "operational_failure", "utility")
        critical_match = branch_steps == expected["policy_steps"] and all(
            observed[key] == expected["outcome"][key] for key in critical
        )
        numeric_match = all(
            np.isclose(observed[key], expected["outcome"][key], rtol=0.0, atol=1e-9)
            for key in ("initial_energy", "remaining_energy")
        )
        if not critical_match or not numeric_match:
            raise RuntimeError(
                "replay did not reproduce the registered branch: "
                + json.dumps({"branch_steps": branch_steps, "observed": observed, "expected": expected}, default=str)
            )
        cycle = env.base.last_battery_cycle_record
        summary = {
            "schema_version": "psps-v1-best-world-gif-v1",
            "role": "POST_HOC_VISUALIZATION_ONLY_NOT_GATE_EVIDENCE",
            "world_identity": WORLD_IDENTITY,
            "world_seed": WORLD_SEED,
            "selection_rule": "MAX_MEAN_FROZEN_UTILITY_THEN_R_RETURN_RATE_THEN_C_RETURN_RATE_THEN_MIN_MEAN_R_STEPS",
            "selection_population": "COMPLETE_ALREADY_ACCESSED_PSPS_DEV_RECORDS_AT_SELECTION_TIME",
            "anchor_step": ANCHOR_STEP,
            "action": "R",
            "replicate": REPLICATE,
            "disturbance_seed": int(expected["disturbance_seed"]),
            "prefix_policy_steps": ANCHOR_STEP,
            "branch_policy_steps": branch_steps,
            "total_visualized_policy_steps": int(env.total_steps),
            "tasks_completed_before_return_commit": int(trace[ANCHOR_STEP]["tasks_completed"]),
            "tasks_completed_at_end": int(env.base.tasks_completed),
            "return_commit_was_mid_task": True,
            "termination": observed["termination"],
            "returned": observed["returned"],
            "collision_count": observed["collision_count"],
            "initial_energy": float(trace[0]["energy"]),
            "energy_at_return_commit": float(trace[ANCHOR_STEP]["energy"]),
            "arrival_energy_before_recharge": None if cycle is None else float(cycle["remaining_energy_at_cycle_end"]),
            "final_post_service_energy": float(trace[-1]["energy"]),
            "registered_outcome_reproduced": True,
            "generic_history_encoder_used": False,
            "psps_selector_used": False,
            "confirm_accessed": False,
        }
        return trace, world, summary
    finally:
        env.close()


def render(trace: list[dict[str, Any]], world: dict[str, Any], summary: dict[str, Any], output: Path) -> None:
    positions = np.stack([row["position"] for row in trace])
    goals = np.stack([row["goal"] for row in trace])
    energy = np.asarray([row["energy"] for row in trace])
    distance = np.asarray([row["distance_to_charger"] for row in trace])
    tasks = np.asarray([row["tasks_completed"] for row in trace])
    modes = np.asarray([row["mode"] for row in trace])
    events = [index for index, row in enumerate(trace) if row["event"]]
    sampled = set(np.linspace(0, len(trace) - 1, min(180, len(trace)), dtype=int).tolist())
    sampled.update(events)
    frame_indices = sorted(sampled)
    commit_index = summary["anchor_step"]
    workspace = world["workspace"]
    charger = np.asarray(world["charger_position"])

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.15,
    })
    fig = plt.figure(figsize=(10.0, 5.6), dpi=110)
    grid = fig.add_gridspec(2, 2, width_ratios=(1.3, 1.0), hspace=0.35, wspace=0.28)
    map_ax = fig.add_subplot(grid[:, 0])
    battery_ax = fig.add_subplot(grid[0, 1])
    distance_ax = fig.add_subplot(grid[1, 1])

    for obstacle in world["static_obstacles"]:
        map_ax.add_patch(Circle(obstacle["position"], obstacle["radius"], facecolor="#9E9E9E", edgecolor="#555555", alpha=0.42, linewidth=0.6))
    map_ax.plot(positions[:, 0], positions[:, 1], color="#C7C7C7", linewidth=1.0, label="full trajectory")
    map_ax.scatter(charger[0], charger[1], marker="P", s=150, color="#0072B2", edgecolor="white", linewidth=0.8, zorder=5, label="charger")
    map_ax.set_xlim(0, workspace["length"])
    map_ax.set_ylim(0, workspace["width"])
    map_ax.set_aspect("equal", adjustable="box")
    map_ax.set_xlabel("x position")
    map_ax.set_ylabel("y position")
    map_ax.set_title("Best observed DEV world: task flight and mid-task return")
    task_line, = map_ax.plot([], [], color="#009E73", linewidth=2.2, label="task mode")
    return_line, = map_ax.plot([], [], color="#E69F00", linewidth=2.5, label="return mode")
    uav = map_ax.scatter([], [], s=75, color="#D55E00", edgecolor="black", linewidth=0.7, zorder=7, label="UAV")
    goal = map_ax.scatter([], [], marker="X", s=95, color="#009E73", edgecolor="black", linewidth=0.6, zorder=6, label="active goal")
    info_text = map_ax.text(0.02, 0.98, "", transform=map_ax.transAxes, va="top", ha="left", bbox={"facecolor": "white", "alpha": 0.86, "edgecolor": "none", "boxstyle": "round,pad=0.35"})
    map_ax.legend(loc="lower left", fontsize=7, frameon=True, ncol=2)

    steps = np.arange(len(trace))
    battery_ax.plot(steps, energy, color="#0072B2", linewidth=1.4, alpha=0.28)
    battery_ax.axvline(commit_index, color="#E69F00", linestyle="--", linewidth=1.1)
    battery_progress, = battery_ax.plot([], [], color="#0072B2", linewidth=2.1)
    battery_dot, = battery_ax.plot([], [], "o", color="#D55E00", markersize=4)
    battery_ax.set_xlim(0, len(trace) - 1)
    battery_ax.set_ylim(0, max(energy) * 1.04)
    battery_ax.set_ylabel("energy (synthetic units)")
    battery_ax.set_title("Battery (dashed line = return commit)")

    distance_ax.plot(steps, distance, color="#999999", linewidth=1.2, alpha=0.3)
    distance_ax.axvline(commit_index, color="#E69F00", linestyle="--", linewidth=1.1)
    distance_progress, = distance_ax.plot([], [], color="#CC79A7", linewidth=2.1)
    distance_dot, = distance_ax.plot([], [], "o", color="#D55E00", markersize=4)
    distance_ax.set_xlim(0, len(trace) - 1)
    distance_ax.set_ylim(0, max(distance) * 1.05)
    distance_ax.set_xlabel("policy step")
    distance_ax.set_ylabel("distance to charger")
    distance_ax.set_title("Return progress")

    def update(frame: int):
        index = frame_indices[frame]
        task_end = min(index, commit_index)
        task_line.set_data(positions[: task_end + 1, 0], positions[: task_end + 1, 1])
        if index >= commit_index:
            return_line.set_data(positions[commit_index : index + 1, 0], positions[commit_index : index + 1, 1])
        else:
            return_line.set_data([], [])
        uav.set_offsets(positions[index, :2][None, :])
        goal.set_offsets(goals[index, :2][None, :])
        event = trace[index]["event"] or "in flight"
        info_text.set_text(
            f"step {index}/{len(trace)-1}\nmode: {modes[index]}\n"
            f"tasks completed: {tasks[index]}\nenergy: {energy[index]:.1f}\n"
            f"collisions: {trace[index]['collision_count']}\nevent: {event}"
        )
        battery_progress.set_data(steps[: index + 1], energy[: index + 1])
        battery_dot.set_data([index], [energy[index]])
        distance_progress.set_data(steps[: index + 1], distance[: index + 1])
        distance_dot.set_data([index], [distance[index]])
        return task_line, return_line, uav, goal, info_text, battery_progress, battery_dot, distance_progress, distance_dot

    movie = animation.FuncAnimation(fig, update, frames=len(frame_indices), interval=90, blit=False, repeat=True)
    movie.save(output, writer=animation.PillowWriter(fps=11))
    plt.close(fig)
    summary["gif_frame_count"] = len(frame_indices)
    summary["gif_sha256"] = file_hash(output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    trace, world, summary = replay(args.record.resolve(), args.contract.resolve(), args.device)
    arrays = {
        "step": np.asarray([row["step"] for row in trace], np.int64),
        "position": np.stack([row["position"] for row in trace]),
        "goal": np.stack([row["goal"] for row in trace]),
        "energy": np.asarray([row["energy"] for row in trace]),
        "tasks_completed": np.asarray([row["tasks_completed"] for row in trace], np.int64),
        "collision_count": np.asarray([row["collision_count"] for row in trace], np.int64),
        "distance_to_charger": np.asarray([row["distance_to_charger"] for row in trace]),
        "mode": np.asarray([row["mode"] for row in trace]),
        "event": np.asarray([row["event"] for row in trace]),
    }
    np.savez_compressed(output / "trajectory.npz", **arrays)
    gif = output / "best_world_mid_task_return.gif"
    render(trace, world, summary, gif)
    summary["trajectory_sha256"] = file_hash(output / "trajectory.npz")
    summary["source_record_sha256"] = file_hash(args.record.resolve())
    atomic_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
