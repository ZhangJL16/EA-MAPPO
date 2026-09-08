from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import argparse
import hashlib
import json
import math
import sys
import time
from collections import deque
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
from experiments.forked_action_safety.core import (
    analytic_clearance,
    candidate_actions,
    sanitize_snapshot_position,
    sanitize_snapshot_velocity,
    select_anchor_indices,
    summarize_paired_gate,
)
from experiments.rechargeability_safety.core import decode_active_goal_state
from experiments.uav_energy_parallel import (
    ParallelUAVEnvPool,
    WorkerForkReset,
    WorkerRetarget,
)
from scripts.evaluate_jseb_checkpoints import reconstruct_environment_args
from scripts.train_uav_energy_delivery_sac import (
    NavigationTask,
    environment_from_args,
    environment_kwargs_from_args,
    generate_stratified_navigation_tasks,
)


PROTOCOL = "LIDAR_FORKED_ACTION_CAUSAL_DATA_GATE_V1"


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


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_value(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def save_npz_atomic(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    payload = json.dumps(json_value(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def observation_sha256(observation: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(observation, dtype=np.float32).tobytes()).hexdigest()


def selected_scene_indices(
    total_scenes: int,
    requested: int,
    *,
    offset_per_bucket: int = 0,
) -> list[int]:
    if total_scenes <= 0 or total_scenes % 5 or requested <= 0 or requested % 5:
        raise ValueError("source and requested scene counts must be positive multiples of five")
    if requested > total_scenes or offset_per_bucket < 0:
        raise ValueError("requested scenes exceed source scenes")
    source_per_bucket = total_scenes // 5
    requested_per_bucket = requested // 5
    if offset_per_bucket + requested_per_bucket > source_per_bucket:
        raise ValueError("requested bucket slice exceeds source scenes")
    return [
        bucket * source_per_bucket + offset
        for bucket in range(5)
        for offset in range(offset_per_bucket, offset_per_bucket + requested_per_bucket)
    ]


def _set_anchor_state(environment, *, position: np.ndarray, velocity: np.ndarray, elapsed: int) -> np.ndarray:
    environment.agent.pos = np.asarray(position, dtype=np.float32).copy()
    environment.agent.prev_pos = environment.agent.pos.copy()
    environment.agent.last_pos = environment.agent.pos.copy()
    environment.agent.spawn_pos = environment.agent.pos.copy()
    environment.agent.vel = np.asarray(velocity, dtype=np.float32).copy()
    environment.agent.goal = environment.current_task_point.copy()
    environment.agent_paths = [[environment.agent.pos.copy()]]
    environment.current_step = int(elapsed)
    environment.steps_in_current_task = int(elapsed)
    environment.simulation_time = elapsed * environment.policy_dt
    environment._current_goal_initial_distance = float(
        np.linalg.norm(environment.active_goal - environment.agent.pos)
    )
    environment._current_goal_path_length = 0.0
    environment._update_lidar()
    return environment._active_goal_sac_observation()


def _elapsed_in_leg(legs: np.ndarray, index: int) -> int:
    leg = int(legs[index])
    start = index
    while start > 0 and int(legs[start - 1]) == leg:
        start -= 1
    return index - start


def prepare_anchors(
    *,
    source: Path,
    output: Path,
    result: dict[str, object],
    policy: SAC,
    environment,
    tasks: list[NavigationTask],
    scene_indices: list[int],
    anchors_per_scene: int,
    hold_steps: int,
    perturbation: float,
) -> tuple[list[dict[str, object]], dict[str, np.ndarray]]:
    anchors: list[dict[str, object]] = []
    observations: list[np.ndarray] = []
    positions: list[np.ndarray] = []
    velocities: list[np.ndarray] = []
    nominal_actions: list[np.ndarray] = []
    all_candidates: list[np.ndarray] = []
    source_prefix_errors: list[float] = []
    for scene in scene_indices:
        task = tasks[scene]
        nominal_path = source / "rollouts" / f"scene_{scene:03d}" / "nominal.npz"
        with np.load(nominal_path) as arrays:
            compact = np.asarray(arrays["compact_observations"], dtype=np.float32)
            legs = np.asarray(arrays["leg"], dtype=np.int8)
            exact_positions = (
                np.asarray(arrays["positions"], dtype=np.float32)
                if "positions" in arrays.files
                else None
            )
            exact_velocities = (
                np.asarray(arrays["velocities"], dtype=np.float32)
                if "velocities" in arrays.files
                else None
            )
        initial_observation, _ = environment.reset(
            seed=int(result["world_seed"]) + scene,
            options={
                "start_position": task.start_position,
                "start_velocity": task.initial_velocity,
                "task_point": task.goal_position,
            },
        )
        obstacle_layout = environment.static_obstacle_layout()
        obstacle_digest = sha256_json(obstacle_layout)
        reconstructed_positions = []
        reconstructed_velocities = []
        charger = environment.charger_position.copy()
        for step, compact_state in enumerate(compact):
            if exact_positions is not None and exact_velocities is not None:
                if exact_positions.shape != (compact.shape[0], 3):
                    raise ValueError(f"invalid exact position shape in {nominal_path}")
                if exact_velocities.shape != (compact.shape[0], 3):
                    raise ValueError(f"invalid exact velocity shape in {nominal_path}")
                position = exact_positions[step]
                velocity = exact_velocities[step]
            else:
                active_goal = task.goal_position if int(legs[step]) == 0 else charger
                position, velocity = decode_active_goal_state(
                    compact_state,
                    active_goal,
                    d_max=environment.d_max,
                    horizontal_v_max=environment.horizontal_v_max,
                    vertical_v_max=environment.vertical_v_max,
                )
            # Compact float32 goal deltas can reconstruct a legal boundary
            # state a few ulps outside the map (observed at z=0.4999504 for a
            # 0.5 m limit).  Fork workers already apply this same ulp-bounded
            # projection.  Canonicalize here so the stored observation hash
            # matches the first reset instead of requiring a failed/resumed
            # run to migrate the still-unstarted anchor.
            reconstructed_positions.append(
                sanitize_snapshot_position(
                    position,
                    world_extent=np.asarray(
                        [environment.length, environment.width, environment.height],
                        dtype=np.float32,
                    ),
                    safe_radius=environment.safe_radius,
                )
            )
            reconstructed_velocities.append(
                sanitize_snapshot_velocity(
                    velocity,
                    horizontal_limit=environment.horizontal_v_max,
                    vertical_limit=environment.vertical_v_max,
                )
            )
        reconstructed_positions_array = np.asarray(reconstructed_positions, dtype=np.float32)
        reconstructed_velocities_array = np.asarray(reconstructed_velocities, dtype=np.float32)
        clearance = analytic_clearance(
            reconstructed_positions_array,
            obstacle_layout,
            safe_radius=environment.safe_radius,
        )
        chosen = select_anchor_indices(
            legs, clearance, anchors_per_scene=anchors_per_scene
        )
        for local_index, source_step in enumerate(chosen):
            leg = int(legs[source_step])
            active_goal = task.goal_position if leg == 0 else charger
            elapsed = _elapsed_in_leg(legs, int(source_step))
            # The original task start is protected by scene generation and is
            # used only for exact static-layout validation.
            generated, _ = environment.reset(
                seed=int(result["world_seed"]) + scene,
                options={
                    "start_position": task.start_position,
                    "start_velocity": np.zeros(3, dtype=np.float32),
                    "task_point": active_goal,
                    "static_obstacles": obstacle_layout,
                },
            )
            del generated
            observation = _set_anchor_state(
                environment,
                position=reconstructed_positions_array[source_step],
                velocity=reconstructed_velocities_array[source_step],
                elapsed=elapsed,
            )
            source_prefix_errors.append(
                float(np.max(np.abs(observation[:7] - compact[source_step])))
            )
            observations.append(observation.astype(np.float32))
            positions.append(reconstructed_positions_array[source_step])
            velocities.append(reconstructed_velocities_array[source_step])
            anchor_id = f"scene_{scene:03d}_anchor_{local_index:02d}"
            anchors.append(
                {
                    "anchor_id": anchor_id,
                    # This is the canonical row in anchors.npz.  The item has
                    # not been appended yet, so the current length is its row.
                    "anchor_index": len(anchors),
                    "scene_index": scene,
                    "source_transition_index": int(source_step),
                    "leg": leg,
                    "elapsed_policy_steps": elapsed,
                    "distance_bucket": task.distance_bucket,
                    "validation_start_position": task.start_position,
                    "position": reconstructed_positions_array[source_step],
                    "velocity": reconstructed_velocities_array[source_step],
                    "task_goal": task.goal_position,
                    "active_goal": active_goal,
                    "charger_goal": charger,
                    "nearest_obstacle_clearance": float(clearance[source_step]),
                    "obstacle_layout": obstacle_layout,
                    "obstacle_layout_sha256": obstacle_digest,
                    "anchor_observation_sha256": observation_sha256(observation),
                }
            )
        source_prefix_errors.append(
            float(np.max(np.abs(initial_observation - np.load(nominal_path)["initial_observation"])))
        )
    observation_batch = np.asarray(observations, dtype=np.float32)
    predicted, _ = policy.predict(observation_batch, deterministic=True)
    for index, nominal in enumerate(np.asarray(predicted, dtype=np.float32)):
        candidates = candidate_actions(
            nominal,
            position=np.asarray(anchors[index]["position"], dtype=np.float32),
            velocity=np.asarray(anchors[index]["velocity"], dtype=np.float32),
            obstacles=list(anchors[index]["obstacle_layout"]),
            perturbation=perturbation,
            hold_steps=hold_steps,
            policy_dt=environment.policy_dt,
            horizontal_a_max=environment.horizontal_a_max,
            vertical_a_max=environment.vertical_a_max,
        )
        nominal_actions.append(nominal)
        all_candidates.append(np.stack([item.action for item in candidates]))
        anchors[index]["candidate_names"] = [item.name for item in candidates]
        anchors[index]["candidate_actions"] = [item.action for item in candidates]
    arrays = {
        "observation": observation_batch,
        "position": np.asarray(positions, dtype=np.float32),
        "velocity": np.asarray(velocities, dtype=np.float32),
        "nominal_action": np.asarray(nominal_actions, dtype=np.float32),
        "candidate_actions": np.asarray(all_candidates, dtype=np.float32),
        "scene_index": np.asarray([item["scene_index"] for item in anchors], dtype=np.int64),
        "leg": np.asarray([item["leg"] for item in anchors], dtype=np.int8),
        "source_transition_index": np.asarray(
            [item["source_transition_index"] for item in anchors], dtype=np.int64
        ),
    }
    save_npz_atomic(output / "anchors.npz", **arrays)
    atomic_json(
        output / "anchors.json",
        {
            "anchors": anchors,
            "maximum_source_observation_prefix_error": max(source_prefix_errors),
        },
    )
    return anchors, arrays


def load_anchors(output: Path) -> tuple[list[dict[str, object]], dict[str, np.ndarray]]:
    metadata = json.loads((output / "anchors.json").read_text(encoding="utf-8"))
    with np.load(output / "anchors.npz") as values:
        arrays = {key: values[key] for key in values.files}
    return list(metadata["anchors"]), arrays


def migrate_unstarted_anchor_states(
    *,
    output: Path,
    anchors: list[dict[str, object]],
    arrays: dict[str, np.ndarray],
    environment,
    policy: SAC,
    hold_steps: int,
    perturbation: float,
) -> dict[str, int]:
    """Repair numerical-limit states without invalidating completed branches."""

    started = {path.parent.name for path in (output / "branches").glob("*/*.json")}
    changed: list[int] = []
    position_changes = 0
    velocity_changes = 0
    for index, anchor in enumerate(anchors):
        if str(anchor["anchor_id"]) in started:
            continue
        position = np.asarray(anchor["position"], dtype=np.float32)
        sanitized_position = sanitize_snapshot_position(
            position,
            world_extent=np.asarray(
                [environment.length, environment.width, environment.height],
                dtype=np.float32,
            ),
            safe_radius=environment.safe_radius,
        )
        velocity = np.asarray(anchor["velocity"], dtype=np.float32)
        sanitized = sanitize_snapshot_velocity(
            velocity,
            horizontal_limit=environment.horizontal_v_max,
            vertical_limit=environment.vertical_v_max,
        )
        position_changed = not np.array_equal(sanitized_position, position)
        velocity_changed = not np.array_equal(sanitized, velocity)
        if not position_changed and not velocity_changed:
            continue
        environment.reset(
            seed=0,
            options={
                "start_position": np.asarray(anchor["validation_start_position"], dtype=np.float32),
                "start_velocity": np.zeros(3, dtype=np.float32),
                "task_point": np.asarray(anchor["active_goal"], dtype=np.float32),
                "static_obstacles": list(anchor["obstacle_layout"]),
            },
        )
        observation = _set_anchor_state(
            environment,
            position=sanitized_position,
            velocity=sanitized,
            elapsed=int(anchor["elapsed_policy_steps"]),
        )
        anchor["position"] = sanitized_position
        anchor["velocity"] = sanitized
        anchor["anchor_observation_sha256"] = observation_sha256(observation)
        arrays["position"][index] = sanitized_position
        arrays["velocity"][index] = sanitized
        arrays["observation"][index] = observation
        changed.append(index)
        position_changes += int(position_changed)
        velocity_changes += int(velocity_changed)
    if not changed:
        return {"anchors": 0, "positions": 0, "velocities": 0}
    predicted, _ = policy.predict(arrays["observation"][changed], deterministic=True)
    for index, nominal in zip(changed, np.asarray(predicted, dtype=np.float32), strict=True):
        anchor = anchors[index]
        candidates = candidate_actions(
            nominal,
            position=np.asarray(anchor["position"], dtype=np.float32),
            velocity=np.asarray(anchor["velocity"], dtype=np.float32),
            obstacles=list(anchor["obstacle_layout"]),
            perturbation=perturbation,
            hold_steps=hold_steps,
            policy_dt=environment.policy_dt,
            horizontal_a_max=environment.horizontal_a_max,
            vertical_a_max=environment.vertical_a_max,
        )
        arrays["nominal_action"][index] = nominal
        arrays["candidate_actions"][index] = np.stack([item.action for item in candidates])
        anchor["candidate_names"] = [item.name for item in candidates]
        anchor["candidate_actions"] = [item.action for item in candidates]
    save_npz_atomic(output / "anchors.npz", **arrays)
    prior = json.loads((output / "anchors.json").read_text(encoding="utf-8"))
    atomic_json(
        output / "anchors.json",
        {
            "anchors": anchors,
            "maximum_source_observation_prefix_error": prior[
                "maximum_source_observation_prefix_error"
            ],
            "numerically_sanitized_unstarted_anchors": len(changed),
            "numerically_sanitized_unstarted_positions": position_changes,
            "numerically_sanitized_unstarted_velocities": velocity_changes,
        },
    )
    return {
        "anchors": len(changed),
        "positions": position_changes,
        "velocities": velocity_changes,
    }


def branch_path(output: Path, anchor_id: str, candidate_name: str) -> Path:
    return output / "branches" / anchor_id / f"{candidate_name}.json"


@dataclass
class ActiveBranch:
    anchor: dict[str, object]
    candidate_index: int
    candidate_name: str
    candidate_action: np.ndarray
    observation: np.ndarray
    current_leg: int
    prefix_steps: int = 0
    total_steps: int = 0
    task_success: bool = False
    return_success: bool = False
    collision: bool = False
    boundary_contact: bool = False
    total_realized_energy: float = 0.0
    first_executed_action: np.ndarray | None = None
    hocbf_intervention_steps: int = 0
    end_reason: str = ""
    censored: bool = False


def save_branch(output: Path, branch: ActiveBranch) -> dict[str, object]:
    if branch.first_executed_action is None:
        raise RuntimeError("cannot save a branch without an executed first action")
    safe_recharge = bool(
        not branch.collision
        and not branch.boundary_contact
        and branch.return_success
        and (branch.current_leg == 1)
        and (branch.task_success or int(branch.anchor["leg"]) == 1)
        and not branch.censored
    )
    row = {
        "protocol": PROTOCOL,
        "anchor_id": branch.anchor["anchor_id"],
        "anchor_index": branch.anchor["anchor_index"],
        "scene_index": branch.anchor["scene_index"],
        "distance_bucket": branch.anchor["distance_bucket"],
        "source_transition_index": branch.anchor["source_transition_index"],
        "source_leg": branch.anchor["leg"],
        "candidate_index": branch.candidate_index,
        "candidate_name": branch.candidate_name,
        "candidate_action": branch.candidate_action,
        "first_executed_action": branch.first_executed_action,
        "task_success": branch.task_success,
        "return_success": branch.return_success,
        "safe_recharge": safe_recharge,
        "collision": branch.collision,
        "boundary_contact": branch.boundary_contact,
        "total_realized_energy": branch.total_realized_energy,
        "total_steps": branch.total_steps,
        "raw_prefix_steps": branch.prefix_steps,
        "hocbf_intervention_steps_after_prefix": branch.hocbf_intervention_steps,
        "end_reason": branch.end_reason,
        "censored": branch.censored,
        "nearest_obstacle_clearance": branch.anchor["nearest_obstacle_clearance"],
        "anchor_observation_sha256": branch.anchor["anchor_observation_sha256"],
        "obstacle_layout_sha256": branch.anchor["obstacle_layout_sha256"],
    }
    atomic_json(branch_path(output, str(branch.anchor["anchor_id"]), branch.candidate_name), row)
    return row


def collect_branches(
    *,
    output: Path,
    anchors: list[dict[str, object]],
    policy: SAC,
    environment_kwargs: dict[str, object],
    workers: int,
    world_seed: int,
    hold_steps: int,
    resume: bool,
    smoke_max_branch_steps: int | None,
) -> list[dict[str, object]]:
    pending = deque()
    completed_rows = []
    for anchor in anchors:
        for candidate_index, (name, action) in enumerate(
            zip(anchor["candidate_names"], anchor["candidate_actions"], strict=True)
        ):
            path = branch_path(output, str(anchor["anchor_id"]), str(name))
            if resume and path.is_file():
                completed_rows.append(json.loads(path.read_text(encoding="utf-8")))
            else:
                pending.append((anchor, candidate_index, str(name), np.asarray(action, dtype=np.float32)))
    total = len(completed_rows) + len(pending)
    if not pending:
        return completed_rows
    with ParallelUAVEnvPool(environment_kwargs, num_workers=min(workers, len(pending))) as pool:
        active: dict[int, ActiveBranch] = {}

        def assign(worker_ids: list[int]) -> None:
            requests = []
            jobs = []
            for worker_id in worker_ids:
                if not pending:
                    break
                job = pending.popleft()
                anchor = job[0]
                requests.append(
                    WorkerForkReset(
                        worker_id=worker_id,
                        seed=world_seed + int(anchor["scene_index"]),
                        validation_start_position=np.asarray(
                            anchor["validation_start_position"], dtype=np.float32
                        ),
                        anchor_position=np.asarray(anchor["position"], dtype=np.float32),
                        anchor_velocity=np.asarray(anchor["velocity"], dtype=np.float32),
                        goal_position=np.asarray(anchor["active_goal"], dtype=np.float32),
                        static_obstacles=list(anchor["obstacle_layout"]),
                        elapsed_policy_steps=int(anchor["elapsed_policy_steps"]),
                    )
                )
                jobs.append((worker_id, job))
            if not requests:
                return
            resets = pool.fork_reset_many(requests)
            for worker_id, job in jobs:
                anchor, candidate_index, candidate_name, action = job
                reset = resets[worker_id]
                digest = observation_sha256(reset.observation)
                if digest != anchor["anchor_observation_sha256"]:
                    raise RuntimeError(
                        f"fork observation mismatch for {anchor['anchor_id']}: {digest}"
                    )
                active[worker_id] = ActiveBranch(
                    anchor=anchor,
                    candidate_index=candidate_index,
                    candidate_name=candidate_name,
                    candidate_action=action,
                    observation=reset.observation,
                    current_leg=int(anchor["leg"]),
                    task_success=int(anchor["leg"]) == 1,
                )

        assign(list(range(pool.num_workers)))
        while active:
            worker_ids = list(active)
            actions = np.empty((len(worker_ids), 3), dtype=np.float32)
            raw = np.zeros(len(worker_ids), dtype=np.bool_)
            continuation_positions = []
            continuation_observations = []
            for position, worker_id in enumerate(worker_ids):
                branch = active[worker_id]
                if branch.prefix_steps < hold_steps:
                    actions[position] = branch.candidate_action
                    raw[position] = True
                else:
                    continuation_positions.append(position)
                    continuation_observations.append(branch.observation)
            if continuation_positions:
                predicted, _ = policy.predict(
                    np.asarray(continuation_observations, dtype=np.float32),
                    deterministic=True,
                )
                actions[np.asarray(continuation_positions)] = np.asarray(predicted, dtype=np.float32)
            results = pool.step_many(worker_ids, actions, disable_cbf=raw)
            retargets = []
            finished_workers = []
            for position, worker_id in enumerate(worker_ids):
                branch = active[worker_id]
                result = results[worker_id]
                if branch.first_executed_action is None:
                    branch.first_executed_action = np.asarray(
                        result.info["executed_action"], dtype=np.float32
                    )
                if raw[position]:
                    branch.prefix_steps += 1
                else:
                    branch.hocbf_intervention_steps += int(
                        bool(result.info.get("hocbf_intervened", False))
                    )
                branch.total_steps += 1
                branch.total_realized_energy += float(result.info["realized_energy_cost"])
                branch.collision |= bool(result.info.get("obstacle_collision", False))
                branch.boundary_contact |= bool(result.info.get("boundary_contact", False))
                branch.observation = result.observation
                if branch.collision or branch.boundary_contact:
                    branch.end_reason = "unsafe_contact_during_raw_prefix" if raw[position] else "unsafe_contact_during_continuation"
                    save_branch(output, branch)
                    completed_rows.append(json.loads(branch_path(output, str(branch.anchor["anchor_id"]), branch.candidate_name).read_text()))
                    finished_workers.append(worker_id)
                    continue
                if smoke_max_branch_steps is not None and branch.total_steps >= smoke_max_branch_steps:
                    branch.end_reason = "smoke_step_cap"
                    branch.censored = True
                    save_branch(output, branch)
                    completed_rows.append(json.loads(branch_path(output, str(branch.anchor["anchor_id"]), branch.candidate_name).read_text()))
                    finished_workers.append(worker_id)
                    continue
                if not (result.terminated or result.truncated):
                    continue
                success = bool(result.info.get("is_success", False))
                if branch.current_leg == 0 and success:
                    branch.task_success = True
                    branch.current_leg = 1
                    branch.prefix_steps = hold_steps
                    retargets.append(
                        WorkerRetarget(
                            worker_id=worker_id,
                            seed=world_seed + int(branch.anchor["scene_index"]),
                            start_position=result.position,
                            start_velocity=np.zeros(3, dtype=np.float32),
                            goal_position=np.asarray(branch.anchor["charger_goal"], dtype=np.float32),
                        )
                    )
                else:
                    branch.return_success = bool(branch.current_leg == 1 and success)
                    branch.end_reason = str(result.info.get("end_reason", "unknown"))
                    save_branch(output, branch)
                    completed_rows.append(json.loads(branch_path(output, str(branch.anchor["anchor_id"]), branch.candidate_name).read_text()))
                    finished_workers.append(worker_id)
            if retargets:
                resets = pool.retarget_many(retargets)
                for request in retargets:
                    active[request.worker_id].observation = resets[request.worker_id].observation
            for worker_id in finished_workers:
                del active[worker_id]
            if finished_workers:
                assign(finished_workers)
                atomic_json(
                    output / "PROGRESS.json",
                    {
                        "completed_branches": len(completed_rows),
                        "total_branches": total,
                        "pending_branches": len(pending),
                        "active_branches": len(active),
                    },
                )
    return completed_rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-scenes", type=int, default=15)
    parser.add_argument("--scene-offset-per-bucket", type=int, default=0)
    parser.add_argument("--anchors-per-scene", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--hold-steps", type=int, default=8)
    parser.add_argument("--perturbation", type=float, default=0.45)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    if args.smoke:
        args.num_scenes = 5
        args.anchors_per_scene = 1
        args.num_workers = min(args.num_workers, 4)
        args.hold_steps = min(args.hold_steps, 2)
    if args.num_scenes <= 0 or args.num_scenes % 5:
        parser.error("--num-scenes must be a positive multiple of five")
    if min(args.anchors_per_scene, args.num_workers, args.hold_steps) <= 0:
        parser.error("anchor, worker, and hold-step counts must be positive")
    if not 0.0 < args.perturbation <= 1.0:
        parser.error("--perturbation must lie in (0, 1]")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.source = args.source.expanduser().resolve()
    args.protocol = args.protocol.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    source_result_path = args.source / "RESULT.json"
    for path in (source_result_path, args.protocol):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.resume:
        raise FileExistsError(f"output directory is not fresh: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.resume:
        (args.output_dir / "FAILED.json").unlink(missing_ok=True)
        (args.output_dir / "COMPLETED.json").unlink(missing_ok=True)
        (args.output_dir / "PAUSED.json").unlink(missing_ok=True)
    result = json.loads(source_result_path.read_text(encoding="utf-8"))
    environment_args, reconstruction = reconstruct_environment_args(
        Path(result["artifact"]), device=args.device, seed=int(result["world_seed"])
    )
    environment_args.phase1_episode_max_policy_steps = int(result["max_policy_steps_per_leg"])
    environment_args.phase1_max_steps = int(result["max_policy_steps_per_leg"])
    environment = environment_from_args(environment_args, phase=SACTrainingPhase.NAVIGATION)
    policy = SAC.load(
        Path(result["checkpoint"]), env=environment, device=args.device, print_system_info=False
    )
    tasks = generate_stratified_navigation_tasks(
        num_tasks=int(result["num_scenes"]), seed=int(result["task_seed"])
    )
    scene_indices = selected_scene_indices(
        int(result["num_scenes"]),
        args.num_scenes,
        offset_per_bucket=args.scene_offset_per_bucket,
    )
    manifest = {
        "status": "RUNNING",
        "protocol": PROTOCOL,
        "formal_evidence": False,
        "source": args.source,
        "source_result_sha256": sha256_file(source_result_path),
        "protocol_document": args.protocol,
        "protocol_sha256": sha256_file(args.protocol),
        "checkpoint": result["checkpoint"],
        "checkpoint_sha256": result["checkpoint_sha256"],
        "scene_indices": scene_indices,
        "num_scenes": args.num_scenes,
        "scene_offset_per_bucket": args.scene_offset_per_bucket,
        "anchors_per_scene": args.anchors_per_scene,
        "branches_per_anchor": 10,
        "num_workers": args.num_workers,
        "hold_steps": args.hold_steps,
        "hold_seconds": args.hold_steps * environment.policy_dt,
        "perturbation": args.perturbation,
        "device": args.device,
        "environment_reconstruction": reconstruction,
        "started_unix": time.time(),
        "exact_command": [sys.executable, *sys.argv],
    }
    atomic_json(args.output_dir / "RUNNING.json", manifest)
    started = time.time()
    try:
        if args.resume and (args.output_dir / "anchors.json").is_file() and (args.output_dir / "anchors.npz").is_file():
            anchors, anchor_arrays = load_anchors(args.output_dir)
            migrated = migrate_unstarted_anchor_states(
                output=args.output_dir,
                anchors=anchors,
                arrays=anchor_arrays,
                environment=environment,
                policy=policy,
                hold_steps=args.hold_steps,
                perturbation=args.perturbation,
            )
            if migrated["anchors"]:
                manifest["numerically_sanitized_unstarted_states"] = migrated
                atomic_json(args.output_dir / "RUNNING.json", manifest)
        else:
            anchors, anchor_arrays = prepare_anchors(
                source=args.source,
                output=args.output_dir,
                result=result,
                policy=policy,
                environment=environment,
                tasks=tasks,
                scene_indices=scene_indices,
                anchors_per_scene=args.anchors_per_scene,
                hold_steps=args.hold_steps,
                perturbation=args.perturbation,
            )
        environment.close()
        kwargs = environment_kwargs_from_args(environment_args, phase=SACTrainingPhase.NAVIGATION)
        kwargs["max_steps_per_task"] = int(result["max_policy_steps_per_leg"])
        kwargs["phase1_episode_max_policy_steps"] = int(result["max_policy_steps_per_leg"])
        rows = collect_branches(
            output=args.output_dir,
            anchors=anchors,
            policy=policy,
            environment_kwargs=kwargs,
            workers=args.num_workers,
            world_seed=int(result["world_seed"]),
            hold_steps=args.hold_steps,
            resume=args.resume,
            smoke_max_branch_steps=4 if args.smoke else None,
        )
        gate = summarize_paired_gate(rows)
        if args.smoke:
            gate["promotable"] = False
            gate["status"] = "SMOKE_COMPLETE_NOT_EVIDENCE"
        else:
            gate["status"] = (
                "PROMOTE_TO_LIDAR_ACTION_CRITIC_GATE"
                if gate["promotable"]
                else "REVISE_ANCHOR_OR_ACTION_SAMPLING"
            )
        final = {
            **manifest,
            **gate,
            "anchor_observation_shape": list(anchor_arrays["observation"].shape),
            "finished_unix": time.time(),
            "wall_clock_seconds": time.time() - started,
        }
        atomic_json(args.output_dir / "RESULT.json", final)
        atomic_json(
            args.output_dir / "COMPLETED.json",
            {"status": final["status"], "result": args.output_dir / "RESULT.json"},
        )
        (args.output_dir / "RUNNING.json").unlink(missing_ok=True)
        return 0
    except Exception as error:
        try:
            environment.close()
        except Exception:
            pass
        atomic_json(
            args.output_dir / "FAILED.json",
            {"type": type(error).__name__, "message": str(error), "failed_unix": time.time()},
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
