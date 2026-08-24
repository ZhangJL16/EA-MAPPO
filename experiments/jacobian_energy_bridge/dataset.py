from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from .safety_buffer import SAFETY_CONTEXT_DIM, projection_context


@dataclass(frozen=True)
class PackedBridgeDataset:
    trajectory_ids: np.ndarray
    compact_energy_states: np.ndarray
    safety_contexts: np.ndarray
    nominal_actions: np.ndarray
    executed_actions: np.ndarray
    jacobians: np.ndarray
    realized_accelerations: np.ndarray
    step_energy: np.ndarray
    transition_dt: np.ndarray
    energy_to_go: np.ndarray
    burden_to_go: np.ndarray
    valid_jacobians: np.ndarray
    goal_types: np.ndarray
    initial_distances: np.ndarray
    distance_buckets: np.ndarray

    def __len__(self) -> int:
        return int(self.energy_to_go.size)

    @property
    def unique_trajectory_ids(self) -> set[int]:
        return set(int(value) for value in np.unique(self.trajectory_ids))

    def subset(self, trajectory_ids: set[int]) -> "PackedBridgeDataset":
        mask = np.isin(self.trajectory_ids, list(trajectory_ids))
        return PackedBridgeDataset(
            **{
                field: getattr(self, field)[mask]
                for field in self.__dataclass_fields__
            }
        )


def concatenate_bridge_datasets(
    *datasets: PackedBridgeDataset,
) -> PackedBridgeDataset:
    if not datasets:
        raise ValueError("at least one bridge dataset is required")
    arrays: dict[str, list[np.ndarray]] = {
        field: [] for field in PackedBridgeDataset.__dataclass_fields__
    }
    next_trajectory_id = 0
    for dataset in datasets:
        if len(dataset) == 0:
            continue
        source_ids = sorted(dataset.unique_trajectory_ids)
        remapping = {
            source_id: next_trajectory_id + index
            for index, source_id in enumerate(source_ids)
        }
        remapped_ids = np.asarray(
            [remapping[int(value)] for value in dataset.trajectory_ids],
            dtype=np.int64,
        )
        arrays["trajectory_ids"].append(remapped_ids)
        for field in PackedBridgeDataset.__dataclass_fields__:
            if field != "trajectory_ids":
                arrays[field].append(getattr(dataset, field))
        next_trajectory_id += len(source_ids)
    if not arrays["trajectory_ids"]:
        raise ValueError("bridge datasets contain no transitions")
    return PackedBridgeDataset(
        **{
            field: np.concatenate(values, axis=0)
            for field, values in arrays.items()
        }
    )


def distance_bucket(distance: float) -> str:
    if distance < 500.0:
        return "100-500"
    if distance < 1500.0:
        return "500-1500"
    if distance < 2500.0:
        return "1500-2500"
    if distance < 4000.0:
        return "2500-4000"
    return ">4000"


class SafetyBridgeTrajectoryWriter:
    def __init__(
        self,
        output_dir: str | Path,
        *,
        num_envs: int,
        maximum_barrier_constraints: int = 16,
        slack_scale: float = 5.0,
        intervention_weight: float = 1.0,
        authority_weight: float = 1.0,
        emergency_weight: float = 5.0,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.output_dir / "manifest.jsonl"
        self.pending: list[list[dict[str, object]]] = [[] for _ in range(num_envs)]
        self.maximum_barrier_constraints = int(maximum_barrier_constraints)
        self.slack_scale = float(slack_scale)
        self.intervention_weight = float(intervention_weight)
        self.authority_weight = float(authority_weight)
        self.emergency_weight = float(emergency_weight)
        self.completed_trajectories = 0
        self.discarded_trajectories = 0
        self.completed_transitions = 0

    def observe(self, env_index: int, info: Mapping[str, object], done: bool) -> None:
        geometry = info.get("projection_geometry")
        if not isinstance(geometry, Mapping):
            raise TypeError("trajectory info lacks projection geometry")
        emergency = bool(info.get("hocbf_emergency_brake", False))
        fallback = bool(info.get("hocbf_fallback_used", False))
        valid = bool(
            geometry.get("valid", False)
            and geometry.get("active_set_stable", False)
            and geometry.get("coordinate_map_stable", False)
            and not emergency
            and not fallback
        )
        context = projection_context(
            geometry,
            intervention_norm=float(info.get("hocbf_intervention_norm", 0.0)),
            emergency=emergency,
            maximum_barrier_constraints=self.maximum_barrier_constraints,
            slack_scale=self.slack_scale,
        )
        authority = float(np.clip(context[0], 0.0, 1.0))
        intervention = float(info.get("hocbf_intervention_norm", 0.0))
        burden = (
            self.intervention_weight * intervention
            + self.authority_weight * (1.0 - authority)
            + self.emergency_weight * float(emergency)
        )
        row = {
            "compact_energy_state": np.asarray(
                info["anchor_compact_energy_state"], dtype=np.float32
            ),
            "safety_context": context,
            "nominal_action": np.asarray(info["nominal_action"], dtype=np.float32),
            "executed_action": np.asarray(
                info["anchor_executed_action"], dtype=np.float32
            ),
            "jacobian": np.asarray(geometry["jacobian_total"], dtype=np.float32),
            "realized_acceleration": np.asarray(
                info["realized_acceleration"], dtype=np.float32
            ),
            "step_energy": float(info["realized_energy_cost"]),
            "transition_dt": float(info["transition_dt"]),
            "burden": burden,
            "valid_jacobian": valid,
            "goal_type": str(info.get("goal_type", "TASK")),
            "initial_distance": float(info.get("goal_initial_distance", 0.0)),
            "position": np.asarray(info["anchor_position"], dtype=np.float32),
            "goal": np.asarray(info["anchor_goal"], dtype=np.float32),
        }
        self.pending[env_index].append(row)
        successful_goal = bool(
            info.get("task_completed_now", False)
            or info.get("charger_reached_now", False)
        )
        censored = bool(
            info.get("energy_exhausted", False)
            or info.get("task_stuck", False)
            or info.get("navigation_failure", False)
        )
        if successful_goal:
            self._finalize(env_index)
        elif done or censored or bool(info.get("switched_now", False)):
            self._discard(env_index)

    def _finalize(self, env_index: int) -> None:
        rows = self.pending[env_index]
        if not rows:
            return
        trajectory_id = self.completed_trajectories
        step_energy = np.asarray([row["step_energy"] for row in rows], dtype=np.float32)
        transition_dt = np.asarray([row["transition_dt"] for row in rows], dtype=np.float32)
        burden = np.asarray([row["burden"] for row in rows], dtype=np.float32)
        energy_to_go = np.cumsum(step_energy[::-1], dtype=np.float64)[::-1].astype(np.float32)
        burden_to_go = np.cumsum(burden[::-1], dtype=np.float64)[::-1].astype(np.float32)
        payload = {
            "compact_energy_states": np.stack(
                [row["compact_energy_state"] for row in rows]
            ),
            "safety_contexts": np.stack([row["safety_context"] for row in rows]),
            "nominal_actions": np.stack([row["nominal_action"] for row in rows]),
            "executed_actions": np.stack([row["executed_action"] for row in rows]),
            "jacobians": np.stack([row["jacobian"] for row in rows]),
            "realized_accelerations": np.stack(
                [row["realized_acceleration"] for row in rows]
            ),
            "positions": np.stack([row["position"] for row in rows]),
            "goals": np.stack([row["goal"] for row in rows]),
            "step_energy": step_energy,
            "transition_dt": transition_dt,
            "energy_to_go": energy_to_go,
            "burden_to_go": burden_to_go,
            "valid_jacobians": np.asarray(
                [row["valid_jacobian"] for row in rows], dtype=np.bool_
            ),
        }
        filename = f"trajectory_{trajectory_id:06d}.npz"
        np.savez_compressed(self.output_dir / filename, **payload)
        initial_distance = float(rows[0]["initial_distance"])
        record = {
            "trajectory_id": trajectory_id,
            "file": filename,
            "num_transitions": len(rows),
            "goal_type": str(rows[0]["goal_type"]),
            "initial_distance": initial_distance,
            "distance_bucket": distance_bucket(initial_distance),
            "total_realized_energy": float(energy_to_go[0]),
            "total_safety_burden": float(burden_to_go[0]),
            "valid_jacobian_fraction": float(np.mean(payload["valid_jacobians"])),
        }
        with self.manifest_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        self.completed_trajectories += 1
        self.completed_transitions += len(rows)
        self.pending[env_index] = []

    def _discard(self, env_index: int) -> None:
        if self.pending[env_index]:
            self.discarded_trajectories += 1
        self.pending[env_index] = []

    def discard_partials(self) -> int:
        count = sum(bool(rows) for rows in self.pending)
        for env_index in range(len(self.pending)):
            self._discard(env_index)
        return count

    def metadata(self) -> dict[str, object]:
        return {
            "completed_trajectories": self.completed_trajectories,
            "discarded_trajectories": self.discarded_trajectories,
            "completed_transitions": self.completed_transitions,
            "partial_trajectories": sum(bool(rows) for rows in self.pending),
            "safety_context_dim": SAFETY_CONTEXT_DIM,
            "burden_definition": {
                "intervention_weight": self.intervention_weight,
                "authority_weight": self.authority_weight,
                "emergency_weight": self.emergency_weight,
            },
        }


def load_bridge_dataset(path: str | Path) -> PackedBridgeDataset:
    root = Path(path)
    manifest = [
        json.loads(line)
        for line in (root / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not manifest:
        raise ValueError("bridge dataset contains no completed trajectories")
    arrays: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "compact_energy_states",
            "safety_contexts",
            "nominal_actions",
            "executed_actions",
            "jacobians",
            "realized_accelerations",
            "step_energy",
            "transition_dt",
            "energy_to_go",
            "burden_to_go",
            "valid_jacobians",
        )
    }
    trajectory_ids: list[np.ndarray] = []
    goal_types: list[np.ndarray] = []
    initial_distances: list[np.ndarray] = []
    distance_buckets: list[np.ndarray] = []
    for row in manifest:
        values = np.load(root / row["file"])
        count = int(row["num_transitions"])
        for name in arrays:
            arrays[name].append(values[name])
        trajectory_ids.append(np.full(count, int(row["trajectory_id"]), dtype=np.int64))
        goal_types.append(np.full(count, str(row["goal_type"]), dtype="U32"))
        initial_distances.append(
            np.full(count, float(row["initial_distance"]), dtype=np.float32)
        )
        distance_buckets.append(
            np.full(count, str(row["distance_bucket"]), dtype="U16")
        )
    return PackedBridgeDataset(
        trajectory_ids=np.concatenate(trajectory_ids),
        compact_energy_states=np.concatenate(arrays["compact_energy_states"]),
        safety_contexts=np.concatenate(arrays["safety_contexts"]),
        nominal_actions=np.concatenate(arrays["nominal_actions"]),
        executed_actions=np.concatenate(arrays["executed_actions"]),
        jacobians=np.concatenate(arrays["jacobians"]),
        realized_accelerations=np.concatenate(arrays["realized_accelerations"]),
        step_energy=np.concatenate(arrays["step_energy"]),
        transition_dt=np.concatenate(arrays["transition_dt"]),
        energy_to_go=np.concatenate(arrays["energy_to_go"]),
        burden_to_go=np.concatenate(arrays["burden_to_go"]),
        valid_jacobians=np.concatenate(arrays["valid_jacobians"]),
        goal_types=np.concatenate(goal_types),
        initial_distances=np.concatenate(initial_distances),
        distance_buckets=np.concatenate(distance_buckets),
    )
