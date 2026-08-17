from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

import numpy as np


ENERGY_GOAL_TYPES = ("TASK", "CHARGER", "TASK_ENDPOINT_TO_CHARGER")
DISTANCE_BUCKETS = (
    ("100-500", 100.0, 500.0),
    ("500-1500", 500.0, 1500.0),
    ("1500-2500", 1500.0, 2500.0),
    ("2500-4000", 2500.0, 4000.0),
    (">4000", 4000.0, float("inf")),
)


class DeterministicPolicy(Protocol):
    def predict(self, observation: np.ndarray, deterministic: bool = True): ...


def monte_carlo_energy_to_go(step_energy: np.ndarray) -> np.ndarray:
    energy = np.asarray(step_energy, dtype=np.float64)
    if energy.ndim != 1 or energy.size == 0:
        raise ValueError("step_energy must be a nonempty vector")
    if not np.all(np.isfinite(energy)) or np.any(energy < 0.0):
        raise ValueError("step_energy must be finite and nonnegative")
    return np.cumsum(energy[::-1])[::-1]


@dataclass(frozen=True)
class EnergyGoalSpec:
    trajectory_id: int
    goal_type: str
    start_position: np.ndarray
    goal_position: np.ndarray
    initial_velocity: np.ndarray
    initial_goal_distance: float
    distance_bucket: str

    def as_dict(self) -> dict[str, object]:
        return {
            "trajectory_id": self.trajectory_id,
            "goal_type": self.goal_type,
            "start_position": self.start_position.tolist(),
            "goal_position": self.goal_position.tolist(),
            "initial_velocity": self.initial_velocity.tolist(),
            "initial_goal_distance": self.initial_goal_distance,
            "distance_bucket": self.distance_bucket,
        }


@dataclass(frozen=True)
class EnergyTrajectory:
    spec: EnergyGoalSpec
    states: np.ndarray
    step_energy: np.ndarray
    transition_dt: np.ndarray
    mc_energy_to_go: np.ndarray
    total_realized_energy: float
    path_length: float
    flight_time: float
    steps: int
    success: bool
    end_reason: str
    had_boundary_contact: bool
    boundary_contact_steps: int
    final_position: np.ndarray | None = None
    final_velocity: np.ndarray | None = None

    def metadata(self) -> dict[str, object]:
        return {
            **self.spec.as_dict(),
            "total_realized_energy": self.total_realized_energy,
            "path_length": self.path_length,
            "flight_time": self.flight_time,
            "steps": self.steps,
            "success": self.success,
            "end_reason": self.end_reason,
            "had_boundary_contact": self.had_boundary_contact,
            "boundary_contact_steps": self.boundary_contact_steps,
            "final_position": None
            if self.final_position is None
            else self.final_position.tolist(),
            "final_velocity": None
            if self.final_velocity is None
            else self.final_velocity.tolist(),
        }


@dataclass(frozen=True)
class PackedEnergyDataset:
    states: np.ndarray
    targets: np.ndarray
    step_energy: np.ndarray
    transition_dt: np.ndarray
    trajectory_ids: np.ndarray
    goal_types: np.ndarray
    distance_buckets: np.ndarray
    boundary_contact_trajectories: np.ndarray
    metadata: list[dict[str, object]]

    @classmethod
    def from_trajectories(cls, trajectories: list[EnergyTrajectory]) -> "PackedEnergyDataset":
        successful = [trajectory for trajectory in trajectories if trajectory.success]
        if not successful:
            raise ValueError("dataset requires at least one successful trajectory")
        states = np.concatenate([row.states for row in successful]).astype(np.float32)
        targets = np.concatenate([row.mc_energy_to_go for row in successful]).astype(np.float32)
        step_energy = np.concatenate([row.step_energy for row in successful]).astype(np.float32)
        transition_dt = np.concatenate([row.transition_dt for row in successful]).astype(np.float32)
        trajectory_ids = np.concatenate(
            [np.full(row.steps, row.spec.trajectory_id, dtype=np.int64) for row in successful]
        )
        goal_types = np.concatenate(
            [np.full(row.steps, row.spec.goal_type, dtype="U32") for row in successful]
        )
        distance_buckets = np.concatenate(
            [np.full(row.steps, row.spec.distance_bucket, dtype="U16") for row in successful]
        )
        boundary = np.concatenate(
            [np.full(row.steps, row.had_boundary_contact, dtype=bool) for row in successful]
        )
        if states.shape[1] != 7:
            raise ValueError("energy states must be 7D")
        return cls(
            states,
            targets,
            step_energy,
            transition_dt,
            trajectory_ids,
            goal_types,
            distance_buckets,
            boundary,
            [row.metadata() for row in trajectories],
        )

    @property
    def successful_trajectory_ids(self) -> set[int]:
        return set(int(value) for value in np.unique(self.trajectory_ids))

    def save(self, directory: str | Path) -> None:
        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(
            destination / "transitions.npz",
            states=self.states,
            targets=self.targets,
            step_energy=self.step_energy,
            transition_dt=self.transition_dt,
            trajectory_ids=self.trajectory_ids,
            goal_types=self.goal_types,
            distance_buckets=self.distance_buckets,
            boundary_contact_trajectories=self.boundary_contact_trajectories,
        )
        with (destination / "trajectories.jsonl").open("w", encoding="utf-8") as handle:
            for row in self.metadata:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        manifest = {
            "trajectory_split_unit": "complete_goal_trajectory",
            "num_requested_trajectories": len(self.metadata),
            "num_successful_trajectories": len(self.successful_trajectory_ids),
            "num_transitions": int(self.states.shape[0]),
            "goal_type_counts": {
                goal_type: sum(row["goal_type"] == goal_type for row in self.metadata)
                for goal_type in ENERGY_GOAL_TYPES
            },
            "success_rate": float(
                np.mean([bool(row["success"]) for row in self.metadata])
            ),
        }
        (destination / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: str | Path) -> "PackedEnergyDataset":
        source = Path(directory)
        with np.load(source / "transitions.npz", allow_pickle=False) as payload:
            arrays = {key: payload[key] for key in payload.files}
        metadata = [
            json.loads(line)
            for line in (source / "trajectories.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return cls(metadata=metadata, **arrays)

    @classmethod
    def concatenate(cls, datasets: list["PackedEnergyDataset"]) -> "PackedEnergyDataset":
        if not datasets:
            raise ValueError("at least one dataset is required")
        combined = cls(
            states=np.concatenate([row.states for row in datasets]),
            targets=np.concatenate([row.targets for row in datasets]),
            step_energy=np.concatenate([row.step_energy for row in datasets]),
            transition_dt=np.concatenate([row.transition_dt for row in datasets]),
            trajectory_ids=np.concatenate([row.trajectory_ids for row in datasets]),
            goal_types=np.concatenate([row.goal_types for row in datasets]),
            distance_buckets=np.concatenate([row.distance_buckets for row in datasets]),
            boundary_contact_trajectories=np.concatenate(
                [row.boundary_contact_trajectories for row in datasets]
            ),
            metadata=[item for row in datasets for item in row.metadata],
        )
        if len(combined.successful_trajectory_ids) != sum(
            len(row.successful_trajectory_ids) for row in datasets
        ):
            raise ValueError("cannot concatenate datasets with overlapping trajectory ids")
        return combined


def _sample_position(
    rng: np.random.Generator,
    *,
    length: float,
    width: float,
    z_min: float,
    z_max: float,
    margin: float,
) -> np.ndarray:
    return np.asarray(
        [
            rng.uniform(margin, length - margin),
            rng.uniform(margin, width - margin),
            rng.uniform(z_min, z_max),
        ],
        dtype=np.float32,
    )


def _sample_pair_for_bucket(
    rng: np.random.Generator,
    bucket: tuple[str, float, float],
    *,
    length: float,
    width: float,
    z_min: float,
    z_max: float,
    margin: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    _, lower, upper = bucket
    for _ in range(500_000):
        start = _sample_position(
            rng,
            length=length,
            width=width,
            z_min=z_min,
            z_max=z_max,
            margin=margin,
        )
        goal = _sample_position(
            rng,
            length=length,
            width=width,
            z_min=z_min,
            z_max=z_max,
            margin=margin,
        )
        distance = float(np.linalg.norm(goal - start))
        if lower <= distance < upper:
            return start, goal, distance
    raise RuntimeError(f"could not sample pair for bucket {bucket[0]}")


def _sample_charger_context(
    rng: np.random.Generator,
    bucket: tuple[str, float, float],
    charger: np.ndarray,
    *,
    length: float,
    width: float,
    z_min: float,
    z_max: float,
    margin: float,
) -> tuple[np.ndarray, float]:
    _, lower, upper = bucket
    for _ in range(500_000):
        start = _sample_position(
            rng,
            length=length,
            width=width,
            z_min=z_min,
            z_max=z_max,
            margin=margin,
        )
        distance = float(np.linalg.norm(charger - start))
        if lower <= distance < upper:
            return start, distance
    raise RuntimeError(f"charger cannot support requested bucket {bucket[0]}")


def generate_energy_goal_specs(
    *,
    num_trajectories: int,
    seed: int,
    charger_position: np.ndarray,
    trajectory_id_offset: int = 0,
    length: float = 4000.0,
    width: float = 4000.0,
    z_min: float = 20.0,
    z_max: float = 380.0,
    margin: float = 100.0,
) -> list[EnergyGoalSpec]:
    if num_trajectories <= 0 or num_trajectories % len(DISTANCE_BUCKETS) != 0:
        raise ValueError("trajectory count must be a positive multiple of five")
    rng = np.random.default_rng(seed)
    charger = np.asarray(charger_position, dtype=np.float32)
    specs: list[EnergyGoalSpec] = []
    per_bucket = num_trajectories // len(DISTANCE_BUCKETS)
    for bucket in DISTANCE_BUCKETS:
        bucket_name = bucket[0]
        for local_index in range(per_bucket):
            if bucket_name == ">4000":
                goal_type = "TASK"
            else:
                goal_type = ENERGY_GOAL_TYPES[local_index % len(ENERGY_GOAL_TYPES)]
            if goal_type == "TASK":
                start, goal, distance = _sample_pair_for_bucket(
                    rng,
                    bucket,
                    length=length,
                    width=width,
                    z_min=z_min,
                    z_max=z_max,
                    margin=margin,
                )
                velocity = np.zeros(3, dtype=np.float32)
            else:
                start, distance = _sample_charger_context(
                    rng,
                    bucket,
                    charger,
                    length=length,
                    width=width,
                    z_min=z_min,
                    z_max=z_max,
                    margin=margin,
                )
                goal = charger.copy()
                if goal_type == "CHARGER":
                    velocity = np.asarray(
                        [rng.uniform(-2.0, 2.0), rng.uniform(-2.0, 2.0), rng.uniform(-0.5, 0.5)],
                        dtype=np.float32,
                    )
                else:
                    velocity = np.zeros(3, dtype=np.float32)
            specs.append(
                EnergyGoalSpec(
                    trajectory_id_offset + len(specs),
                    goal_type,
                    start,
                    goal,
                    velocity,
                    distance,
                    bucket_name,
                )
            )
    return specs


def generate_intersection_stratified_energy_goal_specs(
    *,
    num_trajectories: int,
    seed: int,
    charger_position: np.ndarray,
    trajectory_id_offset: int = 0,
    length: float = 4000.0,
    width: float = 4000.0,
    z_min: float = 20.0,
    z_max: float = 380.0,
    margin: float = 100.0,
) -> list[EnergyGoalSpec]:
    if num_trajectories < 13:
        raise ValueError("intersection-stratified data requires at least 13 trajectories")
    feasible_cells = [
        (goal_type, bucket)
        for bucket in DISTANCE_BUCKETS
        for goal_type in ENERGY_GOAL_TYPES
        if not (bucket[0] == ">4000" and goal_type != "TASK")
    ]
    rng = np.random.default_rng(seed)
    base, remainder = divmod(num_trajectories, len(feasible_cells))
    specs: list[EnergyGoalSpec] = []
    charger = np.asarray(charger_position, dtype=np.float32)
    for cell_index, (goal_type, bucket) in enumerate(feasible_cells):
        count = base + int(cell_index < remainder)
        for _ in range(count):
            if goal_type == "TASK":
                start, goal, distance = _sample_pair_for_bucket(
                    rng,
                    bucket,
                    length=length,
                    width=width,
                    z_min=z_min,
                    z_max=z_max,
                    margin=margin,
                )
                velocity = np.zeros(3, dtype=np.float32)
            else:
                start, distance = _sample_charger_context(
                    rng,
                    bucket,
                    charger,
                    length=length,
                    width=width,
                    z_min=z_min,
                    z_max=z_max,
                    margin=margin,
                )
                goal = charger.copy()
                velocity = (
                    np.asarray(
                        [
                            rng.uniform(-2.0, 2.0),
                            rng.uniform(-2.0, 2.0),
                            rng.uniform(-0.5, 0.5),
                        ],
                        dtype=np.float32,
                    )
                    if goal_type == "CHARGER"
                    else np.zeros(3, dtype=np.float32)
                )
            specs.append(
                EnergyGoalSpec(
                    trajectory_id_offset + len(specs),
                    goal_type,
                    start,
                    goal,
                    velocity,
                    distance,
                    bucket[0],
                )
            )
    return specs


def generate_stratified_task_specs(
    *,
    num_trajectories: int,
    seed: int,
    trajectory_id_offset: int = 0,
    length: float = 4000.0,
    width: float = 4000.0,
    z_min: float = 20.0,
    z_max: float = 380.0,
    margin: float = 100.0,
) -> list[EnergyGoalSpec]:
    if num_trajectories <= 0 or num_trajectories % len(DISTANCE_BUCKETS) != 0:
        raise ValueError("task trajectory count must be a positive multiple of five")
    rng = np.random.default_rng(seed)
    specs: list[EnergyGoalSpec] = []
    per_bucket = num_trajectories // len(DISTANCE_BUCKETS)
    for bucket in DISTANCE_BUCKETS:
        for _ in range(per_bucket):
            start, goal, distance = _sample_pair_for_bucket(
                rng,
                bucket,
                length=length,
                width=width,
                z_min=z_min,
                z_max=z_max,
                margin=margin,
            )
            specs.append(
                EnergyGoalSpec(
                    trajectory_id_offset + len(specs),
                    "TASK",
                    start,
                    goal,
                    np.zeros(3, dtype=np.float32),
                    distance,
                    bucket[0],
                )
            )
    return specs


def collect_energy_trajectory(
    policy: DeterministicPolicy,
    environment_factory: Callable[[], object],
    spec: EnergyGoalSpec,
    *,
    seed: int,
) -> EnergyTrajectory:
    environment = environment_factory()
    observation, _ = environment.reset(
        seed=seed,
        options={
            "start_position": spec.start_position,
            "start_velocity": spec.initial_velocity,
            "task_point": spec.goal_position,
        },
    )
    states: list[np.ndarray] = []
    costs: list[float] = []
    transition_dts: list[float] = []
    path_length = 0.0
    boundary_contacts = 0
    success = False
    end_reason = "unknown"
    while True:
        states.append(environment.energy_state_for_goal(spec.goal_position))
        previous_position = environment.agent.pos.copy()
        action, _ = policy.predict(observation, deterministic=True)
        observation, _, terminated, truncated, info = environment.step(action)
        costs.append(float(info["realized_energy_cost"]))
        transition_dts.append(float(info["transition_dt"]))
        path_length += float(np.linalg.norm(environment.agent.pos - previous_position))
        boundary_contacts += int(info["boundary_contact"])
        if terminated or truncated:
            success = bool(info["is_success"])
            end_reason = str(info["end_reason"])
            break
    final_position = environment.agent.pos.copy()
    final_velocity = environment.agent.vel.copy()
    environment.close()
    energy = np.asarray(costs, dtype=np.float32)
    returns = monte_carlo_energy_to_go(energy).astype(np.float32)
    return EnergyTrajectory(
        spec,
        np.asarray(states, dtype=np.float32),
        energy,
        np.asarray(transition_dts, dtype=np.float32),
        returns,
        float(np.sum(energy, dtype=np.float64)),
        path_length,
        float(np.sum(transition_dts, dtype=np.float64)),
        len(costs),
        success,
        end_reason,
        boundary_contacts > 0,
        boundary_contacts,
        final_position,
        final_velocity,
    )


def assert_disjoint_trajectory_splits(splits: dict[str, PackedEnergyDataset]) -> None:
    names = list(splits)
    for index, left_name in enumerate(names):
        for right_name in names[index + 1 :]:
            overlap = (
                splits[left_name].successful_trajectory_ids
                & splits[right_name].successful_trajectory_ids
            )
            if overlap:
                raise ValueError(
                    f"trajectory leakage between {left_name} and {right_name}: {sorted(overlap)[:5]}"
                )


__all__ = [
    "DISTANCE_BUCKETS",
    "ENERGY_GOAL_TYPES",
    "EnergyGoalSpec",
    "EnergyTrajectory",
    "PackedEnergyDataset",
    "assert_disjoint_trajectory_splits",
    "collect_energy_trajectory",
    "generate_energy_goal_specs",
    "generate_intersection_stratified_energy_goal_specs",
    "generate_stratified_task_specs",
    "monte_carlo_energy_to_go",
]
