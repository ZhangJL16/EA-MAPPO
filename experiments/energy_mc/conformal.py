from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from experiments.energy_mc.core import (
    EnergyGoalSpec,
    EnergyTrajectory,
    PackedEnergyDataset,
    collect_energy_trajectory,
)
from review_bundle.safety.energy.mc_regression import (
    EnergyToGoRegressor,
    HierarchicalConformalEnergyEstimator,
    MissionConformalCalibration,
    energy_distance_bucket,
    energy_regression_metrics,
)


ENERGY_D_MAX = float(np.linalg.norm([4000.0, 4000.0, 400.0]))


def _zero_velocity_energy_state(start: np.ndarray, goal: np.ndarray) -> np.ndarray:
    delta = np.asarray(goal, dtype=np.float32) - np.asarray(start, dtype=np.float32)
    distance = float(np.linalg.norm(delta))
    direction = delta / max(distance, 1e-8)
    return np.concatenate(
        [
            np.zeros(3, dtype=np.float32),
            direction.astype(np.float32),
            np.asarray([distance / ENERGY_D_MAX], dtype=np.float32),
        ]
    )


@dataclass(frozen=True)
class MissionTrajectory:
    mission_id: int
    task: EnergyTrajectory
    return_after_task: EnergyTrajectory | None
    task_states: np.ndarray
    return_after_state: np.ndarray
    true_mission_energy: np.ndarray

    def metadata(self) -> dict[str, object]:
        return {
            "mission_id": self.mission_id,
            "initial_task_distance": self.task.spec.initial_goal_distance,
            "initial_task_distance_bucket": self.task.spec.distance_bucket,
            "task_trajectory_id": self.task.spec.trajectory_id,
            "return_trajectory_id": None
            if self.return_after_task is None
            else self.return_after_task.spec.trajectory_id,
            "task_energy": self.task.total_realized_energy,
            "return_after_task_energy": 0.0
            if self.return_after_task is None
            else self.return_after_task.total_realized_energy,
            "true_total_mission_energy": float(self.true_mission_energy[0]),
            "task_steps": self.task.steps,
            "return_steps": 0
            if self.return_after_task is None
            else self.return_after_task.steps,
            "had_boundary_contact": bool(
                self.task.had_boundary_contact
                or (
                    self.return_after_task is not None
                    and self.return_after_task.had_boundary_contact
                )
            ),
            "task_success": self.task.success,
            "return_success": True
            if self.return_after_task is None
            else self.return_after_task.success,
        }


@dataclass(frozen=True)
class PackedMissionDataset:
    task_states: np.ndarray
    return_after_states: np.ndarray
    true_mission_energy: np.ndarray
    mission_ids: np.ndarray
    initial_distance_buckets: np.ndarray
    metadata: list[dict[str, object]]

    @classmethod
    def from_trajectories(
        cls,
        trajectories: list[MissionTrajectory],
    ) -> "PackedMissionDataset":
        successful = [
            row
            for row in trajectories
            if row.task.success
            and (row.return_after_task is None or row.return_after_task.success)
        ]
        if not successful:
            raise ValueError("mission dataset requires successful task-return trajectories")
        return cls(
            task_states=np.concatenate([row.task_states for row in successful]).astype(
                np.float32
            ),
            return_after_states=np.concatenate(
                [
                    np.repeat(
                        row.return_after_state[None, :],
                        row.task_states.shape[0],
                        axis=0,
                    )
                    for row in successful
                ]
            ).astype(np.float32),
            true_mission_energy=np.concatenate(
                [row.true_mission_energy for row in successful]
            ).astype(np.float32),
            mission_ids=np.concatenate(
                [
                    np.full(row.task_states.shape[0], row.mission_id, dtype=np.int64)
                    for row in successful
                ]
            ),
            initial_distance_buckets=np.concatenate(
                [
                    np.full(
                        row.task_states.shape[0],
                        row.task.spec.distance_bucket,
                        dtype="U16",
                    )
                    for row in successful
                ]
            ),
            metadata=[row.metadata() for row in trajectories],
        )

    @property
    def successful_mission_ids(self) -> set[int]:
        return set(int(value) for value in np.unique(self.mission_ids))

    def save(self, directory: str | Path) -> None:
        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(
            destination / "mission_transitions.npz",
            task_states=self.task_states,
            return_after_states=self.return_after_states,
            true_mission_energy=self.true_mission_energy,
            mission_ids=self.mission_ids,
            initial_distance_buckets=self.initial_distance_buckets,
        )
        with (destination / "missions.jsonl").open("w", encoding="utf-8") as handle:
            for row in self.metadata:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        (destination / "manifest.json").write_text(
            json.dumps(
                {
                    "num_requested_missions": len(self.metadata),
                    "num_successful_missions": len(self.successful_mission_ids),
                    "num_task_states": int(self.task_states.shape[0]),
                    "split_unit": "complete_task_then_charger_mission",
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )


def collect_mission_trajectory(
    policy,
    environment_factory: Callable[[], object],
    task_spec: EnergyGoalSpec,
    *,
    charger_position: np.ndarray,
    seed: int,
) -> MissionTrajectory:
    if task_spec.goal_type != "TASK":
        raise ValueError("mission collection requires a TASK specification")
    task = collect_energy_trajectory(
        policy,
        environment_factory,
        task_spec,
        seed=seed,
    )
    if not task.success or task.final_position is None:
        raise RuntimeError("mission task segment did not reach its goal")
    charger = np.asarray(charger_position, dtype=np.float32)
    nominal_task_endpoint = task_spec.goal_position.astype(np.float32, copy=True)
    return_distance = float(np.linalg.norm(charger - task.final_position))
    return_after_state = _zero_velocity_energy_state(nominal_task_endpoint, charger)
    if return_distance <= 5.0:
        return MissionTrajectory(
            task_spec.trajectory_id,
            task,
            None,
            task.states.copy(),
            return_after_state,
            task.mc_energy_to_go.copy(),
        )
    return_spec = EnergyGoalSpec(
        trajectory_id=task_spec.trajectory_id + 100_000_000,
        goal_type="TASK_ENDPOINT_TO_CHARGER",
        start_position=task.final_position.copy(),
        goal_position=charger.copy(),
        initial_velocity=np.zeros(3, dtype=np.float32),
        initial_goal_distance=return_distance,
        distance_bucket=energy_distance_bucket(return_distance),
    )
    return_after = collect_energy_trajectory(
        policy,
        environment_factory,
        return_spec,
        seed=seed + 50_000_000,
    )
    if not return_after.success:
        raise RuntimeError("mission return-after-task segment did not reach charger")
    true_mission = task.mc_energy_to_go.astype(np.float64) + float(
        return_after.total_realized_energy
    )
    return MissionTrajectory(
        task_spec.trajectory_id,
        task,
        return_after,
        task.states.copy(),
        return_after_state,
        true_mission.astype(np.float32),
    )


def mission_scores(
    point_estimator: EnergyToGoRegressor,
    dataset: PackedMissionDataset,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    task_prediction = point_estimator.predict_batch(dataset.task_states)
    return_prediction = point_estimator.predict_batch(dataset.return_after_states)
    mission_prediction = task_prediction + return_prediction
    scores = []
    buckets = []
    for mission_id in np.unique(dataset.mission_ids):
        mask = dataset.mission_ids == mission_id
        scores.append(
            float(
                np.max(
                    dataset.true_mission_energy[mask]
                    - mission_prediction[mask]
                )
            )
        )
        buckets.append(str(np.unique(dataset.initial_distance_buckets[mask])[0]))
    return mission_prediction, np.asarray(scores), np.asarray(buckets, dtype="U16")


def _trajectory_coverage(
    truth: np.ndarray,
    upper: np.ndarray,
    trajectory_ids: np.ndarray,
    mask: np.ndarray,
) -> dict[str, object]:
    selected_ids = np.unique(trajectory_ids[mask])
    covered = []
    for trajectory_id in selected_ids:
        selected = (trajectory_ids == trajectory_id) & mask
        covered.append(bool(np.all(truth[selected] <= upper[selected])))
    return {
        "num_trajectories": int(selected_ids.size),
        "covered_trajectories": int(sum(covered)),
        "whole_trajectory_simultaneous_coverage": None
        if not covered
        else float(np.mean(covered)),
    }


def evaluate_group_conformal(
    estimator: HierarchicalConformalEnergyEstimator,
    dataset: PackedEnergyDataset,
) -> dict[str, object]:
    point = estimator.predict_batch(dataset.states)
    distances = dataset.states[:, -1].astype(np.float64) * float(
        np.linalg.norm([4000.0, 4000.0, 400.0])
    )
    margins = np.asarray(
        [
            estimator.trajectory_calibration.margin_for(goal_type, distance)
            for goal_type, distance in zip(dataset.goal_types, distances, strict=True)
        ],
        dtype=np.float64,
    )
    upper = point + margins

    def metrics(mask: np.ndarray) -> dict[str, object]:
        if not np.any(mask):
            return {"count": 0, "num_trajectories": 0}
        result = dict(
            energy_regression_metrics(
                point[mask],
                dataset.targets[mask],
                upper_bounds=upper[mask],
            )
        )
        result["pointwise_state_coverage"] = result.pop("upper95_coverage")
        result.update(
            _trajectory_coverage(
                dataset.targets,
                upper,
                dataset.trajectory_ids,
                mask,
            )
        )
        selected_margins = margins[mask]
        selected_truth = dataset.targets[mask]
        result.update(
            {
                "mean_margin": float(np.mean(selected_margins)),
                "median_margin": float(np.median(selected_margins)),
                "mean_margin_over_mean_true_energy": float(
                    np.mean(selected_margins) / np.mean(selected_truth)
                ),
            }
        )
        return result

    overall_mask = np.ones(dataset.targets.shape, dtype=bool)
    by_goal = {
        goal_type: metrics(dataset.goal_types == goal_type)
        for goal_type in np.unique(dataset.goal_types)
    }
    by_distance = {
        bucket: metrics(dataset.distance_buckets == bucket)
        for bucket in np.unique(dataset.distance_buckets)
    }
    intersections = {}
    for goal_type in np.unique(dataset.goal_types):
        for bucket in np.unique(dataset.distance_buckets):
            mask = (dataset.goal_types == goal_type) & (
                dataset.distance_buckets == bucket
            )
            if np.any(mask):
                intersections[f"{goal_type}|{bucket}"] = metrics(mask)
    return {
        "overall": metrics(overall_mask),
        "by_goal_type": by_goal,
        "by_initial_distance_bucket": by_distance,
        "by_goal_type_and_initial_distance": intersections,
    }


def evaluate_mission_conformal(
    estimator: HierarchicalConformalEnergyEstimator,
    dataset: PackedMissionDataset,
) -> dict[str, object]:
    point, _, _ = mission_scores(estimator.point_estimator, dataset)
    distances = dataset.task_states[:, -1].astype(np.float64) * float(
        np.linalg.norm([4000.0, 4000.0, 400.0])
    )
    margins = np.asarray(
        [estimator.mission_calibration.margin_for(distance) for distance in distances],
        dtype=np.float64,
    )
    upper = point + margins

    def metrics(mask: np.ndarray) -> dict[str, object]:
        if not np.any(mask):
            return {"count": 0, "num_trajectories": 0}
        result = dict(
            energy_regression_metrics(
                point[mask],
                dataset.true_mission_energy[mask],
                upper_bounds=upper[mask],
            )
        )
        result["pointwise_state_coverage"] = result.pop("upper95_coverage")
        result.update(
            _trajectory_coverage(
                dataset.true_mission_energy,
                upper,
                dataset.mission_ids,
                mask,
            )
        )
        result["mean_margin"] = float(np.mean(margins[mask]))
        result["median_margin"] = float(np.median(margins[mask]))
        result["mean_margin_over_mean_true_energy"] = float(
            np.mean(margins[mask]) / np.mean(dataset.true_mission_energy[mask])
        )
        return result

    return {
        "overall": metrics(np.ones(dataset.true_mission_energy.shape, dtype=bool)),
        "by_initial_task_distance_bucket": {
            bucket: metrics(dataset.initial_distance_buckets == bucket)
            for bucket in np.unique(dataset.initial_distance_buckets)
        },
    }


def mission_calibration_from_dataset(
    point_estimator: EnergyToGoRegressor,
    dataset: PackedMissionDataset,
    *,
    coverage: float,
) -> MissionConformalCalibration:
    _, scores, buckets = mission_scores(point_estimator, dataset)
    return MissionConformalCalibration.fit(scores, buckets, coverage=coverage)


__all__ = [
    "MissionTrajectory",
    "PackedMissionDataset",
    "collect_mission_trajectory",
    "evaluate_group_conformal",
    "evaluate_mission_conformal",
    "mission_calibration_from_dataset",
    "mission_scores",
]
