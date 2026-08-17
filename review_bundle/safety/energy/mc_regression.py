from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from torch import nn
from torch.nn import functional


ENERGY_ESTIMATOR_TYPE = "mc_supervised_energy_to_go"
GROUP_CONFORMAL_ESTIMATOR_TYPE = "mc_supervised_energy_to_go_group_conformal"
ENERGY_DISTANCE_BUCKETS = (
    ("100-500", 100.0, 500.0),
    ("500-1500", 500.0, 1500.0),
    ("1500-2500", 1500.0, 2500.0),
    ("2500-4000", 2500.0, 4000.0),
    (">4000", 4000.0, float("inf")),
)


def energy_distance_bucket(distance: float) -> str:
    value = float(distance)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError("distance must be finite and nonnegative")
    if value < 100.0:
        return "100-500"
    for name, lower, upper in ENERGY_DISTANCE_BUCKETS:
        if lower <= value < upper:
            return name
    raise ValueError(f"distance {value} is outside configured buckets")


def finite_sample_conformal_margin(
    scores: np.ndarray,
    *,
    coverage: float,
) -> tuple[float, float, int]:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("conformal scores must be a nonempty vector")
    if not np.all(np.isfinite(values)):
        raise ValueError("conformal scores must be finite")
    if not 0.0 < coverage < 1.0:
        raise ValueError("coverage must lie in (0, 1)")
    rank = min(values.size, int(math.ceil((values.size + 1) * coverage)))
    raw_margin = float(np.partition(values, rank - 1)[rank - 1])
    return max(0.0, raw_margin), raw_margin, rank


@dataclass(frozen=True)
class HierarchicalConformalCalibration:
    coverage_target: float
    global_margin: float
    global_raw_margin: float
    global_rank: int
    num_trajectories: int
    goal_type_margins: dict[str, float]
    goal_type_raw_margins: dict[str, float]
    goal_type_ranks: dict[str, int]
    goal_type_counts: dict[str, int]
    distance_margins: dict[str, float]
    distance_raw_margins: dict[str, float]
    distance_ranks: dict[str, int]
    distance_counts: dict[str, int]
    intersection_counts: dict[str, int]

    @classmethod
    def fit(
        cls,
        predictions: np.ndarray,
        targets: np.ndarray,
        trajectory_ids: np.ndarray,
        goal_types: np.ndarray,
        distance_buckets: np.ndarray,
        *,
        coverage: float = 0.95,
    ) -> "HierarchicalConformalCalibration":
        predicted = np.asarray(predictions, dtype=np.float64)
        truth = np.asarray(targets, dtype=np.float64)
        ids = np.asarray(trajectory_ids, dtype=np.int64)
        types = np.asarray(goal_types).astype("U32")
        buckets = np.asarray(distance_buckets).astype("U16")
        if not (
            predicted.shape == truth.shape == ids.shape == types.shape == buckets.shape
            and predicted.ndim == 1
        ):
            raise ValueError("trajectory conformal arrays must be aligned vectors")
        rows: list[tuple[float, str, str]] = []
        for trajectory_id in np.unique(ids):
            mask = ids == trajectory_id
            unique_types = np.unique(types[mask])
            unique_buckets = np.unique(buckets[mask])
            if unique_types.size != 1 or unique_buckets.size != 1:
                raise ValueError("goal type and initial distance bucket must be trajectory-constant")
            rows.append(
                (
                    float(np.max(truth[mask] - predicted[mask])),
                    str(unique_types[0]),
                    str(unique_buckets[0]),
                )
            )
        scores = np.asarray([row[0] for row in rows], dtype=np.float64)
        global_margin, global_raw, global_rank = finite_sample_conformal_margin(
            scores,
            coverage=coverage,
        )

        def grouped(index: int):
            deployed: dict[str, float] = {}
            raw: dict[str, float] = {}
            ranks: dict[str, int] = {}
            counts: dict[str, int] = {}
            for name in sorted({row[index] for row in rows}):
                group_scores = np.asarray(
                    [row[0] for row in rows if row[index] == name],
                    dtype=np.float64,
                )
                margin, raw_margin, rank = finite_sample_conformal_margin(
                    group_scores,
                    coverage=coverage,
                )
                deployed[name] = margin
                raw[name] = raw_margin
                ranks[name] = rank
                counts[name] = int(group_scores.size)
            return deployed, raw, ranks, counts

        goal_margins, goal_raw, goal_ranks, goal_counts = grouped(1)
        distance_margins, distance_raw, distance_ranks, distance_counts = grouped(2)
        intersection_counts: dict[str, int] = {}
        for _, goal_type, bucket in rows:
            key = f"{goal_type}|{bucket}"
            intersection_counts[key] = intersection_counts.get(key, 0) + 1
        return cls(
            float(coverage),
            global_margin,
            global_raw,
            global_rank,
            len(rows),
            goal_margins,
            goal_raw,
            goal_ranks,
            goal_counts,
            distance_margins,
            distance_raw,
            distance_ranks,
            distance_counts,
            dict(sorted(intersection_counts.items())),
        )

    def margin_for(self, goal_type: str, distance: float) -> float:
        bucket = energy_distance_bucket(distance)
        return max(
            self.global_margin,
            self.goal_type_margins.get(str(goal_type), self.global_margin),
            self.distance_margins.get(bucket, self.global_margin),
        )

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "HierarchicalConformalCalibration":
        return cls(**payload)


@dataclass(frozen=True)
class MissionConformalCalibration:
    coverage_target: float
    global_margin: float
    global_raw_margin: float
    global_rank: int
    num_missions: int
    distance_margins: dict[str, float]
    distance_raw_margins: dict[str, float]
    distance_ranks: dict[str, int]
    distance_counts: dict[str, int]

    @classmethod
    def fit(
        cls,
        mission_scores: np.ndarray,
        distance_buckets: np.ndarray,
        *,
        coverage: float = 0.95,
    ) -> "MissionConformalCalibration":
        scores = np.asarray(mission_scores, dtype=np.float64)
        buckets = np.asarray(distance_buckets).astype("U16")
        if scores.ndim != 1 or scores.shape != buckets.shape or scores.size == 0:
            raise ValueError("mission scores and buckets must be aligned nonempty vectors")
        global_margin, global_raw, global_rank = finite_sample_conformal_margin(
            scores,
            coverage=coverage,
        )
        margins: dict[str, float] = {}
        raw_margins: dict[str, float] = {}
        ranks: dict[str, int] = {}
        counts: dict[str, int] = {}
        for bucket in sorted(np.unique(buckets)):
            group_scores = scores[buckets == bucket]
            margin, raw_margin, rank = finite_sample_conformal_margin(
                group_scores,
                coverage=coverage,
            )
            margins[str(bucket)] = margin
            raw_margins[str(bucket)] = raw_margin
            ranks[str(bucket)] = rank
            counts[str(bucket)] = int(group_scores.size)
        return cls(
            float(coverage),
            global_margin,
            global_raw,
            global_rank,
            int(scores.size),
            margins,
            raw_margins,
            ranks,
            counts,
        )

    def margin_for(self, task_distance: float) -> float:
        bucket = energy_distance_bucket(task_distance)
        return max(self.global_margin, self.distance_margins.get(bucket, self.global_margin))

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "MissionConformalCalibration":
        return cls(**payload)


class DeterministicGoalPolicy(Protocol):
    def predict(self, observation: np.ndarray, deterministic: bool = True): ...


@dataclass(frozen=True)
class GoalEnergyPrediction:
    prediction: float
    upper95: float
    rollout_steps: int = 0
    wall_clock_seconds: float = 0.0


@dataclass(frozen=True)
class EnergyTrainingHistory:
    best_epoch: int
    epochs_completed: int
    best_validation_mae: float
    train_loss: list[float]
    validation_mae: list[float]


class EnergyToGoNetwork(nn.Module):
    def __init__(self, input_dim: int = 7, hidden_dim: int = 128) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.backbone = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return functional.softplus(self.backbone(values)).squeeze(-1)


class EnergyToGoRegressor:
    estimator_type = ENERGY_ESTIMATOR_TYPE
    bootstrapping = False
    gamma = None

    def __init__(
        self,
        *,
        battery_capacity: float,
        input_dim: int = 7,
        hidden_dim: int = 128,
        learning_rate: float = 3e-4,
        batch_size: int = 256,
        seed: int = 0,
        device: str = "cpu",
    ) -> None:
        if not np.isfinite(battery_capacity) or battery_capacity <= 0.0:
            raise ValueError("battery_capacity must be finite and positive")
        if input_dim != 7:
            raise ValueError("MC Energy-to-Go input must be 7D")
        if hidden_dim <= 0 or learning_rate <= 0.0 or batch_size <= 0:
            raise ValueError("invalid supervised estimator configuration")
        torch.manual_seed(seed)
        self.battery_capacity = float(battery_capacity)
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.learning_rate = float(learning_rate)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.device = torch.device(device)
        self.model = EnergyToGoNetwork(self.input_dim, self.hidden_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.upper_delta = 0.0
        self.coverage_target = 0.95
        self.calibration_metadata: dict[str, object] = {}
        self.update_count = 0
        self.last_loss: float | None = None
        self.replay: tuple[()] = ()
        self.trainable_replay: tuple[()] = ()

    def normalize_target(self, physical_energy: np.ndarray | float) -> np.ndarray:
        return np.asarray(physical_energy, dtype=np.float32) / self.battery_capacity

    def denormalize_prediction(self, normalized_energy: np.ndarray | float) -> np.ndarray:
        return np.asarray(normalized_energy, dtype=np.float64) * self.battery_capacity

    def predict_normalized_batch(self, states: np.ndarray) -> np.ndarray:
        values = np.asarray(states, dtype=np.float32)
        if values.ndim == 1:
            values = values[None, :]
        if values.ndim != 2 or values.shape[1] != self.input_dim:
            raise ValueError(f"states must have shape [N, {self.input_dim}]")
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(
                torch.as_tensor(values, dtype=torch.float32, device=self.device)
            )
        return predictions.cpu().numpy().astype(np.float64, copy=False)

    def predict_batch(self, states: np.ndarray) -> np.ndarray:
        return self.denormalize_prediction(self.predict_normalized_batch(states))

    def predict(self, energy_state: np.ndarray, action: np.ndarray | None = None, **_: object) -> float:
        del action
        return float(self.predict_batch(np.asarray(energy_state, dtype=np.float32))[0])

    def predict_upper(self, energy_state: np.ndarray) -> float:
        return float(self.predict(energy_state) + self.upper_delta)

    def predict_quantiles(
        self,
        energy_state: np.ndarray,
        action: np.ndarray | None = None,
    ) -> np.ndarray:
        prediction = self.predict(energy_state, action)
        upper = prediction + self.upper_delta
        return np.asarray([prediction, upper, upper, upper], dtype=np.float64)

    def estimate_context(
        self,
        environment,
        goal: np.ndarray,
        *,
        position: np.ndarray | None = None,
        velocity: np.ndarray | None = None,
        goal_type: str | None = None,
    ) -> GoalEnergyPrediction:
        del goal_type
        state = environment.energy_state_for_goal(goal, position=position, velocity=velocity)
        prediction = self.predict(state)
        return GoalEnergyPrediction(prediction, prediction + self.upper_delta)

    def fit(
        self,
        train_states: np.ndarray,
        train_targets: np.ndarray,
        validation_states: np.ndarray,
        validation_targets: np.ndarray,
        *,
        max_epochs: int = 200,
        patience: int = 20,
        loss: str = "huber",
    ) -> EnergyTrainingHistory:
        if max_epochs <= 0 or patience <= 0:
            raise ValueError("max_epochs and patience must be positive")
        train_x = np.asarray(train_states, dtype=np.float32)
        train_y = self.normalize_target(train_targets)
        validation_x = np.asarray(validation_states, dtype=np.float32)
        validation_y = np.asarray(validation_targets, dtype=np.float64)
        if train_x.ndim != 2 or train_x.shape[1] != self.input_dim:
            raise ValueError("train_states have the wrong shape")
        if validation_x.ndim != 2 or validation_x.shape[1] != self.input_dim:
            raise ValueError("validation_states have the wrong shape")
        if train_y.shape != (train_x.shape[0],) or validation_y.shape != (validation_x.shape[0],):
            raise ValueError("targets do not align with states")
        if not np.all(np.isfinite(train_y)) or np.any(train_y < 0.0):
            raise ValueError("training targets must be finite and nonnegative")
        rng = np.random.default_rng(self.seed)
        best_state: dict[str, torch.Tensor] | None = None
        best_validation_mae = math.inf
        best_epoch = 0
        epochs_without_improvement = 0
        train_losses: list[float] = []
        validation_maes: list[float] = []
        for epoch in range(1, max_epochs + 1):
            self.model.train()
            order = rng.permutation(train_x.shape[0])
            epoch_losses: list[float] = []
            for offset in range(0, train_x.shape[0], self.batch_size):
                indices = order[offset : offset + self.batch_size]
                inputs = torch.as_tensor(train_x[indices], dtype=torch.float32, device=self.device)
                targets = torch.as_tensor(train_y[indices], dtype=torch.float32, device=self.device)
                predictions = self.model(inputs)
                if loss == "huber":
                    objective = functional.huber_loss(predictions, targets)
                elif loss == "mse":
                    objective = functional.mse_loss(predictions, targets)
                else:
                    raise ValueError("loss must be huber or mse")
                self.optimizer.zero_grad(set_to_none=True)
                objective.backward()
                self.optimizer.step()
                self.update_count += 1
                self.last_loss = float(objective.detach().cpu())
                epoch_losses.append(self.last_loss)
            validation_predictions = self.predict_batch(validation_x)
            validation_mae = float(np.mean(np.abs(validation_predictions - validation_y)))
            train_losses.append(float(np.mean(epoch_losses)))
            validation_maes.append(validation_mae)
            if validation_mae < best_validation_mae - 1e-8:
                best_validation_mae = validation_mae
                best_epoch = epoch
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in self.model.state_dict().items()
                }
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break
        if best_state is None:
            raise RuntimeError("supervised training did not produce a checkpoint")
        self.model.load_state_dict(best_state)
        self.model.eval()
        return EnergyTrainingHistory(
            best_epoch,
            len(train_losses),
            best_validation_mae,
            train_losses,
            validation_maes,
        )

    def calibrate_upper_bound(
        self,
        states: np.ndarray,
        targets: np.ndarray,
        trajectory_ids: np.ndarray,
        *,
        coverage: float = 0.95,
    ) -> dict[str, object]:
        if not 0.0 < coverage < 1.0:
            raise ValueError("coverage must lie in (0, 1)")
        predictions = self.predict_batch(states)
        truth = np.asarray(targets, dtype=np.float64)
        ids = np.asarray(trajectory_ids, dtype=np.int64)
        if predictions.shape != truth.shape or ids.shape != truth.shape:
            raise ValueError("calibration arrays must align")
        unique_ids = np.unique(ids)
        if unique_ids.size == 0:
            raise ValueError("calibration requires at least one trajectory")
        trajectory_scores = np.asarray(
            [np.max(truth[ids == trajectory_id] - predictions[ids == trajectory_id]) for trajectory_id in unique_ids],
            dtype=np.float64,
        )
        rank = min(unique_ids.size, int(math.ceil((unique_ids.size + 1) * coverage)))
        raw_delta = float(np.partition(trajectory_scores, rank - 1)[rank - 1])
        self.upper_delta = max(0.0, raw_delta)
        self.coverage_target = float(coverage)
        self.calibration_metadata = {
            "method": "one_sided_split_conformal",
            "calibration_unit": "trajectory_max_underprediction_residual",
            "num_calibration_trajectories": int(unique_ids.size),
            "finite_sample_rank": rank,
            "finite_sample_formula": "ceil((n+1)*coverage), capped at n",
            "coverage_target": float(coverage),
            "raw_residual_quantile": raw_delta,
            "upper_delta": self.upper_delta,
        }
        return dict(self.calibration_metadata)

    def checkpoint_payload(self) -> dict[str, object]:
        return {
            "energy_estimator_type": self.estimator_type,
            "bootstrapping": self.bootstrapping,
            "gamma": self.gamma,
            "navigation_policy": "frozen_sac_500k",
            "input_dim": self.input_dim,
            "energy_observation": "velocity3_goal_direction3_linear_distance_over_dmax1",
            "target_normalization": "mc_energy_to_go_over_calibrated_battery_capacity",
            "battery_capacity": self.battery_capacity,
            "hidden_dim": self.hidden_dim,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "seed": self.seed,
            "upper_delta": self.upper_delta,
            "coverage_target": self.coverage_target,
            "calibration_metadata": self.calibration_metadata,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "update_count": self.update_count,
        }

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.checkpoint_payload(), destination)

    @classmethod
    def from_checkpoint_payload(
        cls,
        payload: dict[str, object],
        *,
        device: str = "cpu",
    ) -> "EnergyToGoRegressor":
        if payload.get("energy_estimator_type") != ENERGY_ESTIMATOR_TYPE:
            raise ValueError("checkpoint is not an MC-supervised Energy-to-Go estimator")
        estimator = cls(
            battery_capacity=float(payload["battery_capacity"]),
            input_dim=int(payload["input_dim"]),
            hidden_dim=int(payload["hidden_dim"]),
            learning_rate=float(payload["learning_rate"]),
            batch_size=int(payload["batch_size"]),
            seed=int(payload.get("seed", 0)),
            device=device,
        )
        estimator.model.load_state_dict(payload["model_state_dict"])
        estimator.optimizer.load_state_dict(payload["optimizer_state_dict"])
        estimator.upper_delta = float(payload.get("upper_delta", 0.0))
        estimator.coverage_target = float(payload.get("coverage_target", 0.95))
        estimator.calibration_metadata = dict(payload.get("calibration_metadata", {}))
        estimator.update_count = int(payload.get("update_count", 0))
        estimator.model.eval()
        return estimator

    @classmethod
    def load(cls, path: str | Path, *, device: str = "cpu") -> "EnergyToGoRegressor":
        payload = torch.load(Path(path), map_location=device, weights_only=False)
        return cls.from_checkpoint_payload(payload, device=device)


class HierarchicalConformalEnergyEstimator:
    estimator_type = GROUP_CONFORMAL_ESTIMATOR_TYPE
    bootstrapping = False
    gamma = None

    def __init__(
        self,
        point_estimator: EnergyToGoRegressor,
        trajectory_calibration: HierarchicalConformalCalibration,
        mission_calibration: MissionConformalCalibration,
    ) -> None:
        self.point_estimator = point_estimator
        self.trajectory_calibration = trajectory_calibration
        self.mission_calibration = mission_calibration
        self.update_count = point_estimator.update_count
        self.replay: tuple[()] = ()
        self.trainable_replay: tuple[()] = ()

    @property
    def battery_capacity(self) -> float:
        return self.point_estimator.battery_capacity

    def predict(self, energy_state: np.ndarray, action: np.ndarray | None = None, **kwargs: object) -> float:
        return self.point_estimator.predict(energy_state, action, **kwargs)

    def predict_batch(self, states: np.ndarray) -> np.ndarray:
        return self.point_estimator.predict_batch(states)

    def predict_quantiles(
        self,
        energy_state: np.ndarray,
        action: np.ndarray | None = None,
    ) -> np.ndarray:
        prediction = self.predict(energy_state, action)
        upper = prediction + self.trajectory_calibration.global_margin
        return np.asarray([prediction, upper, upper, upper], dtype=np.float64)

    def estimate_context(
        self,
        environment,
        goal: np.ndarray,
        *,
        position: np.ndarray | None = None,
        velocity: np.ndarray | None = None,
        goal_type: str | None = None,
    ) -> GoalEnergyPrediction:
        selected_position = environment.agent.pos if position is None else np.asarray(position)
        selected_type = "TASK" if goal_type is None else str(goal_type)
        state = environment.energy_state_for_goal(goal, position=position, velocity=velocity)
        prediction = self.point_estimator.predict(state)
        distance = float(np.linalg.norm(np.asarray(goal) - selected_position))
        margin = self.trajectory_calibration.margin_for(selected_type, distance)
        return GoalEnergyPrediction(prediction, prediction + margin)

    def estimate_mission(
        self,
        task_prediction: float,
        return_after_task_prediction: float,
        *,
        task_distance: float,
    ) -> GoalEnergyPrediction:
        point = float(task_prediction + return_after_task_prediction)
        margin = self.mission_calibration.margin_for(task_distance)
        return GoalEnergyPrediction(point, point + margin)

    def checkpoint_payload(self) -> dict[str, object]:
        return {
            "energy_estimator_type": self.estimator_type,
            "bootstrapping": False,
            "gamma": None,
            "navigation_policy": "frozen_sac_500k",
            "point_estimator_payload": self.point_estimator.checkpoint_payload(),
            "trajectory_calibration": self.trajectory_calibration.as_dict(),
            "mission_calibration": self.mission_calibration.as_dict(),
        }

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.checkpoint_payload(), destination)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        device: str = "cpu",
    ) -> "HierarchicalConformalEnergyEstimator":
        payload = torch.load(Path(path), map_location=device, weights_only=False)
        if payload.get("energy_estimator_type") != GROUP_CONFORMAL_ESTIMATOR_TYPE:
            raise ValueError("checkpoint is not a hierarchical conformal energy estimator")
        point = EnergyToGoRegressor.from_checkpoint_payload(
            payload["point_estimator_payload"],
            device=device,
        )
        return cls(
            point,
            HierarchicalConformalCalibration.from_dict(payload["trajectory_calibration"]),
            MissionConformalCalibration.from_dict(payload["mission_calibration"]),
        )


class DistanceEnergyEstimator:
    estimator_type = "distance_times_empirical_energy_per_meter"
    bootstrapping = False
    gamma = None

    def __init__(self, *, energy_per_meter: float, d_max: float, upper_delta: float = 0.0) -> None:
        if not np.isfinite(energy_per_meter) or energy_per_meter <= 0.0:
            raise ValueError("energy_per_meter must be finite and positive")
        if not np.isfinite(d_max) or d_max <= 0.0:
            raise ValueError("d_max must be finite and positive")
        self.energy_per_meter = float(energy_per_meter)
        self.d_max = float(d_max)
        self.upper_delta = max(0.0, float(upper_delta))

    def predict_batch(self, states: np.ndarray) -> np.ndarray:
        values = np.asarray(states, dtype=np.float64)
        if values.ndim == 1:
            values = values[None, :]
        if values.ndim != 2 or values.shape[1] != 7:
            raise ValueError("states must have shape [N, 7]")
        return np.maximum(0.0, values[:, -1]) * self.d_max * self.energy_per_meter


class ModelBasedEnergyRolloutEstimator:
    estimator_type = "model_based_energy_rollout_oracle"
    bootstrapping = False
    gamma = None

    def __init__(
        self,
        policy: DeterministicGoalPolicy,
        *,
        max_policy_steps: int = 4000,
    ) -> None:
        if max_policy_steps <= 0:
            raise ValueError("max_policy_steps must be positive")
        self.policy = policy
        self.max_policy_steps = int(max_policy_steps)
        self.update_count = 0
        self.replay: tuple[()] = ()
        self.last_loss = None

    def estimate_context(
        self,
        environment,
        goal: np.ndarray,
        *,
        position: np.ndarray | None = None,
        velocity: np.ndarray | None = None,
        goal_type: str | None = None,
    ) -> GoalEnergyPrediction:
        del goal_type
        start_position = environment.agent.pos.copy() if position is None else np.asarray(position, dtype=np.float32)
        start_velocity = environment.agent.vel.copy() if velocity is None else np.asarray(velocity, dtype=np.float32)
        horizontal_speed = float(np.linalg.norm(start_velocity[:2]))
        if horizontal_speed > environment.horizontal_v_max:
            start_velocity = start_velocity.copy()
            start_velocity[:2] *= environment.horizontal_v_max / horizontal_speed
        start_velocity = start_velocity.copy()
        start_velocity[2] = np.clip(
            start_velocity[2],
            -environment.vertical_v_max,
            environment.vertical_v_max,
        )
        if float(np.linalg.norm(np.asarray(goal, dtype=np.float32) - start_position)) <= environment.goal_tolerance:
            return GoalEnergyPrediction(0.0, 0.0, 0, 0.0)
        clone = environment.__class__(
            length=environment.length,
            width=environment.width,
            height=environment.height,
            policy_dt=environment.policy_dt,
            physics_dt=environment.physics_dt,
            horizontal_v_max=environment.horizontal_v_max,
            vertical_v_max=environment.vertical_v_max,
            horizontal_a_max=environment.horizontal_a_max,
            vertical_a_max=environment.vertical_a_max,
            goal_radius=environment.goal_tolerance,
            near_goal_distance=environment.near_goal_distance,
            minimum_task_distance=environment.goal_tolerance,
            xy_sampling_margin=environment.xy_sampling_margin,
            task_z_min=environment.task_z_min,
            task_z_max=environment.task_z_max,
            max_steps_per_task=environment.max_steps_per_task,
            phase1_episode_max_policy_steps=environment.phase1_episode_max_policy_steps,
            phase2_episode_limit=environment.phase2_episode_limit,
            safe_radius=environment.safe_radius,
            telemetry_cost_config=environment.telemetry_cost_model.config,
            charger_position=environment.charger_position,
            phase=2,
        )
        observation, _ = clone.reset(
            seed=0,
            options={
                "start_position": start_position,
                "start_velocity": start_velocity,
                "task_point": np.asarray(goal, dtype=np.float32),
            },
        )
        total_energy = 0.0
        started = time.perf_counter()
        steps = 0
        while steps < self.max_policy_steps:
            action, _ = self.policy.predict(observation, deterministic=True)
            observation, _, terminated, truncated, info = clone.step(action)
            total_energy += float(info["realized_energy_cost"])
            steps += 1
            if terminated or truncated:
                if not bool(info["is_success"]):
                    clone.close()
                    raise RuntimeError(f"model-based energy rollout failed: {info['end_reason']}")
                elapsed = time.perf_counter() - started
                clone.close()
                return GoalEnergyPrediction(total_energy, total_energy, steps, elapsed)
        clone.close()
        raise RuntimeError("model-based energy rollout exceeded max_policy_steps")


def energy_regression_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    *,
    upper_bounds: np.ndarray | None = None,
) -> dict[str, float | int | bool]:
    predicted = np.asarray(predictions, dtype=np.float64)
    truth = np.asarray(targets, dtype=np.float64)
    if predicted.shape != truth.shape or predicted.ndim != 1:
        raise ValueError("predictions and targets must be aligned vectors")
    error = predicted - truth
    under = np.maximum(truth - predicted, 0.0)
    result: dict[str, float | int | bool] = {
        "count": int(truth.size),
        "finite_predictions": bool(np.all(np.isfinite(predicted))),
        "positive_predictions": bool(np.all(predicted > 0.0)),
        "mean_true_energy": float(np.mean(truth)),
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "bias": float(np.mean(error)),
        "mean_relative_error": float(np.mean(np.abs(error) / np.maximum(truth, 1e-8))),
        "median_relative_error": float(np.median(np.abs(error) / np.maximum(truth, 1e-8))),
        "underestimation_rate": float(np.mean(predicted < truth)),
        "mean_underestimation_magnitude": float(np.mean(under)),
        "max_underestimation": float(np.max(under)),
        "mean_prediction_true_ratio": float(np.mean(predicted / np.maximum(truth, 1e-8))),
    }
    if upper_bounds is not None:
        upper = np.asarray(upper_bounds, dtype=np.float64)
        if upper.shape != truth.shape:
            raise ValueError("upper_bounds must align with targets")
        result["upper95_coverage"] = float(np.mean(truth <= upper))
        result["mean_upper95_slack"] = float(np.mean(upper - truth))
    return result


__all__ = [
    "ENERGY_DISTANCE_BUCKETS",
    "ENERGY_ESTIMATOR_TYPE",
    "GROUP_CONFORMAL_ESTIMATOR_TYPE",
    "EnergyToGoNetwork",
    "EnergyToGoRegressor",
    "DistanceEnergyEstimator",
    "EnergyTrainingHistory",
    "GoalEnergyPrediction",
    "HierarchicalConformalCalibration",
    "HierarchicalConformalEnergyEstimator",
    "MissionConformalCalibration",
    "ModelBasedEnergyRolloutEstimator",
    "energy_distance_bucket",
    "energy_regression_metrics",
    "finite_sample_conformal_margin",
]
