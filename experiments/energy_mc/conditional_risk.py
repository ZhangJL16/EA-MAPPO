from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch import nn
from torch.nn import functional

from experiments.energy_mc.adaptive_uncertainty import (
    AdaptiveTrainingHistory,
    HeteroscedasticResidualModel,
    TrajectoryConformalCorrection,
)
from experiments.energy_mc.core import DISTANCE_BUCKETS, PackedEnergyDataset
from review_bundle.safety.energy.mc_regression import (
    EnergyToGoRegressor,
    GoalEnergyPrediction,
    HierarchicalConformalCalibration,
    finite_sample_conformal_margin,
)


MAP_EXTENT = np.asarray([4000.0, 4000.0, 400.0], dtype=np.float64)
RESIDUAL_QUANTILE_LEVELS = (0.90, 0.95, 0.975, 0.99)
GoalFeatureMode = Literal[
    "state7",
    "absolute_position",
    "boundary_distances",
    "compact_position_boundary",
    "compact_decision_context",
    "position_and_boundary",
]


def positive_underestimation_target(truth: np.ndarray, point: np.ndarray) -> np.ndarray:
    targets = np.asarray(truth, dtype=np.float64)
    predictions = np.asarray(point, dtype=np.float64)
    if targets.shape != predictions.shape:
        raise ValueError("truth and point prediction must align")
    return np.maximum(targets - predictions, 0.0)


def trajectory_balanced_state_indices(
    trajectory_ids: np.ndarray,
    *,
    max_states_per_trajectory: int,
) -> np.ndarray:
    ids = np.asarray(trajectory_ids, dtype=np.int64)
    if ids.ndim != 1 or ids.size == 0 or max_states_per_trajectory <= 0:
        raise ValueError("trajectory ids must be nonempty and max states must be positive")
    starts = np.concatenate([[0], np.flatnonzero(ids[1:] != ids[:-1]) + 1])
    stops = np.concatenate([starts[1:], [len(ids)]])
    if starts.size != np.unique(ids).size:
        raise ValueError("each trajectory must occupy one contiguous block")
    selected = []
    for start, stop in zip(starts, stops, strict=True):
        count = int(stop - start)
        if count <= max_states_per_trajectory:
            selected.append(np.arange(start, stop, dtype=np.int64))
        else:
            selected.append(
                np.unique(
                    np.linspace(
                        start,
                        stop - 1,
                        max_states_per_trajectory,
                        dtype=np.int64,
                    )
                )
            )
    return np.concatenate(selected)


def distance_bucket_for(distance: float) -> str:
    value = float(distance)
    for name, lower, upper in DISTANCE_BUCKETS:
        if lower <= value < upper:
            return name
    if value < DISTANCE_BUCKETS[0][1]:
        return DISTANCE_BUCKETS[0][0]
    return DISTANCE_BUCKETS[-1][0]


def reconstruct_positions(dataset: PackedEnergyDataset) -> np.ndarray:
    metadata = {int(row["trajectory_id"]): row for row in dataset.metadata}
    positions = np.empty((dataset.states.shape[0], 3), dtype=np.float64)
    ids = np.asarray(dataset.trajectory_ids, dtype=np.int64)
    starts = np.concatenate([[0], np.flatnonzero(ids[1:] != ids[:-1]) + 1])
    stops = np.concatenate([starts[1:], [len(ids)]])
    if len(starts) != len(np.unique(ids)):
        raise ValueError("goal-risk feature construction requires contiguous trajectories")
    for start, stop in zip(starts, stops, strict=True):
        trajectory_id = int(ids[start])
        row = metadata[trajectory_id]
        goal = np.asarray(row["goal_position"], dtype=np.float64)
        states = dataset.states[start:stop].astype(np.float64)
        distance = states[:, -1] * float(np.linalg.norm(MAP_EXTENT))
        positions[start:stop] = goal[None, :] - states[:, 3:6] * distance[:, None]
    return np.clip(positions, 0.0, MAP_EXTENT)


@dataclass(frozen=True)
class GoalRiskFeatureBuilder:
    mode: GoalFeatureMode
    map_extent: tuple[float, float, float] = (4000.0, 4000.0, 400.0)

    @property
    def input_dim(self) -> int:
        return {
            "state7": 7,
            "absolute_position": 10,
            "boundary_distances": 13,
            "compact_position_boundary": 13,
            "compact_decision_context": 12,
            "position_and_boundary": 16,
        }[self.mode]

    def build(self, states: np.ndarray, positions: np.ndarray) -> np.ndarray:
        state = np.asarray(states, dtype=np.float32)
        position = np.asarray(positions, dtype=np.float64)
        if state.ndim != 2 or state.shape[1] != 7:
            raise ValueError("goal-risk base states must be (N, 7)")
        if position.shape != (state.shape[0], 3):
            raise ValueError("positions must align with goal-risk states")
        extent = np.asarray(self.map_extent, dtype=np.float64)
        normalized = np.clip(position / extent[None, :], 0.0, 1.0)
        if self.mode == "state7":
            context = np.empty((state.shape[0], 0), dtype=np.float64)
        elif self.mode == "absolute_position":
            context = normalized
        elif self.mode == "boundary_distances":
            context = np.concatenate([normalized, 1.0 - normalized], axis=1)
        elif self.mode == "compact_position_boundary":
            nearest_by_axis = np.minimum(normalized, 1.0 - normalized)
            context = np.concatenate([normalized, nearest_by_axis], axis=1)
        elif self.mode == "compact_decision_context":
            nearest_by_axis = np.minimum(normalized, 1.0 - normalized)
            minimum_horizontal = np.minimum(nearest_by_axis[:, 0], nearest_by_axis[:, 1])
            minimum_vertical = nearest_by_axis[:, 2]
            context = np.concatenate(
                [
                    normalized,
                    minimum_horizontal[:, None],
                    minimum_vertical[:, None],
                ],
                axis=1,
            )
        elif self.mode == "position_and_boundary":
            boundary = np.concatenate([normalized, 1.0 - normalized], axis=1)
            context = np.concatenate([normalized, boundary], axis=1)
        else:
            raise ValueError(f"unknown goal-risk feature mode: {self.mode}")
        result = np.concatenate([state.astype(np.float64), context], axis=1).astype(np.float32)
        if result.shape[1] != self.input_dim or not np.all(np.isfinite(result)):
            raise RuntimeError("invalid goal-risk features")
        return result

    def build_from_dataset(self, dataset: PackedEnergyDataset) -> np.ndarray:
        return self.build(dataset.states, reconstruct_positions(dataset))

    def build_single(self, state: np.ndarray, position: np.ndarray) -> np.ndarray:
        return self.build(
            np.asarray(state, dtype=np.float32).reshape(1, 7),
            np.asarray(position, dtype=np.float64).reshape(1, 3),
        )[0]

    def as_dict(self) -> dict[str, object]:
        return {"mode": self.mode, "map_extent": list(self.map_extent), "input_dim": self.input_dim}


@dataclass(frozen=True)
class ScaledTrajectoryConformalCorrection:
    coverage_target: float
    multiplier: float
    raw_multiplier: float
    rank: int
    num_trajectories: int

    @classmethod
    def fit(
        cls,
        truth: np.ndarray,
        point: np.ndarray,
        scale: np.ndarray,
        trajectory_ids: np.ndarray,
        *,
        coverage: float,
    ) -> "ScaledTrajectoryConformalCorrection":
        targets = np.asarray(truth, dtype=np.float64)
        center = np.asarray(point, dtype=np.float64)
        sigma = np.maximum(np.asarray(scale, dtype=np.float64), 1e-8)
        ids = np.asarray(trajectory_ids, dtype=np.int64)
        if targets.shape != center.shape or sigma.shape != targets.shape or ids.shape != targets.shape:
            raise ValueError("scaled conformal arrays must align")
        starts = np.concatenate([[0], np.flatnonzero(ids[1:] != ids[:-1]) + 1])
        if starts.size != np.unique(ids).size:
            raise ValueError("each trajectory must occupy one contiguous block")
        scores = np.maximum.reduceat((targets - center) / sigma, starts).astype(np.float64)
        multiplier, raw, rank = finite_sample_conformal_margin(scores, coverage=coverage)
        return cls(float(coverage), float(multiplier), float(raw), int(rank), int(scores.size))

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class HierarchicalScaledConformalCalibration:
    coverage_target: float
    global_multiplier: float
    goal_multipliers: dict[str, float]
    distance_multipliers: dict[str, float]
    intersection_multipliers: dict[str, float]
    intersection_counts: dict[str, int]

    @classmethod
    def fit(
        cls,
        truth: np.ndarray,
        point: np.ndarray,
        scale: np.ndarray,
        trajectory_ids: np.ndarray,
        goal_types: np.ndarray,
        distance_buckets: np.ndarray,
        *,
        coverage: float,
        minimum_group_trajectories: int = 30,
    ) -> "HierarchicalScaledConformalCalibration":
        truth_values = np.asarray(truth, dtype=np.float64)
        point_values = np.asarray(point, dtype=np.float64)
        scale_values = np.maximum(np.asarray(scale, dtype=np.float64), 1e-8)
        ids = np.asarray(trajectory_ids, dtype=np.int64)
        goals = np.asarray(goal_types)
        buckets = np.asarray(distance_buckets)

        def correction(mask: np.ndarray) -> ScaledTrajectoryConformalCorrection | None:
            selected_ids = np.unique(ids[mask])
            if selected_ids.size < minimum_group_trajectories:
                return None
            return ScaledTrajectoryConformalCorrection.fit(
                truth_values[mask],
                point_values[mask],
                scale_values[mask],
                ids[mask],
                coverage=coverage,
            )

        global_fit = ScaledTrajectoryConformalCorrection.fit(
            truth_values,
            point_values,
            scale_values,
            ids,
            coverage=coverage,
        )
        goal_multipliers: dict[str, float] = {}
        distance_multipliers: dict[str, float] = {}
        intersections: dict[str, float] = {}
        counts: dict[str, int] = {}
        for goal in np.unique(goals):
            fit = correction(goals == goal)
            if fit is not None:
                goal_multipliers[str(goal)] = fit.multiplier
        for bucket in np.unique(buckets):
            fit = correction(buckets == bucket)
            if fit is not None:
                distance_multipliers[str(bucket)] = fit.multiplier
        for goal in np.unique(goals):
            for bucket in np.unique(buckets):
                mask = (goals == goal) & (buckets == bucket)
                count = int(np.unique(ids[mask]).size)
                if count == 0:
                    continue
                key = f"{goal}|{bucket}"
                counts[key] = count
                fit = correction(mask)
                if fit is not None:
                    intersections[key] = fit.multiplier
        return cls(
            float(coverage),
            global_fit.multiplier,
            goal_multipliers,
            distance_multipliers,
            intersections,
            counts,
        )

    def multiplier_for(self, goal_type: str, distance: float) -> float:
        bucket = distance_bucket_for(distance)
        intersection = f"{goal_type}|{bucket}"
        if intersection in self.intersection_multipliers:
            return self.intersection_multipliers[intersection]
        if goal_type in self.goal_multipliers:
            return self.goal_multipliers[goal_type]
        if bucket in self.distance_multipliers:
            return self.distance_multipliers[bucket]
        return self.global_multiplier

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class _ResidualQuantileNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return functional.softplus(self.layers(values))


class PositiveResidualQuantileModel:
    def __init__(
        self,
        *,
        input_dim: int,
        quantile_levels: tuple[float, ...] = RESIDUAL_QUANTILE_LEVELS,
        hidden_dim: int = 128,
        learning_rate: float = 3e-4,
        batch_size: int = 1024,
        crossing_weight: float = 0.1,
        seed: int = 0,
        device: str = "cpu",
    ) -> None:
        self.input_dim = int(input_dim)
        self.quantile_levels = tuple(float(value) for value in quantile_levels)
        self.hidden_dim = int(hidden_dim)
        self.learning_rate = float(learning_rate)
        self.batch_size = int(batch_size)
        self.crossing_weight = float(crossing_weight)
        self.seed = int(seed)
        self.device = torch.device(device)
        torch.manual_seed(self.seed)
        self.model = _ResidualQuantileNetwork(
            self.input_dim,
            self.hidden_dim,
            len(self.quantile_levels),
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.quantiles = torch.as_tensor(self.quantile_levels, dtype=torch.float32, device=self.device)
        self.target_scale = 1.0

    def _loss(self, prediction: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
        error = truth[:, None] - prediction
        pinball = torch.maximum((self.quantiles - 1.0) * error, self.quantiles * error).mean()
        crossing = functional.relu(prediction[:, :-1] - prediction[:, 1:]).mean()
        return pinball + self.crossing_weight * crossing

    def fit(
        self,
        train_features: np.ndarray,
        train_positive_residuals: np.ndarray,
        validation_features: np.ndarray,
        validation_positive_residuals: np.ndarray,
        *,
        max_epochs: int = 20,
        patience: int = 4,
    ) -> AdaptiveTrainingHistory:
        train_x = np.asarray(train_features, dtype=np.float32)
        validation_x = np.asarray(validation_features, dtype=np.float32)
        train_y = np.maximum(np.asarray(train_positive_residuals, dtype=np.float32), 0.0)
        validation_y = np.maximum(np.asarray(validation_positive_residuals, dtype=np.float32), 0.0)
        self.target_scale = max(float(np.quantile(train_y, 0.99)), 1e-4)
        train_y = train_y / self.target_scale
        validation_y = validation_y / self.target_scale
        rng = np.random.default_rng(self.seed)
        best_state = None
        best_loss = math.inf
        best_epoch = 0
        stale = 0
        train_history: list[float] = []
        validation_history: list[float] = []
        for epoch in range(1, max_epochs + 1):
            self.model.train()
            order = rng.permutation(train_x.shape[0])
            losses = []
            for offset in range(0, order.size, self.batch_size):
                indices = order[offset : offset + self.batch_size]
                inputs = torch.as_tensor(train_x[indices], device=self.device)
                targets = torch.as_tensor(train_y[indices], device=self.device)
                loss = self._loss(self.model(inputs), targets)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()
                losses.append(float(loss.detach().cpu()))
            validation_loss = self.validation_loss(validation_x, validation_y, normalized=True)
            train_history.append(float(np.mean(losses)))
            validation_history.append(validation_loss)
            if validation_loss < best_loss - 1e-8:
                best_loss = validation_loss
                best_epoch = epoch
                best_state = copy.deepcopy(self.model.state_dict())
                stale = 0
            else:
                stale += 1
            if stale >= patience:
                break
        if best_state is None:
            raise RuntimeError("positive-residual quantile training produced no checkpoint")
        self.model.load_state_dict(best_state)
        self.model.eval()
        return AdaptiveTrainingHistory(
            best_epoch,
            len(train_history),
            best_loss,
            train_history,
            validation_history,
        )

    def validation_loss(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        *,
        normalized: bool = False,
        chunk_size: int = 131_072,
    ) -> float:
        values = np.asarray(features, dtype=np.float32)
        truth = np.asarray(targets, dtype=np.float32)
        if not normalized:
            truth = np.maximum(truth, 0.0) / self.target_scale
        weighted = 0.0
        count = 0
        self.model.eval()
        with torch.no_grad():
            for offset in range(0, values.shape[0], chunk_size):
                inputs = torch.as_tensor(values[offset : offset + chunk_size], device=self.device)
                batch_truth = torch.as_tensor(truth[offset : offset + chunk_size], device=self.device)
                batch_loss = self._loss(self.model(inputs), batch_truth)
                weighted += float(batch_loss.cpu()) * inputs.shape[0]
                count += inputs.shape[0]
        return weighted / count

    def predict_quantiles(self, features: np.ndarray, *, chunk_size: int = 131_072) -> np.ndarray:
        values = np.asarray(features, dtype=np.float32)
        outputs = []
        self.model.eval()
        with torch.no_grad():
            for offset in range(0, values.shape[0], chunk_size):
                inputs = torch.as_tensor(values[offset : offset + chunk_size], device=self.device)
                outputs.append(self.model(inputs).cpu().numpy())
        return np.sort(np.concatenate(outputs).astype(np.float64), axis=1) * self.target_scale

    def predict_level(self, features: np.ndarray, coverage: float) -> np.ndarray:
        index = self.quantile_levels.index(float(coverage))
        return self.predict_quantiles(features)[:, index]

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.model.parameters())

    def checkpoint_payload(self) -> dict[str, object]:
        return {
            "model_type": "positive_residual_quantiles",
            "input_dim": self.input_dim,
            "quantile_levels": self.quantile_levels,
            "hidden_dim": self.hidden_dim,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "crossing_weight": self.crossing_weight,
            "seed": self.seed,
            "target_scale": self.target_scale,
            "model_state_dict": self.model.state_dict(),
        }

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.checkpoint_payload(), destination)

    @classmethod
    def load(cls, path: str | Path, *, device: str = "cpu") -> "PositiveResidualQuantileModel":
        payload = torch.load(Path(path), map_location=device, weights_only=False)
        if payload.get("model_type") != "positive_residual_quantiles":
            raise ValueError("checkpoint is not a positive-residual quantile model")
        model = cls(
            input_dim=int(payload["input_dim"]),
            quantile_levels=tuple(float(value) for value in payload["quantile_levels"]),
            hidden_dim=int(payload["hidden_dim"]),
            learning_rate=float(payload["learning_rate"]),
            batch_size=int(payload["batch_size"]),
            crossing_weight=float(payload["crossing_weight"]),
            seed=int(payload["seed"]),
            device=device,
        )
        model.target_scale = float(payload["target_scale"])
        model.model.load_state_dict(payload["model_state_dict"])
        model.model.eval()
        return model


class SeparatedGoalMissionRiskEstimator:
    estimator_type = "mc_separated_goal_mission_risk"
    bootstrapping = False
    gamma = None

    def __init__(
        self,
        point_estimator: EnergyToGoRegressor,
        goal_feature_builder: GoalRiskFeatureBuilder,
        goal_risk_model: HeteroscedasticResidualModel | PositiveResidualQuantileModel,
        goal_calibration: (
            ScaledTrajectoryConformalCorrection
            | HierarchicalScaledConformalCalibration
            | TrajectoryConformalCorrection
        ),
        mission_risk_model: HeteroscedasticResidualModel,
        mission_correction: TrajectoryConformalCorrection,
        *,
        coverage: float = 0.95,
    ) -> None:
        self.point_estimator = point_estimator
        self.goal_feature_builder = goal_feature_builder
        self.goal_risk_model = goal_risk_model
        self.goal_calibration = goal_calibration
        self.mission_risk_model = mission_risk_model
        self.mission_correction = mission_correction
        self.coverage = float(coverage)
        self.update_count = 0
        self.replay: tuple[()] = ()
        self.trainable_replay: tuple[()] = ()

    def _goal_upper(
        self,
        state: np.ndarray,
        position: np.ndarray,
        *,
        point: float,
        goal_type: str,
    ) -> float:
        features = self.goal_feature_builder.build_single(state, position)[None, :]
        distance = float(np.asarray(state, dtype=np.float64)[-1] * np.linalg.norm(MAP_EXTENT))
        if isinstance(self.goal_risk_model, HeteroscedasticResidualModel):
            scale = float(self.goal_risk_model.predict_scale(features)[0])
            if isinstance(self.goal_calibration, HierarchicalScaledConformalCalibration):
                multiplier = self.goal_calibration.multiplier_for(goal_type, distance)
            elif isinstance(self.goal_calibration, ScaledTrajectoryConformalCorrection):
                multiplier = self.goal_calibration.multiplier
            else:
                raise TypeError("heteroscedastic goal risk requires scaled conformal calibration")
            return float(point + multiplier * scale)
        residual_quantile = float(self.goal_risk_model.predict_level(features, self.coverage)[0])
        if isinstance(self.goal_calibration, HierarchicalScaledConformalCalibration):
            correction = self.goal_calibration.multiplier_for(goal_type, distance)
        elif isinstance(self.goal_calibration, TrajectoryConformalCorrection):
            correction = self.goal_calibration.correction
        else:
            raise TypeError("residual quantile risk requires additive conformal calibration")
        return float(point + residual_quantile + correction)

    def estimate_context(
        self,
        environment,
        goal: np.ndarray,
        *,
        position: np.ndarray | None = None,
        velocity: np.ndarray | None = None,
        goal_type: str | None = None,
    ) -> GoalEnergyPrediction:
        current_position = environment.agent.pos if position is None else np.asarray(position, dtype=np.float32)
        state = environment.energy_state_for_goal(goal, position=position, velocity=velocity)
        point = self.point_estimator.predict(state)
        upper = self._goal_upper(
            state,
            current_position,
            point=point,
            goal_type="TASK" if goal_type is None else goal_type,
        )
        return GoalEnergyPrediction(float(point), float(upper))

    def predict_quantiles_context(
        self,
        environment,
        goal: np.ndarray,
        action: np.ndarray | None = None,
    ) -> np.ndarray:
        del action
        prediction = self.estimate_context(
            environment,
            goal,
            goal_type=("CHARGER" if np.allclose(goal, environment.charger_position) else "TASK"),
        )
        return np.asarray(
            [prediction.prediction, prediction.upper95, prediction.upper95, prediction.upper95],
            dtype=np.float64,
        )

    def estimate_mission_context(self, environment, task_goal: np.ndarray) -> GoalEnergyPrediction:
        task_state = environment.energy_state_for_goal(task_goal)
        return_state = environment.energy_state_for_goal(
            environment.charger_position,
            position=np.asarray(task_goal, dtype=np.float32),
            velocity=np.zeros(3, dtype=np.float32),
        )
        mission_features = np.concatenate([task_state, return_state])[None, :]
        point = self.point_estimator.predict(task_state) + self.point_estimator.predict(return_state)
        upper = (
            point
            + self.mission_risk_model.upper_offset(mission_features, self.coverage)[0]
            + self.mission_correction.correction
        )
        return GoalEnergyPrediction(float(point), float(upper))

    def predict_quantiles(self, energy_state: np.ndarray, action: np.ndarray | None = None) -> np.ndarray:
        del action
        point = self.point_estimator.predict(energy_state)
        return np.asarray([point, point, point, point], dtype=np.float64)

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        calibration_payload = self.goal_calibration.as_dict()
        torch.save(
            {
                "estimator_type": self.estimator_type,
                "coverage": self.coverage,
                "goal_feature_builder": self.goal_feature_builder.as_dict(),
                "goal_risk_model": self.goal_risk_model.checkpoint_payload(),
                "goal_calibration_type": type(self.goal_calibration).__name__,
                "goal_calibration": calibration_payload,
                "mission_risk_model": self.mission_risk_model.checkpoint_payload(),
                "mission_correction": self.mission_correction.as_dict(),
                "point_estimator": self.point_estimator.checkpoint_payload(),
            },
            destination,
        )


__all__ = [
    "GoalRiskFeatureBuilder",
    "GoalFeatureMode",
    "HierarchicalScaledConformalCalibration",
    "MAP_EXTENT",
    "PositiveResidualQuantileModel",
    "RESIDUAL_QUANTILE_LEVELS",
    "ScaledTrajectoryConformalCorrection",
    "SeparatedGoalMissionRiskEstimator",
    "distance_bucket_for",
    "positive_underestimation_target",
    "reconstruct_positions",
    "trajectory_balanced_state_indices",
]
