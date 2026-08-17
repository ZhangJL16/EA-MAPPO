from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import math
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch import nn
from torch.nn import functional

from experiments.energy_mc.adaptive_uncertainty import (
    HeteroscedasticResidualModel,
    trajectory_max_scores,
)
from experiments.energy_mc.conditional_risk import (
    MAP_EXTENT,
    GoalRiskFeatureBuilder,
    PositiveResidualQuantileModel,
    distance_bucket_for,
)
from review_bundle.safety.energy.mc_regression import (
    EnergyToGoRegressor,
    GoalEnergyPrediction,
    finite_sample_conformal_margin,
)


GoalCalibrationMode = Literal["additive", "scaled"]
CHARGER_GOAL_TYPES = frozenset({"CHARGER", "TASK_ENDPOINT_TO_CHARGER"})
NONDEPLOYABLE_FEATURE_PREFIXES = (
    "future_",
    "realized_future_",
    "actual_future_",
    "true_future_",
)


def canonical_goal_type(goal_type: str) -> str:
    value = str(goal_type)
    if value == "TASK":
        return "TASK"
    if value in CHARGER_GOAL_TYPES:
        return "CHARGER"
    raise ValueError(f"unknown Goal type: {value}")


def primary_goal_group(goal_type: str, distance_bucket: str) -> str:
    return f"{canonical_goal_type(goal_type)}|{distance_bucket}"


def assert_decision_time_features(feature_names: list[str] | tuple[str, ...]) -> None:
    invalid = [
        name
        for name in feature_names
        if str(name).startswith(NONDEPLOYABLE_FEATURE_PREFIXES)
        or str(name)
        in {
            "path_ratio",
            "trajectory_steps",
            "boundary_contact",
            "acceleration_statistics",
        }
    ]
    if invalid:
        raise ValueError(f"post-hoc features are not deployable: {invalid}")


def assert_fresh_test_isolation(
    fresh_ids: set[int],
    development_splits: dict[str, set[int]],
) -> None:
    for name, identifiers in development_splits.items():
        overlap = fresh_ids & identifiers
        if overlap:
            raise ValueError(
                f"fresh-test leakage from {name}: {sorted(overlap)[:5]}"
            )


def unnecessary_return_indicator(
    remaining_energy: np.ndarray,
    true_required_energy: np.ndarray,
    predicted_upper: np.ndarray,
    reserve: float,
) -> np.ndarray:
    remaining = np.asarray(remaining_energy, dtype=np.float64)
    truth = np.asarray(true_required_energy, dtype=np.float64)
    upper = np.asarray(predicted_upper, dtype=np.float64)
    if not (remaining.shape == truth.shape == upper.shape):
        raise ValueError("unnecessary-return arrays must align")
    if reserve < 0.0:
        raise ValueError("reserve must be nonnegative")
    return (remaining >= truth + reserve) & (remaining < upper + reserve)


SwitchCause = Literal["point_estimate", "uncertainty_margin", "reserve", "both"]


@dataclass(frozen=True)
class SwitchAttribution:
    cause: SwitchCause
    full_trigger: bool
    point_only_trigger: bool
    without_uncertainty_trigger: bool
    without_reserve_trigger: bool
    remaining_energy: float
    point_estimate: float
    uncertainty_margin: float
    reserve: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def attribute_switch(
    *,
    remaining_energy: float,
    point_estimate: float,
    uncertainty_margin: float,
    reserve: float,
) -> SwitchAttribution:
    values = np.asarray(
        [remaining_energy, point_estimate, uncertainty_margin, reserve], dtype=np.float64
    )
    if not np.all(np.isfinite(values)) or np.any(values[1:] < 0.0):
        raise ValueError("switch inputs must be finite and margins nonnegative")
    full = bool(remaining_energy <= point_estimate + uncertainty_margin + reserve)
    if not full:
        raise ValueError("switch attribution requires an observed full trigger")
    point_only = bool(remaining_energy <= point_estimate)
    without_uncertainty = bool(remaining_energy <= point_estimate + reserve)
    without_reserve = bool(remaining_energy <= point_estimate + uncertainty_margin)
    if point_only:
        cause: SwitchCause = "point_estimate"
    elif without_uncertainty and not without_reserve:
        cause = "reserve"
    elif without_reserve and not without_uncertainty:
        cause = "uncertainty_margin"
    else:
        cause = "both"
    return SwitchAttribution(
        cause,
        full,
        point_only,
        without_uncertainty,
        without_reserve,
        float(remaining_energy),
        float(point_estimate),
        float(uncertainty_margin),
        float(reserve),
    )


class _ContextualEnergyNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return functional.softplus(self.layers(values)).squeeze(-1)


class ContextualEnergyRegressor:
    """Small controlled-ablation regressor; not a new deployment architecture."""

    def __init__(
        self,
        *,
        input_dim: int,
        battery_capacity: float,
        hidden_dim: int = 128,
        learning_rate: float = 3e-4,
        batch_size: int = 1024,
        seed: int = 0,
        device: str = "cpu",
    ) -> None:
        if input_dim <= 0 or battery_capacity <= 0.0:
            raise ValueError("contextual point model dimensions and capacity must be positive")
        self.input_dim = int(input_dim)
        self.battery_capacity = float(battery_capacity)
        self.hidden_dim = int(hidden_dim)
        self.learning_rate = float(learning_rate)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.device = torch.device(device)
        torch.manual_seed(self.seed)
        self.model = _ContextualEnergyNetwork(self.input_dim, self.hidden_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def initialize_from_frozen_7d(self, frozen_model: object) -> None:
        source = getattr(frozen_model, "model", None)
        source_layers = getattr(source, "backbone", None)
        if source_layers is None or self.input_dim < 7:
            raise ValueError("a compatible frozen 7D EnergyToGo model is required")
        with torch.no_grad():
            self.model.layers[0].weight.zero_()
            self.model.layers[0].weight[:, :7].copy_(source_layers[0].weight)
            self.model.layers[0].bias.copy_(source_layers[0].bias)
            self.model.layers[2].weight.copy_(source_layers[2].weight)
            self.model.layers[2].bias.copy_(source_layers[2].bias)
            self.model.layers[4].weight.copy_(source_layers[4].weight)
            self.model.layers[4].bias.copy_(source_layers[4].bias)

    def predict_batch(self, features: np.ndarray, *, chunk_size: int = 131_072) -> np.ndarray:
        values = np.asarray(features, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != self.input_dim:
            raise ValueError("contextual point features have the wrong shape")
        outputs = []
        self.model.eval()
        with torch.no_grad():
            for offset in range(0, values.shape[0], chunk_size):
                inputs = torch.as_tensor(
                    values[offset : offset + chunk_size],
                    dtype=torch.float32,
                    device=self.device,
                )
                outputs.append(self.model(inputs).cpu().numpy())
        return np.concatenate(outputs).astype(np.float64) * self.battery_capacity

    def fit(
        self,
        train_features: np.ndarray,
        train_targets: np.ndarray,
        validation_features: np.ndarray,
        validation_targets: np.ndarray,
        *,
        max_epochs: int = 20,
        patience: int = 4,
    ) -> dict[str, object]:
        train_x = np.asarray(train_features, dtype=np.float32)
        validation_x = np.asarray(validation_features, dtype=np.float32)
        train_y = np.asarray(train_targets, dtype=np.float32) / self.battery_capacity
        validation_y = np.asarray(validation_targets, dtype=np.float64)
        if train_x.ndim != 2 or train_x.shape[1] != self.input_dim:
            raise ValueError("contextual training features have the wrong shape")
        if validation_x.ndim != 2 or validation_x.shape[1] != self.input_dim:
            raise ValueError("contextual validation features have the wrong shape")
        if train_y.shape != (train_x.shape[0],) or validation_y.shape != (
            validation_x.shape[0],
        ):
            raise ValueError("contextual point targets do not align")
        rng = np.random.default_rng(self.seed)
        initial_prediction = self.predict_batch(validation_x)
        best_mae = float(np.mean(np.abs(initial_prediction - validation_y)))
        best_state = copy.deepcopy(self.model.state_dict())
        best_epoch = 0
        stale = 0
        history: list[dict[str, float | int | None]] = [
            {"epoch": 0, "train_loss": None, "validation_mae": best_mae}
        ]
        for epoch in range(1, max_epochs + 1):
            self.model.train()
            order = rng.permutation(train_x.shape[0])
            losses = []
            for offset in range(0, order.size, self.batch_size):
                indices = order[offset : offset + self.batch_size]
                inputs = torch.as_tensor(train_x[indices], device=self.device)
                truth = torch.as_tensor(train_y[indices], device=self.device)
                loss = functional.huber_loss(self.model(inputs), truth)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()
                losses.append(float(loss.detach().cpu()))
            validation_prediction = self.predict_batch(validation_x)
            validation_mae = float(np.mean(np.abs(validation_prediction - validation_y)))
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": float(np.mean(losses)),
                    "validation_mae": validation_mae,
                }
            )
            if validation_mae < best_mae - 1e-8:
                best_mae = validation_mae
                best_epoch = epoch
                best_state = copy.deepcopy(self.model.state_dict())
                stale = 0
            else:
                stale += 1
            if stale >= patience:
                break
        self.model.load_state_dict(best_state)
        self.model.eval()
        return {
            "best_epoch": best_epoch,
            "best_validation_mae": best_mae,
            "epochs_completed": len(history) - 1,
            "history": history,
        }

    def save(self, path: str | Path, *, feature_mode: str) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_type": "controlled_contextual_point_ablation",
                "feature_mode": feature_mode,
                "input_dim": self.input_dim,
                "battery_capacity": self.battery_capacity,
                "hidden_dim": self.hidden_dim,
                "learning_rate": self.learning_rate,
                "batch_size": self.batch_size,
                "seed": self.seed,
                "model_state_dict": self.model.state_dict(),
            },
            destination,
        )

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.model.parameters())


def future_max_underestimation_target(
    truth: np.ndarray,
    point: np.ndarray,
    trajectory_ids: np.ndarray,
) -> np.ndarray:
    targets = np.asarray(truth, dtype=np.float64)
    predictions = np.asarray(point, dtype=np.float64)
    identifiers = np.asarray(trajectory_ids, dtype=np.int64)
    if targets.shape != predictions.shape or identifiers.shape != targets.shape:
        raise ValueError("future-max arrays must align")
    starts = np.concatenate([[0], np.flatnonzero(identifiers[1:] != identifiers[:-1]) + 1])
    stops = np.concatenate([starts[1:], [identifiers.size]])
    if starts.size != np.unique(identifiers).size:
        raise ValueError("each trajectory must occupy one contiguous block")
    residual = np.maximum(targets - predictions, 0.0)
    result = np.empty_like(residual)
    for start, stop in zip(starts, stops, strict=True):
        result[start:stop] = np.maximum.accumulate(residual[start:stop][::-1])[::-1]
    return result


@dataclass(frozen=True)
class MondrianTrajectoryCalibration:
    coverage_target: float
    mode: GoalCalibrationMode
    group_values: dict[str, float]
    group_raw_values: dict[str, float]
    group_ranks: dict[str, int]
    group_counts: dict[str, int]
    global_value: float
    global_raw_value: float
    global_rank: int
    global_count: int
    minimum_group_trajectories: int

    @classmethod
    def fit(
        cls,
        truth: np.ndarray,
        point: np.ndarray,
        risk: np.ndarray,
        trajectory_ids: np.ndarray,
        goal_types: np.ndarray,
        distance_buckets: np.ndarray,
        *,
        coverage: float,
        mode: GoalCalibrationMode,
        minimum_group_trajectories: int = 100,
    ) -> "MondrianTrajectoryCalibration":
        targets = np.asarray(truth, dtype=np.float64)
        centers = np.asarray(point, dtype=np.float64)
        risk_values = np.asarray(risk, dtype=np.float64)
        ids = np.asarray(trajectory_ids, dtype=np.int64)
        goals = np.asarray(goal_types).astype("U32")
        buckets = np.asarray(distance_buckets).astype("U16")
        if not (
            targets.shape
            == centers.shape
            == risk_values.shape
            == ids.shape
            == goals.shape
            == buckets.shape
        ):
            raise ValueError("Mondrian calibration arrays must align")
        if mode == "additive":
            state_scores = targets - centers - risk_values
        elif mode == "scaled":
            state_scores = (targets - centers) / np.maximum(risk_values, 1e-8)
        else:
            raise ValueError("mode must be additive or scaled")
        _, global_scores = trajectory_max_scores(state_scores, np.zeros_like(state_scores), ids)
        global_value, global_raw, global_rank = finite_sample_conformal_margin(
            global_scores,
            coverage=coverage,
        )
        group_values: dict[str, float] = {}
        group_raw: dict[str, float] = {}
        group_ranks: dict[str, int] = {}
        group_counts: dict[str, int] = {}
        canonical = np.asarray([canonical_goal_type(value) for value in goals], dtype="U8")
        group_labels = np.char.add(np.char.add(canonical, "|"), buckets)
        for group in sorted(np.unique(group_labels)):
            mask = group_labels == group
            selected_ids = np.unique(ids[mask])
            if selected_ids.size < minimum_group_trajectories:
                continue
            _, scores = trajectory_max_scores(
                state_scores[mask],
                np.zeros(np.sum(mask), dtype=np.float64),
                ids[mask],
            )
            value, raw, rank = finite_sample_conformal_margin(scores, coverage=coverage)
            group_values[str(group)] = float(value)
            group_raw[str(group)] = float(raw)
            group_ranks[str(group)] = int(rank)
            group_counts[str(group)] = int(scores.size)
        return cls(
            coverage_target=float(coverage),
            mode=mode,
            group_values=group_values,
            group_raw_values=group_raw,
            group_ranks=group_ranks,
            group_counts=group_counts,
            global_value=float(global_value),
            global_raw_value=float(global_raw),
            global_rank=int(global_rank),
            global_count=int(global_scores.size),
            minimum_group_trajectories=int(minimum_group_trajectories),
        )

    def value_for(self, goal_type: str, distance: float) -> tuple[float, bool, str]:
        group = primary_goal_group(goal_type, distance_bucket_for(distance))
        if group in self.group_values:
            return self.group_values[group], True, group
        return self.global_value, False, group

    def apply(
        self,
        point: np.ndarray,
        risk: np.ndarray,
        goal_types: np.ndarray,
        distances: np.ndarray,
    ) -> np.ndarray:
        centers = np.asarray(point, dtype=np.float64)
        risk_values = np.asarray(risk, dtype=np.float64)
        goals = np.asarray(goal_types)
        distance_values = np.asarray(distances, dtype=np.float64)
        if not (centers.shape == risk_values.shape == goals.shape == distance_values.shape):
            raise ValueError("Mondrian inference arrays must align")
        calibration = np.asarray(
            [self.value_for(goal, distance)[0] for goal, distance in zip(goals, distance_values)],
            dtype=np.float64,
        )
        if self.mode == "additive":
            return centers + np.maximum(risk_values + calibration, 0.0)
        return centers + np.maximum(calibration, 0.0) * np.maximum(risk_values, 1e-8)

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class MondrianMissionCalibration:
    coverage_target: float
    distance_corrections: dict[str, float]
    distance_raw_corrections: dict[str, float]
    distance_ranks: dict[str, int]
    distance_counts: dict[str, int]
    minimum_group_missions: int

    @classmethod
    def fit(
        cls,
        truth: np.ndarray,
        base_upper: np.ndarray,
        mission_ids: np.ndarray,
        distance_buckets: np.ndarray,
        *,
        coverage: float,
        minimum_group_missions: int = 100,
    ) -> "MondrianMissionCalibration":
        targets = np.asarray(truth, dtype=np.float64)
        upper = np.asarray(base_upper, dtype=np.float64)
        ids = np.asarray(mission_ids, dtype=np.int64)
        buckets = np.asarray(distance_buckets).astype("U16")
        if not (targets.shape == upper.shape == ids.shape == buckets.shape):
            raise ValueError("Mission Mondrian arrays must align")
        corrections: dict[str, float] = {}
        raw_corrections: dict[str, float] = {}
        ranks: dict[str, int] = {}
        counts: dict[str, int] = {}
        for bucket in sorted(np.unique(buckets)):
            mask = buckets == bucket
            selected_ids = np.unique(ids[mask])
            if selected_ids.size < minimum_group_missions:
                continue
            _, scores = trajectory_max_scores(targets[mask], upper[mask], ids[mask])
            correction, raw, rank = finite_sample_conformal_margin(scores, coverage=coverage)
            corrections[str(bucket)] = float(correction)
            raw_corrections[str(bucket)] = float(raw)
            ranks[str(bucket)] = int(rank)
            counts[str(bucket)] = int(scores.size)
        return cls(
            float(coverage),
            corrections,
            raw_corrections,
            ranks,
            counts,
            int(minimum_group_missions),
        )

    def apply(self, base_upper: np.ndarray, distance_buckets: np.ndarray) -> np.ndarray:
        upper = np.asarray(base_upper, dtype=np.float64)
        buckets = np.asarray(distance_buckets).astype("U16")
        if upper.shape != buckets.shape:
            raise ValueError("Mission Mondrian inference arrays must align")
        missing = sorted(set(str(value) for value in np.unique(buckets)) - self.distance_corrections.keys())
        if missing:
            raise ValueError(f"Mission primary groups lack calibration: {missing}")
        return upper + np.asarray([self.distance_corrections[str(value)] for value in buckets])

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class MondrianGoalMissionRiskEstimator:
    """Frozen point plus adaptive Goal risk and direct Mondrian calibration."""

    estimator_type = "mc_mondrian_goal_mission_risk_v5"
    bootstrapping = False
    gamma = None

    def __init__(
        self,
        point_estimator: EnergyToGoRegressor,
        goal_feature_builder: GoalRiskFeatureBuilder,
        goal_risk_model: PositiveResidualQuantileModel,
        goal_calibration: MondrianTrajectoryCalibration,
        mission_risk_model: HeteroscedasticResidualModel,
        mission_calibration: MondrianMissionCalibration,
        *,
        goal_coverage: float,
        mission_coverage: float,
    ) -> None:
        if goal_calibration.mode != "additive":
            raise ValueError("positive residual Goal risk requires additive calibration")
        self.point_estimator = point_estimator
        self.goal_feature_builder = goal_feature_builder
        self.goal_risk_model = goal_risk_model
        self.goal_calibration = goal_calibration
        self.mission_risk_model = mission_risk_model
        self.mission_calibration = mission_calibration
        self.goal_coverage = float(goal_coverage)
        self.mission_coverage = float(mission_coverage)
        self.update_count = 0
        self.replay: tuple[()] = ()
        self.trainable_replay: tuple[()] = ()

    def _goal_prediction(
        self,
        environment,
        goal: np.ndarray,
        *,
        position: np.ndarray | None,
        velocity: np.ndarray | None,
        goal_type: str,
    ) -> GoalEnergyPrediction:
        current_position = (
            np.asarray(environment.agent.pos, dtype=np.float32)
            if position is None
            else np.asarray(position, dtype=np.float32)
        )
        state = environment.energy_state_for_goal(
            goal,
            position=position,
            velocity=velocity,
        )
        point = float(self.point_estimator.predict(state))
        features = self.goal_feature_builder.build_single(state, current_position)[None, :]
        risk = float(
            self.goal_risk_model.predict_level(features, self.goal_coverage)[0]
        )
        distance = float(np.asarray(state, dtype=np.float64)[-1] * np.linalg.norm(MAP_EXTENT))
        upper = float(
            self.goal_calibration.apply(
                np.asarray([point]),
                np.asarray([risk]),
                np.asarray([goal_type]),
                np.asarray([distance]),
            )[0]
        )
        return GoalEnergyPrediction(point, upper)

    def estimate_context(
        self,
        environment,
        goal: np.ndarray,
        *,
        position: np.ndarray | None = None,
        velocity: np.ndarray | None = None,
        goal_type: str | None = None,
    ) -> GoalEnergyPrediction:
        resolved_type = (
            "CHARGER"
            if goal_type is None and np.allclose(goal, environment.charger_position)
            else "TASK" if goal_type is None else str(goal_type)
        )
        return self._goal_prediction(
            environment,
            np.asarray(goal, dtype=np.float32),
            position=position,
            velocity=velocity,
            goal_type=resolved_type,
        )

    def estimate_mission_context(
        self,
        environment,
        task_goal: np.ndarray,
    ) -> GoalEnergyPrediction:
        task_state = environment.energy_state_for_goal(task_goal)
        return_state = environment.energy_state_for_goal(
            environment.charger_position,
            position=np.asarray(task_goal, dtype=np.float32),
            velocity=np.zeros(3, dtype=np.float32),
        )
        point = float(
            self.point_estimator.predict(task_state)
            + self.point_estimator.predict(return_state)
        )
        features = np.concatenate([task_state, return_state])[None, :]
        risk = float(
            self.mission_risk_model.upper_offset(features, self.mission_coverage)[0]
        )
        bucket = distance_bucket_for(
            float(task_state[-1] * np.linalg.norm(MAP_EXTENT))
        )
        upper = float(
            self.mission_calibration.apply(
                np.asarray([point + risk]),
                np.asarray([bucket]),
            )[0]
        )
        return GoalEnergyPrediction(point, upper)

    def predict_quantiles_context(
        self,
        environment,
        goal: np.ndarray,
        action: np.ndarray | None = None,
    ) -> np.ndarray:
        del action
        prediction = self.estimate_context(environment, goal)
        return np.asarray(
            [
                prediction.prediction,
                prediction.upper95,
                prediction.upper95,
                prediction.upper95,
            ],
            dtype=np.float64,
        )

    def predict_quantiles(
        self,
        energy_state: np.ndarray,
        action: np.ndarray | None = None,
    ) -> np.ndarray:
        del action
        point = float(self.point_estimator.predict(energy_state))
        return np.asarray([point, point, point, point], dtype=np.float64)

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "estimator_type": self.estimator_type,
                "goal_coverage": self.goal_coverage,
                "mission_coverage": self.mission_coverage,
                "goal_feature_builder": self.goal_feature_builder.as_dict(),
                "goal_risk_model": self.goal_risk_model.checkpoint_payload(),
                "goal_calibration": self.goal_calibration.as_dict(),
                "mission_risk_model": self.mission_risk_model.checkpoint_payload(),
                "mission_calibration": self.mission_calibration.as_dict(),
                "point_estimator": self.point_estimator.checkpoint_payload(),
            },
            destination,
        )


__all__ = [
    "MondrianMissionCalibration",
    "MondrianTrajectoryCalibration",
    "MondrianGoalMissionRiskEstimator",
    "ContextualEnergyRegressor",
    "assert_decision_time_features",
    "assert_fresh_test_isolation",
    "attribute_switch",
    "canonical_goal_type",
    "future_max_underestimation_target",
    "primary_goal_group",
    "SwitchAttribution",
    "unnecessary_return_indicator",
]
