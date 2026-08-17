from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from scipy.stats import norm
from torch import nn
from torch.nn import functional

from review_bundle.safety.energy.mc_regression import (
    EnergyToGoRegressor,
    GoalEnergyPrediction,
    finite_sample_conformal_margin,
)


RISK_COVERAGE_LEVELS = (0.90, 0.95, 0.975, 0.99)
QUANTILE_LEVELS = (0.50, 0.70, 0.80, 0.90, 0.95, 0.975, 0.99)


def chunked_point_predictions(
    estimator: EnergyToGoRegressor,
    states: np.ndarray,
    *,
    chunk_size: int = 131_072,
) -> np.ndarray:
    values = np.asarray(states, dtype=np.float32)
    return np.concatenate(
        [
            estimator.predict_batch(values[offset : offset + chunk_size])
            for offset in range(0, values.shape[0], chunk_size)
        ]
    )


def trajectory_max_scores(
    truth: np.ndarray,
    base_upper: np.ndarray,
    trajectory_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    targets = np.asarray(truth, dtype=np.float64)
    upper = np.asarray(base_upper, dtype=np.float64)
    ids = np.asarray(trajectory_ids, dtype=np.int64)
    if targets.shape != upper.shape or ids.shape != targets.shape:
        raise ValueError("trajectory score arrays must align")
    starts = np.concatenate([[0], np.flatnonzero(ids[1:] != ids[:-1]) + 1])
    unique_ids = ids[starts]
    if unique_ids.size != np.unique(ids).size:
        raise ValueError("each trajectory must occupy one contiguous block")
    scores = np.maximum.reduceat(targets - upper, starts).astype(np.float64)
    return unique_ids, scores


@dataclass(frozen=True)
class TrajectoryConformalCorrection:
    coverage_target: float
    correction: float
    raw_correction: float
    rank: int
    num_trajectories: int

    @classmethod
    def fit(
        cls,
        truth: np.ndarray,
        base_upper: np.ndarray,
        trajectory_ids: np.ndarray,
        *,
        coverage: float,
        allow_negative: bool = False,
    ) -> "TrajectoryConformalCorrection":
        _, scores = trajectory_max_scores(truth, base_upper, trajectory_ids)
        correction, raw, rank = finite_sample_conformal_margin(scores, coverage=coverage)
        if allow_negative:
            correction = raw
        return cls(float(coverage), correction, raw, rank, int(scores.size))

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class ResidualScaleNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128) -> None:
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
        return functional.softplus(self.backbone(values)).squeeze(-1) + 1e-4


class SupervisedQuantileNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        quantile_levels: tuple[float, ...] = QUANTILE_LEVELS,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.quantile_levels = tuple(float(level) for level in quantile_levels)
        self.backbone = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, len(self.quantile_levels)),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.backbone(values)


@dataclass(frozen=True)
class AdaptiveTrainingHistory:
    best_epoch: int
    epochs_completed: int
    best_validation_loss: float
    train_loss: list[float]
    validation_loss: list[float]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class HeteroscedasticResidualModel:
    def __init__(
        self,
        *,
        input_dim: int = 7,
        hidden_dim: int = 128,
        distribution: Literal["gaussian", "laplace"] = "laplace",
        learning_rate: float = 3e-4,
        batch_size: int = 1024,
        seed: int = 0,
        device: str = "cpu",
    ) -> None:
        if distribution not in ("gaussian", "laplace"):
            raise ValueError("distribution must be gaussian or laplace")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.distribution = distribution
        self.learning_rate = float(learning_rate)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.device = torch.device(device)
        torch.manual_seed(seed)
        self.model = ResidualScaleNetwork(self.input_dim, self.hidden_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.residual_scale = 1.0

    def _loss(self, predicted_scale: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        if self.distribution == "gaussian":
            return torch.mean(torch.log(predicted_scale) + 0.5 * (residual / predicted_scale) ** 2)
        return torch.mean(torch.log(predicted_scale) + torch.abs(residual) / predicted_scale)

    def fit(
        self,
        train_states: np.ndarray,
        train_residuals: np.ndarray,
        validation_states: np.ndarray,
        validation_residuals: np.ndarray,
        *,
        max_epochs: int = 20,
        patience: int = 4,
    ) -> AdaptiveTrainingHistory:
        train_x = np.asarray(train_states, dtype=np.float32)
        validation_x = np.asarray(validation_states, dtype=np.float32)
        train_r = np.asarray(train_residuals, dtype=np.float32)
        validation_r = np.asarray(validation_residuals, dtype=np.float32)
        self.residual_scale = max(float(np.std(train_r)), 1e-4)
        train_r = train_r / self.residual_scale
        validation_r = validation_r / self.residual_scale
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
                residual = torch.as_tensor(train_r[indices], device=self.device)
                loss = self._loss(self.model(inputs), residual)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()
                losses.append(float(loss.detach().cpu()))
            validation_loss = self.validation_loss(validation_x, validation_r, normalized=True)
            train_history.append(float(np.mean(losses)))
            validation_history.append(validation_loss)
            if validation_loss < best_loss - 1e-7:
                best_loss = validation_loss
                best_epoch = epoch
                best_state = copy.deepcopy(self.model.state_dict())
                stale = 0
            else:
                stale += 1
            if stale >= patience:
                break
        if best_state is None:
            raise RuntimeError("heteroscedastic training produced no checkpoint")
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
        states: np.ndarray,
        residuals: np.ndarray,
        *,
        normalized: bool = False,
        chunk_size: int = 131_072,
    ) -> float:
        values = np.asarray(states, dtype=np.float32)
        targets = np.asarray(residuals, dtype=np.float32)
        if not normalized:
            targets = targets / self.residual_scale
        weighted_loss = 0.0
        count = 0
        self.model.eval()
        with torch.no_grad():
            for offset in range(0, values.shape[0], chunk_size):
                inputs = torch.as_tensor(values[offset : offset + chunk_size], device=self.device)
                residual = torch.as_tensor(targets[offset : offset + chunk_size], device=self.device)
                loss = self._loss(self.model(inputs), residual)
                batch_count = inputs.shape[0]
                weighted_loss += float(loss.cpu()) * batch_count
                count += batch_count
        return weighted_loss / count

    def predict_scale(self, states: np.ndarray, *, chunk_size: int = 131_072) -> np.ndarray:
        values = np.asarray(states, dtype=np.float32)
        outputs = []
        self.model.eval()
        with torch.no_grad():
            for offset in range(0, values.shape[0], chunk_size):
                inputs = torch.as_tensor(values[offset : offset + chunk_size], device=self.device)
                outputs.append(self.model(inputs).cpu().numpy())
        return np.concatenate(outputs).astype(np.float64) * self.residual_scale

    def upper_offset(self, states: np.ndarray, coverage: float) -> np.ndarray:
        scale = self.predict_scale(states)
        if self.distribution == "gaussian":
            multiplier = float(norm.ppf(coverage))
        else:
            multiplier = -math.log(2.0 * (1.0 - coverage))
        return scale * multiplier

    def checkpoint_payload(self) -> dict[str, object]:
        return {
            "model_type": "heteroscedastic_residual_scale",
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "distribution": self.distribution,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "seed": self.seed,
            "residual_scale": self.residual_scale,
            "model_state_dict": self.model.state_dict(),
        }

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.checkpoint_payload(), destination)

    @classmethod
    def load(cls, path: str | Path, *, device: str = "cpu") -> "HeteroscedasticResidualModel":
        payload = torch.load(Path(path), map_location=device, weights_only=False)
        if payload.get("model_type") != "heteroscedastic_residual_scale":
            raise ValueError("checkpoint is not a heteroscedastic residual model")
        model = cls(
            input_dim=int(payload["input_dim"]),
            hidden_dim=int(payload["hidden_dim"]),
            distribution=str(payload["distribution"]),
            learning_rate=float(payload["learning_rate"]),
            batch_size=int(payload["batch_size"]),
            seed=int(payload["seed"]),
            device=device,
        )
        model.residual_scale = float(payload["residual_scale"])
        model.model.load_state_dict(payload["model_state_dict"])
        model.model.eval()
        return model


class MCSupervisedQuantileModel:
    def __init__(
        self,
        *,
        battery_capacity: float,
        input_dim: int = 7,
        quantile_levels: tuple[float, ...] = QUANTILE_LEVELS,
        hidden_dim: int = 128,
        learning_rate: float = 3e-4,
        batch_size: int = 1024,
        crossing_weight: float = 0.1,
        seed: int = 0,
        device: str = "cpu",
    ) -> None:
        self.battery_capacity = float(battery_capacity)
        self.input_dim = int(input_dim)
        self.quantile_levels = tuple(float(level) for level in quantile_levels)
        self.hidden_dim = int(hidden_dim)
        self.learning_rate = float(learning_rate)
        self.batch_size = int(batch_size)
        self.crossing_weight = float(crossing_weight)
        self.seed = int(seed)
        self.device = torch.device(device)
        torch.manual_seed(seed)
        self.model = SupervisedQuantileNetwork(
            self.input_dim,
            self.quantile_levels,
            self.hidden_dim,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.quantiles = torch.as_tensor(self.quantile_levels, dtype=torch.float32, device=self.device)

    def _loss(self, prediction: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
        error = truth[:, None] - prediction
        pinball = torch.maximum((self.quantiles - 1.0) * error, self.quantiles * error).mean()
        crossing = functional.relu(prediction[:, :-1] - prediction[:, 1:]).mean()
        return pinball + self.crossing_weight * crossing

    def fit(
        self,
        train_states: np.ndarray,
        train_targets: np.ndarray,
        validation_states: np.ndarray,
        validation_targets: np.ndarray,
        *,
        max_epochs: int = 25,
        patience: int = 5,
    ) -> AdaptiveTrainingHistory:
        train_x = np.asarray(train_states, dtype=np.float32)
        validation_x = np.asarray(validation_states, dtype=np.float32)
        train_y = np.asarray(train_targets, dtype=np.float32) / self.battery_capacity
        validation_y = np.asarray(validation_targets, dtype=np.float32) / self.battery_capacity
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
                truth = torch.as_tensor(train_y[indices], device=self.device)
                loss = self._loss(self.model(inputs), truth)
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
            raise RuntimeError("quantile training produced no checkpoint")
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
        states: np.ndarray,
        targets: np.ndarray,
        *,
        normalized: bool = False,
        chunk_size: int = 131_072,
    ) -> float:
        values = np.asarray(states, dtype=np.float32)
        truth = np.asarray(targets, dtype=np.float32)
        if not normalized:
            truth = truth / self.battery_capacity
        weighted_loss = 0.0
        count = 0
        self.model.eval()
        with torch.no_grad():
            for offset in range(0, values.shape[0], chunk_size):
                inputs = torch.as_tensor(values[offset : offset + chunk_size], device=self.device)
                batch_truth = torch.as_tensor(truth[offset : offset + chunk_size], device=self.device)
                loss = self._loss(self.model(inputs), batch_truth)
                batch_count = inputs.shape[0]
                weighted_loss += float(loss.cpu()) * batch_count
                count += batch_count
        return weighted_loss / count

    def predict_quantiles(self, states: np.ndarray, *, chunk_size: int = 131_072) -> np.ndarray:
        values = np.asarray(states, dtype=np.float32)
        raw = []
        self.model.eval()
        with torch.no_grad():
            for offset in range(0, values.shape[0], chunk_size):
                inputs = torch.as_tensor(values[offset : offset + chunk_size], device=self.device)
                raw.append(self.model(inputs).cpu().numpy())
        predictions = np.concatenate(raw).astype(np.float64) * self.battery_capacity
        return np.maximum(0.0, np.sort(predictions, axis=1))

    def raw_crossing_rate(self, states: np.ndarray, *, chunk_size: int = 131_072) -> float:
        values = np.asarray(states, dtype=np.float32)
        crossed = 0
        count = 0
        self.model.eval()
        with torch.no_grad():
            for offset in range(0, values.shape[0], chunk_size):
                inputs = torch.as_tensor(values[offset : offset + chunk_size], device=self.device)
                predictions = self.model(inputs).cpu().numpy()
                crossed += int(np.sum(np.any(predictions[:, :-1] > predictions[:, 1:], axis=1)))
                count += predictions.shape[0]
        return float(crossed / count)

    def predict_level(self, states: np.ndarray, coverage: float) -> np.ndarray:
        try:
            index = self.quantile_levels.index(float(coverage))
        except ValueError as error:
            raise ValueError(f"coverage {coverage} was not trained") from error
        return self.predict_quantiles(states)[:, index]

    def checkpoint_payload(self) -> dict[str, object]:
        return {
            "model_type": "mc_supervised_independent_quantiles",
            "battery_capacity": self.battery_capacity,
            "input_dim": self.input_dim,
            "quantile_levels": self.quantile_levels,
            "hidden_dim": self.hidden_dim,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "crossing_weight": self.crossing_weight,
            "seed": self.seed,
            "model_state_dict": self.model.state_dict(),
        }

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.checkpoint_payload(), destination)

    @classmethod
    def load(cls, path: str | Path, *, device: str = "cpu") -> "MCSupervisedQuantileModel":
        payload = torch.load(Path(path), map_location=device, weights_only=False)
        if payload.get("model_type") != "mc_supervised_independent_quantiles":
            raise ValueError("checkpoint is not an MC-supervised quantile model")
        model = cls(
            battery_capacity=float(payload["battery_capacity"]),
            input_dim=int(payload["input_dim"]),
            quantile_levels=tuple(float(value) for value in payload["quantile_levels"]),
            hidden_dim=int(payload["hidden_dim"]),
            learning_rate=float(payload["learning_rate"]),
            batch_size=int(payload["batch_size"]),
            crossing_weight=float(payload["crossing_weight"]),
            seed=int(payload["seed"]),
            device=device,
        )
        model.model.load_state_dict(payload["model_state_dict"])
        model.model.eval()
        return model


class AdaptiveConformalEnergyEstimator:
    estimator_type = "mc_adaptive_conformal_energy_to_go"
    bootstrapping = False
    gamma = None

    def __init__(
        self,
        point_estimator: EnergyToGoRegressor,
        goal_model: HeteroscedasticResidualModel | MCSupervisedQuantileModel,
        goal_correction: TrajectoryConformalCorrection,
        mission_model: HeteroscedasticResidualModel | MCSupervisedQuantileModel,
        mission_correction: TrajectoryConformalCorrection,
        *,
        coverage: float = 0.95,
    ) -> None:
        self.point_estimator = point_estimator
        self.goal_model = goal_model
        self.goal_correction = goal_correction
        self.mission_model = mission_model
        self.mission_correction = mission_correction
        self.coverage = float(coverage)
        self.update_count = 0
        self.replay: tuple[()] = ()
        self.trainable_replay: tuple[()] = ()

    def _goal_base_upper(self, state: np.ndarray, point: float) -> float:
        values = np.asarray(state, dtype=np.float32)[None, :]
        if isinstance(self.goal_model, HeteroscedasticResidualModel):
            return float(point + self.goal_model.upper_offset(values, self.coverage)[0])
        return float(self.goal_model.predict_level(values, self.coverage)[0])

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
        point = self.point_estimator.predict(state)
        upper = self._goal_base_upper(state, point) + self.goal_correction.correction
        return GoalEnergyPrediction(point, upper)

    def estimate_mission_context(self, environment, task_goal: np.ndarray) -> GoalEnergyPrediction:
        task_state = environment.energy_state_for_goal(task_goal)
        return_state = environment.energy_state_for_goal(
            environment.charger_position,
            position=np.asarray(task_goal, dtype=np.float32),
            velocity=np.zeros(3, dtype=np.float32),
        )
        features = np.concatenate([task_state, return_state])[None, :]
        task_point = self.point_estimator.predict(task_state)
        return_point = self.point_estimator.predict(return_state)
        point = task_point + return_point
        if isinstance(self.mission_model, HeteroscedasticResidualModel):
            base_upper = point + self.mission_model.upper_offset(features, self.coverage)[0]
        else:
            base_upper = self.mission_model.predict_level(features, self.coverage)[0]
        return GoalEnergyPrediction(
            float(point),
            float(base_upper + self.mission_correction.correction),
        )

    def predict_quantiles(self, energy_state: np.ndarray, action: np.ndarray | None = None) -> np.ndarray:
        del action
        point = self.point_estimator.predict(energy_state)
        upper = self._goal_base_upper(energy_state, point) + self.goal_correction.correction
        return np.asarray([point, upper, upper, upper], dtype=np.float64)

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "estimator_type": self.estimator_type,
                "coverage": self.coverage,
                "goal_correction": self.goal_correction.as_dict(),
                "mission_correction": self.mission_correction.as_dict(),
                "goal_model": self.goal_model.checkpoint_payload(),
                "mission_model": self.mission_model.checkpoint_payload(),
                "point_estimator": self.point_estimator.checkpoint_payload(),
            },
            destination,
        )


__all__ = [
    "AdaptiveConformalEnergyEstimator",
    "AdaptiveTrainingHistory",
    "HeteroscedasticResidualModel",
    "MCSupervisedQuantileModel",
    "QUANTILE_LEVELS",
    "RISK_COVERAGE_LEVELS",
    "ResidualScaleNetwork",
    "SupervisedQuantileNetwork",
    "TrajectoryConformalCorrection",
    "chunked_point_predictions",
    "trajectory_max_scores",
]
