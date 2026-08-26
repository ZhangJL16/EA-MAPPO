from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional

from review_bundle.safety.energy.mc_regression import (
    GoalEnergyPrediction,
    HierarchicalConformalCalibration,
    MissionConformalCalibration,
)

from .dataset import PackedBridgeDataset
from .safety_buffer import SAFETY_CONTEXT_DIM


@dataclass(frozen=True)
class SupervisedFitResult:
    best_epoch: int
    best_validation_mae: float
    train_loss: list[float]
    validation_mae: list[float]


class _PositiveNetwork(nn.Module):
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


class SafetyBridgeEncoder(nn.Module):
    def __init__(self, *, latent_dim: int = 8, hidden_dim: int = 32) -> None:
        super().__init__()
        if latent_dim <= 0 or hidden_dim <= 0:
            raise ValueError("encoder dimensions must be positive")
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.encoder = nn.Sequential(
            nn.Linear(SAFETY_CONTEXT_DIM, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.Tanh(),
        )
        self.reconstruction = nn.Linear(latent_dim, SAFETY_CONTEXT_DIM)
        self.burden = nn.Sequential(nn.Linear(latent_dim, 1), nn.Softplus())

    def forward(self, contexts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent = self.encoder(contexts)
        return latent, self.reconstruction(latent), self.burden(latent).squeeze(-1)

    def encode(self, contexts: torch.Tensor) -> torch.Tensor:
        return self.encoder(contexts)

    def fit_dataset(
        self,
        train: PackedBridgeDataset,
        validation: PackedBridgeDataset,
        *,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        burden_scale: float,
        seed: int,
        device: str,
    ) -> SupervisedFitResult:
        if burden_scale <= 0.0:
            raise ValueError("burden_scale must be positive")
        generator = np.random.default_rng(seed)
        target_device = torch.device(device)
        self.to(target_device)
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        train_losses: list[float] = []
        validation_maes: list[float] = []
        best_state = None
        best_mae = float("inf")
        best_epoch = 0
        for epoch in range(1, epochs + 1):
            order = generator.permutation(len(train))
            epoch_losses = []
            self.train()
            for start in range(0, len(order), batch_size):
                indices = order[start : start + batch_size]
                contexts = torch.as_tensor(
                    train.safety_contexts[indices], dtype=torch.float32, device=target_device
                )
                burdens = torch.as_tensor(
                    train.burden_to_go[indices] / burden_scale,
                    dtype=torch.float32,
                    device=target_device,
                )
                _, reconstruction, predicted_burden = self(contexts)
                loss = functional.mse_loss(reconstruction, contexts) + functional.smooth_l1_loss(
                    predicted_burden,
                    burdens,
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_losses.append(float(loss.detach().cpu()))
            self.eval()
            with torch.no_grad():
                contexts = torch.as_tensor(
                    validation.safety_contexts,
                    dtype=torch.float32,
                    device=target_device,
                )
                _, _, predicted = self(contexts)
                mae = float(
                    torch.mean(
                        torch.abs(
                            predicted
                            - torch.as_tensor(
                                validation.burden_to_go / burden_scale,
                                dtype=torch.float32,
                                device=target_device,
                            )
                        )
                    ).cpu()
                )
            train_losses.append(float(np.mean(epoch_losses)))
            validation_maes.append(mae)
            if mae < best_mae:
                best_mae = mae
                best_epoch = epoch
                best_state = {
                    key: value.detach().cpu().clone() for key, value in self.state_dict().items()
                }
        if best_state is None:
            raise RuntimeError("SafetyBridge encoder did not train")
        self.load_state_dict(best_state)
        self.to(target_device)
        return SupervisedFitResult(best_epoch, best_mae, train_losses, validation_maes)


class FlexibleEnergyRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dim: int = 128,
        energy_scale: float,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0 or energy_scale <= 0.0:
            raise ValueError("invalid energy regressor configuration")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.energy_scale = float(energy_scale)
        self.network = _PositiveNetwork(input_dim, hidden_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features) * self.energy_scale

    def predict(self, features: np.ndarray, *, device: str | torch.device) -> np.ndarray:
        values = np.asarray(features, dtype=np.float32)
        if values.ndim == 1:
            values = values[None, :]
        if values.ndim != 2 or values.shape[1] != self.input_dim:
            raise ValueError("energy feature dimension mismatch")
        self.eval()
        with torch.no_grad():
            result = self(
                torch.as_tensor(values, dtype=torch.float32, device=device)
            )
        return result.cpu().numpy().astype(np.float64)

    def fit_arrays(
        self,
        train_features: np.ndarray,
        train_targets: np.ndarray,
        validation_features: np.ndarray,
        validation_targets: np.ndarray,
        *,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        seed: int,
        device: str,
    ) -> SupervisedFitResult:
        train_x = np.asarray(train_features, dtype=np.float32)
        train_y = np.asarray(train_targets, dtype=np.float32)
        valid_x = np.asarray(validation_features, dtype=np.float32)
        valid_y = np.asarray(validation_targets, dtype=np.float32)
        if train_x.ndim != 2 or train_x.shape[1] != self.input_dim:
            raise ValueError("invalid training features")
        if valid_x.ndim != 2 or valid_x.shape[1] != self.input_dim:
            raise ValueError("invalid validation features")
        if train_y.shape != (train_x.shape[0],) or valid_y.shape != (valid_x.shape[0],):
            raise ValueError("energy targets must align with features")
        rng = np.random.default_rng(seed)
        target_device = torch.device(device)
        self.to(target_device)
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        best_state = None
        best_mae = float("inf")
        best_epoch = 0
        train_losses: list[float] = []
        validation_maes: list[float] = []
        for epoch in range(1, epochs + 1):
            order = rng.permutation(train_x.shape[0])
            losses = []
            self.train()
            for start in range(0, order.size, batch_size):
                indices = order[start : start + batch_size]
                x = torch.as_tensor(train_x[indices], dtype=torch.float32, device=target_device)
                y = torch.as_tensor(
                    train_y[indices] / self.energy_scale,
                    dtype=torch.float32,
                    device=target_device,
                )
                predicted = self.network(x)
                loss = functional.smooth_l1_loss(predicted, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            predictions = self.predict(valid_x, device=target_device)
            mae = float(np.mean(np.abs(predictions - valid_y)))
            train_losses.append(float(np.mean(losses)))
            validation_maes.append(mae)
            if mae < best_mae:
                best_mae = mae
                best_epoch = epoch
                best_state = {
                    key: value.detach().cpu().clone() for key, value in self.state_dict().items()
                }
        if best_state is None:
            raise RuntimeError("energy regressor did not train")
        self.load_state_dict(best_state)
        self.to(target_device)
        return SupervisedFitResult(best_epoch, best_mae, train_losses, validation_maes)

    def checkpoint_payload(self) -> dict[str, object]:
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "energy_scale": self.energy_scale,
            "state_dict": self.state_dict(),
        }

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.checkpoint_payload(), destination)

    @classmethod
    def from_checkpoint_payload(
        cls,
        payload: Mapping[str, object],
        *,
        device: str | torch.device = "cpu",
    ) -> "FlexibleEnergyRegressor":
        model = cls(
            int(payload["input_dim"]),
            hidden_dim=int(payload["hidden_dim"]),
            energy_scale=float(payload["energy_scale"]),
        )
        model.load_state_dict(payload["state_dict"])
        model.to(device)
        model.eval()
        return model


class ActionConditionedEnergyCritic(FlexibleEnergyRegressor):
    def __init__(
        self,
        *,
        state_dim: int = 7,
        context_dim: int = SAFETY_CONTEXT_DIM,
        action_dim: int = 3,
        hidden_dim: int = 128,
        energy_scale: float,
    ) -> None:
        self.state_dim = int(state_dim)
        self.context_dim = int(context_dim)
        self.action_dim = int(action_dim)
        super().__init__(
            self.state_dim + self.context_dim + self.action_dim,
            hidden_dim=hidden_dim,
            energy_scale=energy_scale,
        )

    def features(
        self,
        states: torch.Tensor,
        contexts: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        if states.ndim != 2 or states.shape[1] != self.state_dim:
            raise ValueError("state batch dimension mismatch")
        if contexts.ndim != 2 or contexts.shape[1] != self.context_dim:
            raise ValueError("context batch dimension mismatch")
        if actions.ndim != 2 or actions.shape[1] != self.action_dim:
            raise ValueError("action batch dimension mismatch")
        return torch.cat([states, contexts, actions], dim=1)

    def energy(
        self,
        states: torch.Tensor,
        contexts: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        return self(self.features(states, contexts, actions))

    def freeze(self) -> None:
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def checkpoint_payload(self) -> dict[str, object]:
        payload = super().checkpoint_payload()
        payload.update(
            {
                "state_dim": self.state_dim,
                "context_dim": self.context_dim,
                "action_dim": self.action_dim,
            }
        )
        return payload


class CalibratedCompactEnergyEstimator:
    estimator_type = "jseb_compact_mc_group_conformal"
    bootstrapping = False
    gamma = None

    def __init__(
        self,
        point_model: FlexibleEnergyRegressor,
        goal_calibration: HierarchicalConformalCalibration,
        mission_calibration: MissionConformalCalibration,
        *,
        device: str,
    ) -> None:
        if point_model.input_dim != 7:
            raise ValueError("deployed compact estimator must use the 7D energy state")
        self.point_model = point_model.to(device)
        self.point_model.eval()
        self.goal_calibration = goal_calibration
        self.mission_calibration = mission_calibration
        self.device = torch.device(device)
        self.coverage = float(goal_calibration.coverage_target)
        self.mission_coverage = float(mission_calibration.coverage_target)
        self.update_count = 0
        self.replay: tuple[()] = ()
        self.trainable_replay: tuple[()] = ()

    def _prediction(
        self,
        environment,
        goal: np.ndarray,
        *,
        position: np.ndarray | None,
        velocity: np.ndarray | None,
        goal_type: str,
    ) -> GoalEnergyPrediction:
        calibration_goal_type = (
            "CHARGER" if goal_type == "TASK_ENDPOINT_TO_CHARGER" else goal_type
        )
        state = environment.compact_energy_state_for_goal(
            goal,
            position=position,
            velocity=velocity,
        )
        point = float(self.point_model.predict(state, device=self.device)[0])
        current_position = (
            np.asarray(environment.agent.pos, dtype=np.float32)
            if position is None
            else np.asarray(position, dtype=np.float32)
        )
        distance = float(np.linalg.norm(np.asarray(goal) - current_position))
        upper = point + self.goal_calibration.margin_for(
            calibration_goal_type,
            distance,
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
        return self._prediction(
            environment,
            np.asarray(goal, dtype=np.float32),
            position=position,
            velocity=velocity,
            goal_type=resolved_type,
        )

    def estimate_mission_context(self, environment, task_goal: np.ndarray) -> GoalEnergyPrediction:
        task = self._prediction(
            environment,
            np.asarray(task_goal, dtype=np.float32),
            position=None,
            velocity=None,
            goal_type="TASK",
        )
        returned = self._prediction(
            environment,
            np.asarray(environment.charger_position, dtype=np.float32),
            position=np.asarray(task_goal, dtype=np.float32),
            velocity=np.zeros(3, dtype=np.float32),
            goal_type="TASK_ENDPOINT_TO_CHARGER",
        )
        point = task.prediction + returned.prediction
        distance = float(np.linalg.norm(np.asarray(task_goal) - environment.agent.pos))
        return GoalEnergyPrediction(
            point,
            point + self.mission_calibration.margin_for(distance),
        )

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

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "estimator_type": self.estimator_type,
                "point_model": self.point_model.checkpoint_payload(),
                "goal_calibration": self.goal_calibration.as_dict(),
                "mission_calibration": self.mission_calibration.as_dict(),
            },
            destination,
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        device: str = "cpu",
    ) -> "CalibratedCompactEnergyEstimator":
        payload = torch.load(Path(path), map_location=device, weights_only=False)
        if payload.get("estimator_type") != cls.estimator_type:
            raise ValueError("checkpoint is not a calibrated compact JSEB estimator")
        point_model = FlexibleEnergyRegressor.from_checkpoint_payload(
            payload["point_model"],
            device=device,
        )
        return cls(
            point_model,
            HierarchicalConformalCalibration.from_dict(payload["goal_calibration"]),
            MissionConformalCalibration.from_dict(payload["mission_calibration"]),
            device=device,
        )


def fit_result_dict(result: SupervisedFitResult) -> dict[str, object]:
    return asdict(result)
