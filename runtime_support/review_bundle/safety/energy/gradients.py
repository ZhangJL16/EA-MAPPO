from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .mc_regression import EnergyToGoRegressor


def _vec3(value: np.ndarray, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite (3,) vector")
    return vector.copy()


@dataclass(frozen=True)
class PhysicalEnergyGradient:
    energy: float
    grad_position: np.ndarray
    grad_velocity: np.ndarray
    energy_state: np.ndarray


def physical_energy_state(
    position: np.ndarray,
    velocity: np.ndarray,
    goal: np.ndarray,
    *,
    horizontal_velocity_limit: float,
    vertical_velocity_limit: float,
    distance_scale: float,
) -> np.ndarray:
    point = _vec3(position, "position")
    speed = _vec3(velocity, "velocity")
    target = _vec3(goal, "goal")
    if horizontal_velocity_limit <= 0.0 or vertical_velocity_limit <= 0.0:
        raise ValueError("velocity limits must be positive")
    if distance_scale <= 0.0:
        raise ValueError("distance_scale must be positive")
    delta = target - point
    distance = float(np.linalg.norm(delta))
    direction = delta / max(distance, 1e-9)
    state = np.concatenate(
        (
            np.array(
                [
                    speed[0] / horizontal_velocity_limit,
                    speed[1] / horizontal_velocity_limit,
                    speed[2] / vertical_velocity_limit,
                ]
            ),
            direction,
            [np.clip(distance / distance_scale, 0.0, 1.0)],
        )
    )
    return np.clip(state, -1.0, 1.0).astype(np.float32)


def mc_energy_physical_gradient(
    estimator: EnergyToGoRegressor,
    position: np.ndarray,
    velocity: np.ndarray,
    goal: np.ndarray,
    *,
    horizontal_velocity_limit: float,
    vertical_velocity_limit: float,
    distance_scale: float,
) -> PhysicalEnergyGradient:
    point = torch.tensor(
        _vec3(position, "position"),
        dtype=torch.float32,
        device=estimator.device,
        requires_grad=True,
    )
    speed = torch.tensor(
        _vec3(velocity, "velocity"),
        dtype=torch.float32,
        device=estimator.device,
        requires_grad=True,
    )
    target = torch.tensor(
        _vec3(goal, "goal"),
        dtype=torch.float32,
        device=estimator.device,
    )
    delta = target - point
    distance = torch.linalg.vector_norm(delta)
    if float(distance.detach().cpu()) <= 1e-6:
        raise ValueError("energy gradient is undefined at the goal center")
    direction = delta / distance
    velocity_features = torch.stack(
        (
            speed[0] / horizontal_velocity_limit,
            speed[1] / horizontal_velocity_limit,
            speed[2] / vertical_velocity_limit,
        )
    )
    state = torch.clamp(
        torch.cat(
            (
                velocity_features,
                direction,
                torch.clamp(distance / distance_scale, 0.0, 1.0).reshape(1),
            )
        ),
        -1.0,
        1.0,
    )
    estimator.model.eval()
    normalized_energy = estimator.model(state.unsqueeze(0)).squeeze(0)
    energy = normalized_energy * estimator.battery_capacity
    grad_position, grad_velocity = torch.autograd.grad(
        energy,
        (point, speed),
        retain_graph=False,
        create_graph=False,
    )
    return PhysicalEnergyGradient(
        energy=float(energy.detach().cpu()),
        grad_position=grad_position.detach().cpu().numpy().astype(np.float64),
        grad_velocity=grad_velocity.detach().cpu().numpy().astype(np.float64),
        energy_state=state.detach().cpu().numpy().astype(np.float64),
    )
