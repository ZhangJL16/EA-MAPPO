from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .state import NavigationState, as_vec3


@dataclass(frozen=True)
class TelemetryCostConfig:
    base_power: float = 0.05
    velocity_coefficients: np.ndarray = field(default_factory=lambda: np.full(3, 0.005))
    acceleration_coefficients: np.ndarray = field(default_factory=lambda: np.full(3, 0.005))
    compute_power: float = 0.005
    communication_power: float = 0.005
    simulation_error: float = 0.0
    flight_energy_multiplier: float = 1.0

    def __post_init__(self) -> None:
        velocity = as_vec3(self.velocity_coefficients, "velocity_coefficients")
        acceleration = as_vec3(self.acceleration_coefficients, "acceleration_coefficients")
        if np.any(velocity < 0.0) or np.any(acceleration < 0.0):
            raise ValueError("cost coefficients must be nonnegative")
        object.__setattr__(self, "velocity_coefficients", velocity)
        object.__setattr__(self, "acceleration_coefficients", acceleration)
        values = (
            self.base_power,
            self.compute_power,
            self.communication_power,
            self.simulation_error,
        )
        if any(not np.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("telemetry cost parameters must be finite and nonnegative")
        if not np.isfinite(self.flight_energy_multiplier) or self.flight_energy_multiplier <= 0.0:
            raise ValueError("flight_energy_multiplier must be finite and positive")


class TelemetryCostModel:
    def __init__(self, config: TelemetryCostConfig | None = None) -> None:
        self.config = TelemetryCostConfig() if config is None else config

    def base_realized_cost(
        self,
        state: NavigationState,
        measured_action: np.ndarray,
        dt: float,
    ) -> float:
        action = as_vec3(measured_action, "measured_action")
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        power = (
            self.config.base_power
            + float(np.sum(self.config.velocity_coefficients * np.abs(state.velocity)))
            + float(np.sum(self.config.acceleration_coefficients * action * action))
            + self.config.compute_power
            + self.config.communication_power
        )
        return float(dt * power + self.config.simulation_error)

    def realized_cost(
        self,
        state: NavigationState,
        measured_action: np.ndarray,
        dt: float,
    ) -> float:
        return float(self.config.flight_energy_multiplier * self.base_realized_cost(state, measured_action, dt))
