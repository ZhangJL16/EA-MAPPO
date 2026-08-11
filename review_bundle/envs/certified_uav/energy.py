from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .state import UAVPhysicalState, as_vec3


@dataclass(frozen=True)
class SimulationEnergyConfig:
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
            raise ValueError("energy coefficients must be nonnegative")
        object.__setattr__(self, "velocity_coefficients", velocity)
        object.__setattr__(self, "acceleration_coefficients", acceleration)
        scalars = (self.base_power, self.compute_power, self.communication_power, self.simulation_error)
        if any(not np.isfinite(value) or value < 0.0 for value in scalars):
            raise ValueError("energy scalar parameters must be finite and nonnegative")
        if not np.isfinite(self.flight_energy_multiplier) or self.flight_energy_multiplier <= 0.0:
            raise ValueError("flight_energy_multiplier must be finite and positive")


@dataclass(frozen=True, slots=True)
class AnalyticEnergyBoxUpper:
    """Exact separable maximum of the synthetic plant cost on an absolute box.

    The plant model is affine in ``|velocity|`` and quadratic with nonnegative
    coefficients in measured acceleration.  Its maximum on a Cartesian box is
    therefore attained at the componentwise absolute endpoints; no sampling is
    involved in this report.
    """

    fixed_cost: float
    velocity_cost: float
    acceleration_cost: float
    total_cost_upper: float
    velocity_abs_max: tuple[float, float, float]
    measured_action_abs_max: tuple[float, float, float]
    dt: float
    flight_energy_multiplier: float


@dataclass(frozen=True, slots=True)
class EnergyDominationReport:
    certified_upper: float
    realized_box_upper: float
    slack: float
    verified: bool
    analytic_box: AnalyticEnergyBoxUpper


class EnergyModel:
    def __init__(self, config: SimulationEnergyConfig | None = None) -> None:
        self.config = SimulationEnergyConfig() if config is None else config

    def base_realized_cost(
        self,
        state: UAVPhysicalState,
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
        state: UAVPhysicalState,
        measured_action: np.ndarray,
        dt: float,
    ) -> float:
        return float(
            self.config.flight_energy_multiplier
            * self.base_realized_cost(state, measured_action, dt)
        )

    def analytic_box_upper(
        self,
        velocity_abs_max: np.ndarray,
        measured_action_abs_max: np.ndarray,
        dt: float,
    ) -> AnalyticEnergyBoxUpper:
        """Return the full-domain realized-cost maximum on two absolute boxes."""

        velocity = as_vec3(velocity_abs_max, "velocity_abs_max")
        action = as_vec3(measured_action_abs_max, "measured_action_abs_max")
        if np.any(velocity < 0.0) or np.any(action < 0.0):
            raise ValueError("absolute box maxima must be nonnegative")
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        multiplier = float(self.config.flight_energy_multiplier)
        fixed = multiplier * (
            dt * (
                self.config.base_power
                + self.config.compute_power
                + self.config.communication_power
            )
            + self.config.simulation_error
        )
        velocity_cost = multiplier * dt * float(
            np.sum(self.config.velocity_coefficients * velocity)
        )
        acceleration_cost = multiplier * dt * float(
            np.sum(self.config.acceleration_coefficients * action * action)
        )
        # The exact algebraic endpoint value is rounded outward once so the
        # audit cannot pass because of a downward floating-point rounding bit.
        total = float(np.nextafter(fixed + velocity_cost + acceleration_cost, np.inf))
        return AnalyticEnergyBoxUpper(
            fixed_cost=float(fixed),
            velocity_cost=float(velocity_cost),
            acceleration_cost=float(acceleration_cost),
            total_cost_upper=total,
            velocity_abs_max=tuple(float(value) for value in velocity),
            measured_action_abs_max=tuple(float(value) for value in action),
            dt=float(dt),
            flight_energy_multiplier=multiplier,
        )

    def audit_certified_upper(
        self,
        certified_upper: float,
        velocity_abs_max: np.ndarray,
        measured_action_abs_max: np.ndarray,
        dt: float,
        *,
        tolerance: float = 0.0,
    ) -> EnergyDominationReport:
        """Compare a uniform certificate upper with the exact plant box maximum."""

        if not np.isfinite(certified_upper) or certified_upper < 0.0:
            raise ValueError("certified_upper must be finite and nonnegative")
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("tolerance must be finite and nonnegative")
        analytic = self.analytic_box_upper(velocity_abs_max, measured_action_abs_max, dt)
        slack = float(certified_upper - analytic.total_cost_upper)
        return EnergyDominationReport(
            certified_upper=float(certified_upper),
            realized_box_upper=analytic.total_cost_upper,
            slack=slack,
            verified=bool(slack >= -tolerance),
            analytic_box=analytic,
        )
