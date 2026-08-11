from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .actuator import ActionTrace
from .plant_env import CertifiedSingleUAVPlantEnv
from .state import UAVPhysicalState
from .telemetry import StepTelemetry


@dataclass(frozen=True, slots=True)
class ChargingConfig:
    battery_capacity: float = 30.0
    charging_rate: float = 2.0
    station_available: bool = True
    checkpoint_steps: int = 5
    departure_energy_margin: float = 0.5
    forced_return_margin: float = 1.0
    version: str = "synthetic-charging-v1"

    def __post_init__(self) -> None:
        if not np.isfinite(self.battery_capacity) or self.battery_capacity <= 0.0:
            raise ValueError("battery capacity must be positive")
        if not np.isfinite(self.charging_rate) or self.charging_rate <= 0.0:
            raise ValueError("net charging rate must be positive")
        if self.checkpoint_steps <= 0:
            raise ValueError("charging checkpoint interval must be positive")
        if self.departure_energy_margin < 0.0 or self.forced_return_margin < 0.0:
            raise ValueError("charging margins must be nonnegative")

    def gain_per_step(self, dt: float) -> float:
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("charging control period must be positive")
        return float(self.charging_rate * dt)


@dataclass(frozen=True, slots=True)
class ChargingStepResult:
    charged_energy: float
    telemetry: StepTelemetry
    truncated: bool


@dataclass(frozen=True, slots=True)
class DepartureGateResult:
    allowed: bool
    required_with_margin: float
    reason: str | None


def verify_departure_energy(
    current_energy: float,
    required_route_energy: float,
    margin: float,
    certificate_valid: bool,
) -> DepartureGateResult:
    values = (current_energy, required_route_energy, margin)
    if not all(np.isfinite(value) for value in values) or required_route_energy < 0.0 or margin < 0.0:
        return DepartureGateResult(False, float("inf"), "INVALID_DEPARTURE_ENERGY_INPUT")
    required = float(required_route_energy + margin)
    if not certificate_valid:
        return DepartureGateResult(False, required, "PERSISTENT_CERTIFICATE_INVALID")
    if current_energy < required:
        return DepartureGateResult(False, required, "INSUFFICIENT_DEPARTURE_ENERGY")
    return DepartureGateResult(True, required, None)


class ChargingDynamics:
    """Synthetic net charging dynamics; no reset, teleport, or velocity correction."""

    def __init__(self, config: ChargingConfig | None = None) -> None:
        self.config = config or ChargingConfig()

    def can_charge(self, plant: CertifiedSingleUAVPlantEnv) -> bool:
        return bool(
            self.config.station_available
            and plant.failure_reason is None
            and plant.terminal.is_charge_admissible(plant.state)
        )

    def step(
        self,
        plant: CertifiedSingleUAVPlantEnv,
        certificate_epoch: str,
        hold_action: np.ndarray | None = None,
    ) -> ChargingStepResult:
        if not self.can_charge(plant):
            raise RuntimeError("CHARGING_NOT_ADMISSIBLE")
        before = plant.state.copy()
        if hold_action is None:
            position = before.position.copy()
            velocity = before.velocity.copy()
            energy_before_charge = before.energy
            energy_cost = 0.0
            collision = False
            published = np.zeros(3, dtype=np.float64)
            measured = published.copy()
            plant.step_count += 1
            truncated = plant.step_count >= plant.config.episode_limit
        else:
            published = np.asarray(hold_action, dtype=np.float64)
            _, _, terminated, truncated, info = plant.step(published)
            motion_telemetry = info["telemetry"]
            measured = motion_telemetry.action_trace.measured.copy()
            position = plant.state.position.copy()
            velocity = plant.state.velocity.copy()
            energy_before_charge = plant.state.energy
            energy_cost = motion_telemetry.energy_cost
            collision = motion_telemetry.collision
            if terminated or not self.can_charge(plant):
                raise RuntimeError("CERTIFIED_CHARGER_HOLD_VIOLATED")
        after_energy = min(
            self.config.battery_capacity,
            energy_before_charge + self.config.gain_per_step(plant.config.dt),
        )
        plant.state = UAVPhysicalState(
            position,
            velocity,
            after_energy,
            before.timestamp + plant.config.dt,
        )
        plant.last_lidar = plant.lidar_model.measure(plant.state, plant.world, plant.np_random)
        trace = ActionTrace(
            None,
            None if hold_action is None else published,
            published,
            published,
            measured,
            False,
            "CHARGER_HOLD",
            certificate_epoch,
        )
        telemetry = StepTelemetry(
            before,
            plant.state.copy(),
            trace,
            energy_cost,
            collision,
            plant.terminal.is_charge_admissible(plant.state),
            plant.last_lidar,
            certificate_epoch,
            None,
            None,
        )
        plant.last_telemetry = telemetry
        return ChargingStepResult(
            after_energy - energy_before_charge,
            telemetry,
            truncated,
        )

    def apply_during_motion_cycle(
        self,
        plant: CertifiedSingleUAVPlantEnv,
        telemetry: StepTelemetry,
    ) -> ChargingStepResult:
        """Apply synthetic net charging after a continuous RL-controlled plant step."""

        if not self.can_charge(plant):
            return ChargingStepResult(0.0, telemetry, plant.step_count >= plant.config.episode_limit)
        before_charge = plant.state.copy()
        after_energy = min(
            self.config.battery_capacity,
            before_charge.energy + self.config.gain_per_step(plant.config.dt),
        )
        plant.state = UAVPhysicalState(
            before_charge.position.copy(),
            before_charge.velocity.copy(),
            after_energy,
            before_charge.timestamp,
        )
        updated = StepTelemetry(
            telemetry.state_before,
            plant.state.copy(),
            telemetry.action_trace,
            telemetry.energy_cost,
            telemetry.collision,
            plant.terminal.is_charge_admissible(plant.state),
            telemetry.lidar_packet,
            telemetry.certificate_version,
            telemetry.geometry_version,
            telemetry.corridor_version,
        )
        plant.last_telemetry = updated
        return ChargingStepResult(
            after_energy - before_charge.energy,
            updated,
            plant.step_count >= plant.config.episode_limit,
        )
