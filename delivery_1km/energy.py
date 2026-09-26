"""Parameterized, non-validated M100 flight-energy accounting.

DJI TB47D rated hover endurance anchors the mass-to-power scale.  The
horizontal induced-inflow and blade-profile terms follow the structure of
the Gao et al. rotary-wing speed model; the thrust term follows momentum
theory.  Their coefficients here are explicit research assumptions, not
identified M100 motor parameters.  No ambient wind or battery aging is used.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, degrees, isfinite, pi, sqrt

import numpy as np

from .scenario import DeliveryScenario, M100_1KM


GRAVITY_MPS2 = 9.80665
AIR_DENSITY_KG_M3 = 1.225  # Sea-level standard atmosphere; model assumption.


@dataclass(frozen=True)
class EnergyAssumption:
    """Coefficients that must be frozen before a production energy experiment."""

    name: str
    usable_energy_fraction: float
    hover_power_multiplier: float
    induced_power_fraction: float
    profile_power_fraction: float
    propeller_diameter_m: float
    rotor_tip_speed_mps: float
    drag_area_m2: float
    propulsive_efficiency: float
    additional_electronics_w: float
    thrust_headroom_over_mtow: float

    def __post_init__(self) -> None:
        values = (
            self.usable_energy_fraction, self.hover_power_multiplier,
            self.induced_power_fraction, self.profile_power_fraction,
            self.propeller_diameter_m, self.rotor_tip_speed_mps,
            self.drag_area_m2, self.propulsive_efficiency,
            self.additional_electronics_w, self.thrust_headroom_over_mtow,
        )
        if not self.name or not all(isfinite(v) for v in values):
            raise ValueError("energy assumptions must be finite and named")
        if not (0 < self.usable_energy_fraction <= 1 and self.hover_power_multiplier > 0):
            raise ValueError("invalid battery or hover-power scaling")
        if not (0 <= self.induced_power_fraction <= 1 and 0 <= self.profile_power_fraction <= 1
                and self.induced_power_fraction + self.profile_power_fraction <= 1):
            raise ValueError("power fractions must sum to at most one")
        if not (self.propeller_diameter_m > 0 and self.rotor_tip_speed_mps > 0
                and self.drag_area_m2 >= 0 and 0 < self.propulsive_efficiency <= 1
                and self.additional_electronics_w >= 0 and self.thrust_headroom_over_mtow >= 1):
            raise ValueError("invalid aerodynamic or thrust assumptions")


@dataclass(frozen=True)
class EnergySample:
    power_w: float
    thrust_n: float
    tilt_deg: float
    feasible: bool
    violated: tuple[str, ...]


@dataclass(frozen=True)
class EnergyTrace:
    energy_wh: float
    max_power_w: float
    max_thrust_n: float
    max_tilt_deg: float
    infeasible_steps: int
    first_infeasible_step: int | None
    violated: tuple[str, ...]


def rated_hover_equivalent_power_w(mass_kg: float, scenario: DeliveryScenario = M100_1KM) -> float:
    """Interpolate DJI's 22/17/13 min TB47D endurance in mass^(3/2).

    This is nameplate-Wh / rated-time, not measured instantaneous electrical
    power.  Valid only for 0-1 kg additional mass beyond the bare M100.
    """
    if not isfinite(mass_kg):
        raise ValueError("mass must be finite")
    base = scenario.base_mass_kg
    masses = np.array((base, base + 0.5, base + 1.0), dtype=float)
    if mass_kg < masses[0] - 1e-10 or mass_kg > masses[-1] + 1e-10:
        raise ValueError("hover-time interpolation is outside DJI payload anchors")
    effective_watts = np.array([scenario.nominal_battery_wh * 60.0 / minutes
                                for minutes in (22.0, 17.0, 13.0)])
    return float(np.interp(mass_kg**1.5, masses**1.5, effective_watts))


def _inflow_factor(horizontal_speed_mps: float, induced_speed_mps: float) -> float:
    ratio_sq = (horizontal_speed_mps / induced_speed_mps) ** 2
    return sqrt(max(0.0, sqrt(1.0 + ratio_sq * ratio_sq / 4.0) - ratio_sq / 2.0))


def step_energy(
    velocity_mps: object,
    acceleration_mps2: object,
    payload_kg: float,
    assumptions: EnergyAssumption,
    scenario: DeliveryScenario = M100_1KM,
) -> EnergySample:
    v = np.asarray(velocity_mps, dtype=float)
    a = np.asarray(acceleration_mps2, dtype=float)
    if v.shape != (3,) or a.shape != (3,) or not np.isfinite(v).all() or not np.isfinite(a).all():
        raise ValueError("velocity and acceleration must be finite 3-vectors")
    if not isfinite(payload_kg) or payload_kg < 0:
        raise ValueError("payload must be finite and nonnegative")
    mass = scenario.base_mass_kg + scenario.equipment_mass_kg + payload_kg
    if mass > scenario.max_takeoff_mass_kg + 1e-10:
        raise ValueError("payload exceeds manufacturer maximum takeoff mass")
    if mass > scenario.base_mass_kg + 1.0 + 1e-10:
        raise ValueError("payload exceeds hover-time interpolation range")

    horizontal_speed = float(np.linalg.norm(v[:2]))
    drag_force_n = 0.5 * AIR_DENSITY_KG_M3 * assumptions.drag_area_m2 * horizontal_speed**2
    drag_per_mass = (drag_force_n / mass) * (v[:2] / horizontal_speed) if horizontal_speed > 1e-12 else np.zeros(2)
    horizontal_force_per_mass = a[:2] + drag_per_mass
    vertical_force_per_mass = GRAVITY_MPS2 + float(a[2])
    thrust_per_mass = float(np.linalg.norm((horizontal_force_per_mass[0], horizontal_force_per_mass[1], vertical_force_per_mass)))
    thrust_n = mass * thrust_per_mass
    tilt_deg = degrees(atan2(float(np.linalg.norm(horizontal_force_per_mass)), vertical_force_per_mass))
    disk_area = 4.0 * pi * (assumptions.propeller_diameter_m / 2.0) ** 2
    induced_speed = sqrt(thrust_n / (2.0 * AIR_DENSITY_KG_M3 * disk_area))
    inflow = _inflow_factor(horizontal_speed, induced_speed)
    thrust_ratio = thrust_per_mass / GRAVITY_MPS2
    remainder_fraction = 1.0 - assumptions.induced_power_fraction - assumptions.profile_power_fraction
    shape = (
        assumptions.induced_power_fraction * thrust_ratio**1.5 * inflow
        + assumptions.profile_power_fraction * (1.0 + 3.0 * (horizontal_speed / assumptions.rotor_tip_speed_mps) ** 2)
        + remainder_fraction
    )
    # DJI's quoted duration is measured to its own end-of-discharge point.
    # Scale both the usable capacity and the implied hover power by the same
    # unknown fraction so the nominal hover-time anchor stays self-consistent.
    hover = (assumptions.usable_energy_fraction * assumptions.hover_power_multiplier
             * rated_hover_equivalent_power_w(mass, scenario))
    parasite = drag_force_n * horizontal_speed / assumptions.propulsive_efficiency
    climb = mass * GRAVITY_MPS2 * max(0.0, float(v[2])) / assumptions.propulsive_efficiency
    power_w = hover * shape + parasite + climb + assumptions.additional_electronics_w

    violations: list[str] = []
    if thrust_n > assumptions.thrust_headroom_over_mtow * scenario.max_takeoff_mass_kg * GRAVITY_MPS2 + 1e-9:
        violations.append("assumed_thrust_limit")
    if tilt_deg > 35.0 + 1e-9:  # DJI M100 published tilt limit.
        violations.append("published_tilt_limit")
    if v[2] > 5.0 + 1e-9 or v[2] < -4.0 - 1e-9:  # DJI M100 published ascent/descent limits.
        violations.append("published_vertical_speed_limit")
    return EnergySample(power_w, thrust_n, tilt_deg, not violations, tuple(violations))


def account_flight_trace(
    velocities_mps: object,
    accelerations_mps2: object,
    payload_kg: float,
    assumptions: EnergyAssumption,
    scenario: DeliveryScenario = M100_1KM,
) -> EnergyTrace:
    """Account executed controls; terminal velocity reset has no energy spike.

    Uses the velocity at each step's start and its applied acceleration.  The
    final goal-arrival zeroing of velocity is not treated as a motor impulse.
    Contacts must be reported separately; collision repair is not a flight
    maneuver that a power model can physically validate.
    """
    velocities = np.asarray(velocities_mps, dtype=float)
    accelerations = np.asarray(accelerations_mps2, dtype=float)
    if velocities.ndim != 2 or accelerations.ndim != 2 or velocities.shape[1:] != (3,) or accelerations.shape[1:] != (3,):
        raise ValueError("trace arrays must have three columns")
    if len(velocities) != len(accelerations) + 1 or not np.isfinite(velocities).all() or not np.isfinite(accelerations).all():
        raise ValueError("invalid trace lengths or nonfinite state")
    if not len(accelerations):
        return EnergyTrace(0.0, 0.0, 0.0, 0.0, 0, None, ())
    samples = [step_energy(v, a, payload_kg, assumptions, scenario)
               for v, a in zip(velocities[:-1], accelerations)]
    violated = tuple(sorted({name for sample in samples for name in sample.violated}))
    first = next((i for i, sample in enumerate(samples) if not sample.feasible), None)
    return EnergyTrace(
        energy_wh=sum(sample.power_w for sample in samples) * scenario.physics_dt_s / 3600.0,
        max_power_w=max(sample.power_w for sample in samples),
        max_thrust_n=max(sample.thrust_n for sample in samples),
        max_tilt_deg=max(sample.tilt_deg for sample in samples),
        infeasible_steps=sum(not sample.feasible for sample in samples),
        first_infeasible_step=first,
        violated=violated,
    )
