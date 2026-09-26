"""Array implementation of the frozen per-step energy accounting formula."""

from __future__ import annotations

from math import pi

import numpy as np

from .energy import (AIR_DENSITY_KG_M3, GRAVITY_MPS2, EnergyAssumption,
                     EnergyTrace, rated_hover_equivalent_power_w, step_energy)
from .scenario import DeliveryScenario, M100_1KM


def account_flight_trace_vectorized(
    velocities_mps: object,
    accelerations_mps2: object,
    payload_kg: float,
    assumptions: EnergyAssumption,
    scenario: DeliveryScenario = M100_1KM,
) -> EnergyTrace:
    velocities = np.asarray(velocities_mps, dtype=float)
    accelerations = np.asarray(accelerations_mps2, dtype=float)
    if (velocities.ndim != 2 or accelerations.ndim != 2
            or velocities.shape[1:] != (3,) or accelerations.shape[1:] != (3,)
            or len(velocities) != len(accelerations) + 1
            or not np.isfinite(velocities).all() or not np.isfinite(accelerations).all()):
        raise ValueError("invalid trace lengths or nonfinite state")
    if not len(accelerations):
        return EnergyTrace(0.0, 0.0, 0.0, 0.0, 0, None, ())
    step_energy(velocities[0], accelerations[0], payload_kg, assumptions, scenario)

    velocity = velocities[:-1]
    mass = scenario.base_mass_kg + scenario.equipment_mass_kg + payload_kg
    horizontal_speed = np.linalg.norm(velocity[:, :2], axis=1)
    drag = 0.5 * AIR_DENSITY_KG_M3 * assumptions.drag_area_m2 * horizontal_speed**2
    direction = np.divide(velocity[:, :2], horizontal_speed[:, None],
                          out=np.zeros_like(velocity[:, :2]),
                          where=horizontal_speed[:, None] > 1e-12)
    horizontal_force_per_mass = accelerations[:, :2] + drag[:, None] * direction / mass
    horizontal_force_norm = np.linalg.norm(horizontal_force_per_mass, axis=1)
    vertical_force_per_mass = GRAVITY_MPS2 + accelerations[:, 2]
    thrust_per_mass = np.sqrt(horizontal_force_norm**2 + vertical_force_per_mass**2)
    thrust = mass * thrust_per_mass
    tilt = np.degrees(np.arctan2(horizontal_force_norm, vertical_force_per_mass))

    disk_area = 4.0 * pi * (assumptions.propeller_diameter_m / 2.0) ** 2
    induced_speed = np.sqrt(thrust / (2.0 * AIR_DENSITY_KG_M3 * disk_area))
    ratio_sq = (horizontal_speed / induced_speed) ** 2
    inflow = np.sqrt(np.maximum(0.0, np.sqrt(1.0 + ratio_sq**2 / 4.0) - ratio_sq / 2.0))
    remainder = 1.0 - assumptions.induced_power_fraction - assumptions.profile_power_fraction
    shape = (
        assumptions.induced_power_fraction * (thrust_per_mass / GRAVITY_MPS2) ** 1.5 * inflow
        + assumptions.profile_power_fraction * (1.0 + 3.0 * (horizontal_speed / assumptions.rotor_tip_speed_mps) ** 2)
        + remainder
    )
    hover = (assumptions.usable_energy_fraction * assumptions.hover_power_multiplier
             * rated_hover_equivalent_power_w(mass, scenario))
    power = (hover * shape + drag * horizontal_speed / assumptions.propulsive_efficiency
             + mass * GRAVITY_MPS2 * np.maximum(0.0, velocity[:, 2]) / assumptions.propulsive_efficiency
             + assumptions.additional_electronics_w)

    thrust_violation = thrust > assumptions.thrust_headroom_over_mtow * scenario.max_takeoff_mass_kg * GRAVITY_MPS2 + 1e-9
    tilt_violation = tilt > 35.0 + 1e-9
    speed_violation = (velocity[:, 2] > 5.0 + 1e-9) | (velocity[:, 2] < -4.0 - 1e-9)
    infeasible = thrust_violation | tilt_violation | speed_violation
    violated = tuple(name for name, mask in (
        ("assumed_thrust_limit", thrust_violation),
        ("published_tilt_limit", tilt_violation),
        ("published_vertical_speed_limit", speed_violation),
    ) if np.any(mask))
    first = np.flatnonzero(infeasible)
    return EnergyTrace(
        energy_wh=float(np.sum(power) * scenario.physics_dt_s / 3600.0),
        max_power_w=float(np.max(power)), max_thrust_n=float(np.max(thrust)),
        max_tilt_deg=float(np.max(tilt)), infeasible_steps=int(np.count_nonzero(infeasible)),
        first_infeasible_step=int(first[0]) if len(first) else None,
        violated=violated,
    )
