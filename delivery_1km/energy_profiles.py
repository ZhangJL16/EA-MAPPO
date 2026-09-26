"""User-approved one-at-a-time model sensitivity profiles (2026-09-25)."""

from __future__ import annotations

from dataclasses import replace

from .energy import EnergyAssumption


BASE_ENERGY_ASSUMPTION = EnergyAssumption(
    name="base",
    usable_energy_fraction=0.90,
    hover_power_multiplier=1.00,
    induced_power_fraction=0.45,
    profile_power_fraction=0.35,
    propeller_diameter_m=0.3302,
    rotor_tip_speed_mps=80.0,
    drag_area_m2=0.10,
    propulsive_efficiency=0.70,
    additional_electronics_w=21.5,
    thrust_headroom_over_mtow=1.35,
)


SENSITIVITY_ENDPOINTS: tuple[tuple[str, float, float], ...] = (
    ("usable_energy_fraction", 0.80, 1.00),
    ("hover_power_multiplier", 0.85, 1.15),
    ("induced_power_fraction", 0.30, 0.60),
    ("profile_power_fraction", 0.20, 0.50),
    ("rotor_tip_speed_mps", 60.0, 100.0),
    ("drag_area_m2", 0.05, 0.20),
    ("propulsive_efficiency", 0.55, 0.85),
    ("additional_electronics_w", 13.5, 30.0),
    ("thrust_headroom_over_mtow", 1.20, 1.50),
)


def energy_profiles() -> tuple[EnergyAssumption, ...]:
    profiles = [BASE_ENERGY_ASSUMPTION]
    for field_name, low, high in SENSITIVITY_ENDPOINTS:
        profiles.append(replace(BASE_ENERGY_ASSUMPTION, name=f"{field_name}_low", **{field_name: low}))
        profiles.append(replace(BASE_ENERGY_ASSUMPTION, name=f"{field_name}_high", **{field_name: high}))
    return tuple(profiles)


ENERGY_PROFILES = energy_profiles()
