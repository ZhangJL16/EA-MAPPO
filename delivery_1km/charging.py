"""Explicit sensitivity assumptions for M100 partial charging.

DJI rates the standard TB47D charger at 100 W. Efficiency and taper shape are
research assumptions, not an observed M100 SOC-versus-time curve. This module
only estimates time while docked; idle alone never increases battery SOC.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, log


@dataclass(frozen=True)
class ChargingAssumption:
    name: str
    charger_input_limit_w: float = 100.0
    battery_efficiency: float = 0.90
    taper_start_soc: float = 0.80
    end_power_fraction: float = 0.20

    def __post_init__(self) -> None:
        values = (
            self.charger_input_limit_w, self.battery_efficiency,
            self.taper_start_soc, self.end_power_fraction,
        )
        if not all(isfinite(value) for value in values):
            raise ValueError("charging parameters must be finite")
        if not self.name or self.charger_input_limit_w <= 0:
            raise ValueError("charging assumption needs a name and positive input limit")
        if not (0 < self.battery_efficiency <= 1):
            raise ValueError("battery efficiency must lie in (0, 1]")
        if not (0 < self.taper_start_soc < 1 and 0 < self.end_power_fraction <= 1):
            raise ValueError("invalid taper assumptions")

    def battery_power_w(self, soc: float) -> float:
        if not isfinite(soc) or not 0 <= soc <= 1:
            raise ValueError("SOC must lie in [0, 1]")
        if soc <= self.taper_start_soc:
            fraction = 1.0
        else:
            progress = (soc - self.taper_start_soc) / (1.0 - self.taper_start_soc)
            fraction = 1.0 - (1.0 - self.end_power_fraction) * progress
        return self.charger_input_limit_w * self.battery_efficiency * fraction

    def seconds_to_soc(self, start_soc: float, target_soc: float, nominal_battery_wh: float) -> float:
        """Integrate a constant-power phase followed by linear-power taper."""
        if not all(isfinite(value) for value in (start_soc, target_soc, nominal_battery_wh)):
            raise ValueError("charge request must be finite")
        if not (0 <= start_soc <= target_soc <= 1) or nominal_battery_wh <= 0:
            raise ValueError("invalid charging interval or battery capacity")
        if start_soc == target_soc:
            return 0.0
        base_seconds = 3600.0 * nominal_battery_wh / (self.charger_input_limit_w * self.battery_efficiency)
        constant_end = min(target_soc, self.taper_start_soc)
        constant_fraction = max(0.0, constant_end - start_soc)
        result = base_seconds * constant_fraction
        taper_begin = max(start_soc, self.taper_start_soc)
        if target_soc <= taper_begin:
            return result
        slope = (1.0 - self.end_power_fraction) / (1.0 - self.taper_start_soc)
        if slope == 0:
            return result + base_seconds * (target_soc - taper_begin)
        power_fraction_begin = 1.0 - slope * (taper_begin - self.taper_start_soc)
        power_fraction_end = 1.0 - slope * (target_soc - self.taper_start_soc)
        return result + base_seconds * log(power_fraction_begin / power_fraction_end) / slope


# Same 100 W charger cap in all cases; only unmeasured conversion/taper vary.
CHARGING_SENSITIVITY = (
    ChargingAssumption("optimistic_flat", battery_efficiency=0.95, taper_start_soc=0.90, end_power_fraction=1.0),
    ChargingAssumption("reference_taper", battery_efficiency=0.90, taper_start_soc=0.80, end_power_fraction=0.20),
    ChargingAssumption("conservative_taper", battery_efficiency=0.80, taper_start_soc=0.70, end_power_fraction=0.10),
)
