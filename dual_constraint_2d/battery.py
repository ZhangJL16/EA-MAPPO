"""M100 external-validation SOC accounting and event-driven dock charging.

The usable-energy fraction defines a lower physical-SOC cutoff.  It does not
rescale the battery's Wh-per-SOC conversion or the charger input power.  This
module is not the core battery model of the synthetic 2D mechanism study.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from delivery_1km.charging import ChargingAssumption
from delivery_1km.energy import EnergyAssumption
from delivery_1km.scenario import DeliveryScenario, M100_1KM


@dataclass(frozen=True)
class BatteryClock:
    physical_soc: float
    elapsed_s: float = 0.0

    def __post_init__(self) -> None:
        if not isfinite(self.physical_soc) or not 0.0 <= self.physical_soc <= 1.0:
            raise ValueError("physical SOC must lie in [0, 1]")
        if not isfinite(self.elapsed_s) or self.elapsed_s < 0.0:
            raise ValueError("elapsed time must be finite and nonnegative")


@dataclass(frozen=True)
class BatteryModel:
    energy: EnergyAssumption
    charger: ChargingAssumption
    scenario: DeliveryScenario = M100_1KM

    @property
    def minimum_physical_soc(self) -> float:
        return 1.0 - self.energy.usable_energy_fraction

    def usable_wh(self, state: BatteryClock) -> float:
        return max(0.0, state.physical_soc - self.minimum_physical_soc) * self.scenario.nominal_battery_wh

    def spend(self, state: BatteryClock, energy_wh: float, duration_s: float) -> tuple[BatteryClock, bool]:
        """Advance flight time; report when the usable cutoff is reached/exceeded."""
        if not isfinite(energy_wh) or energy_wh < 0.0:
            raise ValueError("flight energy must be finite and nonnegative")
        if not isfinite(duration_s) or duration_s < 0.0:
            raise ValueError("flight duration must be finite and nonnegative")
        next_soc = state.physical_soc - energy_wh / self.scenario.nominal_battery_wh
        return BatteryClock(max(0.0, next_soc), state.elapsed_s + duration_s), next_soc <= self.minimum_physical_soc + 1e-12

    def charge_to(self, state: BatteryClock, target_physical_soc: float) -> BatteryClock:
        """Explicit station action: jump the clock to target SOC, including taper."""
        if not isfinite(target_physical_soc) or not state.physical_soc <= target_physical_soc <= 1.0:
            raise ValueError("target physical SOC must be within [current, 1]")
        duration_s = self.charger.seconds_to_soc(
            state.physical_soc, target_physical_soc, self.scenario.nominal_battery_wh
        )
        return BatteryClock(target_physical_soc, state.elapsed_s + duration_s)
