"""Continuing single-target synthetic environment with joint safety filtering.

This is a diagnostic simulator, not an M100 model or a Gym training adapter.
Policy actions occur at fixed decision instants; goal completion is checked at
the end of each certified block.  Return takeover cannot be interrupted.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import floor, isfinite

import numpy as np

from nav3d.simulation import FlightState

from .backup import ReturnRollout
from .routes import PlanarRouter
from .shield import certify_policy_block, continue_verified_return, execute_certified_block
from .synthetic import SyntheticChargeState, SyntheticCharger, SyntheticEnergy, make_synthetic_map
from .targets import SelectedTarget, select_next_target


@dataclass(frozen=True)
class EnvironmentAction:
    desired_velocity_xy: tuple[float, float] = (0.0, 0.0)
    charge_target_fraction: float | None = None


class DualConstraintEnv:
    def __init__(
        self,
        map_id: int,
        capacity: float,
        full_charge_seconds: float,
        *,
        horizon_s: float = 600.0,
        energy_model: SyntheticEnergy | None = None,
    ) -> None:
        if not all(isfinite(x) and x > 0 for x in (capacity, full_charge_seconds, horizon_s)):
            raise ValueError("environment capacity, charge time and horizon must be positive")
        self.case = make_synthetic_map(map_id)
        self.router = PlanarRouter(self.case)
        self.energy_model = energy_model or SyntheticEnergy()
        self.charger = SyntheticCharger(capacity, full_charge_seconds)
        self.horizon_s = horizon_s
        self.reset()

    def reset(self) -> dict:
        self.state = FlightState(self.case.xyz(self.case.station_xy), np.zeros(3))
        self.energy = self.charger.capacity
        self.time_s = 0.0
        self.mode = "docked"
        self.done = False
        self.failure_reason: str | None = None
        self.completed_targets = 0
        self.next_candidate_id = 0
        self.backup: ReturnRollout | None = None
        self.target = self._draw_target()
        self.charge_events = 0
        self.return_takeovers = 0
        self.collision_intervention_steps = 0
        self.energy_filter_events = 0
        self.rejected_departures = 0
        return self.observe()

    def _draw_target(self) -> SelectedTarget:
        result = select_next_target(
            self.case, self.router, self.charger.capacity,
            next_candidate_id=self.next_candidate_id,
        )
        self.next_candidate_id = result.next_candidate_id
        return result

    def observe(self) -> dict:
        return {
            "time_s": self.time_s,
            "position_xy": tuple(float(x) for x in self.state.position[:2]),
            "velocity_xy": tuple(float(x) for x in self.state.velocity[:2]),
            "energy": self.energy,
            "capacity": self.charger.capacity,
            "station_xy": self.case.station_xy,
            "target_xy": self.target.position_xy,
            "mode": self.mode,
            "obstacles": tuple(
                (float(o.center_xy[0]), float(o.center_xy[1]), float(o.radius))
                for o in self.case.world.obstacles
            ),
        }

    def _flight_step_limit(self) -> int:
        remaining = self.horizon_s - self.time_s
        return max(0, min(
            self.case.config.policy_hold_steps,
            floor((remaining + 1e-10) / self.case.config.physics_dt_s),
        ))

    def _at_station(self) -> bool:
        return (
            np.linalg.norm(self.state.position[:2] - self.case.station_xy) <= 1.5
            and np.linalg.norm(self.state.velocity[:2]) <= 0.1
        )

    def _at_target(self) -> bool:
        return (
            np.linalg.norm(self.state.position[:2] - self.target.position_xy) <= 1.5
            and np.linalg.norm(self.state.velocity[:2]) <= 0.1
        )

    def _perform_block(self, block) -> None:
        self.state, self.energy = execute_certified_block(
            self.case, self.state, self.energy, block
        )
        self.backup = block.endpoint_backup
        self.time_s += len(block.actions) * self.case.config.physics_dt_s
        self.collision_intervention_steps += block.collision_intervention_steps
        self.energy_filter_events += int(block.energy_filter_triggered)

    def step(self, action: EnvironmentAction | None = None) -> tuple[dict, float, bool, dict]:
        if self.done:
            raise RuntimeError("environment is complete; reset before another step")
        if self.time_s >= self.horizon_s - 1e-10:
            self.done = True
            return self.observe(), 0.0, True, {"event": "horizon"}
        reward = 0.0
        event = "none"
        block = None
        raw_action = None if action is None else {
            "desired_velocity_xy": tuple(action.desired_velocity_xy),
            "charge_target_fraction": action.charge_target_fraction,
        }
        if self.mode == "docked":
            if action is None:
                raise ValueError("docked state requires a charge or departure action")
            if action.charge_target_fraction is not None:
                current_fraction = self.energy / self.charger.capacity
                target_fraction = action.charge_target_fraction
                if not isfinite(target_fraction) or not current_fraction + 1e-8 < target_fraction <= 1.0:
                    raise ValueError("charge target must strictly exceed current fraction and be at most one")
                charged = self.charger.charge_to_fraction(
                    SyntheticChargeState(self.energy, self.time_s), target_fraction
                )
                if charged.elapsed_s > self.horizon_s:
                    remaining = self.horizon_s - self.time_s
                    increment = self.charger.capacity * remaining / self.charger.full_charge_seconds
                    self.energy = min(self.charger.capacity, self.energy + increment)
                    self.time_s = self.horizon_s
                else:
                    self.energy = charged.energy
                    self.time_s = charged.elapsed_s
                self.charge_events += 1
                event = "charged"
            elif np.linalg.norm(action.desired_velocity_xy) <= 1e-12:
                self.time_s = min(self.horizon_s, self.time_s + self.case.config.policy_dt_s)
                event = "dock_idle"
            else:
                count = self._flight_step_limit()
                if count == 0:
                    self.time_s = self.horizon_s
                    event = "horizon"
                else:
                    block = certify_policy_block(
                        self.case, self.state, self.energy, action.desired_velocity_xy,
                        router=self.router, current_backup=self.backup,
                        energy_model=self.energy_model, hold_steps=count,
                    )
                    if block.actions:
                        self._perform_block(block)
                        self.mode = "flight" if block.accepted_policy else "return"
                        self.return_takeovers += int(not block.accepted_policy)
                        if block.accepted_policy and self._at_target():
                            self.completed_targets += 1
                            reward = 1.0
                            self.target = self._draw_target()
                            event = "target_completed"
                        else:
                            event = "departed" if block.accepted_policy else "return_takeover"
                    elif block.reason == "no_verified_start_backup":
                        self.done = True
                        self.failure_reason = block.reason
                        event = "safety_failure"
                    else:
                        # A rejected attempt at a dock does not create motion;
                        # charge remains an explicit separate policy action.
                        self.time_s = min(self.horizon_s, self.time_s + self.case.config.policy_dt_s)
                        self.rejected_departures += 1
                        self.energy_filter_events += int(block.energy_filter_triggered)
                        event = "departure_rejected"
        elif self.mode == "flight":
            if action is None or action.charge_target_fraction is not None:
                raise ValueError("flight state requires a planar velocity action")
            count = self._flight_step_limit()
            if count == 0:
                self.time_s = self.horizon_s
                event = "horizon"
            else:
                block = certify_policy_block(
                    self.case, self.state, self.energy, action.desired_velocity_xy,
                    router=self.router, current_backup=self.backup,
                    energy_model=self.energy_model, hold_steps=count,
                )
                if block.actions:
                    self._perform_block(block)
                    if not block.accepted_policy:
                        self.mode = "return"
                        self.return_takeovers += 1
                        event = "return_takeover"
                    elif self._at_target():
                        self.completed_targets += 1
                        reward = 1.0
                        self.target = self._draw_target()
                        event = "target_completed"
                    elif self._at_station():
                        self.mode = "docked"
                        event = "docked"
                    else:
                        event = "flight"
                else:
                    self.done = True
                    self.failure_reason = block.reason
                    event = "safety_failure"
        else:
            if self.mode != "return" or self.backup is None or not self.backup.feasible:
                raise RuntimeError("invalid return takeover state")
            if self.backup.steps == 0:
                if not self._at_station():
                    self.done = True
                    self.failure_reason = "return_ended_away_from_station"
                    event = "safety_failure"
                else:
                    self.mode = "docked"
                    event = "docked"
            else:
                count = self._flight_step_limit()
                if count == 0:
                    self.time_s = self.horizon_s
                    event = "horizon"
                else:
                    block = continue_verified_return(self.backup, hold_steps=count)
                    self._perform_block(block)
                    if self.backup.steps == 0:
                        if not self._at_station():
                            self.done = True
                            self.failure_reason = "return_ended_away_from_station"
                            event = "safety_failure"
                        else:
                            self.mode = "docked"
                            event = "docked"
                    else:
                        event = "returning"
        if self.time_s >= self.horizon_s - 1e-10:
            self.done = True
        info = {
            "event": event,
            "raw_action": raw_action,
            "executed_accelerations_xy": () if block is None else tuple(
                (float(a[0]), float(a[1])) for a in block.actions
            ),
            "filter_reason": None if block is None else block.reason,
            "block_accepted_policy": None if block is None else block.accepted_policy,
            "verified_return_energy_at_block_end": None if block is None or block.endpoint_backup is None else block.endpoint_backup.energy_needed,
            "failure_reason": self.failure_reason,
            "completed_targets": self.completed_targets,
            "collision_count": self.state.collision_count,
            "return_takeovers": self.return_takeovers,
            "collision_intervention_steps": self.collision_intervention_steps,
            "energy_filter_events": self.energy_filter_events,
            "charge_events": self.charge_events,
            "rejected_departures": self.rejected_departures,
        }
        return self.observe(), reward, self.done, info
