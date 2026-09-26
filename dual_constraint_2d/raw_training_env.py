"""Trial-and-error training plant for the same synthetic 2D task.

This path bypasses the evaluation safety filter.  Contact uses the frozen
step_flight repair and raw penalty; depletion ends the training episode without
rescue.  It never changes the protected historical simulator source.
"""

from __future__ import annotations

from math import floor, isfinite

import numpy as np

from nav3d.simulation import step_flight

from .environment import DualConstraintEnv, EnvironmentAction
from .shield import _controller
from .synthetic import SyntheticChargeState


class RawTrainingEnv(DualConstraintEnv):
    def step(self, action: EnvironmentAction | None = None) -> tuple[dict, float, bool, dict]:
        if self.done:
            raise RuntimeError("training episode complete; reset before another step")
        if action is None:
            raise ValueError("raw training requires an explicit action")
        event = "flight"
        reward = 0.0
        last_collision_count = self.state.collision_count
        last_penalty = self.state.raw_contact_penalty
        executed: list[tuple[float, float]] = []
        if self.mode == "docked":
            if action.charge_target_fraction is not None:
                fraction = float(action.charge_target_fraction)
                current = self.energy / self.charger.capacity
                if not isfinite(fraction) or not current + 1e-8 < fraction <= 1.0:
                    raise ValueError("invalid explicit charging target")
                charged = self.charger.charge_to_fraction(
                    SyntheticChargeState(self.energy, self.time_s), fraction
                )
                if charged.elapsed_s > self.horizon_s:
                    duration = self.horizon_s - self.time_s
                    self.energy = min(
                        self.charger.capacity,
                        self.energy + duration * self.charger.capacity / self.charger.full_charge_seconds,
                    )
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
                self.mode = "flight"
                event = "departed"
        elif self.mode != "flight" or action.charge_target_fraction is not None:
            raise ValueError("raw flight requires a planar velocity action")
        if self.mode == "flight" and event in ("flight", "departed"):
            velocity = np.asarray(action.desired_velocity_xy, dtype=float)
            if velocity.shape != (2,) or not np.isfinite(velocity).all():
                raise ValueError("desired velocity must be a finite 2-vector")
            if np.linalg.norm(velocity) > self.case.config.max_speed_mps + 1e-9:
                raise ValueError("desired velocity exceeds maximum speed")
            count = min(
                self.case.config.policy_hold_steps,
                max(0, floor((self.horizon_s - self.time_s + 1e-10) / self.case.config.physics_dt_s)),
            )
            controller = _controller(self.case)
            desired = np.array((velocity[0], velocity[1], 0.0))
            for _ in range(count):
                nominal = controller.nominal_acceleration(
                    self.state.position, self.state.velocity, desired
                )
                executed.append((float(nominal[0]), float(nominal[1])))
                cost = self.energy_model.flight_cost(
                    self.state.velocity[:2], nominal[:2], self.case.config.physics_dt_s
                )
                self.state, _ = step_flight(
                    self.case.world, self.state, nominal, controller.config
                )
                self.energy = max(0.0, self.energy - cost)
                self.time_s += self.case.config.physics_dt_s
                if self.energy <= 1e-10:
                    self.done = True
                    self.failure_reason = "depletion"
                    event = "depletion"
                    break
            if not self.done and self._at_target():
                self.completed_targets += 1
                reward += 1.0
                self.target = self._draw_target()
                event = "target_completed"
            elif not self.done and self._at_station():
                self.mode = "docked"
                event = "docked"
        reward += self.state.raw_contact_penalty - last_penalty
        if self.time_s >= self.horizon_s - 1e-10:
            self.done = True
            if event not in ("depletion", "target_completed"):
                event = "horizon"
        info = {
            "event": event,
            "raw_action": {
                "desired_velocity_xy": tuple(action.desired_velocity_xy),
                "charge_target_fraction": action.charge_target_fraction,
            },
            "executed_accelerations_xy": tuple(executed),
            "filter_reason": None,
            "block_accepted_policy": None,
            "failure_reason": self.failure_reason,
            "completed_targets": self.completed_targets,
            "collision_count": self.state.collision_count,
            "safety_cost": self.state.safety_cost,
            "new_collision_steps": self.state.collision_count - last_collision_count,
            "raw_contact_penalty": self.state.raw_contact_penalty - last_penalty,
            "charge_events": self.charge_events,
        }
        return self.observe(), reward, self.done, info
