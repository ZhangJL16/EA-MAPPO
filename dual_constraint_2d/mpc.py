"""Finite-horizon full-model predictive control on the shared action set.

This is a strong, explicitly bounded comparator, not a global optimum.  It
checks each candidate with the exact deterministic 0.05 s plant and joint
return certificate, then uses the same visibility graph as a terminal cost.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from nav3d.simulation import FlightState

from .action_adapter import ActionAdapter
from .environment import DualConstraintEnv
from .shield import CertifiedBlock, certify_policy_block


@dataclass(frozen=True)
class MPCStats:
    decisions: int
    candidates: int
    wall_seconds: float


class FullMapMPC:
    def __init__(self, case, *, depth: int = 2, beam_width: int = 2) -> None:
        if depth not in (1, 2) or beam_width <= 0:
            raise ValueError("MPC depth must be one or two and beam width positive")
        self.case = case
        self.adapter = ActionAdapter(case)
        self.depth = depth
        self.beam_width = beam_width
        self.decisions = 0
        self.candidates = 0
        self.wall_seconds = 0.0
        self._return_cost_cache: dict[tuple[float, float], float] = {}

    @property
    def stats(self) -> MPCStats:
        return MPCStats(self.decisions, self.candidates, self.wall_seconds)

    def _target_return_cost(self, env: DualConstraintEnv) -> float:
        goal = env.target.position_xy
        if goal not in self._return_cost_cache:
            from .backup import return_rollout
            state = FlightState(env.case.xyz(goal), np.zeros(3))
            witness = return_rollout(
                env.case, state, env.charger.capacity * 100.0,
                router=env.router, energy_model=env.energy_model,
            )
            self._return_cost_cache[goal] = (
                witness.energy_needed if witness.feasible else env.charger.capacity
            )
        return self._return_cost_cache[goal]

    def _terminal_score(
        self, env: DualConstraintEnv, block: CertifiedBlock,
        energy_before: float, return_from_goal: float,
    ) -> float:
        goal = env.target.position_xy
        endpoint = np.asarray(block.positions_xy[-1])
        route_length = self.adapter.route_direction(endpoint, goal)[1]
        remaining = energy_before - block.cumulative_energy[-1]
        nominal_goal_energy = route_length * 0.30
        reserve_need = nominal_goal_energy + return_from_goal + 0.05 * env.charger.capacity
        if remaining + 1e-9 < reserve_need:
            return -1000.0 - route_length
        return (
            -route_length / 5.0
            - 0.03 * block.cumulative_energy[-1]
            + 0.002 * (remaining - block.endpoint_backup.energy_needed)
            - 0.002 * block.collision_intervention_steps
        )

    def _candidate(
        self, env: DualConstraintEnv, state: FlightState, energy: float,
        backup, action_id: int, *, hold_steps: int,
    ) -> CertifiedBlock:
        observation = {
            **env.observe(),
            "position_xy": tuple(float(x) for x in state.position[:2]),
            "velocity_xy": tuple(float(x) for x in state.velocity[:2]),
            "energy": energy,
            "mode": "flight",
        }
        command = self.adapter.decode(action_id, observation)
        self.candidates += 1
        return certify_policy_block(
            env.case, state, energy, command.desired_velocity_xy,
            router=env.router, current_backup=backup,
            energy_model=env.energy_model, hold_steps=hold_steps,
        )

    def act(self, env: DualConstraintEnv) -> int:
        if env.mode == "return":
            return 9
        # The plant treats a final sub-physics-step remainder as horizon end.
        # No zero-length block may be sent to the safety certificate.
        if env._flight_step_limit() == 0:
            return 9
        began = perf_counter()
        self.decisions += 1
        if env.mode == "docked":
            # Current target only: choose the least of the available charge
            # levels that covers the shared reference witness plus a fixed
            # five-percent mismatch margin.  Full charge remains a candidate.
            needed = min(
                env.charger.capacity,
                env.target.reference_round_trip_energy + 0.05 * env.charger.capacity,
            )
            if env.energy + 1e-8 < needed:
                for action_id, fraction in ((10, 0.5), (11, 0.75), (12, 1.0)):
                    if env.charger.capacity * fraction + 1e-8 >= needed:
                        self.wall_seconds += perf_counter() - began
                        return action_id
            self.wall_seconds += perf_counter() - began
            return 4
        return_from_goal = self._target_return_cost(env)
        target_route = env.router.plan(env.state.position[:2], env.target.position_xy)
        if (
            env.energy <
            target_route.length_m * 0.30 + return_from_goal + 0.05 * env.charger.capacity
        ):
            self.wall_seconds += perf_counter() - began
            return 9
        first: list[tuple[float, int, CertifiedBlock]] = []
        count = env._flight_step_limit()
        for action_id in range(10):
            block = self._candidate(
                env, env.state, env.energy, env.backup, action_id, hold_steps=count
            )
            if not block.accepted_policy:
                continue
            score = self._terminal_score(env, block, env.energy, return_from_goal)
            first.append((score, action_id, block))
        if not first:
            self.wall_seconds += perf_counter() - began
            return 9
        first.sort(key=lambda entry: (-entry[0], entry[1]))
        best_score, best_action, _ = first[0]
        if self.depth == 2 and env.horizon_s - env.time_s >= 2 * env.case.config.policy_dt_s:
            for _, first_action, first_block in first[:self.beam_width]:
                next_state = FlightState(
                    env.case.xyz(first_block.positions_xy[-1]),
                    np.array((*first_block.velocities_xy[-1], 0.0)),
                    collision_count=env.state.collision_count,
                    safety_cost=env.state.safety_cost,
                    raw_contact_penalty=env.state.raw_contact_penalty,
                )
                next_energy = env.energy - first_block.cumulative_energy[-1]
                for next_action in (1, 2, 4, 5, 7, 8, 9):
                    second = self._candidate(
                        env, next_state, next_energy, first_block.endpoint_backup,
                        next_action, hold_steps=count,
                    )
                    if not second.accepted_policy:
                        continue
                    score = self._terminal_score(
                        env, second, next_energy, return_from_goal
                    ) - 0.01 * first_block.cumulative_energy[-1]
                    if score > best_score:
                        best_score = score
                        best_action = first_action
        self.wall_seconds += perf_counter() - began
        return best_action
