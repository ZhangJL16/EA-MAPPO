from __future__ import annotations

import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from cert_runtime.generator_sac import GeneratorSAC, GeneratorSACConfig, PersistentGeneratorSAC
from cert_runtime.persistent_authority import ExecutionAuthority
from cert_runtime.goal_exposure import (
    GoalExposureAccumulator,
    batch_goal_diversity,
    goal_exposure_reset_boundary,
    goal_exposure_reset_seed,
    training_protocol_name,
)
from envs.certified_uav import PersistentMissionMode, make_random_persistent_uav_env
from envs.certified_uav.runtime_wrapper import RuntimeCyclePreparation
from scripts.persistent_generator_common import successor_context_for_transition, transition_from_cycle


class MultiGoalExposureProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.environment = make_random_persistent_uav_env("random_persistent_open.json", seed=0)

    def setUp(self) -> None:
        self.observation, self.reset_info = self.environment.reset(seed=0)

    def _collector_transition(self):
        context = self.reset_info["action_context"]
        actor_u = np.zeros(3, dtype=np.float64)
        next_observation, reward, terminated, truncated, info = self.environment.step(actor_u)
        next_context = successor_context_for_transition(self.environment, terminated=terminated)
        item = transition_from_cycle(
            self.observation,
            next_observation,
            actor_u,
            reward,
            terminated,
            truncated,
            0,
            context,
            next_context,
            info,
            collector_boundary=True,
        )
        return item, next_observation, next_context

    def _roundtrip_update(self, item):
        agent = PersistentGeneratorSAC(
            self.observation.size,
            GeneratorSACConfig(batch_size=1, hidden_dim=16, warmup_steps=0),
            seed=31,
        )
        self.assertTrue(agent.observe(item))
        sampled = agent.replay.sample(1)
        self.assertIs(sampled[0], item)
        return agent.update(sampled)

    def test_real_rl_runtime_transition_replay_update_roundtrip(self) -> None:
        item, _, _ = self._collector_transition()
        self.assertEqual(item.execution_authority, ExecutionAuthority.RL_GENERATOR.value)
        self.assertTrue(item.accepted)
        metrics = self._roundtrip_update(item)
        self.assertEqual(metrics["actor_status"], "updated")
        self.assertEqual(metrics["accepted_batch_count"], 1)

    def test_real_kappa_runtime_transition_replay_update_roundtrip(self) -> None:
        environment = self.environment
        environment._begin_backup("TEST_KAPPA_ROUNDTRIP")
        environment._context_cache_key = None
        context = environment._refresh_context()
        self.assertEqual(context["execution_authority"], ExecutionAuthority.KAPPA_BACKUP.value)
        next_observation, reward, terminated, truncated, info = environment.step(np.zeros(3))
        next_context = None if terminated or truncated else environment._refresh_context()
        if next_context is not None and next_context["persistent_mode"] == PersistentMissionMode.BACKUP_RECOVERY.name:
            self.assertLess(next_context["recovery_level"], context["recovery_level"])
            committed_child = environment.atlas._cells_by_id[
                context["recovery_cell_id"]
            ].successor_target_cell
            self.assertEqual(
                next_context["recovery_cell_id"],
                committed_child,
            )
        item = transition_from_cycle(
            self.observation, next_observation, np.zeros(3), reward,
            terminated, truncated, 0, context, next_context, info,
        )
        self.assertEqual(item.execution_authority, ExecutionAuthority.KAPPA_BACKUP.value)
        self.assertFalse(item.accepted)
        metrics = self._roundtrip_update(item)
        self.assertEqual(metrics["actor_status"], "zero-accepted-sample")
        self.assertEqual(metrics["fallback_batch_count"], 1)

    def test_positive_rank_kappa_terminal_overlap_preserves_committed_child(self) -> None:
        environment = self.environment
        environment.plant.state.position = np.array([0.4999, 0.5, 1.0], dtype=np.float64)
        environment.plant.state.velocity = np.zeros(3, dtype=np.float64)
        environment.plant.state.energy = 30.0
        environment.task_env.mode = PersistentMissionMode.BACKUP_RECOVERY
        environment.task_env.phase = environment.task_env.mode
        environment.atlas.reset()
        environment.atlas.recovery_active = True
        environment._context_cache_key = None
        context = environment._refresh_context()
        self.assertEqual(context["execution_authority"], ExecutionAuthority.KAPPA_BACKUP.value)
        self.assertGreater(context["recovery_successor_level"], 0)
        committed_child = context["recovery_successor_cell_id"]
        self.assertTrue(environment.plant.terminal.is_charge_admissible(environment.plant.state))

        _, _, terminated, truncated, info = environment.step(np.zeros(3))

        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["command_source"], "kappa")
        self.assertTrue(info["covered_at_publication"])
        self.assertEqual(info["persistent_mode"], PersistentMissionMode.BACKUP_RECOVERY.name)
        self.assertFalse(info["kappa_station_arrival"])
        self.assertEqual(environment.atlas.active_cell_id, committed_child)
        successor_context = environment._refresh_context()
        self.assertEqual(successor_context["recovery_cell_id"], committed_child)
        self.assertEqual(
            successor_context["recovery_level"],
            context["recovery_successor_level"],
        )

    def test_level_one_kappa_arrives_only_through_committed_level_zero_child(self) -> None:
        environment = self.environment
        level_one = next(cell for cell in environment.atlas.manifest.cells if cell.level == 1)
        environment.plant.state.position = np.asarray(level_one.reference_position, dtype=np.float64)
        environment.plant.state.velocity = np.asarray(level_one.reference_velocity, dtype=np.float64)
        environment.plant.state.energy = 30.0
        environment.task_env.mode = PersistentMissionMode.BACKUP_RECOVERY
        environment.task_env.phase = environment.task_env.mode
        environment.atlas.recovery_active = True
        environment.atlas.active_cell_id = level_one.cell_id
        environment._context_cache_key = None
        context = environment._refresh_context()
        self.assertEqual(context["recovery_level"], 1)
        self.assertEqual(context["recovery_successor_level"], 0)
        pending_task_id = environment.task_env.manager.current_task.task_id
        charged_before = environment.metrics.energy_charged

        with patch.object(environment, "_apply_charging", wraps=environment._apply_charging) as apply_charging:
            _, _, terminated, truncated, info = environment.step(np.zeros(3))

        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["command_source"], "kappa")
        self.assertTrue(info["covered_at_publication"])
        self.assertTrue(info["kappa_station_arrival"])
        self.assertEqual(info["persistent_mode"], PersistentMissionMode.CHARGING_RL.name)
        self.assertEqual(environment.task_env.manager.current_task.task_id, pending_task_id)
        self.assertEqual(environment.metrics.energy_charged, charged_before)
        apply_charging.assert_not_called()

    def test_kappa_committed_child_dependency_drift_stops_before_charging(self) -> None:
        environment = self.environment
        level_one = next(cell for cell in environment.atlas.manifest.cells if cell.level == 1)
        environment.plant.state.position = np.asarray(level_one.reference_position, dtype=np.float64)
        environment.plant.state.velocity = np.asarray(level_one.reference_velocity, dtype=np.float64)
        environment.plant.state.energy = 30.0
        environment.task_env.mode = PersistentMissionMode.BACKUP_RECOVERY
        environment.task_env.phase = environment.task_env.mode
        environment.atlas.recovery_active = True
        environment.atlas.active_cell_id = level_one.cell_id
        environment._context_cache_key = None
        context = environment._refresh_context()
        self.assertEqual(context["recovery_successor_level"], 0)
        provider = environment.certificate_provider
        original_check = provider.committed_recovery_successor_is_valid
        original_geometry_version = environment.runtime.geometry.version
        charged_before = environment.metrics.energy_charged

        def drift_after_child_check(state, selected=None):
            valid = original_check(state, selected)
            environment.runtime.geometry.version += 1
            return valid

        try:
            with (
                patch.object(provider, "committed_recovery_successor_is_valid", side_effect=drift_after_child_check),
                patch.object(environment, "_apply_charging", wraps=environment._apply_charging) as apply_charging,
            ):
                _, _, terminated, truncated, info = environment.step(np.zeros(3))
        finally:
            environment.runtime.geometry.version = original_geometry_version
            environment._context_cache_key = None

        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["failure_reason"], "kappa_successor_snapshot_changed")
        self.assertEqual(info["command_source"], "kappa")
        self.assertTrue(info["covered_at_publication"])
        self.assertEqual(info["execution_authority"], ExecutionAuthority.KAPPA_BACKUP.value)
        self.assertEqual(info["persistent_mode"], PersistentMissionMode.FAILURE.name)
        self.assertFalse(info["kappa_station_arrival"])
        self.assertEqual(environment.metrics.energy_charged, charged_before)
        apply_charging.assert_not_called()

    def test_kappa_committed_child_identity_cannot_change_after_readback(self) -> None:
        environment = self.environment
        environment._begin_backup("TEST_KAPPA_COMMITMENT_IDENTITY_RACE")
        environment._context_cache_key = None
        context = environment._refresh_context()
        self.assertGreater(context["recovery_successor_level"], 0)
        provider = environment.certificate_provider
        original_check = provider.committed_recovery_successor_is_valid

        def clear_commitment_after_valid_readback(state, selected=None):
            valid = original_check(state, selected)
            environment.atlas.active_cell_id = None
            return valid

        with patch.object(
            provider,
            "committed_recovery_successor_is_valid",
            side_effect=clear_commitment_after_valid_readback,
        ):
            _, _, terminated, truncated, info = environment.step(np.zeros(3))

        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["failure_reason"], "kappa_successor_snapshot_changed")
        self.assertEqual(info["command_source"], "kappa")
        self.assertTrue(info["covered_at_publication"])
        self.assertEqual(info["persistent_mode"], PersistentMissionMode.FAILURE.name)
        self.assertFalse(info["kappa_station_arrival"])

    def test_real_charger_hold_runtime_transition_replay_update_roundtrip(self) -> None:
        environment = self.environment
        environment.plant.state.position = environment.plant.scenario.station_position.copy()
        environment.plant.state.velocity = np.zeros(3, dtype=np.float64)
        environment.plant.state.energy = max(
            environment.plant.scenario.terminal.minimum_energy + 1.0,
            5.0,
        )
        environment.task_env.mode = PersistentMissionMode.CHARGING_RL
        environment.task_env.phase = environment.task_env.mode
        environment._context_cache_key = None
        real_context = environment._refresh_context()
        context = dict(real_context)
        context.update({
            "persistent_mode": PersistentMissionMode.CHARGING_RL.name,
            "execution_authority": ExecutionAuthority.CHARGER_CONSTRAINED.value,
            "execution_authority_reason": "TEST_CERTIFIED_HOLD",
            "generator_executable": False,
            "backup_required": False,
            "charging_restriction": True,
            "charging_support_verified": False,
            "station_hold_valid": True,
            "departure_allowed": False,
        })
        decision = SimpleNamespace(
            authority=ExecutionAuthority.CHARGER_CONSTRAINED,
            reason="TEST_CERTIFIED_HOLD",
            generator_executable=False,
            kappa_required=False,
            departure_allowed=False,
            charging_restriction=True,
            station_hold_required=True,
        )
        environment._last_authority_decision = decision
        environment._context_cache = context
        next_observation, reward, terminated, truncated, info = environment.step(np.zeros(3))
        next_context = environment._refresh_context()
        item = transition_from_cycle(
            self.observation, next_observation, np.zeros(3), reward,
            terminated, truncated, 0, context, next_context, info,
        )
        self.assertEqual(item.execution_authority, ExecutionAuthority.CHARGER_CONSTRAINED.value)
        self.assertFalse(item.accepted)
        self.assertIsNotNone(item.task_goal)
        environment._refresh_context()
        rebuilt = environment.task_env.build_observation(
            environment.runtime._map_encoding(),
            environment.runtime._corridor_encoding(),
        )
        np.testing.assert_array_equal(next_observation, rebuilt)
        metrics = self._roundtrip_update(item)
        self.assertEqual(metrics["generator_target_count"], 1)
        self.assertEqual(metrics["charger_atomic_target_count"], 0)
        self.assertEqual(metrics["actor_status"], "zero-accepted-sample")

    def test_unavailable_charger_hold_is_explicit_fail_closed_without_recovery_commit(self) -> None:
        environment = self.environment
        environment.plant.state.position = environment.plant.scenario.station_position.copy()
        environment.plant.state.velocity = np.zeros(3, dtype=np.float64)
        environment.plant.state.energy = max(
            environment.plant.scenario.terminal.minimum_energy + 1.0,
            5.0,
        )
        environment.task_env.mode = PersistentMissionMode.CHARGING_RL
        environment.task_env.phase = environment.task_env.mode
        environment._context_cache_key = None
        context = dict(environment._refresh_context())
        context.update({
            "persistent_mode": PersistentMissionMode.CHARGING_RL.name,
            "execution_authority": ExecutionAuthority.CHARGER_CONSTRAINED.value,
            "execution_authority_reason": "TEST_HOLD_UNAVAILABLE",
            "generator_executable": False,
            "backup_required": False,
            "charging_restriction": True,
            "charging_support_verified": False,
            "station_hold_valid": True,
            "departure_allowed": False,
        })
        environment._last_authority_decision = SimpleNamespace(
            authority=ExecutionAuthority.CHARGER_CONSTRAINED,
            reason="TEST_HOLD_UNAVAILABLE",
            generator_executable=False,
            kappa_required=False,
            departure_allowed=False,
            charging_restriction=True,
            station_hold_required=True,
        )
        runtime_provider = environment.runtime.mission_provider
        self.assertIsNotNone(runtime_provider)
        active_cell_before = runtime_provider.active_cell_id
        with (
            patch.object(environment, "_refresh_context", return_value=context),
            patch.object(environment.certificate_provider, "certified_station_hold_action", return_value=None),
            patch.object(runtime_provider, "commit_execution", wraps=runtime_provider.commit_execution) as commit_execution,
        ):
            next_observation, reward, terminated, truncated, info = environment.step(np.zeros(3))

        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["execution_authority"], ExecutionAuthority.FAIL_CLOSED.value)
        self.assertEqual(info["command_source"], "uncertified_emergency_brake")
        self.assertFalse(info["covered_at_publication"])
        self.assertFalse(info["station_hold_valid"])
        self.assertEqual(runtime_provider.active_cell_id, active_cell_before)
        commit_execution.assert_not_called()
        item = transition_from_cycle(
            self.observation,
            next_observation,
            np.zeros(3),
            reward,
            terminated,
            truncated,
            0,
            context,
            None,
            info,
        )
        metrics = self._roundtrip_update(item)
        self.assertEqual(metrics["fail_closed_target_count"], 1)
        self.assertEqual(metrics["charger_atomic_target_count"], 0)

    def test_charger_hold_version_change_is_fail_closed_before_motion(self) -> None:
        environment = self.environment
        environment.plant.state.position = environment.plant.scenario.station_position.copy()
        environment.plant.state.velocity = np.zeros(3, dtype=np.float64)
        environment.plant.state.energy = max(
            environment.plant.scenario.terminal.minimum_energy + 1.0,
            5.0,
        )
        environment.task_env.mode = PersistentMissionMode.CHARGING_RL
        environment.task_env.phase = environment.task_env.mode
        environment._context_cache_key = None
        context = dict(environment._refresh_context())
        context.update({
            "persistent_mode": PersistentMissionMode.CHARGING_RL.name,
            "execution_authority": ExecutionAuthority.CHARGER_CONSTRAINED.value,
            "execution_authority_reason": "TEST_HOLD_VERSION_RACE",
            "generator_executable": False,
            "backup_required": False,
            "charging_restriction": True,
            "charging_support_verified": False,
            "station_hold_valid": True,
            "departure_allowed": False,
        })
        environment._last_authority_decision = SimpleNamespace(
            authority=ExecutionAuthority.CHARGER_CONSTRAINED,
            reason="TEST_HOLD_VERSION_RACE",
            generator_executable=False,
            kappa_required=False,
            departure_allowed=False,
            charging_restriction=True,
            station_hold_required=True,
        )
        provider = environment.certificate_provider
        self.assertIsNotNone(provider)
        original_hold_action = provider.certified_station_hold_action
        original_geometry_version = environment.runtime.geometry.version
        before_fail_closed_steps = environment.metrics.fail_closed_steps
        before_charger_steps = environment.metrics.charger_constrained_steps
        mutated = False

        def mutate_during_hold_check(state):
            nonlocal mutated
            if not mutated:
                environment.runtime.geometry.version += 1
                mutated = True
            return original_hold_action(state)

        try:
            with (
                patch.object(environment, "_refresh_context", return_value=context),
                patch.object(provider, "certified_station_hold_action", side_effect=mutate_during_hold_check),
                patch.object(environment.charging, "step", wraps=environment.charging.step) as charging_step,
            ):
                _, _, terminated, truncated, info = environment.step(np.zeros(3))
        finally:
            environment.runtime.geometry.version = original_geometry_version

        self.assertTrue(mutated)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["failure_reason"], "certificate_version_changed")
        self.assertEqual(info["execution_authority"], ExecutionAuthority.FAIL_CLOSED.value)
        self.assertEqual(info["command_source"], "uncertified_emergency_brake")
        self.assertFalse(info["covered_at_publication"])
        self.assertEqual(environment.metrics.fail_closed_steps, before_fail_closed_steps + 1)
        self.assertEqual(environment.metrics.charger_constrained_steps, before_charger_steps)
        charging_step.assert_not_called()

    def test_charger_hold_rejects_a_different_unverified_selected_action(self) -> None:
        environment = self.environment
        environment.plant.state.position = np.array([0.32, 0.42, 0.92], dtype=np.float64)
        environment.plant.state.velocity = np.array([-0.04, -0.04, -0.032], dtype=np.float64)
        environment.plant.state.energy = 5.0
        environment.task_env.mode = PersistentMissionMode.CHARGING_RL
        environment.task_env.phase = environment.task_env.mode
        environment._context_cache_key = None
        context = dict(environment._refresh_context())
        context.update({
            "persistent_mode": PersistentMissionMode.CHARGING_RL.name,
            "execution_authority": ExecutionAuthority.CHARGER_CONSTRAINED.value,
            "execution_authority_reason": "TEST_EXACT_HOLD_ACTION_BINDING",
            "generator_executable": False,
            "backup_required": False,
            "charging_restriction": True,
            "charging_support_verified": False,
            "station_hold_valid": True,
            "departure_allowed": False,
        })
        environment._last_authority_decision = SimpleNamespace(
            authority=ExecutionAuthority.CHARGER_CONSTRAINED,
            reason="TEST_EXACT_HOLD_ACTION_BINDING",
            generator_executable=False,
            kappa_required=False,
            departure_allowed=False,
            charging_restriction=True,
            station_hold_required=True,
        )
        provider = environment.certificate_provider
        bad_hold = np.array([-0.18, -0.18, -0.08], dtype=np.float64)
        state_before = environment.plant.state.copy()
        self.assertFalse(provider.certified_station_hold_action_is_valid(
            environment.runtime._certificate_state(),
            bad_hold,
        ))

        with (
            patch.object(environment, "_refresh_context", return_value=context),
            patch.object(provider, "certified_station_hold_action", return_value=bad_hold),
            patch.object(environment.charging, "step", wraps=environment.charging.step) as charging_step,
        ):
            _, _, terminated, truncated, info = environment.step(np.zeros(3))

        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["execution_authority"], ExecutionAuthority.FAIL_CLOSED.value)
        self.assertEqual(info["execution_authority_reason"], "CERTIFIED_CHARGER_HOLD_UNAVAILABLE")
        self.assertEqual(info["command_source"], "uncertified_emergency_brake")
        self.assertFalse(info["covered_at_publication"])
        self.assertFalse(np.array_equal(
            np.asarray(info["telemetry"].action_trace.published),
            bad_hold,
        ))
        charging_step.assert_not_called()

    def test_charger_hold_rejects_nominal_terminal_boundary_without_uncertainty_membership(self) -> None:
        environment = self.environment
        terminal = environment.plant.scenario.terminal
        environment.plant.state.position = terminal.position_low.copy()
        environment.plant.state.velocity = np.array([0.025, 0.025, 0.02], dtype=np.float64)
        environment.plant.state.energy = 5.0
        environment.task_env.mode = PersistentMissionMode.CHARGING_RL
        environment.task_env.phase = environment.task_env.mode
        environment._context_cache_key = None
        state = environment.runtime._certificate_state()
        self.assertTrue(terminal.is_charge_admissible(environment.plant.state))
        self.assertFalse(
            environment.certificate_provider.certified_station_hold_action_is_valid(
                state,
                np.zeros(3, dtype=np.float64),
            )
        )
        self.assertFalse(environment.certificate_provider.certified_station_hold(state))

    def test_charger_hold_rejects_unavailable_station_before_motion(self) -> None:
        environment = self.environment
        environment.plant.state.position = environment.plant.scenario.station_position.copy()
        environment.plant.state.velocity = np.zeros(3, dtype=np.float64)
        environment.plant.state.energy = 5.0
        environment.task_env.mode = PersistentMissionMode.CHARGING_RL
        environment.task_env.phase = environment.task_env.mode
        environment._context_cache_key = None
        context = dict(environment._refresh_context())
        unavailable = replace(environment.charging.config, station_available=False)

        with (
            patch.object(environment.charging, "config", unavailable),
            patch.object(environment.charging, "step", wraps=environment.charging.step) as charging_step,
        ):
            result, failure = environment._execute_certified_station_hold(context)

        self.assertIsNone(result)
        self.assertEqual(failure, "CERTIFIED_CHARGER_HOLD_UNAVAILABLE")
        charging_step.assert_not_called()

    def test_open_gate_task_publication_atomically_departs_even_inside_station_overlap(self) -> None:
        environment = self.environment
        environment.plant.state.position = environment.plant.scenario.station_position.copy()
        environment.plant.state.velocity = np.zeros(3, dtype=np.float64)
        environment.plant.state.energy = environment.charging.config.battery_capacity
        environment.task_env.enter_charging(voluntary=True)
        environment._context_cache_key = None
        context = environment._refresh_context()
        self.assertTrue(context["departure_allowed"])
        self.assertEqual(context["execution_authority"], ExecutionAuthority.RL_GENERATOR.value)
        self.assertTrue(context["generator_executable"])
        pending_task = environment.task_env.manager.current_task
        self.assertIsNotNone(pending_task)
        pending_task_id = pending_task.task_id
        charged_before = environment.metrics.energy_charged

        with patch.object(environment, "_apply_charging", wraps=environment._apply_charging) as apply_charging:
            _, _, terminated, truncated, info = environment.step(np.zeros(3))

        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertTrue(info["accepted"])
        self.assertEqual(info["command_source"], "task")
        self.assertTrue(info["covered_at_publication"])
        self.assertEqual(info["execution_authority"], ExecutionAuthority.RL_GENERATOR.value)
        self.assertTrue(environment.plant.terminal.is_charge_admissible(environment.plant.state))
        self.assertEqual(info["persistent_mode"], PersistentMissionMode.TASK_RL.name)
        self.assertFalse(info["charging"])
        self.assertTrue(info["departure_attempt"])
        self.assertEqual(environment.task_env.manager.current_task.task_id, pending_task_id)
        self.assertEqual(environment.metrics.energy_charged, charged_before)
        apply_charging.assert_not_called()

    def test_normal_arrival_is_certified_zero_motion_mode_update_without_same_step_charge(self) -> None:
        environment = self.environment
        environment.plant.state.position = environment.plant.scenario.station_position.copy()
        environment.plant.state.velocity = np.zeros(3, dtype=np.float64)
        environment.plant.state.energy = environment.charging.config.battery_capacity
        environment.task_env.mode = PersistentMissionMode.TASK_RL
        environment.task_env.phase = environment.task_env.mode
        environment._context_cache_key = None
        context = environment._refresh_context()
        self.assertEqual(context["execution_authority"], ExecutionAuthority.RL_GENERATOR.value)
        pending_task = environment.task_env.manager.current_task
        self.assertIsNotNone(pending_task)
        pending_task_id = pending_task.task_id
        charged_before = environment.metrics.energy_charged
        energy_before = environment.plant.state.energy

        with patch.object(environment, "_apply_charging", wraps=environment._apply_charging) as apply_charging:
            _, _, terminated, truncated, info = environment.step(np.zeros(3))

        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertTrue(info["accepted"])
        self.assertEqual(info["command_source"], "task")
        self.assertTrue(info["covered_at_publication"])
        self.assertEqual(info["execution_authority"], ExecutionAuthority.RL_GENERATOR.value)
        self.assertEqual(info["persistent_mode"], PersistentMissionMode.CHARGING_RL.name)
        self.assertTrue(info["charging"])
        self.assertTrue(info["voluntary_station_arrival"])
        self.assertFalse(info["departure_attempt"])
        self.assertEqual(environment.task_env.manager.current_task.task_id, pending_task_id)
        self.assertEqual(environment.metrics.energy_charged, charged_before)
        self.assertLessEqual(environment.plant.state.energy, energy_before)
        apply_charging.assert_not_called()

    def test_arrival_dependency_drift_terminates_without_charging_mode_or_gain(self) -> None:
        environment = self.environment
        runtime = environment.runtime
        environment.plant.state.position = environment.plant.scenario.station_position.copy()
        environment.plant.state.velocity = np.zeros(3, dtype=np.float64)
        environment.plant.state.energy = environment.charging.config.battery_capacity
        environment.task_env.mode = PersistentMissionMode.TASK_RL
        environment.task_env.phase = environment.task_env.mode
        environment._context_cache_key = None
        context = environment._refresh_context()
        self.assertEqual(context["execution_authority"], ExecutionAuthority.RL_GENERATOR.value)
        provider = environment.certificate_provider
        self.assertIsNotNone(provider)
        original_arrival_check = provider.certified_station_hold
        original_geometry_version = runtime.geometry.version
        charged_before = environment.metrics.energy_charged

        def drift_after_valid_arrival_check(state):
            valid = original_arrival_check(state)
            runtime.geometry.version += 1
            return valid

        try:
            with (
                patch.object(provider, "certified_station_hold", side_effect=drift_after_valid_arrival_check),
                patch.object(environment, "_apply_charging", wraps=environment._apply_charging) as apply_charging,
            ):
                _, _, terminated, truncated, info = environment.step(np.zeros(3))
        finally:
            runtime.geometry.version = original_geometry_version
            environment._context_cache_key = None

        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["failure_reason"], "arrival_certificate_version_changed")
        self.assertEqual(info["command_source"], "task")
        self.assertTrue(info["covered_at_publication"])
        self.assertEqual(info["execution_authority"], ExecutionAuthority.RL_GENERATOR.value)
        self.assertEqual(info["persistent_mode"], PersistentMissionMode.FAILURE.name)
        self.assertFalse(info["voluntary_station_arrival"])
        self.assertFalse(info["charging"])
        self.assertEqual(environment.metrics.energy_charged, charged_before)
        apply_charging.assert_not_called()

    def test_open_charger_without_departure_support_fails_closed_without_kappa_or_charge(self) -> None:
        environment = self.environment
        environment.plant.state.position = environment.plant.scenario.station_position.copy()
        environment.plant.state.velocity = np.zeros(3, dtype=np.float64)
        environment.plant.state.energy = environment.charging.config.battery_capacity
        environment.task_env.enter_charging(voluntary=True)
        environment._context_cache_key = None
        original = environment.runtime.preview_next_action_context()
        unavailable = dict(original)
        unavailable.update({
            "generator_available": False,
            "c": None,
            "G": None,
            "recoverability_action_verified": False,
        })
        before_kappa = environment.metrics.kappa_backup_steps
        before_charge = environment.metrics.energy_charged
        with patch.object(environment.runtime, "preview_next_action_context", return_value=unavailable):
            environment._context_cache_key = None
            context = environment._refresh_context()
            self.assertTrue(context["departure_allowed"])
            self.assertEqual(context["execution_authority"], ExecutionAuthority.FAIL_CLOSED.value)
            _, _, terminated, _, info = environment.step(np.zeros(3))

        self.assertTrue(terminated)
        self.assertEqual(info["execution_authority"], ExecutionAuthority.FAIL_CLOSED.value)
        self.assertEqual(info["execution_authority_reason"], "NO_DEPARTURE_GENERATOR_SET")
        self.assertEqual(info["command_source"], "uncertified_emergency_brake")
        self.assertFalse(info["charging"])
        self.assertEqual(environment.metrics.kappa_backup_steps, before_kappa)
        self.assertEqual(environment.metrics.energy_charged, before_charge)

    def test_normal_zero_rank_refusal_fails_closed_without_fictitious_kappa(self) -> None:
        environment = self.environment
        environment.plant.state.position = environment.plant.scenario.station_position.copy()
        environment.plant.state.velocity = np.zeros(3, dtype=np.float64)
        environment.task_env.mode = PersistentMissionMode.TASK_RL
        environment.task_env.phase = environment.task_env.mode
        environment._context_cache_key = None
        original = environment.runtime.preview_next_action_context()
        unavailable = dict(original)
        unavailable.update({"generator_available": False, "c": None, "G": None})
        before_kappa = environment.metrics.kappa_backup_steps
        with patch.object(environment.runtime, "preview_next_action_context", return_value=unavailable):
            environment._context_cache_key = None
            context = environment._refresh_context()
            self.assertEqual(context["recovery_level"], 0)
            self.assertEqual(context["execution_authority"], ExecutionAuthority.FAIL_CLOSED.value)
            _, _, terminated, _, info = environment.step(np.zeros(3))

        self.assertTrue(terminated)
        self.assertEqual(info["execution_authority"], ExecutionAuthority.FAIL_CLOSED.value)
        self.assertEqual(info["execution_authority_reason"], "ZERO_RANK_RECOVERY_ACTION_UNDEFINED")
        self.assertEqual(environment.metrics.kappa_backup_steps, before_kappa)

    def test_transition_uses_actual_publication_context_not_stale_collector_argument(self) -> None:
        environment = self.environment
        next_observation, reward, terminated, truncated, info = environment.step(np.zeros(3))
        self.assertFalse(terminated)
        publication_context = info["action_context"]
        stale_context = dict(publication_context)
        stale_context.update({
            "certificate_epoch": "stale-collector-epoch",
            "geometry_version": "stale-geometry",
            "c": np.asarray(publication_context["c"]) + 0.25,
            "G": np.asarray(publication_context["G"]) * 0.5,
        })
        next_context = successor_context_for_transition(environment, terminated=terminated)
        item = transition_from_cycle(
            self.observation,
            next_observation,
            np.zeros(3),
            reward,
            terminated,
            truncated,
            0,
            stale_context,
            next_context,
            info,
        )
        self.assertEqual(item.certificate_epoch, publication_context["certificate_epoch"])
        self.assertEqual(item.geometry_version, str(publication_context["geometry_version"]))
        np.testing.assert_allclose(item.c, publication_context["c"])
        np.testing.assert_allclose(item.G, publication_context["G"])

    def test_returned_observation_matches_refreshed_successor_after_normal_and_kappa(self) -> None:
        environment = self.environment
        normal_next, _, normal_terminated, _, _ = environment.step(np.zeros(3))
        self.assertFalse(normal_terminated)
        environment._refresh_context()
        normal_rebuilt = environment.task_env.build_observation(
            environment.runtime._map_encoding(),
            environment.runtime._corridor_encoding(),
        )
        np.testing.assert_array_equal(normal_next, normal_rebuilt)

        environment.reset(seed=0)
        environment._begin_backup("TEST_SUCCESSOR_OBSERVATION")
        environment._context_cache_key = None
        kappa_next, _, kappa_terminated, _, _ = environment.step(np.zeros(3))
        self.assertFalse(kappa_terminated)
        environment._refresh_context()
        kappa_rebuilt = environment.task_env.build_observation(
            environment.runtime._map_encoding(),
            environment.runtime._corridor_encoding(),
        )
        np.testing.assert_array_equal(kappa_next, kappa_rebuilt)

    def test_real_fail_closed_runtime_transition_replay_update_roundtrip(self) -> None:
        environment = self.environment
        context = dict(self.reset_info["action_context"])
        context.update({
            "execution_authority": ExecutionAuthority.FAIL_CLOSED.value,
            "execution_authority_reason": "TEST_FAIL_CLOSED",
            "generator_executable": False,
            "backup_required": False,
        })
        decision = SimpleNamespace(
            authority=ExecutionAuthority.FAIL_CLOSED,
            reason="TEST_FAIL_CLOSED",
            generator_executable=False,
            kappa_required=False,
            departure_allowed=False,
            charging_restriction=False,
            station_hold_required=False,
        )
        environment._last_authority_decision = decision
        provider = environment.runtime.mission_provider
        self.assertIsNotNone(provider)
        active_cell_before = provider.active_cell_id
        with (
            patch.object(environment, "_refresh_context", return_value=context),
            patch.object(provider, "commit_execution", wraps=provider.commit_execution) as commit_execution,
        ):
            next_observation, reward, terminated, truncated, info = environment.step(np.zeros(3))
            item = transition_from_cycle(
                self.observation, next_observation, np.zeros(3), reward,
                terminated, truncated, 0, context, None, info,
            )
        self.assertTrue(item.terminated)
        self.assertEqual(item.execution_authority, ExecutionAuthority.FAIL_CLOSED.value)
        self.assertFalse(item.backup_triggered)
        self.assertEqual(info["command_source"], "uncertified_emergency_brake")
        self.assertFalse(info["covered_at_publication"])
        self.assertEqual(provider.active_cell_id, active_cell_before)
        commit_execution.assert_not_called()
        metrics = self._roundtrip_update(item)
        self.assertEqual(metrics["fail_closed_target_count"], 1)
        self.assertEqual(metrics["actor_status"], "zero-accepted-sample")

    def test_kappa_preview_losing_publication_coverage_is_fail_closed(self) -> None:
        environment = self.environment
        environment._begin_backup("TEST_KAPPA_PUBLICATION_RACE")
        environment._context_cache_key = None
        context = environment._refresh_context()
        self.assertEqual(context["execution_authority"], ExecutionAuthority.KAPPA_BACKUP.value)
        valid_preparation = environment.runtime.last_preparation
        self.assertIsNotNone(valid_preparation)
        invalid_preparation = RuntimeCyclePreparation(
            valid_preparation.state,
            valid_preparation.task_observation,
            valid_preparation.closure_result,
            None,
            "INJECTED_PUBLICATION_COVERAGE_LOSS",
        )
        before_kappa_steps = environment.metrics.kappa_backup_steps
        before_fail_closed_steps = environment.metrics.fail_closed_steps
        before_backup_entries = environment.metrics.backup_recovery_count
        provider = environment.runtime.mission_provider
        self.assertIsNotNone(provider)
        active_cell_before = provider.active_cell_id
        with (
            patch.object(environment.runtime, "prepare_certificate_cycle", return_value=invalid_preparation),
            patch.object(provider, "commit_execution", wraps=provider.commit_execution) as commit_execution,
        ):
            next_observation, reward, terminated, truncated, info = environment.step(np.zeros(3))

        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["failure_reason"], "injected_publication_coverage_loss")
        self.assertEqual(info["command_source"], "uncertified_emergency_brake")
        self.assertFalse(info["covered_at_publication"])
        self.assertEqual(info["execution_authority"], ExecutionAuthority.FAIL_CLOSED.value)
        self.assertFalse(info["backup_triggered"])
        self.assertEqual(environment.metrics.kappa_backup_steps, before_kappa_steps)
        self.assertEqual(environment.metrics.fail_closed_steps, before_fail_closed_steps + 1)
        self.assertEqual(environment.metrics.backup_recovery_count, before_backup_entries)
        commit_execution.assert_not_called()
        self.assertEqual(provider.active_cell_id, active_cell_before)

        item = transition_from_cycle(
            self.observation,
            next_observation,
            np.zeros(3),
            reward,
            terminated,
            truncated,
            0,
            context,
            None,
            info,
        )
        self.assertTrue(item.terminated)
        self.assertEqual(item.execution_authority, ExecutionAuthority.FAIL_CLOSED.value)
        metrics = self._roundtrip_update(item)
        self.assertEqual(metrics["fail_closed_target_count"], 1)
        self.assertEqual(metrics["kappa_target_count"], 0)

    def test_rl_preview_losing_final_recovery_coverage_is_fail_closed_without_child_commit(self) -> None:
        environment = self.environment
        environment._context_cache_key = None
        context = environment._refresh_context()
        self.assertEqual(context["execution_authority"], ExecutionAuthority.RL_GENERATOR.value)
        valid_preparation = environment.runtime.last_preparation
        self.assertIsNotNone(valid_preparation)
        invalid_preparation = RuntimeCyclePreparation(
            valid_preparation.state,
            valid_preparation.task_observation,
            valid_preparation.closure_result,
            None,
            "INJECTED_FINAL_RECOVERY_COVERAGE_LOSS",
        )
        provider = environment.runtime.mission_provider
        self.assertIsNotNone(provider)
        active_cell_before = provider.active_cell_id
        before_kappa_steps = environment.metrics.kappa_backup_steps
        before_fail_closed_steps = environment.metrics.fail_closed_steps
        with (
            patch.object(environment.runtime, "prepare_certificate_cycle", return_value=invalid_preparation),
            patch.object(provider, "commit_execution", wraps=provider.commit_execution) as commit_execution,
        ):
            next_observation, reward, terminated, truncated, info = environment.step(np.zeros(3))

        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["failure_reason"], "injected_final_recovery_coverage_loss")
        self.assertEqual(info["command_source"], "uncertified_emergency_brake")
        self.assertFalse(info["covered_at_publication"])
        self.assertEqual(info["execution_authority"], ExecutionAuthority.FAIL_CLOSED.value)
        self.assertFalse(info["backup_triggered"])
        self.assertEqual(environment.metrics.kappa_backup_steps, before_kappa_steps)
        self.assertEqual(environment.metrics.fail_closed_steps, before_fail_closed_steps + 1)
        self.assertEqual(provider.active_cell_id, active_cell_before)
        commit_execution.assert_not_called()

        item = transition_from_cycle(
            self.observation,
            next_observation,
            np.zeros(3),
            reward,
            terminated,
            truncated,
            0,
            context,
            None,
            info,
        )
        metrics = self._roundtrip_update(item)
        self.assertEqual(metrics["fail_closed_target_count"], 1)
        self.assertEqual(metrics["kappa_target_count"], 0)

    def test_midcycle_certificate_version_change_is_fail_closed_without_child_commit(self) -> None:
        environment = self.environment
        runtime = environment.runtime
        context = environment._refresh_context()
        self.assertEqual(context["execution_authority"], ExecutionAuthority.RL_GENERATOR.value)
        provider = runtime.mission_provider
        self.assertIsNotNone(provider)
        active_cell_before = provider.active_cell_id
        before_kappa_steps = environment.metrics.kappa_backup_steps
        before_fail_closed_steps = environment.metrics.fail_closed_steps
        original_sample = runtime.actor.sample_u
        original_geometry_version = runtime.geometry.version

        def mutate_certificate_version(observation):
            runtime.geometry.version += 1
            return original_sample(observation)

        try:
            with (
                patch.object(runtime.actor, "sample_u", side_effect=mutate_certificate_version),
                patch.object(provider, "commit_execution", wraps=provider.commit_execution) as commit_execution,
            ):
                _, _, terminated, truncated, info = environment.step(np.zeros(3))
        finally:
            runtime.geometry.version = original_geometry_version

        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["fallback_reason"], "CERTIFICATE_VERSION_CHANGED")
        self.assertEqual(info["command_source"], "uncertified_emergency_brake")
        self.assertFalse(info["covered_at_publication"])
        self.assertEqual(info["execution_authority"], ExecutionAuthority.FAIL_CLOSED.value)
        self.assertEqual(environment.metrics.kappa_backup_steps, before_kappa_steps)
        self.assertEqual(environment.metrics.fail_closed_steps, before_fail_closed_steps + 1)
        self.assertEqual(provider.active_cell_id, active_cell_before)
        commit_execution.assert_not_called()

    def test_all_copied_bound_version_changes_are_seen_by_live_watchdog(self) -> None:
        environment = self.environment
        for version_owner, version_field in (
            ("sensor", "version"),
            ("dynamics", "version"),
            ("tracking", "version"),
            ("energy", "version"),
            ("terminal", "version"),
            ("kappa", "parameter_version"),
        ):
            with self.subTest(version_owner=version_owner):
                self.observation, self.reset_info = environment.reset(seed=0)
                environment._context_cache_key = None
                context = environment._refresh_context()
                self.assertEqual(context["execution_authority"], ExecutionAuthority.RL_GENERATOR.value)
                runtime = environment.runtime
                provider = runtime.mission_provider
                self.assertIsNotNone(provider)
                active_cell_before = provider.active_cell_id
                before_kappa_steps = environment.metrics.kappa_backup_steps
                before_fail_closed_steps = environment.metrics.fail_closed_steps
                original_sample = runtime.actor.sample_u
                owner = (
                    runtime.recovery_policy.config
                    if version_owner == "kappa"
                    else getattr(runtime.calibration, version_owner)
                )
                original_version = getattr(owner, version_field)

                def mutate_bound_version(observation):
                    object.__setattr__(owner, version_field, f"{original_version}-MIDCYCLE")
                    return original_sample(observation)

                try:
                    with (
                        patch.object(runtime.actor, "sample_u", side_effect=mutate_bound_version),
                        patch.object(provider, "commit_execution", wraps=provider.commit_execution) as commit_execution,
                    ):
                        _, _, terminated, truncated, info = environment.step(np.zeros(3))
                finally:
                    object.__setattr__(owner, version_field, original_version)

                self.assertTrue(terminated)
                self.assertFalse(truncated)
                self.assertEqual(info["fallback_reason"], "CERTIFICATE_VERSION_CHANGED")
                self.assertEqual(info["command_source"], "uncertified_emergency_brake")
                self.assertFalse(info["covered_at_publication"])
                self.assertEqual(info["execution_authority"], ExecutionAuthority.FAIL_CLOSED.value)
                self.assertEqual(environment.metrics.kappa_backup_steps, before_kappa_steps)
                self.assertEqual(environment.metrics.fail_closed_steps, before_fail_closed_steps + 1)
                self.assertEqual(provider.active_cell_id, active_cell_before)
                commit_execution.assert_not_called()

    def test_early_task_closure_failure_rechecks_snapshot_before_kappa_publication(self) -> None:
        environment = self.environment
        runtime = environment.runtime
        environment._context_cache_key = None
        context = environment._refresh_context()
        self.assertGreater(context["recovery_level"], 0)
        valid_preparation = runtime.last_preparation
        self.assertIsNotNone(valid_preparation)
        stale_failure = RuntimeCyclePreparation(
            valid_preparation.state,
            valid_preparation.task_observation,
            valid_preparation.closure_result,
            valid_preparation.recovery,
            "INJECTED_TASK_CLOSURE_FAILURE",
        )
        provider = runtime.mission_provider
        self.assertIsNotNone(provider)
        active_cell_before = provider.active_cell_id
        before_kappa = environment.metrics.kappa_backup_steps
        original_geometry_version = runtime.geometry.version

        def mutate_after_preparation():
            runtime.geometry.version += 1
            return stale_failure

        try:
            with (
                patch.object(runtime, "prepare_certificate_cycle", side_effect=mutate_after_preparation),
                patch.object(provider, "commit_execution", wraps=provider.commit_execution) as commit_execution,
            ):
                _, _, terminated, truncated, info = environment.step(np.zeros(3))
        finally:
            runtime.geometry.version = original_geometry_version

        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["fallback_reason"], "CERTIFICATE_VERSION_CHANGED")
        self.assertEqual(info["command_source"], "uncertified_emergency_brake")
        self.assertFalse(info["covered_at_publication"])
        self.assertEqual(info["execution_authority"], ExecutionAuthority.FAIL_CLOSED.value)
        self.assertEqual(environment.metrics.kappa_backup_steps, before_kappa)
        self.assertEqual(provider.active_cell_id, active_cell_before)
        commit_execution.assert_not_called()

    def test_rank_zero_runtime_candidate_failure_cannot_fallback_to_kappa(self) -> None:
        environment = self.environment
        runtime = environment.runtime
        environment.plant.state.position = environment.plant.scenario.station_position.copy()
        environment.plant.state.velocity = np.zeros(3, dtype=np.float64)
        environment.plant.state.energy = environment.charging.config.battery_capacity
        environment.task_env.mode = PersistentMissionMode.TASK_RL
        environment.task_env.phase = environment.task_env.mode
        environment._context_cache_key = None
        context = environment._refresh_context()
        self.assertEqual(context["recovery_level"], 0)
        self.assertEqual(context["execution_authority"], ExecutionAuthority.RL_GENERATOR.value)
        provider = runtime.mission_provider
        self.assertIsNotNone(provider)
        active_cell_before = provider.active_cell_id
        before_kappa = environment.metrics.kappa_backup_steps
        with (
            patch.object(
                runtime.runtime_certifier,
                "prepare_candidate_from_certificate",
                side_effect=RuntimeError("injected candidate failure"),
            ),
            patch.object(provider, "commit_execution", wraps=provider.commit_execution) as commit_execution,
        ):
            _, _, terminated, truncated, info = environment.step(np.zeros(3))

        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["fallback_reason"], "KAPPA_FALLBACK_DISALLOWED")
        self.assertEqual(info["command_source"], "uncertified_emergency_brake")
        self.assertFalse(info["covered_at_publication"])
        self.assertEqual(info["execution_authority"], ExecutionAuthority.FAIL_CLOSED.value)
        self.assertEqual(environment.metrics.kappa_backup_steps, before_kappa)
        self.assertEqual(provider.active_cell_id, active_cell_before)
        commit_execution.assert_not_called()

    def test_terminal_mission_phase_cannot_publish_kappa(self) -> None:
        runtime = self.environment.runtime
        state = runtime._certificate_state()
        provider = runtime.mission_provider
        self.assertIsNotNone(provider)
        context = provider.evaluate(state, runtime.plant.state.timestamp)
        self.assertGreater(context.recovery_level, 0)
        self.assertTrue(runtime._kappa_publication_eligible(state, context))

        for phase in ("FAILURE", "SUCCESS", "CHARGING", "CHARGING_RL"):
            with self.subTest(phase=phase):
                state.explicit_task_state["mission_phase"] = phase
                self.assertFalse(runtime._kappa_publication_eligible(state, context))

    def test_prestep_recovery_dependency_version_or_hash_change_invalidates_kappa(self) -> None:
        environment = self.environment
        runtime = environment.runtime
        for field, replacement_suffix in (("version", "-PRESTEP"), ("contract_hash", "-REPLACED")):
            with self.subTest(field=field):
                self.observation, self.reset_info = environment.reset(seed=0)
                provider = runtime.mission_provider
                self.assertIsNotNone(provider)
                active_cell_before = provider.active_cell_id
                before_kappa = environment.metrics.kappa_backup_steps
                owner = runtime.calibration.energy
                original = getattr(owner, field)
                try:
                    object.__setattr__(owner, field, f"{original}{replacement_suffix}")
                    environment._context_cache_key = None
                    context = environment._refresh_context()
                    self.assertFalse(context["certificate_valid"])
                    self.assertEqual(context["execution_authority"], ExecutionAuthority.FAIL_CLOSED.value)
                    with patch.object(
                        provider,
                        "commit_execution",
                        wraps=provider.commit_execution,
                    ) as commit_execution:
                        _, _, terminated, truncated, info = environment.step(np.zeros(3))
                finally:
                    object.__setattr__(owner, field, original)
                    environment._context_cache_key = None

                self.assertTrue(terminated)
                self.assertFalse(truncated)
                self.assertEqual(info["command_source"], "uncertified_emergency_brake")
                self.assertFalse(info["covered_at_publication"])
                self.assertEqual(info["execution_authority"], ExecutionAuthority.FAIL_CLOSED.value)
                self.assertEqual(environment.metrics.kappa_backup_steps, before_kappa)
                self.assertEqual(provider.active_cell_id, active_cell_before)
                commit_execution.assert_not_called()

    def test_midcycle_timestamp_expiry_is_seen_by_publication_snapshot(self) -> None:
        environment = self.environment
        runtime = environment.runtime
        environment._context_cache_key = None
        context = environment._refresh_context()
        self.assertEqual(context["execution_authority"], ExecutionAuthority.RL_GENERATOR.value)
        provider = runtime.mission_provider
        self.assertIsNotNone(provider)
        active_cell_before = provider.active_cell_id
        before_kappa = environment.metrics.kappa_backup_steps
        original_sample = runtime.actor.sample_u
        original_timestamp = runtime.plant.state.timestamp
        expired_timestamp = max(cell.expiry for cell in provider.manifest.cells) + 1.0

        def expire_during_actor(observation):
            runtime.plant.state.timestamp = expired_timestamp
            return original_sample(observation)

        try:
            with (
                patch.object(runtime.actor, "sample_u", side_effect=expire_during_actor),
                patch.object(provider, "commit_execution", wraps=provider.commit_execution) as commit_execution,
            ):
                _, _, terminated, truncated, info = environment.step(np.zeros(3))
        finally:
            runtime.plant.state.timestamp = original_timestamp
            environment._context_cache_key = None

        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["fallback_reason"], "CERTIFICATE_VERSION_CHANGED")
        self.assertEqual(info["command_source"], "uncertified_emergency_brake")
        self.assertFalse(info["covered_at_publication"])
        self.assertEqual(info["execution_authority"], ExecutionAuthority.FAIL_CLOSED.value)
        self.assertEqual(environment.metrics.kappa_backup_steps, before_kappa)
        self.assertEqual(provider.active_cell_id, active_cell_before)
        commit_execution.assert_not_called()

    def test_kappa_reprepare_version_change_is_fail_closed_without_child_commit(self) -> None:
        environment = self.environment
        runtime = environment.runtime
        environment._begin_backup("TEST_KAPPA_REPREPARE_VERSION_RACE")
        environment._context_cache_key = None
        context = environment._refresh_context()
        self.assertEqual(context["execution_authority"], ExecutionAuthority.KAPPA_BACKUP.value)
        provider = runtime.mission_provider
        self.assertIsNotNone(provider)
        active_cell_before = provider.active_cell_id
        before_kappa_steps = environment.metrics.kappa_backup_steps
        before_fail_closed_steps = environment.metrics.fail_closed_steps
        original_prepare = runtime.prepare_certificate_cycle
        original_geometry_version = runtime.geometry.version
        mutated = False

        def mutate_before_reprepare():
            nonlocal mutated
            if not mutated:
                runtime.geometry.version += 1
                mutated = True
            return original_prepare()

        try:
            with (
                patch.object(runtime, "prepare_certificate_cycle", side_effect=mutate_before_reprepare),
                patch.object(provider, "commit_execution", wraps=provider.commit_execution) as commit_execution,
            ):
                _, _, terminated, truncated, info = environment.step(np.zeros(3))
        finally:
            runtime.geometry.version = original_geometry_version

        self.assertTrue(mutated)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["failure_reason"], "certificate_version_changed")
        self.assertEqual(info["fallback_reason"], "CERTIFICATE_VERSION_CHANGED")
        self.assertEqual(info["command_source"], "uncertified_emergency_brake")
        self.assertFalse(info["covered_at_publication"])
        self.assertEqual(info["execution_authority"], ExecutionAuthority.FAIL_CLOSED.value)
        self.assertEqual(environment.metrics.kappa_backup_steps, before_kappa_steps)
        self.assertEqual(environment.metrics.fail_closed_steps, before_fail_closed_steps + 1)
        self.assertEqual(provider.active_cell_id, active_cell_before)
        commit_execution.assert_not_called()

    def test_nominal_preview_losing_final_recovery_coverage_is_fail_closed_without_child_commit(self) -> None:
        environment = self.environment
        runtime = environment.runtime
        environment._context_cache_key = None
        environment._refresh_context()
        valid_preparation = runtime.last_preparation
        self.assertIsNotNone(valid_preparation)
        invalid_preparation = RuntimeCyclePreparation(
            valid_preparation.state,
            valid_preparation.task_observation,
            valid_preparation.closure_result,
            None,
            "INJECTED_NOMINAL_PUBLICATION_COVERAGE_LOSS",
        )
        provider = runtime.mission_provider
        self.assertIsNotNone(provider)
        active_cell_before = provider.active_cell_id
        with (
            patch.object(runtime, "prepare_certificate_cycle", return_value=invalid_preparation),
            patch.object(provider, "commit_execution", wraps=provider.commit_execution) as commit_execution,
        ):
            _, _, terminated, truncated, info = runtime.step_nominal_action(np.zeros(3))

        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["failure_reason"], "injected_nominal_publication_coverage_loss")
        self.assertEqual(info["fallback_reason"], "INJECTED_NOMINAL_PUBLICATION_COVERAGE_LOSS")
        self.assertEqual(info["command_source"], "uncertified_emergency_brake")
        self.assertFalse(info["covered_at_publication"])
        np.testing.assert_array_equal(
            info["telemetry"].action_trace.fallback,
            info["telemetry"].action_trace.published,
        )
        self.assertEqual(provider.active_cell_id, active_cell_before)
        commit_execution.assert_not_called()

    def test_nominal_reprepare_version_change_is_uncovered_without_child_commit(self) -> None:
        environment = self.environment
        runtime = environment.runtime
        provider = runtime.mission_provider
        self.assertIsNotNone(provider)
        active_cell_before = provider.active_cell_id
        original_prepare = runtime.prepare_certificate_cycle
        original_geometry_version = runtime.geometry.version
        mutated = False

        def mutate_before_reprepare():
            nonlocal mutated
            if not mutated:
                runtime.geometry.version += 1
                mutated = True
            return original_prepare()

        try:
            with (
                patch.object(runtime, "prepare_certificate_cycle", side_effect=mutate_before_reprepare),
                patch.object(provider, "commit_execution", wraps=provider.commit_execution) as commit_execution,
            ):
                _, _, terminated, truncated, info = runtime.step_nominal_action(np.zeros(3))
        finally:
            runtime.geometry.version = original_geometry_version

        self.assertTrue(mutated)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertFalse(info["accepted"])
        self.assertEqual(info["failure_reason"], "certificate_version_changed")
        self.assertEqual(info["fallback_reason"], "CERTIFICATE_VERSION_CHANGED")
        self.assertEqual(info["command_source"], "uncertified_emergency_brake")
        self.assertFalse(info["covered_at_publication"])
        self.assertEqual(provider.active_cell_id, active_cell_before)
        commit_execution.assert_not_called()

    def test_goal_exposure_reset_samples_new_goal(self) -> None:
        first_goal = np.asarray(self.reset_info["sampled_goal"], dtype=np.float64)
        reset_seed = goal_exposure_reset_seed(0, 1)
        _, reset_info = self.environment.reset(seed=reset_seed)
        self.assertFalse(np.allclose(first_goal, reset_info["sampled_goal"]))

    def test_goal_exposure_reset_samples_certified_initial_state(self) -> None:
        reset_seed = goal_exposure_reset_seed(2, 3)
        _, reset_info = self.environment.reset(seed=reset_seed)
        context = reset_info["action_context"]
        self.assertTrue(context["certificate_valid"])
        self.assertTrue(context["recoverable_set_member"])
        self.assertTrue(context["rl_authority_set_member"])

    def test_exposure_reset_preserves_agent(self) -> None:
        agent = GeneratorSAC(self.observation.size, GeneratorSACConfig(hidden_dim=16), seed=4)
        before = [parameter.detach().clone() for parameter in agent.actor.parameters()]
        self.environment.reset(seed=goal_exposure_reset_seed(4, 1))
        self.assertTrue(all(torch.equal(left, right) for left, right in zip(before, agent.actor.parameters())))

    def test_exposure_reset_preserves_replay(self) -> None:
        item, _, _ = self._collector_transition()
        agent = GeneratorSAC(self.observation.size, GeneratorSACConfig(hidden_dim=16), seed=5)
        agent.observe(item)
        self.environment.reset(seed=goal_exposure_reset_seed(5, 1))
        self.assertEqual(len(agent.replay), 1)
        self.assertTrue(agent.replay.transitions[0].collector_boundary)

    def test_collector_reset_state_not_used_as_previous_successor(self) -> None:
        item, physical_successor, _ = self._collector_transition()
        reset_observation, _ = self.environment.reset(seed=goal_exposure_reset_seed(0, 1))
        np.testing.assert_array_equal(item.next_observation, physical_successor)
        self.assertFalse(np.array_equal(item.next_observation, reset_observation))

    def test_collector_boundary_preserves_real_successor(self) -> None:
        item, physical_successor, next_context = self._collector_transition()
        np.testing.assert_array_equal(item.next_observation, physical_successor)
        self.assertEqual(item.next_execution_authority, next_context["execution_authority"])
        self.assertEqual(item.next_certificate_epoch, next_context["certificate_epoch"])
        if item.next_generator_executable:
            np.testing.assert_array_equal(item.next_c, next_context["c"])
            np.testing.assert_array_equal(item.next_G, next_context["G"])

    def test_collector_boundary_does_not_force_terminal_target(self) -> None:
        item, _, _ = self._collector_transition()
        agent = PersistentGeneratorSAC(
            self.observation.size,
            GeneratorSACConfig(batch_size=2, hidden_dim=16, bootstrap_on_truncation=True),
            seed=6,
        )
        without_boundary = replace(item, collector_boundary=False)
        torch.manual_seed(17)
        boundary_targets, boundary_counts = agent.bellman_target([item, item])
        torch.manual_seed(17)
        ordinary_targets, ordinary_counts = agent.bellman_target([without_boundary, without_boundary])
        torch.testing.assert_close(boundary_targets, ordinary_targets)
        self.assertEqual(boundary_counts["generator_target_count"], ordinary_counts["generator_target_count"])
        self.assertEqual(boundary_counts["collector_boundary_target_count"], 2)

    def test_true_termination_still_zero_bootstraps(self) -> None:
        item, _, _ = self._collector_transition()
        terminated = replace(item, terminated=True, collector_boundary=True)
        agent = PersistentGeneratorSAC(
            self.observation.size,
            GeneratorSACConfig(batch_size=2, hidden_dim=16, bootstrap_on_truncation=True),
            seed=16,
        )
        targets, _ = agent.bellman_target([terminated, terminated])
        torch.testing.assert_close(targets, torch.full((2,), float(item.reward)))

    def test_truncation_semantics_unchanged(self) -> None:
        item, _, _ = self._collector_transition()
        item = replace(
            item,
            collector_boundary=False,
            truncated=True,
            next_generator_available=False,
            next_certificate_valid=False,
            next_c=None,
            next_G=None,
        )
        agent = GeneratorSAC(
            self.observation.size,
            GeneratorSACConfig(batch_size=2, hidden_dim=16, bootstrap_on_truncation=True),
            seed=7,
        )
        with torch.no_grad():
            for network in (agent.target_critic_1, agent.target_critic_2):
                for parameter in network.parameters():
                    parameter.zero_()
                network.network[-1].bias.fill_(1.0)
        targets, _ = agent.bellman_target([item, item])
        expected = float(item.reward) + agent.config.gamma
        torch.testing.assert_close(targets, torch.full((2,), expected))

        no_bootstrap_agent = GeneratorSAC(
            self.observation.size,
            GeneratorSACConfig(batch_size=2, hidden_dim=16, bootstrap_on_truncation=False),
            seed=7,
        )
        terminal_targets, _ = no_bootstrap_agent.bellman_target([item, item])
        torch.testing.assert_close(terminal_targets, torch.full((2,), float(item.reward)))

    def test_truncation_records_real_successor_authority_before_reset(self) -> None:
        context = self.reset_info["action_context"]
        actor_u = np.zeros(3, dtype=np.float64)
        next_observation, reward, terminated, _, info = self.environment.step(actor_u)
        self.assertFalse(terminated)
        next_context = successor_context_for_transition(self.environment, terminated=terminated)
        self.assertIsNotNone(next_context)
        item = transition_from_cycle(
            self.observation,
            next_observation,
            actor_u,
            reward,
            terminated,
            True,
            0,
            context,
            next_context,
            info,
        )
        self.assertTrue(item.truncated)
        self.assertEqual(item.next_execution_authority, next_context["execution_authority"])
        self.assertNotEqual(item.next_execution_authority, ExecutionAuthority.FAIL_CLOSED.value)
        self.assertEqual(item.next_certificate_epoch, next_context["certificate_epoch"])
        rebuilt = self.environment.task_env.build_observation(
            self.environment.runtime._map_encoding(),
            self.environment.runtime._corridor_encoding(),
        )
        np.testing.assert_array_equal(next_observation, rebuilt)

    def test_true_termination_has_no_successor_context(self) -> None:
        environment = SimpleNamespace(_refresh_context=lambda: self.fail("terminal context refreshed"))
        self.assertIsNone(successor_context_for_transition(environment, terminated=True))

    def test_exposure_disabled_is_backward_compatible(self) -> None:
        self.assertEqual(training_protocol_name(None), "persistent_only")
        self.assertFalse(goal_exposure_reset_boundary(250, 5000, None, terminated=False, truncated=False))
        self.assertFalse(goal_exposure_reset_boundary(250, 5000, 0, terminated=False, truncated=False))

    def test_persistent_evaluation_never_uses_exposure_reset(self) -> None:
        goal_before = self.environment.task_env.manager.current_task.goal_position.copy()
        self.environment.task_env.manager.goal_radius = 1e-6
        self.environment.task_env.step(np.zeros(3, dtype=np.float64))
        np.testing.assert_array_equal(self.environment.task_env.manager.current_task.goal_position, goal_before)

    def test_collector_reset_is_not_natural_completion(self) -> None:
        tracker = GoalExposureAccumulator()
        tracker.assign(np.ones(3), np.zeros(3), 0, "initial_reset", 0)
        tracker.observe_step(False)
        tracker.assign(np.full(3, 2.0), np.zeros(3), 250, "collector_reset", 1_000_003)
        summary = tracker.summary()
        self.assertEqual(summary["collector_resets"], 1)
        self.assertEqual(summary["natural_task_completions"], 0)

    def test_batch_goal_diversity_uses_explicit_goal_metadata(self) -> None:
        item, _, _ = self._collector_transition()
        goal = np.array((1.2345678, 2.3456789, 1.0), dtype=np.float32)
        first = replace(item, task_goal=goal, observation=item.observation.copy())
        shifted = item.observation.copy()
        shifted[self.environment.task_env.observation_layout["position"]] += 1e-4
        second = replace(item, task_goal=goal, observation=shifted)
        metrics = batch_goal_diversity(
            [first, second],
            self.environment.task_env.observation_layout["position"],
            self.environment.task_env.observation_layout["goal_delta"],
            self.environment.plant.config.world_size,
        )
        self.assertEqual(metrics["batch_unique_goal_count"], 1)


if __name__ == "__main__":
    unittest.main()
