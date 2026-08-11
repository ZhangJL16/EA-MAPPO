from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch import nn

from cert_runtime.generator_sac import GeneratorSACConfig, GeneratorTransition, PersistentGeneratorSAC
from cert_runtime.persistent_authority import (
    ExecutionAuthority,
    PersistentAuthorityInput,
    PersistentExecutionAuthority,
)
from cert_runtime.types import Interval3, Zonotope3
from envs.certified_uav.recoverability import RecoverabilityVerifier
from scripts.persistent_generator_common import transition_from_cycle


def authority_input(**updates) -> PersistentAuthorityInput:
    values = {
        "persistent_mode": "TASK_RL",
        "energy_margin": 3.0,
        "backup_switch_margin": 1.0,
        "persistent_certificate_valid": True,
        "certificate_valid": True,
        "kappa_valid": True,
        "generator_available": True,
        "recoverable_set_member": True,
        "recoverability_action_verified": True,
        "policy_authority_pass": True,
        "charging_state": False,
        "departure_allowed": True,
        "charging_support_verified": False,
        "station_hold_valid": False,
        "recovery_level": 1,
    }
    values.update(updates)
    return PersistentAuthorityInput(**values)


def persistent_transition(
    authority: ExecutionAuthority,
    *,
    generator_executable: bool,
    backup_required: bool = False,
    recoverable: bool = True,
    action_verified: bool = True,
    policy_authority: bool = True,
) -> GeneratorTransition:
    observation = np.zeros(6, dtype=np.float32)
    center = np.zeros(3, dtype=np.float32)
    generators = np.diag([0.05, 0.06, 0.07]).astype(np.float32)
    kappa = np.array([0.03, -0.02, 0.01], dtype=np.float32)
    return GeneratorTransition(
        observation=observation,
        next_observation=observation + 0.1,
        reward=1.0,
        terminated=False,
        truncated=False,
        episode_id=0,
        mission_phase="TASK_RL",
        next_mission_phase="TASK_RL",
        certificate_epoch="manifest",
        next_certificate_epoch="manifest",
        u=np.zeros(3, dtype=np.float32),
        eta=np.zeros(3, dtype=np.float32),
        c=center,
        G=generators,
        candidate_action=center,
        kappa_action=kappa,
        executed_action=center,
        measured_action=center,
        accepted=True,
        fallback_reason=None,
        next_c=center if generator_executable else None,
        next_G=generators if generator_executable else None,
        next_kappa=kappa,
        next_generator_available=generator_executable,
        next_certificate_valid=True,
        geometry_version="geometry-v1",
        corridor_version="corridor-v1",
        energy_version="energy-v1",
        certificate_hashes=("recovery", "zonotope"),
        certificate_manifest_hash="manifest",
        execution_authority=ExecutionAuthority.RL_GENERATOR.value,
        next_execution_authority=authority.value,
        next_generator_executable=generator_executable,
        next_backup_required=backup_required,
        next_backup_reason="test" if backup_required else None,
        next_recoverable_set_member=recoverable,
        next_recoverability_action_verified=action_verified,
        next_policy_authority_pass=policy_authority,
        next_energy_margin=3.0,
        next_departure_allowed=True,
        next_charging_state=authority == ExecutionAuthority.CHARGER_CONSTRAINED,
        next_charging_restriction=authority == ExecutionAuthority.CHARGER_CONSTRAINED,
        next_charging_support_verified=(
            authority == ExecutionAuthority.CHARGER_CONSTRAINED and generator_executable
        ),
        next_station_hold_valid=(
            authority == ExecutionAuthority.CHARGER_CONSTRAINED and not generator_executable
        ),
        next_authority_action=kappa,
    )


class _CaptureQ(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.actions = None

    def forward(self, observation, action):
        del observation
        self.actions = action.detach().clone()
        return action.sum(dim=-1)


class PersistentBellmanAuthorityTests(unittest.TestCase):
    def setUp(self):
        config = GeneratorSACConfig(batch_size=2, hidden_dim=8, warmup_steps=0)
        self.agent = PersistentGeneratorSAC(6, config, seed=0)

    def test_next_backup_state_uses_kappa_bellman_branch(self):
        transition = persistent_transition(
            ExecutionAuthority.KAPPA_BACKUP,
            generator_executable=False,
            backup_required=True,
        )
        first, second = _CaptureQ(), _CaptureQ()
        self.agent.target_critic_1, self.agent.target_critic_2 = first, second
        before = self.agent.generator_log_density_calls
        _, counts = self.agent.bellman_target([transition])
        np.testing.assert_allclose(first.actions.cpu().numpy()[0], transition.next_kappa)
        self.assertEqual(self.agent.generator_log_density_calls, before)
        self.assertEqual(counts["kappa_target_count"], 1)

    def test_current_kappa_atom_must_match_certified_executed_action(self):
        transition = persistent_transition(
            ExecutionAuthority.KAPPA_BACKUP,
            generator_executable=False,
            backup_required=True,
        )
        malformed = replace(
            transition,
            execution_authority=ExecutionAuthority.KAPPA_BACKUP.value,
            accepted=False,
            u=None,
            eta=None,
            c=None,
            G=None,
            candidate_action=None,
            executed_action=np.zeros(3, dtype=np.float32),
        )
        with self.assertRaisesRegex(ValueError, "must equal kappa_action"):
            self.agent.observe(malformed)

    def test_next_kappa_atom_requires_valid_recovery_certificate(self):
        transition = persistent_transition(
            ExecutionAuthority.KAPPA_BACKUP,
            generator_executable=False,
            backup_required=True,
        )
        with self.assertRaisesRegex(ValueError, "valid recovery certificate"):
            self.agent.observe(replace(transition, next_certificate_valid=False))

    def test_next_interior_state_uses_generator_branch(self):
        transition = persistent_transition(ExecutionAuthority.RL_GENERATOR, generator_executable=True)
        before = self.agent.generator_log_density_calls
        _, counts = self.agent.bellman_target([transition])
        self.assertEqual(counts["generator_target_count"], 1)
        self.assertEqual(self.agent.generator_log_density_calls - before, 1)

    def test_transition_rejects_nonfinite_or_singular_generator_contracts(self):
        transition = persistent_transition(ExecutionAuthority.RL_GENERATOR, generator_executable=True)
        with self.assertRaisesRegex(ValueError, "G must be invertible"):
            replace(transition, G=np.zeros((3, 3), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "next_G must be invertible"):
            replace(transition, next_G=np.zeros((3, 3), dtype=np.float32))
        invalid_action = transition.executed_action.copy()
        invalid_action[1] = np.nan
        with self.assertRaisesRegex(ValueError, "executed_action must be a finite vector"):
            replace(transition, executed_action=invalid_action)

    def test_next_policy_authority_failure_uses_kappa_branch(self):
        decision = PersistentExecutionAuthority.evaluate(authority_input(policy_authority_pass=False))
        self.assertEqual(decision.authority, ExecutionAuthority.KAPPA_BACKUP)
        self.assertEqual(decision.reason, "POLICY_AUTHORITY_GATE_FAILED")

    def test_next_a_rec_failure_uses_kappa_branch(self):
        decision = PersistentExecutionAuthority.evaluate(authority_input(recoverability_action_verified=False))
        self.assertEqual(decision.authority, ExecutionAuthority.KAPPA_BACKUP)
        self.assertEqual(decision.reason, "GENERATOR_NOT_CONTAINED_IN_A_REC")

    def test_next_recoverable_set_failure_uses_kappa_branch(self):
        decision = PersistentExecutionAuthority.evaluate(authority_input(recoverable_set_member=False))
        self.assertEqual(decision.authority, ExecutionAuthority.KAPPA_BACKUP)
        self.assertEqual(decision.reason, "RECOVERABLE_SET_CERTIFICATE_INVALID")

    def test_closed_charger_generator_target_uses_constrained_support(self):
        transition = persistent_transition(
            ExecutionAuthority.CHARGER_CONSTRAINED,
            generator_executable=True,
        )
        before = self.agent.generator_log_density_calls
        _, counts = self.agent.bellman_target([transition])
        self.assertEqual(counts["generator_target_count"], 1)
        self.assertEqual(counts["charger_atomic_target_count"], 0)
        self.assertEqual(self.agent.generator_log_density_calls - before, 1)

    def test_closed_charger_hold_uses_atomic_target_without_density(self):
        transition = persistent_transition(
            ExecutionAuthority.CHARGER_CONSTRAINED,
            generator_executable=False,
        )
        first, second = _CaptureQ(), _CaptureQ()
        self.agent.target_critic_1, self.agent.target_critic_2 = first, second
        before = self.agent.generator_log_density_calls
        _, counts = self.agent.bellman_target([transition])
        np.testing.assert_allclose(first.actions.cpu().numpy()[0], transition.next_authority_action)
        self.assertEqual(self.agent.generator_log_density_calls, before)
        self.assertEqual(counts["charger_atomic_target_count"], 1)

    def test_fail_closed_has_no_bootstrap_or_generator_density(self):
        transition = persistent_transition(
            ExecutionAuthority.FAIL_CLOSED,
            generator_executable=False,
        )
        before = self.agent.generator_log_density_calls
        target, counts = self.agent.bellman_target([transition])
        self.assertAlmostEqual(float(target[0]), transition.reward)
        self.assertEqual(self.agent.generator_log_density_calls, before)
        self.assertEqual(counts["fail_closed_target_count"], 1)

    def test_charger_atomic_target_rejects_missing_hold_validity(self):
        transition = persistent_transition(
            ExecutionAuthority.CHARGER_CONSTRAINED,
            generator_executable=False,
        )
        transition = replace(transition, next_station_hold_valid=False)
        with self.assertRaisesRegex(ValueError, "valid certified hold"):
            self.agent.bellman_target([transition])

    def test_actor_target_matches_runtime_execution_authority(self):
        decision = PersistentExecutionAuthority.evaluate(authority_input(energy_margin=1.0))
        transition = persistent_transition(
            decision.authority,
            generator_executable=decision.generator_executable,
            backup_required=decision.kappa_required,
        )
        _, counts = self.agent.bellman_target([transition])
        self.assertEqual(decision.authority, ExecutionAuthority.KAPPA_BACKUP)
        self.assertEqual(counts["generator_target_count"], 0)
        self.assertEqual(counts["kappa_target_count"], 1)

    def test_open_charger_without_departure_support_fails_closed(self):
        decision = PersistentExecutionAuthority.evaluate(authority_input(
            persistent_mode="CHARGING_RL",
            charging_state=True,
            departure_allowed=True,
            generator_available=False,
            recoverability_action_verified=False,
            policy_authority_pass=False,
            station_hold_valid=True,
        ))
        self.assertEqual(decision.authority, ExecutionAuthority.FAIL_CLOSED)
        self.assertEqual(decision.reason, "NO_DEPARTURE_GENERATOR_SET")
        self.assertFalse(decision.kappa_required)
        self.assertFalse(decision.station_hold_required)

    def test_open_charger_state_failure_never_routes_to_kappa(self):
        decision = PersistentExecutionAuthority.evaluate(authority_input(
            persistent_mode="CHARGING_RL",
            charging_state=True,
            departure_allowed=True,
            recoverable_set_member=False,
        ))
        self.assertEqual(decision.authority, ExecutionAuthority.FAIL_CLOSED)
        self.assertEqual(decision.reason, "RECOVERABLE_SET_CERTIFICATE_INVALID")
        self.assertFalse(decision.kappa_required)

    def test_closed_charger_state_failure_never_routes_to_kappa(self):
        decision = PersistentExecutionAuthority.evaluate(authority_input(
            persistent_mode="CHARGING_RL",
            charging_state=True,
            departure_allowed=False,
            persistent_certificate_valid=False,
            station_hold_valid=True,
        ))
        self.assertEqual(decision.authority, ExecutionAuthority.FAIL_CLOSED)
        self.assertEqual(decision.reason, "PERSISTENT_CERTIFICATE_GATE_FAILED")
        self.assertFalse(decision.kappa_required)

    def test_normal_zero_rank_refusal_has_no_fictitious_kappa_action(self):
        decision = PersistentExecutionAuthority.evaluate(authority_input(
            recovery_level=0,
            generator_available=False,
        ))
        self.assertEqual(decision.authority, ExecutionAuthority.FAIL_CLOSED)
        self.assertEqual(decision.reason, "ZERO_RANK_RECOVERY_ACTION_UNDEFINED")
        self.assertFalse(decision.kappa_required)


class ChargingSupportTests(unittest.TestCase):
    def _fixture(self):
        class Envelope:
            dynamics_bound_version = "dynamics-v1"
            energy_bound_version = "energy-v1"
            energy_low = 20.0
            energy_prefix_low = 20.0
            velocity = Interval3((-0.01, -0.01, -0.01), (0.01, 0.01, 0.01))

            def __init__(self, action_set):
                bounds = action_set.action_bounds
                self.position = Interval3(
                    tuple(0.5 + np.asarray(bounds.low)),
                    tuple(0.5 + np.asarray(bounds.high)),
                )
                self.swept_position = self.position

        class Builder:
            dynamics = SimpleNamespace(version="dynamics-v1")
            energy = SimpleNamespace(version="energy-v1")

            @staticmethod
            def propagate_zonotope(state, action_set):
                del state
                return Envelope(action_set)

        target = SimpleNamespace(
            cell_id="target",
            energy_upper=5.0,
            hash_valid=True,
            complete_successor_containment=True,
            minimum_geometry_slack=0.1,
            state_bounds=SimpleNamespace(velocity=Interval3((-0.1,) * 3, (0.1,) * 3)),
            dynamics_version="dynamics-v1",
            tracking_version="tracking-v1",
            energy_version="energy-v1",
            terminal_version="terminal-v1",
            kappa_version="kappa-v1",
            recovery_certificate_hash="recovery",
        )
        provider = SimpleNamespace(
            energy_reserve=1.0,
            _target_root=lambda index: target,
            _candidate_successor_in_root=lambda state, action_set, selected: True,
            complete_swept_tube_slack=lambda swept_position: 0.1,
            _versions=lambda: ("geometry-v1", "dynamics-v1", "tracking-v1", "energy-v1", "terminal-v1", "kappa-v1"),
        )
        terminal = SimpleNamespace(
            minimum_energy=1.0,
            position_low=np.full(3, 0.45),
            position_high=np.full(3, 0.55),
            velocity_abs_max=np.full(3, 0.05),
        )
        runtime = SimpleNamespace(
            config=SimpleNamespace(a_max=np.ones(3), minimum_generator_sigma=0.05, generator_bisection_iterations=8),
            calibration=SimpleNamespace(
                dynamics=SimpleNamespace(version="dynamics-v1"),
                tracking=SimpleNamespace(version="tracking-v1"),
                energy=SimpleNamespace(version="energy-v1"),
                terminal=SimpleNamespace(version="terminal-v1"),
            ),
            envelope_builder=Builder(),
            scenario=SimpleNamespace(terminal=terminal),
        )
        state = SimpleNamespace(
            position=(0.5, 0.5, 0.5),
            bound_versions={"dynamics": "dynamics-v1", "tracking": "tracking-v1", "energy": "energy-v1", "terminal": "terminal-v1"},
            snapshot=lambda: {"state": "charging"},
        )
        context = SimpleNamespace(
            recovery=SimpleNamespace(certified=True, certificate_hash="recovery"),
            required_energy=7.0,
            current_energy_margin=2.0,
            recovery_cell_id="current",
            root_index=1,
            task_successor_cell_id=None,
        )
        return RecoverabilityVerifier(runtime, provider), state, context

    def test_closed_departure_gate_generator_stays_in_charge_set(self):
        verifier, state, context = self._fixture()
        full = Zonotope3.diagonal(np.zeros(3), (0.1, 0.1, 0.1))
        restricted, certificate = verifier.restrict_to_charging_set(state, full, context)
        self.assertIsNotNone(restricted)
        self.assertTrue(certificate.verified)
        self.assertTrue(verifier.action_set_stays_in_charging_set(state, restricted))
        self.assertGreaterEqual(restricted.sigma_min_lower_bound, 0.05 - 1e-12)

    def test_closed_departure_gate_has_no_normal_action_aliasing(self):
        decision = PersistentExecutionAuthority.evaluate(authority_input(
            persistent_mode="CHARGING_RL",
            charging_state=True,
            departure_allowed=False,
            charging_support_verified=True,
        ))
        self.assertEqual(decision.authority, ExecutionAuthority.CHARGER_CONSTRAINED)
        self.assertTrue(decision.generator_executable)
        self.assertFalse(decision.station_hold_required)

    def test_open_departure_gate_allows_recoverable_departure(self):
        decision = PersistentExecutionAuthority.evaluate(authority_input(
            persistent_mode="CHARGING_RL",
            charging_state=True,
            departure_allowed=True,
            energy_margin=0.1,
        ))
        self.assertEqual(decision.authority, ExecutionAuthority.RL_GENERATOR)
        self.assertTrue(decision.generator_executable)
        self.assertTrue(decision.departure_allowed)


class PersistentReplayPhaseTests(unittest.TestCase):
    def test_mission_phase_and_next_phase_follow_before_and_after_contexts(self):
        zero = np.zeros(3, dtype=np.float32)
        trace = SimpleNamespace(candidate=None, fallback=zero, published=zero, measured=zero)
        telemetry = SimpleNamespace(action_trace=trace, state_before=SimpleNamespace(energy=10.0))
        context = {
            "certificate_epoch": "manifest",
            "persistent_mode": "TASK_RL",
            "geometry_version": "g",
            "corridor_version": "c",
            "energy_version": "e",
        }
        next_context = {
            "certificate_epoch": "manifest",
            "persistent_mode": "BACKUP_RECOVERY",
            "execution_authority": ExecutionAuthority.KAPPA_BACKUP.value,
            "execution_authority_reason": "ENERGY_MARGIN_BACKUP_SWITCH",
            "backup_required": True,
            "generator_executable": False,
            "certificate_valid": True,
            "kappa": zero,
            "energy_margin": 0.5,
        }
        item = transition_from_cycle(
            zero,
            zero,
            zero,
            0.0,
            False,
            False,
            0,
            context,
            next_context,
            {"telemetry": telemetry, "persistent_mode": "BACKUP_RECOVERY"},
        )
        self.assertEqual(item.mission_phase, "TASK_RL")
        self.assertEqual(item.next_mission_phase, "BACKUP_RECOVERY")


if __name__ == "__main__":
    unittest.main()
