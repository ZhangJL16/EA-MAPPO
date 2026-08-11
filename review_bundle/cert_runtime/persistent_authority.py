from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite


class ExecutionAuthority(str, Enum):
    RL_GENERATOR = "RL_GENERATOR"
    KAPPA_BACKUP = "KAPPA_BACKUP"
    CHARGER_CONSTRAINED = "CHARGER_CONSTRAINED"
    FAIL_CLOSED = "FAIL_CLOSED"


@dataclass(frozen=True, slots=True)
class PersistentAuthorityInput:
    persistent_mode: str
    energy_margin: float
    backup_switch_margin: float
    persistent_certificate_valid: bool
    certificate_valid: bool
    kappa_valid: bool
    generator_available: bool
    recoverable_set_member: bool
    recoverability_action_verified: bool
    policy_authority_pass: bool
    charging_state: bool
    departure_allowed: bool
    charging_support_verified: bool
    station_hold_valid: bool
    rl_authority_member: bool = True
    continuation_action_verified: bool = True
    recovery_level: int | None = None


@dataclass(frozen=True, slots=True)
class PersistentAuthorityDecision:
    authority: ExecutionAuthority
    reason: str
    generator_executable: bool
    kappa_required: bool
    departure_allowed: bool
    charging_restriction: bool
    station_hold_required: bool


class PersistentExecutionAuthority:
    """Pure execution classification shared by runtime and persistent SAC."""

    @staticmethod
    def evaluate(inputs: PersistentAuthorityInput) -> PersistentAuthorityDecision:
        positive_recovery_rank = bool(
            inputs.recovery_level is not None and int(inputs.recovery_level) > 0
        )

        def recovery_or_fail(reason: str) -> PersistentAuthorityDecision:
            if positive_recovery_rank:
                return PersistentAuthorityDecision(
                    ExecutionAuthority.KAPPA_BACKUP,
                    reason,
                    False,
                    True,
                    False,
                    False,
                    False,
                )
            return PersistentAuthorityDecision(
                ExecutionAuthority.FAIL_CLOSED,
                "ZERO_RANK_RECOVERY_ACTION_UNDEFINED",
                False,
                False,
                False,
                False,
                False,
            )

        if not inputs.kappa_valid:
            return PersistentAuthorityDecision(
                ExecutionAuthority.FAIL_CLOSED,
                "KAPPA_CERTIFICATE_INVALID",
                False,
                False,
                False,
                inputs.charging_state and not inputs.departure_allowed,
                False,
            )
        if inputs.persistent_mode == "BACKUP_RECOVERY":
            return recovery_or_fail("BACKUP_RECOVERY_CONTINUATION")
        state_checks = (
            (inputs.persistent_certificate_valid, "PERSISTENT_CERTIFICATE_GATE_FAILED"),
            (inputs.certificate_valid, "RECOVERY_CERTIFICATE_INVALID"),
            (inputs.recoverable_set_member, "RECOVERABLE_SET_CERTIFICATE_INVALID"),
            (isfinite(inputs.energy_margin), "ENERGY_MARGIN_NONFINITE"),
        )
        failed_state_reason = next((reason for valid, reason in state_checks if not valid), None)

        # A charging-source state is never sent to kappa.  The covered hybrid
        # relation has exactly two positive charging exits: closed-gate
        # CHARGE/HOLD and open-gate DEPART.  If the certificate needed by the
        # applicable exit is absent, termination is fail-closed instead of an
        # unmodelled CHARGING -> KAPPA (and possibly charging) transition.
        if inputs.charging_state and not inputs.departure_allowed:
            if failed_state_reason is not None:
                return PersistentAuthorityDecision(
                    ExecutionAuthority.FAIL_CLOSED,
                    failed_state_reason,
                    False,
                    False,
                    False,
                    True,
                    False,
                )
            if (
                inputs.generator_available
                and inputs.recoverability_action_verified
                and inputs.policy_authority_pass
                and inputs.charging_support_verified
            ):
                return PersistentAuthorityDecision(
                    ExecutionAuthority.CHARGER_CONSTRAINED,
                    "VERIFIED_CHARGING_SUPPORT",
                    True,
                    False,
                    False,
                    True,
                    False,
                )
            if inputs.station_hold_valid:
                return PersistentAuthorityDecision(
                    ExecutionAuthority.CHARGER_CONSTRAINED,
                    "CHARGING_SUPPORT_UNAVAILABLE_USE_HOLD",
                    False,
                    False,
                    False,
                    True,
                    True,
                )
            return PersistentAuthorityDecision(
                ExecutionAuthority.FAIL_CLOSED,
                "CHARGING_SUPPORT_AND_HOLD_INVALID",
                False,
                False,
                False,
                True,
                False,
            )

        if inputs.charging_state and inputs.departure_allowed:
            if failed_state_reason is not None:
                return PersistentAuthorityDecision(
                    ExecutionAuthority.FAIL_CLOSED,
                    failed_state_reason,
                    False,
                    False,
                    False,
                    False,
                    False,
                )
            departure_checks = (
                (inputs.generator_available, "NO_DEPARTURE_GENERATOR_SET"),
                (inputs.rl_authority_member, "DEPARTURE_RL_AUTHORITY_SET_MEMBERSHIP_FAILED"),
                (inputs.recoverability_action_verified, "DEPARTURE_GENERATOR_NOT_CONTAINED_IN_A_REC"),
                (inputs.continuation_action_verified, "DEPARTURE_GENERATOR_NOT_CONTAINED_IN_A_CONT"),
                (inputs.policy_authority_pass, "DEPARTURE_POLICY_AUTHORITY_GATE_FAILED"),
            )
            failed_departure_reason = next(
                (reason for valid, reason in departure_checks if not valid),
                None,
            )
            if failed_departure_reason is not None:
                return PersistentAuthorityDecision(
                    ExecutionAuthority.FAIL_CLOSED,
                    failed_departure_reason,
                    False,
                    False,
                    False,
                    False,
                    False,
                )
            return PersistentAuthorityDecision(
                ExecutionAuthority.RL_GENERATOR,
                "VERIFIED_DEPARTURE_AUTHORITY",
                True,
                False,
                True,
                False,
                False,
            )

        for valid, reason in state_checks:
            if not valid:
                return recovery_or_fail(reason)
        if inputs.energy_margin <= inputs.backup_switch_margin:
            return recovery_or_fail("ENERGY_MARGIN_BACKUP_SWITCH")
        generator_checks = (
            (inputs.generator_available, "NO_GENERATOR_SET"),
            (inputs.rl_authority_member, "RL_AUTHORITY_SET_MEMBERSHIP_FAILED"),
            (inputs.recoverability_action_verified, "GENERATOR_NOT_CONTAINED_IN_A_REC"),
            (inputs.continuation_action_verified, "GENERATOR_NOT_CONTAINED_IN_A_CONT"),
            (inputs.policy_authority_pass, "POLICY_AUTHORITY_GATE_FAILED"),
        )
        for valid, reason in generator_checks:
            if not valid:
                return recovery_or_fail(reason)
        return PersistentAuthorityDecision(
            ExecutionAuthority.RL_GENERATOR,
            "VERIFIED_RL_GENERATOR_AUTHORITY",
            True,
            False,
            inputs.departure_allowed if inputs.charging_state else True,
            False,
            False,
        )
