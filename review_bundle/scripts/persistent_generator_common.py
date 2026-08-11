from __future__ import annotations

from typing import Any

import numpy as np

from cert_runtime.generator_sac import GeneratorTransition
from cert_runtime.persistent_authority import ExecutionAuthority


def successor_context_for_transition(
    environment: Any,
    *,
    terminated: bool,
) -> dict[str, Any] | None:
    """Return the physical successor context whenever Bellman bootstrapping is legal.

    A Gymnasium time-limit truncation ends the collector episode, not the
    underlying MDP transition.  Its real post-step certificate/authority context
    must therefore be recorded before reset when ``bootstrap_on_truncation`` is
    enabled.  Only a true terminal transition has no continuation context.
    """
    if terminated:
        return None
    return environment._refresh_context()


def transition_from_cycle(
    observation: np.ndarray,
    next_observation: np.ndarray,
    actor_u: np.ndarray,
    reward: float,
    terminated: bool,
    truncated: bool,
    episode_id: int,
    context: dict[str, Any],
    next_context: dict[str, Any] | None,
    info: dict[str, Any],
    *,
    collector_boundary: bool = False,
) -> GeneratorTransition:
    """Build an immutable persistent transition from actual runtime execution."""
    telemetry = info["telemetry"]
    trace = telemetry.action_trace
    accepted = bool(info.get("accepted", False))
    runtime_publication_context = info.get("action_context")
    if accepted and not isinstance(runtime_publication_context, dict):
        raise ValueError("accepted transition requires publication-time action_context")
    publication_context = dict(context)
    if isinstance(runtime_publication_context, dict):
        publication_context.update(runtime_publication_context)
    if publication_context.get("certificate_epoch") is None:
        raise ValueError("transition requires a publication-time certificate epoch")
    if accepted and (
        publication_context.get("c") is None
        or publication_context.get("G") is None
    ):
        raise ValueError("accepted transition requires publication-time c and G")
    epoch = str(publication_context["certificate_epoch"])
    selected_next = {} if next_context is None else next_context
    next_epoch = epoch if next_context is None else str(selected_next["certificate_epoch"])
    next_authority = str(selected_next.get("execution_authority", ExecutionAuthority.FAIL_CLOSED.value))
    next_generator_executable = bool(selected_next.get("generator_executable", False))
    next_kappa = np.asarray(selected_next.get("kappa", trace.fallback), dtype=np.float32)
    if next_authority == ExecutionAuthority.CHARGER_CONSTRAINED.value and not next_generator_executable:
        next_authority_action = np.asarray(
            selected_next.get("certified_station_hold_action", next_kappa),
            dtype=np.float32,
        )
    else:
        next_authority_action = next_kappa.copy()
    candidate = None if trace.candidate is None else np.asarray(trace.candidate, dtype=np.float32)
    eta = np.tanh(np.asarray(actor_u, dtype=np.float32)) if accepted else None
    task_goal_value = info.get("goal_before", info.get("current_goal"))
    task_goal = None if task_goal_value is None else np.asarray(task_goal_value, dtype=np.float32)
    current_required_energy = publication_context.get("recovery_energy_required")
    if current_required_energy is None:
        current_required_energy = info.get("required_return_energy", np.nan)
    current_energy_margin = publication_context.get("energy_margin")
    if current_energy_margin is None:
        current_energy_margin = info.get("energy_margin", np.nan)
    return GeneratorTransition(
        observation=np.asarray(observation, dtype=np.float32),
        next_observation=np.asarray(next_observation, dtype=np.float32),
        reward=float(reward),
        terminated=bool(terminated),
        truncated=bool(truncated),
        episode_id=int(episode_id),
        mission_phase=str(publication_context.get("persistent_mode", "TASK_RL")),
        next_mission_phase=str(selected_next.get("persistent_mode", info.get("persistent_mode", "TASK_RL"))),
        certificate_epoch=epoch,
        next_certificate_epoch=next_epoch,
        u=np.asarray(actor_u, dtype=np.float32) if accepted else None,
        eta=eta,
        c=None if not accepted else publication_context.get("c"),
        G=None if not accepted else publication_context.get("G"),
        candidate_action=candidate,
        kappa_action=np.asarray(trace.fallback, dtype=np.float32),
        executed_action=np.asarray(trace.published, dtype=np.float32),
        measured_action=np.asarray(trace.measured, dtype=np.float32),
        accepted=accepted,
        fallback_reason=info.get("fallback_reason"),
        next_c=selected_next.get("c") if next_generator_executable else None,
        next_G=selected_next.get("G") if next_generator_executable else None,
        next_kappa=next_kappa,
        next_generator_available=next_generator_executable,
        next_certificate_valid=bool(selected_next.get("certificate_valid", False)),
        geometry_version=str(publication_context.get("geometry_version", "")),
        corridor_version=str(publication_context.get("corridor_version", "")),
        energy_version=str(publication_context.get("energy_version", "")),
        certificate_hashes=(
            publication_context.get("recovery_hash"),
            publication_context.get("zonotope_hash"),
        ),
        scenario_id=str(info.get("scenario_id", "persistent")),
        certificate_manifest_hash=publication_context.get("certificate_manifest_hash"),
        backup_triggered=bool(info.get("backup_triggered", False)),
        backup_started_now=bool(info.get("backup_started_now", False)),
        backup_reason=info.get("backup_reason"),
        energy=float(telemetry.state_before.energy),
        required_return_energy=float(current_required_energy),
        energy_margin=float(current_energy_margin),
        charging=bool(info.get("charging", False)),
        station_arrival=bool(info.get("voluntary_station_arrival", False)),
        departure_attempt=bool(info.get("departure_attempt", False)),
        departure_rejected=bool(info.get("departure_rejected", False)),
        task_id=info.get("task_id"),
        goal_id=info.get("current_goal_id"),
        tasks_completed=int(info.get("tasks_completed", 0)),
        recoverable_set_version=publication_context.get("recoverable_set_version"),
        recoverability_action_rule_version=publication_context.get("recoverability_action_rule_version"),
        execution_authority=str(
            info.get("execution_authority", publication_context.get("execution_authority", ""))
        ),
        next_execution_authority=next_authority,
        next_generator_executable=next_generator_executable,
        next_backup_required=bool(selected_next.get("backup_required", False)),
        next_backup_reason=selected_next.get("execution_authority_reason") if selected_next.get("backup_required", False) else None,
        next_recoverable_set_member=selected_next.get("recoverable_set_member") is True,
        next_recoverability_action_verified=selected_next.get("recoverability_action_verified") is True,
        next_policy_authority_pass=selected_next.get("policy_authority_pass") is True,
        next_energy_margin=float(selected_next.get("energy_margin", np.nan)),
        next_departure_allowed=selected_next.get("departure_allowed") is True,
        next_charging_state=str(selected_next.get("persistent_mode", "")) == "CHARGING_RL",
        next_charging_restriction=selected_next.get("charging_restriction") is True,
        next_charging_support_verified=selected_next.get("charging_support_verified") is True,
        next_station_hold_valid=selected_next.get("station_hold_valid") is True,
        next_authority_action=next_authority_action,
        task_goal=task_goal,
        collector_boundary=bool(collector_boundary),
    )
