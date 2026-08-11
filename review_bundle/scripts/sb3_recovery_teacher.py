from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from cert_runtime.certificates import certificate_hash
from envs.certified_uav import make_random_persistent_uav_env
from envs.certified_uav.charging import verify_departure_energy
from envs.certified_uav.persistent_task import PersistentMissionMode


CERTIFICATE_SCHEMA = "certified-recovery-atlas-energy-binding-v1"


def certificate_descriptor(certified_environment) -> dict[str, Any]:
    manifest = certified_environment.atlas.persistent_manifest
    energy = certified_environment.runtime.calibration.energy
    descriptor = {
        "schema": CERTIFICATE_SCHEMA,
        "scenario": manifest.scenario_id,
        "flight_energy_multiplier": manifest.flight_energy_multiplier,
        "energy_version": manifest.energy_version,
        "energy_contract_hash": energy.contract_hash,
        "recovery_manifest_hash": manifest.recovery_manifest_hash,
        "recovery_atlas_hash": manifest.manifest_hash,
        "terminal_recovery_certificate_hash": manifest.terminal_recovery_certificate_hash,
        "kappa_version": manifest.kappa_version,
        "number_of_cells": manifest.number_of_cells,
        "gate_pass": manifest.gate_pass,
    }
    return descriptor | {"binding_hash": certificate_hash(descriptor)}


def write_certificate_artifact(path: Path, certified_environment) -> dict[str, Any]:
    descriptor = certificate_descriptor(certified_environment)
    if not descriptor["gate_pass"]:
        raise RuntimeError("RECOVERY_CERTIFICATE_GATE_FAILED")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != descriptor:
            raise FileExistsError(f"refusing to overwrite incompatible certificate artifact {path}")
        return descriptor
    path.write_text(json.dumps(descriptor, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return descriptor


def load_and_verify_certificate_artifact(path: Path, certified_environment) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"2x certificate artifact does not exist: {path}")
    stored = json.loads(path.read_text(encoding="utf-8"))
    expected = certificate_descriptor(certified_environment)
    if stored != expected:
        raise RuntimeError(
            "CERTIFICATE_RUNTIME_DEPENDENCY_MISMATCH: "
            f"artifact multiplier={stored.get('flight_energy_multiplier')} "
            f"runtime multiplier={expected['flight_energy_multiplier']}"
        )
    if not stored["gate_pass"]:
        raise RuntimeError("RECOVERY_CERTIFICATE_GATE_FAILED")
    return stored


@dataclass(slots=True)
class TeacherTransition:
    observation: np.ndarray
    normalized_action: np.ndarray
    reward: float
    next_observation: np.ndarray
    terminated: bool
    truncated: bool
    info: dict[str, Any]
    authority: str


@dataclass(slots=True)
class TeacherCycleResult:
    success: bool
    failure_reason: str | None
    transitions: list[TeacherTransition]
    kappa_observations: list[np.ndarray]
    kappa_actions: list[np.ndarray]
    recovery_transitions: int
    charging_steps: int
    resume_successes: int
    original_goal_completions: int


class CertifiedRecoveryOracle:
    """Certificate/κ oracle isolated from the clean SB3 actor observation."""

    def __init__(
        self,
        scenario: str,
        flight_energy_multiplier: float,
        certificate_artifact: Path | None,
    ) -> None:
        self.environment = make_random_persistent_uav_env(
            scenario,
            timing_mode="functional",
            flight_energy_multiplier=flight_energy_multiplier,
        )
        self.runtime = self.environment.runtime
        self.atlas = self.environment.atlas
        self.flight_energy_multiplier = float(flight_energy_multiplier)
        self.certificate = (
            certificate_descriptor(self.environment)
            if certificate_artifact is None
            else load_and_verify_certificate_artifact(certificate_artifact, self.environment)
        )
        if not self.certificate["gate_pass"]:
            raise RuntimeError("RECOVERY_CERTIFICATE_GATE_FAILED")

    @property
    def guidance_margin(self) -> float:
        return float(self.atlas.trigger_margin)

    def reset(self) -> None:
        self.atlas.reset()

    def _sync(self, actor_environment, mode: PersistentMissionMode) -> Any:
        self.runtime.plant.state = actor_environment.state.copy()
        self.runtime.plant.step_count = actor_environment.episode_step
        self.runtime.plant.last_lidar = self.runtime.plant.lidar_model.measure(
            self.runtime.plant.state,
            self.runtime.plant.world,
            self.runtime.plant.np_random,
        )
        self.runtime.task_env.mode = mode
        self.runtime.task_env.phase = mode
        self.runtime.task_env.episode_step = actor_environment.episode_step
        return self.runtime._certificate_state()

    def context(self, actor_environment, *, recovery: bool) -> tuple[Any, Any]:
        mode = PersistentMissionMode.BACKUP_RECOVERY if recovery else PersistentMissionMode.TASK_RL
        state = self._sync(actor_environment, mode)
        if recovery:
            self.atlas.recovery_active = True
        context = self.atlas.evaluate(state, actor_environment.state.timestamp)
        return state, context

    def begin_recovery(self) -> None:
        self.atlas.recovery_active = True
        self.atlas.active_cell_id = None

    def end_recovery(self) -> None:
        self.atlas.recovery_active = False
        self.atlas.active_cell_id = None

    def recovery_action(self, actor_environment) -> tuple[np.ndarray, Any]:
        _, context = self.context(actor_environment, recovery=True)
        if not context.recovery.certified:
            raise RuntimeError(f"KAPPA_RECOVERY_UNCERTIFIED:{context.recovery.reason}")
        physical = np.asarray(context.recovery.action, dtype=np.float64)
        normalized = np.clip(physical / actor_environment.config.a_max, -1.0, 1.0)
        if not np.allclose(actor_environment.normalized_to_physical_action(normalized), physical, atol=1e-8):
            raise RuntimeError("KAPPA_ACTION_NORMALIZATION_MISMATCH")
        return normalized.astype(np.float32), context

    def commit_recovery(self, context: Any) -> None:
        self.atlas.commit_execution(context, False)

    def commit_policy(self, context: Any) -> None:
        self.atlas.commit_execution(context, True)

    def hold_action(self, actor_environment) -> np.ndarray:
        state = self._sync(actor_environment, PersistentMissionMode.CHARGING_RL)
        physical = self.atlas.certified_station_hold_action(state)
        if physical is None:
            raise RuntimeError("CERTIFIED_CHARGER_HOLD_UNAVAILABLE")
        return np.clip(
            np.asarray(physical, dtype=np.float64) / actor_environment.config.a_max,
            -1.0,
            1.0,
        ).astype(np.float32)

    def departure_allowed(self, actor_environment) -> bool:
        result = verify_departure_energy(
            actor_environment.state.energy,
            self.atlas.required_departure_energy(),
            self.environment.charging.config.departure_energy_margin,
            self.atlas.gate_pass,
        )
        return bool(result.allowed)

    def certified_departure_action(
        self,
        actor_environment,
        normalized_action: np.ndarray,
    ) -> bool:
        if not self.departure_allowed(actor_environment):
            return False
        self.atlas.configure_charging_support(False)
        state = self._sync(actor_environment, PersistentMissionMode.CHARGING_RL)
        context = self.atlas.evaluate(state, actor_environment.state.timestamp)
        certificate = context.closure.zonotope_certificate
        physical = actor_environment.normalized_to_physical_action(normalized_action)
        return bool(
            context.recovery.certified
            and certificate is not None
            and certificate.verified
            and certificate.zonotope.contains(physical)
            and self.atlas.last_recoverability_action_certificate is not None
            and self.atlas.last_recoverability_action_certificate.verified
            and self.atlas.last_continuation_verified
            and self.atlas.contains_rl_authority_state(state)
        )

    def certified_support_action(
        self,
        actor_environment,
        normalized_support_action: np.ndarray,
        *,
        charging_state: bool,
        charging_support_required: bool,
    ) -> tuple[np.ndarray, Any] | None:
        selected = np.asarray(normalized_support_action, dtype=np.float64)
        if selected.shape != (3,) or not np.all(np.isfinite(selected)):
            return None
        selected = np.clip(selected, -1.0, 1.0)
        self.atlas.configure_charging_support(charging_support_required)
        mode = PersistentMissionMode.CHARGING_RL if charging_state else PersistentMissionMode.TASK_RL
        state = self._sync(actor_environment, mode)
        context = self.atlas.evaluate(state, actor_environment.state.timestamp)
        certificate = context.closure.zonotope_certificate
        if (
            not context.recovery.certified
            or certificate is None
            or not certificate.verified
            or self.atlas.last_recoverability_action_certificate is None
            or not self.atlas.last_recoverability_action_certificate.verified
            or not self.atlas.last_continuation_verified
        ):
            return None
        if charging_support_required and not self.atlas.last_charging_support_verified:
            return None
        if not charging_state and (
            context.current_energy_margin <= self.guidance_margin
            or not self.atlas.contains_rl_authority_state(state)
        ):
            return None
        zonotope = certificate.zonotope
        physical = np.asarray(zonotope.center, dtype=np.float64) + np.asarray(
            zonotope.generators,
            dtype=np.float64,
        ) @ selected
        if not zonotope.contains(physical):
            raise RuntimeError("CERTIFIED_SUPPORT_ACTION_MAPPING_FAILED")
        normalized_physical = np.clip(
            physical / actor_environment.config.a_max,
            -1.0,
            1.0,
        )
        if not np.allclose(
            actor_environment.normalized_to_physical_action(normalized_physical),
            physical,
            atol=1e-8,
        ):
            raise RuntimeError("CERTIFIED_SUPPORT_ACTION_NORMALIZATION_MISMATCH")
        return normalized_physical.astype(np.float32), context

    def certified_policy_action(self, actor_environment, normalized_action: np.ndarray) -> bool:
        self.atlas.configure_charging_support(False)
        state, context = self.context(actor_environment, recovery=False)
        certificate = context.closure.zonotope_certificate
        physical = actor_environment.normalized_to_physical_action(normalized_action)
        return bool(
            context.recovery.certified
            and context.current_energy_margin > self.guidance_margin
            and certificate is not None
            and certificate.verified
            and certificate.zonotope.contains(physical)
            and self.atlas.contains_rl_authority_state(state)
        )


class CertifiedRecoveryRolloutTeacher:
    def __init__(self, navigation_teacher_model, oracle: CertifiedRecoveryOracle) -> None:
        self.navigation_teacher_model = navigation_teacher_model
        self.oracle = oracle

    def _task_action(self, observation: np.ndarray) -> np.ndarray:
        action, _ = self.navigation_teacher_model.predict(observation, deterministic=True)
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (3,) or not np.all(np.isfinite(action)):
            raise RuntimeError("NAVIGATION_TEACHER_ACTION_INVALID")
        return np.clip(action, -1.0, 1.0)

    @staticmethod
    def _transition(
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_observation: np.ndarray,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
        authority: str,
    ) -> TeacherTransition:
        return TeacherTransition(
            observation.copy(),
            action.copy(),
            float(reward),
            next_observation.copy(),
            bool(terminated),
            bool(truncated),
            dict(info),
            authority,
        )

    def generate_cycle(self, actor_environment, seed: int, max_steps: int) -> TeacherCycleResult:
        self.oracle.reset()
        certified_start = self.oracle.atlas.sample_initial_state(
            seed,
            actor_environment.energy_navigation_config.battery_capacity,
        )
        goal_rng = np.random.default_rng(seed + 1_000_003)
        goal = self.oracle.atlas.sample_goal(
            goal_rng,
            certified_start.position,
            actor_environment.minimum_goal_separation,
        )
        observation, _ = actor_environment.reset(
            seed=seed,
            options={
                "start_position": certified_start.position,
                "start_velocity": certified_start.velocity,
                "goal_position": goal,
                "initial_energy_fraction": certified_start.energy
                / actor_environment.energy_navigation_config.battery_capacity,
            },
        )
        _, initial_context = self.oracle.context(actor_environment, recovery=False)
        if not initial_context.recovery.certified:
            return TeacherCycleResult(False, "INITIAL_STATE_NOT_RECOVERABLE", [], [], [], 0, 0, 0, 0)
        guided_initial_energy = min(
            actor_environment.energy_navigation_config.battery_capacity,
            initial_context.required_energy + 0.5 * self.oracle.guidance_margin,
        )
        actor_environment.state.energy = guided_initial_energy
        observation = actor_environment._observation()
        original_goal = actor_environment.goal.copy()
        original_goal_id = str(actor_environment._goal_attempt["goal_id"])
        transitions: list[TeacherTransition] = []
        kappa_observations: list[np.ndarray] = []
        kappa_actions: list[np.ndarray] = []
        phase = "task_before_recovery"
        recovery_transitions = 0
        charging_steps = 0
        resume_successes = 0
        original_goal_completions = 0
        charged_energy = 0.0
        task_steps_before_recovery = 0

        for _ in range(max_steps):
            if not np.array_equal(actor_environment.goal, original_goal):
                return TeacherCycleResult(False, "PENDING_GOAL_CHANGED_BEFORE_COMPLETION", transitions, kappa_observations, kappa_actions, recovery_transitions, charging_steps, resume_successes, original_goal_completions)
            authority = "task_teacher"
            context = None
            if phase in {"task_before_recovery", "resume_task"}:
                _, task_context = self.oracle.context(actor_environment, recovery=False)
                should_recover = (
                    task_steps_before_recovery > 0
                    and task_context.recovery.certified
                    and task_context.current_energy_margin <= self.oracle.guidance_margin
                )
                if should_recover:
                    self.oracle.begin_recovery()
                    phase = "recovery"
                else:
                    action = self._task_action(observation)
                    task_steps_before_recovery += 1
            if phase == "recovery":
                action, context = self.oracle.recovery_action(actor_environment)
                authority = "frozen_kappa"
                kappa_observations.append(observation.copy())
                kappa_actions.append(action.copy())
            elif phase == "charging_hold":
                if charged_energy > 0.0 and self.oracle.departure_allowed(actor_environment):
                    self.oracle.end_recovery()
                    phase = "resume_task"
                    resume_successes += 1
                    action = self._task_action(observation)
                    authority = "task_teacher_resume"
                else:
                    action = self.oracle.hold_action(actor_environment)
                    authority = "certified_charger_hold"

            next_observation, reward, terminated, truncated, info = actor_environment.step(action)
            transitions.append(self._transition(observation, action, reward, next_observation, terminated, truncated, info, authority))
            if context is not None:
                self.oracle.commit_recovery(context)
                recovery_transitions += 1
            if authority == "frozen_kappa" and info["collision"]:
                return TeacherCycleResult(False, "COLLISION_DURING_CERTIFIED_RECOVERY", transitions, kappa_observations, kappa_actions, recovery_transitions, charging_steps, resume_successes, original_goal_completions)
            if info["energy_stranded"]:
                return TeacherCycleResult(False, "ENERGY_STRANDED", transitions, kappa_observations, kappa_actions, recovery_transitions, charging_steps, resume_successes, original_goal_completions)
            if phase == "recovery" and info["charging"]:
                phase = "charging_hold"
            if info["charging"]:
                charging_steps += 1
                charged_energy += float(info["gross_charge_received"])
            if info["task_completed_now"]:
                if str(info["goal_id_before"]) != original_goal_id:
                    return TeacherCycleResult(False, "WRONG_GOAL_COMPLETED", transitions, kappa_observations, kappa_actions, recovery_transitions, charging_steps, resume_successes, original_goal_completions)
                original_goal_completions = 1
                success = recovery_transitions > 0 and charging_steps > 0 and resume_successes > 0
                return TeacherCycleResult(success, None if success else "INCOMPLETE_RECOVERY_CYCLE", transitions, kappa_observations, kappa_actions, recovery_transitions, charging_steps, resume_successes, original_goal_completions)
            if terminated:
                return TeacherCycleResult(False, "UNEXPECTED_TERMINATION", transitions, kappa_observations, kappa_actions, recovery_transitions, charging_steps, resume_successes, original_goal_completions)
            if truncated:
                return TeacherCycleResult(False, "TRAJECTORY_TIMEOUT", transitions, kappa_observations, kappa_actions, recovery_transitions, charging_steps, resume_successes, original_goal_completions)
            observation = next_observation
        return TeacherCycleResult(False, "TRAJECTORY_TIMEOUT", transitions, kappa_observations, kappa_actions, recovery_transitions, charging_steps, resume_successes, original_goal_completions)


def collect_successful_teacher_trajectories(
    teacher: CertifiedRecoveryRolloutTeacher,
    environment_factory,
    target_transitions: int,
    max_rollout_steps: int,
    teacher_seed: int,
    max_cycles: int,
) -> tuple[list[TeacherTransition], np.ndarray, np.ndarray, dict[str, Any]]:
    accepted: list[TeacherTransition] = []
    kappa_observations: list[np.ndarray] = []
    kappa_actions: list[np.ndarray] = []
    telemetry = {
        "teacher_dataset_selection": "OUTCOME_CONDITIONED_COMPLETE_SUCCESS_CYCLES",
        "teacher_bellman_unbiasedness_claim": False,
        "teacher_cycles_attempted": 0,
        "teacher_cycles_successful": 0,
        "teacher_transitions_generated": 0,
        "teacher_transitions_inserted": 0,
        "teacher_recovery_transitions": 0,
        "teacher_charging_steps": 0,
        "teacher_resume_successes": 0,
        "teacher_original_goal_completions": 0,
        "teacher_failures": {},
    }
    for cycle_index in range(max_cycles):
        environment = environment_factory()
        try:
            result = teacher.generate_cycle(
                environment,
                teacher_seed + cycle_index,
                max_rollout_steps,
            )
        finally:
            environment.close()
        telemetry["teacher_cycles_attempted"] += 1
        telemetry["teacher_transitions_generated"] += len(result.transitions)
        if not result.success:
            reason = result.failure_reason or "UNKNOWN"
            telemetry["teacher_failures"][reason] = telemetry["teacher_failures"].get(reason, 0) + 1
            continue
        telemetry["teacher_cycles_successful"] += 1
        accepted.extend(result.transitions)
        kappa_observations.extend(result.kappa_observations)
        kappa_actions.extend(result.kappa_actions)
        telemetry["teacher_recovery_transitions"] += result.recovery_transitions
        telemetry["teacher_charging_steps"] += result.charging_steps
        telemetry["teacher_resume_successes"] += result.resume_successes
        telemetry["teacher_original_goal_completions"] += result.original_goal_completions
        if len(accepted) >= target_transitions:
            break
    telemetry["teacher_transitions_inserted"] = len(accepted)
    telemetry["teacher_full_trajectory_success_rate"] = (
        telemetry["teacher_cycles_successful"] / max(1, telemetry["teacher_cycles_attempted"])
    )
    if target_transitions > 0 and not accepted:
        raise RuntimeError(f"NO_COMPLETE_TEACHER_TRAJECTORY:{telemetry['teacher_failures']}")
    return (
        accepted,
        np.asarray(kappa_observations, dtype=np.float32),
        np.asarray(kappa_actions, dtype=np.float32),
        telemetry,
    )


def prefill_sb3_replay_buffer(model, transitions: Iterable[TeacherTransition]) -> int:
    inserted = 0
    for transition in transitions:
        info = dict(transition.info)
        info["teacher_authority"] = transition.authority
        info["TimeLimit.truncated"] = bool(transition.truncated and not transition.terminated)
        model.replay_buffer.add(
            transition.observation.reshape(1, -1),
            transition.next_observation.reshape(1, -1),
            transition.normalized_action.reshape(1, -1),
            np.asarray([transition.reward], dtype=np.float32),
            np.asarray([transition.terminated or transition.truncated], dtype=np.float32),
            [info],
        )
        inserted += 1
    return inserted


def supervised_actor_warm_start(
    model,
    observations: np.ndarray,
    normalized_actions: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> list[float]:
    if observations.ndim != 2 or normalized_actions.shape != (len(observations), 3):
        raise ValueError("invalid κ warm-start dataset")
    if len(observations) == 0:
        raise ValueError("κ warm-start requires at least one recovery transition")
    critic_before = {key: value.detach().clone() for key, value in model.critic.state_dict().items()}
    target_before = {key: value.detach().clone() for key, value in model.critic_target.state_dict().items()}
    entropy_before = None if model.log_ent_coef is None else model.log_ent_coef.detach().clone()
    optimizer = torch.optim.Adam(model.actor.parameters(), lr=learning_rate)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    losses: list[float] = []
    device = model.device
    for _ in range(epochs):
        permutation = torch.randperm(len(observations), generator=generator)
        epoch_losses = []
        for start in range(0, len(observations), batch_size):
            indices = permutation[start : start + batch_size].numpy()
            obs = torch.as_tensor(observations[indices], device=device)
            target = torch.as_tensor(normalized_actions[indices], device=device)
            prediction = model.actor(obs, deterministic=True)
            loss = torch.nn.functional.mse_loss(prediction, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        losses.append(float(np.mean(epoch_losses)))
    for key, before in critic_before.items():
        if not torch.equal(before, model.critic.state_dict()[key]):
            raise RuntimeError("ACTOR_WARMSTART_MODIFIED_CRITIC")
    for key, before in target_before.items():
        if not torch.equal(before, model.critic_target.state_dict()[key]):
            raise RuntimeError("ACTOR_WARMSTART_MODIFIED_TARGET_CRITIC")
    if entropy_before is not None and not torch.equal(entropy_before, model.log_ent_coef.detach()):
        raise RuntimeError("ACTOR_WARMSTART_MODIFIED_ENTROPY_COEFFICIENT")
    return losses
