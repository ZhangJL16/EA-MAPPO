from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from envs.UAVEnergyDeliverySAC import SACTrainingPhase, UAVEnergyDeliverySACEnv
from experiments.energy_mc.gate_b_prerequisites import navigation_artifact_view
from experiments.energy_mc.return_decision import (
    CloneRolloutVarianceAudit,
    clone_rollout_variance_audit,
    probability_semantics_gate,
)
from review_bundle.safety.energy.mc_regression import ModelBasedEnergyRolloutEstimator
from scripts.evaluate_jseb_checkpoints import file_sha256, reconstruct_environment_args
from scripts.run_return_decision_stage_b import FrozenPolicy, load_policy
from scripts.train_uav_energy_delivery_sac import environment_kwargs_from_args


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit the conditional probability semantics of the frozen "
            "return-energy Oracle without authorizing downstream Gates"
        )
    )
    parser.add_argument("--navigation-artifact", type=Path, required=True)
    parser.add_argument("--navigation-checkpoint", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=170_001)
    parser.add_argument("--num-rollouts", type=int, default=3)
    parser.add_argument("--max-policy-steps", type=int, default=4000)
    args = parser.parse_args(argv)
    if args.num_rollouts < 2:
        parser.error("--num-rollouts must be at least two")
    if args.max_policy_steps <= 0:
        parser.error("--max-policy-steps must be positive")
    if args.output_json.exists():
        parser.error(f"output already exists: {args.output_json}")
    return args


def _checkpoint_from_wrapper(artifact: Path) -> tuple[Path, dict[str, object]]:
    wrapper_candidates = (
        artifact / "EVALUATION_COMPLETED.json",
        artifact / "STOPPED_NAVIGATION_NOT_READY.json",
    )
    wrapper_path = next((path for path in wrapper_candidates if path.is_file()), None)
    if wrapper_path is None:
        raise FileNotFoundError(
            "navigation artifact has no completion/stop wrapper"
        )
    wrapper = json.loads(wrapper_path.read_text(encoding="utf-8"))
    if not isinstance(wrapper, dict):
        raise TypeError("navigation wrapper must contain a JSON object")
    checkpoint = wrapper.get("checkpoint")
    if not isinstance(checkpoint, str):
        raise ValueError("navigation wrapper does not name its checkpoint")
    path = Path(checkpoint)
    if not path.is_absolute():
        path = artifact / path
    return path.resolve(), wrapper


def _snapshot_payload(environment: UAVEnergyDeliverySACEnv) -> dict[str, object]:
    return {
        "position": environment.agent.pos.astype(float).tolist(),
        "velocity": environment.agent.vel.astype(float).tolist(),
        "task_goal": environment.current_task_point.astype(float).tolist(),
        "charger_position": environment.charger_position.astype(float).tolist(),
        "remaining_energy": float(environment.agent.energy),
        "obstacles": environment.static_obstacle_layout(),
        "policy_step": int(environment.current_step),
    }


def _sha256_json(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def probability_semantics_payload(
    *,
    variance_audit: dict[str, object],
    estimator_declares_exact_deterministic_return: bool,
    snapshot_sha256: str,
    repeated_rollout_seconds: float,
) -> dict[str, object]:
    audit = CloneRolloutVarianceAudit(**variance_audit)
    gate = probability_semantics_gate(
        audit,
        physically_specified_future_disturbance=False,
    )
    if not estimator_declares_exact_deterministic_return and gate["passed"] is True:
        gate = {
            "status": "UNRESOLVED_ESTIMATOR_SEMANTICS",
            "passed": False,
            "probability_object": "unresolved_conditional_resource_object",
            "aleatoric_q90_q95_claim_authorized": False,
            "reason": "the estimator does not declare an exact deterministic return",
        }
    return {
        **gate,
        "reason_aleatoric_claim_not_authorized": (
            "the executed simulator has no physically specified future "
            "disturbance law conditioned on deployment information"
        ),
        "fixed_information": [
            "state",
            "task_goal",
            "charger",
            "frozen_deterministic_policy",
            "deterministic_safety_operator",
            "static_obstacle_realization",
            "telemetry_cost_model",
        ],
        "randomness_audit": {
            "task_and_obstacle_sampling": (
                "conditioned_as_part_of_the_fixed_snapshot"
            ),
            "transition_noise_model": None,
            "wind_model": None,
            "moving_obstacle_model": None,
            "future_disturbance_seed_consumed_by_oracle": False,
            "repetition_index_is_not_a_disturbance_seed": True,
        },
        "snapshot_sha256": snapshot_sha256,
        "estimator_declares_exact_deterministic_return": bool(
            estimator_declares_exact_deterministic_return
        ),
        "repeated_rollout_wall_clock_seconds": float(repeated_rollout_seconds),
        "clone_rollout_variance_audit": variance_audit,
        "claim_boundary": (
            "this audit identifies simulator probability semantics only; it "
            "does not establish navigation readiness, calibration, Oracle "
            "decision headroom, or Pareto improvement"
        ),
    }


def audit_platform(args: argparse.Namespace) -> dict[str, object]:
    artifact = args.navigation_artifact.resolve()
    checkpoint_from_wrapper, wrapper = _checkpoint_from_wrapper(artifact)
    checkpoint = (
        checkpoint_from_wrapper
        if args.navigation_checkpoint is None
        else args.navigation_checkpoint.resolve()
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    view = navigation_artifact_view(wrapper)
    checkpoint_sha = file_sha256(checkpoint)
    if view.checkpoint_sha256 != checkpoint_sha:
        raise RuntimeError(
            "loaded checkpoint SHA does not match the navigation wrapper"
        )

    source_args, reconstruction = reconstruct_environment_args(
        artifact,
        device=args.device,
        seed=args.seed,
    )
    kwargs = environment_kwargs_from_args(
        source_args,
        phase=SACTrainingPhase.NAVIGATION,
    )
    environment = UAVEnergyDeliverySACEnv(**kwargs)
    policy_args = argparse.Namespace(
        navigation_checkpoint=checkpoint,
        smoke=False,
        device=args.device,
    )
    policy: FrozenPolicy = load_policy(policy_args, environment)
    environment.bind_navigation_policy(policy.action)
    environment.reset(seed=args.seed)
    snapshot = _snapshot_payload(environment)
    snapshot_sha = _sha256_json(snapshot)
    goal = environment.current_task_point.copy()

    started = time.perf_counter()

    def rollout(_: int) -> float:
        oracle = ModelBasedEnergyRolloutEstimator(
            policy,
            max_policy_steps=args.max_policy_steps,
            cache_mission_suffixes=False,
        )
        return float(oracle.estimate_context(environment, goal).prediction)

    variance = clone_rollout_variance_audit(
        rollout,
        num_rollouts=args.num_rollouts,
    )
    elapsed = time.perf_counter() - started
    environment.close()
    semantics = probability_semantics_payload(
        variance_audit=variance.as_dict(),
        estimator_declares_exact_deterministic_return=bool(
            ModelBasedEnergyRolloutEstimator.returns_deterministic_exact
        ),
        snapshot_sha256=snapshot_sha,
        repeated_rollout_seconds=elapsed,
    )
    return {
        "schema_version": "return-probability-semantics-v1",
        "evidence_mode": "PLATFORM_DIAGNOSTIC_NOT_GATE_AUTHORIZATION",
        "navigation_artifact": str(artifact),
        "navigation_artifact_schema": view.schema,
        "navigation_wrapper_status": wrapper.get("status"),
        "navigation_gate_passed": wrapper.get("navigation_gate_passed"),
        "downstream_stages_authorized": wrapper.get(
            "downstream_stages_authorized"
        ),
        "wrapper_authorization_passed": view.authorization_passed,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "navigation_config_sha256": file_sha256(artifact / "config.json"),
        "environment_reconstruction": reconstruction.get(
            "environment_reconstruction"
        ),
        "seed": int(args.seed),
        "num_repeated_fixed_snapshot_rollouts": int(args.num_rollouts),
        "snapshot": snapshot,
        "probability_semantics": semantics,
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    payload = audit_platform(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["probability_semantics"], sort_keys=True))


if __name__ == "__main__":
    main()
