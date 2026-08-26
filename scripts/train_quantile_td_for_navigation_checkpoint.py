from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import torch

from envs.UAVEnergyDeliverySAC import GoalConditionedQuantileTDEnergyEstimator
from experiments.jacobian_energy_bridge.sac import JacobianBridgeSAC
from scripts.calibrate_battery_for_navigation_checkpoint import (
    FORMAL_NAVIGATION_TRANSITIONS,
    navigation_readiness_failures,
)
from scripts.evaluate_jseb_checkpoints import reconstruct_environment_args
from scripts.train_uav_energy_delivery_sac import (
    freeze_navigation_and_start_td,
    generate_stratified_navigation_tasks,
    navigation_observation_dim,
    run_td_pretraining,
    save_navigation_tasks,
)


FORMAL_TD_TRANSITIONS = 500_000
FORMAL_ENERGY_EVAL_TASKS = 500
FORMAL_ENERGY_EVAL_FREQUENCY = 50_000


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def policy_parameter_hash(policy: JacobianBridgeSAC) -> str:
    digest = hashlib.sha256()
    for name, parameter in sorted(policy.policy.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(parameter.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def prerequisite_failures(
    navigation: dict[str, object],
    oracle_headroom: dict[str, object],
) -> list[str]:
    failures = navigation_readiness_failures(navigation)
    if oracle_headroom.get("evaluable") is not True:
        failures.append("Oracle headroom gate is not statistically evaluable")
    if oracle_headroom.get("passed") is not True:
        failures.append("Oracle headroom gate did not pass")
    if oracle_headroom.get("status") != "PASS":
        failures.append("Oracle headroom gate status is not PASS")
    return failures


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train goal-conditioned Quantile-TD under one frozen 500k JSEB policy"
    )
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--navigation-evaluation-json", type=Path, required=True)
    parser.add_argument("--oracle-headroom-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--transition-budget", type=int, default=FORMAL_TD_TRANSITIONS)
    parser.add_argument(
        "--energy-eval-tasks",
        type=int,
        default=FORMAL_ENERGY_EVAL_TASKS,
    )
    parser.add_argument(
        "--energy-eval-freq-transitions",
        type=int,
        default=FORMAL_ENERGY_EVAL_FREQUENCY,
    )
    parser.add_argument("--td-collection-seed", type=int, default=100_001)
    parser.add_argument("--energy-eval-seed", type=int, default=120_001)
    parser.add_argument("--evaluation-num-envs", type=int, default=6)
    parser.add_argument("--log-freq-transitions", type=int, default=1_000)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.smoke:
        args.transition_budget = 2_000
        args.energy_eval_tasks = 10
        args.energy_eval_freq_transitions = 1_000
        args.evaluation_num_envs = 2
    elif (
        args.transition_budget != FORMAL_TD_TRANSITIONS
        or args.energy_eval_tasks != FORMAL_ENERGY_EVAL_TASKS
        or args.energy_eval_freq_transitions != FORMAL_ENERGY_EVAL_FREQUENCY
    ):
        parser.error(
            "formal Quantile-TD requires 500k transitions, 500 held-out tasks, "
            "and evaluation every 50k transitions"
        )
    if args.evaluation_num_envs <= 0:
        parser.error("evaluation worker count must be positive")
    seeds = [args.td_collection_seed, args.energy_eval_seed]
    if len(set(seeds)) != len(seeds):
        parser.error("TD collection and held-out evaluation seeds must differ")
    return args


def main(argv: list[str] | None = None) -> int:
    cli = parse_args(argv)
    artifact = cli.artifact.expanduser().resolve()
    checkpoint = cli.checkpoint.expanduser().resolve()
    navigation_path = cli.navigation_evaluation_json.expanduser().resolve()
    oracle_path = cli.oracle_headroom_json.expanduser().resolve()
    output = cli.output_dir.expanduser().resolve()
    for path in (artifact, checkpoint, navigation_path, oracle_path):
        if not path.exists():
            raise FileNotFoundError(path)
    if checkpoint.name != "checkpoint_transition_500000.zip" and not cli.smoke:
        raise ValueError("formal Quantile-TD requires checkpoint_transition_500000.zip")
    navigation = json.loads(navigation_path.read_text(encoding="utf-8"))
    oracle_headroom = json.loads(oracle_path.read_text(encoding="utf-8"))
    failures = prerequisite_failures(navigation, oracle_headroom)
    if cli.dry_run:
        print(
            json.dumps(
                {
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": file_sha256(checkpoint),
                    "prerequisite_failures": failures,
                    "would_run": not failures,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return int(bool(failures))
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Quantile-TD output is not fresh: {output}")
    output.mkdir(parents=True, exist_ok=True)
    git_status = subprocess.check_output(
        ["git", "status", "--porcelain"],
        text=True,
    ).splitlines()
    provenance = {
        "status": "RUNNING",
        "pid": os.getpid(),
        "source_artifact": str(artifact),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "navigation_evaluation": str(navigation_path),
        "oracle_headroom_gate": str(oracle_path),
        "prerequisite_failures": failures,
        "git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
        ).strip(),
        "git_status": git_status,
        "exact_command": [sys.executable, *sys.argv],
        "transition_budget": int(cli.transition_budget),
        "energy_eval_tasks": int(cli.energy_eval_tasks),
        "energy_eval_freq_transitions": int(cli.energy_eval_freq_transitions),
        "td_collection_seed": int(cli.td_collection_seed),
        "energy_eval_seed": int(cli.energy_eval_seed),
    }
    write_json(output / "RUNNING.json", provenance)
    if git_status and not cli.smoke:
        write_json(
            output / "FAILED.json",
            {"status": "FAILED_DIRTY_WORKTREE", "dirty_paths": git_status},
        )
        (output / "RUNNING.json").unlink(missing_ok=True)
        return 2
    if failures:
        write_json(
            output / "STOPPED_PREREQUISITES.json",
            {"status": "STOPPED_PREREQUISITES", "failures": failures},
        )
        (output / "RUNNING.json").unlink(missing_ok=True)
        return 3
    try:
        args, _ = reconstruct_environment_args(
            artifact,
            device=cli.device,
            seed=cli.td_collection_seed,
        )
        args.device = cli.device
        args.smoke = bool(cli.smoke)
        args.pilot = False
        args.td_collection_seed = int(cli.td_collection_seed)
        args.energy_eval_seed = int(cli.energy_eval_seed)
        args.energy_eval_freq_transitions = int(
            cli.energy_eval_freq_transitions
        )
        args.evaluation_num_envs = int(cli.evaluation_num_envs)
        args.log_freq_transitions = int(cli.log_freq_transitions)
        torch.set_num_threads(cli.torch_threads)
        os.environ.setdefault("OMP_NUM_THREADS", str(cli.torch_threads))
        os.environ.setdefault("MKL_NUM_THREADS", str(cli.torch_threads))
        policy = JacobianBridgeSAC.load(checkpoint, device=cli.device)
        if not cli.smoke and int(policy.num_timesteps) != FORMAL_NAVIGATION_TRANSITIONS:
            raise ValueError("checkpoint payload does not record 500k transitions")
        policy_hash_before = policy_parameter_hash(policy)
        td_state_dim = navigation_observation_dim(args)
        estimator = GoalConditionedQuantileTDEnergyEstimator(
            state_dim=td_state_dim,
            seed=cli.td_collection_seed,
            device=cli.device,
        )
        boundary = freeze_navigation_and_start_td(policy, estimator)
        tasks = generate_stratified_navigation_tasks(
            num_tasks=cli.energy_eval_tasks,
            seed=cli.energy_eval_seed,
        )
        save_navigation_tasks(
            output / "energy_eval_tasks.json",
            tasks,
            seed=cli.energy_eval_seed,
            role="heldout_quantile_td_evaluation_not_training",
        )
        write_json(
            output / "phase1_td" / "phase_boundary.json",
            {
                **boundary,
                "navigation_checkpoint": str(checkpoint),
                "navigation_checkpoint_sha256": file_sha256(checkpoint),
                "policy_hash_before": policy_hash_before,
                "td_state_dim": td_state_dim,
                "td_state_contract": (
                    "velocity_goal_lidar_executed_closed_loop_context"
                    if bool(args.lidar_enabled)
                    else "velocity_goal_context"
                ),
                "td_replay_contains_navigation_training_data": False,
            },
        )
        summary = run_td_pretraining(
            policy,
            estimator,
            args,
            tasks,
            transition_budget=cli.transition_budget,
            output=output,
        )
        policy_hash_after = policy_parameter_hash(policy)
        exact_budget = int(summary["actual_transitions"]) == int(
            cli.transition_budget
        )
        expected_evaluations = int(
            cli.transition_budget // cli.energy_eval_freq_transitions
        )
        passed = bool(
            exact_budget
            and policy_hash_before == policy_hash_after
            and int(summary["td_update_count"]) > 0
            and int(summary["td_replay_size"]) > 0
            and int(summary["heldout_evaluation_count"])
            == expected_evaluations
            and summary["td_readiness"] is not None
            and bool(summary["td_readiness"]["td_energy_ready"])
        )
        completed = {
            "status": "COMPLETED" if passed else "STOPPED_TD_NOT_READY",
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": file_sha256(checkpoint),
            "requested_transitions": int(cli.transition_budget),
            "actual_transitions": int(summary["actual_transitions"]),
            "exact_budget_match": exact_budget,
            "policy_hash_before": policy_hash_before,
            "policy_hash_after": policy_hash_after,
            "frozen_policy_unchanged": policy_hash_before == policy_hash_after,
            "td_update_count": int(summary["td_update_count"]),
            "td_replay_size": int(summary["td_replay_size"]),
            "td_state_dim": td_state_dim,
            "heldout_evaluation_count": int(
                summary["heldout_evaluation_count"]
            ),
            "td_readiness": summary["td_readiness"],
            "td_checkpoint": str(output / "phase1_td" / "td_energy_phase1.pt"),
        }
        sentinel = "COMPLETED.json" if passed else "STOPPED_TD_NOT_READY.json"
        write_json(output / sentinel, completed)
        (output / "RUNNING.json").unlink(missing_ok=True)
        return 0 if passed else 4
    except Exception as error:
        write_json(
            output / "FAILED.json",
            {
                "status": "FAILED",
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        (output / "RUNNING.json").unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
