from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import SAC

from envs.UAVEnergyDeliverySAC import UAVEnergyDeliverySACEnv
from envs.UAVEnergyDeliverySAC import GoalConditionedQuantileTDEnergyEstimator
from experiments.jacobian_energy_bridge import CalibratedCompactEnergyEstimator
from experiments.energy_mc.gate_b_prerequisites import (
    GateBPrerequisiteAudit,
    audit_gate_b_prerequisites,
)
from experiments.energy_mc.return_decision import (
    ReturnDecisionOutcome,
    clone_rollout_variance_audit,
    oracle_headroom_gate,
    pareto_frontier,
    wilson_interval,
)
from review_bundle.safety.energy.mc_regression import ModelBasedEnergyRolloutEstimator
from review_bundle.safety.switching import (
    DistanceEnergyReturnManager,
    FixedSOCThresholdReturnManager,
    QuantileEnergyReturnManager,
)
from scripts.train_uav_energy_delivery_sac import HeuristicGoalPolicy


_STAGE_B_WORKER_ARGS: argparse.Namespace | None = None
_STAGE_B_WORKER_POLICY: FrozenPolicy | None = None


class FrozenPolicy:
    def __init__(self, policy) -> None:
        self.policy = policy

    def predict(self, observation: np.ndarray, deterministic: bool = True):
        return self.policy.predict(observation, deterministic=deterministic)

    def action(self, observation: np.ndarray) -> np.ndarray:
        action, _ = self.predict(observation, deterministic=True)
        return np.asarray(action, dtype=np.float32)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage-B probability audit and Oracle return-decision headroom test"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--navigation-checkpoint", type=Path)
    parser.add_argument("--td-checkpoint", type=Path)
    parser.add_argument("--jseb-estimator-checkpoint", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--evaluation-seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
        help="paired evaluation seeds shared by every method/parameter point",
    )
    parser.add_argument("--battery-capacity", type=float)
    parser.add_argument("--navigation-evaluation-json", type=Path)
    parser.add_argument("--battery-calibration-json", type=Path)
    parser.add_argument("--battery-validation-json", type=Path)
    parser.add_argument(
        "--oracle-headroom-json",
        type=Path,
        help="passed Oracle headroom Gate inherited by post-Gate learned-method comparisons",
    )
    parser.add_argument("--cycles-per-point", type=int, default=20)
    parser.add_argument(
        "--evaluation-num-envs",
        type=int,
        default=1,
        help="independent evaluation seeds executed concurrently",
    )
    parser.add_argument("--minimum-cycles-for-gate", type=int, default=100)
    parser.add_argument("--oracle-stranding-ceiling", type=float, default=0.05)
    parser.add_argument(
        "--oracle-min-throughput-gain-fraction",
        type=float,
        default=0.05,
    )
    parser.add_argument("--maximum-policy-steps", type=int, default=400_000)
    parser.add_argument("--oracle-max-policy-steps", type=int, default=4000)
    parser.add_argument("--decision-interval-policy-steps", type=int, default=10)
    parser.add_argument("--p0-clone-rollouts", type=int, default=20)
    parser.add_argument("--energy-per-meter", type=float)
    parser.add_argument("--soc-thresholds", type=float, nargs="+", default=[0.10, 0.20, 0.30, 0.40])
    parser.add_argument("--reserve-fractions", type=float, nargs="+", default=[0.0, 0.05, 0.10, 0.15])
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=[
            "soc",
            "distance",
            "td_frozen",
            "td_online",
            "jseb_frozen",
            "oracle",
        ],
        default=["soc", "distance", "oracle"],
    )
    parser.add_argument("--length", type=float, default=4000.0)
    parser.add_argument("--width", type=float, default=4000.0)
    parser.add_argument("--height", type=float, default=400.0)
    parser.add_argument("--minimum-task-distance", type=float, default=100.0)
    parser.add_argument("--xy-sampling-margin", type=float, default=100.0)
    parser.add_argument("--task-z-min", type=float, default=20.0)
    parser.add_argument("--task-z-max", type=float, default=380.0)
    parser.add_argument("--lidar-enabled", action="store_true")
    parser.add_argument("--lidar-horizontal-sectors", type=int, default=128)
    parser.add_argument("--lidar-vertical-sectors", type=int, default=8)
    parser.add_argument("--lidar-range", type=float, default=100.0)
    parser.add_argument("--num-obstacles", type=int, default=0)
    parser.add_argument("--obstacle-radius-min", type=float, default=50.0)
    parser.add_argument("--obstacle-radius-max", type=float, default=120.0)
    parser.add_argument("--hocbf-enabled", action="store_true")
    parser.add_argument("--hocbf-top-k", type=int, default=16)
    parser.add_argument("--projection-geometry-enabled", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    if args.smoke:
        args.cycles_per_point = 2
        args.maximum_policy_steps = 20_000
        args.oracle_max_policy_steps = 1000
        args.decision_interval_policy_steps = 20
        args.p0_clone_rollouts = 4
        args.soc_thresholds = [0.20]
        args.reserve_fractions = [0.05]
        args.length = 400.0
        args.width = 400.0
        args.height = 80.0
        args.minimum_task_distance = 30.0
        args.xy_sampling_margin = 20.0
        args.task_z_min = 10.0
        args.task_z_max = 70.0
        args.battery_capacity = 60.0
        args.energy_per_meter = 0.01
        args.num_obstacles = 0
        args.lidar_enabled = False
        args.hocbf_enabled = False
        args.methods = ["soc", "distance", "oracle"]
        args.evaluation_seeds = [args.seed]
    if args.hocbf_enabled and not args.lidar_enabled:
        parser.error("--hocbf-enabled requires --lidar-enabled")
    if (
        args.cycles_per_point <= 0
        or args.evaluation_num_envs <= 0
        or args.minimum_cycles_for_gate <= 0
        or args.p0_clone_rollouts < 2
        or args.decision_interval_policy_steps <= 0
    ):
        parser.error("cycle count must be positive and P0 needs at least two clones")
    if len(set(args.evaluation_seeds)) != len(args.evaluation_seeds):
        parser.error("--evaluation-seeds must be unique")
    if not args.evaluation_seeds:
        parser.error("at least one evaluation seed is required")
    if not 0.0 <= args.oracle_stranding_ceiling <= 1.0:
        parser.error("--oracle-stranding-ceiling must lie in [0, 1]")
    if args.oracle_min_throughput_gain_fraction < 0.0:
        parser.error("--oracle-min-throughput-gain-fraction must be nonnegative")
    if (
        not args.smoke
        and args.cycles_per_point * len(args.evaluation_seeds)
        < args.minimum_cycles_for_gate
    ):
        parser.error(
            "formal Oracle headroom evaluation is underpowered: "
            "cycles-per-point * evaluation-seeds must reach minimum-cycles-for-gate"
        )
    if not args.smoke and args.oracle_headroom_json is None:
        if "oracle" not in args.methods or not {"soc", "distance"}.intersection(
            args.methods
        ):
            parser.error(
                "a formal headroom run must include Oracle and at least one SOC/distance "
                "heuristic, or inherit a passed --oracle-headroom-json"
            )
    return args


def environment_from_args(
    args: argparse.Namespace,
    *,
    reserve_fraction: float,
) -> UAVEnergyDeliverySACEnv:
    if args.battery_capacity is None:
        raise RuntimeError("battery capacity must come from a passed prerequisite audit")
    return UAVEnergyDeliverySACEnv(
        length=args.length,
        width=args.width,
        height=args.height,
        minimum_task_distance=args.minimum_task_distance,
        xy_sampling_margin=args.xy_sampling_margin,
        task_z_min=args.task_z_min,
        task_z_max=args.task_z_max,
        operational_energy_capacity=args.battery_capacity,
        energy_reserve_fraction=reserve_fraction,
        mission_decision_interval_policy_steps=args.decision_interval_policy_steps,
        lidar_enabled=args.lidar_enabled,
        lidar_horizontal_sectors=args.lidar_horizontal_sectors,
        lidar_vertical_sectors=args.lidar_vertical_sectors,
        lidar_max_range=args.lidar_range,
        num_obstacles=args.num_obstacles,
        obstacle_radius_min=args.obstacle_radius_min,
        obstacle_radius_max=args.obstacle_radius_max,
        cbf_enabled=args.hocbf_enabled,
        hocbf_top_k=args.hocbf_top_k,
        projection_geometry_enabled=args.projection_geometry_enabled,
    )


def load_policy(args: argparse.Namespace, probe_environment: UAVEnergyDeliverySACEnv) -> FrozenPolicy:
    if args.navigation_checkpoint is None:
        if not args.smoke:
            raise ValueError("formal Stage-B evaluation requires --navigation-checkpoint")
        return FrozenPolicy(HeuristicGoalPolicy())
    model = SAC.load(
        args.navigation_checkpoint,
        env=probe_environment,
        device=args.device,
        print_system_info=False,
    )
    return FrozenPolicy(model)


def configure_method(
    environment: UAVEnergyDeliverySACEnv,
    policy: FrozenPolicy,
    *,
    method: str,
    parameter: float,
    args: argparse.Namespace,
) -> None:
    environment.bind_navigation_policy(policy.action)
    if method == "soc":
        environment.bind_return_manager(FixedSOCThresholdReturnManager(parameter))
    elif method == "distance":
        environment.bind_return_manager(DistanceEnergyReturnManager(args.energy_per_meter))
    elif method in {"td_frozen", "td_online"}:
        if args.td_checkpoint is None:
            raise ValueError(f"{method} requires --td-checkpoint")
        estimator = GoalConditionedQuantileTDEnergyEstimator.load(
            args.td_checkpoint,
            device=args.device,
        )
        environment.bind_energy_learning(
            energy_estimator=estimator,
            goal_action_provider=policy.action,
            training_enabled=method == "td_online",
            return_manager=QuantileEnergyReturnManager(),
        )
    elif method == "jseb_frozen":
        if args.jseb_estimator_checkpoint is None:
            raise ValueError("jseb_frozen requires --jseb-estimator-checkpoint")
        estimator = CalibratedCompactEnergyEstimator.load(
            args.jseb_estimator_checkpoint,
            device=args.device,
        )
        environment.bind_energy_learning(
            energy_estimator=estimator,
            goal_action_provider=policy.action,
            training_enabled=False,
            return_manager=QuantileEnergyReturnManager(),
        )
    elif method == "oracle":
        oracle = ModelBasedEnergyRolloutEstimator(
            policy,
            max_policy_steps=args.oracle_max_policy_steps,
        )
        environment.bind_energy_learning(
            energy_estimator=oracle,
            goal_action_provider=policy.action,
            training_enabled=False,
            return_manager=QuantileEnergyReturnManager(),
        )
    else:
        raise ValueError(f"unknown return method: {method}")
    environment.enable_phase_two()


def evaluate_method(
    args: argparse.Namespace,
    policy: FrozenPolicy,
    *,
    method: str,
    parameter: float,
    evaluation_seed: int,
) -> tuple[ReturnDecisionOutcome, dict[str, object], list[dict[str, object]]]:
    reserve_fraction = 0.0 if method == "soc" else float(parameter)
    environment = environment_from_args(args, reserve_fraction=reserve_fraction)
    configure_method(environment, policy, method=method, parameter=parameter, args=args)
    observation, _ = environment.reset(seed=evaluation_seed)
    attempted_cycles = 0
    successful_returns = 0
    energy_exhaustions = 0
    emergency_guards = 0
    tasks_completed = 0
    simulated_seconds = 0.0
    unused_energy_fractions: list[float] = []
    policy_steps = 0
    cycle_tasks = 0
    cycle_start_seconds = 0.0
    cycle_start_policy_steps = 0
    cycle_records: list[dict[str, object]] = []
    while attempted_cycles < args.cycles_per_point:
        action = policy.action(observation)
        observation, _, terminated, truncated, info = environment.step(action)
        policy_steps += 1
        simulated_seconds += float(info["transition_dt"])
        task_completed = int(bool(info["task_completed_now"]))
        tasks_completed += task_completed
        cycle_tasks += task_completed
        if bool(info["charger_reached_now"]):
            cycle_record = info["battery_cycle_record"]
            if not isinstance(cycle_record, dict):
                environment.close()
                raise RuntimeError("charger arrival is missing its pre-recharge cycle record")
            unused_fraction = float(
                cycle_record["remaining_energy_fraction_at_charger"]
            )
            cycle_records.append(
                {
                    "method": method,
                    "parameter": float(parameter),
                    "evaluation_seed": int(evaluation_seed),
                    "cycle_index": int(attempted_cycles),
                    "end_reason": "charger_reached",
                    "return_success": True,
                    "energy_exhausted": False,
                    "tasks_completed": int(cycle_tasks),
                    "simulation_seconds": float(
                        simulated_seconds - cycle_start_seconds
                    ),
                    "policy_steps": int(policy_steps - cycle_start_policy_steps),
                    "unused_energy_fraction_at_charger": unused_fraction,
                    "battery_cycle_record": cycle_record,
                }
            )
            attempted_cycles += 1
            successful_returns += 1
            unused_energy_fractions.append(unused_fraction)
            cycle_tasks = 0
            cycle_start_seconds = simulated_seconds
            cycle_start_policy_steps = policy_steps
        elif terminated:
            if info["end_reason"] != "energy_exhausted":
                environment.close()
                raise RuntimeError(
                    f"unexpected terminal during Stage-B evaluation: {info['end_reason']}"
                )
            cycle_records.append(
                {
                    "method": method,
                    "parameter": float(parameter),
                    "evaluation_seed": int(evaluation_seed),
                    "cycle_index": int(attempted_cycles),
                    "end_reason": "energy_exhausted",
                    "return_success": False,
                    "energy_exhausted": True,
                    "tasks_completed": int(cycle_tasks),
                    "simulation_seconds": float(
                        simulated_seconds - cycle_start_seconds
                    ),
                    "policy_steps": int(policy_steps - cycle_start_policy_steps),
                    "unused_energy_fraction_at_charger": None,
                    "battery_cycle_record": info.get("battery_cycle_record"),
                }
            )
            attempted_cycles += 1
            energy_exhaustions += 1
            observation, _ = environment.reset(
                seed=evaluation_seed + attempted_cycles
            )
            cycle_tasks = 0
            cycle_start_seconds = simulated_seconds
            cycle_start_policy_steps = policy_steps
        elif truncated:
            emergency_guards += 1
            environment.close()
            raise RuntimeError(
                "Phase-2 emergency step guard fired during a battery cycle; "
                "the partial cycle is invalid and must not be counted as safe"
            )
        if policy_steps >= args.maximum_policy_steps:
            environment.close()
            raise RuntimeError(
                f"{method} parameter={parameter} exceeded maximum policy-step audit budget"
            )
    outcome = ReturnDecisionOutcome(
        method=method,
        parameter=float(parameter),
        stranding_rate=float(energy_exhaustions / attempted_cycles),
        tasks_per_simulated_hour=float(
            tasks_completed / max(simulated_seconds / 3600.0, np.finfo(np.float64).eps)
        ),
        tasks_per_battery_cycle=float(tasks_completed / attempted_cycles),
        return_success_rate=float(successful_returns / attempted_cycles),
        mean_unused_energy_fraction_at_charger=float(
            np.mean(unused_energy_fractions) if unused_energy_fractions else 0.0
        ),
    )
    audit = {
        **outcome.as_dict(),
        "evaluation_seed": int(evaluation_seed),
        "attempted_cycles": attempted_cycles,
        "successful_returns": successful_returns,
        "energy_exhaustions": energy_exhaustions,
        "emergency_guards": emergency_guards,
        "tasks_completed": tasks_completed,
        "simulated_seconds": simulated_seconds,
        "policy_steps": policy_steps,
        "unused_energy_fraction_sum": float(np.sum(unused_energy_fractions)),
        "hocbf_interventions": int(environment.safety_interventions),
        "obstacle_collision_count": int(environment.obstacle_collision_count),
    }
    if hasattr(environment.energy_estimator, "cache_diagnostics"):
        audit["oracle_cache_diagnostics"] = (
            environment.energy_estimator.cache_diagnostics()
        )
    environment.close()
    return outcome, audit, cycle_records


def _initialize_stage_b_worker(args: argparse.Namespace) -> None:
    global _STAGE_B_WORKER_ARGS, _STAGE_B_WORKER_POLICY
    _STAGE_B_WORKER_ARGS = args
    probe = environment_from_args(args, reserve_fraction=0.0)
    _STAGE_B_WORKER_POLICY = load_policy(args, probe)
    probe.close()


def _evaluate_stage_b_seed_job(
    job: tuple[str, float, int],
) -> tuple[dict[str, object], dict[str, object], list[dict[str, object]]]:
    if _STAGE_B_WORKER_ARGS is None or _STAGE_B_WORKER_POLICY is None:
        raise RuntimeError("Stage-B worker was not initialized")
    method, parameter, evaluation_seed = job
    outcome, audit, cycle_records = evaluate_method(
        _STAGE_B_WORKER_ARGS,
        _STAGE_B_WORKER_POLICY,
        method=method,
        parameter=parameter,
        evaluation_seed=evaluation_seed,
    )
    return outcome.as_dict(), audit, cycle_records


def evaluate_parameter_across_seeds(
    args: argparse.Namespace,
    policy: FrozenPolicy,
    *,
    method: str,
    parameter: float,
) -> list[tuple[ReturnDecisionOutcome, dict[str, object], list[dict[str, object]]]]:
    if args.evaluation_num_envs == 1 or len(args.evaluation_seeds) == 1:
        return [
            evaluate_method(
                args,
                policy,
                method=method,
                parameter=parameter,
                evaluation_seed=evaluation_seed,
            )
            for evaluation_seed in args.evaluation_seeds
        ]
    process_count = min(args.evaluation_num_envs, len(args.evaluation_seeds))
    jobs = [
        (method, float(parameter), int(evaluation_seed))
        for evaluation_seed in args.evaluation_seeds
    ]
    context = mp.get_context("spawn")
    with context.Pool(
        processes=process_count,
        initializer=_initialize_stage_b_worker,
        initargs=(args,),
    ) as pool:
        raw_results = pool.map(_evaluate_stage_b_seed_job, jobs)
    return [
        (ReturnDecisionOutcome(**outcome), audit, cycle_records)
        for outcome, audit, cycle_records in raw_results
    ]


def aggregate_seed_audits(
    *,
    method: str,
    parameter: float,
    seed_audits: list[dict[str, object]],
) -> tuple[ReturnDecisionOutcome, dict[str, object]]:
    if not seed_audits:
        raise ValueError("at least one seed audit is required")
    attempted_cycles = sum(int(item["attempted_cycles"]) for item in seed_audits)
    successful_returns = sum(int(item["successful_returns"]) for item in seed_audits)
    energy_exhaustions = sum(int(item["energy_exhaustions"]) for item in seed_audits)
    emergency_guards = sum(int(item["emergency_guards"]) for item in seed_audits)
    tasks_completed = sum(int(item["tasks_completed"]) for item in seed_audits)
    simulated_seconds = sum(float(item["simulated_seconds"]) for item in seed_audits)
    policy_steps = sum(int(item["policy_steps"]) for item in seed_audits)
    unused_energy_sum = sum(
        float(item["unused_energy_fraction_sum"]) for item in seed_audits
    )
    if attempted_cycles <= 0:
        raise RuntimeError("return-decision evaluation produced no battery cycles")
    if emergency_guards:
        raise RuntimeError("emergency guards invalidate aggregate headroom evidence")
    outcome = ReturnDecisionOutcome(
        method=method,
        parameter=float(parameter),
        stranding_rate=float(energy_exhaustions / attempted_cycles),
        tasks_per_simulated_hour=float(
            tasks_completed
            / max(simulated_seconds / 3600.0, np.finfo(np.float64).eps)
        ),
        tasks_per_battery_cycle=float(tasks_completed / attempted_cycles),
        return_success_rate=float(successful_returns / attempted_cycles),
        mean_unused_energy_fraction_at_charger=float(
            unused_energy_sum / successful_returns if successful_returns else 0.0
        ),
    )
    lower, upper = wilson_interval(energy_exhaustions, attempted_cycles)
    aggregate = {
        **outcome.as_dict(),
        "evaluation_seeds": [int(item["evaluation_seed"]) for item in seed_audits],
        "attempted_cycles": attempted_cycles,
        "successful_returns": successful_returns,
        "energy_exhaustions": energy_exhaustions,
        "emergency_guards": emergency_guards,
        "tasks_completed": tasks_completed,
        "simulated_seconds": simulated_seconds,
        "policy_steps": policy_steps,
        "stranding_rate_wilson95_lower": lower,
        "stranding_rate_wilson95_upper": upper,
        "hocbf_interventions": sum(
            int(item["hocbf_interventions"]) for item in seed_audits
        ),
        "obstacle_collision_count": sum(
            int(item["obstacle_collision_count"]) for item in seed_audits
        ),
    }
    oracle_diagnostics = [
        item["oracle_cache_diagnostics"]
        for item in seed_audits
        if "oracle_cache_diagnostics" in item
    ]
    if oracle_diagnostics:
        aggregate["oracle_cache_diagnostics"] = {
            "cache_mission_suffixes": all(
                bool(item["cache_mission_suffixes"])
                for item in oracle_diagnostics
            ),
            "rollout_request_count": sum(
                int(item["rollout_request_count"])
                for item in oracle_diagnostics
            ),
            "full_rollout_count": sum(
                int(item["full_rollout_count"])
                for item in oracle_diagnostics
            ),
            "mission_cache_hits": sum(
                int(item["mission_cache_hits"])
                for item in oracle_diagnostics
            ),
            "mission_cache_misses": sum(
                int(item["mission_cache_misses"])
                for item in oracle_diagnostics
            ),
        }
    return outcome, aggregate


def run_p0_audit(
    args: argparse.Namespace,
    policy: FrozenPolicy,
) -> dict[str, object]:
    environment = environment_from_args(args, reserve_fraction=0.0)
    environment.bind_navigation_policy(policy.action)
    environment.reset(seed=args.seed + 100_000)
    oracle = ModelBasedEnergyRolloutEstimator(
        policy,
        max_policy_steps=args.oracle_max_policy_steps,
    )
    goal = environment.current_task_point.copy()
    audit = clone_rollout_variance_audit(
        lambda disturbance_seed: oracle.estimate_context(environment, goal).prediction,
        num_rollouts=args.p0_clone_rollouts,
    )
    environment.close()
    payload = audit.as_dict()
    payload["fixed_information"] = [
        "state",
        "goal",
        "frozen_policy",
        "safety_operator",
        "obstacle_realization",
    ]
    payload["varied_information"] = ["future_disturbance_seed"]
    payload["current_environment_has_seeded_future_disturbance"] = False
    return payload


def parameter_grid(args: argparse.Namespace) -> Iterable[tuple[str, float]]:
    for method in args.methods:
        values = args.soc_thresholds if method == "soc" else args.reserve_fractions
        for value in values:
            yield method, float(value)


def write_results(
    output: Path,
    *,
    args: argparse.Namespace,
    p0: dict[str, object],
    prerequisite_audit: dict[str, object],
    outcomes: list[ReturnDecisionOutcome],
    aggregate_audits: list[dict[str, object]],
    seed_audits: list[dict[str, object]],
    cycle_records: list[dict[str, object]],
    inherited_oracle_gate: dict[str, object] | None,
) -> None:
    (output / "probability_semantics_audit.json").write_text(
        json.dumps(p0, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    frontier = pareto_frontier(outcomes)
    cycles_per_point = {
        (str(item["method"]), float(item["parameter"])): int(
            item["attempted_cycles"]
        )
        for item in aggregate_audits
    }
    stranding_upper_bounds = {
        (str(item["method"]), float(item["parameter"])): float(
            item["stranding_rate_wilson95_upper"]
        )
        for item in aggregate_audits
    }
    if inherited_oracle_gate is None:
        gate = oracle_headroom_gate(
            outcomes,
            cycles_per_point=cycles_per_point,
            stranding_upper_bounds=stranding_upper_bounds,
            minimum_cycles_per_point=args.minimum_cycles_for_gate,
            stranding_ceiling=args.oracle_stranding_ceiling,
            minimum_throughput_gain_fraction=(
                args.oracle_min_throughput_gain_fraction
            ),
        )
        gate_payload = {
            **gate.as_dict(),
            "stranding_statistic": "two_sided_wilson_95_upper_bound",
            "point_estimates_not_used_as_safety_gate": True,
            "gate_source": "computed_from_current_soc_distance_oracle_outcomes",
        }
        gate_passed = gate.passed
    else:
        gate_payload = {
            **inherited_oracle_gate,
            "gate_source": str(args.oracle_headroom_json),
            "inherited_for_post_gate_comparison": True,
        }
        gate_passed = True
    (output / "oracle_headroom_gate.json").write_text(
        json.dumps(gate_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = {
        "protocol": "stage_b_oracle_decision_headroom",
        "navigation_checkpoint": None
        if args.navigation_checkpoint is None
        else str(args.navigation_checkpoint),
        "battery_capacity": float(args.battery_capacity),
        "distance_baseline_energy_per_meter": float(args.energy_per_meter),
        "distance_baseline_source": (
            "smoke_constant"
            if args.smoke
            else "successful_calibration_total_energy_divided_by_total_path_length"
        ),
        "cycles_per_seed": int(args.cycles_per_point),
        "evaluation_seeds": list(args.evaluation_seeds),
        "evaluation_num_envs": int(args.evaluation_num_envs),
        "seed_parallelism_semantics": "independent_seed_processes_same_config_and_budget",
        "minimum_cycles_per_point": int(args.minimum_cycles_for_gate),
        "actual_cycles_per_point": {
            f"{method}|{parameter}": count
            for (method, parameter), count in cycles_per_point.items()
        },
        "training_env_transitions": 0,
        "evaluation_env_transitions": sum(
            int(item["policy_steps"]) for item in aggregate_audits
        ),
        "decision_interval_policy_steps": int(
            args.decision_interval_policy_steps
        ),
        "decision_interval_applied_equally_to_all_methods": True,
        "probability_semantics": p0,
        "prerequisite_audit": prerequisite_audit,
        "outcomes": [item.as_dict() for item in outcomes],
        "aggregate_audits": aggregate_audits,
        "pareto_frontier": [item.as_dict() for item in frontier],
        "oracle_headroom_gate": gate_payload,
        "formal_claim_status": (
            "SMOKE_ONLY_NOT_FORMAL_EVIDENCE"
            if args.smoke
            else (
                "POST_ORACLE_HEADROOM_DECISION_COMPARISON"
                if inherited_oracle_gate is not None
                else (
                "ORACLE_HEADROOM_GATE_PASS"
                if gate_passed is True
                else "STOP_AFTER_ORACLE_HEADROOM_GATE"
                )
            )
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_csv(output / "outcomes.csv", aggregate_audits)
    write_csv(output / "seed_outcomes.csv", seed_audits)
    write_csv(
        output / "battery_cycles.csv",
        [
            {key: value for key, value in item.items() if key != "battery_cycle_record"}
            for item in cycle_records
        ],
    )
    with (output / "battery_cycles.jsonl").open("w", encoding="utf-8") as handle:
        for item in cycle_records:
            handle.write(json.dumps(item, sort_keys=True) + "\n")
    figure, axis = plt.subplots(figsize=(7.0, 5.0))
    for method in sorted({item.method for item in outcomes}):
        selected = sorted(
            (item for item in outcomes if item.method == method),
            key=lambda item: item.stranding_rate,
        )
        axis.plot(
            [item.stranding_rate for item in selected],
            [item.tasks_per_simulated_hour for item in selected],
            marker="o",
            label=method,
        )
    axis.set_xlabel("Stranding rate (lower is better)")
    axis.set_ylabel("Tasks per simulated hour (higher is better)")
    axis.set_title("Stage-B return-decision headroom")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "stranding_throughput_pareto.png", dpi=180)
    plt.close(figure)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty audit table: {path}")
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def jsonable_args(args: argparse.Namespace) -> dict[str, object]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def git_provenance() -> dict[str, object]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return {
        "commit_sha": commit,
        "worktree_clean": not status,
        "dirty_path_count": len(status),
    }


def load_prerequisite_audit(args: argparse.Namespace) -> GateBPrerequisiteAudit:
    if args.smoke:
        return GateBPrerequisiteAudit(
            passed=True,
            status="SMOKE_BYPASS",
            failures=(),
            calibrated_battery_capacity=float(args.battery_capacity),
            navigation_success_rate=None,
            calibration_success_rate=None,
            observed_endurance_seconds=None,
        )
    required = {
        "--navigation-evaluation-json": args.navigation_evaluation_json,
        "--battery-calibration-json": args.battery_calibration_json,
        "--battery-validation-json": args.battery_validation_json,
    }
    missing = [name for name, path in required.items() if path is None]
    if missing:
        raise ValueError(
            "formal Stage-B evaluation requires audited prerequisites: "
            + ", ".join(missing)
        )
    payloads: list[dict[str, object]] = []
    for path in required.values():
        assert path is not None
        if not path.is_file():
            raise FileNotFoundError(path)
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise TypeError(f"prerequisite JSON must contain an object: {path}")
        payloads.append(value)
    audit = audit_gate_b_prerequisites(*payloads)
    calibration = payloads[1]
    calibrated_capacity = audit.calibrated_battery_capacity
    if audit.passed and calibrated_capacity is None:
        raise RuntimeError("passed prerequisite audit did not provide battery capacity")
    if (
        audit.passed
        and args.battery_capacity is not None
        and not np.isclose(
        args.battery_capacity,
        calibrated_capacity,
        rtol=1e-9,
        atol=1e-9,
        )
    ):
        raise ValueError(
            "manual --battery-capacity differs from the audited calibrated capacity"
        )
    if audit.passed:
        args.battery_capacity = calibrated_capacity
        empirical_energy_per_meter = derive_successful_energy_per_meter(calibration)
        if args.energy_per_meter is not None and not np.isclose(
            args.energy_per_meter,
            empirical_energy_per_meter,
            rtol=1e-9,
            atol=1e-12,
        ):
            raise ValueError(
                "manual --energy-per-meter differs from the audited calibration estimate"
            )
        args.energy_per_meter = empirical_energy_per_meter
    return audit


def load_passed_oracle_headroom_gate(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("Oracle headroom Gate JSON must contain an object")
    if (
        payload.get("status") != "PASS"
        or payload.get("evaluable") is not True
        or payload.get("passed") is not True
    ):
        raise RuntimeError("post-Gate comparison requires an evaluable PASS Gate")
    return payload


def derive_successful_energy_per_meter(
    calibration: dict[str, object],
) -> float:
    tasks = calibration.get("tasks")
    if not isinstance(tasks, list):
        raise ValueError("calibration tasks are required for the distance baseline")
    successful = [
        item
        for item in tasks
        if isinstance(item, dict) and item.get("success") is True
    ]
    if not successful:
        raise ValueError("calibration contains no successful distance-energy samples")
    total_energy = sum(float(item["total_realized_energy"]) for item in successful)
    total_path_length = sum(float(item["actual_path_length"]) for item in successful)
    value = total_energy / total_path_length
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("empirical calibration energy per meter must be positive")
    return float(value)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(f"output directory already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    launch = {
        "arguments": jsonable_args(args),
        "command": [sys.executable, *sys.argv],
        "git": git_provenance(),
    }
    (args.output_dir / "launch.json").write_text(
        json.dumps(launch, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    try:
        prerequisite = load_prerequisite_audit(args)
        (args.output_dir / "prerequisite_audit.json").write_text(
            json.dumps(prerequisite.as_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if not prerequisite.passed:
            raise RuntimeError(
                "formal Gate-B prerequisites failed: "
                + "; ".join(prerequisite.failures)
            )
        inherited_oracle_gate = (
            None
            if args.oracle_headroom_json is None
            else load_passed_oracle_headroom_gate(args.oracle_headroom_json)
        )
        probe = environment_from_args(args, reserve_fraction=0.0)
        policy = load_policy(args, probe)
        probe.close()
        p0 = run_p0_audit(args, policy)
        outcomes: list[ReturnDecisionOutcome] = []
        aggregate_audits: list[dict[str, object]] = []
        all_seed_audits: list[dict[str, object]] = []
        all_cycle_records: list[dict[str, object]] = []
        for method, parameter in parameter_grid(args):
            point_seed_audits: list[dict[str, object]] = []
            point_results = evaluate_parameter_across_seeds(
                args,
                policy,
                method=method,
                parameter=parameter,
            )
            for _, seed_audit, cycle_records in point_results:
                point_seed_audits.append(seed_audit)
                all_seed_audits.append(seed_audit)
                all_cycle_records.extend(cycle_records)
                print(json.dumps(seed_audit, sort_keys=True), flush=True)
            outcome, aggregate = aggregate_seed_audits(
                method=method,
                parameter=parameter,
                seed_audits=point_seed_audits,
            )
            outcomes.append(outcome)
            aggregate_audits.append(aggregate)
            print(json.dumps(aggregate, sort_keys=True), flush=True)
        write_results(
            args.output_dir,
            args=args,
            p0=p0,
            prerequisite_audit=prerequisite.as_dict(),
            outcomes=outcomes,
            aggregate_audits=aggregate_audits,
            seed_audits=all_seed_audits,
            cycle_records=all_cycle_records,
            inherited_oracle_gate=inherited_oracle_gate,
        )
    except Exception as error:
        (args.output_dir / "FAILED.json").write_text(
            json.dumps(
                {
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "protocol": "stage_b_oracle_decision_headroom",
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        raise


if __name__ == "__main__":
    main()
