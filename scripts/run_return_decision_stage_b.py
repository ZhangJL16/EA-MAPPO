from __future__ import annotations

import os

# Stage-B is dominated by independent simulator/HOCBF jobs.  Numerical-library
# thread pools must be capped before Matplotlib/NumPy/Torch are imported; setting
# these variables in the ProcessPool initializer is too late under ``spawn``
# because the child imports this module before invoking the initializer.
for _thread_environment in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_thread_environment] = "1"

import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import multiprocessing as mp
import subprocess
import sys
import traceback
from dataclasses import asdict, is_dataclass
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch
from stable_baselines3 import SAC

from envs.UAVEnergyDeliverySAC import UAVEnergyDeliverySACEnv
from envs.UAVEnergyDeliverySAC import GoalConditionedQuantileTDEnergyEstimator
from envs.UAVEnergyDeliverySAC import SACTrainingPhase
from experiments.jacobian_energy_bridge import CalibratedCompactEnergyEstimator
from experiments.energy_mc.gate_b_prerequisites import (
    GateBPrerequisiteAudit,
    audit_gate_b_prerequisites,
    navigation_artifact_view,
)
from experiments.energy_mc.return_decision import (
    ReturnDecisionOutcome,
    audit_oracle_shadow_coupling,
    clone_rollout_variance_audit,
    oracle_headroom_gate,
    paired_frontier_throughput_interval,
    pareto_frontier,
    probability_semantics_gate,
    simultaneous_stranding_upper_bounds,
    stranding_certification_power_audit,
    wilson_interval,
)
from experiments.energy_mc.oracle_pareto_records import (
    empty_oracle_pareto_bundle,
    validate_oracle_pareto_bundle,
)
from review_bundle.safety.energy.mc_regression import ModelBasedEnergyRolloutEstimator
from review_bundle.envs.navigation.telemetry_cost import TelemetryCostConfig
from review_bundle.safety.switching import (
    DistanceEnergyReturnManager,
    FixedSOCThresholdReturnManager,
    QuantileEnergyReturnManager,
)
from scripts.evaluate_jseb_checkpoints import reconstruct_environment_args
from scripts.train_uav_energy_delivery_sac import (
    HeuristicGoalPolicy,
    environment_kwargs_from_args,
)


_STAGE_B_WORKER_ARGS: argparse.Namespace | None = None
_STAGE_B_WORKER_POLICY: FrozenPolicy | None = None
FORMAL_EVALUATION_SEEDS = list(range(110_001, 110_321))
FAMILYWISE_STRANDING_STATISTIC = (
    "familywise_one_sided_clopper_pearson_upper_bound_with_bonferroni"
)
SIMULTANEOUS_THROUGHPUT_STATISTIC = (
    "paired_schedule_simultaneous_max_t_rate_band_over_full_candidate_family"
)
SIMULTANEOUS_THROUGHPUT_INFERENCE = (
    "paired_schedule_studentized_max_t_rate_band_over_full_candidate_family"
)


def _telemetry_cost_config_from_kwargs(
    environment_kwargs: dict[str, object],
) -> TelemetryCostConfig:
    raw = environment_kwargs.get("telemetry_cost_config")
    if raw is None:
        return TelemetryCostConfig()
    if isinstance(raw, TelemetryCostConfig):
        return raw
    if isinstance(raw, dict):
        return TelemetryCostConfig(**raw)
    raise TypeError("telemetry_cost_config must be a TelemetryCostConfig or mapping")


def stage_b_episode_guard_contract(
    args: argparse.Namespace,
    environment_kwargs: dict[str, object] | None = None,
) -> dict[str, object]:
    """Certify a fail-closed guard that cannot pre-empt energy exhaustion."""

    if args.battery_capacity is None:
        raise RuntimeError("battery capacity is required to certify the episode guard")
    kwargs = {} if environment_kwargs is None else dict(environment_kwargs)
    policy_dt = float(kwargs.get("policy_dt", 0.20))
    physics_dt = float(kwargs.get("physics_dt", 0.05))
    substep_ratio = policy_dt / physics_dt
    if not np.isclose(substep_ratio, round(substep_ratio)):
        raise ValueError("policy_dt must be an integer multiple of physics_dt")
    physics_substeps = int(round(substep_ratio))
    config = _telemetry_cost_config_from_kwargs(kwargs)
    minimum_energy_per_policy_step = float(
        config.flight_energy_multiplier
        * (
            policy_dt
            * (config.base_power + config.compute_power + config.communication_power)
            + physics_substeps * config.simulation_error
        )
    )
    if not np.isfinite(minimum_energy_per_policy_step) or minimum_energy_per_policy_step <= 0.0:
        raise ValueError(
            "Stage-B needs a strictly positive certified per-step energy lower bound"
        )
    maximum_steps_to_exhaustion_per_cycle = int(
        math.ceil(float(args.battery_capacity) / minimum_energy_per_policy_step)
    )
    source_limit = int(kwargs.get("phase2_episode_limit", 20_000))
    certified_limit = int(
        args.cycles_per_point * maximum_steps_to_exhaustion_per_cycle + 1
    )
    return {
        "source_phase2_episode_limit": source_limit,
        "effective_phase2_episode_limit": max(source_limit, certified_limit),
        "minimum_energy_per_policy_step": minimum_energy_per_policy_step,
        "maximum_steps_to_exhaustion_per_cycle": (
            maximum_steps_to_exhaustion_per_cycle
        ),
        "cycles_per_point": int(args.cycles_per_point),
        "certified_limit_formula": (
            "max(source_limit, cycles_per_point*"
            "ceil(capacity/minimum_energy_per_policy_step)+1)"
        ),
        "guard_cannot_preempt_energy_exhaustion": True,
    }


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
    parser.add_argument("--navigation-artifact", type=Path)
    parser.add_argument("--navigation-checkpoint", type=Path)
    parser.add_argument("--td-checkpoint", type=Path)
    parser.add_argument("--jseb-estimator-checkpoint", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--evaluation-seeds",
        type=int,
        nargs="+",
        default=FORMAL_EVALUATION_SEEDS,
        help=(
            "independent battery-cycle seeds shared by every method/parameter "
            "point"
        ),
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
    parser.add_argument(
        "--cycles-per-point",
        type=int,
        default=1,
        help=(
            "cycles executed per evaluation seed; formal evidence requires one "
            "cycle so safety inference uses independent seeded units"
        ),
    )
    parser.add_argument(
        "--evaluation-num-envs",
        type=int,
        default=12,
        help="independent evaluation seeds executed concurrently",
    )
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=1,
        help="Torch CPU threads per Stage-B worker",
    )
    parser.add_argument(
        "--formal-seed-retries",
        type=int,
        default=2,
        help="whole-pool retries after a worker failure; completed seed files resume",
    )
    parser.add_argument("--minimum-cycles-for-gate", type=int, default=320)
    parser.add_argument("--oracle-stranding-ceiling", type=float, default=0.05)
    parser.add_argument(
        "--oracle-familywise-safety-confidence-level",
        type=float,
        default=0.95,
    )
    parser.add_argument(
        "--oracle-design-stranding-rate",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--oracle-min-joint-safety-certification-power",
        type=float,
        default=0.90,
    )
    parser.add_argument(
        "--oracle-min-throughput-gain-fraction",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--oracle-throughput-confidence-level",
        type=float,
        default=0.95,
    )
    parser.add_argument(
        "--oracle-throughput-bootstrap-replicates",
        type=int,
        default=10_000,
    )
    parser.add_argument(
        "--oracle-throughput-bootstrap-seed",
        type=int,
        default=20_260_830,
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
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume formal seed-family files from an existing output directory",
    )
    args = parser.parse_args(argv)
    if args.smoke:
        args.cycles_per_point = 1
        args.maximum_policy_steps = 20_000
        args.oracle_max_policy_steps = 1000
        args.decision_interval_policy_steps = 20
        args.p0_clone_rollouts = 4
        args.soc_thresholds = [0.20]
        args.reserve_fractions = [0.0, 0.05]
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
        or args.torch_threads <= 0
        or args.formal_seed_retries < 0
    ):
        parser.error("cycle count must be positive and P0 needs at least two clones")
    if len(set(args.evaluation_seeds)) != len(args.evaluation_seeds):
        parser.error("--evaluation-seeds must be unique")
    if not args.evaluation_seeds:
        parser.error("at least one evaluation seed is required")
    if not 0.0 <= args.oracle_stranding_ceiling <= 1.0:
        parser.error("--oracle-stranding-ceiling must lie in [0, 1]")
    if not 0.5 < args.oracle_familywise_safety_confidence_level < 1.0:
        parser.error(
            "--oracle-familywise-safety-confidence-level must lie in (0.5, 1)"
        )
    if not 0.0 <= args.oracle_design_stranding_rate < args.oracle_stranding_ceiling:
        parser.error(
            "--oracle-design-stranding-rate must lie in [0, stranding ceiling)"
        )
    if not 0.5 < args.oracle_min_joint_safety_certification_power < 1.0:
        parser.error(
            "--oracle-min-joint-safety-certification-power must lie in (0.5, 1)"
        )
    if args.oracle_min_throughput_gain_fraction < 0.0:
        parser.error("--oracle-min-throughput-gain-fraction must be nonnegative")
    if not 0.5 < args.oracle_throughput_confidence_level < 1.0:
        parser.error("--oracle-throughput-confidence-level must lie in (0.5, 1)")
    if args.oracle_throughput_bootstrap_replicates < 1000:
        parser.error("--oracle-throughput-bootstrap-replicates must be at least 1000")
    if (
        not args.smoke
        and args.cycles_per_point * len(args.evaluation_seeds)
        < args.minimum_cycles_for_gate
    ):
        parser.error(
            "formal Oracle headroom evaluation is underpowered: "
            "cycles-per-point * evaluation-seeds must reach minimum-cycles-for-gate"
        )
    if not args.smoke and args.cycles_per_point != 1:
        parser.error(
            "formal Stage-B evidence requires exactly one battery cycle per "
            "independent evaluation seed"
        )
    if not args.smoke and args.oracle_headroom_json is None:
        if "oracle" not in args.methods or not {"soc", "distance"}.intersection(
            args.methods
        ):
            parser.error(
                "a formal headroom run must include Oracle and at least one SOC/distance "
                "heuristic, or inherit a passed --oracle-headroom-json"
            )
        safety_power = stranding_certification_power_audit(
            num_candidate_points=len(list(parameter_grid(args))),
            planned_independent_cycles_per_point=(
                args.cycles_per_point * len(args.evaluation_seeds)
            ),
            stranding_ceiling=args.oracle_stranding_ceiling,
            design_stranding_rate=args.oracle_design_stranding_rate,
            confidence_level=args.oracle_familywise_safety_confidence_level,
            target_joint_certification_power=(
                args.oracle_min_joint_safety_certification_power
            ),
        )
        if not safety_power.passed:
            parser.error(
                "formal Oracle safety design is underpowered: "
                f"power={safety_power.planned_certification_power:.6f}, "
                "joint lower="
                f"{safety_power.planned_joint_certification_power_lower_bound:.6f}, "
                f"joint target={safety_power.target_joint_certification_power:.6f}, "
                "minimum cycles per point="
                f"{safety_power.minimum_cycles_for_target_power}"
            )
    grid = list(parameter_grid(args))
    if len(grid) != len(set(grid)):
        parser.error("method/parameter candidate points must be unique")
    if not args.smoke and args.oracle_headroom_json is None:
        oracle_reserves = {
            round(parameter, 12)
            for method, parameter in grid
            if method == "oracle"
        }
        required_oracle_reserves = {
            round(0.0 if method == "soc" else parameter, 12)
            for method, parameter in grid
            if method != "oracle"
        }
        missing_oracle_reserves = sorted(
            required_oracle_reserves - oracle_reserves
        )
        if missing_oracle_reserves:
            parser.error(
                "formal same-schedule coupling requires an Oracle point at every "
                "non-Oracle reserve; missing reserves="
                f"{missing_oracle_reserves}"
            )
    return args


def environment_from_args(
    args: argparse.Namespace,
    *,
    reserve_fraction: float,
) -> UAVEnergyDeliverySACEnv:
    if args.battery_capacity is None:
        raise RuntimeError("battery capacity must come from a passed prerequisite audit")
    navigation_kwargs = getattr(args, "navigation_environment_kwargs", None)
    if navigation_kwargs is not None:
        kwargs = dict(navigation_kwargs)
        guard_contract = stage_b_episode_guard_contract(args, kwargs)
        kwargs.update(
            {
                "phase": SACTrainingPhase.ENERGY_MANAGED,
                "operational_energy_capacity": args.battery_capacity,
                "energy_reserve_fraction": reserve_fraction,
                "mission_decision_interval_policy_steps": (
                    args.decision_interval_policy_steps
                ),
                "phase2_episode_limit": guard_contract[
                    "effective_phase2_episode_limit"
                ],
            }
        )
        return UAVEnergyDeliverySACEnv(**kwargs)
    fallback_kwargs = {
        "policy_dt": 0.20,
        "physics_dt": 0.05,
        "phase2_episode_limit": 20_000,
    }
    guard_contract = stage_b_episode_guard_contract(args, fallback_kwargs)
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
        phase2_episode_limit=guard_contract["effective_phase2_episode_limit"],
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
    oracle_bundle_cache: dict[tuple[bytes, bytes, bytes, bytes], tuple] | None = None,
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
            shared_bundle_cache=oracle_bundle_cache,
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
    environment.enable_continuous_task_workload()


def evaluate_method(
    args: argparse.Namespace,
    policy: FrozenPolicy,
    *,
    method: str,
    parameter: float,
    evaluation_seed: int,
    oracle_bundle_cache: dict[tuple[bytes, bytes, bytes, bytes], tuple] | None = None,
) -> tuple[ReturnDecisionOutcome, dict[str, object], list[dict[str, object]]]:
    reserve_fraction = 0.0 if method == "soc" else float(parameter)
    environment = environment_from_args(args, reserve_fraction=reserve_fraction)
    configure_method(
        environment,
        policy,
        method=method,
        parameter=parameter,
        args=args,
        oracle_bundle_cache=oracle_bundle_cache,
    )
    environment.bind_keyed_task_schedule(evaluation_seed)
    oracle_shadow = (
        environment.energy_estimator
        if method == "oracle"
        else ModelBasedEnergyRolloutEstimator(
            policy,
            max_policy_steps=args.oracle_max_policy_steps,
            shared_bundle_cache=oracle_bundle_cache,
        )
    )
    environment.bind_oracle_shadow(
        oracle_shadow,
        stop_after_first_disagreement=(method != "oracle"),
    )
    observation, _ = environment.reset(seed=evaluation_seed)
    attempted_cycles = 0
    successful_returns = 0
    energy_exhaustions = 0
    emergency_guards = 0
    task_workload_rollovers = 0
    tasks_completed = 0
    simulated_seconds = 0.0
    unused_energy_fractions: list[float] = []
    policy_steps = 0
    cycle_tasks = 0
    cycle_start_seconds = 0.0
    cycle_start_policy_steps = 0
    cycle_records: list[dict[str, object]] = []
    decision_records: list[dict[str, object]] = []
    decision_cursor = 0
    while attempted_cycles < args.cycles_per_point:
        action = policy.action(observation)
        try:
            observation, _, terminated, truncated, info = environment.step(action)
        except Exception as error:
            error.stage_b_context = {
                "method": str(method),
                "parameter": float(parameter),
                "reserve_fraction": float(reserve_fraction),
                "evaluation_seed": int(evaluation_seed),
                "cycle_index": int(attempted_cycles),
                "policy_steps_before_failure": int(policy_steps),
                "simulated_seconds_before_failure": float(simulated_seconds),
                "steps_in_current_task": int(environment.steps_in_current_task),
                "position": np.asarray(
                    environment.agent.pos, dtype=np.float64
                ).tolist(),
                "velocity": np.asarray(
                    environment.agent.vel, dtype=np.float64
                ).tolist(),
                "task_goal": np.asarray(
                    environment.current_task_point, dtype=np.float64
                ).tolist(),
                "charger_position": np.asarray(
                    environment.charger_position, dtype=np.float64
                ).tolist(),
                "remaining_energy": float(environment.agent.energy),
            }
            environment.close()
            raise
        policy_steps += 1
        simulated_seconds += float(info["transition_dt"])
        task_completed = int(bool(info["task_completed_now"]))
        tasks_completed += task_completed
        cycle_tasks += task_completed
        task_workload_rollovers += int(
            bool(info.get("continuous_task_workload_rollover", False))
        )
        for raw_record in environment.return_decision_records[decision_cursor:]:
            record = dict(raw_record)
            for vector_field in ("position", "velocity", "task_goal"):
                record[vector_field] = np.asarray(
                    record[vector_field],
                    dtype=np.float64,
                ).tolist()
            schedule_seed = int(record["task_schedule_seed"])
            schedule_id = (
                f"keyed-task-v1:{schedule_seed}:cycle:{attempted_cycles}"
            )
            record.update(
                {
                    "record_id": (
                        f"{method}|{parameter}|{evaluation_seed}|"
                        f"{attempted_cycles}|{record['decision_index']}"
                    ),
                    "method": method,
                    "parameter": float(parameter),
                    "reserve_fraction": float(reserve_fraction),
                    "evaluation_seed": int(evaluation_seed),
                    "paired_schedule_id": schedule_id,
                    "cycle_index": int(attempted_cycles),
                }
            )
            decision_records.append(record)
        decision_cursor = len(environment.return_decision_records)
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
                    "record_id": (
                        f"{method}|{parameter}|{evaluation_seed}|{attempted_cycles}"
                    ),
                    "method": method,
                    "parameter": float(parameter),
                    "evaluation_seed": int(evaluation_seed),
                    "paired_schedule_id": (
                        f"keyed-task-v1:{cycle_record['task_schedule_seed']}:"
                        f"cycle:{attempted_cycles}"
                    ),
                    "cycle_index": int(attempted_cycles),
                    "end_reason": "charger_reached",
                    "return_success": True,
                    "energy_exhausted": False,
                    "tasks_completed": int(cycle_tasks),
                    "simulation_seconds": float(
                        simulated_seconds - cycle_start_seconds
                    ),
                    "simulated_seconds": float(
                        simulated_seconds - cycle_start_seconds
                    ),
                    "policy_steps": int(policy_steps - cycle_start_policy_steps),
                    "unused_energy_fraction_at_charger": unused_fraction,
                    "stranded": False,
                    "tasks_per_simulated_hour": float(
                        cycle_tasks
                        / max(
                            (simulated_seconds - cycle_start_seconds) / 3600.0,
                            np.finfo(np.float64).eps,
                        )
                    ),
                    "first_disagreement_step": cycle_record.get(
                        "first_disagreement_step"
                    ),
                    "first_disagreement_direction": cycle_record.get(
                        "first_disagreement_direction"
                    ),
                    "reserve_fraction": reserve_fraction,
                    "paired_oracle_return_success": None,
                    "paired_oracle_stranded": None,
                    "paired_oracle_tasks_completed": None,
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
                    "record_id": (
                        f"{method}|{parameter}|{evaluation_seed}|{attempted_cycles}"
                    ),
                    "method": method,
                    "parameter": float(parameter),
                    "evaluation_seed": int(evaluation_seed),
                    "paired_schedule_id": (
                        f"keyed-task-v1:{info['battery_cycle_record']['task_schedule_seed']}:"
                        f"cycle:{attempted_cycles}"
                    ),
                    "cycle_index": int(attempted_cycles),
                    "end_reason": "energy_exhausted",
                    "return_success": False,
                    "energy_exhausted": True,
                    "tasks_completed": int(cycle_tasks),
                    "simulation_seconds": float(
                        simulated_seconds - cycle_start_seconds
                    ),
                    "simulated_seconds": float(
                        simulated_seconds - cycle_start_seconds
                    ),
                    "policy_steps": int(policy_steps - cycle_start_policy_steps),
                    "unused_energy_fraction_at_charger": None,
                    "stranded": True,
                    "tasks_per_simulated_hour": float(
                        cycle_tasks
                        / max(
                            (simulated_seconds - cycle_start_seconds) / 3600.0,
                            np.finfo(np.float64).eps,
                        )
                    ),
                    "first_disagreement_step": info["battery_cycle_record"].get(
                        "first_disagreement_step"
                    ),
                    "first_disagreement_direction": info[
                        "battery_cycle_record"
                    ].get("first_disagreement_direction"),
                    "reserve_fraction": reserve_fraction,
                    "paired_oracle_return_success": None,
                    "paired_oracle_stranded": None,
                    "paired_oracle_tasks_completed": None,
                    "battery_cycle_record": info.get("battery_cycle_record"),
                }
            )
            attempted_cycles += 1
            energy_exhaustions += 1
            environment.bind_keyed_task_schedule(
                evaluation_seed + attempted_cycles
            )
            observation, _ = environment.reset(
                seed=evaluation_seed + attempted_cycles
            )
            decision_cursor = 0
            cycle_tasks = 0
            cycle_start_seconds = simulated_seconds
            cycle_start_policy_steps = policy_steps
        elif truncated:
            end_reason = str(info.get("end_reason", "missing_end_reason"))
            emergency_guards += int(end_reason == "episode_emergency_step_guard")
            error = RuntimeError(
                "Stage-B encountered an invalid truncation; "
                f"end_reason={end_reason}, policy_steps={policy_steps}, "
                f"task_workload_rollovers={task_workload_rollovers}"
            )
            error.stage_b_context = {
                "method": str(method),
                "parameter": float(parameter),
                "reserve_fraction": float(reserve_fraction),
                "evaluation_seed": int(evaluation_seed),
                "cycle_index": int(attempted_cycles),
                "policy_steps_before_failure": int(policy_steps),
                "simulated_seconds_before_failure": float(simulated_seconds),
                "end_reason": end_reason,
                "task_workload_rollovers": int(task_workload_rollovers),
                "remaining_energy": float(environment.agent.energy),
                "phase2_episode_limit": int(environment.phase2_episode_limit),
            }
            environment.close()
            raise error
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
        "independent_cycles": (
            attempted_cycles if args.cycles_per_point == 1 else 0
        ),
        "successful_returns": successful_returns,
        "energy_exhaustions": energy_exhaustions,
        "stranding_count": energy_exhaustions,
        "emergency_guards": emergency_guards,
        "task_workload_rollovers": task_workload_rollovers,
        "tasks_completed": tasks_completed,
        "simulated_seconds": simulated_seconds,
        "policy_steps": policy_steps,
        "unused_energy_fraction_sum": float(np.sum(unused_energy_fractions)),
        "hocbf_interventions": int(environment.safety_interventions),
        "obstacle_collision_count": int(environment.obstacle_collision_count),
        "decision_events": decision_records,
        "decision_event_count": len(decision_records),
        "exact_return_now_deadline_infeasible_event_count": sum(
            int(item.get("exact_return_now_deadline_feasible") is False)
            for item in decision_records
        ),
        "exact_task_then_return_deadline_infeasible_event_count": sum(
            int(item.get("exact_task_then_return_deadline_feasible") is False)
            for item in decision_records
        ),
        "estimated_return_now_deadline_infeasible_event_count": sum(
            int(item.get("estimated_return_now_deadline_feasible") is False)
            for item in decision_records
        ),
        "estimated_task_then_return_deadline_infeasible_event_count": sum(
            int(item.get("estimated_task_then_return_deadline_feasible") is False)
            for item in decision_records
        ),
        "first_disagreement_count": sum(
            int(bool(item["first_disagreement"])) for item in decision_records
        ),
        "task_schedule_semantics": (
            "method_invariant_keyed_by_seed_cycle_and_task_index"
        ),
    }
    if hasattr(environment.energy_estimator, "cache_diagnostics"):
        audit["oracle_cache_diagnostics"] = (
            environment.energy_estimator.cache_diagnostics()
        )
    environment.close()
    return outcome, audit, cycle_records


def _initialize_stage_b_worker(args: argparse.Namespace) -> None:
    global _STAGE_B_WORKER_ARGS, _STAGE_B_WORKER_POLICY
    os.environ["OMP_NUM_THREADS"] = str(args.torch_threads)
    os.environ["MKL_NUM_THREADS"] = str(args.torch_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(args.torch_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(args.torch_threads)
    os.environ["BLIS_NUM_THREADS"] = str(args.torch_threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(args.torch_threads)
    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(1)
    _STAGE_B_WORKER_ARGS = args
    probe = environment_from_args(args, reserve_fraction=0.0)
    _STAGE_B_WORKER_POLICY = load_policy(args, probe)
    probe.close()
    _atomic_write_json(
        args.output_dir / "worker_health" / f"worker_{os.getpid()}.json",
        {
            "protocol": "stage_b_worker_health_v1",
            "pid": os.getpid(),
            "os_thread_count": _linux_thread_count(),
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
            "thread_environment": {
                name: os.environ.get(name)
                for name in (
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                    "BLIS_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS",
                )
            },
        },
    )


def _linux_thread_count() -> int | None:
    status = Path("/proc/self/status")
    if not status.is_file():
        return None
    for line in status.read_text(encoding="utf-8").splitlines():
        if line.startswith("Threads:"):
            return int(line.split(":", 1)[1].strip())
    return None


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


def _seed_family_result_path(args: argparse.Namespace, evaluation_seed: int) -> Path:
    return args.output_dir / "seed_results" / f"seed_{evaluation_seed}.json"


def _seed_family_failure_path(args: argparse.Namespace, evaluation_seed: int) -> Path:
    return args.output_dir / "seed_failures" / f"seed_{evaluation_seed}.json"


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_seed_family_result(
    args: argparse.Namespace,
    evaluation_seed: int,
) -> dict[str, object]:
    path = _seed_family_result_path(args, evaluation_seed)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"seed-family result must be an object: {path}")
    if payload.get("protocol") != "stage_b_seed_family_v1":
        raise ValueError(f"unexpected seed-family protocol: {path}")
    if int(payload.get("evaluation_seed", -1)) != int(evaluation_seed):
        raise ValueError(f"seed-family result seed mismatch: {path}")
    points = payload.get("points")
    if not isinstance(points, list):
        raise TypeError(f"seed-family points must be a list: {path}")
    expected = [(method, float(parameter)) for method, parameter in parameter_grid(args)]
    observed = [
        (str(item["method"]), float(item["parameter"]))
        for item in points
        if isinstance(item, dict)
    ]
    if observed != expected:
        raise ValueError(f"seed-family candidate grid mismatch: {path}")
    return payload


def _evaluate_stage_b_seed_family_job(evaluation_seed: int) -> str:
    if _STAGE_B_WORKER_ARGS is None or _STAGE_B_WORKER_POLICY is None:
        raise RuntimeError("Stage-B worker was not initialized")
    args = _STAGE_B_WORKER_ARGS
    oracle_bundle_cache: dict[tuple[bytes, bytes, bytes, bytes], tuple] = {}
    points: list[dict[str, object]] = []
    for candidate_index, (method, parameter) in enumerate(parameter_grid(args)):
        try:
            outcome, audit, cycle_records = evaluate_method(
                args,
                _STAGE_B_WORKER_POLICY,
                method=method,
                parameter=parameter,
                evaluation_seed=evaluation_seed,
                oracle_bundle_cache=oracle_bundle_cache,
            )
        except Exception as error:
            failure: dict[str, object] = {
                "protocol": "stage_b_seed_family_failure_v1",
                "evaluation_seed": int(evaluation_seed),
                "candidate_index": int(candidate_index),
                "method": str(method),
                "parameter": float(parameter),
                "completed_candidate_count": len(points),
                "completed_candidates": [
                    {
                        "method": str(item["method"]),
                        "parameter": float(item["parameter"]),
                    }
                    for item in points
                ],
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
            for attribute in ("stage_b_context", "rollout_diagnostics"):
                context = getattr(error, attribute, None)
                if context is not None:
                    failure[attribute] = context
            _atomic_write_json(
                _seed_family_failure_path(args, evaluation_seed),
                failure,
            )
            raise
        points.append(
            {
                "method": method,
                "parameter": float(parameter),
                "outcome": outcome.as_dict(),
                "audit": audit,
                "cycle_records": cycle_records,
            }
        )
    payload = {
        "protocol": "stage_b_seed_family_v1",
        "evaluation_seed": int(evaluation_seed),
        "candidate_count": len(points),
        "shared_oracle_bundle_count": len(oracle_bundle_cache),
        "points": points,
    }
    path = _seed_family_result_path(args, evaluation_seed)
    _atomic_write_json(path, payload)
    return str(path)


def evaluate_formal_seed_families(
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    seed_directory = args.output_dir / "seed_results"
    seed_directory.mkdir(parents=True, exist_ok=True)
    expected_seeds = [int(seed) for seed in args.evaluation_seeds]
    completed: dict[int, dict[str, object]] = {}
    for seed in expected_seeds:
        path = _seed_family_result_path(args, seed)
        if path.is_file():
            completed[seed] = _load_seed_family_result(args, seed)
    attempts = int(args.formal_seed_retries) + 1
    for attempt in range(1, attempts + 1):
        pending = [seed for seed in expected_seeds if seed not in completed]
        if not pending:
            break
        context = mp.get_context("spawn")
        try:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=min(args.evaluation_num_envs, len(pending)),
                mp_context=context,
                initializer=_initialize_stage_b_worker,
                initargs=(args,),
            ) as executor:
                futures = {
                    executor.submit(_evaluate_stage_b_seed_family_job, seed): seed
                    for seed in pending
                }
                for future in concurrent.futures.as_completed(futures):
                    seed = futures[future]
                    future.result()
                    completed[seed] = _load_seed_family_result(args, seed)
                    print(
                        json.dumps(
                            {
                                "status": "SEED_FAMILY_COMPLETED",
                                "evaluation_seed": seed,
                                "completed_seed_families": len(completed),
                                "total_seed_families": len(expected_seeds),
                                "attempt": attempt,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
        except BrokenProcessPool as error:
            print(
                json.dumps(
                    {
                        "status": "WORKER_POOL_BROKEN_RETRYING",
                        "attempt": attempt,
                        "error": str(error),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        for seed in expected_seeds:
            path = _seed_family_result_path(args, seed)
            if seed not in completed and path.is_file():
                completed[seed] = _load_seed_family_result(args, seed)
    missing = [seed for seed in expected_seeds if seed not in completed]
    if missing:
        raise RuntimeError(
            "formal Stage-B exhausted worker retries with missing seed families: "
            f"{missing}"
        )
    return [completed[seed] for seed in expected_seeds]


def evaluate_parameter_across_seeds(
    args: argparse.Namespace,
    policy: FrozenPolicy,
    *,
    method: str,
    parameter: float,
    worker_pool: object | None = None,
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
    if worker_pool is None:
        context = mp.get_context("spawn")
        with context.Pool(
            processes=process_count,
            initializer=_initialize_stage_b_worker,
            initargs=(args,),
        ) as pool:
            raw_results = pool.map(
                _evaluate_stage_b_seed_job,
                jobs,
                chunksize=1,
            )
    else:
        raw_results = worker_pool.map(
            _evaluate_stage_b_seed_job,
            jobs,
            chunksize=1,
        )
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
        "independent_cycles": sum(
            int(item.get("independent_cycles", item["attempted_cycles"]))
            for item in seed_audits
        ),
        "successful_returns": successful_returns,
        "energy_exhaustions": energy_exhaustions,
        "stranding_count": energy_exhaustions,
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
        "exact_return_now_deadline_infeasible_event_count": sum(
            int(item.get("exact_return_now_deadline_infeasible_event_count", 0))
            for item in seed_audits
        ),
        "exact_task_then_return_deadline_infeasible_event_count": sum(
            int(
                item.get(
                    "exact_task_then_return_deadline_infeasible_event_count",
                    0,
                )
            )
            for item in seed_audits
        ),
        "estimated_return_now_deadline_infeasible_event_count": sum(
            int(
                item.get(
                    "estimated_return_now_deadline_infeasible_event_count",
                    0,
                )
            )
            for item in seed_audits
        ),
        "estimated_task_then_return_deadline_infeasible_event_count": sum(
            int(
                item.get(
                    "estimated_task_then_return_deadline_infeasible_event_count",
                    0,
                )
            )
            for item in seed_audits
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
            "shared_bundle_cache_hits": sum(
                int(item.get("shared_bundle_cache_hits", 0))
                for item in oracle_diagnostics
            ),
            "shared_bundle_cache_misses": sum(
                int(item.get("shared_bundle_cache_misses", 0))
                for item in oracle_diagnostics
            ),
        }
    return outcome, aggregate


def attach_paired_oracle_outcomes(
    cycle_records: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Join cycle outcomes to the Oracle point with the same keyed schedule/reserve."""

    oracle_by_pair: dict[tuple[str, float], dict[str, object]] = {}
    for record in cycle_records:
        if record.get("method") != "oracle":
            continue
        key = (
            str(record["paired_schedule_id"]),
            float(record["reserve_fraction"]),
        )
        if key in oracle_by_pair:
            raise ValueError(f"duplicate Oracle cycle for paired key {key}")
        oracle_by_pair[key] = record

    paired: list[dict[str, object]] = []
    for original in cycle_records:
        record = dict(original)
        key = (
            str(record["paired_schedule_id"]),
            float(record["reserve_fraction"]),
        )
        oracle = oracle_by_pair.get(key)
        if record.get("method") == "oracle":
            oracle = record
        record["paired_oracle_available"] = oracle is not None
        if oracle is not None:
            record["paired_oracle_return_success"] = bool(
                oracle["return_success"]
            )
            record["paired_oracle_stranded"] = bool(oracle["stranded"])
            record["paired_oracle_tasks_completed"] = int(
                oracle["tasks_completed"]
            )
        paired.append(record)
    return paired


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
        lambda repetition_index: oracle.estimate_context(
            environment,
            goal,
        ).prediction,
        num_rollouts=args.p0_clone_rollouts,
    )
    environment.close()
    payload = audit.as_dict()
    payload.update(
        probability_semantics_gate(
            audit,
            physically_specified_future_disturbance=False,
        )
    )
    payload["fixed_information"] = [
        "state",
        "goal",
        "frozen_policy",
        "safety_operator",
        "obstacle_realization",
    ]
    payload["varied_information"] = []
    payload["repetition_index_is_not_a_disturbance_seed"] = True
    payload["current_environment_has_seeded_future_disturbance"] = False
    return payload


def parameter_grid(args: argparse.Namespace) -> Iterable[tuple[str, float]]:
    for method in args.methods:
        values = args.soc_thresholds if method == "soc" else args.reserve_fractions
        for value in values:
            yield method, float(value)


def formal_design_manifest(args: argparse.Namespace) -> dict[str, object]:
    grid = list(parameter_grid(args))
    independent_cycles = int(args.cycles_per_point * len(args.evaluation_seeds))
    safety_power = None
    if not args.smoke and args.oracle_headroom_json is None:
        safety_power = stranding_certification_power_audit(
            num_candidate_points=len(grid),
            planned_independent_cycles_per_point=independent_cycles,
            stranding_ceiling=args.oracle_stranding_ceiling,
            design_stranding_rate=args.oracle_design_stranding_rate,
            confidence_level=args.oracle_familywise_safety_confidence_level,
            target_joint_certification_power=(
                args.oracle_min_joint_safety_certification_power
            ),
        ).as_dict()
    process_count = min(args.evaluation_num_envs, len(args.evaluation_seeds))
    return {
        "evidence_mode": "EXPLORATORY" if args.smoke else "FORMAL",
        "candidate_points": [
            {"method": method, "parameter": parameter}
            for method, parameter in grid
        ],
        "num_candidate_points": len(grid),
        "independent_cycles_per_point": independent_cycles,
        "total_battery_cycle_jobs": len(grid) * independent_cycles,
        "evaluation_seed_count": len(args.evaluation_seeds),
        "evaluation_seed_sha256": hashlib.sha256(
            json.dumps(list(args.evaluation_seeds), separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest(),
        "parallel_worker_processes": process_count,
        "worker_pool_lifetime": (
            "single_process_no_pool"
            if process_count == 1
            else "restartable_pool_over_atomic_seed_families"
        ),
        "maximum_policy_worker_initializations": process_count,
        "execution_unit": "one_seed_across_complete_candidate_grid",
        "cross_candidate_oracle_cache": (
            "exact_float32_state_goal_and_charger_key_within_seed"
        ),
        "torch_threads_per_worker": int(args.torch_threads),
        "oracle_shadow_scope": (
            "exact_through_and_including_first_disagreement_for_non_oracle_methods;"
            "full_cycle_for_oracle"
        ),
        "oracle_horizon_semantics": (
            "finite_deadline_completion_resource_as_tagged_extended_real"
        ),
        "oracle_horizon_failure_policy": (
            "continue_if_task_then_return_is_certified_even_when_direct_return_is_"
            "uncertified;commit_if_task_then_return_is_uncertified;"
            "label_emergency_if_direct_return_is_also_uncertified"
        ),
        "oracle_horizon_non_claim": (
            "deadline_infeasible_does_not_assert_infinite_horizon_unreachability;"
            "truncated_prefix_energy_is_not_completed_goal_resource_to_go"
        ),
        "task_timeout_semantics": (
            "failed_TASK_goal_rolls_to_next_keyed_task_without_position_battery_"
            "simulation_time_or_distance_reset"
        ),
        "committed_return_timeout_semantics": (
            "CHARGER_COMMITTED_mode_cannot_use_task_rollover"
        ),
        "episode_guard_semantics": (
            "fail_closed_and_strictly_beyond_certified_physical_energy_exhaustion_bound"
        ),
        "paired_schedule_join": "same_seed_cycle_and_reserve",
        "stranding_certification_power_audit": safety_power,
        "throughput_design_status": (
            "VALID_SIMULTANEOUS_THREE_WAY_GATE_WITHOUT_ASSUMED_EFFECT_POWER"
        ),
        "throughput_design_note": (
            "fixed-n max-t inference controls the selection-aware decision; "
            "without a preregistered paired-rate variance, 320 cycles do not "
            "guarantee power to resolve exactly five percent and may return "
            "INCONCLUSIVE"
        ),
    }


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
) -> dict[str, object]:
    (output / "probability_semantics_audit.json").write_text(
        json.dumps(p0, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    frontier = pareto_frontier(outcomes)
    paired_cycle_records = attach_paired_oracle_outcomes(cycle_records)
    cycles_per_point = {
        (str(item["method"]), float(item["parameter"])): int(
            item["independent_cycles"]
        )
        for item in aggregate_audits
    }
    familywise_stranding_audit = None
    stranding_power_audit = None
    if inherited_oracle_gate is None:
        relevant_aggregates = [
            item
            for item in aggregate_audits
            if item["method"] in {"oracle", "soc", "distance"}
        ]
        familywise_stranding_audit = simultaneous_stranding_upper_bounds(
            {
                (str(item["method"]), float(item["parameter"])): (
                    int(item["stranding_count"]),
                    int(item["independent_cycles"]),
                )
                for item in relevant_aggregates
            },
            confidence_level=args.oracle_familywise_safety_confidence_level,
        )
        if 0.0 < args.oracle_stranding_ceiling < 1.0:
            stranding_power_audit = stranding_certification_power_audit(
                num_candidate_points=len(relevant_aggregates),
                planned_independent_cycles_per_point=min(
                    int(item["independent_cycles"])
                    for item in relevant_aggregates
                ),
                stranding_ceiling=args.oracle_stranding_ceiling,
                design_stranding_rate=args.oracle_design_stranding_rate,
                confidence_level=(
                    args.oracle_familywise_safety_confidence_level
                ),
                target_joint_certification_power=(
                    args.oracle_min_joint_safety_certification_power
                ),
            )
        stranding_upper_bounds = {
            (point.method, point.parameter): point.upper_confidence_bound
            for point in familywise_stranding_audit.points
        }
        for item in relevant_aggregates:
            item["stranding_rate_familywise_upper"] = stranding_upper_bounds[
                (str(item["method"]), float(item["parameter"]))
            ]
        preliminary_gate = oracle_headroom_gate(
            outcomes,
            cycles_per_point=cycles_per_point,
            stranding_upper_bounds=stranding_upper_bounds,
            minimum_cycles_per_point=args.minimum_cycles_for_gate,
            stranding_ceiling=args.oracle_stranding_ceiling,
            minimum_throughput_gain_fraction=(
                args.oracle_min_throughput_gain_fraction
            ),
        )
        eligible_oracle_points = [
            (str(item["method"]), float(item["parameter"]))
            for item in aggregate_audits
            if item["method"] == "oracle"
            and float(item["stranding_rate_familywise_upper"])
            <= args.oracle_stranding_ceiling
        ]
        eligible_heuristic_points = [
            (str(item["method"]), float(item["parameter"]))
            for item in aggregate_audits
            if item["method"] in {"soc", "distance"}
            and float(item["stranding_rate_familywise_upper"])
            <= args.oracle_stranding_ceiling
        ]
        paired_throughput_interval = None
        if eligible_oracle_points and eligible_heuristic_points:
            all_oracle_points = [
                (str(item["method"]), float(item["parameter"]))
                for item in aggregate_audits
                if item["method"] == "oracle"
            ]
            all_heuristic_points = [
                (str(item["method"]), float(item["parameter"]))
                for item in aggregate_audits
                if item["method"] in {"soc", "distance"}
            ]
            paired_throughput_interval = paired_frontier_throughput_interval(
                paired_cycle_records,
                oracle_points=all_oracle_points,
                heuristic_points=all_heuristic_points,
                eligible_oracle_points=eligible_oracle_points,
                eligible_heuristic_points=eligible_heuristic_points,
                confidence_level=args.oracle_throughput_confidence_level,
                bootstrap_replicates=(
                    args.oracle_throughput_bootstrap_replicates
                ),
                bootstrap_seed=args.oracle_throughput_bootstrap_seed,
            )
            if (
                preliminary_gate.throughput_gain_fraction is not None
                and not np.isclose(
                    preliminary_gate.throughput_gain_fraction,
                    paired_throughput_interval.throughput_gain_fraction,
                    rtol=1e-10,
                    atol=1e-12,
                )
            ):
                raise RuntimeError(
                    "aggregate and paired Oracle-headroom point estimates disagree"
                )
            gate = oracle_headroom_gate(
                outcomes,
                cycles_per_point=cycles_per_point,
                stranding_upper_bounds=stranding_upper_bounds,
                minimum_cycles_per_point=args.minimum_cycles_for_gate,
                stranding_ceiling=args.oracle_stranding_ceiling,
                minimum_throughput_gain_fraction=(
                    args.oracle_min_throughput_gain_fraction
                ),
                throughput_gain_lower_confidence_bound=(
                    paired_throughput_interval.throughput_gain_lower_confidence_bound
                ),
                throughput_gain_upper_confidence_bound=(
                    paired_throughput_interval.throughput_gain_upper_confidence_bound
                ),
                throughput_confidence_level=(
                    paired_throughput_interval.confidence_level
                ),
            )
        else:
            gate = preliminary_gate
        gate_payload = {
            **gate.as_dict(),
            "stranding_statistic": FAMILYWISE_STRANDING_STATISTIC,
            "simultaneous_stranding_audit": familywise_stranding_audit.as_dict(),
            "stranding_certification_power_audit": (
                None
                if stranding_power_audit is None
                else stranding_power_audit.as_dict()
            ),
            "point_estimates_not_used_as_safety_gate": True,
            "throughput_statistic": SIMULTANEOUS_THROUGHPUT_STATISTIC,
            "point_estimate_not_sufficient_for_throughput_gate": True,
            "paired_throughput_interval": (
                None
                if paired_throughput_interval is None
                else paired_throughput_interval.as_dict()
            ),
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
    if (
        not args.smoke
        and inherited_oracle_gate is None
        and (
            stranding_power_audit is None
            or not stranding_power_audit.passed
        )
    ):
        gate_payload["statistical_gate_before_power_audit"] = {
            "status": gate_payload.get("status"),
            "evaluable": gate_payload.get("evaluable"),
            "passed": gate_payload.get("passed"),
        }
        gate_payload.update(
            {
                "status": "FAIL_UNDERPOWERED_STRANDING_CERTIFICATION",
                "evaluable": False,
                "passed": False,
                "reason": (
                    "formal evidence requires the preregistered minimum "
                    "stranding-certification power"
                ),
            }
        )
        gate_passed = False
    decision_events = [
        event
        for seed_audit in seed_audits
        for event in seed_audit.get("decision_events", [])
    ]
    paired_cycle_evidence_complete = bool(paired_cycle_records) and all(
        bool(record["paired_oracle_available"])
        for record in paired_cycle_records
    )
    coupling_audit = audit_oracle_shadow_coupling(decision_events)
    coupling_payload = coupling_audit.as_dict()
    (output / "oracle_shadow_coupling_audit.json").write_text(
        json.dumps(coupling_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    evidence_integrity_passed = bool(
        paired_cycle_evidence_complete and coupling_audit.passed
    )
    gate_payload["oracle_shadow_coupling_audit"] = coupling_payload
    gate_payload["paired_cycle_evidence_complete"] = (
        paired_cycle_evidence_complete
    )
    gate_payload["evidence_integrity_passed"] = evidence_integrity_passed
    if not args.smoke and not evidence_integrity_passed:
        gate_payload["statistical_gate_before_evidence_integrity"] = {
            "status": gate_payload.get("status"),
            "evaluable": gate_payload.get("evaluable"),
            "passed": gate_payload.get("passed"),
        }
        gate_payload.update(
            {
                "status": "FAIL_ORACLE_SHADOW_EVIDENCE_INTEGRITY",
                "evaluable": False,
                "passed": False,
                "reason": (
                    "formal evidence requires complete same-schedule Oracle cycle "
                    "joins and exact pathwise coupling through first disagreement"
                ),
            }
        )
        gate_passed = False
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
        "cycle_evidence_unit": (
            "independent_seeded_battery_cycle"
            if args.cycles_per_point == 1
            else "smoke_only_continuous_cycles_within_seed"
        ),
        "wilson_interval_independence_design": bool(
            not args.smoke and args.cycles_per_point == 1
        ),
        "familywise_stranding_audit": (
            None
            if familywise_stranding_audit is None
            else familywise_stranding_audit.as_dict()
        ),
        "stranding_certification_power_audit": (
            None
            if stranding_power_audit is None
            else stranding_power_audit.as_dict()
        ),
        "seed_parallelism_semantics": (
            "independent_seed_processes_same_config_and_one_cycle_budget"
        ),
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
        "task_schedule_semantics": (
            "method_invariant_keyed_by_evaluation_seed_cycle_and_task_index"
        ),
        "decision_event_count": len(decision_events),
        "first_disagreement_count": sum(
            int(bool(event["first_disagreement"]))
            for event in decision_events
        ),
        "paired_cycle_evidence_complete": paired_cycle_evidence_complete,
        "oracle_shadow_coupling_audit": coupling_payload,
        "evidence_integrity_passed": evidence_integrity_passed,
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
                else "INCONCLUSIVE_ORACLE_HEADROOM"
                if gate_payload.get("status") == "INCONCLUSIVE_ORACLE_HEADROOM"
                else "STOP_AFTER_ORACLE_HEADROOM_GATE"
                )
            )
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    formal_eligible = bool(
        not args.smoke
        and prerequisite_audit.get("passed") is True
        and decision_events
        and paired_cycle_evidence_complete
        and coupling_audit.passed
        and aggregate_audits
        and (
            inherited_oracle_gate is not None
            or (
                stranding_power_audit is not None
                and stranding_power_audit.passed
            )
        )
    )
    evidence_bundle = empty_oracle_pareto_bundle()
    evidence_bundle.update(
        {
            "evidence_mode": "FORMAL" if not args.smoke else "EXPLORATORY",
            "claim_status": (
                "SMOKE_ONLY_NOT_FORMAL_EVIDENCE"
                if args.smoke
                else "PAIRED_EVIDENCE_INCOMPLETE"
                if not formal_eligible
                else "POST_ORACLE_HEADROOM_DECISION_COMPARISON"
                if inherited_oracle_gate is not None
                else "ORACLE_HEADROOM_GATE_PASS"
                if gate_passed is True
                else "ORACLE_HEADROOM_GATE_INCONCLUSIVE"
                if gate_payload.get("status") == "INCONCLUSIVE_ORACLE_HEADROOM"
                else "ORACLE_HEADROOM_GATE_FAIL"
            ),
            "formal_eligible": formal_eligible,
            "provenance": {
                "navigation_artifact": (
                    None
                    if args.navigation_artifact is None
                    else str(args.navigation_artifact)
                ),
                "navigation_artifact_config_sha256": (
                    None
                    if getattr(args, "navigation_environment_contract", None)
                    is None
                    else args.navigation_environment_contract.get(
                        "config_sha256"
                    )
                ),
                "navigation_environment_contract": getattr(
                    args,
                    "navigation_environment_contract",
                    None,
                ),
                "navigation_checkpoint": (
                    None
                    if args.navigation_checkpoint is None
                    else str(args.navigation_checkpoint)
                ),
                "navigation_evaluation_artifact": (
                    None
                    if args.navigation_evaluation_json is None
                    else str(args.navigation_evaluation_json)
                ),
                "battery_calibration_artifact": (
                    None
                    if args.battery_calibration_json is None
                    else str(args.battery_calibration_json)
                ),
                "battery_validation_artifact": (
                    None
                    if args.battery_validation_json is None
                    else str(args.battery_validation_json)
                ),
                "navigation_artifact_schema": prerequisite_audit.get(
                    "navigation_artifact_schema"
                ),
                "navigation_checkpoint_sha256": optional_file_sha256(
                    args.navigation_checkpoint
                ),
                "navigation_wrapper_checkpoint_sha256": (
                    prerequisite_audit.get("navigation_checkpoint_sha256")
                ),
                "navigation_evaluation_sha256": optional_file_sha256(
                    args.navigation_evaluation_json
                ),
                "battery_calibration_sha256": optional_file_sha256(
                    args.battery_calibration_json
                ),
                "battery_validation_sha256": optional_file_sha256(
                    args.battery_validation_json
                ),
            },
            "protocol": {
                "evaluation_seeds": list(args.evaluation_seeds),
                "cycles_per_point": int(args.cycles_per_point),
                "minimum_cycles_per_point": int(args.minimum_cycles_for_gate),
                "stranding_ceiling": float(args.oracle_stranding_ceiling),
                "familywise_safety_confidence_level": float(
                    args.oracle_familywise_safety_confidence_level
                ),
                "familywise_safety_correction": (
                    "bonferroni_one_sided_clopper_pearson"
                ),
                "design_stranding_rate": float(
                    args.oracle_design_stranding_rate
                ),
                "minimum_joint_safety_certification_power": float(
                    args.oracle_min_joint_safety_certification_power
                ),
                "minimum_throughput_gain_fraction": float(
                    args.oracle_min_throughput_gain_fraction
                ),
                "throughput_confidence_level": float(
                    args.oracle_throughput_confidence_level
                ),
                "throughput_bootstrap_replicates": int(
                    args.oracle_throughput_bootstrap_replicates
                ),
                "throughput_bootstrap_seed": int(
                    args.oracle_throughput_bootstrap_seed
                ),
                "decision_interval_policy_steps": int(
                    args.decision_interval_policy_steps
                ),
                "task_schedule_semantics": summary["task_schedule_semantics"],
            },
            "gates": {
                "navigation": {
                    "status": prerequisite_audit.get("status"),
                    "passed": prerequisite_audit.get("passed"),
                    "artifact": (
                        None
                        if args.navigation_evaluation_json is None
                        else str(args.navigation_evaluation_json)
                    ),
                },
                "probability_semantics": {
                    "status": p0.get("status"),
                    "passed": p0.get("passed"),
                    "artifact": str(output / "probability_semantics_audit.json"),
                },
                "battery_calibration": {
                    "status": prerequisite_audit.get("status"),
                    "passed": prerequisite_audit.get("passed"),
                    "artifact": (
                        None
                        if args.battery_calibration_json is None
                        else str(args.battery_calibration_json)
                    ),
                },
                "battery_validation": {
                    "status": prerequisite_audit.get("status"),
                    "passed": prerequisite_audit.get("passed"),
                    "artifact": (
                        None
                        if args.battery_validation_json is None
                        else str(args.battery_validation_json)
                    ),
                },
                "oracle_decision_headroom": {
                    "status": gate_payload.get("status"),
                    "passed": gate_payload.get("passed"),
                    "artifact": str(output / "oracle_headroom_gate.json"),
                },
            },
            "oracle_shadow_coupling_audit": coupling_payload,
            "simultaneous_stranding_audit": gate_payload.get(
                "simultaneous_stranding_audit",
                {
                    "status": "NOT_EVALUATED",
                    "correction": None,
                    "num_candidate_points": 0,
                    "points": [],
                },
            ),
            "stranding_certification_power_audit": gate_payload.get(
                "stranding_certification_power_audit"
            ),
            "paired_throughput_interval": gate_payload.get(
                "paired_throughput_interval"
            ),
            "decision_events": decision_events,
            "cycles": paired_cycle_records,
            "method_summaries": aggregate_audits,
            "theory_diagnostics": [],
            "pareto_frontier": [item.as_dict() for item in frontier],
        }
    )
    evidence_errors = validate_oracle_pareto_bundle(evidence_bundle)
    if evidence_errors:
        raise RuntimeError(
            "Oracle/Pareto evidence bundle failed validation: "
            + "; ".join(evidence_errors)
        )
    (output / "oracle_pareto_evidence.json").write_text(
        json.dumps(evidence_bundle, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_csv(output / "outcomes.csv", aggregate_audits)
    write_csv(
        output / "seed_outcomes.csv",
        [
            {
                key: value
                for key, value in item.items()
                if key != "decision_events"
            }
            for item in seed_audits
        ],
    )
    write_csv(
        output / "battery_cycles.csv",
        [
            {key: value for key, value in item.items() if key != "battery_cycle_record"}
            for item in paired_cycle_records
        ],
    )
    with (output / "battery_cycles.jsonl").open("w", encoding="utf-8") as handle:
        for item in paired_cycle_records:
            handle.write(json.dumps(item, sort_keys=True) + "\n")
    with (output / "decision_events.jsonl").open("w", encoding="utf-8") as handle:
        for item in decision_events:
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
    return gate_payload


def stage_b_completion_contract(
    args: argparse.Namespace,
    gate_payload: dict[str, object],
) -> tuple[str, int, str]:
    if args.smoke:
        return "SMOKE_COMPLETED.json", 0, "SMOKE_ONLY_NOT_FORMAL_EVIDENCE"
    if gate_payload.get("evidence_integrity_passed") is not True:
        return (
            "STOPPED_AFTER_EVIDENCE_INTEGRITY.json",
            4,
            "FAIL_ORACLE_SHADOW_EVIDENCE_INTEGRITY",
        )
    if args.oracle_headroom_json is not None:
        return "COMPLETED.json", 0, "POST_ORACLE_HEADROOM_DECISION_COMPARISON"
    if gate_payload.get("passed") is True and gate_payload.get("status") == "PASS":
        return "COMPLETED.json", 0, "ORACLE_HEADROOM_GATE_PASS"
    return (
        "STOPPED_AFTER_ORACLE_HEADROOM_GATE.json",
        4,
        str(gate_payload.get("status", "UNKNOWN_GATE_STATUS")),
    )


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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def optional_file_sha256(path: Path | None) -> str | None:
    return None if path is None or not path.is_file() else file_sha256(path)


def _jsonable_environment_contract(
    kwargs: dict[str, object],
) -> dict[str, object]:
    contract: dict[str, object] = {}
    for key, value in kwargs.items():
        if is_dataclass(value):
            contract[key] = asdict(value)
        elif isinstance(value, SACTrainingPhase):
            contract[key] = int(value)
        else:
            contract[key] = value
    return contract


def _bind_navigation_environment_contract(
    args: argparse.Namespace,
) -> tuple[list[str], str | None]:
    failures: list[str] = []
    if args.navigation_artifact is None:
        failures.append("formal Stage-B requires a named navigation artifact")
        return failures, None
    artifact = args.navigation_artifact.resolve()
    config_path = artifact / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    if args.navigation_checkpoint is not None:
        try:
            args.navigation_checkpoint.resolve().relative_to(artifact)
        except ValueError:
            failures.append(
                "Stage-B checkpoint is not contained in the named navigation artifact"
            )
    source_args, reconstruction = reconstruct_environment_args(
        artifact,
        device=args.device,
        seed=args.seed,
    )
    kwargs = environment_kwargs_from_args(
        source_args,
        phase=SACTrainingPhase.ENERGY_MANAGED,
    )
    args.navigation_environment_kwargs = kwargs
    args.navigation_environment_contract = {
        "source_artifact": str(artifact),
        "config_sha256": file_sha256(config_path),
        "reconstruction_audit": reconstruction.get(
            "environment_reconstruction"
        ),
        "environment_kwargs": _jsonable_environment_contract(kwargs),
    }
    return failures, file_sha256(config_path)


def _audit_with_additional_failures(
    audit: GateBPrerequisiteAudit,
    failures: list[str],
) -> GateBPrerequisiteAudit:
    if not failures:
        return audit
    return GateBPrerequisiteAudit(
        passed=False,
        status="FAIL_GATE_B_PREREQUISITES",
        failures=tuple([*audit.failures, *failures]),
        calibrated_battery_capacity=audit.calibrated_battery_capacity,
        navigation_success_rate=audit.navigation_success_rate,
        calibration_success_rate=audit.calibration_success_rate,
        observed_endurance_seconds=audit.observed_endurance_seconds,
        navigation_artifact_schema=audit.navigation_artifact_schema,
        navigation_checkpoint_sha256=audit.navigation_checkpoint_sha256,
        fixed_baseline_contract=audit.fixed_baseline_contract,
        baseline_id=audit.baseline_id,
        navigation_performance_metrics_descriptive=(
            audit.navigation_performance_metrics_descriptive
        ),
        battery_validation_schema=audit.battery_validation_schema,
        endurance_estimand=audit.endurance_estimand,
        continuous_workload_until_depletion=(
            audit.continuous_workload_until_depletion
        ),
        censored_run_count=audit.censored_run_count,
        battery_validation_runs=audit.battery_validation_runs,
    )


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
    navigation, calibration, validation = payloads
    audit = audit_gate_b_prerequisites(navigation, calibration, validation)
    navigation_path, calibration_path, _ = required.values()
    assert navigation_path is not None
    assert calibration_path is not None
    navigation_view = navigation_artifact_view(navigation)
    identity_failures: list[str] = []
    if not navigation_view.wrapped:
        identity_failures.append(
            "formal Stage-B requires a navigation completion wrapper with "
            "explicit downstream authorization"
        )
    checkpoint_sha = None
    if args.navigation_checkpoint is None:
        identity_failures.append(
            "formal Stage-B requires a named navigation checkpoint"
        )
    elif not args.navigation_checkpoint.is_file():
        raise FileNotFoundError(args.navigation_checkpoint)
    else:
        checkpoint_sha = file_sha256(args.navigation_checkpoint)
        if navigation_view.checkpoint_sha256 != checkpoint_sha:
            identity_failures.append(
                "navigation completion wrapper checkpoint SHA does not match "
                "the Stage-B checkpoint"
            )
    navigation_sha = file_sha256(navigation_path)
    calibration_sha = file_sha256(calibration_path)
    environment_failures, navigation_config_sha = (
        _bind_navigation_environment_contract(args)
    )
    identity_failures.extend(environment_failures)

    def require_sha(
        payload: dict[str, object],
        field: str,
        expected: str | None,
        label: str,
    ) -> None:
        if expected is None or payload.get(field) != expected:
            identity_failures.append(f"{label} provenance mismatch: {field}")

    require_sha(
        calibration,
        "navigation_checkpoint_sha256",
        checkpoint_sha,
        "battery calibration",
    )
    require_sha(
        calibration,
        "navigation_evaluation_sha256",
        navigation_sha,
        "battery calibration",
    )
    require_sha(
        validation,
        "navigation_checkpoint_sha256",
        checkpoint_sha,
        "battery validation",
    )
    require_sha(
        calibration,
        "navigation_artifact_config_sha256",
        navigation_config_sha,
        "battery calibration",
    )
    require_sha(
        validation,
        "navigation_artifact_config_sha256",
        navigation_config_sha,
        "battery validation",
    )
    require_sha(
        validation,
        "navigation_evaluation_sha256",
        navigation_sha,
        "battery validation",
    )
    require_sha(
        validation,
        "battery_calibration_sha256",
        calibration_sha,
        "battery validation",
    )
    audit = _audit_with_additional_failures(audit, identity_failures)
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
        args.navigation_environment_contract["stage_b_runtime_overrides"] = {
            "phase": "ENERGY_MANAGED",
            "operational_energy_capacity": float(calibrated_capacity),
            "energy_reserve_fractions": [
                float(value) for value in args.reserve_fractions
            ],
            "mission_decision_interval_policy_steps": int(
                args.decision_interval_policy_steps
            ),
            "continuous_task_workload": True,
            "task_step_limit_semantics": (
                "roll_to_next_keyed_task_without_physical_state_reset"
            ),
            "episode_guard_contract": stage_b_episode_guard_contract(
                args,
                args.navigation_environment_kwargs,
            ),
        }
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
        or payload.get("evidence_integrity_passed") is not True
    ):
        raise RuntimeError(
            "post-Gate comparison requires an evaluable PASS Gate with passed "
            "Oracle-shadow evidence integrity"
        )
    safety_audit = payload.get("simultaneous_stranding_audit")
    if (
        payload.get("stranding_statistic") != FAMILYWISE_STRANDING_STATISTIC
        or not isinstance(safety_audit, dict)
        or safety_audit.get("correction")
        != "bonferroni_one_sided_clopper_pearson"
        or not isinstance(safety_audit.get("points"), list)
        or len(safety_audit["points"]) != safety_audit.get("num_candidate_points")
    ):
        raise RuntimeError(
            "post-Gate comparison requires selection-valid simultaneous "
            "stranding safety evidence"
        )
    power_audit = payload.get("stranding_certification_power_audit")
    if (
        not isinstance(power_audit, dict)
        or power_audit.get("passed") is not True
        or power_audit.get("status") != "PASS_DESIGN_POWER"
    ):
        raise RuntimeError(
            "post-Gate comparison requires a passed stranding-certification "
            "power audit"
        )
    throughput_audit = payload.get("paired_throughput_interval")
    throughput_lower = (
        None
        if not isinstance(throughput_audit, dict)
        else throughput_audit.get("throughput_gain_lower_confidence_bound")
    )
    throughput_upper = (
        None
        if not isinstance(throughput_audit, dict)
        else throughput_audit.get("throughput_gain_upper_confidence_bound")
    )
    if (
        payload.get("throughput_statistic") != SIMULTANEOUS_THROUGHPUT_STATISTIC
        or not isinstance(throughput_audit, dict)
        or throughput_audit.get("inference")
        != SIMULTANEOUS_THROUGHPUT_INFERENCE
        or not isinstance(throughput_audit.get("oracle_points"), list)
        or not isinstance(throughput_audit.get("heuristic_points"), list)
        or not isinstance(throughput_audit.get("eligible_oracle_points"), list)
        or not isinstance(throughput_audit.get("eligible_heuristic_points"), list)
        or not isinstance(throughput_audit.get("max_t_critical_value"), (int, float))
        or not np.isfinite(float(throughput_audit.get("max_t_critical_value")))
        or float(throughput_audit.get("max_t_critical_value")) < 0.0
        or not isinstance(
            throughput_audit.get("upper_max_t_critical_value"), (int, float)
        )
        or not np.isfinite(
            float(throughput_audit.get("upper_max_t_critical_value"))
        )
        or float(throughput_audit.get("upper_max_t_critical_value")) < 0.0
        or not all(
            isinstance(throughput_audit.get(field), list)
            and bool(throughput_audit.get(field))
            for field in (
                "oracle_rate_lower_bounds",
                "oracle_rate_upper_bounds",
                "heuristic_rate_lower_bounds",
                "heuristic_rate_upper_bounds",
            )
        )
        or not isinstance(throughput_lower, (int, float))
        or not np.isfinite(float(throughput_lower))
        or not isinstance(throughput_upper, (int, float))
        or not np.isfinite(float(throughput_upper))
        or float(throughput_lower) > float(throughput_upper)
    ):
        raise RuntimeError(
            "post-Gate comparison requires a full-family simultaneous "
            "throughput lower/upper-bound audit"
        )
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
        if not args.resume:
            raise FileExistsError(f"output directory already exists: {args.output_dir}")
        existing_launch_path = args.output_dir / "launch.json"
        if not existing_launch_path.is_file():
            raise FileNotFoundError(
                f"resume requires an existing launch manifest: {existing_launch_path}"
            )
        existing_launch = json.loads(existing_launch_path.read_text(encoding="utf-8"))
        existing_arguments = dict(existing_launch.get("arguments", {}))
        requested_arguments = jsonable_args(args)
        existing_arguments.pop("resume", None)
        requested_arguments.pop("resume", None)
        if existing_arguments != requested_arguments:
            raise ValueError("resume arguments do not match the existing launch manifest")
    else:
        args.output_dir.mkdir(parents=True, exist_ok=False)
    launch = {
        "arguments": jsonable_args(args),
        "command": [sys.executable, *sys.argv],
        "git": git_provenance(),
        "formal_design": formal_design_manifest(args),
    }
    if not args.resume:
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
            prerequisite_stop = {
                "protocol": "stage_b_oracle_decision_headroom",
                "status": prerequisite.status,
                "exit_code": 4,
                "downstream_authorized": False,
                "prerequisite_audit": prerequisite.as_dict(),
            }
            (args.output_dir / "STOPPED_PREREQUISITES_NOT_READY.json").write_text(
                json.dumps(prerequisite_stop, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            raise SystemExit(4)
        inherited_oracle_gate = (
            None
            if args.oracle_headroom_json is None
            else load_passed_oracle_headroom_gate(args.oracle_headroom_json)
        )
        probe = environment_from_args(args, reserve_fraction=0.0)
        policy = load_policy(args, probe)
        probe.close()
        p0 = run_p0_audit(args, policy)
        (args.output_dir / "probability_semantics_audit.json").write_text(
            json.dumps(p0, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if not args.smoke and p0.get("passed") is not True:
            probability_stop = {
                "protocol": "stage_b_oracle_decision_headroom",
                "status": p0.get("status"),
                "exit_code": 4,
                "downstream_authorized": False,
                "probability_semantics": p0,
            }
            (
                args.output_dir
                / "STOPPED_PROBABILITY_SEMANTICS_NOT_READY.json"
            ).write_text(
                json.dumps(probability_stop, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            raise SystemExit(4)
        outcomes: list[ReturnDecisionOutcome] = []
        aggregate_audits: list[dict[str, object]] = []
        all_seed_audits: list[dict[str, object]] = []
        all_cycle_records: list[dict[str, object]] = []
        if not args.smoke:
            del policy
            seed_families = evaluate_formal_seed_families(args)
            grid = list(parameter_grid(args))
            for point_index, (method, parameter) in enumerate(grid):
                point_seed_audits: list[dict[str, object]] = []
                for family in seed_families:
                    point = family["points"][point_index]
                    seed_audit = point["audit"]
                    cycle_records = point["cycle_records"]
                    point_seed_audits.append(seed_audit)
                    all_seed_audits.append(seed_audit)
                    all_cycle_records.extend(cycle_records)
                outcome, aggregate = aggregate_seed_audits(
                    method=method,
                    parameter=parameter,
                    seed_audits=point_seed_audits,
                )
                outcomes.append(outcome)
                aggregate_audits.append(aggregate)
                print(json.dumps(aggregate, sort_keys=True), flush=True)
        else:
            worker_pool = None
            if args.evaluation_num_envs > 1 and len(args.evaluation_seeds) > 1:
                process_count = min(
                    args.evaluation_num_envs,
                    len(args.evaluation_seeds),
                )
                context = mp.get_context("spawn")
                worker_pool = context.Pool(
                    processes=process_count,
                    initializer=_initialize_stage_b_worker,
                    initargs=(args,),
                )
            try:
                for method, parameter in parameter_grid(args):
                    point_seed_audits = []
                    point_results = evaluate_parameter_across_seeds(
                        args,
                        policy,
                        method=method,
                        parameter=parameter,
                        worker_pool=worker_pool,
                    )
                    for _, seed_audit, cycle_records in point_results:
                        point_seed_audits.append(seed_audit)
                        all_seed_audits.append(seed_audit)
                        all_cycle_records.extend(cycle_records)
                    outcome, aggregate = aggregate_seed_audits(
                        method=method,
                        parameter=parameter,
                        seed_audits=point_seed_audits,
                    )
                    outcomes.append(outcome)
                    aggregate_audits.append(aggregate)
            except BaseException:
                if worker_pool is not None:
                    worker_pool.terminate()
                    worker_pool.join()
                raise
            else:
                if worker_pool is not None:
                    worker_pool.close()
                    worker_pool.join()
        gate_payload = write_results(
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
        sentinel_name, exit_code, status = stage_b_completion_contract(
            args,
            gate_payload,
        )
        (args.output_dir / sentinel_name).write_text(
            json.dumps(
                {
                    "protocol": "stage_b_oracle_decision_headroom",
                    "status": status,
                    "exit_code": exit_code,
                    "oracle_headroom_gate": gate_payload,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        if exit_code:
            raise SystemExit(exit_code)
    except Exception as error:
        failure_payload: dict[str, object] = {
            "error_type": type(error).__name__,
            "error": str(error),
            "protocol": "stage_b_oracle_decision_headroom",
        }
        for attribute in ("stage_b_context", "rollout_diagnostics"):
            context = getattr(error, attribute, None)
            if context is not None:
                failure_payload[attribute] = context
        (args.output_dir / "FAILED.json").write_text(
            json.dumps(failure_payload, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        raise


if __name__ == "__main__":
    main()
