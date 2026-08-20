#!/usr/bin/env python3
from __future__ import annotations

import argparse
import heapq
import json
import platform
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np

from review_bundle.safety.collision.feasibility import joint_feasibility_margin
from review_bundle.safety.collision.hocbf import (
    HOCBFConfig,
    SphericalObstacle,
    hocbf_sampled_data_residual_bound,
    sphere_hocbf_constraint,
    strengthen_constraint_for_sample_hold,
)


WORLD_SIZE = np.array([4000.0, 4000.0, 400.0], dtype=np.float64)
HORIZONTAL_V_MAX = 20.0
VERTICAL_V_MAX = 5.0
HORIZONTAL_A_MAX = 5.0
VERTICAL_A_MAX = 3.0
UAV_RADIUS = 0.5
HOCBF = HOCBFConfig(k1=1.0, k2=1.0, uav_radius=UAV_RADIUS)
FAMILIES = (
    "high_closing_speed",
    "low_clearance",
    "moving_multi_obstacle_conflict",
    "narrow_passage",
    "horizontal_velocity_saturation",
    "vertical_velocity_saturation",
    "sample_delay",
)


@dataclass(frozen=True)
class HardStateRecord:
    state_index: int
    family: str
    position: list[float]
    velocity: list[float]
    hold_dt: float
    obstacles: list[dict[str, object]]
    minimum_h: float
    minimum_psi1: float
    continuous_margin: float
    sampled_margin: float
    continuous_feasible: bool
    sampled_feasible: bool
    maximizing_action: list[float]
    primal_dual_gap: float
    solve_seconds: float
    global_certificate_fallback_used: bool


def _mostly_horizontal_direction(rng: np.random.Generator) -> np.ndarray:
    angle = float(rng.uniform(-np.pi, np.pi))
    z = float(rng.uniform(-0.2, 0.2))
    horizontal = float(np.sqrt(1.0 - z * z))
    return np.array(
        [horizontal * np.cos(angle), horizontal * np.sin(angle), z],
        dtype=np.float64,
    )


def _orthogonal_horizontal(direction: np.ndarray) -> np.ndarray:
    lateral = np.array([-direction[1], direction[0], 0.0], dtype=np.float64)
    return lateral / max(float(np.linalg.norm(lateral)), 1e-12)


def _obstacle(
    position: np.ndarray,
    relative_direction: np.ndarray,
    distance: float,
    radius: float,
    identifier: str,
    *,
    velocity: np.ndarray | None = None,
) -> SphericalObstacle:
    direction = np.asarray(relative_direction, dtype=np.float64)
    direction /= max(float(np.linalg.norm(direction)), 1e-12)
    relative_position = distance * direction
    return SphericalObstacle(
        center=position - relative_position,
        radius=radius,
        velocity=np.zeros(3, dtype=np.float64) if velocity is None else velocity,
        identifier=identifier,
    )


def generate_case(
    rng: np.random.Generator,
    state_index: int,
) -> tuple[str, np.ndarray, np.ndarray, float, list[SphericalObstacle]]:
    family = FAMILIES[state_index % len(FAMILIES)]
    position = np.array(
        [
            rng.uniform(600.0, 3400.0),
            rng.uniform(600.0, 3400.0),
            rng.uniform(100.0, 300.0),
        ],
        dtype=np.float64,
    )
    direction = _mostly_horizontal_direction(rng)
    lateral = _orthogonal_horizontal(direction)
    radius = float(rng.uniform(5.0, 35.0))
    safe_distance = radius + UAV_RADIUS
    hold_dt = 0.05
    obstacles: list[SphericalObstacle]

    if family == "high_closing_speed":
        speed = float(rng.uniform(12.0, HORIZONTAL_V_MAX))
        velocity = -speed * direction
        clearance = float(rng.uniform(0.02, 120.0))
        obstacles = [
            _obstacle(
                position,
                direction,
                safe_distance + clearance,
                radius,
                "high_closing",
            )
        ]
    elif family == "low_clearance":
        speed = float(rng.uniform(0.0, 15.0))
        velocity = -speed * direction
        clearance = float(rng.uniform(0.005, 5.0))
        obstacles = [
            _obstacle(
                position,
                direction,
                safe_distance + clearance,
                radius,
                "low_clearance",
            )
        ]
    elif family == "moving_multi_obstacle_conflict":
        velocity = rng.uniform(
            [-8.0, -8.0, -2.0],
            [8.0, 8.0, 2.0],
        )
        obstacles = []
        count = int(rng.integers(2, 5))
        for obstacle_index in range(count):
            angle = 2.0 * np.pi * obstacle_index / count + rng.uniform(-0.2, 0.2)
            normal = (
                np.cos(angle) * direction
                + np.sin(angle) * lateral
                + rng.uniform(-0.2, 0.2) * np.array([0.0, 0.0, 1.0])
            )
            normal /= max(float(np.linalg.norm(normal)), 1e-12)
            obstacle_radius = float(rng.uniform(8.0, 30.0))
            clearance = float(rng.uniform(0.05, 35.0))
            closing_speed = float(rng.uniform(5.0, 20.0))
            obstacle_velocity = velocity + closing_speed * normal
            obstacles.append(
                _obstacle(
                    position,
                    normal,
                    obstacle_radius + UAV_RADIUS + clearance,
                    obstacle_radius,
                    f"moving_conflict_{obstacle_index}",
                    velocity=obstacle_velocity,
                )
            )
    elif family == "narrow_passage":
        velocity = float(rng.uniform(8.0, 20.0)) * direction
        side_radius = float(rng.uniform(15.0, 35.0))
        side_clearance = float(rng.uniform(0.05, 4.0))
        forward_radius = float(rng.uniform(10.0, 25.0))
        obstacles = [
            _obstacle(
                position,
                lateral,
                side_radius + UAV_RADIUS + side_clearance,
                side_radius,
                "narrow_left",
                velocity=velocity + rng.uniform(1.0, 8.0) * lateral,
            ),
            _obstacle(
                position,
                -lateral,
                side_radius + UAV_RADIUS + side_clearance,
                side_radius,
                "narrow_right",
                velocity=velocity - rng.uniform(1.0, 8.0) * lateral,
            ),
            _obstacle(
                position,
                -direction,
                forward_radius + UAV_RADIUS + rng.uniform(5.0, 60.0),
                forward_radius,
                "narrow_forward",
            ),
        ]
    elif family == "horizontal_velocity_saturation":
        direction[2] = 0.0
        direction /= max(float(np.linalg.norm(direction)), 1e-12)
        velocity = -float(rng.uniform(19.0, HORIZONTAL_V_MAX)) * direction
        obstacles = [
            _obstacle(
                position,
                direction,
                safe_distance + rng.uniform(0.05, 100.0),
                radius,
                "horizontal_saturation",
            )
        ]
    elif family == "vertical_velocity_saturation":
        sign = float(rng.choice((-1.0, 1.0)))
        normal = np.array([0.0, 0.0, -sign], dtype=np.float64)
        velocity = np.array(
            [rng.uniform(-2.0, 2.0), rng.uniform(-2.0, 2.0), sign * rng.uniform(4.8, 5.0)],
            dtype=np.float64,
        )
        obstacles = [
            _obstacle(
                position,
                normal,
                safe_distance + rng.uniform(0.05, 50.0),
                radius,
                "vertical_saturation",
            )
        ]
    else:
        speed = float(rng.uniform(10.0, HORIZONTAL_V_MAX))
        velocity = -speed * direction
        hold_dt = float(rng.uniform(0.05, 0.20))
        obstacles = [
            _obstacle(
                position,
                direction,
                safe_distance + rng.uniform(0.02, 100.0),
                radius,
                "sample_delay",
            )
        ]

    horizontal_speed = float(np.linalg.norm(velocity[:2]))
    if horizontal_speed > HORIZONTAL_V_MAX:
        velocity[:2] *= HORIZONTAL_V_MAX / horizontal_speed
    velocity[2] = float(np.clip(velocity[2], -VERTICAL_V_MAX, VERTICAL_V_MAX))
    return family, position, velocity, hold_dt, obstacles


def evaluate_case(
    state_index: int,
    family: str,
    position: np.ndarray,
    velocity: np.ndarray,
    hold_dt: float,
    obstacles: list[SphericalObstacle],
) -> HardStateRecord:
    continuous_constraints = [
        sphere_hocbf_constraint(position, velocity, obstacle, HOCBF)
        for obstacle in obstacles
    ]
    sampled_constraints = [
        strengthen_constraint_for_sample_hold(
            constraint,
            hocbf_sampled_data_residual_bound(
                position,
                velocity,
                obstacle,
                HOCBF,
                hold_dt=hold_dt,
                horizontal_acceleration_limit=HORIZONTAL_A_MAX,
                vertical_acceleration_limit=VERTICAL_A_MAX,
            ),
        )
        for constraint, obstacle in zip(continuous_constraints, obstacles)
    ]
    continuous = joint_feasibility_margin(
        np.stack([constraint.row for constraint in continuous_constraints]),
        np.asarray([constraint.lower_bound for constraint in continuous_constraints]),
        HORIZONTAL_A_MAX,
        VERTICAL_A_MAX,
    )
    sampled = joint_feasibility_margin(
        np.stack([constraint.row for constraint in sampled_constraints]),
        np.asarray([constraint.lower_bound for constraint in sampled_constraints]),
        HORIZONTAL_A_MAX,
        VERTICAL_A_MAX,
    )
    if len(sampled_constraints) > 1 and (
        not sampled.dual_success or sampled.duality_gap > 1e-6
    ):
        sampled = joint_feasibility_margin(
            np.stack([constraint.row for constraint in sampled_constraints]),
            np.asarray([constraint.lower_bound for constraint in sampled_constraints]),
            HORIZONTAL_A_MAX,
            VERTICAL_A_MAX,
            global_certificate_fallback=True,
        )
    return HardStateRecord(
        state_index=state_index,
        family=family,
        position=position.tolist(),
        velocity=velocity.tolist(),
        hold_dt=float(hold_dt),
        obstacles=[
            {
                "identifier": obstacle.identifier,
                "center": obstacle.center.tolist(),
                "radius": obstacle.radius,
                "velocity": obstacle.velocity.tolist(),
            }
            for obstacle in obstacles
        ],
        minimum_h=float(min(constraint.h for constraint in continuous_constraints)),
        minimum_psi1=float(min(constraint.psi1 for constraint in continuous_constraints)),
        continuous_margin=continuous.margin,
        sampled_margin=sampled.margin,
        continuous_feasible=continuous.feasible,
        sampled_feasible=sampled.feasible,
        maximizing_action=sampled.maximizing_action.tolist(),
        primal_dual_gap=sampled.duality_gap,
        solve_seconds=continuous.solve_seconds + sampled.solve_seconds,
        global_certificate_fallback_used=sampled.global_certificate_fallback_used,
    )


def _worker(
    worker_index: int,
    start: int,
    count: int,
    seed: int,
    keep: int,
) -> dict[str, object]:
    rng = np.random.default_rng(seed + 1_000_003 * worker_index)
    hardest: list[tuple[float, int, HardStateRecord]] = []
    family_counts = {
        family: {
            "states": 0,
            "continuous_infeasible": 0,
            "sampled_infeasible": 0,
            "continuous_feasible_sampled_infeasible": 0,
            "hocbf_domain_sampled_infeasible": 0,
        }
        for family in FAMILIES
    }
    continuous_margins = np.empty(count, dtype=np.float64)
    sampled_margins = np.empty(count, dtype=np.float64)
    solve_seconds = np.empty(count, dtype=np.float64)
    global_fallbacks = 0
    for local_index in range(count):
        state_index = start + local_index
        family, position, velocity, hold_dt, obstacles = generate_case(rng, state_index)
        record = evaluate_case(
            state_index,
            family,
            position,
            velocity,
            hold_dt,
            obstacles,
        )
        continuous_margins[local_index] = record.continuous_margin
        sampled_margins[local_index] = record.sampled_margin
        solve_seconds[local_index] = record.solve_seconds
        global_fallbacks += int(record.global_certificate_fallback_used)
        stats = family_counts[family]
        stats["states"] += 1
        stats["continuous_infeasible"] += int(not record.continuous_feasible)
        stats["sampled_infeasible"] += int(not record.sampled_feasible)
        stats["continuous_feasible_sampled_infeasible"] += int(
            record.continuous_feasible and not record.sampled_feasible
        )
        stats["hocbf_domain_sampled_infeasible"] += int(
            record.minimum_h >= 0.0
            and record.minimum_psi1 >= 0.0
            and not record.sampled_feasible
        )
        item = (-record.sampled_margin, state_index, record)
        if len(hardest) < keep:
            heapq.heappush(hardest, item)
        elif record.sampled_margin < -hardest[0][0]:
            heapq.heapreplace(hardest, item)
    records = sorted(
        (item[2] for item in hardest),
        key=lambda item: item.sampled_margin,
    )
    return {
        "records": [asdict(item) for item in records[:keep]],
        "continuous_margins": continuous_margins,
        "sampled_margins": sampled_margins,
        "solve_seconds": solve_seconds,
        "family_counts": family_counts,
        "global_fallbacks": global_fallbacks,
    }


def _merge_family_counts(results: list[dict[str, object]]) -> dict[str, dict[str, int]]:
    merged = {
        family: {
            "states": 0,
            "continuous_infeasible": 0,
            "sampled_infeasible": 0,
            "continuous_feasible_sampled_infeasible": 0,
            "hocbf_domain_sampled_infeasible": 0,
        }
        for family in FAMILIES
    }
    for result in results:
        family_counts = result["family_counts"]
        for family in FAMILIES:
            for key in merged[family]:
                merged[family][key] += int(family_counts[family][key])
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", type=int, default=100_000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--keep-hardest", type=int, default=1000)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.states <= 0 or args.workers <= 0 or args.keep_hardest <= 0:
        raise ValueError("states, workers, and keep-hardest must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = perf_counter()

    base = args.states // args.workers
    remainder = args.states % args.workers
    jobs = []
    start = 0
    for worker_index in range(args.workers):
        count = base + int(worker_index < remainder)
        jobs.append((worker_index, start, count, args.seed, args.keep_hardest))
        start += count
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(_worker, *job) for job in jobs]
        results = [future.result() for future in futures]

    continuous_margins = np.concatenate(
        [np.asarray(result["continuous_margins"]) for result in results]
    )
    sampled_margins = np.concatenate(
        [np.asarray(result["sampled_margins"]) for result in results]
    )
    solve_seconds = np.concatenate(
        [np.asarray(result["solve_seconds"]) for result in results]
    )
    hard_records = [record for result in results for record in result["records"]]
    hard_records.sort(key=lambda item: float(item["sampled_margin"]))
    hard_records = hard_records[: args.keep_hardest]
    family_counts = _merge_family_counts(results)

    with (args.output_dir / "hard_states.jsonl").open("w") as handle:
        for record in hard_records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    method_audit = {
        "saved_hard_states": len(hard_records),
        "sampled_safe_action_set_nonempty": int(
            sum(float(record["sampled_margin"]) >= -1e-8 for record in hard_records)
        ),
        "sampled_safe_action_set_empty": int(
            sum(float(record["sampled_margin"]) < -1e-8 for record in hard_records)
        ),
        "standard_hocbf_certified_actions_available": int(
            sum(float(record["sampled_margin"]) >= -1e-8 for record in hard_records)
        ),
        "candidate_a_certified_actions_available": int(
            sum(float(record["sampled_margin"]) >= -1e-8 for record in hard_records)
        ),
        "lexicographic_candidate_certified_actions_available": int(
            sum(float(record["sampled_margin"]) >= -1e-8 for record in hard_records)
        ),
        "reason": (
            "all three selectors share the same sampled-data hard feasible set; "
            "an energy objective cannot create an action when the exact joint margin is negative"
        ),
        "scope_limit": (
            "static feasibility audit only; it does not fabricate rollout collision or task-success outcomes"
        ),
    }
    (args.output_dir / "method_audit.json").write_text(
        json.dumps(method_audit, indent=2, sort_keys=True) + "\n"
    )

    quantile_levels = (0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "physical adversarial initial-state search for HOCBF feasibility",
        "states": args.states,
        "seed": args.seed,
        "workers": args.workers,
        "families": list(FAMILIES),
        "physical_model": {
            "world_size_m": WORLD_SIZE.tolist(),
            "horizontal_velocity_limit_mps": HORIZONTAL_V_MAX,
            "vertical_velocity_limit_mps": VERTICAL_V_MAX,
            "horizontal_acceleration_limit_mps2": HORIZONTAL_A_MAX,
            "vertical_acceleration_limit_mps2": VERTICAL_A_MAX,
            "uav_radius_m": UAV_RADIUS,
            "hocbf_k1_per_s": HOCBF.k1,
            "hocbf_k2_per_s": HOCBF.k2,
        },
        "continuous_infeasible": int(np.sum(continuous_margins < -1e-8)),
        "sampled_infeasible": int(np.sum(sampled_margins < -1e-8)),
        "continuous_feasible_sampled_infeasible": int(
            np.sum((continuous_margins >= -1e-8) & (sampled_margins < -1e-8))
        ),
        "continuous_margin_quantiles": {
            str(level): float(np.quantile(continuous_margins, level))
            for level in quantile_levels
        },
        "sampled_margin_quantiles": {
            str(level): float(np.quantile(sampled_margins, level))
            for level in quantile_levels
        },
        "family_counts": family_counts,
        "global_certificate_fallbacks": int(
            sum(int(result["global_fallbacks"]) for result in results)
        ),
        "solve_timing_ms": {
            "mean": float(np.mean(solve_seconds) * 1e3),
            "p95": float(np.quantile(solve_seconds, 0.95) * 1e3),
            "p99": float(np.quantile(solve_seconds, 0.99) * 1e3),
            "maximum": float(np.max(solve_seconds) * 1e3),
            "deadline_misses_50ms": int(np.sum(solve_seconds > 0.05)),
        },
        "hard_states_saved": len(hard_records),
        "hardest_sampled_margin": float(hard_records[0]["sampled_margin"]),
        "least_hard_saved_sampled_margin": float(hard_records[-1]["sampled_margin"]),
        "prior_34_state_reproduction": {
            "exact_replay_available": False,
            "reason": "the historical rollout artifact retained aggregate fallback counts but no per-step physical state trace",
            "replacement": "same physical model with targeted same-or-harder failure families and exact joint margins",
        },
        "wall_seconds": perf_counter() - started,
        "python": platform.python_version(),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    (args.output_dir / "config.json").write_text(
        json.dumps(vars(args) | {"output_dir": str(args.output_dir)}, indent=2, default=str)
        + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
