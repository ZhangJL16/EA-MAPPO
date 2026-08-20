from __future__ import annotations

import json
import hashlib
import platform
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from review_bundle.safety.collision.hocbf import SphericalObstacle
from review_bundle.safety.trajectory import (
    CandidateEvaluation,
    PhysicsRolloutConfig,
    ProposalConfig,
    certify_trajectory,
    generate_trajectory_proposals,
    rollout_action_sequences,
    rollout_action_sequences_torch,
    select_safe_trajectory,
    trajectory_energy,
    trajectory_progress,
)
from review_bundle.safety.trajectory.trajectory_proposals import ProposalSource


@dataclass(frozen=True)
class BenchmarkConfig:
    seed: int = 20260819
    initial_states: int = 100_000
    proposals_per_state: int = 1_000
    hard_states: int = 5
    hard_proposals_per_state: int = 10_000
    exact_reference_states: int = 50
    horizon: int = 5
    coarse_keep: int = 20
    chunk_states: int = 100
    output_dir: str = "artifacts/history_safe_trajectory_100k"

    def __post_init__(self) -> None:
        if self.initial_states != 100_000:
            raise ValueError("controlled protocol fixes initial_states=100000")
        if self.proposals_per_state < 1000:
            raise ValueError("each state requires at least 1000 coarse proposals")
        if self.hard_proposals_per_state < 10_000:
            raise ValueError("hard states require at least 10000 candidates")
        if self.horizon not in (5, 10, 20, 40):
            raise ValueError("unsupported horizon")
        if self.exact_reference_states <= 0 or self.coarse_keep <= 0:
            raise ValueError("reference and survivor counts must be positive")


def _quantiles(values: list[float] | np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {name: float("nan") for name in ("mean", "p50", "p90", "p95", "p99", "max")}
    return {
        "mean": float(np.mean(array)),
        "p50": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "max": float(np.max(array)),
    }


def _sample_states(
    rng: np.random.Generator,
    count: int,
    *,
    hard: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    directions = rng.normal(size=(count, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    radii = rng.uniform(2.0, 8.0, size=count)
    clearance = rng.uniform(4.0, 15.0, size=count) if hard else rng.uniform(
        0.25, 35.0, size=count
    )
    obstacle_centers = rng.uniform(
        np.array([300.0, 300.0, 80.0]),
        np.array([3700.0, 3700.0, 320.0]),
        size=(count, 3),
    )
    positions = obstacle_centers + directions * (radii + 0.5 + clearance)[:, None]
    closing = rng.uniform(3.0, 8.0, size=count) if hard else rng.uniform(0.0, 5.0, size=count)
    tangential = rng.normal(scale=1.0 if hard else 0.5, size=(count, 3))
    tangential -= np.sum(tangential * directions, axis=1, keepdims=True) * directions
    velocities = -directions * closing[:, None] + tangential
    if not hard:
        speed = np.linalg.norm(velocities[:, :2], axis=1)
        velocities[:, :2] *= np.minimum(1.0, 5.0 / np.maximum(speed, 1e-12))[:, None]
        velocities[:, 2] = np.clip(velocities[:, 2], -2.0, 2.0)
    if hard:
        tangent = rng.normal(size=(count, 3))
        tangent -= np.sum(tangent * directions, axis=1, keepdims=True) * directions
        tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-12)
        goal_offset = (
            tangent * rng.uniform(50.0, 120.0, size=(count, 1))
            - directions * rng.uniform(5.0, 20.0, size=(count, 1))
        )
    else:
        goal_offset = -directions * rng.uniform(25.0, 120.0, size=(count, 1))
        goal_offset += rng.normal(scale=15.0, size=(count, 3))
    goals = np.clip(
        obstacle_centers + goal_offset,
        np.array([10.0, 10.0, 10.0]),
        np.array([3990.0, 3990.0, 390.0]),
    )
    return positions, velocities, goals, obstacle_centers, radii


def _clip_actions(actions: np.ndarray) -> np.ndarray:
    result = np.asarray(actions, dtype=np.float32).copy()
    horizontal = np.linalg.norm(result[..., :2], axis=-1)
    result[..., :2] *= np.minimum(1.0, 5.0 / np.maximum(horizontal, 1e-12))[..., None]
    result[..., 2] = np.clip(result[..., 2], -3.0, 3.0)
    return result


def _mass_proposals(
    rng: np.random.Generator,
    positions: np.ndarray,
    velocities: np.ndarray,
    goals: np.ndarray,
    count: int,
    horizon: int,
) -> np.ndarray:
    delta = goals - positions
    delta /= np.maximum(np.linalg.norm(delta, axis=1, keepdims=True), 1e-12)
    nominal = 2.0 * delta - 0.2 * velocities
    noise = rng.normal(0.0, 1.2, size=(positions.shape[0], count, horizon, 3)).astype(np.float32)
    correlated = np.cumsum(noise, axis=2) / np.sqrt(
        np.arange(1, horizon + 1, dtype=np.float32)[None, None, :, None]
    )
    actions = nominal[:, None, None, :].astype(np.float32) + correlated
    actions[:, 0, :, :] = 0.0
    braking = -velocities / np.maximum(np.linalg.norm(velocities, axis=1, keepdims=True), 1e-12)
    actions[:, 1, :, :] = (5.0 * braking)[:, None, :]
    actions[:, 2, :, :] = nominal[:, None, :]
    return _clip_actions(actions)


def _cheap_rollout(
    positions: np.ndarray,
    velocities: np.ndarray,
    actions: np.ndarray,
    *,
    dt: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    position = np.broadcast_to(positions[:, None, :], actions.shape[:2] + (3,)).copy()
    velocity = np.broadcast_to(velocities[:, None, :], actions.shape[:2] + (3,)).copy()
    paths = np.empty(actions.shape[:3] + (3,), dtype=np.float32)
    for step in range(actions.shape[2]):
        acceleration = actions[:, :, step]
        position = position + velocity * dt + 0.5 * acceleration * dt**2
        velocity = velocity + acceleration * dt
        paths[:, :, step] = position
    return paths, velocity


def _coarse_metrics(
    paths: np.ndarray,
    initial_positions: np.ndarray,
    goals: np.ndarray,
    obstacle_centers: np.ndarray,
    radii: np.ndarray,
    keep: int,
) -> dict[str, np.ndarray]:
    clearance = np.linalg.norm(
        paths - obstacle_centers[:, None, None, :], axis=3
    ) - (radii + 0.5)[:, None, None]
    minimum_clearance = np.min(clearance, axis=2)
    initial_distance = np.linalg.norm(initial_positions - goals, axis=1)
    terminal_distance = np.linalg.norm(paths[:, :, -1] - goals[:, None, :], axis=2)
    progress = initial_distance[:, None] - terminal_distance
    safe_progress = (minimum_clearance >= 0.0) & (progress > 0.0)
    score = minimum_clearance + 0.1 * progress
    keep = min(keep, score.shape[1])
    top = np.argpartition(score, -keep, axis=1)[:, -keep:]
    survivor_safe = np.take_along_axis(safe_progress, top, axis=1)
    return {
        "minimum_clearance": minimum_clearance,
        "progress": progress,
        "safe_progress": safe_progress,
        "top_indices": top,
        "reference_exists": np.any(safe_progress, axis=1),
        "pipeline_exists": np.any(survivor_safe, axis=1),
    }


def _run_massive_coarse(config: BenchmarkConfig, rng: np.random.Generator) -> dict[str, object]:
    state_count = 0
    reference_count = 0
    pipeline_count = 0
    safe_fraction_sum = 0.0
    diversity_sum = 0.0
    coarse_times: list[float] = []
    while state_count < config.initial_states:
        count = min(config.chunk_states, config.initial_states - state_count)
        positions, velocities, goals, centers, radii = _sample_states(rng, count, hard=False)
        started = perf_counter()
        actions = _mass_proposals(
            rng, positions, velocities, goals, config.proposals_per_state, config.horizon
        )
        paths, _ = _cheap_rollout(positions, velocities, actions)
        metrics = _coarse_metrics(
            paths, positions, goals, centers, radii, config.coarse_keep
        )
        coarse_times.append((perf_counter() - started) * 1000.0 / count)
        reference_count += int(np.sum(metrics["reference_exists"]))
        pipeline_count += int(np.sum(metrics["pipeline_exists"]))
        safe_fraction_sum += float(np.sum(np.mean(metrics["safe_progress"], axis=1)))
        diversity_sum += float(np.sum(np.mean(np.std(actions[:, :, 0], axis=1), axis=1)))
        state_count += count
    safe_recall = pipeline_count / max(reference_count, 1)
    return {
        "initial_states": state_count,
        "proposals_per_state": config.proposals_per_state,
        "total_coarse_sequences": state_count * config.proposals_per_state,
        "reference_safe_candidate_states": reference_count,
        "pipeline_safe_candidate_states": pipeline_count,
        "coarse_safe_recall": safe_recall,
        "mean_safe_candidate_fraction": safe_fraction_sum / state_count,
        "mean_first_action_diversity": diversity_sum / state_count,
        "per_state_coarse_latency_ms": _quantiles(coarse_times),
    }


def _candidate_batch(
    rng: np.random.Generator,
    position: np.ndarray,
    velocity: np.ndarray,
    goal: np.ndarray,
    center: np.ndarray,
    count: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    nominal_delta = goal - position
    nominal_delta /= max(float(np.linalg.norm(nominal_delta)), 1e-12)
    proposal = generate_trajectory_proposals(
        position=position,
        velocity=velocity,
        goal=goal,
        nominal_action=2.0 * nominal_delta - 0.2 * velocity,
        config=ProposalConfig(horizon=horizon, count=count),
        rng=rng,
        obstacles=center[None, :],
    )
    random_mask = np.asarray(
        [source is ProposalSource.RANDOM_SAC for source in proposal.sources], dtype=bool
    )
    return proposal.action_sequences, random_mask


def _evaluate_state(
    rng: np.random.Generator,
    position: np.ndarray,
    velocity: np.ndarray,
    goal: np.ndarray,
    center: np.ndarray,
    radius: float,
    *,
    count: int,
    horizon: int,
    coarse_keep: int,
    rollout_config: PhysicsRolloutConfig,
) -> dict[str, object]:
    timing: dict[str, float] = {}
    started = perf_counter()
    actions, random_mask = _candidate_batch(
        rng, position, velocity, goal, center, count, horizon
    )
    timing["proposal_ms"] = (perf_counter() - started) * 1000.0
    started = perf_counter()
    approximate_paths, _ = _cheap_rollout(
        position[None, :], velocity[None, :], actions[None, :, :, :]
    )
    coarse = _coarse_metrics(
        approximate_paths,
        position[None, :],
        goal[None, :],
        center[None, :],
        np.array([radius]),
        coarse_keep,
    )
    top = coarse["top_indices"][0]
    timing["coarse_ms"] = (perf_counter() - started) * 1000.0
    started = perf_counter()
    rollout = rollout_action_sequences(position, velocity, actions, rollout_config)
    timing["physics_all_ms"] = (perf_counter() - started) * 1000.0
    obstacle = SphericalObstacle(center=center, radius=float(radius))
    started = perf_counter()
    reference_certificate = certify_trajectory(
        rollout, [obstacle], rollout_config, uav_radius=0.5
    )
    timing["exact_reference_ms"] = (perf_counter() - started) * 1000.0
    progress = trajectory_progress(rollout.positions, rollout.velocities, goal)
    safe_progress = reference_certificate.certified & progress.admissible
    reference_exists = bool(np.any(safe_progress))
    b3_random_exists = bool(np.any(safe_progress & random_mask))
    started = perf_counter()
    selected_rollout = rollout_action_sequences(
        position, velocity, actions[top], rollout_config
    )
    timing["physics_survivor_ms"] = (perf_counter() - started) * 1000.0
    started = perf_counter()
    selected_certificate = certify_trajectory(
        selected_rollout, [obstacle], rollout_config, uav_radius=0.5
    )
    timing["exact_survivor_ms"] = (perf_counter() - started) * 1000.0
    selected_progress = trajectory_progress(
        selected_rollout.positions, selected_rollout.velocities, goal
    )
    selected_energy = trajectory_energy(selected_rollout, goal)
    evaluation = CandidateEvaluation(
        action_sequences=actions[top],
        certified=selected_certificate.certified,
        progress_admissible=selected_progress.admissible,
        progress_decrease=selected_progress.decrease,
        energy_upper=selected_energy.total_upper,
        minimum_clearance=selected_certificate.minimum_clearance,
    )
    selection = select_safe_trajectory(evaluation, actions[1])
    pipeline_exists = not selection.used_fallback
    exact_safe_indices = np.flatnonzero(safe_progress)
    selected_set = set(int(value) for value in top)
    false_pruned = sum(int(value) not in selected_set for value in exact_safe_indices)
    timing["b5_total_ms"] = (
        timing["proposal_ms"]
        + timing["coarse_ms"]
        + timing["physics_survivor_ms"]
        + timing["exact_survivor_ms"]
    )
    return {
        "b3_safe_mppi_random_exists": b3_random_exists,
        "b4_mixture_exact_exists": reference_exists,
        "b5_coarse_to_fine_exists": pipeline_exists,
        "reference_exists": reference_exists,
        "pipeline_exists": pipeline_exists,
        "exact_safe_progress_count": int(exact_safe_indices.size),
        "coarse_survivor_count": int(len(top)),
        "final_certified_count": int(np.sum(selected_certificate.certified)),
        "final_progress_count": int(
            np.sum(selected_certificate.certified & selected_progress.admissible)
        ),
        "false_pruned_safe_candidates": int(false_pruned),
        "false_prune_fraction_of_safe_candidates": (
            float(false_pruned / exact_safe_indices.size) if exact_safe_indices.size else 0.0
        ),
        "used_fallback": bool(selection.used_fallback),
        "uncertified_fallback": bool(selection.used_fallback),
        "certified_selection": bool(not selection.used_fallback),
        "state_level_false_prune": bool(reference_exists and not pipeline_exists),
        "timing_ms": timing,
    }


def _run_exact_reference(
    config: BenchmarkConfig,
    rng: np.random.Generator,
    *,
    hard: bool,
    states: int,
    proposals: int,
) -> dict[str, object]:
    rollout_config = PhysicsRolloutConfig(
        policy_dt=0.2,
        physics_dt=0.05,
        horizontal_speed_limit=20.0,
        vertical_speed_limit=5.0,
        horizontal_acceleration_limit=5.0,
        vertical_acceleration_limit=3.0,
        world_min=np.zeros(3),
        world_max=np.array([4000.0, 4000.0, 400.0]),
        body_radius=0.5,
    )
    rows: list[dict[str, object]] = []
    sampled = _sample_states(rng, states, hard=hard)
    for index in range(states):
        rows.append(
            _evaluate_state(
                rng,
                sampled[0][index],
                sampled[1][index],
                sampled[2][index],
                sampled[3][index],
                float(sampled[4][index]),
                count=proposals,
                horizon=config.horizon,
                coarse_keep=config.coarse_keep,
                rollout_config=rollout_config,
            )
        )
    reference_positive = sum(bool(row["reference_exists"]) for row in rows)
    pipeline_positive = sum(
        bool(row["reference_exists"]) and bool(row["pipeline_exists"]) for row in rows
    )
    b3_positive = sum(
        bool(row["reference_exists"]) and bool(row["b3_safe_mppi_random_exists"])
        for row in rows
    )
    timing_names = next(iter(rows))["timing_ms"].keys() if rows else []
    timings = {
        name: _quantiles([float(row["timing_ms"][name]) for row in rows])
        for name in timing_names
    }
    b5 = [float(row["timing_ms"]["b5_total_ms"]) for row in rows]
    return {
        "hard": hard,
        "states": states,
        "proposals_per_state": proposals,
        "reference_positive_states": reference_positive,
        "pipeline_positive_states": pipeline_positive,
        "method_recall": {
            "B3_safe_mppi_random_exact": b3_positive / max(reference_positive, 1),
            "B4_mixture_exact_all": 1.0 if reference_positive else 0.0,
            "B5_mixture_coarse_to_fine": pipeline_positive / max(reference_positive, 1),
        },
        "safe_candidate_recall": pipeline_positive / max(reference_positive, 1),
        "fallback_rate": float(np.mean([row["used_fallback"] for row in rows])),
        "uncertified_fallbacks": int(sum(row["uncertified_fallback"] for row in rows)),
        "certified_selections": int(sum(row["certified_selection"] for row in rows)),
        "mean_final_certified_count": float(
            np.mean([row["final_certified_count"] for row in rows])
        ),
        "mean_final_progress_count": float(
            np.mean([row["final_progress_count"] for row in rows])
        ),
        "mean_false_prune_fraction_of_safe_candidates": float(
            np.mean([row["false_prune_fraction_of_safe_candidates"] for row in rows])
        ),
        "state_level_false_prune_rate": float(
            np.mean([row["state_level_false_prune"] for row in rows])
        ),
        "latency_ms": timings,
        "deadline_miss_rate_50ms": float(np.mean(np.asarray(b5) > 50.0)),
        "rows": rows,
    }


def _run_backend_latency(
    config: BenchmarkConfig,
    rng: np.random.Generator,
) -> dict[str, object]:
    rollout_config = PhysicsRolloutConfig()
    results: dict[str, object] = {}
    for count in (1_000, 10_000, 100_000):
        actions = rng.normal(size=(count, config.horizon, 3)).astype(np.float64)
        repeats = 10 if count == 1_000 else 3 if count == 10_000 else 1
        numpy_times = []
        torch_cpu_times = []
        torch_cuda_times = []
        for _ in range(repeats):
            started = perf_counter()
            rollout_action_sequences(np.zeros(3), np.zeros(3), actions, rollout_config)
            numpy_times.append((perf_counter() - started) * 1000.0)
            started = perf_counter()
            rollout_action_sequences_torch(
                np.zeros(3), np.zeros(3), actions, rollout_config, device="cpu"
            )
            torch_cpu_times.append((perf_counter() - started) * 1000.0)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                started = perf_counter()
                rollout_action_sequences_torch(
                    np.zeros(3), np.zeros(3), actions, rollout_config, device="cuda"
                )
                torch.cuda.synchronize()
                torch_cuda_times.append((perf_counter() - started) * 1000.0)
        results[str(count)] = {
            "numpy_vectorized_e2e_ms": _quantiles(numpy_times),
            "torch_cpu_e2e_ms": _quantiles(torch_cpu_times),
            "torch_cuda_e2e_including_transfer_ms": _quantiles(torch_cuda_times),
        }
    return results


def run_benchmark(config: BenchmarkConfig) -> dict[str, object]:
    output = Path(config.output_dir)
    if output.exists():
        raise FileExistsError(f"output directory already exists: {output}")
    output.mkdir(parents=True)
    (output / "config.json").write_text(
        json.dumps(asdict(config), indent=2, sort_keys=True), encoding="utf-8"
    )
    rng = np.random.default_rng(config.seed)
    started = perf_counter()
    massive = _run_massive_coarse(config, rng)
    exact = _run_exact_reference(
        config,
        rng,
        hard=False,
        states=config.exact_reference_states,
        proposals=config.proposals_per_state,
    )
    hard = _run_exact_reference(
        config,
        rng,
        hard=True,
        states=config.hard_states,
        proposals=config.hard_proposals_per_state,
    )
    backends = _run_backend_latency(config, rng)
    source_files = sorted(
        list(Path("review_bundle/safety/trajectory").glob("*.py"))
        + [Path(__file__), Path("scripts/run_history_safe_trajectory_benchmark.py")]
    )
    digest = hashlib.sha256()
    for path in source_files:
        digest.update(path.as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        git_sha = None
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "formal_500k": False,
        "git_sha": git_sha,
        "trajectory_code_sha256": digest.hexdigest(),
        "config": asdict(config),
        "massive_coarse_survey": massive,
        "exact_reference": {key: value for key, value in exact.items() if key != "rows"},
        "hard_reference": {key: value for key, value in hard.items() if key != "rows"},
        "backend_latency": backends,
        "wall_seconds": float(perf_counter() - started),
    }
    (output / "exact_reference_rows.json").write_text(
        json.dumps(exact["rows"], indent=2), encoding="utf-8"
    )
    (output / "hard_reference_rows.json").write_text(
        json.dumps(hard["rows"], indent=2), encoding="utf-8"
    )
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary
