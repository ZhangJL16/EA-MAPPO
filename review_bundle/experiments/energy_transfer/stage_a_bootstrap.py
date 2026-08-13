from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
import platform
import shlex
import sys
import traceback
from typing import Protocol

import numpy as np
import torch
from torch.nn import functional

from agents.goal_conditioned_sac import FrozenGoalConditionedSAC
from envs.navigation import NavigationEnv, OperationalEnergyConfig, OperationalEnergyWrapper, load_scenario
from experiments.energy_transfer.analysis import (
    build_figures,
    coverage_report,
    learned_value_decision,
    paired_sortie_comparisons,
    quantile_metrics,
    regression_metrics,
    residual_relationships,
    safe_correlation,
)
from experiments.new_route.provenance import append_jsonl, code_hash, git_sha, sha256_file, utc_now, write_json
from safety.energy import (
    EnergyReturnEpisode,
    EnergyTransition,
    MonotoneQuantileCritic,
    ScalarEnergyCritic,
    quantile_huber_loss,
    quantile_ssp_target,
    returns_to_go,
    scalar_ssp_target,
)


ROOT = Path(__file__).resolve().parents[2]
QUANTILES = (0.50, 0.90, 0.95, 0.99)


class GoalPolicy(Protocol):
    def action(self, observation: np.ndarray, *, deterministic: bool = True) -> np.ndarray: ...


@dataclass(frozen=True)
class StageABootstrapConfig:
    seed: int
    checkpoint: str
    output_dir: str
    sorties: int = 60
    task_prefix_min_steps: int = 0
    task_prefix_max_steps: int = 80
    max_return_steps: int = 800
    operational_energy_capacity: float = 3.0
    training_updates: int = 800
    scenario: str = "random_persistent_open.json"
    device: str = "cpu"
    run_kind: str = "smoke"

    def __post_init__(self) -> None:
        if self.sorties < 10:
            raise ValueError("Stage A requires at least 10 sorties for a nonempty sortie-level split")
        if self.task_prefix_min_steps < 0 or self.task_prefix_max_steps < self.task_prefix_min_steps:
            raise ValueError("invalid task-prefix range")
        if self.max_return_steps <= 0 or self.training_updates <= 0:
            raise ValueError("return and training budgets must be positive")
        if not np.isfinite(self.operational_energy_capacity) or self.operational_energy_capacity <= 0.0:
            raise ValueError("operational energy capacity must be finite and positive")
        if self.run_kind not in {"smoke", "validation", "formal"}:
            raise ValueError("run_kind must be smoke, validation, or formal")
        if self.run_kind == "validation" and not 100 <= self.sorties <= 300:
            raise ValueError("validation runs require 100-300 sorties")
        if self.run_kind == "formal" and self.sorties < 300:
            raise ValueError("formal runs require at least 300 sorties")


@dataclass(frozen=True)
class BootstrapTrajectory:
    episode: EnergyReturnEpisode
    sortie_seed: int
    prefix_steps_planned: int
    prefix_steps_executed: int
    task_completions_during_prefix: int
    start_position: np.ndarray
    task_goal: np.ndarray
    charger_position: np.ndarray
    operational_energy_capacity: float
    operational_energy_at_reset: float
    operational_energy_at_commitment: float
    operational_soc_at_commitment: float
    prefix_energy: float
    prefix_path_length: float


@dataclass(frozen=True)
class EnergyDataset:
    features: np.ndarray
    next_features: np.ndarray
    costs: np.ndarray
    terminals: np.ndarray
    returns: np.ndarray
    distances: np.ndarray
    path_lengths_to_go: np.ndarray
    horizons: np.ndarray
    sortie_indices: np.ndarray
    sortie_seeds: np.ndarray
    velocity_magnitudes: np.ndarray
    velocity_alignment_to_charger: np.ndarray
    action_magnitudes: np.ndarray
    altitude_differences: np.ndarray
    prefix_lengths: np.ndarray
    positions: np.ndarray


def _prefix_schedule(config: StageABootstrapConfig) -> np.ndarray:
    values = np.rint(
        np.linspace(config.task_prefix_min_steps, config.task_prefix_max_steps, config.sorties)
    ).astype(int)
    return values[np.random.default_rng(config.seed + 4_001).permutation(config.sorties)]


def _sortie_seeds(config: StageABootstrapConfig) -> np.ndarray:
    return config.seed * 1_000_000 + 20_000 + np.arange(config.sorties, dtype=np.int64)


def collect_bootstrap_trajectory(
    policy: GoalPolicy,
    *,
    scenario: str,
    sortie_seed: int,
    prefix_steps: int,
    max_return_steps: int,
    operational_energy_capacity: float,
    policy_hash: str,
) -> BootstrapTrajectory:
    environment = NavigationEnv(scenario, max_episode_steps=prefix_steps + max_return_steps + 2)
    wrapped = OperationalEnergyWrapper(
        environment,
        OperationalEnergyConfig(capacity=operational_energy_capacity, enforce_exhaustion=False),
    )
    observation, reset_info = wrapped.reset(
        seed=sortie_seed,
        options={"randomize_station": True, "operational_initial_energy": operational_energy_capacity},
    )
    start_position = environment.state.position.copy()
    task_goal = environment.goal.copy()
    charger_position = environment.station_position.copy()
    prefix_energy = 0.0
    prefix_path_length = 0.0
    prefix_executed = 0
    tasks_before = environment.tasks_completed
    for _ in range(prefix_steps):
        position_before = environment.state.position.copy()
        action = policy.action(observation, deterministic=True)
        observation, _, terminated, truncated, info = wrapped.step(action)
        prefix_energy += float(info["operational_energy_step_cost"])
        prefix_path_length += float(np.linalg.norm(environment.state.position - position_before))
        prefix_executed += 1
        if terminated or truncated:
            break

    commitment_energy = wrapped.operational_energy
    commitment_soc = wrapped.operational_soc
    task_completions_during_prefix = environment.tasks_completed - tasks_before
    observation, _ = environment.set_external_goal(charger_position)
    transitions: list[EnergyTransition] = []
    completed = False
    for _ in range(max_return_steps):
        action = policy.action(observation, deterministic=True)
        next_observation, _, terminated, truncated, info = wrapped.step(action)
        charger_hit = bool(info["distance_to_goal_after"] <= environment.goal_radius + 1e-12)
        transitions.append(
            EnergyTransition(
                observation.copy(),
                action.copy(),
                float(info["operational_energy_step_cost"]),
                next_observation.copy(),
                charger_hit,
            )
        )
        observation = next_observation
        completed = charger_hit
        if completed or terminated or truncated:
            break
    wrapped.close()
    episode = EnergyReturnEpisode(
        transitions=tuple(transitions),
        completed=completed,
        policy_hash=policy_hash,
        context={
            "scenario": scenario,
            "sortie_seed": float(sortie_seed),
            "task_prefix_steps": float(prefix_executed),
        },
    )
    return BootstrapTrajectory(
        episode=episode,
        sortie_seed=sortie_seed,
        prefix_steps_planned=prefix_steps,
        prefix_steps_executed=prefix_executed,
        task_completions_during_prefix=task_completions_during_prefix,
        start_position=start_position,
        task_goal=task_goal,
        charger_position=charger_position,
        operational_energy_capacity=operational_energy_capacity,
        operational_energy_at_reset=float(reset_info["operational_energy_remaining"]),
        operational_energy_at_commitment=commitment_energy,
        operational_soc_at_commitment=commitment_soc,
        prefix_energy=prefix_energy,
        prefix_path_length=prefix_path_length,
    )


def trajectory_payload(
    index: int,
    trajectory: BootstrapTrajectory,
    world_size: np.ndarray,
    velocity_limit: np.ndarray | None = None,
) -> dict:
    episode = trajectory.episode
    velocity_scale = np.array([0.30, 0.30, 0.12]) if velocity_limit is None else np.asarray(velocity_limit)
    costs = np.asarray([transition.cost for transition in episode.transitions], dtype=np.float64)
    positions = np.asarray([transition.state[:3] for transition in episode.transitions]) * world_size
    next_positions = np.asarray([transition.next_state[:3] for transition in episode.transitions]) * world_size
    velocities = np.asarray([transition.state[3:6] for transition in episode.transitions]) * velocity_scale
    step_lengths = np.linalg.norm(next_positions - positions, axis=1)
    accumulated_energy = np.cumsum(costs)
    energy_before = trajectory.operational_energy_at_commitment - np.concatenate(([0.0], accumulated_energy[:-1]))
    energy_after = trajectory.operational_energy_at_commitment - accumulated_energy
    if episode.completed:
        return_values: list[float | None] = returns_to_go(costs).tolist()
        path_to_go: list[float | None] = np.cumsum(step_lengths[::-1])[::-1].tolist()
        total_return_energy: float | None = float(costs.sum())
        total_return_path: float | None = float(step_lengths.sum())
    else:
        return_values = [None] * len(episode.transitions)
        path_to_go = [None] * len(episode.transitions)
        total_return_energy = None
        total_return_path = None
    return {
        "sortie_id": index,
        "sortie_seed": trajectory.sortie_seed,
        "completed": episode.completed,
        "censored": not episode.completed,
        "censoring_reason": None if episode.completed else "charger_not_reached_within_max_return_steps",
        "policy_hash": episode.policy_hash,
        "scenario": episode.context["scenario"],
        "start_position": trajectory.start_position.tolist(),
        "task_goal": trajectory.task_goal.tolist(),
        "charger_position": trajectory.charger_position.tolist(),
        "return_start_position": positions[0].tolist(),
        "return_start_velocity": velocities[0].tolist(),
        "return_start_charger_distance": float(np.linalg.norm(positions[0] - trajectory.charger_position)),
        "task_prefix_steps_planned": trajectory.prefix_steps_planned,
        "task_prefix_steps_executed": trajectory.prefix_steps_executed,
        "task_completions_during_prefix": trajectory.task_completions_during_prefix,
        "prefix_energy": trajectory.prefix_energy,
        "prefix_path_length": trajectory.prefix_path_length,
        "operational_energy_capacity": trajectory.operational_energy_capacity,
        "operational_energy_at_reset": trajectory.operational_energy_at_reset,
        "operational_energy_at_commitment": trajectory.operational_energy_at_commitment,
        "operational_soc_at_commitment": trajectory.operational_soc_at_commitment,
        "return_transition_count": len(episode.transitions),
        "total_return_energy": total_return_energy,
        "total_return_path_length": total_return_path,
        "observed_partial_return_energy": float(costs.sum()),
        "observed_partial_return_path_length": float(step_lengths.sum()),
        "transitions": [
            {
                "sortie_id": index,
                "seed": trajectory.sortie_seed,
                "task_prefix_length": trajectory.prefix_steps_executed,
                "state": transition.state.tolist(),
                "action": transition.action.tolist(),
                "next_state": transition.next_state.tolist(),
                "velocity": velocities[row].tolist(),
                "per_step_energy": transition.cost,
                "accumulated_return_energy": float(accumulated_energy[row]),
                "return_energy_to_go": return_values[row],
                "operational_energy_before": float(max(0.0, energy_before[row])),
                "operational_energy_after": float(max(0.0, energy_after[row])),
                "operational_soc_before": float(
                    np.clip(energy_before[row] / trajectory.operational_energy_capacity, 0.0, 1.0)
                ),
                "distance_to_charger": float(np.linalg.norm(positions[row] - trajectory.charger_position)),
                "step_path_length": float(step_lengths[row]),
                "path_length_remaining": path_to_go[row],
                "charger_hit": transition.charger_hit,
            }
            for row, transition in enumerate(episode.transitions)
        ],
    }


def build_dataset(
    trajectories: list[BootstrapTrajectory],
    sortie_indices: np.ndarray,
    *,
    world_size: np.ndarray | None = None,
    velocity_limit: np.ndarray | None = None,
) -> EnergyDataset:
    world = np.array([4.0, 4.0, 2.0]) if world_size is None else np.asarray(world_size)
    velocity_scale = np.array([0.30, 0.30, 0.12]) if velocity_limit is None else np.asarray(velocity_limit)
    columns: dict[str, list] = {
        name: []
        for name in (
            "features",
            "next_features",
            "costs",
            "terminals",
            "returns",
            "distances",
            "path_lengths",
            "horizons",
            "sortie_indices",
            "sortie_seeds",
            "velocity_magnitudes",
            "velocity_alignment",
            "action_magnitudes",
            "altitude_differences",
            "prefix_lengths",
            "positions",
        )
    }
    for sortie_index in sortie_indices:
        trajectory = trajectories[int(sortie_index)]
        if not trajectory.episode.completed:
            continue
        episode = trajectory.episode
        episode_returns = returns_to_go(np.asarray([transition.cost for transition in episode.transitions]))
        positions = np.asarray([transition.state[:3] for transition in episode.transitions]) * world
        next_positions = np.asarray([transition.next_state[:3] for transition in episode.transitions]) * world
        path_to_go = np.cumsum(np.linalg.norm(next_positions - positions, axis=1)[::-1])[::-1]
        for row, transition in enumerate(episode.transitions):
            position = positions[row]
            velocity = transition.state[3:6] * velocity_scale
            charger_direction = trajectory.charger_position - position
            velocity_norm = float(np.linalg.norm(velocity))
            direction_norm = float(np.linalg.norm(charger_direction))
            alignment = 0.0 if velocity_norm <= 1e-12 or direction_norm <= 1e-12 else float(
                np.dot(velocity, charger_direction) / (velocity_norm * direction_norm)
            )
            next_action = np.zeros_like(transition.action) if transition.charger_hit else episode.transitions[row + 1].action
            columns["features"].append(np.concatenate((transition.state, transition.action)))
            columns["next_features"].append(np.concatenate((transition.next_state, next_action)))
            columns["costs"].append(transition.cost)
            columns["terminals"].append(transition.charger_hit)
            columns["returns"].append(float(episode_returns[row]))
            columns["distances"].append(direction_norm)
            columns["path_lengths"].append(float(path_to_go[row]))
            columns["horizons"].append(len(episode.transitions) - row)
            columns["sortie_indices"].append(int(sortie_index))
            columns["sortie_seeds"].append(trajectory.sortie_seed)
            columns["velocity_magnitudes"].append(velocity_norm)
            columns["velocity_alignment"].append(alignment)
            columns["action_magnitudes"].append(float(np.linalg.norm(transition.action)))
            columns["altitude_differences"].append(float(abs(charger_direction[2])))
            columns["prefix_lengths"].append(trajectory.prefix_steps_executed)
            columns["positions"].append(position)
    if not columns["features"]:
        raise RuntimeError("split contains no completed charger-return supervision")
    return EnergyDataset(
        features=np.asarray(columns["features"], dtype=np.float32),
        next_features=np.asarray(columns["next_features"], dtype=np.float32),
        costs=np.asarray(columns["costs"], dtype=np.float32),
        terminals=np.asarray(columns["terminals"], dtype=bool),
        returns=np.asarray(columns["returns"], dtype=np.float32),
        distances=np.asarray(columns["distances"], dtype=np.float32),
        path_lengths_to_go=np.asarray(columns["path_lengths"], dtype=np.float32),
        horizons=np.asarray(columns["horizons"], dtype=np.int32),
        sortie_indices=np.asarray(columns["sortie_indices"], dtype=np.int32),
        sortie_seeds=np.asarray(columns["sortie_seeds"], dtype=np.int64),
        velocity_magnitudes=np.asarray(columns["velocity_magnitudes"], dtype=np.float32),
        velocity_alignment_to_charger=np.asarray(columns["velocity_alignment"], dtype=np.float32),
        action_magnitudes=np.asarray(columns["action_magnitudes"], dtype=np.float32),
        altitude_differences=np.asarray(columns["altitude_differences"], dtype=np.float32),
        prefix_lengths=np.asarray(columns["prefix_lengths"], dtype=np.int32),
        positions=np.asarray(columns["positions"], dtype=np.float32),
    )


def split_sorties(count: int, seed: int) -> dict[str, np.ndarray]:
    indices = np.random.default_rng(seed + 8_003).permutation(count)
    train_end = max(1, int(0.60 * count))
    calibration_end = max(train_end + 1, int(0.80 * count))
    if calibration_end >= count:
        calibration_end = count - 1
    return {
        "train": indices[:train_end],
        "calibration": indices[train_end:calibration_end],
        "test": indices[calibration_end:],
    }


def _split_manifest(split: dict[str, np.ndarray], trajectories: list[BootstrapTrajectory], seed: int) -> dict:
    splits: dict[str, dict] = {}
    all_ids: list[int] = []
    for name, indices in split.items():
        selected = [trajectories[int(index)] for index in indices]
        all_ids.extend(int(index) for index in indices)
        splits[name] = {
            "sortie_ids": [int(index) for index in indices],
            "sortie_seeds": [item.sortie_seed for item in selected],
            "sortie_count": len(selected),
            "completed_sorties": sum(item.episode.completed for item in selected),
            "censored_sorties": sum(not item.episode.completed for item in selected),
            "observed_transition_count": sum(len(item.episode.transitions) for item in selected),
            "supervision_transition_count": sum(
                len(item.episode.transitions) for item in selected if item.episode.completed
            ),
        }
    return {
        "split_unit": "sortie",
        "split_seed": seed + 8_003,
        "split_fraction": [0.60, 0.20, 0.20],
        "splits": splits,
        "all_sortie_ids_unique": len(all_ids) == len(set(all_ids)),
        "all_sorties_assigned_exactly_once": set(all_ids) == set(range(len(trajectories))),
        "transition_leakage_possible_by_construction": False,
    }


def _loss_report(losses: list[float]) -> dict[str, float | bool]:
    values = np.asarray(losses, dtype=np.float64)
    return {
        "initial_loss": float(values[0]),
        "final_loss": float(values[-1]),
        "minimum_loss": float(values.min()),
        "maximum_loss": float(values.max()),
        "all_finite": bool(np.all(np.isfinite(values))),
    }


def _train_mc(dataset: EnergyDataset, updates: int) -> tuple[ScalarEnergyCritic, dict]:
    model = ScalarEnergyCritic(dataset.features.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    features = torch.from_numpy(dataset.features)
    labels = torch.from_numpy(dataset.returns)
    losses: list[float] = []
    for _ in range(updates):
        loss = functional.mse_loss(model(features), labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return model, _loss_report(losses)


def _train_scalar_td(dataset: EnergyDataset, updates: int) -> tuple[ScalarEnergyCritic, dict]:
    model = ScalarEnergyCritic(dataset.features.shape[1])
    target = ScalarEnergyCritic(dataset.features.shape[1])
    target.load_state_dict(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    features = torch.from_numpy(dataset.features)
    next_features = torch.from_numpy(dataset.next_features)
    costs = torch.from_numpy(dataset.costs)
    terminals = torch.from_numpy(dataset.terminals)
    losses: list[float] = []
    for update in range(updates):
        with torch.no_grad():
            labels = scalar_ssp_target(costs, target(next_features), terminals)
        loss = functional.mse_loss(model(features), labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
        if (update + 1) % 20 == 0:
            target.load_state_dict(model.state_dict())
    return model, _loss_report(losses)


def _train_quantile_td(dataset: EnergyDataset, updates: int) -> tuple[MonotoneQuantileCritic, dict]:
    model = MonotoneQuantileCritic(dataset.features.shape[1], QUANTILES)
    target = MonotoneQuantileCritic(dataset.features.shape[1], QUANTILES)
    target.load_state_dict(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    features = torch.from_numpy(dataset.features)
    next_features = torch.from_numpy(dataset.next_features)
    costs = torch.from_numpy(dataset.costs)
    terminals = torch.from_numpy(dataset.terminals)
    levels = torch.tensor(QUANTILES, dtype=torch.float32)
    losses: list[float] = []
    for update in range(updates):
        with torch.no_grad():
            labels = quantile_ssp_target(costs, target(next_features), terminals)
        loss = quantile_huber_loss(model(features), labels, levels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
        if (update + 1) % 20 == 0:
            target.load_state_dict(model.state_dict())
    return model, _loss_report(losses)


def _save_model(path: Path, model: torch.nn.Module, model_kind: str) -> str:
    torch.save(
        {
            "model_kind": model_kind,
            "input_dim": next(model.parameters()).shape[1],
            "quantile_levels": list(QUANTILES) if isinstance(model, MonotoneQuantileCritic) else None,
            "state_dict": model.state_dict(),
        },
        path,
    )
    return sha256_file(path)


def _cost_per_meter(trajectories: list[BootstrapTrajectory], indices: np.ndarray, world_size: np.ndarray) -> float:
    energy = 0.0
    path_length = 0.0
    for index in indices:
        trajectory = trajectories[int(index)]
        if not trajectory.episode.completed:
            continue
        episode = trajectory.episode
        positions = np.asarray([transition.state[:3] for transition in episode.transitions]) * world_size
        next_positions = np.asarray([transition.next_state[:3] for transition in episode.transitions]) * world_size
        energy += sum(transition.cost for transition in episode.transitions)
        path_length += float(np.linalg.norm(next_positions - positions, axis=1).sum())
    return energy / max(path_length, 1e-12)


def _predict(
    dataset: EnergyDataset,
    cost_per_meter: float,
    mc_model: ScalarEnergyCritic,
    scalar_td: ScalarEnergyCritic,
    quantile_td: MonotoneQuantileCritic,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    with torch.no_grad():
        features = torch.from_numpy(dataset.features)
        mc = mc_model(features).numpy()
        scalar = scalar_td(features).numpy()
        quantiles = quantile_td(features).numpy()
    return {
        "B0_distance": dataset.distances * cost_per_meter,
        "B1_monte_carlo": mc,
        "B2_scalar_td": scalar,
        "B3_distributional_median": quantiles[:, 0],
    }, quantiles


def _return_start_arrays(
    trajectories: list[BootstrapTrajectory], world_size: np.ndarray, velocity_limit: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    positions = []
    velocities = []
    distances = []
    prefixes = []
    for trajectory in trajectories:
        transition = trajectory.episode.transitions[0]
        position = transition.state[:3] * world_size
        positions.append(position)
        velocities.append(transition.state[3:6] * velocity_limit)
        distances.append(np.linalg.norm(position - trajectory.charger_position))
        prefixes.append(trajectory.prefix_steps_executed)
    return np.asarray(positions), np.asarray(velocities), np.asarray(distances), np.asarray(prefixes)


def run_stage_a(config: StageABootstrapConfig) -> dict:
    output = Path(config.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty run directory: {output}")
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = Path(config.checkpoint).resolve()
    checkpoint_hash = sha256_file(checkpoint)
    sortie_seeds = _sortie_seeds(config)
    config_payload = asdict(config) | {
        "experiment": "STAGE_A_4X4_ENERGY_ESTIMATION",
        "status": "RUNNING",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "navigation_policy": "FROZEN_SAC_1M",
        "navigation_energy_semantics": "1000_UNIT_CHECKPOINT_COMPATIBILITY_ONLY",
        "operational_energy_semantics": "SEPARATE_TELEMETRY_ACCOUNTING_NON_ENFORCING_IN_STAGE_A",
        "comparison_methods": ["B0_DISTANCE", "B1_MC", "B2_SCALAR_TD_GAMMA_1", "B3_QUANTILE_TD_GAMMA_1"],
        "split_unit": "sortie",
        "split_fraction": [0.60, 0.20, 0.20],
        "split_seed": config.seed + 8_003,
        "sortie_seeds": sortie_seeds.tolist(),
        "git_commit_sha": git_sha(ROOT),
        "code_hash": code_hash(ROOT),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "exact_command": shlex.join([sys.executable, *sys.argv]),
        "started_at": utc_now(),
        "pid": os.getpid(),
        "interrupted_runs_invalid_for_comparison": True,
    }
    write_json(output / "config.json", config_payload)
    write_json(output / "RUNNING.json", {"status": "RUNNING", "started_at": config_payload["started_at"]})
    try:
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        torch.set_num_threads(1)
        policy = FrozenGoalConditionedSAC(checkpoint, device=config.device)
        prefixes = _prefix_schedule(config)
        scenario = load_scenario(config.scenario)
        world_size = scenario.world_size
        velocity_limit = np.asarray(scenario.configuration_overrides.get("v_max", [0.30, 0.30, 0.12]))
        if not np.allclose(world_size, np.array([4.0, 4.0, 2.0])):
            raise ValueError("Stage A is restricted to the 4x4x2 bootstrap environment")
        trajectories: list[BootstrapTrajectory] = []
        raw_path = output / "raw_sorties.jsonl"
        for index, (prefix_steps, sortie_seed) in enumerate(zip(prefixes, sortie_seeds, strict=True)):
            trajectory = collect_bootstrap_trajectory(
                policy,
                scenario=config.scenario,
                sortie_seed=int(sortie_seed),
                prefix_steps=int(prefix_steps),
                max_return_steps=config.max_return_steps,
                operational_energy_capacity=config.operational_energy_capacity,
                policy_hash=checkpoint_hash,
            )
            trajectories.append(trajectory)
            append_jsonl(raw_path, trajectory_payload(index, trajectory, world_size, velocity_limit))
        completed_count = sum(item.episode.completed for item in trajectories)
        censored_count = len(trajectories) - completed_count
        write_json(
            output / "COLLECTION_COMPLETED.json",
            {
                "status": "COLLECTION_COMPLETED",
                "sorties": len(trajectories),
                "completed_returns": completed_count,
                "censored_returns": censored_count,
                "completion_rate": completed_count / len(trajectories),
                "valid_return_to_go_transitions": sum(
                    len(item.episode.transitions) for item in trajectories if item.episode.completed
                ),
                "censored_transitions_without_return_to_go_labels": sum(
                    len(item.episode.transitions) for item in trajectories if not item.episode.completed
                ),
                "data_sha256": sha256_file(raw_path),
            },
        )
        split = split_sorties(len(trajectories), config.seed)
        split_payload = _split_manifest(split, trajectories, config.seed)
        write_json(output / "split.json", split_payload)
        datasets = {
            name: build_dataset(
                trajectories,
                indices,
                world_size=world_size,
                velocity_limit=velocity_limit,
            )
            for name, indices in split.items()
        }
        train = datasets["train"]
        calibration = datasets["calibration"]
        test = datasets["test"]
        mc_model, mc_training = _train_mc(train, config.training_updates)
        scalar_td, scalar_training = _train_scalar_td(train, config.training_updates)
        quantile_td, quantile_training = _train_quantile_td(train, config.training_updates)
        cost_per_meter = _cost_per_meter(trajectories, split["train"], world_size)
        calibration_predictions, calibration_quantiles = _predict(
            calibration, cost_per_meter, mc_model, scalar_td, quantile_td
        )
        test_predictions, test_quantiles = _predict(test, cost_per_meter, mc_model, scalar_td, quantile_td)
        model_hashes = {
            "B1_MC": _save_model(output / "b1_mc_model.pt", mc_model, "ScalarEnergyCritic_MC"),
            "B2_SCALAR_TD": _save_model(output / "b2_scalar_td_model.pt", scalar_td, "ScalarEnergyCritic_TD"),
            "B3_QUANTILE_TD": _save_model(
                output / "b3_quantile_td_model.pt", quantile_td, "MonotoneQuantileCritic_TD"
            ),
        }
        paired = paired_sortie_comparisons(
            test_predictions,
            test.returns,
            test.sortie_indices,
            seed=config.seed + 12_007,
        )
        learned_decision = learned_value_decision(paired)
        return_positions, return_velocities, return_distances, return_prefixes = _return_start_arrays(
            trajectories, world_size, velocity_limit
        )
        data_coverage = coverage_report(
            return_positions,
            return_velocities,
            return_distances,
            return_prefixes,
            world_size,
        )
        q_metrics = quantile_metrics(test_quantiles, test.returns, QUANTILES)
        terminal_mask = test.terminals
        terminal_prediction_diagnostics = {
            name: regression_metrics(prediction[terminal_mask], test.returns[terminal_mask])
            for name, prediction in test_predictions.items()
        }
        b3_coverage_reasonable = bool(
            abs(q_metrics["empirical_coverage"]["0.5"] - 0.50) <= 0.20
            and q_metrics["empirical_coverage"]["0.9"] >= 0.80
            and q_metrics["empirical_coverage"]["0.95"] >= 0.85
            and q_metrics["empirical_coverage"]["0.99"] >= 0.90
            and not q_metrics["quantile_crossing_occurred"]
        )
        training_finite = bool(
            mc_training["all_finite"] and scalar_training["all_finite"] and quantile_training["all_finite"]
        )
        b2_semantically_stable = bool(
            regression_metrics(test_predictions["B2_scalar_td"], test.returns)["mae"]
            <= 2.0 * regression_metrics(test_predictions["B0_distance"], test.returns)["mae"]
            and terminal_prediction_diagnostics["B2_scalar_td"]["mae"] <= 0.05
        )
        b3_semantically_stable = bool(
            regression_metrics(test_predictions["B3_distributional_median"], test.returns)["mae"]
            <= 2.0 * regression_metrics(test_predictions["B0_distance"], test.returns)["mae"]
            and terminal_prediction_diagnostics["B3_distributional_median"]["mae"] <= 0.05
        )
        no_leakage = bool(
            split_payload["all_sortie_ids_unique"] and split_payload["all_sorties_assigned_exactly_once"]
        )
        completion_sufficient = completed_count / len(trajectories) >= 0.98
        blockers: list[str] = []
        if not completion_sufficient:
            blockers.append("charger_return_completion_rate_below_0.98")
        if not data_coverage["DATA_COVERAGE_SUFFICIENT"]:
            blockers.append("return_start_state_coverage_below_preregistered_validation_threshold")
        if not training_finite:
            blockers.append("nonfinite_B1_B2_or_B3_training")
        if not b2_semantically_stable:
            blockers.append("B2_scalar_TD_not_semantically_stable_on_heldout_or_terminal_samples")
        if not b3_semantically_stable:
            blockers.append("B3_quantile_TD_semantic_collapse_or_terminal_anchor_failure")
        if not b3_coverage_reasonable:
            blockers.append("B3_quantile_coverage_not_reasonable")
        if not no_leakage:
            blockers.append("sortie_split_leakage")
        if learned_decision["LEARNED_MODEL_ADDS_VALUE"] != "TRUE":
            blockers.append("learned_critic_extra_value_over_B0_not_established")
        if config.run_kind != "formal":
            blockers.append("formal_multi_seed_stage_A_not_run")
        figure_paths = build_figures(
            output / "figures",
            predictions=test_predictions,
            quantile_predictions=test_quantiles,
            quantile_levels=QUANTILES,
            target=test.returns,
            distances=test.distances,
            path_lengths=test.path_lengths_to_go,
            return_start_positions=return_positions,
            world_size=world_size,
        )
        results = {
            "status": "COMPLETED",
            "run_kind": config.run_kind,
            "completed_at": utc_now(),
            "sorties": len(trajectories),
            "completed_returns": completed_count,
            "censored_returns": censored_count,
            "completion_rate": completed_count / len(trajectories),
            "valid_return_to_go_transitions": sum(
                len(item.episode.transitions) for item in trajectories if item.episode.completed
            ),
            "censored_transitions_without_return_to_go_labels": sum(
                len(item.episode.transitions) for item in trajectories if not item.episode.completed
            ),
            "prefix_steps_min": int(prefixes.min()),
            "prefix_steps_max": int(prefixes.max()),
            "prefix_steps_unique": int(np.unique(prefixes).size),
            "split": split_payload,
            "training_diagnostics": {
                "B1_monte_carlo": mc_training,
                "B2_scalar_td": scalar_training,
                "B3_quantile_td": quantile_training,
                "heldout_terminal_prediction_metrics": terminal_prediction_diagnostics,
            },
            "calibration_split_metrics": {
                name: regression_metrics(prediction, calibration.returns)
                for name, prediction in calibration_predictions.items()
            }
            | {"B3_quantiles": quantile_metrics(calibration_quantiles, calibration.returns, QUANTILES)},
            "heldout_test_metrics": {
                name: regression_metrics(prediction, test.returns)
                for name, prediction in test_predictions.items()
            }
            | {"B3_quantiles": q_metrics},
            "distance_only_analysis": {
                "actual_energy_vs_euclidean_distance_correlation": safe_correlation(
                    test.returns, test.distances
                ),
                "actual_energy_vs_actual_path_length_correlation": safe_correlation(
                    test.returns, test.path_lengths_to_go
                ),
                "B0_signed_residual_relationships": residual_relationships(
                    test_predictions["B0_distance"] - test.returns,
                    {
                        "current_velocity_magnitude": test.velocity_magnitudes,
                        "velocity_direction_relative_to_charger": test.velocity_alignment_to_charger,
                        "action_magnitude": test.action_magnitudes,
                        "altitude_difference": test.altitude_differences,
                        "task_prefix_length": test.prefix_lengths,
                    },
                ),
            },
            "paired_sortie_comparison_vs_B0": paired,
            "learned_model_value": learned_decision,
            "data_coverage": data_coverage,
            "gate_checks": {
                "completion_rate_sufficient": completion_sufficient,
                "data_coverage_sufficient": data_coverage["DATA_COVERAGE_SUFFICIENT"],
                "B1_B2_B3_training_finite": training_finite,
                "B2_prediction_semantically_stable": b2_semantically_stable,
                "B3_prediction_semantically_stable": b3_semantically_stable,
                "B3_quantile_coverage_reasonable": b3_coverage_reasonable,
                "sortie_split_no_leakage": no_leakage,
                "learned_model_adds_value": learned_decision["LEARNED_MODEL_ADDS_VALUE"],
                "formal_stage_A_completed": config.run_kind == "formal",
                "READY_FOR_16X16": len(blockers) == 0,
                "blockers": blockers,
            },
            "cost_per_meter_train": cost_per_meter,
            "figures": figure_paths,
            "model_sha256": model_hashes,
            "data_sha256": sha256_file(raw_path),
            "split_sha256": sha256_file(output / "split.json"),
            "validation_result_eligible": config.run_kind == "validation",
            "formal_result_eligible": config.run_kind == "formal",
        }
        write_json(output / "results.json", results)
        write_json(output / "COMPLETED.json", results)
        write_json(output / "RUNNING.json", {"status": "COMPLETED", "completed_at": results["completed_at"]})
        return results
    except Exception as error:
        failure = {
            "status": "INVALID_FOR_COMPARISON",
            "failed_at": utc_now(),
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        write_json(output / "FAILED.json", failure)
        write_json(output / "RUNNING.json", failure)
        raise
