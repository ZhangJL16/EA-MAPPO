from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from scipy.spatial import cKDTree
from stable_baselines3 import SAC

from envs.UAVEnergyDeliverySAC import (
    GoalConditionedQuantileTDEnergyEstimator,
    SACTrainingPhase,
    UAVEnergyDeliverySACEnv,
)
from experiments.energy_td_diagnostics.core import (
    DiagnosticDataset,
    TrainingConfig,
    compute_mc_returns,
    terminal_anchor_statistics,
    train_diagnostic_estimator,
)
from review_bundle.safety.energy.td import quantile_ssp_target
from scripts.train_uav_energy_delivery_sac import (
    NavigationTask,
    generate_stratified_navigation_tasks,
    load_navigation_tasks,
    save_navigation_tasks,
)


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    return value


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_value(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def policy_parameter_sha256(policy: SAC) -> str:
    digest = hashlib.sha256()
    for name, parameter in sorted(policy.policy.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def freeze_policy(checkpoint: Path, device: str) -> SAC:
    policy = SAC.load(checkpoint, device=device)
    for parameter in policy.policy.parameters():
        parameter.requires_grad_(False)
    policy.policy.set_training_mode(False)
    if not all(not parameter.requires_grad for parameter in policy.policy.parameters()):
        raise RuntimeError("navigation policy was not completely frozen")
    return policy


def _collect_tasks(
    policy: SAC,
    tasks: list[NavigationTask],
    *,
    seed: int,
    max_steps: int,
) -> tuple[DiagnosticDataset, list[dict[str, object]]]:
    columns: dict[str, list] = {
        name: []
        for name in DiagnosticDataset.__dataclass_fields__
        if name != "returns"
    }
    task_summaries: list[dict[str, object]] = []
    trajectory_id = 0
    environment = UAVEnergyDeliverySACEnv(
        phase=SACTrainingPhase.TD_PRETRAINING,
        max_steps_per_task=max_steps,
        phase1_episode_max_policy_steps=max_steps,
    )
    for task_index, task in enumerate(tasks):
        observation, _ = environment.reset(
            seed=seed + task_index,
            options={
                "start_position": task.start_position,
                "start_velocity": task.initial_velocity,
                "task_point": task.goal_position,
            },
        )
        trajectory_rows: list[dict[str, object]] = []
        path_length = 0.0
        previous_position = environment.agent.pos.copy()
        for step_index in range(max_steps):
            state = environment.energy_state_for_goal(task.goal_position)
            position = environment.agent.pos.copy()
            velocity = environment.agent.vel.copy()
            distance = float(np.linalg.norm(task.goal_position - position))
            action, _ = policy.predict(observation, deterministic=True)
            next_observation, _, terminated, truncated, info = environment.step(action)
            next_state = environment.energy_state_for_goal(task.goal_position)
            if terminated:
                next_action = np.zeros(3, dtype=np.float32)
            else:
                next_action, _ = policy.predict(next_observation, deterministic=True)
            current_position = environment.agent.pos.copy()
            path_length += float(np.linalg.norm(current_position - previous_position))
            previous_position = current_position
            trajectory_rows.append(
                {
                    "state": state,
                    "action": np.asarray(action, dtype=np.float32),
                    "cost": float(info["realized_energy_cost"]),
                    "next_state": next_state,
                    "next_action": np.asarray(next_action, dtype=np.float32),
                    "terminal": bool(terminated and info["is_success"]),
                    "step_index": step_index,
                    "position": position,
                    "velocity": velocity,
                    "goal": task.goal_position.copy(),
                    "distance": distance,
                    "initial_distance": task.straight_line_distance,
                    "boundary_contact": bool(info["boundary_contact"]),
                }
            )
            observation = next_observation
            if terminated or truncated:
                success = bool(terminated and info["is_success"])
                task_summaries.append(
                    {
                        "task_index": task_index,
                        "distance_bucket": task.distance_bucket,
                        "straight_line_distance": task.straight_line_distance,
                        "policy_steps": len(trajectory_rows),
                        "path_length": path_length,
                        "path_ratio": path_length / max(task.straight_line_distance, 1e-8),
                        "success": success,
                        "end_reason": info["end_reason"],
                        "boundary_contact_steps": int(
                            sum(bool(row["boundary_contact"]) for row in trajectory_rows)
                        ),
                    }
                )
                if not success:
                    environment.close()
                    raise RuntimeError(
                        f"diagnostic task {task_index} failed with {info['end_reason']}"
                    )
                break
        else:
            environment.close()
            raise RuntimeError(f"diagnostic task {task_index} exceeded {max_steps} steps")
        trajectory_length = len(trajectory_rows)
        for row_index, row in enumerate(trajectory_rows):
            columns["states"].append(row["state"])
            columns["actions"].append(row["action"])
            columns["costs"].append(row["cost"])
            columns["next_states"].append(row["next_state"])
            columns["next_actions"].append(row["next_action"])
            columns["terminals"].append(row["terminal"])
            columns["horizons"].append(trajectory_length - row_index)
            columns["trajectory_ids"].append(trajectory_id)
            columns["step_indices"].append(row["step_index"])
            columns["positions"].append(row["position"])
            columns["velocities"].append(row["velocity"])
            columns["goals"].append(row["goal"])
            columns["distances"].append(row["distance"])
            columns["initial_distances"].append(row["initial_distance"])
            columns["boundary_contacts"].append(row["boundary_contact"])
        trajectory_id += 1
    environment.close()
    costs = np.asarray(columns["costs"], dtype=np.float32)
    trajectory_ids = np.asarray(columns["trajectory_ids"], dtype=np.int32)
    terminals = np.asarray(columns["terminals"], dtype=bool)
    returns = compute_mc_returns(costs, trajectory_ids, terminals)
    dataset = DiagnosticDataset(
        states=np.asarray(columns["states"], dtype=np.float32),
        actions=np.asarray(columns["actions"], dtype=np.float32),
        costs=costs,
        next_states=np.asarray(columns["next_states"], dtype=np.float32),
        next_actions=np.asarray(columns["next_actions"], dtype=np.float32),
        terminals=terminals,
        returns=returns,
        horizons=np.asarray(columns["horizons"], dtype=np.int32),
        trajectory_ids=trajectory_ids,
        step_indices=np.asarray(columns["step_indices"], dtype=np.int32),
        positions=np.asarray(columns["positions"], dtype=np.float32),
        velocities=np.asarray(columns["velocities"], dtype=np.float32),
        goals=np.asarray(columns["goals"], dtype=np.float32),
        distances=np.asarray(columns["distances"], dtype=np.float32),
        initial_distances=np.asarray(columns["initial_distances"], dtype=np.float32),
        boundary_contacts=np.asarray(columns["boundary_contacts"], dtype=bool),
    )
    return dataset, task_summaries


def _dataset_summary(dataset: DiagnosticDataset, task_summaries: list[dict[str, object]]) -> dict:
    return {
        "trajectories": dataset.trajectory_count,
        "transitions": dataset.transition_count,
        "success_rate": float(np.mean([row["success"] for row in task_summaries])),
        "mc_return_scale": {
            "median": float(np.median(dataset.returns)),
            "p90": float(np.quantile(dataset.returns, 0.90)),
            "p95": float(np.quantile(dataset.returns, 0.95)),
            "p99": float(np.quantile(dataset.returns, 0.99)),
            "max": float(np.max(dataset.returns)),
        },
        "step_energy_scale": {
            "mean": float(np.mean(dataset.costs)),
            "p99": float(np.quantile(dataset.costs, 0.99)),
            "max": float(np.max(dataset.costs)),
        },
        "terminal_anchor": terminal_anchor_statistics(
            dataset.terminals, dataset.trajectory_ids, batch_size=128
        ),
        "boundary_contact_transition_rate": float(np.mean(dataset.boundary_contacts)),
        "tasks": task_summaries,
    }


def _terminal_audit(
    dataset: DiagnosticDataset,
    estimator: GoalConditionedQuantileTDEnergyEstimator,
    count: int = 100,
) -> list[dict[str, object]]:
    terminal_indices = np.flatnonzero(dataset.terminals)[:count]
    rows: list[dict[str, object]] = []
    for transition_index in terminal_indices:
        next_feature = np.concatenate(
            (dataset.next_states[transition_index], dataset.next_actions[transition_index])
        ).astype(np.float32)
        with torch.no_grad():
            next_online = estimator.model(
                torch.as_tensor(next_feature[None, :], device=estimator.device)
            ).squeeze(0).cpu().numpy()
            next_target_tensor = estimator.target_model(
                torch.as_tensor(next_feature[None, :], device=estimator.device)
            )
            next_target = next_target_tensor.squeeze(0).cpu().numpy()
        cost = float(dataset.costs[transition_index])
        computed_target = quantile_ssp_target(
            torch.as_tensor([cost], dtype=torch.float32, device=estimator.device),
            next_target_tensor,
            torch.ones(1, dtype=torch.bool, device=estimator.device),
            gamma=estimator.gamma_energy,
        ).squeeze(0).detach().cpu().numpy()
        rows.append(
            {
                "trajectory_id": int(dataset.trajectory_ids[transition_index]),
                "transition_index": int(transition_index),
                "last_transition_cost": cost,
                "goal_reached": True,
                "next_online_quantiles": next_online.tolist(),
                "next_target_quantiles": next_target.tolist(),
                "bootstrap_mask": 0.0,
                "computed_target_quantiles": computed_target.tolist(),
                "target_equals_realized_cost": bool(
                    np.allclose(computed_target, cost, rtol=0.0, atol=1e-7)
                ),
            }
        )
    return rows


def collect_command(args: argparse.Namespace) -> None:
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = Path(args.sac_checkpoint).resolve()
    td_checkpoint = Path(args.td_checkpoint).resolve()
    policy = freeze_policy(checkpoint, args.device)
    policy_hash_before = policy_parameter_sha256(policy)
    heldout_tasks = load_navigation_tasks(Path(args.heldout_tasks))
    if len(heldout_tasks) < 500:
        raise ValueError("the held-out diagnostic set must contain at least 500 tasks")
    train_tasks = generate_stratified_navigation_tasks(
        num_tasks=args.train_tasks,
        seed=args.train_task_seed,
    )
    save_navigation_tasks(
        output / "train_tasks.json",
        train_tasks,
        seed=args.train_task_seed,
        role="energy_td_diagnostic_train",
    )
    train_dataset, train_summaries = _collect_tasks(
        policy,
        train_tasks,
        seed=args.train_rollout_seed,
        max_steps=args.max_steps,
    )
    heldout_dataset, heldout_summaries = _collect_tasks(
        policy,
        heldout_tasks,
        seed=args.heldout_rollout_seed,
        max_steps=args.max_steps,
    )
    train_dataset.save(output / "train_dataset.npz")
    heldout_dataset.save(output / "heldout_dataset.npz")
    estimator = GoalConditionedQuantileTDEnergyEstimator.load(td_checkpoint, device=args.device)
    terminal_rows = _terminal_audit(train_dataset, estimator, count=100)
    write_json(output / "terminal_target_audit.json", terminal_rows)
    policy_hash_after = policy_parameter_sha256(policy)
    audit = {
        "sac_checkpoint": str(checkpoint),
        "sac_checkpoint_sha256": file_sha256(checkpoint),
        "policy_parameter_sha256_before_collection": policy_hash_before,
        "policy_parameter_sha256_after_collection": policy_hash_after,
        "policy_parameters_unchanged": policy_hash_before == policy_hash_after,
        "all_policy_parameters_require_grad_false": all(
            not parameter.requires_grad for parameter in policy.policy.parameters()
        ),
        "policy_training_mode": bool(policy.policy.training),
        "optimizer_calls_during_collection": 0,
        "td_checkpoint": str(td_checkpoint),
        "td_checkpoint_sha256": file_sha256(td_checkpoint),
        "train": _dataset_summary(train_dataset, train_summaries),
        "heldout": _dataset_summary(heldout_dataset, heldout_summaries),
        "terminal_audit_samples": len(terminal_rows),
    }
    write_json(output / "dataset_audit.json", audit)
    print(json.dumps(json_value(audit), indent=2, sort_keys=True))


def _matrix_configs(args: argparse.Namespace) -> list[tuple[str, TrainingConfig]]:
    common = {
        "updates": args.updates,
        "batch_size": args.batch_size,
        "replay_capacity": args.replay_capacity,
        "learning_starts": args.learning_starts,
        "learning_rate": args.learning_rate,
        "target_tau": args.target_tau,
        "seed": args.seed,
        "device": args.device,
        "log_every": args.log_every,
    }
    requested = [name.strip() for name in args.methods.split(",") if name.strip()]
    def variant(**changes) -> dict[str, object]:
        return {**common, **changes}

    available = {
        "current_quantile_td": TrainingConfig(kind="quantile_td", **common),
        "scalar_td": TrainingConfig(kind="scalar_td", **common),
        "mc_scalar": TrainingConfig(kind="mc_scalar", **common),
        "mc_quantile": TrainingConfig(kind="mc_quantile", **common),
        "nstep20_scalar": TrainingConfig(kind="n_step_scalar_td", n_step=20, **common),
        "terminal10_quantile": TrainingConfig(
            kind="quantile_td", terminal_fraction_per_batch=0.10, **common
        ),
        "terminal25_quantile": TrainingConfig(
            kind="quantile_td", terminal_fraction_per_batch=0.25, **common
        ),
        "uniform32_quantile_td": TrainingConfig(
            kind="uniform_quantile_td", **common
        ),
        "free_four_quantile_td": TrainingConfig(
            kind="free_four_quantile_td", **common
        ),
        "unweighted_four_quantile_td": TrainingConfig(
            kind="unweighted_four_quantile_td", **common
        ),
        "uniform_four_quantile_td": TrainingConfig(
            kind="uniform_four_quantile_td", **common
        ),
        "quantile_tau005": TrainingConfig(
            kind="quantile_td", **variant(target_tau=0.005)
        ),
        "quantile_tau001": TrainingConfig(
            kind="quantile_td", **variant(target_tau=0.001)
        ),
        "quantile_tau0001": TrainingConfig(
            kind="quantile_td", **variant(target_tau=0.0001)
        ),
        "quantile_lr1e4": TrainingConfig(
            kind="quantile_td", **variant(learning_rate=1e-4)
        ),
        "quantile_lr3e5": TrainingConfig(
            kind="quantile_td", **variant(learning_rate=3e-5)
        ),
        "quantile_lr1e5": TrainingConfig(
            kind="quantile_td", **variant(learning_rate=1e-5)
        ),
        "quantile_hard500": TrainingConfig(
            kind="quantile_td",
            **variant(target_update="hard", hard_update_interval=500),
        ),
        "quantile_hard2000": TrainingConfig(
            kind="quantile_td",
            **variant(target_update="hard", hard_update_interval=2000),
        ),
        "quantile_hard10000": TrainingConfig(
            kind="quantile_td",
            **variant(target_update="hard", hard_update_interval=10000),
        ),
        "state_only_mc_scalar": TrainingConfig(
            kind="mc_scalar", include_action=False, **common
        ),
    }
    unknown = sorted(set(requested) - set(available))
    if unknown:
        raise ValueError(f"unknown diagnostic methods: {unknown}")
    return [(name, available[name]) for name in requested]


def matrix_command(args: argparse.Namespace) -> None:
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    train = DiagnosticDataset.load(args.train_dataset)
    heldout = DiagnosticDataset.load(args.heldout_dataset)
    results: dict[str, object] = {}
    for method_name, config in _matrix_configs(args):
        print(f"=== {method_name} ===", flush=True)
        results[method_name] = train_diagnostic_estimator(
            train,
            heldout,
            config,
            output_dir=output / method_name,
        )
    summary = {
        "train_dataset": str(Path(args.train_dataset).resolve()),
        "heldout_dataset": str(Path(args.heldout_dataset).resolve()),
        "same_data_for_all_methods": True,
        "methods": results,
    }
    write_json(output / "matrix_results.json", summary)
    print(json.dumps(json_value({name: result["final"] for name, result in results.items()}), indent=2))


def _synthetic_dataset(
    *,
    length: int,
    trajectories: int,
    random_costs: bool,
    seed: int,
) -> DiagnosticDataset:
    rng = np.random.default_rng(seed)
    transition_count = length * trajectories
    trajectory_ids = np.repeat(np.arange(trajectories, dtype=np.int32), length)
    step_indices = np.tile(np.arange(length, dtype=np.int32), trajectories)
    horizons = np.tile(np.arange(length, 0, -1, dtype=np.int32), trajectories)
    if random_costs:
        costs = rng.lognormal(mean=np.log(0.05), sigma=0.25, size=transition_count).astype(np.float32)
    else:
        base_cost = 1.0 if length == 10 else 0.05
        costs = np.full(transition_count, base_cost, dtype=np.float32)
    terminals = np.zeros(transition_count, dtype=bool)
    terminals[length - 1 :: length] = True
    states = np.zeros((transition_count, 7), dtype=np.float32)
    states[:, 6] = horizons / float(length)
    states[:, 0] = 0.25
    next_states = np.zeros_like(states)
    for trajectory_index in range(trajectories):
        start = trajectory_index * length
        stop = start + length
        next_states[start : stop - 1] = states[start + 1 : stop]
    actions = np.zeros((transition_count, 3), dtype=np.float32)
    returns = compute_mc_returns(costs, trajectory_ids, terminals)
    zeros3 = np.zeros((transition_count, 3), dtype=np.float32)
    return DiagnosticDataset(
        states=states,
        actions=actions,
        costs=costs,
        next_states=next_states,
        next_actions=actions.copy(),
        terminals=terminals,
        returns=returns,
        horizons=horizons,
        trajectory_ids=trajectory_ids,
        step_indices=step_indices,
        positions=zeros3.copy(),
        velocities=zeros3.copy(),
        goals=zeros3.copy(),
        distances=horizons.astype(np.float32),
        initial_distances=np.full(transition_count, float(length), dtype=np.float32),
        boundary_contacts=np.zeros(transition_count, dtype=bool),
    )


def synthetic_command(args: argparse.Namespace) -> None:
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    environments = {
        "A_fixed_10_cost_1": _synthetic_dataset(
            length=10, trajectories=200, random_costs=False, seed=args.seed
        ),
        "B_fixed_500_cost_005": _synthetic_dataset(
            length=500, trajectories=200, random_costs=False, seed=args.seed + 1
        ),
        "C_random_positive_100": _synthetic_dataset(
            length=100, trajectories=300, random_costs=True, seed=args.seed + 2
        ),
    }
    results: dict[str, object] = {}
    for environment_name, dataset in environments.items():
        environment_updates = args.long_updates if environment_name.startswith("B_") else args.updates
        results[environment_name] = {}
        for kind in ("scalar_td", "quantile_td", "mc_scalar"):
            config = TrainingConfig(
                kind=kind,
                updates=environment_updates,
                batch_size=args.batch_size,
                replay_capacity=min(100_000, dataset.transition_count),
                learning_starts=args.batch_size,
                learning_rate=args.learning_rate,
                target_tau=args.target_tau,
                seed=args.seed,
                device=args.device,
                log_every=args.log_every,
            )
            results[environment_name][kind] = train_diagnostic_estimator(
                dataset,
                dataset,
                config,
                output_dir=output / environment_name / kind,
            )
    write_json(output / "synthetic_results.json", results)


def rollout_command(args: argparse.Namespace) -> None:
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    dataset = DiagnosticDataset.load(args.dataset)
    policy = freeze_policy(Path(args.sac_checkpoint), args.device)
    candidate_indices = np.linspace(
        0, dataset.transition_count - 1, args.samples, dtype=np.int64
    )
    estimates: list[float] = []
    truths: list[float] = []
    rollout_steps: list[int] = []
    elapsed_times: list[float] = []
    records: list[dict[str, object]] = []
    maximum_velocity_projection = 0.0
    for sample_number, transition_index in enumerate(candidate_indices):
        environment = UAVEnergyDeliverySACEnv(
            phase=SACTrainingPhase.TD_PRETRAINING,
            minimum_task_distance=5.0001,
        )
        reset_velocity = dataset.velocities[transition_index].astype(np.float64).copy()
        horizontal_speed = float(np.linalg.norm(reset_velocity[:2]))
        if horizontal_speed > 20.0:
            reset_velocity[:2] *= 20.0 / horizontal_speed
        reset_velocity[2] = np.clip(reset_velocity[2], -5.0, 5.0)
        maximum_velocity_projection = max(
            maximum_velocity_projection,
            float(np.linalg.norm(reset_velocity - dataset.velocities[transition_index])),
        )
        observation, _ = environment.reset(
            seed=args.seed + sample_number,
            options={
                "start_position": dataset.positions[transition_index],
                "start_velocity": reset_velocity.astype(np.float32),
                "task_point": dataset.goals[transition_index],
            },
        )
        total_energy = 0.0
        start_time = time.perf_counter()
        for policy_step in range(args.max_steps):
            action, _ = policy.predict(observation, deterministic=True)
            observation, _, terminated, truncated, info = environment.step(action)
            total_energy += float(info["realized_energy_cost"])
            if terminated or truncated:
                if not bool(terminated and info["is_success"]):
                    environment.close()
                    raise RuntimeError("model-based rollout did not reach the goal")
                break
        elapsed = time.perf_counter() - start_time
        truth = float(dataset.returns[transition_index])
        estimates.append(total_energy)
        truths.append(truth)
        rollout_steps.append(policy_step + 1)
        elapsed_times.append(elapsed)
        records.append(
            {
                "transition_index": int(transition_index),
                "true_mc_return": truth,
                "rollout_estimate": total_energy,
                "absolute_error": abs(total_energy - truth),
                "rollout_steps": policy_step + 1,
                "inference_seconds": elapsed,
            }
        )
        environment.close()
    errors = np.asarray(estimates) - np.asarray(truths)
    summary = {
        "samples": len(records),
        "MAE": float(np.mean(np.abs(errors))),
        "RMSE": float(np.sqrt(np.mean(np.square(errors)))),
        "bias": float(np.mean(errors)),
        "mean_rollout_steps": float(np.mean(rollout_steps)),
        "mean_inference_seconds": float(np.mean(elapsed_times)),
        "p95_inference_seconds": float(np.quantile(elapsed_times, 0.95)),
        "mean_seconds_per_simulated_step": float(
            np.sum(elapsed_times) / max(np.sum(rollout_steps), 1)
        ),
        "device": args.device,
        "maximum_reset_velocity_projection_mps": maximum_velocity_projection,
        "records": records,
    }
    write_json(output / "model_based_rollout.json", summary)
    print(json.dumps(summary, indent=2))


def _different_trajectory_neighbors(
    features: np.ndarray,
    trajectory_ids: np.ndarray,
    *,
    neighbor_count: int,
    max_query_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(features, dtype=np.float64)
    identifiers = np.asarray(trajectory_ids)
    scale = np.std(values, axis=0)
    scale[scale < 1e-8] = 1.0
    standardized = (values - np.mean(values, axis=0)) / scale
    tree = cKDTree(standardized)
    sample_count = min(values.shape[0], max_query_samples)
    query_indices = np.linspace(0, values.shape[0] - 1, sample_count, dtype=np.int64)
    query_count = min(max(neighbor_count * 16, 256), values.shape[0])
    distances, indices = tree.query(standardized[query_indices], k=query_count, workers=-1)
    selected_indices = np.full((sample_count, neighbor_count), -1, dtype=np.int64)
    selected_distances = np.full((sample_count, neighbor_count), np.nan, dtype=np.float64)
    counts = np.zeros(sample_count, dtype=np.int32)
    for row_index, source_index in enumerate(query_indices):
        for distance, candidate in zip(distances[row_index, 1:], indices[row_index, 1:]):
            if identifiers[candidate] == identifiers[source_index]:
                continue
            slot = counts[row_index]
            selected_indices[row_index, slot] = int(candidate)
            selected_distances[row_index, slot] = float(distance)
            counts[row_index] += 1
            if counts[row_index] == neighbor_count:
                break
    valid = counts == neighbor_count
    for row_index in np.flatnonzero(~valid):
        source_index = query_indices[row_index]
        fallback_count = min(max(query_count * 8, 2048), values.shape[0])
        fallback_distances, fallback_indices = tree.query(
            standardized[source_index], k=fallback_count, workers=-1
        )
        counts[row_index] = 0
        for distance, candidate in zip(fallback_distances[1:], fallback_indices[1:]):
            if identifiers[candidate] == identifiers[source_index]:
                continue
            slot = counts[row_index]
            selected_indices[row_index, slot] = int(candidate)
            selected_distances[row_index, slot] = float(distance)
            counts[row_index] += 1
            if counts[row_index] == neighbor_count:
                break
    valid = counts == neighbor_count
    if not np.all(valid):
        raise RuntimeError(
            f"failed to find {neighbor_count} cross-trajectory neighbors for "
            f"{int((~valid).sum())} samples"
        )
    return query_indices, selected_indices, selected_distances, scale


def _ambiguity_summary(
    dataset: DiagnosticDataset,
    features: np.ndarray,
    *,
    neighbor_count: int,
    max_query_samples: int,
) -> dict[str, object]:
    query_indices, neighbor_indices, neighbor_distances, feature_scale = _different_trajectory_neighbors(
        features,
        dataset.trajectory_ids,
        neighbor_count=neighbor_count,
        max_query_samples=max_query_samples,
    )
    nearest = neighbor_indices[:, 0]
    nearest_distance = neighbor_distances[:, 0]
    nearest_return_difference = np.abs(dataset.returns[query_indices] - dataset.returns[nearest])
    nearest_position_difference = np.linalg.norm(
        dataset.positions[query_indices] - dataset.positions[nearest], axis=1
    )
    boundary_mismatch = (
        dataset.boundary_contacts[query_indices] != dataset.boundary_contacts[nearest]
    )
    local_returns = dataset.returns[neighbor_indices]
    local_return_std = np.std(local_returns, axis=1)
    local_return_range = np.ptp(local_returns, axis=1)
    close_threshold = float(np.quantile(nearest_distance, 0.01))
    close = nearest_distance <= close_threshold
    close_far_position = close & (nearest_position_difference >= 500.0)
    close_boundary_mismatch = close & boundary_mismatch

    def selected_metrics(mask: np.ndarray) -> dict[str, float | int | None]:
        count = int(mask.sum())
        if count == 0:
            return {
                "count": 0,
                "return_difference_mean": None,
                "return_difference_p95": None,
                "position_difference_mean": None,
            }
        return {
            "count": count,
            "return_difference_mean": float(nearest_return_difference[mask].mean()),
            "return_difference_p95": float(np.quantile(nearest_return_difference[mask], 0.95)),
            "position_difference_mean": float(nearest_position_difference[mask].mean()),
        }

    return {
        "feature_dimension": int(features.shape[1]),
        "query_samples": int(query_indices.size),
        "feature_standard_deviation": feature_scale.tolist(),
        "cross_trajectory_neighbor_count": int(neighbor_count),
        "nearest_standardized_distance": {
            "median": float(np.median(nearest_distance)),
            "p01": close_threshold,
            "p05": float(np.quantile(nearest_distance, 0.05)),
            "p95": float(np.quantile(nearest_distance, 0.95)),
        },
        "nearest_return_absolute_difference": {
            "mean": float(nearest_return_difference.mean()),
            "median": float(np.median(nearest_return_difference)),
            "p90": float(np.quantile(nearest_return_difference, 0.90)),
            "p95": float(np.quantile(nearest_return_difference, 0.95)),
            "p99": float(np.quantile(nearest_return_difference, 0.99)),
        },
        "local_return_std": {
            "mean": float(local_return_std.mean()),
            "median": float(np.median(local_return_std)),
            "p95": float(np.quantile(local_return_std, 0.95)),
        },
        "local_return_range": {
            "mean": float(local_return_range.mean()),
            "p95": float(np.quantile(local_return_range, 0.95)),
        },
        "closest_one_percent": selected_metrics(close),
        "closest_one_percent_with_position_gap_ge_500m": selected_metrics(close_far_position),
        "closest_one_percent_with_boundary_mismatch": selected_metrics(close_boundary_mismatch),
        "nearest_boundary_mismatch_rate": float(boundary_mismatch.mean()),
    }


def ambiguity_command(args: argparse.Namespace) -> None:
    dataset = DiagnosticDataset.load(args.dataset)
    result = {
        "dataset": str(Path(args.dataset).resolve()),
        "transitions": dataset.transition_count,
        "trajectories": dataset.trajectory_count,
        "state_action_10d": _ambiguity_summary(
            dataset,
            dataset.features,
            neighbor_count=args.neighbors,
            max_query_samples=args.max_query_samples,
        ),
        "state_only_7d": _ambiguity_summary(
            dataset,
            dataset.states,
            neighbor_count=args.neighbors,
            max_query_samples=args.max_query_samples,
        ),
        "interpretation_scope": (
            "Nearest neighbors are restricted to different trajectories and standardized by "
            "held-out feature variance; this is an empirical ambiguity diagnostic, not an "
            "identifiability theorem."
        ),
    }
    write_json(Path(args.output), result)
    print(json.dumps(json_value(result), indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose UAV Energy TD divergence")
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect = subparsers.add_parser("collect")
    collect.add_argument("--sac-checkpoint", required=True)
    collect.add_argument("--td-checkpoint", required=True)
    collect.add_argument("--heldout-tasks", required=True)
    collect.add_argument("--output-dir", required=True)
    collect.add_argument("--train-tasks", type=int, default=200)
    collect.add_argument("--train-task-seed", type=int, default=130_001)
    collect.add_argument("--train-rollout-seed", type=int, default=131_001)
    collect.add_argument("--heldout-rollout-seed", type=int, default=132_001)
    collect.add_argument("--max-steps", type=int, default=4000)
    collect.add_argument("--device", default="cuda")
    collect.set_defaults(function=collect_command)

    matrix = subparsers.add_parser("matrix")
    matrix.add_argument("--train-dataset", required=True)
    matrix.add_argument("--heldout-dataset", required=True)
    matrix.add_argument("--output-dir", required=True)
    matrix.add_argument(
        "--methods",
        default="current_quantile_td,scalar_td,mc_scalar,mc_quantile",
    )
    matrix.add_argument("--updates", type=int, default=100_000)
    matrix.add_argument("--batch-size", type=int, default=128)
    matrix.add_argument("--replay-capacity", type=int, default=100_000)
    matrix.add_argument("--learning-starts", type=int, default=512)
    matrix.add_argument("--learning-rate", type=float, default=3e-4)
    matrix.add_argument("--target-tau", type=float, default=0.01)
    matrix.add_argument("--log-every", type=int, default=5_000)
    matrix.add_argument("--seed", type=int, default=0)
    matrix.add_argument("--device", default="cuda")
    matrix.set_defaults(function=matrix_command)

    synthetic = subparsers.add_parser("synthetic")
    synthetic.add_argument("--output-dir", required=True)
    synthetic.add_argument("--updates", type=int, default=25_000)
    synthetic.add_argument("--long-updates", type=int, default=100_000)
    synthetic.add_argument("--batch-size", type=int, default=128)
    synthetic.add_argument("--learning-rate", type=float, default=3e-4)
    synthetic.add_argument("--target-tau", type=float, default=0.01)
    synthetic.add_argument("--log-every", type=int, default=5_000)
    synthetic.add_argument("--seed", type=int, default=0)
    synthetic.add_argument("--device", default="cuda")
    synthetic.set_defaults(function=synthetic_command)

    rollout = subparsers.add_parser("rollout")
    rollout.add_argument("--dataset", required=True)
    rollout.add_argument("--sac-checkpoint", required=True)
    rollout.add_argument("--output-dir", required=True)
    rollout.add_argument("--samples", type=int, default=100)
    rollout.add_argument("--max-steps", type=int, default=4000)
    rollout.add_argument("--seed", type=int, default=140_001)
    rollout.add_argument("--device", default="cuda")
    rollout.set_defaults(function=rollout_command)

    ambiguity = subparsers.add_parser("ambiguity")
    ambiguity.add_argument("--dataset", required=True)
    ambiguity.add_argument("--output", required=True)
    ambiguity.add_argument("--neighbors", type=int, default=20)
    ambiguity.add_argument("--max-query-samples", type=int, default=20_000)
    ambiguity.set_defaults(function=ambiguity_command)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
