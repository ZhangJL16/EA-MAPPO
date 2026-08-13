from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import platform
import shlex
import sys
import traceback

import numpy as np
import torch
from torch.nn import functional as functional

from agents.goal_conditioned_sac import FrozenGoalConditionedSAC
from envs.navigation import NavigationEnv
from safety.calibration import SplitConformalUpperBound
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

from .provenance import append_jsonl, code_hash, git_sha, sha256_file, utc_now, write_json


ROOT = Path(__file__).resolve().parents[2]
QUANTILES = (0.50, 0.90, 0.95, 0.99)


def initialize(args: argparse.Namespace) -> None:
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty run directory: {output}")
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = Path(args.checkpoint).resolve()
    execution_argv = [argument for argument in sys.argv if argument != "--initialize-only"]
    config = {
        "status": "INITIALIZED",
        "experiment": "E1_OPEN_WORLD_ENERGY_TO_CHARGER",
        "comparison_methods": [
            "distance_empirical_cost_per_meter",
            "monte_carlo_return_regression",
            "scalar_td_gamma_1",
            "distributional_quantile_td_gamma_1",
            "distributional_quantile_td_plus_split_conformal",
        ],
        "navigation_policy": "FROZEN",
        "seed": args.seed,
        "sorties": args.sorties,
        "max_return_steps": args.max_return_steps,
        "training_updates": args.training_updates,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "git_commit_sha": git_sha(ROOT),
        "code_hash": code_hash(ROOT),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "device_requested": args.device,
        "exact_command": shlex.join([sys.executable, *execution_argv]),
        "initialized_at": utc_now(),
        "train_calibration_test_split": [0.60, 0.20, 0.20],
        "heldout_split_unit": "sortie",
        "gamma_energy": 1.0,
        "alpha_energy": 0.05,
        "invalid_for_comparison_on_interrupt": True,
    }
    write_json(output / "config.json", config)
    write_json(output / "RUNNING.json", {"status": "INITIALIZED", "initialized_at": config["initialized_at"]})


def _set_charger_goal(environment: NavigationEnv) -> np.ndarray:
    environment.goal = environment.scenario.station_position.copy()
    return environment._observation()


def collect_episode(
    policy: FrozenGoalConditionedSAC,
    *,
    seed: int,
    max_return_steps: int,
    policy_hash: str,
) -> EnergyReturnEpisode:
    environment = NavigationEnv(max_episode_steps=max_return_steps + 1)
    observation, _ = environment.reset(seed=seed)
    transitions: list[EnergyTransition] = []
    action = policy.action(observation, deterministic=True)
    next_observation, _, _, _, info = environment.step(action)
    charger_hit = bool(np.linalg.norm(environment.state.position - environment.scenario.station_position) <= environment.goal_radius)
    if not charger_hit:
        next_observation = _set_charger_goal(environment)
    transitions.append(
        EnergyTransition(observation.copy(), action.copy(), float(info["energy_usage"]), next_observation.copy(), charger_hit)
    )
    completed = charger_hit
    observation = next_observation
    for _ in range(max_return_steps - 1):
        if completed:
            break
        action = policy.action(observation, deterministic=True)
        next_observation, _, _, _, info = environment.step(action)
        charger_hit = bool(info["distance_to_goal_after"] <= environment.goal_radius + 1e-12)
        transitions.append(
            EnergyTransition(
                observation.copy(),
                action.copy(),
                float(info["energy_usage"]),
                next_observation.copy(),
                charger_hit,
            )
        )
        completed = charger_hit
        observation = next_observation
    environment.close()
    return EnergyReturnEpisode(
        transitions=tuple(transitions),
        completed=completed,
        policy_hash=policy_hash,
        context={"environment": "random_persistent_open", "seed": float(seed)},
    )


def _episode_payload(index: int, episode: EnergyReturnEpisode) -> dict:
    return {
        "sortie_index": index,
        "completed": episode.completed,
        "censored": not episode.completed,
        "policy_hash": episode.policy_hash,
        "context": episode.context,
        "transition_count": len(episode.transitions),
        "total_energy": float(sum(transition.cost for transition in episode.transitions)),
        "transitions": [
            {
                "state": transition.state.tolist(),
                "action": transition.action.tolist(),
                "cost": transition.cost,
                "next_state": transition.next_state.tolist(),
                "charger_hit": transition.charger_hit,
            }
            for transition in episode.transitions
        ],
    }


def _completed_rows(episodes: list[EnergyReturnEpisode]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    next_features: list[np.ndarray] = []
    costs: list[float] = []
    terminals: list[bool] = []
    returns: list[float] = []
    for episode in episodes:
        if not episode.completed:
            continue
        return_values = returns_to_go(np.asarray([transition.cost for transition in episode.transitions]))
        for index, transition in enumerate(episode.transitions):
            features.append(np.concatenate((transition.state, transition.action)))
            costs.append(transition.cost)
            terminals.append(transition.charger_hit)
            returns.append(float(return_values[index]))
            if transition.charger_hit:
                next_action = np.zeros_like(transition.action)
            else:
                next_action = episode.transitions[index + 1].action
            next_features.append(np.concatenate((transition.next_state, next_action)))
    if not features:
        raise RuntimeError("no completed return transitions available")
    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(next_features, dtype=np.float32),
        np.asarray(costs, dtype=np.float32),
        np.asarray(terminals, dtype=bool),
        np.asarray(returns, dtype=np.float32),
    )


def _train_scalar_supervised(features: np.ndarray, labels: np.ndarray, updates: int) -> ScalarEnergyCritic:
    model = ScalarEnergyCritic(features.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.from_numpy(features)
    y = torch.from_numpy(labels)
    for _ in range(updates):
        loss = functional.mse_loss(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model


def _train_scalar_td(
    features: np.ndarray,
    next_features: np.ndarray,
    costs: np.ndarray,
    terminals: np.ndarray,
    updates: int,
) -> ScalarEnergyCritic:
    model = ScalarEnergyCritic(features.shape[1])
    target = ScalarEnergyCritic(features.shape[1])
    target.load_state_dict(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.from_numpy(features)
    next_x = torch.from_numpy(next_features)
    cost = torch.from_numpy(costs)
    terminal = torch.from_numpy(terminals)
    for update in range(updates):
        with torch.no_grad():
            y = scalar_ssp_target(cost, target(next_x), terminal)
        loss = functional.mse_loss(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (update + 1) % 20 == 0:
            target.load_state_dict(model.state_dict())
    return model


def _train_quantile_td(
    features: np.ndarray,
    next_features: np.ndarray,
    costs: np.ndarray,
    terminals: np.ndarray,
    updates: int,
) -> MonotoneQuantileCritic:
    model = MonotoneQuantileCritic(features.shape[1], QUANTILES)
    target = MonotoneQuantileCritic(features.shape[1], QUANTILES)
    target.load_state_dict(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.from_numpy(features)
    next_x = torch.from_numpy(next_features)
    cost = torch.from_numpy(costs)
    terminal = torch.from_numpy(terminals)
    levels = torch.tensor(QUANTILES, dtype=torch.float32)
    for update in range(updates):
        with torch.no_grad():
            y = quantile_ssp_target(cost, target(next_x), terminal)
        loss = quantile_huber_loss(model(x), y, levels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (update + 1) % 20 == 0:
            target.load_state_dict(model.state_dict())
    return model


def _regression_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    error = np.asarray(prediction) - np.asarray(target)
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(math.sqrt(np.mean(error**2))),
        "underestimation_rate": float(np.mean(error < 0.0)),
    }


def execute(args: argparse.Namespace) -> None:
    output = Path(args.output_dir)
    config_path = output / "config.json"
    if not config_path.exists() or (output / "COMPLETED.json").exists() or (output / "FAILED.json").exists():
        raise RuntimeError("run is not in an initialized executable state")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    started_at = utc_now()
    write_json(output / "RUNNING.json", {"status": "RUNNING", "pid": os.getpid(), "started_at": started_at})
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    policy = FrozenGoalConditionedSAC(args.checkpoint, device=args.device)
    policy_hash = config["checkpoint_sha256"]
    episodes: list[EnergyReturnEpisode] = []
    for index in range(args.sorties):
        episode = collect_episode(
            policy,
            seed=args.seed * 1_000_000 + 10_000 + index,
            max_return_steps=args.max_return_steps,
            policy_hash=policy_hash,
        )
        episodes.append(episode)
        append_jsonl(output / "raw_sorties.jsonl", _episode_payload(index, episode))
    indices = np.random.default_rng(args.seed).permutation(len(episodes))
    train_end = max(1, int(0.60 * len(indices)))
    calibration_end = max(train_end + 1, int(0.80 * len(indices)))
    train_episodes = [episodes[index] for index in indices[:train_end]]
    calibration_episodes = [episodes[index] for index in indices[train_end:calibration_end]]
    test_episodes = [episodes[index] for index in indices[calibration_end:]]
    train = _completed_rows(train_episodes)
    calibration = _completed_rows(calibration_episodes)
    test = _completed_rows(test_episodes)
    mc_model = _train_scalar_supervised(train[0], train[4], args.training_updates)
    scalar_td = _train_scalar_td(train[0], train[1], train[2], train[3], args.training_updates)
    quantile_td = _train_quantile_td(train[0], train[1], train[2], train[3], args.training_updates)
    with torch.no_grad():
        mc_prediction = mc_model(torch.from_numpy(test[0])).numpy()
        scalar_prediction = scalar_td(torch.from_numpy(test[0])).numpy()
        calibration_quantiles = quantile_td(torch.from_numpy(calibration[0])).numpy()
        test_quantiles = quantile_td(torch.from_numpy(test[0])).numpy()
    calibrator = SplitConformalUpperBound(alpha=0.05).fit(calibration_quantiles[:, 2], calibration[4])
    calibrated_upper = calibrator.predict(test_quantiles[:, 2])
    world = np.array([4.0, 4.0, 2.0], dtype=np.float32)
    station = np.array([0.40, 0.50, 1.0], dtype=np.float32)
    train_steps = np.linalg.norm((train[1][:, :3] - train[0][:, :3]) * world, axis=1)
    cost_per_meter = float(np.sum(train[2]) / max(np.sum(train_steps), 1e-8))
    successor_position = test[1][:, :3] * world
    distance_prediction = test[2] + np.linalg.norm(successor_position - station, axis=1) * cost_per_meter
    result = {
        "status": "COMPLETED",
        "completed_at": utc_now(),
        "sorties": len(episodes),
        "completed_sorties": sum(episode.completed for episode in episodes),
        "censored_sorties": sum(not episode.completed for episode in episodes),
        "split_sortie_counts": [len(train_episodes), len(calibration_episodes), len(test_episodes)],
        "distance_baseline": _regression_metrics(distance_prediction, test[4]),
        "monte_carlo": _regression_metrics(mc_prediction, test[4]),
        "scalar_td": _regression_metrics(scalar_prediction, test[4]),
        "distributional_median": _regression_metrics(test_quantiles[:, 0], test[4]),
        "quantile_coverage": {
            str(level): float(np.mean(test[4] <= test_quantiles[:, index]))
            for index, level in enumerate(QUANTILES)
        },
        "calibrated_upper": SplitConformalUpperBound.report(calibrated_upper, test[4]).__dict__,
        "conformal_correction": calibrator.correction,
        "cost_per_meter_train": cost_per_meter,
    }
    torch.save(mc_model.state_dict(), output / "mc_model.pt")
    torch.save(scalar_td.state_dict(), output / "scalar_td.pt")
    torch.save(quantile_td.state_dict(), output / "quantile_td.pt")
    write_json(output / "results.json", result)
    write_json(output / "COMPLETED.json", result)
    write_json(output / "RUNNING.json", {"status": "COMPLETED", "completed_at": result["completed_at"]})


def run(args: argparse.Namespace) -> None:
    try:
        if args.initialize_only:
            initialize(args)
        else:
            execute(args)
    except Exception as error:
        output = Path(args.output_dir)
        if output.exists():
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="E1 energy-to-charger estimator comparison")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sorties", type=int, default=100)
    parser.add_argument("--max-return-steps", type=int, default=800)
    parser.add_argument("--training-updates", type=int, default=800)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--initialize-only", action="store_true")
    args = parser.parse_args()
    if args.sorties < 15 or args.max_return_steps <= 0 or args.training_updates <= 0:
        parser.error("formal E1 requires at least 15 sorties and positive budgets")
    return args
