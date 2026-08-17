from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch import nn
from torch.nn import functional

from review_bundle.safety.energy.critics import MonotoneQuantileCritic, ScalarEnergyCritic
from review_bundle.safety.energy.td import (
    quantile_atom_weights,
    quantile_huber_loss,
    quantile_ssp_target,
    scalar_ssp_target,
)


ENERGY_QUANTILES = (0.50, 0.90, 0.95, 0.99)
EstimatorKind = Literal[
    "scalar_td",
    "quantile_td",
    "mc_scalar",
    "mc_quantile",
    "n_step_scalar_td",
    "uniform_quantile_td",
    "free_four_quantile_td",
    "unweighted_four_quantile_td",
    "uniform_four_quantile_td",
]


class UniformQuantileCritic(nn.Module):
    def __init__(self, input_dim: int, quantile_count: int = 32, hidden_dim: int = 128) -> None:
        super().__init__()
        if input_dim <= 0 or quantile_count <= 1:
            raise ValueError("uniform quantile critic requires positive input and multiple atoms")
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, quantile_count),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


class FreeFourQuantileCritic(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden_dim, len(ENERGY_QUANTILES))
        nn.init.zeros_(self.head.weight)
        with torch.no_grad():
            self.head.bias.copy_(torch.tensor([1.0, 1.1, 1.2, 1.3]))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(features))


@dataclass(frozen=True)
class DiagnosticDataset:
    states: np.ndarray
    actions: np.ndarray
    costs: np.ndarray
    next_states: np.ndarray
    next_actions: np.ndarray
    terminals: np.ndarray
    returns: np.ndarray
    horizons: np.ndarray
    trajectory_ids: np.ndarray
    step_indices: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray
    goals: np.ndarray
    distances: np.ndarray
    initial_distances: np.ndarray
    boundary_contacts: np.ndarray

    def __post_init__(self) -> None:
        transition_count = int(self.costs.shape[0])
        if transition_count <= 0:
            raise ValueError("diagnostic dataset cannot be empty")
        expected_rows = {
            "states": self.states,
            "actions": self.actions,
            "next_states": self.next_states,
            "next_actions": self.next_actions,
            "terminals": self.terminals,
            "returns": self.returns,
            "horizons": self.horizons,
            "trajectory_ids": self.trajectory_ids,
            "step_indices": self.step_indices,
            "positions": self.positions,
            "velocities": self.velocities,
            "goals": self.goals,
            "distances": self.distances,
            "initial_distances": self.initial_distances,
            "boundary_contacts": self.boundary_contacts,
        }
        for name, values in expected_rows.items():
            if values.shape[0] != transition_count:
                raise ValueError(f"{name} does not align with transition count")
        if self.states.ndim != 2 or self.states.shape[1] != 7:
            raise ValueError("energy states must be [N, 7]")
        if self.next_states.shape != self.states.shape:
            raise ValueError("next states must align with states")
        if self.actions.ndim != 2 or self.actions.shape[1] != 3:
            raise ValueError("actions must be [N, 3]")
        if self.next_actions.shape != self.actions.shape:
            raise ValueError("next actions must align with actions")
        if np.any(self.costs < 0.0) or not np.all(np.isfinite(self.costs)):
            raise ValueError("costs must be finite and nonnegative")
        if not np.all(np.isfinite(self.returns)) or np.any(self.returns < 0.0):
            raise ValueError("returns must be finite and nonnegative")
        reconstructed = compute_mc_returns(self.costs, self.trajectory_ids, self.terminals)
        if not np.allclose(reconstructed, self.returns, rtol=1e-5, atol=1e-6):
            raise ValueError("stored returns do not match reconstructed Monte-Carlo returns")

    @property
    def transition_count(self) -> int:
        return int(self.costs.shape[0])

    @property
    def trajectory_count(self) -> int:
        return int(np.unique(self.trajectory_ids).size)

    @property
    def features(self) -> np.ndarray:
        return np.concatenate((self.states, self.actions), axis=1).astype(np.float32, copy=False)

    @property
    def next_features(self) -> np.ndarray:
        return np.concatenate((self.next_states, self.next_actions), axis=1).astype(
            np.float32, copy=False
        )

    def subset(self, indices: np.ndarray) -> "DiagnosticDataset":
        selected = np.asarray(indices, dtype=np.int64)
        return DiagnosticDataset(
            **{
                name: np.asarray(getattr(self, name))[selected]
                for name in self.__dataclass_fields__
            }
        )

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination,
            **{name: getattr(self, name) for name in self.__dataclass_fields__},
        )

    @classmethod
    def load(cls, path: str | Path) -> "DiagnosticDataset":
        with np.load(Path(path), allow_pickle=False) as payload:
            return cls(**{name: payload[name] for name in cls.__dataclass_fields__})


@dataclass(frozen=True)
class TrainingConfig:
    kind: EstimatorKind
    updates: int = 100_000
    batch_size: int = 128
    replay_capacity: int = 100_000
    learning_starts: int = 512
    learning_rate: float = 3e-4
    target_tau: float = 0.01
    n_step: int = 1
    terminal_fraction_per_batch: float = 0.0
    hidden_dim: int = 128
    seed: int = 0
    device: str = "cpu"
    log_every: int = 5_000
    value_alarm_factor: float = 100.0
    include_action: bool = True
    target_update: Literal["polyak", "hard"] = "polyak"
    hard_update_interval: int = 0

    def __post_init__(self) -> None:
        if self.updates <= 0 or self.batch_size <= 0 or self.replay_capacity < self.batch_size:
            raise ValueError("invalid update, batch, or replay configuration")
        if self.learning_starts < self.batch_size:
            raise ValueError("learning_starts must be at least one batch")
        if self.learning_rate <= 0.0 or not 0.0 < self.target_tau <= 1.0:
            raise ValueError("invalid optimizer or target update configuration")
        if self.n_step <= 0:
            raise ValueError("n_step must be positive")
        if not 0.0 <= self.terminal_fraction_per_batch < 1.0:
            raise ValueError("terminal fraction must lie in [0, 1)")
        if self.kind == "n_step_scalar_td" and self.n_step == 1:
            raise ValueError("n-step scalar TD requires n_step > 1")
        if self.kind != "n_step_scalar_td" and self.n_step != 1:
            raise ValueError("n_step only applies to n_step_scalar_td")
        if self.target_update == "hard" and self.hard_update_interval <= 0:
            raise ValueError("hard target updates require a positive interval")
        if self.target_update == "polyak" and self.hard_update_interval != 0:
            raise ValueError("Polyak target updates do not use a hard interval")


def compute_mc_returns(
    costs: np.ndarray,
    trajectory_ids: np.ndarray,
    terminals: np.ndarray,
) -> np.ndarray:
    energy_costs = np.asarray(costs, dtype=np.float64)
    identifiers = np.asarray(trajectory_ids)
    terminal_flags = np.asarray(terminals, dtype=bool)
    if energy_costs.ndim != 1 or identifiers.shape != energy_costs.shape:
        raise ValueError("costs and trajectory ids must be aligned vectors")
    if terminal_flags.shape != energy_costs.shape:
        raise ValueError("terminal flags must align with costs")
    returns = np.empty_like(energy_costs)
    for trajectory_id in np.unique(identifiers):
        indices = np.flatnonzero(identifiers == trajectory_id)
        if indices.size == 0 or not np.all(np.diff(indices) == 1):
            raise ValueError("each trajectory must occupy one contiguous block")
        if terminal_flags[indices].sum() != 1 or not terminal_flags[indices[-1]]:
            raise ValueError("each successful trajectory must end in exactly one terminal")
        returns[indices] = np.cumsum(energy_costs[indices][::-1], dtype=np.float64)[::-1]
    return returns.astype(np.float32)


def build_n_step_targets(
    dataset: DiagnosticDataset,
    n_step: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if n_step <= 0:
        raise ValueError("n_step must be positive")
    cumulative_costs = np.zeros(dataset.transition_count, dtype=np.float32)
    bootstrap_indices = np.zeros(dataset.transition_count, dtype=np.int64)
    terminal_within_window = np.zeros(dataset.transition_count, dtype=bool)
    for trajectory_id in np.unique(dataset.trajectory_ids):
        indices = np.flatnonzero(dataset.trajectory_ids == trajectory_id)
        trajectory_end = int(indices[-1])
        for current_index in indices:
            window_end = min(int(current_index) + n_step - 1, trajectory_end)
            cumulative_costs[current_index] = float(dataset.costs[current_index : window_end + 1].sum())
            terminal_within_window[current_index] = bool(window_end == trajectory_end)
            bootstrap_indices[current_index] = window_end
    return cumulative_costs, bootstrap_indices, terminal_within_window


def terminal_anchor_statistics(
    terminals: np.ndarray,
    trajectory_ids: np.ndarray,
    *,
    batch_size: int,
) -> dict[str, float | int]:
    flags = np.asarray(terminals, dtype=bool)
    identifiers = np.asarray(trajectory_ids)
    if flags.ndim != 1 or identifiers.shape != flags.shape or batch_size <= 0:
        raise ValueError("invalid terminal statistics inputs")
    terminal_fraction = float(flags.mean())
    lengths = np.asarray(
        [np.sum(identifiers == trajectory_id) for trajectory_id in np.unique(identifiers)],
        dtype=np.int64,
    )
    no_terminal_probability = float((1.0 - terminal_fraction) ** batch_size)
    return {
        "total_trainable_transitions": int(flags.size),
        "terminal_transitions": int(flags.sum()),
        "terminal_fraction": terminal_fraction,
        "mean_segment_length": float(lengths.mean()),
        "median_segment_length": float(np.median(lengths)),
        "p90_segment_length": float(np.quantile(lengths, 0.90)),
        "p95_segment_length": float(np.quantile(lengths, 0.95)),
        "max_segment_length": int(lengths.max()),
        "batch_size": int(batch_size),
        "probability_batch_has_zero_terminal": no_terminal_probability,
        "probability_batch_has_at_least_one_terminal": 1.0 - no_terminal_probability,
        "mean_terminal_samples_per_batch": float(batch_size * terminal_fraction),
    }


def value_scale_alarm(
    predictions: np.ndarray,
    empirical_returns: np.ndarray,
    *,
    factor: float = 100.0,
) -> dict[str, float | bool]:
    predicted = np.asarray(predictions, dtype=np.float64)
    returns = np.asarray(empirical_returns, dtype=np.float64)
    if predicted.size == 0 or returns.size == 0 or factor <= 0.0:
        raise ValueError("value scale alarm requires nonempty arrays and a positive factor")
    predicted_median = float(np.median(predicted))
    empirical_p99 = float(np.quantile(returns, 0.99))
    threshold = factor * max(empirical_p99, np.finfo(np.float64).eps)
    return {
        "predicted_median": predicted_median,
        "empirical_mc_p99": empirical_p99,
        "alarm_factor": float(factor),
        "alarm_threshold": threshold,
        "prediction_to_empirical_p99_ratio": predicted_median
        / max(empirical_p99, np.finfo(np.float64).eps),
        "VALUE_SCALE_DIVERGENCE": bool(predicted_median > threshold),
    }


def _parameter_norm(model: nn.Module) -> float:
    squared = sum(float(parameter.detach().square().sum().cpu()) for parameter in model.parameters())
    return math.sqrt(squared)


def _gradient_norm(model: nn.Module) -> float:
    squared = sum(
        float(parameter.grad.detach().square().sum().cpu())
        for parameter in model.parameters()
        if parameter.grad is not None
    )
    return math.sqrt(squared)


def _module_gradient_norm(module: nn.Module) -> float:
    squared = sum(
        float(parameter.grad.detach().square().sum().cpu())
        for parameter in module.parameters()
        if parameter.grad is not None
    )
    return math.sqrt(squared)


def _batched_predictions(
    model: nn.Module,
    features: np.ndarray,
    *,
    device: torch.device,
    batch_size: int = 8192,
) -> np.ndarray:
    output: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, features.shape[0], batch_size):
            values = torch.as_tensor(
                features[start : start + batch_size], dtype=torch.float32, device=device
            )
            output.append(model(values).detach().cpu().numpy())
    return np.concatenate(output, axis=0)


def _regression_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    predicted = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    errors = predicted - truth
    return {
        "MAE": float(np.mean(np.abs(errors))),
        "RMSE": float(np.sqrt(np.mean(np.square(errors)))),
        "bias": float(np.mean(errors)),
        "underestimation_rate": float(np.mean(errors < 0.0)),
        "mean_underestimation_magnitude": float(np.mean(np.maximum(-errors, 0.0))),
    }


def _sample_batch_indices(
    rng: np.random.Generator,
    available_indices: np.ndarray,
    terminals: np.ndarray,
    batch_size: int,
    terminal_fraction: float,
    *,
    global_terminal_indices: np.ndarray | None = None,
    global_nonterminal_indices: np.ndarray | None = None,
) -> np.ndarray:
    if terminal_fraction <= 0.0:
        return rng.choice(available_indices, size=batch_size, replace=True)
    terminal_indices = (
        available_indices[terminals[available_indices]]
        if global_terminal_indices is None
        else global_terminal_indices
    )
    nonterminal_indices = (
        available_indices[~terminals[available_indices]]
        if global_nonterminal_indices is None
        else global_nonterminal_indices
    )
    if terminal_indices.size == 0 or nonterminal_indices.size == 0:
        return rng.choice(available_indices, size=batch_size, replace=True)
    terminal_count = max(1, int(round(batch_size * terminal_fraction)))
    nonterminal_count = batch_size - terminal_count
    selected = np.concatenate(
        (
            rng.choice(terminal_indices, size=terminal_count, replace=True),
            rng.choice(nonterminal_indices, size=nonterminal_count, replace=True),
        )
    )
    rng.shuffle(selected)
    return selected


def _quantile_metrics(
    predictions: np.ndarray,
    returns: np.ndarray,
    quantile_levels: tuple[float, ...] = ENERGY_QUANTILES,
) -> dict[str, object]:
    median_index = int(np.argmin(np.abs(np.asarray(quantile_levels) - 0.5)))
    return {
        "median": _regression_metrics(predictions[:, median_index], returns),
        "median_index": median_index,
        "coverage": {
            str(level): float(np.mean(returns <= predictions[:, index]))
            for index, level in enumerate(quantile_levels)
        },
        "quantile_means": predictions.mean(axis=0).tolist(),
        "quantile_p99": np.quantile(predictions, 0.99, axis=0).tolist(),
        "quantile_max": predictions.max(axis=0).tolist(),
        "increment_means": np.diff(predictions, axis=1).mean(axis=0).tolist(),
    }


def _grouped_error_metrics(
    prediction: np.ndarray,
    dataset: DiagnosticDataset,
) -> dict[str, object]:
    horizon_bins = (
        ("horizon_1", 1, 1),
        ("horizon_2_5", 2, 5),
        ("horizon_6_20", 6, 20),
        ("horizon_21_100", 21, 100),
        ("horizon_101_500", 101, 500),
        ("horizon_gt_500", 501, np.iinfo(np.int32).max),
    )
    distance_bins = (
        ("distance_0_500", 0.0, 500.0),
        ("distance_500_1500", 500.0, 1500.0),
        ("distance_1500_2500", 1500.0, 2500.0),
        ("distance_2500_4000", 2500.0, 4000.0),
        ("distance_gt_4000", 4000.0, float("inf")),
    )

    def summarize(mask: np.ndarray) -> dict[str, object]:
        if not np.any(mask):
            return {"count": 0}
        return {"count": int(mask.sum()), **_regression_metrics(prediction[mask], dataset.returns[mask])}

    return {
        "terminal": summarize(dataset.terminals),
        "by_horizon": {
            name: summarize((dataset.horizons >= lower) & (dataset.horizons <= upper))
            for name, lower, upper in horizon_bins
        },
        "by_distance": {
            name: summarize((dataset.distances >= lower) & (dataset.distances < upper))
            for name, lower, upper in distance_bins
        },
    }


def train_diagnostic_estimator(
    train: DiagnosticDataset,
    heldout: DiagnosticDataset,
    config: TrainingConfig,
    *,
    output_dir: str | Path | None = None,
) -> dict[str, object]:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    rng = np.random.default_rng(config.seed)
    device = torch.device(config.device)
    train_features = train.features if config.include_action else train.states
    train_next_features = train.next_features if config.include_action else train.next_states
    heldout_features = heldout.features if config.include_action else heldout.states
    heldout_next_features = heldout.next_features if config.include_action else heldout.next_states
    input_dim = int(train_features.shape[1])
    uniform_quantile_kind = config.kind == "uniform_quantile_td"
    uniform_four_quantile_kind = config.kind == "uniform_four_quantile_td"
    free_four_quantile_kind = config.kind == "free_four_quantile_td"
    unweighted_four_quantile_kind = config.kind == "unweighted_four_quantile_td"
    quantile_kind = config.kind in {
        "quantile_td",
        "mc_quantile",
        "uniform_quantile_td",
        "free_four_quantile_td",
        "unweighted_four_quantile_td",
        "uniform_four_quantile_td",
    }
    quantile_levels = (
        tuple((index + 0.5) / 32.0 for index in range(32))
        if uniform_quantile_kind
        else (0.125, 0.375, 0.625, 0.875)
        if uniform_four_quantile_kind
        else ENERGY_QUANTILES
    )
    if uniform_quantile_kind:
        model = UniformQuantileCritic(input_dim, 32, hidden_dim=config.hidden_dim).to(device)
        target_model = UniformQuantileCritic(input_dim, 32, hidden_dim=config.hidden_dim).to(device)
    elif free_four_quantile_kind:
        model = FreeFourQuantileCritic(input_dim, hidden_dim=config.hidden_dim).to(device)
        target_model = FreeFourQuantileCritic(input_dim, hidden_dim=config.hidden_dim).to(device)
    elif quantile_kind:
        model: nn.Module = MonotoneQuantileCritic(
            input_dim,
            ENERGY_QUANTILES,
            hidden_dim=config.hidden_dim,
            initial_base=1.0,
            initial_increment=0.1,
        ).to(device)
        target_model: nn.Module = MonotoneQuantileCritic(
            input_dim,
            ENERGY_QUANTILES,
            hidden_dim=config.hidden_dim,
            initial_base=1.0,
            initial_increment=0.1,
        ).to(device)
    else:
        model = ScalarEnergyCritic(
            input_dim, hidden_dim=config.hidden_dim, initial_output=1.0
        ).to(device)
        target_model = ScalarEnergyCritic(
            input_dim, hidden_dim=config.hidden_dim, initial_output=1.0
        ).to(device)
    target_model.load_state_dict(model.state_dict())
    target_model.eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    levels = torch.tensor(quantile_levels, dtype=torch.float32, device=device)
    atom_weights = (
        None
        if uniform_quantile_kind
        or uniform_four_quantile_kind
        or unweighted_four_quantile_kind
        else quantile_atom_weights(levels)
    )
    n_step_costs, n_step_bootstrap_indices, n_step_terminals = build_n_step_targets(
        train, config.n_step
    )
    replay_index_buffer = np.empty(config.replay_capacity, dtype=np.int64)
    replay_size = 0
    replay_position = 0
    global_terminal_indices = np.flatnonzero(train.terminals)
    global_nonterminal_indices = np.flatnonzero(~train.terminals)
    total_source_transitions = train.transition_count
    source_cursor = 0
    trace: list[dict[str, object]] = []
    last_loss = float("nan")
    last_gradient_norm = 0.0
    last_head_gradient_norms: dict[str, float] = {}
    checkpoints = {0, config.updates}
    checkpoints.update(range(config.log_every, config.updates + 1, config.log_every))
    empirical_scale = {
        "median": float(np.median(heldout.returns)),
        "p90": float(np.quantile(heldout.returns, 0.90)),
        "p95": float(np.quantile(heldout.returns, 0.95)),
        "p99": float(np.quantile(heldout.returns, 0.99)),
        "max": float(np.max(heldout.returns)),
    }

    def evaluate(update: int) -> dict[str, object]:
        predictions = _batched_predictions(model, heldout_features, device=device)
        target_predictions = _batched_predictions(target_model, heldout_features, device=device)
        if predictions.ndim == 1:
            primary = predictions
            target_primary = target_predictions
            metrics: dict[str, object] = _regression_metrics(primary, heldout.returns)
        else:
            primary = predictions[:, 0]
            target_primary = target_predictions[:, 0]
            metrics = _quantile_metrics(predictions, heldout.returns, quantile_levels)
            median_index = int(metrics["median_index"])
            primary = predictions[:, median_index]
            target_primary = target_predictions[:, median_index]
        diagnostic_count = min(4096, heldout.transition_count)
        diagnostic_indices = np.linspace(
            0, heldout.transition_count - 1, diagnostic_count, dtype=np.int64
        )
        next_target = _batched_predictions(
            target_model,
            heldout_next_features[diagnostic_indices],
            device=device,
        )
        if next_target.ndim == 1:
            bellman_targets = (
                heldout.costs[diagnostic_indices]
                + (~heldout.terminals[diagnostic_indices]) * next_target
            )
        else:
            bellman_targets = heldout.costs[diagnostic_indices, None] + (
                ~heldout.terminals[diagnostic_indices, None]
            ) * next_target
        result = {
            "update": update,
            "loss": last_loss,
            "gradient_l2_norm": last_gradient_norm,
            "online_parameter_l2_norm": _parameter_norm(model),
            "target_parameter_l2_norm": _parameter_norm(target_model),
            "online_target_primary_gap_mean": float(np.mean(primary - target_primary)),
            "online_target_primary_gap_abs_mean": float(np.mean(np.abs(primary - target_primary))),
            "prediction_primary_mean": float(np.mean(primary)),
            "prediction_primary_median": float(np.median(primary)),
            "prediction_primary_p90": float(np.quantile(primary, 0.90)),
            "prediction_primary_p99": float(np.quantile(primary, 0.99)),
            "prediction_primary_max": float(np.max(primary)),
            "target_primary_mean": float(np.mean(target_primary)),
            "bellman_target_mean": float(np.mean(bellman_targets)),
            "bellman_target_p99": float(np.quantile(bellman_targets, 0.99)),
            "bellman_target_max": float(np.max(bellman_targets)),
            "metrics": metrics,
            "grouped_errors": _grouped_error_metrics(primary, heldout),
            "head_gradient_l2_norms": last_head_gradient_norms,
            "value_scale_alarm": value_scale_alarm(
                primary,
                heldout.returns,
                factor=config.value_alarm_factor,
            ),
        }
        if predictions.ndim == 2:
            result["online_quantiles"] = _quantile_metrics(
                predictions, heldout.returns, quantile_levels
            )
            result["target_quantiles"] = _quantile_metrics(
                target_predictions, heldout.returns, quantile_levels
            )
        if isinstance(model, MonotoneQuantileCritic):
            diagnostic_features = torch.as_tensor(
                heldout_features[diagnostic_indices], dtype=torch.float32, device=device
            )
            with torch.no_grad():
                hidden = model.backbone(diagnostic_features)
                raw_base = model.base_head(hidden).squeeze(-1)
                raw_increments = model.increment_head(hidden)
            result["monotone_head_outputs"] = {
                "raw_base_mean": float(raw_base.mean().cpu()),
                "raw_base_p99": float(torch.quantile(raw_base, 0.99).cpu()),
                "raw_increment_means": raw_increments.mean(dim=0).cpu().tolist(),
                "softplus_increment_means": functional.softplus(raw_increments)
                .mean(dim=0)
                .cpu()
                .tolist(),
            }
        return result

    trace.append(evaluate(0))
    for update in range(1, config.updates + 1):
        replay_index_buffer[replay_position] = source_cursor
        replay_position = (replay_position + 1) % config.replay_capacity
        replay_size = min(replay_size + 1, config.replay_capacity)
        source_cursor = (source_cursor + 1) % total_source_transitions
        if replay_size < config.learning_starts:
            if update in checkpoints:
                trace.append(evaluate(update))
            continue
        available = replay_index_buffer[:replay_size]
        batch_indices = _sample_batch_indices(
            rng,
            available,
            train.terminals,
            config.batch_size,
            config.terminal_fraction_per_batch,
            global_terminal_indices=(
                global_terminal_indices
                if config.terminal_fraction_per_batch > 0.0
                else None
            ),
            global_nonterminal_indices=(
                global_nonterminal_indices
                if config.terminal_fraction_per_batch > 0.0
                else None
            ),
        )
        features = torch.as_tensor(train_features[batch_indices], dtype=torch.float32, device=device)
        predictions = model(features)
        if config.kind in {"mc_scalar", "mc_quantile"}:
            labels = torch.as_tensor(train.returns[batch_indices], dtype=torch.float32, device=device)
            if config.kind == "mc_quantile":
                labels = labels.unsqueeze(-1)
        elif config.kind == "n_step_scalar_td":
            bootstrap_features = train_next_features[n_step_bootstrap_indices[batch_indices]]
            with torch.no_grad():
                next_values = target_model(
                    torch.as_tensor(bootstrap_features, dtype=torch.float32, device=device)
                )
            labels = scalar_ssp_target(
                torch.as_tensor(n_step_costs[batch_indices], dtype=torch.float32, device=device),
                next_values,
                torch.as_tensor(n_step_terminals[batch_indices], dtype=torch.bool, device=device),
            )
        else:
            next_features = torch.as_tensor(
                train_next_features[batch_indices], dtype=torch.float32, device=device
            )
            terminals = torch.as_tensor(
                train.terminals[batch_indices], dtype=torch.bool, device=device
            )
            costs = torch.as_tensor(train.costs[batch_indices], dtype=torch.float32, device=device)
            with torch.no_grad():
                next_values = target_model(next_features)
                if config.kind in {
                    "quantile_td",
                    "uniform_quantile_td",
                    "free_four_quantile_td",
                    "unweighted_four_quantile_td",
                    "uniform_four_quantile_td",
                }:
                    labels = quantile_ssp_target(costs, next_values, terminals)
                else:
                    labels = scalar_ssp_target(costs, next_values, terminals)
        if quantile_kind:
            loss = quantile_huber_loss(
                predictions,
                labels,
                levels,
                target_weights=atom_weights,
            )
        else:
            loss = functional.mse_loss(predictions, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        should_log = update in checkpoints
        if should_log:
            last_gradient_norm = _gradient_norm(model)
            if isinstance(model, MonotoneQuantileCritic):
                last_head_gradient_norms = {
                    "base_head": _module_gradient_norm(model.base_head),
                    "increment_head": _module_gradient_norm(model.increment_head),
                    "backbone": _module_gradient_norm(model.backbone),
                }
        optimizer.step()
        if config.kind not in {"mc_scalar", "mc_quantile"}:
            if config.target_update == "hard":
                if update % config.hard_update_interval == 0:
                    target_model.load_state_dict(model.state_dict())
            else:
                with torch.no_grad():
                    for target_parameter, parameter in zip(
                        target_model.parameters(), model.parameters(), strict=True
                    ):
                        target_parameter.mul_(1.0 - config.target_tau).add_(
                            parameter, alpha=config.target_tau
                        )
        else:
            target_model.load_state_dict(model.state_dict())
        if should_log:
            last_loss = float(loss.detach().cpu())
            trace.append(evaluate(update))

    result = {
        "config": asdict(config),
        "empirical_mc_scale": empirical_scale,
        "terminal_anchor": terminal_anchor_statistics(
            train.terminals, train.trajectory_ids, batch_size=config.batch_size
        ),
        "trace": trace,
        "final": trace[-1],
    }
    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        torch.save(
            {
                "config": asdict(config),
                "model_state_dict": model.state_dict(),
                "target_state_dict": target_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            destination / "model.pt",
        )
    return result
