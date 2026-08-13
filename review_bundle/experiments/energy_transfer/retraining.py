from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import shlex
import shutil
import sys
import traceback
from typing import Any

import numpy as np
import torch
from torch.nn import functional

from experiments.energy_transfer.analysis import (
    horizon_error_report,
    learned_value_decision,
    paired_sortie_comparisons,
    quantile_metrics,
    regression_metrics,
)
from experiments.energy_transfer.stage_a_bootstrap import EnergyDataset, QUANTILES
from experiments.new_route.provenance import code_hash, git_sha, sha256_file, utc_now, write_json
from safety.energy import (
    MonotoneQuantileCritic,
    QuantileEnergyPredictor,
    ScalarEnergyCritic,
    ScalarEnergyPredictor,
    quantile_atom_weights,
    quantile_huber_loss,
    quantile_ssp_target,
    scalar_ssp_target,
)


ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class CriticTrainingProtocol:
    total_updates: int = 800
    mc_pretrain_updates: int = 0
    normalize_returns: bool = False
    anchor_near_terminal: bool = False
    low_scale_initialization: bool = False
    target_update: str = "hard20"
    weight_nonuniform_quantile_atoms: bool = False

    def __post_init__(self) -> None:
        if self.total_updates <= 0 or not 0 <= self.mc_pretrain_updates <= self.total_updates:
            raise ValueError("invalid pretraining/total update budget")
        if self.target_update != "hard20" and self.target_update != "polyak0.01":
            raise ValueError("target update must be hard20 or polyak0.01")


@dataclass(frozen=True)
class StageARetrainConfig:
    source_dir: str
    output_dir: str
    seed: int = 0
    total_updates: int = 800
    selected_mc_pretrain_updates: int = 600
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.total_updates != 800:
            raise ValueError("the bounded repair comparison fixes the per-model budget at 800 updates")
        if not 0 < self.selected_mc_pretrain_updates < self.total_updates:
            raise ValueError("selected protocol must include both MC pretraining and TD fine-tuning")
        if self.device != "cpu":
            raise ValueError("the deterministic repair audit currently requires cpu")


def load_reused_stage_a_data(source_dir: str | Path) -> tuple[dict[str, EnergyDataset], dict, list[dict]]:
    source = Path(source_dir)
    split = json.loads((source / "split.json").read_text(encoding="utf-8"))
    records = [json.loads(line) for line in (source / "raw_sorties.jsonl").open(encoding="utf-8")]
    by_id = {int(record["sortie_id"]): record for record in records}
    if len(by_id) != len(records):
        raise ValueError("raw Stage A data contains duplicate sortie ids")
    datasets = {
        name: _dataset_from_records(by_id, details["sortie_ids"])
        for name, details in split["splits"].items()
    }
    return datasets, split, records


def _dataset_from_records(records: dict[int, dict], sortie_ids: list[int]) -> EnergyDataset:
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
    for sortie_id in sortie_ids:
        record = records[int(sortie_id)]
        if not record["completed"]:
            continue
        transitions = record["transitions"]
        charger = np.asarray(record["charger_position"], dtype=np.float32)
        for row, transition in enumerate(transitions):
            action = np.asarray(transition["action"], dtype=np.float32)
            next_action = np.zeros_like(action) if transition["charger_hit"] else np.asarray(
                transitions[row + 1]["action"], dtype=np.float32
            )
            state = np.asarray(transition["state"], dtype=np.float32)
            next_state = np.asarray(transition["next_state"], dtype=np.float32)
            position = state[:3] * np.array([4.0, 4.0, 2.0], dtype=np.float32)
            velocity = np.asarray(transition["velocity"], dtype=np.float32)
            charger_direction = charger - position
            velocity_norm = float(np.linalg.norm(velocity))
            direction_norm = float(np.linalg.norm(charger_direction))
            alignment = 0.0 if velocity_norm <= 1e-12 or direction_norm <= 1e-12 else float(
                np.dot(velocity, charger_direction) / (velocity_norm * direction_norm)
            )
            columns["features"].append(np.concatenate((state, action)))
            columns["next_features"].append(np.concatenate((next_state, next_action)))
            columns["costs"].append(transition["per_step_energy"])
            columns["terminals"].append(transition["charger_hit"])
            columns["returns"].append(transition["return_energy_to_go"])
            columns["distances"].append(transition["distance_to_charger"])
            columns["path_lengths"].append(transition["path_length_remaining"])
            columns["horizons"].append(len(transitions) - row)
            columns["sortie_indices"].append(int(sortie_id))
            columns["sortie_seeds"].append(int(record["sortie_seed"]))
            columns["velocity_magnitudes"].append(velocity_norm)
            columns["velocity_alignment"].append(alignment)
            columns["action_magnitudes"].append(float(np.linalg.norm(action)))
            columns["altitude_differences"].append(float(abs(charger_direction[2])))
            columns["prefix_lengths"].append(int(record["task_prefix_steps_executed"]))
            columns["positions"].append(position)
    if not columns["features"]:
        raise RuntimeError("reuse split contains no completed return supervision")
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


def deterministic_energy_dataset(costs: list[float]) -> EnergyDataset:
    values = np.asarray(costs, dtype=np.float32)
    if values.ndim != 1 or values.size == 0 or np.any(values < 0.0):
        raise ValueError("deterministic costs must be a nonempty nonnegative vector")
    count = values.size
    features = np.zeros((count, 80), dtype=np.float32)
    for index in range(count):
        features[index, index] = 1.0
        features[index, -1] = index / max(count - 1, 1)
    next_features = np.zeros_like(features)
    next_features[:-1] = features[1:]
    terminals = np.zeros(count, dtype=bool)
    terminals[-1] = True
    returns = np.cumsum(values[::-1], dtype=np.float64)[::-1].astype(np.float32)
    zeros = np.zeros(count, dtype=np.float32)
    integer_zeros = np.zeros(count, dtype=np.int32)
    return EnergyDataset(
        features=features,
        next_features=next_features,
        costs=values,
        terminals=terminals,
        returns=returns,
        distances=zeros,
        path_lengths_to_go=zeros,
        horizons=np.arange(count, 0, -1, dtype=np.int32),
        sortie_indices=integer_zeros,
        sortie_seeds=integer_zeros.astype(np.int64),
        velocity_magnitudes=zeros,
        velocity_alignment_to_charger=zeros,
        action_magnitudes=zeros,
        altitude_differences=zeros,
        prefix_lengths=integer_zeros,
        positions=np.zeros((count, 3), dtype=np.float32),
    )


def _return_scale(dataset: EnergyDataset, enabled: bool) -> float:
    return float(np.quantile(dataset.returns, 0.95)) if enabled else 1.0


def _sample_weights(dataset: EnergyDataset, enabled: bool) -> torch.Tensor:
    weights = np.ones(dataset.horizons.shape, dtype=np.float32)
    if enabled:
        weights[dataset.horizons == 1] = 20.0
        weights[dataset.horizons == 2] = 10.0
        weights[dataset.horizons == 3] = 5.0
        weights /= weights.mean()
    return torch.from_numpy(weights)


def _update_target(target: torch.nn.Module, model: torch.nn.Module, protocol: CriticTrainingProtocol, td_step: int) -> None:
    if protocol.target_update == "hard20":
        if td_step % 20 == 0:
            target.load_state_dict(model.state_dict())
        return
    with torch.no_grad():
        for target_parameter, parameter in zip(target.parameters(), model.parameters(), strict=True):
            target_parameter.mul_(0.99).add_(parameter, alpha=0.01)


def _scalar_predictions(model: ScalarEnergyCritic, datasets: dict[str, EnergyDataset], scale: float) -> dict[str, np.ndarray]:
    with torch.no_grad():
        return {
            name: model(torch.from_numpy(dataset.features)).numpy() * scale
            for name, dataset in datasets.items()
        }


def _quantile_predictions(
    model: MonotoneQuantileCritic, datasets: dict[str, EnergyDataset], scale: float
) -> dict[str, np.ndarray]:
    with torch.no_grad():
        return {
            name: model(torch.from_numpy(dataset.features)).numpy() * scale
            for name, dataset in datasets.items()
        }


def _trace_scalar(
    update: int,
    phase: str,
    loss: float | None,
    model: ScalarEnergyCritic,
    datasets: dict[str, EnergyDataset],
    scale: float,
) -> dict:
    predictions = _scalar_predictions(model, datasets, scale)
    return {
        "update": update,
        "phase": phase,
        "training_loss": loss,
        "mc_mae": {
            name: float(np.mean(np.abs(prediction - datasets[name].returns)))
            for name, prediction in predictions.items()
        },
        "terminal_mae": {
            name: float(np.mean(np.abs(prediction[datasets[name].terminals] - datasets[name].returns[datasets[name].terminals])))
            for name, prediction in predictions.items()
        },
    }


def _trace_quantile(
    update: int,
    phase: str,
    loss: float | None,
    model: MonotoneQuantileCritic,
    datasets: dict[str, EnergyDataset],
    scale: float,
) -> dict:
    predictions = _quantile_predictions(model, datasets, scale)
    return {
        "update": update,
        "phase": phase,
        "training_loss": loss,
        "median_mc_mae": {
            name: float(np.mean(np.abs(prediction[:, 0] - datasets[name].returns)))
            for name, prediction in predictions.items()
        },
        "terminal_median_mae": {
            name: float(
                np.mean(
                    np.abs(
                        prediction[datasets[name].terminals, 0]
                        - datasets[name].returns[datasets[name].terminals]
                    )
                )
            )
            for name, prediction in predictions.items()
        },
        "quantile_means": {name: prediction.mean(axis=0).tolist() for name, prediction in predictions.items()},
        "increment_means": {name: np.diff(prediction, axis=1).mean(axis=0).tolist() for name, prediction in predictions.items()},
    }


def train_scalar_td_repair(
    datasets: dict[str, EnergyDataset],
    protocol: CriticTrainingProtocol,
    *,
    seed: int,
) -> tuple[ScalarEnergyCritic, float, dict]:
    torch.manual_seed(seed)
    train = datasets["train"]
    scale = _return_scale(train, protocol.normalize_returns)
    initial = 0.05 if protocol.low_scale_initialization else None
    model = ScalarEnergyCritic(train.features.shape[1], initial_output=initial)
    target = ScalarEnergyCritic(train.features.shape[1], initial_output=initial)
    target.load_state_dict(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    features = torch.from_numpy(train.features)
    next_features = torch.from_numpy(train.next_features)
    costs = torch.from_numpy(train.costs / scale)
    terminals = torch.from_numpy(train.terminals)
    returns = torch.from_numpy(train.returns / scale)
    sample_weights = _sample_weights(train, protocol.anchor_near_terminal)
    trace = [_trace_scalar(0, "initial", None, model, datasets, scale)]
    losses: list[float] = []
    for update in range(protocol.total_updates):
        if update == protocol.mc_pretrain_updates and protocol.mc_pretrain_updates:
            target.load_state_dict(model.state_dict())
        if update < protocol.mc_pretrain_updates:
            labels = returns
            phase = "mc_pretrain"
        else:
            with torch.no_grad():
                labels = scalar_ssp_target(costs, target(next_features), terminals)
            phase = "td_finetune"
        per_sample = functional.mse_loss(model(features), labels, reduction="none")
        loss = (per_sample * sample_weights).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
        if phase == "td_finetune":
            _update_target(target, model, protocol, update - protocol.mc_pretrain_updates + 1)
        completed = update + 1
        if completed % 200 == 0 or completed == protocol.mc_pretrain_updates or completed == protocol.total_updates:
            trace.append(_trace_scalar(completed, phase, losses[-1], model, datasets, scale))
    return model, scale, {
        "protocol": asdict(protocol),
        "output_scale": scale,
        "terminal_sample_count": int(train.terminals.sum()),
        "nonterminal_sample_count": int((~train.terminals).sum()),
        "terminal_fraction": float(train.terminals.mean()),
        "trace": trace,
        "final_loss": losses[-1],
        "all_losses_finite": bool(np.all(np.isfinite(losses))),
    }


def train_quantile_td_repair(
    datasets: dict[str, EnergyDataset],
    protocol: CriticTrainingProtocol,
    *,
    seed: int,
) -> tuple[MonotoneQuantileCritic, float, dict]:
    torch.manual_seed(seed)
    train = datasets["train"]
    scale = _return_scale(train, protocol.normalize_returns)
    initial_base = 0.05 if protocol.low_scale_initialization else None
    initial_increment = 0.005 if protocol.low_scale_initialization else None
    model = MonotoneQuantileCritic(
        train.features.shape[1], QUANTILES, initial_base=initial_base, initial_increment=initial_increment
    )
    target = MonotoneQuantileCritic(
        train.features.shape[1], QUANTILES, initial_base=initial_base, initial_increment=initial_increment
    )
    target.load_state_dict(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    features = torch.from_numpy(train.features)
    next_features = torch.from_numpy(train.next_features)
    costs = torch.from_numpy(train.costs / scale)
    terminals = torch.from_numpy(train.terminals)
    returns = torch.from_numpy(train.returns / scale).unsqueeze(-1)
    levels = torch.tensor(QUANTILES, dtype=torch.float32)
    atom_weights = quantile_atom_weights(levels) if protocol.weight_nonuniform_quantile_atoms else None
    sample_weights = _sample_weights(train, protocol.anchor_near_terminal)
    trace = [_trace_quantile(0, "initial", None, model, datasets, scale)]
    losses: list[float] = []
    for update in range(protocol.total_updates):
        if update == protocol.mc_pretrain_updates and protocol.mc_pretrain_updates:
            target.load_state_dict(model.state_dict())
        if update < protocol.mc_pretrain_updates:
            labels = returns
            target_atom_weights = None
            phase = "mc_pretrain"
        else:
            with torch.no_grad():
                labels = quantile_ssp_target(costs, target(next_features), terminals)
            target_atom_weights = atom_weights
            phase = "td_finetune"
        per_sample = quantile_huber_loss(
            model(features),
            labels,
            levels,
            target_weights=target_atom_weights,
            reduction="none",
        )
        loss = (per_sample * sample_weights).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
        if phase == "td_finetune":
            _update_target(target, model, protocol, update - protocol.mc_pretrain_updates + 1)
        completed = update + 1
        if completed % 200 == 0 or completed == protocol.mc_pretrain_updates or completed == protocol.total_updates:
            trace.append(_trace_quantile(completed, phase, losses[-1], model, datasets, scale))
    return model, scale, {
        "protocol": asdict(protocol),
        "output_scale": scale,
        "target_atom_weights": None if atom_weights is None else atom_weights.tolist(),
        "terminal_sample_count": int(train.terminals.sum()),
        "nonterminal_sample_count": int((~train.terminals).sum()),
        "terminal_fraction": float(train.terminals.mean()),
        "trace": trace,
        "final_loss": losses[-1],
        "all_losses_finite": bool(np.all(np.isfinite(losses))),
    }


def _save_model(path: Path, model: torch.nn.Module, kind: str, scale: float, protocol: CriticTrainingProtocol) -> str:
    torch.save(
        {
            "model_kind": kind,
            "input_dim": next(model.parameters()).shape[1],
            "quantile_levels": list(QUANTILES) if isinstance(model, MonotoneQuantileCritic) else None,
            "output_scale": scale,
            "training_protocol": asdict(protocol),
            "state_dict": model.state_dict(),
        },
        path,
    )
    return sha256_file(path)


def _regression_bundle(prediction: np.ndarray, dataset: EnergyDataset) -> dict:
    return {
        "overall": regression_metrics(prediction, dataset.returns),
        "terminal": regression_metrics(prediction[dataset.terminals], dataset.returns[dataset.terminals]),
        "horizon": horizon_error_report(prediction, dataset.returns, dataset.horizons),
    }


def _old_predictions(source: Path, datasets: dict[str, EnergyDataset], cost_per_meter: float) -> dict[str, dict[str, np.ndarray]]:
    b1 = ScalarEnergyPredictor.load(source / "b1_mc_model.pt")
    b2 = ScalarEnergyPredictor.load(source / "b2_scalar_td_model.pt")
    b3 = QuantileEnergyPredictor.load(source / "b3_quantile_td_model.pt")
    output: dict[str, dict[str, np.ndarray]] = {}
    for name, dataset in datasets.items():
        with torch.no_grad():
            features = torch.from_numpy(dataset.features)
            output[name] = {
                "B0_distance": dataset.distances * cost_per_meter,
                "B1_monte_carlo": b1.model(features).numpy() * b1.output_scale,
                "B2_original": b2.model(features).numpy() * b2.output_scale,
                "B3_original_quantiles": b3.model(features).numpy() * b3.output_scale,
            }
    return output


def _ablation_protocols(total_updates: int, pretrain_updates: int, *, quantile: bool) -> dict[str, CriticTrainingProtocol]:
    common = {"total_updates": total_updates}
    protocols = {
        "controlled_current": CriticTrainingProtocol(**common),
        "mc_pretrain_only_fix": CriticTrainingProtocol(
            **common, mc_pretrain_updates=pretrain_updates
        ),
        "terminal_anchor_only": CriticTrainingProtocol(
            **common, anchor_near_terminal=True
        ),
        "normalization_only": CriticTrainingProtocol(
            **common, normalize_returns=True
        ),
        "polyak_only": CriticTrainingProtocol(
            **common, target_update="polyak0.01"
        ),
        "selected_hard20": CriticTrainingProtocol(
            **common,
            mc_pretrain_updates=pretrain_updates,
            normalize_returns=True,
            anchor_near_terminal=True,
            low_scale_initialization=True,
            target_update="hard20",
            weight_nonuniform_quantile_atoms=quantile,
        ),
        "selected_polyak001": CriticTrainingProtocol(
            **common,
            mc_pretrain_updates=pretrain_updates,
            normalize_returns=True,
            anchor_near_terminal=True,
            low_scale_initialization=True,
            target_update="polyak0.01",
            weight_nonuniform_quantile_atoms=quantile,
        ),
    }
    if quantile:
        protocols["low_scale_initialization_only"] = CriticTrainingProtocol(
            **common, low_scale_initialization=True
        )
        protocols["atom_weighting_only"] = CriticTrainingProtocol(
            **common, weight_nonuniform_quantile_atoms=True
        )
    return protocols


def run_stage_a_retraining(config: StageARetrainConfig) -> dict:
    source = Path(config.source_dir).resolve()
    output = Path(config.output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite retraining output: {output}")
    output.mkdir(parents=True)
    write_json(
        output / "RUNNING.json",
        {
            "status": "RUNNING",
            "started_at": utc_now(),
            "pid": __import__("os").getpid(),
        },
    )
    try:
        torch.set_num_threads(1)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        datasets, split, records = load_reused_stage_a_data(source)
        source_results = json.loads((source / "results.json").read_text(encoding="utf-8"))
        raw_hash = sha256_file(source / "raw_sorties.jsonl")
        split_hash = sha256_file(source / "split.json")
        if raw_hash != source_results["data_sha256"] or split_hash != source_results["split_sha256"]:
            raise ValueError("source data/split hashes do not match the immutable validation result")
        shutil.copyfile(source / "split.json", output / "split.json")
        exact_command = " ".join(shlex.quote(value) for value in sys.argv)
        config_payload = asdict(config) | {
            "experiment": "STAGE_A_4X4_TD_SEMANTIC_REPAIR",
            "status": "RUNNING",
            "exact_command": exact_command,
            "source_raw_sorties": str(source / "raw_sorties.jsonl"),
            "source_data_sha256": raw_hash,
            "source_split_sha256": split_hash,
            "source_sortie_count": len(records),
            "git_commit_sha": git_sha(ROOT),
            "code_hash": code_hash(ROOT),
            "data_recollected": False,
            "frozen_sac_loaded_or_modified": False,
        }
        write_json(output / "config.json", config_payload)

        scalar_runs: dict[str, tuple[ScalarEnergyCritic, float, dict]] = {}
        for name, protocol in _ablation_protocols(
            config.total_updates, config.selected_mc_pretrain_updates, quantile=False
        ).items():
            scalar_runs[name] = train_scalar_td_repair(datasets, protocol, seed=config.seed)
        quantile_runs: dict[str, tuple[MonotoneQuantileCritic, float, dict]] = {}
        for name, protocol in _ablation_protocols(
            config.total_updates, config.selected_mc_pretrain_updates, quantile=True
        ).items():
            quantile_runs[name] = train_quantile_td_repair(datasets, protocol, seed=config.seed)

        selected_scalar, scalar_scale, scalar_training = scalar_runs["selected_polyak001"]
        selected_quantile, quantile_scale, quantile_training = quantile_runs["selected_polyak001"]
        scalar_predictions = _scalar_predictions(selected_scalar, datasets, scalar_scale)
        quantile_predictions = _quantile_predictions(selected_quantile, datasets, quantile_scale)
        old = _old_predictions(source, datasets, float(source_results["cost_per_meter_train"]))
        test = datasets["test"]
        test_predictions = {
            "B0_distance": old["test"]["B0_distance"],
            "B1_monte_carlo": old["test"]["B1_monte_carlo"],
            "B2_scalar_td_repaired": scalar_predictions["test"],
            "B3_distributional_median_repaired": quantile_predictions["test"][:, 0],
        }
        paired = paired_sortie_comparisons(
            test_predictions,
            test.returns,
            test.sortie_indices,
            seed=config.seed + 12_007,
        )
        learned = learned_value_decision({name: value for name, value in paired.items() if name != "B0_distance"})
        q_metrics = quantile_metrics(quantile_predictions["test"], test.returns, QUANTILES)
        old_q_metrics = quantile_metrics(old["test"]["B3_original_quantiles"], test.returns, QUANTILES)

        def scalar_ablation_summary(run: tuple[ScalarEnergyCritic, float, dict]) -> dict:
            model, scale, diagnostics = run
            prediction = _scalar_predictions(model, {"test": test}, scale)["test"]
            return {
                "protocol": diagnostics["protocol"],
                "test_mae": regression_metrics(prediction, test.returns)["mae"],
                "terminal_mae": regression_metrics(prediction[test.terminals], test.returns[test.terminals])["mae"],
                "trace": diagnostics["trace"],
            }

        def quantile_ablation_summary(run: tuple[MonotoneQuantileCritic, float, dict]) -> dict:
            model, scale, diagnostics = run
            prediction = _quantile_predictions(model, {"test": test}, scale)["test"]
            return {
                "protocol": diagnostics["protocol"],
                "median_test_mae": regression_metrics(prediction[:, 0], test.returns)["mae"],
                "terminal_median_mae": regression_metrics(
                    prediction[test.terminals, 0], test.returns[test.terminals]
                )["mae"],
                "coverage": quantile_metrics(prediction, test.returns, QUANTILES)["empirical_coverage"],
                "trace": diagnostics["trace"],
            }

        b2_before = _regression_bundle(old["test"]["B2_original"], test)
        b2_after = _regression_bundle(scalar_predictions["test"], test)
        b3_before = _regression_bundle(old["test"]["B3_original_quantiles"][:, 0], test)
        b3_after = _regression_bundle(quantile_predictions["test"][:, 0], test)
        ordering = bool(np.all(quantile_predictions["test"][:, 1:] >= quantile_predictions["test"][:, :-1]))
        b2_semantic = bool(b2_after["overall"]["mae"] < 0.08 and b2_after["terminal"]["mae"] < 0.03)
        b3_semantic = bool(
            b3_after["overall"]["mae"] < 0.10
            and b3_after["terminal"]["mae"] < 0.05
            and 0.30 <= q_metrics["empirical_coverage"]["0.5"] <= 0.70
            and ordering
            and float(np.max(quantile_predictions["test"][:, 0])) < 1.5
        )
        b3_upper_coverage_reasonable = bool(
            q_metrics["empirical_coverage"]["0.9"] >= 0.80
            and q_metrics["empirical_coverage"]["0.95"] >= 0.85
            and q_metrics["empirical_coverage"]["0.99"] >= 0.90
        )
        model_hashes = {
            "B2_REPAIRED": _save_model(
                output / "b2_scalar_td_repaired.pt",
                selected_scalar,
                "ScalarEnergyCritic_MC_PRETRAIN_TD",
                scalar_scale,
                CriticTrainingProtocol(**scalar_training["protocol"]),
            ),
            "B3_REPAIRED": _save_model(
                output / "b3_quantile_td_repaired.pt",
                selected_quantile,
                "MonotoneQuantileCritic_MC_PRETRAIN_TD",
                quantile_scale,
                CriticTrainingProtocol(**quantile_training["protocol"]),
            ),
        }
        results = {
            "status": "COMPLETED",
            "completed_at": utc_now(),
            "source_artifact": str(source),
            "source_data_sha256": raw_hash,
            "source_split_sha256": split_hash,
            "data_recollected": False,
            "ablation_initialization_seed_shared": True,
            "sorties": len(records),
            "split_transition_counts": {name: int(dataset.returns.size) for name, dataset in datasets.items()},
            "terminal_anchor": {
                "train_terminal": int(datasets["train"].terminals.sum()),
                "train_nonterminal": int((~datasets["train"].terminals).sum()),
                "train_terminal_fraction": float(datasets["train"].terminals.mean()),
            },
            "B2_before": b2_before,
            "B2_after": b2_after,
            "B3_before": b3_before | {"quantiles": old_q_metrics},
            "B3_after": b3_after | {"quantiles": q_metrics, "quantile_ordering_preserved": ordering},
            "B2_training": scalar_training,
            "B3_training": quantile_training,
            "B2_ablations": {name: scalar_ablation_summary(run) for name, run in scalar_runs.items()},
            "B3_ablations": {name: quantile_ablation_summary(run) for name, run in quantile_runs.items()},
            "heldout_test_metrics": {
                name: regression_metrics(prediction, test.returns)
                for name, prediction in test_predictions.items()
            }
            | {"B3_quantiles_repaired": q_metrics},
            "paired_sortie_comparison_vs_B0": paired,
            "learned_model_value": learned,
            "semantic_gates": {
                "B2_SEMANTIC_CORRECTNESS": b2_semantic,
                "B3_SEMANTIC_CORRECTNESS": b3_semantic,
                "GATE_1_SEMANTICS": b2_semantic and b3_semantic,
                "B3_UPPER_QUANTILE_COVERAGE_REASONABLE": b3_upper_coverage_reasonable,
                "GATE_2_LEARNED_VALUE_OVER_B0": learned["LEARNED_MODEL_ADDS_VALUE"],
                "READY_FOR_16X16": False,
            },
            "model_sha256": model_hashes,
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
