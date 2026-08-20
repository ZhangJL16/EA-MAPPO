from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn

from review_bundle.safety.grounded_memory import (
    HistoricalPositionConstraint,
    SetMembershipConfig,
    VerifiedHistorySetUpdater,
)


METHODS = (
    "C0_current_only",
    "C1_fixed_history",
    "C2_all_history",
    "C3_generic_GRU_proposal_verifier",
    "C4_FOGM_proposal_verifier",
    "C5_oracle_valid_history",
)


@dataclass(frozen=True)
class ContractionDiagnosticConfig:
    dataset_dir: str
    output_dir: str
    training_seeds: tuple[int, ...] = (11, 23, 37)
    proposal_budget: int = 4
    fixed_history: int = 4
    training_epochs: int = 12
    hidden_size: int = 32
    batch_size: int = 128
    maximum_training_snapshots: int = 8000
    maximum_evaluation_snapshots: int = 2000
    sensor_error_bound: float = 0.10
    velocity_bound: float = 30.0
    acceleration_bound: float = 8.0
    jerk_bound: float = 12.0
    maximum_history_seconds: float = 2.0


class GenericHistoryProposal(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.encoder = nn.GRU(input_dim, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        encoded, _ = self.encoder(features)
        return self.head(encoded).squeeze(-1)


class FOGMConstraintProposal(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.physical = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.physical(features).squeeze(-1)


def _load_rows(dataset_dir: Path, split_name: str) -> list[dict[str, object]]:
    splits = json.loads((dataset_dir / "splits.json").read_text(encoding="utf-8"))
    ids = set(int(value) for value in splits[split_name])
    return [
        json.loads(line)
        for line in (dataset_dir / "contraction_snapshots.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip() and int(json.loads(line)["trajectory_id"]) in ids
    ]


def _constraint(
    measurement: list[float] | np.ndarray,
    age: float,
    track_id: str,
    epoch: int,
    association_verified: bool,
    sensor_error_bound: float,
) -> HistoricalPositionConstraint:
    return HistoricalPositionConstraint(
        measurement=np.asarray(measurement, dtype=np.float64),
        age_seconds=float(age),
        track_id=str(track_id),
        epoch=int(epoch),
        sensor_error_bound=sensor_error_bound,
        association_verified=bool(association_verified),
        provenance="real_closed_loop_bounded_measurement",
    )


def _constraints_from_row(
    row: dict[str, object],
    sensor_error_bound: float,
) -> tuple[HistoricalPositionConstraint, list[HistoricalPositionConstraint]]:
    base = _constraint(
        row["base_measurement"],
        row["base_age_seconds"],
        row["track_id"],
        row["base_epoch"],
        True,
        sensor_error_bound,
    )
    candidates = [
        _constraint(measurement, age, track, epoch, association, sensor_error_bound)
        for measurement, age, track, epoch, association in zip(
            row["candidate_measurements"],
            row["candidate_ages_seconds"],
            row["candidate_track_ids"],
            row["candidate_epochs"],
            row["candidate_association_verified"],
        )
    ]
    return base, candidates


def _candidate_features(
    row: dict[str, object],
    maximum_history_seconds: float,
    *,
    grounded: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base = np.asarray(row["base_measurement"], dtype=np.float64)
    measurement = np.asarray(row["candidate_measurements"], dtype=np.float64)
    ages = np.asarray(row["candidate_ages_seconds"], dtype=np.float64)
    epochs = np.asarray(row["candidate_epochs"], dtype=np.float64)
    same_track = np.asarray(
        [track == row["track_id"] for track in row["candidate_track_ids"]],
        dtype=np.float64,
    )
    association = np.asarray(row["candidate_association_verified"], dtype=np.float64)
    if measurement.size == 0:
        width = 10 if grounded else 5
        return np.zeros((0, width), dtype=np.float32), np.zeros(0, dtype=np.float32), np.zeros(0, dtype=bool)
    delta = (measurement - base[None, :]) / 60.0
    age = ages[:, None] / maximum_history_seconds
    epoch_gap = (float(row["base_epoch"]) - epochs)[:, None] / 40.0
    generic = np.concatenate((delta, age, epoch_gap), axis=1)
    valid = (
        same_track.astype(bool)
        & association.astype(bool)
        & (ages <= maximum_history_seconds + 1e-12)
    )
    valid_indices = np.flatnonzero(valid)
    labels = np.zeros(ages.shape[0], dtype=np.float32)
    if valid_indices.size:
        selected = valid_indices[np.argsort(ages[valid_indices])[-min(4, valid_indices.size) :]]
        labels[selected] = 1.0
    if not grounded:
        return generic.astype(np.float32), labels, valid
    physically_reachable_scale = 0.1 + 30.0 * ages + 0.5 * 8.0 * ages**2 + 12.0 * ages**3 / 6.0
    normalized_residual = np.linalg.norm(measurement - base[None, :], axis=1) / np.maximum(physically_reachable_scale, 1e-6)
    grounded_features = np.concatenate(
        (
            generic,
            same_track[:, None],
            association[:, None],
            normalized_residual[:, None],
            (ages <= maximum_history_seconds)[:, None].astype(np.float64),
            (np.abs(epochs - float(row["base_epoch"])) > 0.0)[:, None].astype(np.float64),
        ),
        axis=1,
    )
    return grounded_features.astype(np.float32), labels, valid


def _pack_rows(
    rows: list[dict[str, object]],
    maximum_history_seconds: float,
    *,
    grounded: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    width = 10 if grounded else 5
    maximum = max((len(row["candidate_ages_seconds"]) for row in rows), default=0)
    features = np.zeros((len(rows), maximum, width), dtype=np.float32)
    labels = np.zeros((len(rows), maximum), dtype=np.float32)
    mask = np.zeros((len(rows), maximum), dtype=bool)
    for index, row in enumerate(rows):
        row_features, row_labels, _ = _candidate_features(
            row,
            maximum_history_seconds,
            grounded=grounded,
        )
        count = row_features.shape[0]
        features[index, :count] = row_features
        labels[index, :count] = row_labels
        mask[index, :count] = True
    return features, labels, mask


def _fit_model(
    model: nn.Module,
    rows: list[dict[str, object]],
    config: ContractionDiagnosticConfig,
    *,
    grounded: bool,
    seed: int,
) -> dict[str, float]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    if len(rows) > config.maximum_training_snapshots:
        rows = [rows[index] for index in rng.choice(len(rows), config.maximum_training_snapshots, replace=False)]
    features, labels, mask = _pack_rows(
        rows,
        config.maximum_history_seconds,
        grounded=grounded,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    losses = []
    for _ in range(config.training_epochs):
        order = rng.permutation(features.shape[0])
        for start in range(0, order.size, config.batch_size):
            indices = order[start : start + config.batch_size]
            x = torch.from_numpy(features[indices])
            target = torch.from_numpy(labels[indices])
            valid_mask = torch.from_numpy(mask[indices])
            logits = model(x)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits[valid_mask],
                target[valid_mask],
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
    return {"final_loss": float(np.mean(losses[-max(1, len(losses) // config.training_epochs) :]))}


def _scores(
    model: nn.Module,
    row: dict[str, object],
    config: ContractionDiagnosticConfig,
    *,
    grounded: bool,
) -> np.ndarray:
    features, _, _ = _candidate_features(
        row,
        config.maximum_history_seconds,
        grounded=grounded,
    )
    if features.shape[0] == 0:
        return np.zeros(0, dtype=np.float64)
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(features[None, ...]))[0].numpy().astype(np.float64)


def _selection(
    method: str,
    row: dict[str, object],
    candidates: list[HistoricalPositionConstraint],
    config: ContractionDiagnosticConfig,
    generic: nn.Module,
    fogm: nn.Module,
) -> list[HistoricalPositionConstraint]:
    if method == "C0_current_only":
        return []
    if method == "C1_fixed_history":
        return sorted(candidates, key=lambda item: item.age_seconds)[: config.fixed_history]
    if method == "C2_all_history":
        return candidates
    if method == "C3_generic_GRU_proposal_verifier":
        scores = _scores(generic, row, config, grounded=False)
    elif method == "C4_FOGM_proposal_verifier":
        scores = _scores(fogm, row, config, grounded=True)
    elif method == "C5_oracle_valid_history":
        valid = [
            index
            for index, item in enumerate(candidates)
            if item.association_verified
            and item.track_id == row["track_id"]
            and item.age_seconds <= config.maximum_history_seconds + 1e-12
        ]
        valid.sort(key=lambda index: candidates[index].age_seconds, reverse=True)
        return [candidates[index] for index in valid[: config.proposal_budget]]
    else:
        raise ValueError(method)
    if scores.size == 0:
        return []
    count = min(config.proposal_budget, scores.size)
    indices = np.argpartition(scores, -count)[-count:]
    return [candidates[int(index)] for index in indices]


def _evaluate_seed(
    rows: list[dict[str, object]],
    config: ContractionDiagnosticConfig,
    generic: nn.Module,
    fogm: nn.Module,
) -> dict[str, object]:
    set_config = SetMembershipConfig(
        sensor_error_bound=config.sensor_error_bound,
        velocity_bound=config.velocity_bound,
        acceleration_bound=config.acceleration_bound,
        jerk_bound=config.jerk_bound,
        maximum_history_seconds=config.maximum_history_seconds,
    )
    updater = VerifiedHistorySetUpdater(set_config)
    by_method: dict[str, list[dict[str, float | int | bool]]] = {method: [] for method in METHODS}
    for row in rows:
        base, candidates = _constraints_from_row(row, config.sensor_error_bound)
        truth = np.asarray(row["true_state"], dtype=np.float64)
        valid_candidate_count = sum(
            item.association_verified
            and item.track_id == base.track_id
            and item.age_seconds <= config.maximum_history_seconds + 1e-12
            for item in candidates
        )
        for method in METHODS:
            selected = _selection(method, row, candidates, config, generic, fogm)
            started = perf_counter()
            update = updater.update_batch(base, selected)
            compute = perf_counter() - started
            accepted_candidate_count = len(update.accepted_epochs) - 1
            selected_valid = sum(
                item.association_verified
                and item.track_id == base.track_id
                and item.age_seconds <= config.maximum_history_seconds + 1e-12
                for item in selected
            )
            by_method[method].append(
                {
                    "contains": update.state_box.contains(truth, tolerance=1e-7),
                    "log_volume": update.state_box.log_volume,
                    "width_sum": float(np.sum(update.state_box.widths)),
                    "position_support": float(np.linalg.norm(update.state_box.radius[:3])),
                    "accepted": accepted_candidate_count,
                    "false_accepted": max(0, accepted_candidate_count - selected_valid),
                    "false_rejected": max(0, valid_candidate_count - accepted_candidate_count),
                    "compute_seconds": compute,
                }
            )
    summary: dict[str, object] = {}
    base_log_volume = np.asarray([item["log_volume"] for item in by_method["C0_current_only"]])
    for method, values in by_method.items():
        log_volume = np.asarray([item["log_volume"] for item in values], dtype=np.float64)
        width = np.asarray([item["width_sum"] for item in values], dtype=np.float64)
        support = np.asarray([item["position_support"] for item in values], dtype=np.float64)
        compute = np.asarray([item["compute_seconds"] for item in values], dtype=np.float64)
        summary[method] = {
            "snapshots": len(values),
            "containment_rate": float(np.mean([item["contains"] for item in values])),
            "containment_failures": int(sum(not item["contains"] for item in values)),
            "mean_log_volume": float(np.mean(log_volume)),
            "median_log_volume": float(np.median(log_volume)),
            "p95_log_volume": float(np.quantile(log_volume, 0.95)),
            "geometric_mean_volume_ratio_to_base": float(np.exp(np.mean(log_volume - base_log_volume))),
            "mean_width_sum": float(np.mean(width)),
            "median_width_sum": float(np.median(width)),
            "p95_width_sum": float(np.quantile(width, 0.95)),
            "mean_position_directional_support_l2": float(np.mean(support)),
            "mean_accepted_constraints": float(np.mean([item["accepted"] for item in values])),
            "false_accepted_constraints": int(sum(item["false_accepted"] for item in values)),
            "mean_false_rejected_constraints": float(np.mean([item["false_rejected"] for item in values])),
            "mean_compute_ms": float(1000.0 * np.mean(compute)),
            "p95_compute_ms": float(1000.0 * np.quantile(compute, 0.95)),
        }
    return summary


def run_contraction_diagnostic(config: ContractionDiagnosticConfig) -> dict[str, object]:
    torch.set_num_threads(1)
    dataset = Path(config.dataset_dir)
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    train_rows = _load_rows(dataset, "train")
    test_rows = _load_rows(dataset, "test")
    rng = np.random.default_rng(71)
    if len(test_rows) > config.maximum_evaluation_snapshots:
        test_rows = [
            test_rows[index]
            for index in rng.choice(len(test_rows), config.maximum_evaluation_snapshots, replace=False)
        ]
    seed_results = []
    for seed in config.training_seeds:
        generic = GenericHistoryProposal(5, config.hidden_size)
        fogm = FOGMConstraintProposal(10, config.hidden_size)
        generic_training = _fit_model(
            generic,
            train_rows,
            config,
            grounded=False,
            seed=seed,
        )
        fogm_training = _fit_model(
            fogm,
            train_rows,
            config,
            grounded=True,
            seed=seed,
        )
        seed_results.append(
            {
                "seed": seed,
                "generic_training": generic_training,
                "fogm_training": fogm_training,
                "methods": _evaluate_seed(test_rows, config, generic, fogm),
            }
        )
    aggregate: dict[str, object] = {}
    for method in METHODS:
        aggregate[method] = {
            metric: {
                "mean": float(np.mean([seed["methods"][method][metric] for seed in seed_results])),
                "std": float(np.std([seed["methods"][method][metric] for seed in seed_results])),
                "values": [seed["methods"][method][metric] for seed in seed_results],
            }
            for metric in (
                "containment_rate",
                "geometric_mean_volume_ratio_to_base",
                "mean_width_sum",
                "mean_position_directional_support_l2",
                "mean_accepted_constraints",
                "mean_false_rejected_constraints",
                "mean_compute_ms",
            )
        }
        aggregate[method]["false_accepted_constraints"] = int(
            sum(seed["methods"][method]["false_accepted_constraints"] for seed in seed_results)
        )
    result = {
        "config": asdict(config),
        "split_unit": "whole_trajectory",
        "training_snapshots_available": len(train_rows),
        "evaluation_snapshots": len(test_rows),
        "seed_results": seed_results,
        "aggregate": aggregate,
        "certificate_semantics": {
            "network_role": "propose history constraints only",
            "verifier_role": "structural checks plus set-membership feasibility",
            "certificate_input": "accepted constraints only",
            "incorrect_proposal_effect": "less contraction, never direct radius reduction",
        },
    }
    (output / "contraction_diagnostic.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["aggregate"], indent=2), flush=True)
    return result
