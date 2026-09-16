#!/usr/bin/env python3
"""Frozen legal-feature and ridge estimators for CMI-v3."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal

import numpy as np


ACTION_ORDER = ("R", "C")
CHANNELS = ("latest", "full")
FAMILIES = ("LINEAR_RIDGE", "RFF_RIDGE")
OUTER_FOLDS = 6
INNER_FOLDS = 4
RIDGE_GRID = (0.1, 1.0, 10.0, 100.0)
RFF_DIM = 256
RFF_SEEDS = (2_026_091_221, 2_026_091_222, 2_026_091_223)
FRAME_DIM = 4128
COMPACT_DIM = 416
FULL_DIM = COMPACT_DIM * 7
STD_FLOOR = 1e-6


def fold_for_world(identity: str, *, outer: bool = True) -> int:
    suffix = "cmi-v3-outer-fold-v1" if outer else "cmi-v3-inner-fold-v1"
    digest = hashlib.sha256(f"{identity}|{suffix}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (OUTER_FOLDS if outer else INNER_FOLDS)


def calibration_worlds(identities: np.ndarray) -> set[str]:
    """Choose exactly the first ceil(20%) training worlds by a frozen hash."""
    worlds = sorted(set(identities.tolist()))
    ordered = sorted(
        worlds,
        key=lambda value: hashlib.sha256(
            f"{value}|cmi-v3-calibration-split-v1".encode()
        ).digest(),
    )
    count = max(1, int(np.ceil(0.2 * len(ordered))))
    return set(ordered[:count])


def _last_dim(value: np.ndarray) -> np.ndarray:
    return value[..., None] if value.ndim == 2 else value


def raw_legal_frames(arrays: dict[str, np.ndarray]) -> np.ndarray:
    fields = (
        "nav_observation", "return_observation", "battery", "distance_to_charger",
        "task_progress", "task_clock", "step", "previous_nominal_action",
        "previous_executed_action", "previous_realized_acceleration",
        "previous_action_valid", "previous_contact",
    )
    values = [_last_dim(np.asarray(arrays[field], dtype=np.float32)) for field in fields]
    result = np.concatenate(values, axis=-1).astype(np.float32)
    if result.ndim != 3 or result.shape[1:] != (64, FRAME_DIM):
        raise ValueError(f"invalid CMI-v3 legal history tensor {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError("non-finite CMI-v3 legal history")
    return result


def _compact_observation(value: np.ndarray) -> np.ndarray:
    if value.shape[-1] != 2056:
        raise ValueError("invalid legal observation width")
    scalar = np.concatenate((value[..., :7], value[..., 2055:2056]), axis=-1)
    distance = value[..., 7:1031].reshape(*value.shape[:-1], 8, 8, 16)
    hit = value[..., 1031:2055].reshape(*value.shape[:-1], 8, 8, 16)
    pooled = np.concatenate(
        (distance.mean(axis=-1), distance.min(axis=-1), hit.mean(axis=-1)), axis=-1
    )
    return np.concatenate((scalar, pooled.reshape(*value.shape[:-1], -1)), axis=-1)


def legal_features(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    raw = raw_legal_frames(arrays)
    compact = np.concatenate(
        (
            _compact_observation(raw[..., :2056]),
            _compact_observation(raw[..., 2056:4112]),
            raw[..., 4112:],
        ),
        axis=-1,
    ).astype(np.float64)
    if compact.shape[1:] != (64, COMPACT_DIM):
        raise ValueError(f"invalid compact legal feature tensor {compact.shape}")
    latest = compact[:, -1]
    summaries = (
        compact.mean(axis=1),
        compact.std(axis=1),
        compact.min(axis=1),
        compact.max(axis=1),
        compact[:, -1] - compact[:, 0],
        compact[:, -16:].mean(axis=1) - compact[:, :16].mean(axis=1),
    )
    full = np.concatenate((latest, *summaries), axis=1)
    if latest.shape[1] != COMPACT_DIM or full.shape[1] != FULL_DIM:
        raise AssertionError("CMI-v3 feature dimension drift")
    if not np.isfinite(latest).all() or not np.isfinite(full).all():
        raise ValueError("non-finite CMI-v3 compact feature")
    return {"latest": latest, "full": full}


@dataclass
class RidgeState:
    family: str
    seed: int
    alpha: float
    mean: np.ndarray
    std: np.ndarray
    target_mean: np.ndarray
    coefficient: np.ndarray
    rff_weight: np.ndarray | None
    rff_phase: np.ndarray | None


def fit_standardizer(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(x, np.float64).mean(axis=0)
    std = np.asarray(x, np.float64).std(axis=0)
    return mean, np.maximum(std, STD_FLOOR)


def _design(
    x: np.ndarray,
    family: Literal["LINEAR_RIDGE", "RFF_RIDGE"],
    seed: int,
    mean: np.ndarray,
    std: np.ndarray,
    rff_weight: np.ndarray | None = None,
    rff_phase: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    standardized = (np.asarray(x, np.float64) - mean) / std
    if family == "LINEAR_RIDGE":
        return standardized, None, None
    if family != "RFF_RIDGE":
        raise ValueError(f"unknown CMI-v3 family {family}")
    if rff_weight is None or rff_phase is None:
        rng = np.random.Generator(np.random.PCG64(seed))
        rff_weight = rng.normal(
            0.0, 1.0 / np.sqrt(standardized.shape[1]),
            size=(standardized.shape[1], RFF_DIM),
        )
        rff_phase = rng.uniform(0.0, 2.0 * np.pi, size=RFF_DIM)
    nonlinear = np.sqrt(2.0 / RFF_DIM) * np.cos(standardized @ rff_weight + rff_phase)
    return np.concatenate((standardized, nonlinear), axis=1), rff_weight, rff_phase


def fit_ridge(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    *,
    family: Literal["LINEAR_RIDGE", "RFF_RIDGE"],
    seed: int,
    alpha: float,
) -> RidgeState:
    if len(x) != len(y) or y.ndim != 2 or y.shape[1] != 2:
        raise ValueError("invalid CMI-v3 ridge training arrays")
    mean, std = fit_standardizer(x)
    design, rff_weight, rff_phase = _design(x, family, seed, mean, std)
    weight = np.asarray(weights, np.float64)
    if weight.shape != (len(x),) or np.any(weight <= 0) or not np.isfinite(weight).all():
        raise ValueError("invalid CMI-v3 sample weights")
    target_mean = np.average(y, axis=0, weights=weight)
    root = np.sqrt(weight / weight.mean())[:, None]
    weighted_x = design * root
    weighted_y = (np.asarray(y, np.float64) - target_mean) * root
    gram = weighted_x @ weighted_x.T
    dual = np.linalg.solve(gram + float(alpha) * np.eye(len(gram)), weighted_y)
    coefficient = weighted_x.T @ dual
    if not all(
        np.isfinite(value).all()
        for value in (mean, std, target_mean, coefficient)
    ):
        raise ValueError("non-finite CMI-v3 ridge state")
    return RidgeState(
        family=family,
        seed=int(seed),
        alpha=float(alpha),
        mean=mean,
        std=std,
        target_mean=target_mean,
        coefficient=coefficient,
        rff_weight=rff_weight,
        rff_phase=rff_phase,
    )


def predict_ridge(state: RidgeState, x: np.ndarray) -> np.ndarray:
    design, _, _ = _design(
        x, state.family, state.seed, state.mean, state.std,
        state.rff_weight, state.rff_phase,
    )
    result = design @ state.coefficient + state.target_mean
    if not np.isfinite(result).all():
        raise ValueError("non-finite CMI-v3 ridge prediction")
    return result


def select_alpha(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    worlds: np.ndarray,
    *,
    family: Literal["LINEAR_RIDGE", "RFF_RIDGE"],
    seed: int,
) -> float:
    folds = np.asarray([fold_for_world(str(world), outer=False) for world in worlds])
    losses: dict[float, float] = {}
    for alpha in RIDGE_GRID:
        total, mass = 0.0, 0.0
        for fold in range(INNER_FOLDS):
            train, test = folds != fold, folds == fold
            if not train.any() or not test.any():
                raise ValueError("empty CMI-v3 inner world fold")
            state = fit_ridge(
                x[train], y[train], weights[train],
                family=family, seed=seed + fold * 1009, alpha=alpha,
            )
            error = np.square(predict_ridge(state, x[test]) - y[test]).mean(axis=1)
            total += float(np.sum(error * weights[test]))
            mass += float(np.sum(weights[test]))
        losses[alpha] = total / mass
    # Larger alpha is the prospectively frozen exact-tie choice.
    return min(RIDGE_GRID, key=lambda alpha: (losses[alpha], -alpha))


def family_seeds(family: str) -> tuple[int, ...]:
    if family == "LINEAR_RIDGE":
        return (RFF_SEEDS[0],)
    if family == "RFF_RIDGE":
        return RFF_SEEDS
    raise ValueError(f"unknown CMI-v3 family {family}")


def load_states(npz_path: str, manifest: list[dict[str, object]]) -> list[tuple[str, str, RidgeState]]:
    loaded: list[tuple[str, str, RidgeState]] = []
    with np.load(npz_path, allow_pickle=False) as archive:
        for row in manifest:
            prefix = str(row["prefix"])
            has_rff = bool(row["has_rff"])
            state = RidgeState(
                family=str(row["family"]),
                seed=int(row["seed"]),
                alpha=float(row["alpha"]),
                mean=archive[f"{prefix}_mean"].copy(),
                std=archive[f"{prefix}_std"].copy(),
                target_mean=archive[f"{prefix}_target_mean"].copy(),
                coefficient=archive[f"{prefix}_coefficient"].copy(),
                rff_weight=archive[f"{prefix}_rff_weight"].copy() if has_rff else None,
                rff_phase=archive[f"{prefix}_rff_phase"].copy() if has_rff else None,
            )
            loaded.append((str(row["channel"]), str(row["family"]), state))
    return loaded


def predict_frozen_ensemble(
    states: list[tuple[str, str, RidgeState]],
    features: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, np.ndarray]]]:
    family: dict[str, dict[str, list[np.ndarray]]] = {
        channel: {name: [] for name in FAMILIES} for channel in CHANNELS
    }
    for channel, name, state in states:
        family[channel][name].append(predict_ridge(state, features[channel]))
    family_mean = {
        channel: {
            name: np.mean(values, axis=0)
            for name, values in by_family.items()
        }
        for channel, by_family in family.items()
    }
    ensemble = {
        channel: np.mean(list(by_family.values()), axis=0)
        for channel, by_family in family_mean.items()
    }
    return ensemble, family_mean
