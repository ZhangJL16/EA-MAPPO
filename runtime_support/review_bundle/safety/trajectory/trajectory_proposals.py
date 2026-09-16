from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

import numpy as np

from .history_buffer import HistoryFrame


class ProposalSource(str, Enum):
    RANDOM_SAC = "random_sac"
    HISTORY_EXTRAPOLATION = "history_extrapolation"
    BRAKING = "braking"
    GOAL_DIRECTED = "goal_directed"
    OBSTACLE_ESCAPE = "obstacle_escape"
    SHIFTED_PREVIOUS = "shifted_previous"
    CLF_GUIDED = "clf_guided"
    CBF_GRADIENT = "cbf_gradient"
    LEARNED = "learned"
    MIXTURE = "mixture"


class LearnedProposalModel(Protocol):
    def propose(
        self,
        *,
        history: tuple[HistoryFrame, ...],
        position: np.ndarray,
        velocity: np.ndarray,
        nominal_action: np.ndarray,
        goal: np.ndarray,
        count: int,
        horizon: int,
    ) -> np.ndarray: ...


@dataclass(frozen=True)
class ProposalConfig:
    horizon: int = 10
    count: int = 1000
    random_fraction: float = 0.50
    perturbation_std: float = 0.8
    horizontal_acceleration_limit: float = 5.0
    vertical_acceleration_limit: float = 3.0
    goal_gain: float = 1.0
    velocity_damping: float = 0.4

    def __post_init__(self) -> None:
        if self.horizon not in (5, 10, 20, 40):
            raise ValueError("horizon must be one of 5, 10, 20, 40")
        if self.count <= 0 or self.perturbation_std < 0.0:
            raise ValueError("proposal count must be positive and std nonnegative")
        if not 0.0 <= self.random_fraction <= 1.0:
            raise ValueError("random_fraction must lie in [0, 1]")
        if self.horizontal_acceleration_limit <= 0.0 or self.vertical_acceleration_limit <= 0.0:
            raise ValueError("acceleration limits must be positive")


@dataclass(frozen=True)
class ProposalBatch:
    action_sequences: np.ndarray
    sources: tuple[ProposalSource, ...]

    def __post_init__(self) -> None:
        actions = np.asarray(self.action_sequences, dtype=np.float64)
        if actions.ndim != 3 or actions.shape[2] != 3 or actions.shape[0] != len(self.sources):
            raise ValueError("proposal actions and sources must be aligned")
        object.__setattr__(self, "action_sequences", actions.copy())


def _clip(actions: np.ndarray, config: ProposalConfig) -> np.ndarray:
    result = np.asarray(actions, dtype=np.float64).copy()
    horizontal = np.linalg.norm(result[..., :2], axis=-1)
    scale = np.minimum(1.0, config.horizontal_acceleration_limit / np.maximum(horizontal, 1e-15))
    result[..., :2] *= scale[..., None]
    result[..., 2] = np.clip(
        result[..., 2], -config.vertical_acceleration_limit, config.vertical_acceleration_limit
    )
    return result


def _constant(action: np.ndarray, horizon: int) -> np.ndarray:
    return np.repeat(np.asarray(action, dtype=np.float64)[None, :], horizon, axis=0)


def generate_trajectory_proposals(
    *,
    position: np.ndarray,
    velocity: np.ndarray,
    goal: np.ndarray,
    nominal_action: np.ndarray,
    config: ProposalConfig,
    rng: np.random.Generator,
    history: tuple[HistoryFrame, ...] = (),
    obstacles: np.ndarray | None = None,
    previous_safe_sequence: np.ndarray | None = None,
    learned_model: LearnedProposalModel | None = None,
) -> ProposalBatch:
    position = np.asarray(position, dtype=np.float64)
    velocity = np.asarray(velocity, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)
    nominal = np.asarray(nominal_action, dtype=np.float64)
    if any(value.shape != (3,) for value in (position, velocity, goal, nominal)):
        raise ValueError("position, velocity, goal and nominal_action must be (3,) vectors")
    horizon = config.horizon
    deterministic: list[tuple[np.ndarray, ProposalSource]] = []
    if history and history[-1].executed_control is not None:
        history_controls = [
            frame.executed_control for frame in history if frame.executed_control is not None
        ]
        extrapolated = np.mean(np.stack(history_controls[-min(4, len(history_controls)) :]), axis=0)
    else:
        extrapolated = nominal
    deterministic.append((_constant(extrapolated, horizon), ProposalSource.HISTORY_EXTRAPOLATION))
    speed = float(np.linalg.norm(velocity))
    braking = np.zeros(3) if speed <= 1e-12 else -velocity / speed * config.horizontal_acceleration_limit
    deterministic.append((_constant(braking, horizon), ProposalSource.BRAKING))
    goal_delta = goal - position
    goal_distance = float(np.linalg.norm(goal_delta))
    goal_direction = np.zeros(3) if goal_distance <= 1e-12 else goal_delta / goal_distance
    goal_action = config.goal_gain * goal_direction * config.horizontal_acceleration_limit - config.velocity_damping * velocity
    deterministic.append((_constant(goal_action, horizon), ProposalSource.GOAL_DIRECTED))
    deterministic.append((_constant(goal_action, horizon), ProposalSource.CLF_GUIDED))
    if obstacles is not None:
        centers = np.asarray(obstacles, dtype=np.float64)
        if centers.ndim != 2 or centers.shape[1] != 3:
            raise ValueError("obstacles must have shape (n, 3)")
        deltas = position[None, :] - centers
        nearest = deltas[int(np.argmin(np.linalg.norm(deltas, axis=1)))]
        direction = nearest / max(float(np.linalg.norm(nearest)), 1e-12)
        escape = direction * config.horizontal_acceleration_limit
        deterministic.append((_constant(escape, horizon), ProposalSource.OBSTACLE_ESCAPE))
        deterministic.append((_constant(escape, horizon), ProposalSource.CBF_GRADIENT))
    if previous_safe_sequence is not None:
        previous = np.asarray(previous_safe_sequence, dtype=np.float64)
        if previous.shape != (horizon, 3):
            raise ValueError("previous_safe_sequence must match configured horizon")
        shifted = np.concatenate((previous[1:], previous[-1:]), axis=0)
        deterministic.append((shifted, ProposalSource.SHIFTED_PREVIOUS))

    sequences: list[np.ndarray] = []
    sources: list[ProposalSource] = []
    for sequence, source in deterministic[: config.count]:
        sequences.append(sequence)
        sources.append(source)
    remaining = config.count - len(sequences)
    learned_count = 0
    if learned_model is not None and remaining > 0:
        learned_count = min(remaining, max(1, remaining // 4))
        learned = np.asarray(
            learned_model.propose(
                history=history,
                position=position,
                velocity=velocity,
                nominal_action=nominal,
                goal=goal,
                count=learned_count,
                horizon=horizon,
            ),
            dtype=np.float64,
        )
        if learned.shape != (learned_count, horizon, 3):
            raise ValueError("learned proposal model returned an invalid shape")
        sequences.extend(learned)
        sources.extend([ProposalSource.LEARNED] * learned_count)
        remaining -= learned_count
    if remaining > 0:
        nominal_sequence = np.broadcast_to(nominal, (remaining, horizon, 3)).copy()
        noise = rng.normal(0.0, config.perturbation_std, size=nominal_sequence.shape)
        time_correlation = np.cumsum(noise, axis=1) / np.sqrt(
            np.arange(1, horizon + 1, dtype=np.float64)[None, :, None]
        )
        random_sequences = nominal_sequence + time_correlation
        sequences.extend(random_sequences)
        sources.extend([ProposalSource.RANDOM_SAC] * remaining)
    actions = _clip(np.asarray(sequences, dtype=np.float64), config)
    return ProposalBatch(action_sequences=actions, sources=tuple(sources))
