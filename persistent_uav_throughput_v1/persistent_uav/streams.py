"""Private exogenous workload. Schedulers only receive already arrived Task objects."""
from dataclasses import asdict, dataclass
import hashlib
import json

import numpy as np

from .config import ceil_grid


@dataclass(frozen=True)
class Task:
    id: int
    arrival: float
    original_arrival: float
    position: tuple[float, float, float]


def legal_position(rng, layout):
    """Environment generation only; no agent-relative rejection or rollout oracle."""
    for _ in range(100_000):
        pos = rng.uniform([100., 100., 20.], [3900., 3900., 380.])
        # Match goal-clearance legality used by the legacy cylinder physics.
        if all(np.linalg.norm(pos[:2] - np.asarray(o['position'])[:2]) > o['radius'] + 5.5
               for o in layout):
            return tuple(float(x) for x in pos)
    raise RuntimeError('legal task-space sampling exhausted; do not silently change map')


def workload(seed, config, layout):
    arrival_rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
    position_rng = np.random.default_rng(np.random.SeedSequence([seed, 2]))
    rows = [Task(i, 0., 0., legal_position(position_rng, layout))
            for i in range(config.initial_tasks)]
    raw_time = 0.
    while True:
        raw_time += float(arrival_rng.exponential(1. / config.arrival_rate))
        effective = ceil_grid(raw_time)
        if effective > config.cutoff + 1e-8:
            break
        rows.append(Task(len(rows), effective, raw_time, legal_position(position_rng, layout)))
    return tuple(rows)


def stream_hash(tasks):
    return hashlib.sha256(json.dumps([asdict(t) for t in tasks], sort_keys=True).encode()).hexdigest()
