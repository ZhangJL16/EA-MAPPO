from dataclasses import dataclass
from fractions import Fraction as F
from itertools import product
from .problem import PublicProblem
from .protocols import PlanningLimit


@dataclass(frozen=True)
class ObservationHistory:
    released: tuple[tuple[str, int], ...] = ()


@dataclass(frozen=True)
class PlannerState:
    posterior: tuple[F, ...]
    remaining: int
    resource: int
    history: ObservationHistory = ObservationHistory()


def update(p: PublicProblem, prior: tuple[F, ...], feedback):
    if len(prior) != len(p.hypotheses) or min(prior) < 0 or sum(prior) != 1:
        raise ValueError("invalid belief")
    weights = list(prior)
    for channel, bit in feedback:
        if bit not in (0, 1):
            raise ValueError("Bernoulli outcome required")
        j = p.channels.index(channel)
        for i, row in enumerate(p.hypotheses):
            weights[i] *= row[j] if bit else 1-row[j]
    mass = sum(weights)
    if not mass:
        raise ValueError("observation impossible under public belief")
    return tuple(w/mass for w in weights)


def expected_reward(p, prior, channels):
    return sum((w*sum((row[p.channels.index(c)] for c in channels), F(0))
                for w, row in zip(prior, p.hypotheses)), F(0))


def outcomes(p, prior, channels, max_outcomes=4096, budget=None):
    if 2**len(channels) > max_outcomes:
        raise PlanningLimit("joint-feedback enumeration limit")
    for bits in product((0, 1), repeat=len(channels)):
        if budget is not None:
            budget.consume(model_calls=len(p.hypotheses)*max(1,len(channels)))
        feedback = tuple(zip(channels, bits))
        mass = F(0)
        weights = []
        for w, row in zip(prior, p.hypotheses):
            likelihood = w
            for c, bit in feedback:
                v = row[p.channels.index(c)]
                likelihood *= v if bit else 1-v
            mass += likelihood
            weights.append(likelihood)
        if mass:
            yield feedback, mass, tuple(w/mass for w in weights)
