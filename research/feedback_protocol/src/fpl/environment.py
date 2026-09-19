"""Evaluator-owned truth/RNG; policies receive only PlannerState and PublicProblem."""
from dataclasses import dataclass
import random
from .problem import PublicProblem
from .protocols import validate_protocol
from .belief import PlannerState, ObservationHistory, update


@dataclass(frozen=True)
class PrivateTruth:
    hypothesis_index: int
    noise_seed: int


@dataclass(frozen=True)
class BatchResult:
    operations: tuple[dict, ...]
    feedback: tuple[tuple[str, int], ...]
    reward: int
    released_at: int


class Environment:
    def __init__(self, problem: PublicProblem, truth: PrivateTruth):
        if not 0 <= truth.hypothesis_index < len(problem.hypotheses):
            raise ValueError("invalid private label")
        self.problem = problem
        self._truth = truth
        self._rng = random.Random(truth.noise_seed)
        self.time = 0
        self._belief = problem.prior
        self._history = ObservationHistory()

    def planner_state(self):
        return PlannerState(self._belief, self.problem.budget-self.time,
                            self.problem.capacity, self._history)

    def execute(self, names: tuple[str, ...]) -> BatchResult:
        p = self.problem
        route = validate_protocol(p, names, p.budget-self.time)
        feedback, events = [], []
        battery = p.capacity
        for name in route.operations:
            op = next(o for o in p.operations if o.name == name)
            start, before = self.time, battery
            battery -= op.energy
            after_debit = battery
            self.time += op.duration
            for channel in op.channels:
                mean = p.hypotheses[self._truth.hypothesis_index][p.channels.index(channel)]
                feedback.append((channel, int(self._rng.random() < mean)))
            if op.target == p.reset:
                battery = p.capacity
            events.append(dict(operation=name, start=start, end=self.time,
                               resource_before=before, resource_after_debit=after_debit,
                               resource_after=battery, source=op.source, target=op.target,
                               measured_channels=op.channels))
        # No callback or policy call is made while physical execution is pending.
        self._belief = update(p, self._belief, feedback)
        self._history = ObservationHistory(self._history.released+tuple(feedback))
        return BatchResult(tuple(events), tuple(feedback), sum(bit for _, bit in feedback), self.time)
