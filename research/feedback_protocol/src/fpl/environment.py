"""Evaluator-owned truth/RNG; policies receive only PlannerState and PublicProblem."""
from dataclasses import dataclass
import random
import hashlib
import json
from fractions import Fraction as F
from .utility import operation_score
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
    reward: F  # Evaluator score; never passed to policy as feedback.
    released_at: int


class Environment:
    def __init__(self, problem: PublicProblem, truth: PrivateTruth, crn_key=None):
        if not 0 <= truth.hypothesis_index < len(problem.hypotheses):
            raise ValueError("invalid private label")
        self.problem = problem
        self._truth = truth
        self._rng = random.Random(truth.noise_seed)
        self._crn_key = crn_key
        self._query_counts = {}
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
        reward = F(0)
        battery = p.capacity
        for name in route.operations:
            op = next(o for o in p.operations if o.name == name)
            start, before = self.time, battery
            battery -= op.energy
            after_debit = battery
            self.time += op.duration
            bits = []
            for channel in op.channels:
                mean = p.hypotheses[self._truth.hypothesis_index][p.channels.index(channel)]
                if self._crn_key is None:
                    draw = self._rng.random()
                else:
                    index = self._query_counts.get(channel,0)
                    payload = ("fpl-crn-v1",self._crn_key,self._truth.hypothesis_index,
                               self._truth.noise_seed,channel,index)
                    seed = int.from_bytes(hashlib.sha256(json.dumps(payload).encode()).digest(),"big")
                    draw = random.Random(seed).random()
                    self._query_counts[channel] = index+1
                bit = int(draw < mean)
                feedback.append((channel, bit))
                bits.append(bit)
            reward += operation_score(op, self._truth.hypothesis_index, bits)
            if op.target == p.reset:
                battery = p.capacity
            events.append(dict(operation=name, start=start, end=self.time,
                               resource_before=before, resource_after_debit=after_debit,
                               resource_after=battery, source=op.source, target=op.target,
                               measured_channels=op.channels))
        # No callback or policy call is made while physical execution is pending.
        self._belief = update(p, self._belief, feedback)
        self._history = ObservationHistory(self._history.released+tuple(feedback))
        return BatchResult(tuple(events), tuple(feedback), reward, self.time)
