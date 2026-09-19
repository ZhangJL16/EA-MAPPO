"""Catalogue-free exact completion mask for deterministic committed protocols.

Boolean dynamic programming, not path enumeration. Worst-case state space is
pseudo-polynomial in time/energy and exponential in channel count; a work cap
raises PlanningLimit, never silently declares a feasible action illegal.
"""
from dataclasses import dataclass
import json
from fpl.protocols import PlanningLimit


def token(kind, name=None):
    return json.dumps([kind, name], separators=(",", ":"))


STOP, END = token("stop"), token("end")


@dataclass(frozen=True)
class ProtocolDecoderState:
    prefix: tuple
    node: str
    time_left: int
    energy_left: int
    counts: tuple
    complete: bool


class ScalableFeasibilityMask:
    def __init__(self, problem, max_states=100000, budget=None):
        self.p, self.max_states, self.budget = problem, max_states, budget
        self.ops = {o.name: o for o in problem.operations}
        self.adj = {n: tuple(sorted((o for o in problem.operations if o.source == n),
                                   key=lambda o: o.name)) for n in problem.nodes}
        self.index = {c: i for i, c in enumerate(problem.channels)}
        self.cache, self.states, self.edge_checks = {}, 0, 0

    def step(self, node, time, energy, counts, op):
        self.edge_checks += 1
        if self.budget is not None:
            self.budget.consume(expansions=1)
        if op.source != node or op.duration > time or op.energy > energy:
            return None
        cs = list(counts)
        for c in op.channels:
            cs[self.index[c]] += 1
        if sum(cs) > self.p.max_measurements or any(c > self.p.per_channel_limit for c in cs):
            return None
        return op.target, time-op.duration, energy-op.energy, tuple(cs)

    def can_return(self, node, time, energy, counts):
        if node == self.p.reset:
            return True  # Debit has already happened; first reset ends batch.
        key = node, time, energy, counts
        if key not in self.cache:
            if self.states >= self.max_states:
                raise PlanningLimit("decoder completion DP state cap")
            self.states += 1
            self.cache[key] = False
            # Non-observing return arcs first; exact fallback covers general graphs.
            for op in sorted(self.adj[node], key=lambda o: (bool(o.channels), o.name)):
                nxt = self.step(*key, op)
                if nxt is not None and self.can_return(*nxt):
                    self.cache[key] = True
                    break
        return self.cache[key]

    def state(self, prefix, remaining):
        if type(remaining) is not int or remaining < 0:
            raise ValueError("invalid remaining horizon")
        key = self.p.reset, remaining, self.p.capacity, (0,)*len(self.p.channels)
        for i, name in enumerate(prefix):
            if i and key[0] == self.p.reset:
                raise ValueError("prefix continues after reset")
            if name not in self.ops:
                raise ValueError("unknown operation")
            nxt = self.step(*key, self.ops[name])
            if nxt is None:
                raise ValueError("infeasible prefix")
            key = nxt
        return ProtocolDecoderState(tuple(prefix), *key, bool(prefix) and key[0] == self.p.reset)

    def legal_next(self, prefix, remaining):
        s = self.state(prefix, remaining)
        if s.complete:
            return [END]
        result = [STOP] if not prefix else []
        for op in self.adj[s.node]:
            nxt = self.step(s.node, s.time_left, s.energy_left, s.counts, op)
            if nxt is not None and self.can_return(*nxt):
                result.append(token("operation", op.name))
        return sorted(result)
