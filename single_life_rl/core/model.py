"""Exact product version sets, avoiding enumeration of 2**34 models."""
from dataclasses import dataclass, field
import math


@dataclass
class VersionSet:
    dimensions: int
    known: dict = field(default_factory=dict)
    log_odds: dict = field(default_factory=dict)
    counts: dict = field(default_factory=dict)

    def cardinality(self):
        return 2 ** (self.dimensions - len(self.known))

    def common_optimal_policy(self, decision_dimensions):
        if all(i in self.known for i in range(decision_dimensions)):
            return tuple(self.known[i] for i in range(decision_dimensions))
        return None

    def includes(self, truth):
        return all(truth[i] == v for i, v in self.known.items())

    def update(self, experiment, observation, error_budget):
        i = experiment.bit
        if i in self.known:
            return False
        self.counts[i] = self.counts.get(i, 0) + 1
        k = experiment.kappa
        increment = (2 * observation - 1) * math.log((.5 + k) / (.5 - k)) if k else 0.
        self.log_odds[i] = self.log_odds.get(i, 0.) + increment
        threshold = math.log(1. / error_budget)
        if abs(self.log_odds[i]) >= threshold:
            self.known[i] = int(self.log_odds[i] > 0)
            return True
        return False
