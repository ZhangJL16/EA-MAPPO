from collections import Counter
from fractions import Fraction
from ..belief import expected_reward
from ..protocols import enumerate_protocols
from ..utility import protocol_value


class ChannelCover:
    """Cover informative primitive channels once, then posterior reward/time.

    A simple baseline, not a KL allocation algorithm or optimality claim.
    """
    def __init__(self, problem):
        self.problem = problem

    def select(self, state):
        p = self.problem
        if state.resource != p.capacity:
            raise ValueError("selection only at reset")
        routes = enumerate_protocols(p, state.remaining)
        if not routes:
            return None
        seen = Counter(c for c, _ in state.history.released)
        uncovered = {c for j, c in enumerate(p.channels)
                     if not seen[c] and len({row[j] for row in p.hypotheses}) > 1}
        def score(route):
            coverage = Fraction(len(set(route.channels) & uncovered), route.duration)
            rate = protocol_value(p, state.posterior, route)/route.duration
            return coverage, rate
        return max(routes, key=score)
