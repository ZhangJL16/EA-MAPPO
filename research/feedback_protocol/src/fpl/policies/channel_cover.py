from collections import Counter
from fractions import Fraction
from ..belief import expected_reward
from ..protocols import enumerate_protocols
from ..utility import protocol_value
from ..budget import SearchBudget, reset_fallback
from ..protocols import PlanningLimit


class ChannelCover:
    """Cover informative primitive channels once, then posterior reward/time.

    A simple baseline, not a KL allocation algorithm or optimality claim.
    """
    def __init__(self, problem, budget_limits=None):
        self.problem = problem
        self.budget_limits = budget_limits or {}

    def select(self, state, budget=None):
        p = self.problem
        if state.resource != p.capacity:
            raise ValueError("selection only at reset")
        budget = budget if budget is not None else SearchBudget(**self.budget_limits)
        best = None
        self.stats = {"truncated":False}
        try:
            best = reset_fallback(p,state.remaining,budget)
            routes = enumerate_protocols(p,state.remaining,budget=budget)
            seen = Counter(c for c, _ in state.history.released)
            budget.consume(model_calls=len(p.channels)*len(p.hypotheses))
            uncovered = {c for j,c in enumerate(p.channels)
                         if not seen[c] and len({row[j] for row in p.hypotheses})>1}
            def score(route):
                return (Fraction(len(set(route.channels)&uncovered),route.duration),
                        protocol_value(p,state.posterior,route,budget)/route.duration)
            best = max(routes,key=score,default=best)
        except PlanningLimit as exc:
            self.stats.update(truncated=True,limit_reason=str(exc))
        self.stats.update(budget.snapshot())
        return best
