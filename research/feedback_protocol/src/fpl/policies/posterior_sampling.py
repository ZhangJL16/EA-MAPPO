"""Independent posterior-sampling baseline over bounded generated protocols."""
import random
from fractions import Fraction as F
from .beam_bayes import beam_protocols
from ..utility import protocol_value
from ..budget import SearchBudget, reset_fallback
from ..protocols import PlanningLimit


class PosteriorSampling:
    def __init__(self,problem,seed=0,width=8,max_expansions=2000,budget_limits=None):
        self.problem,self.rng = problem,random.Random(seed)
        self.width,self.max_expansions = width,max_expansions
        self.budget_limits = budget_limits or dict(max_expansions=max_expansions)

    def select(self,state):
        if state.resource != self.problem.capacity:
            raise ValueError("selection only at reset")
        index = self.rng.choices(range(len(state.posterior)),weights=list(map(float,state.posterior)))[0]
        belief = tuple(F(int(i == index)) for i in range(len(state.posterior)))
        budget = SearchBudget(**self.budget_limits)
        best = None
        self.stats = {"truncated":False}
        try:
            best = reset_fallback(self.problem,state.remaining,budget)
            routes,n,pruned = beam_protocols(self.problem,state.remaining,belief,self.width,
                                            self.max_expansions,budget=budget)
            self.stats["pruned"] = pruned
            best = max(routes,key=lambda r:protocol_value(self.problem,belief,r,budget)/r.duration,default=best)
        except PlanningLimit as exc:
            self.stats.update(truncated=True,limit_reason=str(exc))
        self.stats.update(budget.snapshot())
        return best
