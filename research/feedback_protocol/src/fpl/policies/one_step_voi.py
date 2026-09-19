"""One posterior update followed by a budgeted nonadaptive task plan.

This is a bounded candidate knowledge-gradient baseline, not exact VOI over
all protocols. The tail solves an open-loop unbounded knapsack over the same
bounded route generator, with no further feedback adaptation.
"""
from fractions import Fraction as F
from ..budget import SearchBudget,reset_fallback
from ..protocols import PlanningLimit
from ..belief import outcomes
from ..utility import protocol_value
from .beam_bayes import beam_protocols


class OneStepVOI:
    def __init__(self,problem,width=8,budget_limits=None):
        self.problem,self.width = problem,width
        self.budget_limits = budget_limits or {}

    def select(self,state):
        if state.resource!=self.problem.capacity:
            raise ValueError("selection only at reset")
        p,budget = self.problem,SearchBudget(**self.budget_limits)
        best,bestq = None,F(0)
        self.stats = dict(truncated=False,likelihood_branches=0)
        cache = {}
        def tail(t,b):
            key = (t,b)
            if key not in cache:
                routes,_,_ = beam_protocols(p,t,b,self.width,budget=budget)
                values = [(r.duration,protocol_value(p,b,r,budget)) for r in routes]
                dp = [F(0)]*(t+1)
                for h in range(1,t+1):
                    for d,v in values:
                        budget.consume(expansions=1)
                        if d<=h:
                            dp[h] = max(dp[h],v+dp[h-d])
                cache[key] = dp[t]
            return cache[key]
        try:
            best = reset_fallback(p,state.remaining,budget)
            routes,_,pruned = beam_protocols(p,state.remaining,state.posterior,self.width,budget=budget)
            self.stats["pruned"] = pruned
            for route in routes:
                q = protocol_value(p,state.posterior,route,budget)
                for _,mass,post in outcomes(p,state.posterior,route.channels,budget=budget):
                    self.stats["likelihood_branches"] += 1
                    q += mass*tail(state.remaining-route.duration,post)
                if q>bestq:
                    best,bestq = route,q
            if bestq<=0:
                best = None
        except PlanningLimit as exc:
            self.stats.update(truncated=True,limit_reason=str(exc))
        self.stats.update(budget.snapshot())
        return best
