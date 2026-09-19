"""Prefix beam and finite-depth Bayes search with one global selection budget."""
from collections import Counter
from fractions import Fraction as F
from ..protocols import Protocol,PlanningLimit
from ..utility import protocol_value
from ..belief import outcomes
from ..budget import SearchBudget,reset_fallback

def route_score(problem,prior,route,budget=None):
    uncertainty = F(0)
    for c in set(route.channels):
        if budget is not None:
            budget.consume(model_calls=2*len(prior))
        j = problem.channels.index(c)
        mean = sum(w*row[j] for w,row in zip(prior,problem.hypotheses))
        uncertainty += sum(w*(row[j]-mean)**2 for w,row in zip(prior,problem.hypotheses))
    return (protocol_value(problem,prior,route,budget)+uncertainty)/route.duration

def beam_protocols(problem,remaining,prior,width=8,max_expansions=2000,budget=None):
    if width<1:
        raise ValueError("positive beam width required")
    budget = budget if budget is not None else SearchBudget(max_expansions=max_expansions)
    start_exp = budget.expansions
    frontier = [(problem.reset,problem.capacity,0,(),())]
    completed,pruned = {},False
    while frontier:
        pending = []
        for node,battery,t,names,channels in frontier:
            for op in problem.operations:
                if op.source != node:
                    continue
                budget.consume(expansions=1)
                cs = channels+op.channels
                if (op.energy>battery or t+op.duration>remaining or len(cs)>problem.max_measurements
                    or any(v>problem.per_channel_limit for v in Counter(cs).values())):
                    continue
                ns,duration = names+(op.name,),t+op.duration
                if op.target == problem.reset:
                    completed[ns] = Protocol(ns,duration,cs)
                else:
                    pending.append((op.target,battery-op.energy,duration,ns,cs))
        pending.sort(key=lambda x:route_score(problem,prior,Protocol(x[3],x[2],x[4]),budget),reverse=True)
        pruned |= len(pending)>width
        frontier = pending[:width]
        if len(completed)>width:
            keep = sorted(completed.values(),key=lambda r:route_score(problem,prior,r,budget),reverse=True)[:width]
            completed = {r.operations:r for r in keep}
            pruned = True
    return tuple(completed.values()),budget.expansions-start_exp,pruned

class BeamBayes:
    def __init__(self,problem,width=8,depth=2,max_states=1000,max_expansions=2000,
                 max_seconds=2,max_outcomes=256,budget_limits=None):
        if width<1 or depth<1 or max_states<1:
            raise ValueError("positive search limits required")
        self.problem,self.width,self.depth = problem,width,depth
        self.max_states,self.max_outcomes = max_states,max_outcomes
        self.limits = budget_limits or dict(max_expansions=max_expansions,max_model_calls=100000,max_seconds=max_seconds)

    def select(self,state):
        if state.resource != self.problem.capacity:
            raise ValueError("selection only at reset")
        budget = SearchBudget(**self.limits)
        self.stats = dict(states=0,likelihood_branches=0,truncated=False)
        best = None

        def candidates(t,prior):
            routes,_,pruned = beam_protocols(self.problem,t,prior,self.width,budget=budget)
            self.stats["truncated"] |= pruned
            return routes

        def tail(t,prior):
            return max([F(0)]+[(t//r.duration)*protocol_value(self.problem,prior,r,budget)
                              for r in candidates(t,prior)])

        def q(route,t,prior,depth):
            result = protocol_value(self.problem,prior,route,budget)
            if depth>1:
                for _,mass,posterior in outcomes(self.problem,prior,route.channels,self.max_outcomes,budget):
                    self.stats["likelihood_branches"] += 1
                    result += mass*value(t-route.duration,posterior,depth-1)
            else:
                result += tail(t-route.duration,prior)
            return result

        def value(t,prior,depth):
            budget.consume()
            if self.stats["states"]>=self.max_states:
                raise PlanningLimit("state limit")
            self.stats["states"] += 1
            return max([F(0)]+[q(r,t,prior,depth) for r in candidates(t,prior)])

        try:
            best = reset_fallback(self.problem,state.remaining,budget)
            routes = candidates(state.remaining,state.posterior)
            best_value,selected = F(0),None
            for route in routes:
                score = q(route,state.remaining,state.posterior,self.depth)
                if score>best_value:
                    best_value,selected = score,route
                    best = selected
            best = selected
        except PlanningLimit as exc:
            self.stats.update(truncated=True,limit_reason=str(exc))
        self.stats.update(budget.snapshot())
        return best
