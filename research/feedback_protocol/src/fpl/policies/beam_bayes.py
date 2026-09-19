"""Bounded prefix generation + finite-depth posterior lookahead, no full catalogue."""
from collections import Counter
from fractions import Fraction as F
from time import monotonic
from ..protocols import Protocol, PlanningLimit
from ..utility import protocol_value
from ..belief import outcomes


def route_score(problem, prior, route):
    # Heuristic for candidate generation only, not a Bayes value certificate.
    uncertainty = F(0)
    for c in set(route.channels):
        j = problem.channels.index(c)
        mean = sum(w*row[j] for w,row in zip(prior,problem.hypotheses))
        uncertainty += sum(w*(row[j]-mean)**2 for w,row in zip(prior,problem.hypotheses))
    return (protocol_value(problem,prior,route)+uncertainty)/route.duration


def beam_protocols(problem, remaining, prior, width=8, max_expansions=2000):
    if width < 1 or max_expansions < 1:
        raise ValueError("positive search limits required")
    frontier = [(problem.reset,problem.capacity,0,(),())]
    completed, expanded, truncated = {}, 0, False
    while frontier:
        pending = []
        for node,battery,t,names,channels in frontier:
            for op in problem.operations:
                if op.source != node:
                    continue
                expanded += 1
                if expanded > max_expansions:
                    return tuple(completed.values()), expanded-1, True
                cs = channels+op.channels
                if (op.energy > battery or t+op.duration > remaining or
                    len(cs)>problem.max_measurements or
                    any(v>problem.per_channel_limit for v in Counter(cs).values())):
                    continue
                ns, duration = names+(op.name,), t+op.duration
                if op.target == problem.reset:
                    completed[ns] = Protocol(ns,duration,cs)
                else:
                    pending.append((op.target,battery-op.energy,duration,ns,cs))
        pending.sort(key=lambda x: route_score(problem,prior,Protocol(x[3],x[2],x[4])), reverse=True)
        truncated |= len(pending)>width
        frontier = pending[:width]
        # Bound completed storage too; pruning changes approximation, not legality.
        if len(completed)>width:
            keep = sorted(completed.values(),key=lambda r:route_score(problem,prior,r),reverse=True)[:width]
            completed = {r.operations:r for r in keep}
            truncated = True
    return tuple(completed.values()), expanded, truncated


class BeamBayes:
    def __init__(self, problem, width=8, depth=2, max_states=1000,
                 max_expansions=2000, max_seconds=2, max_outcomes=256):
        if depth<1 or max_states<1 or max_seconds<=0:
            raise ValueError("positive planner limits required")
        self.problem, self.width, self.depth = problem,width,depth
        self.max_states,self.max_expansions,self.max_seconds = max_states,max_expansions,max_seconds
        self.max_outcomes = max_outcomes

    def select(self,state):
        if state.resource != self.problem.capacity:
            raise ValueError("selection only at reset")
        start = monotonic()
        self.stats = dict(states=0, expansions=0, likelihood_branches=0, truncated=False)
        candidates, count, pruned = beam_protocols(self.problem,state.remaining,state.posterior,
                                                  self.width,self.max_expansions)
        self.stats["expansions"] += count
        self.stats["truncated"] |= pruned
        best = max(candidates,key=lambda r:route_score(self.problem,state.posterior,r),default=None)

        def q(route,t,prior,depth):
            result = protocol_value(self.problem,prior,route)
            if depth>1:
                for _,mass,posterior in outcomes(self.problem,prior,route.channels,self.max_outcomes):
                    self.stats["likelihood_branches"] += 1
                    result += mass*value(t-route.duration,posterior,depth-1)
            else:
                # Feasible open-loop tail: repeat one protocol, ignore feedback.
                routes,n,pruned = beam_protocols(self.problem,t-route.duration,prior,
                                                 self.width,self.max_expansions)
                self.stats["expansions"] += n
                self.stats["truncated"] |= pruned
                result += max([F(0)]+[(t-route.duration)//r.duration * protocol_value(self.problem,prior,r)
                                      for r in routes])
            return result

        def value(t,prior,depth):
            if self.stats["states"] >= self.max_states or monotonic()-start>self.max_seconds:
                raise PlanningLimit("approximate search exhausted")
            self.stats["states"] += 1
            routes,n,pruned = beam_protocols(self.problem,t,prior,self.width,self.max_expansions)
            self.stats["expansions"] += n
            self.stats["truncated"] |= pruned
            return max([F(0)]+[q(r,t,prior,depth) for r in routes])

        best_value = None
        try:
            for route in candidates:
                if monotonic()-start>self.max_seconds:
                    raise PlanningLimit("approximate search time exhausted")
                score = q(route,state.remaining,state.posterior,self.depth)
                if best_value is None or score>best_value:
                    best_value,best = score,route
        except PlanningLimit:
            self.stats["truncated"] = True
        self.stats["seconds"] = monotonic()-start
        return best
