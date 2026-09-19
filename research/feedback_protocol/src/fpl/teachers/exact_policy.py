"""Exact feedback-tree integration of a fixed, complete policy state.

No optimization and no belief-only memoization. deepcopy preserves RNG state,
history-sensitive logic and remaining episode work separately on every branch.
Only full-support public priors are accepted for hypothesis-wise certification.
"""
from copy import deepcopy
from fractions import Fraction as F
from time import monotonic
from ..belief import PlannerState,ObservationHistory,outcomes
from ..utility import protocol_vector
from ..protocols import validate_protocol,PlanningLimit
from ..work import WatchdogExpired


class ExactPolicyEvaluator:
    def __init__(self,problem,max_nodes=10000,max_feedback=4096,watchdog_seconds=60):
        if min(problem.prior)<=0:
            raise ValueError("hypothesis-wise evaluation requires full-support initial prior")
        self.problem,self.max_nodes,self.max_feedback = problem,max_nodes,max_feedback
        self.watchdog_seconds = watchdog_seconds

    def evaluate(self,policy):
        p,n = self.problem,len(self.problem.hypotheses)
        started,nodes,branches = monotonic(),0,0
        worst_exp,worst_calls = 0,0
        actions = {}
        def visit(agent,state):
            nonlocal nodes,branches,worst_exp,worst_calls
            if monotonic()-started>self.watchdog_seconds:
                raise WatchdogExpired("fixed-policy evaluation watchdog")
            if nodes>=self.max_nodes:
                raise PlanningLimit("fixed-policy evaluation node cap")
            nodes += 1
            if state.remaining==0:
                return ((F(0),)*n,)*4
            tick = monotonic()
            route = agent.select(state)
            seconds = monotonic()-tick
            stats = getattr(agent,"stats",{})
            exp,calls = stats.get("expansions",0),stats.get("model_calls",0)
            worst_exp = max(worst_exp,getattr(agent,"expansions",0))
            worst_calls = max(worst_calls,getattr(agent,"model_calls",0))
            actions[str(nodes)] = dict(history=state.history.released,remaining=state.remaining,
                                      operations=None if route is None else route.operations)
            if route is None:
                return ((F(0),)*n,(F(exp),)*n,(F(calls),)*n,(F(seconds),)*n)
            route = validate_protocol(p,route.operations,state.remaining)
            vectors = [list(protocol_vector(p,route)),[F(exp)]*n,[F(calls)]*n,[F(seconds)]*n]
            for feedback,_,posterior in outcomes(p,state.posterior,route.channels,self.max_feedback):
                branches += 1
                child_state = PlannerState(posterior,state.remaining-route.duration,p.capacity,
                                           ObservationHistory(state.history.released+feedback))
                tails = visit(deepcopy(agent),child_state)
                for h,row in enumerate(p.hypotheses):
                    probability = F(1)
                    for c,y in feedback:
                        mu = row[p.channels.index(c)]
                        probability *= mu if y else 1-mu
                    for v,tail in zip(vectors,tails):
                        v[h] += probability*tail[h]
            return tuple(tuple(v) for v in vectors)
        try:
            vectors = visit(deepcopy(policy),PlannerState(p.prior,p.budget,p.capacity))
            return dict(status="exact_conditional_policy",utility_by_hypothesis=list(map(str,vectors[0])),
                        expansions_by_hypothesis=list(map(str,vectors[1])),model_calls_by_hypothesis=list(map(str,vectors[2])),
                        measured_latency_by_hypothesis=list(map(float,vectors[3])),
                        nodes=nodes,feedback_branches=branches,wall_seconds=monotonic()-started,
                        worst_episode_expansions=worst_exp,worst_episode_model_calls=worst_calls,
                        action_map=actions)
        except (PlanningLimit,WatchdogExpired) as exc:
            return dict(status="unresolved",reason=str(exc),nodes=nodes,feedback_branches=branches,
                        wall_seconds=monotonic()-started)
