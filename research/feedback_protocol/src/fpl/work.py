"""Deterministic policy identity; elapsed time can invalidate but not choose."""
from dataclasses import dataclass,field
from time import monotonic
from .protocols import PlanningLimit,Protocol


class WatchdogExpired(RuntimeError):
    """Abnormal computation: no fallback action/result is authorized."""


@dataclass
class PlanningWorkBudget:
    max_expansions: int = 3000
    max_model_calls: int = 30000
    watchdog_seconds: float = 60
    expansions: int = 0
    model_calls: int = 0
    started: float = field(default_factory=monotonic)

    def __post_init__(self):
        if (type(self.max_expansions) is not int or type(self.max_model_calls) is not int
            or min(self.max_expansions,self.max_model_calls)<0 or self.watchdog_seconds<=0):
            raise ValueError("invalid deterministic work limits")

    def consume(self,expansions=0,model_calls=0):
        if min(expansions,model_calls)<0:
            raise ValueError("negative work")
        if monotonic()-self.started>self.watchdog_seconds:
            raise WatchdogExpired("watchdog invalidates the evaluation; not a policy cutoff")
        if self.expansions+expansions>self.max_expansions:
            raise PlanningLimit("deterministic expansion allowance")
        if self.model_calls+model_calls>self.max_model_calls:
            raise PlanningLimit("deterministic model allowance")
        self.expansions += expansions
        self.model_calls += model_calls

    def snapshot(self):
        return dict(expansions=self.expansions,model_calls=self.model_calls,seconds=monotonic()-self.started)


class WorkPolicy:
    """Same per-select limits, optionally ALSO a shared whole-episode allowance.

    Physical execution continues with a fixed reset self-loop if planning
    exhausts its allowance. This public, score-blind autopilot needs no model
    calls. It is identical across methods and can be suboptimal.
    """
    def __init__(self,problem,policy,selection_limits,episode_limits=None,watchdog_seconds=60):
        self.problem,self.policy = problem,policy
        self.selection_limits,self.episode_limits = selection_limits,episode_limits
        self.watchdog_seconds = watchdog_seconds
        self.expansions,self.model_calls = 0,0
        self.fallback = next((o for o in problem.operations if o.source==o.target==problem.reset
                              and not o.channels and o.energy<=problem.capacity),None)
        self.stats = {}

    def select(self,state):
        limits = dict(self.selection_limits)
        if self.episode_limits is not None:
            limits["max_expansions"] = min(limits["max_expansions"],self.episode_limits["max_expansions"]-self.expansions)
            limits["max_model_calls"] = min(limits["max_model_calls"],self.episode_limits["max_model_calls"]-self.model_calls)
        account = PlanningWorkBudget(**limits,watchdog_seconds=self.watchdog_seconds)
        route = self.policy.select(state,budget=account)
        self.expansions += account.expansions
        self.model_calls += account.model_calls
        self.stats = dict(getattr(self.policy,"stats",{}),**account.snapshot())
        if route is None and self.stats.get("limit_reason") and self.fallback is not None:
            o = self.fallback
            if o.duration<=state.remaining:
                route = Protocol((o.name,),o.duration,())
                self.stats["autopilot"] = True
        return route
