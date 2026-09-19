"""One cooperative account shared by an entire planner selection."""
from dataclasses import dataclass,field
from time import monotonic
from .protocols import PlanningLimit

@dataclass
class SearchBudget:
    max_expansions: int = 20000
    max_model_calls: int = 100000
    max_seconds: float = 2.0
    expansions: int = 0
    model_calls: int = 0
    started: float = field(default_factory=monotonic)

    def __post_init__(self):
        if self.max_expansions<0 or self.max_model_calls<0 or self.max_seconds<=0:
            raise ValueError("invalid search budget")

    def consume(self,expansions=0,model_calls=0):
        if expansions<0 or model_calls<0:
            raise ValueError("negative charge")
        if monotonic()-self.started>self.max_seconds:
            raise PlanningLimit("global wall-time budget")
        if self.expansions+expansions>self.max_expansions:
            raise PlanningLimit("global expansion budget")
        if self.model_calls+model_calls>self.max_model_calls:
            raise PlanningLimit("global model-call budget")
        self.expansions += expansions
        self.model_calls += model_calls

    def snapshot(self):
        return dict(expansions=self.expansions,model_calls=self.model_calls,seconds=monotonic()-self.started)

def reset_fallback(problem,remaining,budget):
    from .protocols import Protocol
    for op in problem.operations:
        if op.source == problem.reset:
            budget.consume(expansions=1)
            if (op.target == problem.reset and op.duration<=remaining and op.energy<=problem.capacity
                and len(op.channels)<=problem.max_measurements
                and all(op.channels.count(c)<=problem.per_channel_limit for c in op.channels)):
                return Protocol((op.name,),op.duration,op.channels)
    return None
