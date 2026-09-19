"""Immutable public contract. Probabilities are exact rational numbers."""
from dataclasses import dataclass, asdict
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path


@dataclass(frozen=True)
class Operation:
    name: str
    source: str
    target: str
    duration: int
    energy: int
    channels: tuple[str, ...] = ()


@dataclass(frozen=True)
class PublicProblem:
    instance_id: str
    family_id: str
    split_id: str
    nodes: tuple[str, ...]
    reset: str
    channels: tuple[str, ...]
    hypotheses: tuple[tuple[F, ...], ...]
    prior: tuple[F, ...]
    operations: tuple[Operation, ...]
    capacity: int
    budget: int
    max_measurements: int
    per_channel_limit: int = 1
    terminal_rule: str = "reset_by_budget"
    feedback_rule: str = "batch_end"
    reset_rule: str = "debit_before_arrival_reload"
    objective: str = "cumulative_bernoulli_reward"

    def __post_init__(self):
        for name in ("capacity", "budget", "max_measurements", "per_channel_limit"):
            value = getattr(self, name)
            if type(value) is not int or value < (1 if name == "per_channel_limit" else 0):
                raise ValueError(f"invalid {name}")
        if (self.terminal_rule, self.feedback_rule, self.reset_rule, self.objective) != (
            "reset_by_budget", "batch_end", "debit_before_arrival_reload", "cumulative_bernoulli_reward"
        ):
            raise ValueError("unsupported contract; explicit adapter required")
        if not self.nodes or len(set(self.nodes)) != len(self.nodes) or self.reset not in self.nodes:
            raise ValueError("invalid nodes/reset")
        if len(set(self.channels)) != len(self.channels) or not self.hypotheses:
            raise ValueError("invalid channels/hypotheses")
        if len(self.prior) != len(self.hypotheses) or min(self.prior) < 0 or sum(self.prior) != 1:
            raise ValueError("prior must be a probability vector")
        if any(len(row) != len(self.channels) or any(p < 0 or p > 1 for p in row)
               for row in self.hypotheses):
            raise ValueError("invalid Bernoulli model matrix")
        if len({o.name for o in self.operations}) != len(self.operations):
            raise ValueError("duplicate operation name")
        for op in self.operations:
            if op.source not in self.nodes or op.target not in self.nodes:
                raise ValueError("unknown operation endpoint")
            if type(op.duration) is not int or op.duration <= 0 or type(op.energy) is not int or op.energy < 0:
                raise ValueError("duration must be positive; energy nonnegative integers")
            if any(c not in self.channels for c in op.channels):
                raise ValueError("unknown observation channel")

    @property
    def instance_hash(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True, default=str).encode()).hexdigest()


def load_problem(path: str | Path) -> PublicProblem:
    data = json.loads(Path(path).read_text())
    data["nodes"] = tuple(data["nodes"])
    data["channels"] = tuple(data["channels"])
    data["hypotheses"] = tuple(tuple(F(str(p)) for p in row) for row in data["hypotheses"])
    data["prior"] = tuple(F(str(p)) for p in data["prior"])
    data["operations"] = tuple(Operation(**{**o, "channels": tuple(o.get("channels", []))})
                               for o in data["operations"])
    return PublicProblem(**data)
