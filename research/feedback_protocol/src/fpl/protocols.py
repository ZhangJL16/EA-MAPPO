"""Bounded enumeration reference; deliberately not a scalable decoder."""
from dataclasses import dataclass
from collections import Counter
from .problem import PublicProblem


class PlanningLimit(RuntimeError):
    """No exact result is returned after a computation limit is reached."""


@dataclass(frozen=True)
class Protocol:
    operations: tuple[str, ...]
    duration: int
    channels: tuple[str, ...]


def validate_protocol(p: PublicProblem, names: tuple[str, ...], remaining: int) -> Protocol:
    if not names:
        raise ValueError("empty protocol; use stop instead")
    node, battery, elapsed = p.reset, p.capacity, 0
    channels = []
    ops = {op.name: op for op in p.operations}
    for index, name in enumerate(names):
        if name not in ops:
            raise ValueError("unknown operation")
        op = ops[name]
        if op.source != node or op.energy > battery:
            raise ValueError("disconnected or resource-infeasible protocol")
        battery -= op.energy  # Pre-reload non-depletion is mandatory.
        elapsed += op.duration
        node = op.target
        channels.extend(op.channels)
        counts = Counter(channels)
        if elapsed > remaining or len(channels) > p.max_measurements or any(
            n > p.per_channel_limit for n in counts.values()
        ):
            raise ValueError("budget or measurement restriction")
        if node == p.reset:
            if index != len(names)-1:
                raise ValueError("protocol must end at its first reset return")
            battery = p.capacity
    if node != p.reset:
        raise ValueError("protocol must finish at reset")
    return Protocol(tuple(names), elapsed, tuple(channels))


def enumerate_protocols(p: PublicProblem, remaining: int, max_nodes: int = 10000, budget=None) -> tuple[Protocol, ...]:
    if type(remaining) is not int or not 0 <= remaining <= p.budget:
        raise ValueError("invalid remaining budget")
    stack = [(p.reset, p.capacity, 0, (), ())]
    result, expanded = [], 0
    while stack:
        node, battery, elapsed, names, channels = stack.pop()
        expanded += 1
        if expanded > max_nodes:
            raise PlanningLimit("protocol enumeration node limit")
        for op in p.operations:
            if op.source != node:
                continue
            if budget is not None:
                budget.consume(expansions=1)
            next_channels = channels + op.channels
            if (op.source != node or op.energy > battery or elapsed+op.duration > remaining
                or len(next_channels) > p.max_measurements
                or any(n > p.per_channel_limit for n in Counter(next_channels).values())):
                continue
            next_names = names + (op.name,)
            if op.target == p.reset:
                result.append(Protocol(next_names, elapsed+op.duration, next_channels))
            else:
                stack.append((op.target, battery-op.energy, elapsed+op.duration, next_names, next_channels))
    return tuple(sorted(result, key=lambda route: route.operations))
