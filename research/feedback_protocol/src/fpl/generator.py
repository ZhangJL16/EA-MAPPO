"""Public finite Bernoulli task families; no hidden truth used in generation."""
from dataclasses import asdict
from fractions import Fraction as F
import json
import random
from .problem import Operation, PublicProblem, UtilitySpec


def generate(channels=3, hypotheses=3, topology="complete", seed=0, capacity=None,
             budget=8, duration=1, energy=1, sharing="independent", split="debug-only"):
    if channels < 1 or hypotheses < 2 or topology not in ("complete", "chain", "star", "ring"):
        raise ValueError("unsupported family")
    if sharing not in ("independent", "paired") or duration < 1 or energy < 1:
        raise ValueError("invalid sharing/duration/energy")
    rng = random.Random(seed)
    cs = tuple(f"c{i}" for i in range(channels))
    group = tuple(i if sharing == "independent" else i//2 for i in range(channels))
    means = []
    for _ in range(hypotheses):
        values = [F(rng.randint(1,9),10) for _ in range(max(group)+1)]
        means.append(tuple(values[g] for g in group))
    ops = [Operation("prepare", "q", "ready", duration, energy, utility=UtilitySpec()),
           Operation("empty", "ready", "q", duration, energy, utility=UtilitySpec())]
    for i,c in enumerate(cs):
        ops.extend([Operation(f"probe{i}","ready",c,duration,energy,(c,),UtilitySpec()),
                    Operation(f"return{i}",c,"q",duration,energy,utility=UtilitySpec())])
        for j,d in enumerate(cs):
            if j != i and (topology == "complete" or topology == "chain" and j == i+1
                           or topology == "ring" and j == (i+1)%channels):
                ops.append(Operation(f"probe{i}_{j}",c,d,duration,energy,(d,),UtilitySpec()))
    # Task actions at reset have no sensing feedback. Their utility is assessed
    # by the evaluator only, so no implicit true-label channel is introduced.
    for h in range(hypotheses):
        ops.append(Operation(f"task{h}","q","q",duration,energy,utility=UtilitySpec(
            by_hypothesis=tuple(F(int(i == h)) for i in range(hypotheses)))))
    return PublicProblem(f"{topology}-{channels}-{seed}", f"generated-template-{topology}",
                         split, ("q","ready")+cs,"q",cs,tuple(means),
                         (F(1,hypotheses),)*hypotheses,tuple(ops),
                         (channels+2)*energy if capacity is None else capacity,budget,channels,
                         objective="cumulative_task_utility")


def problem_json(problem):
    return json.dumps(asdict(problem), sort_keys=True, indent=2, default=str)
