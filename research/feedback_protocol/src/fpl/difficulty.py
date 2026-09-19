"""Predefined hierarchical diagnosis axes; never performance-filtered."""
from dataclasses import replace
from fractions import Fraction as F
import random
from .problem import Operation,PublicProblem,UtilitySpec


def hierarchical(*,root_seed=2021,depth=2,redundancy=0,quality="4/5",budget=12,capacity=4,
                 prepare=1,probe=1,transit=1,cleanup=1,task=1,energy=1,topology="dense"):
    if depth<1 or redundancy<0 or topology not in ("dense","sparse"):
        raise ValueError("invalid hierarchy")
    q = F(quality)
    if not F(1,2)<q<=1 or min(prepare,probe,transit,cleanup,task,energy)<=0:
        raise ValueError("invalid quality/cost")
    k = 2**depth
    # Each internal binary-tree channel distinguishes its two descendants;
    # outside that subtree it returns a fair, uninformative coin.
    descriptors = [(level,prefix) for level in range(depth) for prefix in range(2**level)]
    descriptors += [(depth-1,j%(2**(depth-1))) for j in range(redundancy)]
    rng = random.Random(root_seed)
    order = list(range(k))
    rng.shuffle(order)
    polarity = [rng.randrange(2) for _ in descriptors]
    channels = tuple(f"c{i}" for i in range(len(descriptors)))
    rows = []
    for h in order:
        row = []
        for j,(level,prefix) in enumerate(descriptors):
            if h//(2**(depth-level))!=prefix:
                row.append(F(1,2))
            else:
                bit = (h//(2**(depth-level-1)))%2 ^ polarity[j]
                row.append(q if bit else 1-q)
        rows.append(tuple(row))
    # Root-dependent nonuniform prior and physical travel edges, drawn before
    # any policy execution; not just cosmetic names or reward reskins.
    weights = [rng.randint(2,5) for _ in range(k)]
    prior = tuple(F(w,sum(weights)) for w in weights)
    ops = [Operation("prepare","q","ready",prepare,energy,utility=UtilitySpec()),
           Operation("empty","ready","q",cleanup,energy,utility=UtilitySpec())]
    for i,c in enumerate(channels):
        ops += [Operation(f"probe{i}","ready",c,probe,energy,(c,),UtilitySpec()),
                Operation(f"return{i}",c,"q",cleanup,energy,utility=UtilitySpec())]
        for j,d in enumerate(channels):
            if i==j:
                continue
            allowed = topology=="dense" or rng.random()<0.25
            if allowed:
                # A joint move-and-measure operation pays both costs.
                ops.append(Operation(f"transit{i}_{j}",c,d,transit+probe,2*energy,(d,),UtilitySpec()))
    for h in range(k):
        ops.append(Operation(f"task{h}","q","q",task,energy,utility=UtilitySpec(
            by_hypothesis=tuple(F(int(h==v)) for v in range(k)))))
    return PublicProblem(f"hierarchy-{root_seed}-d{depth}",f"hierarchy-root-{root_seed}","dev",
                         ("q","ready")+channels,"q",channels,tuple(rows),prior,tuple(ops),capacity,budget,
                         len(channels),objective="cumulative_task_utility")


def tasks(plan):
    for seed in plan["root_seeds"]:
        for spec in plan["conditions"]:
            kwargs = dict(plan["base"],**spec["overrides"],root_seed=seed)
            p = hierarchical(**kwargs)
            yield replace(p,instance_id=f"{p.instance_id}-{spec['id']}"),dict(root_seed=seed,
                group_id=f"hierarchy-{seed}",condition=spec["id"],axes=kwargs,split="dev")
