"""Frozen public task distribution, separate structure and parameter seeds.

No oracle, policy, reward trajectory, or risk is used to select instances.
Only cross-split structural duplicates are rejected. IID is therefore
same-generator / clone-disjoint, not unconditional iid graph sampling.
"""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random
from .generator import generate,problem_json
from .registry import RegistryV2,structural_key


def random_problem(*,structure_seed,parameter_seed,channels=3,hypotheses=2,
                   template="directed",edge_probability=0.5,duration=1,energy=1,
                   split="dev",capacity=None,budget=8,sharing="independent"):
    if template not in ("directed","dag") or not 0<=edge_probability<=1:
        raise ValueError("invalid random graph distribution")
    p = generate(channels,hypotheses,"complete",parameter_seed,capacity,budget,
                 duration,energy,sharing,split)
    rng = random.Random(structure_seed)
    ops = []
    for op in p.operations:
        if op.source in p.channels and op.target in p.channels:
            allowed = template!="dag" or p.channels.index(op.source)<p.channels.index(op.target)
            if not allowed or rng.random()>=edge_probability:
                continue
        ops.append(op)
    return replace(p,instance_id=f"{template}-m{channels}-s{structure_seed}-p{parameter_seed}",
                   family_id=f"random-{template}",operations=tuple(ops))


def build(plan):
    registry,tasks,rejections = RegistryV2(),[],[]
    for group in plan["groups"]:
        for root in range(group["roots"]):
            ss = group["structure_seed"]+root*1000
            ps = group["parameter_seed"]+root
            for attempt in range(1000):
                seed = ss+attempt
                kwargs = dict(channels=group["channels"],hypotheses=group["hypotheses"],
                              template=group["template"],edge_probability=group.get("edge_probability",0.5),
                              duration=group.get("duration",1),energy=group.get("energy",1),
                              split=group["split"],sharing=group.get("sharing","independent"))
                base = random_problem(structure_seed=seed,parameter_seed=ps,**kwargs)
                key = structural_key(base)
                # Separate independent root groups even within the same split.
                if key not in registry.clone_splits:
                    break
                rejections.append(dict(group=group["name"],root=root,structure_seed=seed,reason="clone"))
            else:
                raise ValueError("clone-disjoint generator exhausted; widen distribution")
            gid = f"{group['name']}-{root}"
            for cap in group["capacities"]:
                for horizon in group["budgets"]:
                    p = replace(base,capacity=cap,budget=horizon,instance_id=f"{base.instance_id}-B{cap}-H{horizon}")
                    registry.register(p,distribution_id=group["distribution_id"],template_id=group["template"],
                                      parameter_seed=ps,structure_seed=seed,ood_axis=group.get("ood_axis"),group_id=gid)
                    tasks.append(p)
    return tasks,registry,rejections


def freeze(plan_path,output):
    plan_path,output = Path(plan_path),Path(output)
    plan = json.loads(plan_path.read_text())
    tasks,registry,rejections = build(plan)
    output.mkdir(parents=True,exist_ok=False)
    for p in tasks:
        (output/f"{p.instance_hash}.json").write_text(problem_json(p)+"\n")
    manifest = dict(schema="fpl-scientific-dataset-v1",plan=plan,
                    plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),
                    registry=registry.to_dict(),clone_rejections=rejections,
                    sampling="same-generator clone-disjoint IID; no outcome-based filtering")
    (output/"manifest.json").write_text(json.dumps(manifest,sort_keys=True,indent=2)+"\n")
    return manifest


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan",required=True)
    parser.add_argument("--output",required=True)
    args = parser.parse_args()
    result = freeze(args.plan,args.output)
    print(json.dumps(dict(tasks=len(result["registry"]["entries"]),rejections=len(result["clone_rejections"]))))
