"""Generate a manifest and grouped public tasks from a versioned plan."""
import argparse
import hashlib
import json
from pathlib import Path
from .generator import generate, problem_json
from .registry import Registry


def build(plan):
    registry, problems = Registry(), []
    for family in plan["families"]:
        for seed in family["seeds"]:
            for capacity in family["capacities"]:
                for budget in family["budgets"]:
                    p = generate(**family["generator"],seed=seed,capacity=capacity,
                                 budget=budget,split=family["split"])
                    registry.register(p)
                    problems.append(p)
    if len({p.instance_hash for p in problems}) != len(problems):
        raise ValueError("duplicate instance")
    return problems,registry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    problems,registry = build(json.loads(args.plan.read_text()))
    args.output.mkdir(parents=True,exist_ok=False)
    for p in problems:
        (args.output/(p.instance_hash+".json")).write_text(problem_json(p)+"\n")
    manifest = registry.to_dict()
    manifest["plan_sha256"] = hashlib.sha256(args.plan.read_bytes()).hexdigest()
    manifest["selection"] = "unfiltered public tasks; no benefit or oracle-equality screening"
    (args.output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(f"Generated {len(problems)} public tasks with validated split registry")


if __name__ == "__main__":
    main()
