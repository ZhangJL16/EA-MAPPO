"""Freeze TRAIN inputs from public exclusion metadata before any label solve."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fpl.teacher_data import export_exclusions, build_tasks
from fpl.evaluation import digest, save_new


def default_plan(exclusions):
    specs = [
        ("binary-short", dict(depth=1, quality="3/5", budget=4, capacity=3)),
        ("binary-long", dict(depth=1, quality="9/10", budget=12, capacity=4)),
        ("binary-redundant", dict(depth=1, redundancy=1, quality="7/10", budget=8, capacity=5)),
        ("hierarchy-sparse", dict(depth=2, quality="4/5", budget=8, capacity=3, topology="sparse")),
        ("hierarchy-dense", dict(depth=2, quality="9/10", budget=12, capacity=5)),
        ("hierarchy-cost", dict(depth=2, quality="7/10", budget=10, capacity=4, prepare=2, cleanup=2)),
        ("hierarchy-four-channels", dict(depth=2, redundancy=1, quality="3/5", budget=6, capacity=4)),
        ("hierarchy-five-channels", dict(depth=2, redundancy=2, quality="4/5", budget=12, capacity=5)),
    ]
    return dict(schema="fpl-teacher-plan-v1", split="train", root_start=100001,
                roots_per_condition=3, max_structure_attempts=100, exclusions_hash=digest(exclusions),
                conditions=[dict(id=name, parameters=spec) for name, spec in specs],
                max_rollout_batches=4, states_per_source=6,
                sampling_work=dict(max_expansions=5000, max_model_calls=50000, watchdog_seconds=15),
                solver=dict(max_states=3000, max_protocol_nodes=10000, max_outcomes=64,
                            max_expansions=1000000, max_model_calls=3000000, watchdog_seconds=5),
                statement="No outcome filtering; unresolved labels and empty structural strata remain visible. No network or final evaluation.")


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--scientific-manifest", required=True)
    parser.add_argument("--dev-plan", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    excluded = export_exclusions(args.scientific_manifest, args.dev_plan)
    plan = default_plan(excluded)
    jobs, rejected = build_tasks(plan, excluded)
    save_new(out / "exclusions.json", excluded)
    save_new(out / "plan.json", plan)
    save_new(out / "admission.json", dict(jobs=[dict(instance_hash=p.instance_hash, **m) for p, m in jobs],
                                         rejections=rejected, planned_slots=24))
    print(json.dumps(dict(admitted=len(jobs), rejected=len(rejected), output=str(out)), indent=2))


if __name__ == "__main__":
    main()
