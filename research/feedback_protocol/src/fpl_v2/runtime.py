"""Sealed admission, then only the authorized 8-group/32-variant sizing run."""
import argparse
import fcntl
from hashlib import sha256
import json
from pathlib import Path
from time import monotonic, process_time
from fpl.problem import load_problem
from fpl.belief import PlannerState
from fractions import Fraction as F
from fpl.teacher_data import exact_label
from fpl.evaluation import save_new, digest, source_hashes
from .generator import admit
from .collection import collect


def closure():
    return {**source_hashes(), **{f"fpl_v2/{p.name}": sha256(p.read_bytes()).hexdigest()
                               for p in Path(__file__).parent.glob("*.py")}}


def run(design, admission_dir, output, max_new=None, resume=False):
    base, out = Path(admission_dir), Path(output)
    manifest = json.loads((base/"admission.json").read_text())
    if manifest["design_hash"] != digest(design):
        raise ValueError("design/admission mismatch")
    if len(manifest["groups"]) != 32 or len({g["clone_hash"] for g in manifest["groups"]}) != 32:
        raise ValueError("all 32 structural identities must be admitted before labeling")
    for job in manifest["jobs"]:
        if load_problem(base/job["problem_file"]).instance_hash != job["instance_hash"]:
            raise ValueError("admitted public problem changed")
    jobs = [j for j in manifest["jobs"] if j["pilot"]]
    if len(jobs) != 32:
        raise ValueError("pilot schedule must contain exactly 32 variants")
    seal = dict(schema="teacher-v2-pilot-seal-v1", admission_hash=digest(manifest),
                source_hashes=closure(), design_hash=digest(design), scheduled=jobs)
    if out.exists():
        if not resume or json.loads((out/"seal.json").read_text()) != seal:
            raise ValueError("resume seal mismatch")
    else:
        out.mkdir(parents=True)
        save_new(out/"seal.json", seal)
    with (out/"writer.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        count = 0
        for job in jobs:
            stem = Path(job["problem_file"]).stem
            target, checkpoint = out/(stem+".shard.json"), out/(stem+".states.json")
            journal = out/(stem+".started.json")
            if target.exists():
                packed = json.loads(target.read_text())
                shard = packed["payload"]
                if packed["hash"] != digest(shard):
                    raise ValueError("completed shard content hash mismatch")
                if shard["metadata"] != job or shard["seal_hash"] != digest(seal):
                    raise ValueError("completed shard identity mismatch")
                continue
            if max_new is not None and count >= max_new:
                break
            if journal.exists():
                raise RuntimeError(f"interrupted variant retained, explicit recovery required: {stem}")
            save_new(journal, dict(status="started", metadata=job, retry=False))
            p = load_problem(base/job["problem_file"])
            wall, cpu = monotonic(), process_time()
            selected = collect(p, job["root_seed"], design)
            selected.update(metadata=job, seal_hash=digest(seal))
            save_new(checkpoint, selected)  # State IDs are sealed BEFORE any label.
            records = []
            for row in selected["records"]:
                s = row["state"]
                state = PlannerState(tuple(map(F, s["posterior"])), s["remaining"], s["resource"])
                records.append(dict(row, label=exact_label(p, state, design["work_limits"]["exact_label"])))
            shard = dict(schema="teacher-v2-shard-v1", metadata=job, seal_hash=digest(seal),
                         selection_hash=digest(selected), problem=json.loads((base/job["problem_file"]).read_text()),
                         sampling=selected["sampling"], pool_states=selected["pool_states"], records=records,
                         cpu_seconds=process_time()-cpu, wall_seconds=monotonic()-wall)
            save_new(target, dict(hash=digest(shard), payload=shard))
            count += 1
            print(json.dumps(dict(variant=stem, states=len(records),
                                  exact=sum(r["label"]["status"] == "exact" for r in records),
                                  wall_seconds=shard["wall_seconds"])), flush=True)
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("admit", "run"))
    parser.add_argument("--design", required=True)
    parser.add_argument("--admission", required=True)
    parser.add_argument("--exclusions")
    parser.add_argument("--v1-admission")
    parser.add_argument("--output")
    parser.add_argument("--max-new", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    design = json.loads(Path(args.design).read_text())
    if args.command == "admit":
        result = admit(design, json.loads(Path(args.exclusions).read_text()),
                       json.loads(Path(args.v1_admission).read_text()), args.admission)
        print(json.dumps(dict(groups=len(result["groups"]), variants=len(result["jobs"]), rejected=len(result["rejected"]))))
    else:
        run(design, args.admission, args.output, args.max_new, args.resume)


if __name__ == "__main__":
    main()
