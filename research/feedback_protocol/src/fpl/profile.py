"""Bounded CPU profiling; completed rows append durably, resume checks source seal."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter
import tracemalloc
from .generator import generate
from .protocols import enumerate_protocols, PlanningLimit, validate_protocol
from .teachers.exact_bayes import ExactBayes
from .policies.beam_bayes import BeamBayes
from .belief import PlannerState


def measure(call):
    tracemalloc.start()
    start = perf_counter()
    try:
        result = {"status":"completed","result":call()}
    except PlanningLimit as exc:
        result = {"status":"limit_reached","reason":str(exc)}
    finally:
        seconds = perf_counter()-start
        _,peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    return {**result,"seconds":seconds,"python_peak_bytes":peak}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--resume",action="store_true")
    args = parser.parse_args()
    raw = args.config.read_bytes()
    config = json.loads(raw)
    root = Path(__file__).parent
    source = {str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(root.rglob("*.py"))}
    seal = hashlib.sha256(json.dumps([hashlib.sha256(raw).hexdigest(),source],sort_keys=True).encode()).hexdigest()
    completed = set()
    if args.output.exists():
        if not args.resume:
            parser.error("output exists; use explicit --resume")
        for line in args.output.read_text().splitlines():
            row = json.loads(line)  # Incomplete line: fail, never silently discard data.
            if row["seal"] != seal:
                parser.error("source/config changed; refusing resume")
            completed.add((row["topology"],row["channels"]))
    with args.output.open("a" if args.resume else "x") as handle:
        for topology in config["topologies"]:
            for m in config["channels"]:
                if (topology,m) in completed:
                    continue
                p = generate(m,config["hypotheses"],topology,config["seed"],budget=m+2)
                enumeration = measure(lambda: len(enumerate_protocols(p,p.budget,config["enumeration_nodes"])))
                exact = ExactBayes(p,max_states=config["bayes_states"],max_seconds=config["bayes_seconds"],
                                   max_protocol_nodes=config["enumeration_nodes"])
                bayes = measure(lambda: str(exact.value(p.budget,p.prior)))
                bayes.update(states=exact.states,likelihood_branches=exact.likelihood_branches)
                beam = BeamBayes(p,width=config["beam_width"],depth=config["beam_depth"],
                                 max_expansions=config["beam_expansions"],max_seconds=config["beam_seconds"])
                def choose():
                    route = beam.select(PlannerState(p.prior,p.budget,p.capacity))
                    if route is not None:
                        validate_protocol(p,route.operations,p.budget)
                    return asdict(route) if route else None
                approximate = measure(choose)
                approximate["search"] = beam.stats
                row = dict(schema="fpl-profile-row-v1",seal=seal,source_sha256=source,
                           config=config,topology=topology,channels=m,instance_hash=p.instance_hash,
                           enumeration=enumeration,bayes=bayes,beam=approximate)
                handle.write(json.dumps(row,default=str)+"\n")
                handle.flush()
                os.fsync(handle.fileno())
                print(json.dumps({"topology":topology,"channels":m,"enumeration":enumeration["status"],
                                  "bayes":bayes["status"],"beam":approximate["status"]}),flush=True)


if __name__ == "__main__":
    main()
