"""TRAIN-only exact Bayes distillation data; no private truth or student model.

Sampling and labeling are separate. A failed solve cannot supply a partial label.
Old runtimes are unchanged. All persisted fractions use their canonical strings.
"""
import argparse
from collections import Counter
from dataclasses import replace
from fractions import Fraction as F
import fcntl
import hashlib
import json
from math import lcm
from pathlib import Path
import random
from time import monotonic, process_time

from .belief import PlannerState, ObservationHistory, outcomes, update
from .difficulty import hierarchical, tasks
from .evaluation import digest, source_hashes, save_new
from .generator import problem_json
from .problem import load_problem
from .protocols import enumerate_protocols, validate_protocol, PlanningLimit
from .registry import structural_key
from .teachers.exact_bayes import ExactBayes
from .utility import protocol_value
from .work import PlanningWorkBudget, WorkPolicy, WatchdogExpired
from .policies.beam_bayes import BeamBayes
from .policies.one_step_voi import OneStepVOI
from .policies.posterior_sampling import PosteriorSampling

SOURCES = ("exact", "random", "voi", "beam", "posterior_sampling")


def export_exclusions(scientific_manifest, dev_plan):
    """Read registry metadata only, never test problem files or private seeds."""
    data = json.loads(Path(scientific_manifest).read_text())
    registry = data["registry"]
    entries = list(registry["entries"].values())
    held = [x for x in entries if x["split"] != "train"]
    plan = json.loads(Path(dev_plan).read_text())
    dev = [p for p, _ in tasks(plan)]  # Already public, already evaluated DEV only.
    return dict(schema="fpl-teacher-exclusions-v1",
                source_hashes={"scientific_manifest": hashlib.sha256(Path(scientific_manifest).read_bytes()).hexdigest(),
                               "difficulty_dev_plan": hashlib.sha256(Path(dev_plan).read_bytes()).hexdigest()},
                clone_hashes=sorted({x["clone_hash"] for x in held} | {structural_key(p) for p in dev}),
                forbidden_seeds=sorted({x[k] for x in entries for k in ("structure_seed", "parameter_seed")} | set(plan["root_seeds"])),
                private_confirm="not read; no confirmation instances generated")


def build_tasks(plan, excluded):
    if plan.get("schema") != "fpl-teacher-plan-v1" or plan.get("split") != "train":
        raise ValueError("teacher builder accepts explicit TRAIN plans only")
    if digest(excluded) != plan["exclusions_hash"]:
        raise ValueError("exclusion commitment mismatch")
    forbidden = set(excluded["forbidden_seeds"])
    clones = set(excluded["clone_hashes"])
    jobs, rejected = [], []
    used = set()
    for index, spec in enumerate(plan["conditions"]):
        for replica in range(plan["roots_per_condition"]):
            slot = index * plan["roots_per_condition"] + replica
            for attempt in range(plan["max_structure_attempts"]):
                seed = plan["root_start"] + slot * plan["max_structure_attempts"] + attempt
                if seed in forbidden or seed in used:
                    raise ValueError("reserved or repeated TRAIN root seed")
                p = hierarchical(root_seed=seed, **spec["parameters"])
                p = replace(p, split_id="train", instance_id=f"teacher-{seed}-{spec['id']}",
                            family_id=f"teacher-train-root-{seed}")
                key = structural_key(p)
                if key in clones:
                    rejected.append(dict(slot=slot, root_seed=seed, clone_hash=key, reason="reserved structural clone"))
                    continue
                jobs.append((p, dict(root_seed=seed, condition=spec["id"], clone_hash=key,
                                     parameters=spec["parameters"], split="train")))
                used.add(seed)
                break
            else:
                # Keep impossible strata visible. Never alter topology to force admission.
                rejected.append(dict(slot=slot, condition=spec["id"], reason="stratum exhausted", unresolved=True))
    return jobs, rejected


def solver(p, conf):
    obj = ExactBayes(p, max_states=conf["max_states"], max_seconds=conf["watchdog_seconds"],
                     max_protocol_nodes=conf["max_protocol_nodes"], max_outcomes=conf["max_outcomes"])
    obj.search_budget = PlanningWorkBudget(max_expansions=conf["max_expansions"],
                                          max_model_calls=conf["max_model_calls"],
                                          watchdog_seconds=conf["watchdog_seconds"])
    return obj


def exact_label(p, state, conf):
    if state.resource != p.capacity:
        raise ValueError("teacher decision must be at reset")
    obj = solver(p, conf)
    wall, cpu = monotonic(), process_time()
    try:
        v = obj.value(state.remaining, state.posterior)
        candidates = [dict(operations=None, q="0", immediate="0", successors=[])]
        for route in obj.route_cache[state.remaining]:
            immediate = protocol_value(p, state.posterior, route, obj.search_budget)
            q, children = immediate, []
            for feedback, mass, belief in outcomes(p, state.posterior, route.channels,
                                                   conf["max_outcomes"], obj.search_budget):
                continuation = obj.value(state.remaining-route.duration, belief)
                q += mass * continuation
                children.append(dict(feedback=feedback, mass=str(mass), posterior=list(map(str, belief)),
                                     remaining=state.remaining-route.duration, value=str(continuation)))
            candidates.append(dict(operations=list(route.operations), duration=route.duration,
                                   channels=list(route.channels), q=str(q), immediate=str(immediate), successors=children))
        best = max(F(x["q"]) for x in candidates)
        if best != v:
            raise AssertionError("Bellman V/Q disagreement")
        action = obj.actions[(state.remaining, tuple(state.posterior))]
        result = dict(status="exact", value=str(v), candidates=candidates,
                      optimal_protocol=None if action is None else list(action.operations),
                      optimal_set=[x["operations"] for x in candidates if F(x["q"]) == v],
                      legal_first_operations=sorted({r.operations[0] for r in obj.route_cache[state.remaining]}))
    except (PlanningLimit, WatchdogExpired) as exc:
        result = dict(status="unresolved", reason=str(exc))
    result.update(solver_states=obj.states, likelihood_branches=obj.likelihood_branches,
                  work=obj.search_budget.snapshot(), wall_seconds=monotonic()-wall,
                  cpu_seconds=process_time()-cpu, limits=conf)
    return result


def state_key(state):
    return digest(dict(posterior=list(map(str, state.posterior)), remaining=state.remaining,
                       resource=state.resource))


def draw_index(masses, rng):
    """Exact rational prior-predictive sampling (no float rounding, no truth)."""
    denominator = lcm(*(m.denominator for m in masses))
    weights = [int(m * denominator) for m in masses]
    target = rng.randrange(sum(weights))
    for i, weight in enumerate(weights):
        target -= weight
        if target < 0:
            return i
    raise AssertionError("invalid probability mass")


def advance(p, state, route, feedback):
    return PlannerState(update(p, state.posterior, feedback), state.remaining-route.duration,
                        p.capacity, ObservationHistory(state.history.released+tuple(feedback)))


def collect_states(p, seed, plan):
    """One rollout per source, plus ALL one-step branches of each chosen route.

    Branches not followed still become candidate label states. Hash-priority
    sampling is independent of label values/status. Duplicate sufficient states
    merge only for the history-independent Bayes teacher, retaining provenance.
    """
    collected, logs = {}, []
    limits = plan["sampling_work"]
    for source in SOURCES:
        rng = random.Random(f"teacher-history-v1:{seed}:{source}")
        state = PlannerState(p.prior, p.budget, p.capacity)
        path, pool = [], {}
        account = PlanningWorkBudget(**limits)
        policy = None
        if source in ("voi", "beam", "posterior_sampling"):
            underlying = (OneStepVOI(p, width=8) if source == "voi" else
                          BeamBayes(p, width=8, depth=2) if source == "beam" else
                          PosteriorSampling(p, seed=seed, width=8))
            policy = WorkPolicy(p, underlying,
                                {k: limits[k] for k in ("max_expansions", "max_model_calls")},
                                {k: limits[k] for k in ("max_expansions", "max_model_calls")},
                                watchdog_seconds=limits["watchdog_seconds"])
        source_solves = []

        def add(s, history, kind):
            key = state_key(s)
            pool.setdefault(key, (s, []))[1].append(dict(source=source, kind=kind, history=history))

        add(state, [], "root")
        start, cpu = monotonic(), process_time()
        status, reason = "complete", None
        try:
            for _ in range(plan["max_rollout_batches"]):
                if state.remaining == 0:
                    break
                if source == "exact":
                    obj = solver(p, plan["solver"])
                    try:
                        obj.value(state.remaining, state.posterior)
                    finally:
                        source_solves.append(dict(states=obj.states, work=obj.search_budget.snapshot()))
                    route = obj.actions[(state.remaining, tuple(state.posterior))]
                elif source == "random":
                    routes = enumerate_protocols(p, state.remaining, plan["solver"]["max_protocol_nodes"], account)
                    route = rng.choice(routes) if routes else None
                else:
                    route = policy.select(state)
                if route is None:
                    break
                route = validate_protocol(p, route.operations, state.remaining)
                branches = list(outcomes(p, state.posterior, route.channels,
                                         plan["solver"]["max_outcomes"], account))
                for feedback, _, _ in branches:
                    next_path = path + [dict(operations=list(route.operations), feedback=list(feedback))]
                    add(advance(p, state, route, feedback), next_path, "reachable_feedback")
                choice = draw_index([b[1] for b in branches], rng)
                feedback = branches[choice][0]
                path = path + [dict(operations=list(route.operations), feedback=list(feedback))]
                state = advance(p, state, route, feedback)
                add(state, path, "rollout")
        except (PlanningLimit, WatchdogExpired) as exc:
            status, reason = "truncated", str(exc)
        root = state_key(PlannerState(p.prior, p.budget, p.capacity))
        ordered = sorted(pool, key=lambda k: (k != root, digest([source, seed, k])))
        chosen = ordered[:plan["states_per_source"]]
        for key in chosen:
            s, provenance = pool[key]
            collected.setdefault(key, (s, []))[1].extend(provenance)
        logs.append(dict(source=source, status=status, reason=reason, available_states=len(pool),
                         chosen_states=len(chosen), sampling_work=account.snapshot(),
                         policy_work=None if policy is None else dict(expansions=policy.expansions, model_calls=policy.model_calls),
                         teacher_solves=source_solves, wall_seconds=monotonic()-start, cpu_seconds=process_time()-cpu))
    return collected, logs


def audit_label(p, record):
    """Arithmetic/likelihood replay, NOT an independent optimality proof."""
    state = record["state"]
    belief = tuple(map(F, state["posterior"]))
    remaining = state["remaining"]
    if "id" in record and record["id"] != state_key(PlannerState(belief, remaining, state["resource"])):
        raise ValueError("state identity mismatch")
    for origin in record["provenance"]:
        replay = PlannerState(p.prior, p.budget, p.capacity)
        for step in origin["history"]:
            route = validate_protocol(p, tuple(step["operations"]), replay.remaining)
            feedback = tuple((c, b) for c, b in step["feedback"])
            if tuple(c for c, _ in feedback) != route.channels:
                raise ValueError("history feedback does not match route")
            replay = advance(p, replay, route, feedback)
        if replay.posterior != belief or replay.remaining != remaining or state["resource"] != p.capacity:
            raise ValueError("unreachable teacher state")
    label = record["label"]
    if label["status"] != "exact":
        if any(k in label for k in ("value", "candidates", "optimal_set")):
            raise ValueError("partial label exposed")
        return
    expected_routes = {r.operations for r in enumerate_protocols(p, remaining, label["limits"]["max_protocol_nodes"])}
    candidates = label["candidates"]
    actual = [None if c["operations"] is None else tuple(c["operations"]) for c in candidates]
    if len(set(actual)) != len(actual) or set(actual) != expected_routes | {None}:
        raise ValueError("candidate catalogue incomplete")
    for c in candidates:
        if c["operations"] is None:
            if F(c["q"]) != 0 or c["successors"]:
                raise ValueError("invalid stop value")
            continue
        route = validate_protocol(p, tuple(c["operations"]), remaining)
        immediate = protocol_value(p, belief, route)
        branches = list(outcomes(p, belief, route.channels, label["limits"]["max_outcomes"]))
        if len(branches) != len(c["successors"]):
            raise ValueError("missing outcome branch")
        q = immediate
        for (feedback, mass, posterior), child in zip(branches, c["successors"]):
            if (tuple(map(tuple, child["feedback"])) != feedback or F(child["mass"]) != mass
                or tuple(map(F, child["posterior"])) != posterior or child["remaining"] != remaining-route.duration):
                raise ValueError("likelihood mismatch")
            q += mass*F(child["value"])
        if q != F(c["q"]) or immediate != F(c["immediate"]):
            raise ValueError("Q arithmetic mismatch")
    v = max(F(c["q"]) for c in candidates)
    if F(label["value"]) != v or label["optimal_set"] != [c["operations"] for c in candidates if F(c["q"]) == v]:
        raise ValueError("optimal set mismatch")
    if label["optimal_protocol"] not in label["optimal_set"]:
        raise ValueError("selected action is not optimal")


def make_shard(p, meta, plan):
    start, cpu = monotonic(), process_time()
    states, sampling = collect_states(p, meta["root_seed"], plan)
    records = []
    for key, (state, provenance) in sorted(states.items()):
        label = exact_label(p, state, plan["solver"])
        record = dict(id=key, state=dict(posterior=list(map(str, state.posterior)), remaining=state.remaining,
                                       resource=state.resource), provenance=provenance, label=label)
        audit_label(p, record)
        records.append(record)
    return dict(schema="fpl-teacher-shard-v1", instance_hash=p.instance_hash, metadata=meta,
                problem=json.loads(problem_json(p)), sampling=sampling, records=records,
                wall_seconds=monotonic()-start, cpu_seconds=process_time()-cpu)


def coverage(shards, scheduled):
    labels = Counter()
    sources = Counter()
    channels, hypotheses, horizons, beliefs, clones = set(), set(), set(), set(), set()
    nonprior = informative = ties = 0
    work = Counter()
    by_condition = {}
    for shard in shards:
        p = shard["problem"]
        channels.add(len(p["channels"]))
        hypotheses.add(len(p["hypotheses"]))
        clones.add(shard["metadata"]["clone_hash"])
        condition = by_condition.setdefault(shard["metadata"]["condition"], Counter())
        condition["roots"] += 1
        for source in shard["sampling"]:
            for account in [source["sampling_work"], source["policy_work"]] + [s["work"] for s in source["teacher_solves"]]:
                if account:
                    for key in ("expansions", "model_calls"):
                        work["sampling_"+key] += account[key]
        for row in shard["records"]:
            label = row["label"]
            labels[label["status"]] += 1
            condition[label["status"]] += 1
            for key in ("expansions", "model_calls"):
                work["label_"+key] += label["work"][key]
            if label["status"] != "exact":
                continue
            horizons.add(row["state"]["remaining"])
            beliefs.add(tuple(row["state"]["posterior"]))
            nonprior += row["state"]["posterior"] != p["prior"]
            condition["exact_nonprior"] += row["state"]["posterior"] != p["prior"]
            sources.update(set(x["source"] for x in row["provenance"]))
            ties += len(label["optimal_set"]) > 1
            informative += any(c.get("channels") for c in label["candidates"] if c["operations"] in label["optimal_set"])
    return dict(status="complete" if len(shards) == scheduled else "partial",
                completed_roots=len(shards), scheduled_roots=scheduled, labels=dict(labels),
                exact_source_membership=dict(sources), exact_nonprior_states=nonprior,
                exact_unique_beliefs=len(beliefs), exact_remaining_horizons=sorted(horizons),
                channels=sorted(channels), hypothesis_counts=sorted(hypotheses), structural_groups=len(clones),
                exact_states_with_optimal_measurement=informative, exact_states_with_ties=ties,
                by_condition={k:dict(v) for k,v in by_condition.items()}, recorded_work=dict(work),
                total_cpu_seconds=sum(s["cpu_seconds"] for s in shards),
                total_wall_seconds=sum(s["wall_seconds"] for s in shards),
                warning="Descriptive TRAIN coverage, not generalization, population evidence, or neural advantage.")


def run(plan_path, exclusions_path, output, resume=False, max_new=1):
    if max_new < 0:
        raise ValueError("nonnegative checkpoint count required")
    plan = json.loads(Path(plan_path).read_text())
    exclusions = json.loads(Path(exclusions_path).read_text())
    jobs, rejected = build_tasks(plan, exclusions)
    out = Path(output)
    seal = dict(schema="fpl-teacher-seal-v1", plan=plan, exclusions=exclusions, sources=source_hashes(),
                jobs=[dict(instance_hash=p.instance_hash, metadata=m) for p, m in jobs], rejections=rejected)
    if out.exists() and not resume:
        raise FileExistsError("output exists; use explicit --resume")
    out.mkdir(parents=True, exist_ok=True)
    with (out/"writer.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        seal_path = out/"seal.json"
        if seal_path.exists():
            if json.loads(seal_path.read_text()) != seal:
                raise ValueError("source/plan/exclusion seal drift")
        elif resume:
            raise ValueError("missing seal; refuse ambiguous resume")
        else:
            save_new(seal_path, seal)
        completed, count = [], 0
        expected = {p.instance_hash for p, _ in jobs}
        if any(f.name.removesuffix(".shard.json") not in expected for f in out.glob("*.shard.json")):
            raise ValueError("foreign shard")
        for p, meta in jobs:
            path = out/f"{p.instance_hash}.shard.json"
            if path.exists():
                envelope = json.loads(path.read_text())
                shard = envelope["payload"]
                if (digest(shard) != envelope["hash"] or shard["instance_hash"] != p.instance_hash
                    or shard["metadata"] != meta or shard["problem"] != json.loads(problem_json(p))):
                    raise ValueError("corrupt or mismatched shard")
            elif count < max_new:
                shard = make_shard(p, meta, plan)
                save_new(path, dict(hash=digest(shard), payload=shard))
                count += 1
                print(json.dumps(dict(checkpoint=p.instance_id, records=len(shard["records"]))), flush=True)
            else:
                continue
            completed.append(shard)
        summary = coverage(completed, len(jobs))
        summary["rejected_candidates"] = len(rejected)
        summary["exhausted_strata"] = sum(bool(x.get("unresolved")) for x in rejected)
        save_new(out/"coverage.json", summary)
        return summary


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--exclusions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-new", type=int, default=1, help="default: first-checkpoint handoff only")
    args = parser.parse_args()
    print(json.dumps(run(args.plan, args.exclusions, args.output, args.resume, args.max_new), indent=2))


if __name__ == "__main__":
    main()
