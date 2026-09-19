"""Metadata-only admission; no solver, environment or private truth imports."""
from dataclasses import replace, asdict
from fractions import Fraction as F
from hashlib import sha256
import json
from pathlib import Path
import random
from time import monotonic, process_time
from fpl.problem import PublicProblem, Operation, UtilitySpec
from fpl.registry import structural_key
from fpl.evaluation import digest, save_new

NAMESPACE = "fpl-teacher-v2-design-v1"
FAMILIES = ("adaptive_hierarchy", "random_resource_graph", "cost_horizon", "decoy_redundancy")


def rng_for(seed, stream):
    raw = json.dumps([NAMESPACE, seed, stream], separators=(",", ":")).encode()
    return random.Random(int.from_bytes(sha256(raw).digest(), "big"))


def as_json(p):
    return json.loads(json.dumps(asdict(p), default=str))


def from_json(data):
    d = dict(data)
    for key in ("nodes", "channels"):
        d[key] = tuple(d[key])
    d["prior"] = tuple(map(F, d["prior"]))
    d["hypotheses"] = tuple(tuple(map(F, row)) for row in d["hypotheses"])
    ops = []
    for raw in d["operations"]:
        o = dict(raw, channels=tuple(raw.get("channels", ())))
        if o.get("utility") is not None:
            o["utility"] = UtilitySpec(**{k: tuple(map(F, v)) for k, v in o["utility"].items()})
        ops.append(Operation(**o))
    d["operations"] = tuple(ops)
    return PublicProblem(**d)


def generate(seed, family, horizon=12, capacity=5):
    if family not in FAMILIES:
        raise ValueError("unknown family")
    top, ret, cost = (rng_for(seed, s) for s in ("topology", "return_structure", "cost"))
    quality = rng_for(seed, "quality").choice((F(4, 5), F(9, 10)))
    sensors = [("coarse", quality), ("left", quality), ("right", quality)]
    prob = F(1, 2)
    if family == "random_resource_graph":
        prob = top.choice((F(1, 4), F(1, 2), F(3, 4)))
        if top.choice((0, 1)):
            sensors.append((top.choice(("left", "right")), F(3, 5)))
    if family == "decoy_redundancy":
        for _ in range(top.choice((1, 2))):
            typ = top.choice(("duplicate", "weak", "strong"))
            sensors.append((top.choice(("left", "right")) if typ == "duplicate" else typ,
                            quality if typ == "duplicate" else F(3, 5) if typ == "weak" else F(9, 10)))
    channels = tuple(f"c{i}" for i in range(len(sensors)))
    permutation = list(range(4))
    rng_for(seed, "hypothesis_permutation").shuffle(permutation)
    pol = rng_for(seed, "channel_polarity")
    flips = [pol.randrange(2) for _ in channels]
    table = []
    for h in permutation:
        row = []
        for (typ, q), flip in zip(sensors, flips):
            if typ in ("coarse", "weak", "strong"):
                v = q if h < 2 else 1-q
            elif typ == "left":
                v = (q if h == 0 else 1-q) if h < 2 else F(1, 2)
            else:
                v = (q if h == 2 else 1-q) if h >= 2 else F(1, 2)
            row.append(1-v if flip else v)
        table.append(tuple(row))
    pr = rng_for(seed, "prior")
    weights = [pr.randint(2, 5) for _ in range(4)]
    prior = tuple(F(w, sum(weights)) for w in weights)
    prepare = cost.choice((1, 2)) if family == "cost_horizon" else 1
    cleanup = cost.choice((1, 2)) if family == "cost_horizon" else 1
    transit = 1 if family == "adaptive_hierarchy" else cost.choice((1, 2))
    probes = [cost.choice((1, 2)) if family == "decoy_redundancy" else 1 for _ in channels]
    for i, (typ, _) in enumerate(sensors):
        if typ == "strong":
            probes[i] = 2
    shared = [bool(ret.randrange(2)) for _ in channels]
    nodes = ("q", "ready") + channels + (("cleanup",) if any(shared) else ())
    zero = UtilitySpec()
    ops = [Operation("prepare", "q", "ready", prepare, 1, (), zero),
           Operation("empty_return", "ready", "q", cleanup, 1, (), zero)]
    for i, c in enumerate(channels):
        ops.append(Operation(f"probe_{c}", "ready", c, probes[i], 1, (c,), zero))
        ops.append(Operation(f"return_{c}", c, "cleanup" if shared[i] else "q",
                             1 if shared[i] else cleanup, 1, (), zero))
    if any(shared):
        ops.append(Operation("cleanup_return", "cleanup", "q", cleanup, 1, (), zero))
    for i, c in enumerate(channels):
        for j, d in enumerate(channels):
            if i != j and top.randrange(prob.denominator) < prob.numerator:
                ops.append(Operation(f"transit_{c}_{d}", c, d, transit+probes[j], 2, (d,), zero))
    for h in range(4):
        ops.append(Operation(f"task_{h}", "q", "q", 1, 1, (),
                             UtilitySpec(by_hypothesis=tuple(F(h == i) for i in range(4)))))
    return PublicProblem(f"v2-{seed}-h{horizon}-b{capacity}", f"v2-root-{seed}", "train", nodes,
                         "q", channels, tuple(table), prior, tuple(ops), capacity, horizon,
                         len(channels), objective="cumulative_task_utility")


def structural_check(p):
    # Explicit singleton routes from the prescribed graph. No optimization/label.
    by = {o.name: o for o in p.operations}
    durations = []
    for c in p.channels:
        route = [by["prepare"], by[f"probe_{c}"], by[f"return_{c}"]]
        if route[-1].target != p.reset:
            route.append(by["cleanup_return"])
        if sum(o.energy for o in route) > 5:
            return "probe not safely reachable at B5"
        durations.append(sum(o.duration for o in route))
    if sum(sorted(durations)[:2])+1 > 12:
        return "two distinct singleton batches plus task not feasible at H12"
    return None


def admit(design, exclusions, v1_admission, output):
    out = Path(output)
    if out.exists():
        raise FileExistsError(out)
    start, cpu = monotonic(), process_time()
    # V1 admission is a metadata file, never a public-test problem registry.
    v1jobs = v1_admission["jobs"]
    forbidden = set(exclusions["forbidden_seeds"]) | {j["root_seed"] for j in v1jobs}
    seen = set(exclusions["clone_hashes"]) | {j["clone_hash"] for j in v1jobs}
    groups, rejected = [], []
    cfg = design["admission"]
    for fi, family in enumerate(FAMILIES):
        accepted = 0
        for offset in range(cfg["candidate_limit_per_family"]):
            seed = cfg["candidate_root_start"] + fi*cfg["candidate_limit_per_family"] + offset
            if seed in forbidden:
                raise ValueError("reserved seed")
            p = generate(seed, family)
            clone = structural_key(p)
            reason = structural_check(p) or ("excluded or duplicate structural clone" if clone in seen else None)
            if reason:
                rejected.append(dict(root_seed=seed, family=family, clone_hash=clone, reason=reason))
                continue
            seen.add(clone)
            groups.append(dict(root_seed=seed, family=family, clone_hash=clone,
                               pilot=accepted < cfg["sizing_pilot_structures_per_family"]))
            accepted += 1
            if accepted == cfg["target_unique_structures_per_family"]:
                break
        if accepted < cfg["target_unique_structures_per_family"]:
            rejected.append(dict(family=family, reason="stratum exhausted", missing=8-accepted))
    order = sorted(groups, key=lambda g: digest([design["student_split"]["salt"], g["clone_hash"]]))
    val = {g["clone_hash"] for g in order[:len(groups)//4]}
    out.mkdir(parents=True)
    jobs = []
    for g in groups:
        g["fold"] = "student_validation" if g["clone_hash"] in val else "student_train"
        for h in design["scope"]["horizons"]:
            for b in design["scope"]["capacities"]:
                p = generate(g["root_seed"], g["family"], h, b)
                name = p.instance_id+".json"
                save_new(out/name, as_json(p))
                jobs.append(dict(g, problem_file=name, instance_hash=p.instance_hash, horizon=h, capacity=b))
    manifest = dict(schema="teacher-v2-admission-v1", design_hash=digest(design),
                    exclusions_hash=digest(exclusions), v1_admission_hash=digest(v1_admission),
                    groups=groups, jobs=jobs, rejected=rejected, labels_generated=0,
                    wall_seconds=monotonic()-start, cpu_seconds=process_time()-cpu)
    save_new(out/"admission.json", manifest)
    return manifest
