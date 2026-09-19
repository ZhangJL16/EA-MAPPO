"""Public prior-predictive history coverage; selection precedes label solves."""
from collections import defaultdict, deque
from fractions import Fraction as F
import json
from time import monotonic, process_time
from fpl.belief import PlannerState, outcomes
from fpl.protocols import validate_protocol, PlanningLimit
from fpl.teacher_data import advance, draw_index, state_key, solver
from fpl.evaluation import digest
from fpl.work import PlanningWorkBudget, WatchdogExpired
from fpl.policies.beam_bayes import BeamBayes
from fpl.policies.one_step_voi import OneStepVOI
from fpl.policies.posterior_sampling import PosteriorSampling
from .decoder import ScalableFeasibilityMask
from .generator import rng_for


def dispersion(p, state, channels, duration, budget):
    if not channels:
        return F(0)
    return sum(m*sum((v-u)**2 for v, u in zip(post, state.posterior))
               for _, m, post in outcomes(p, state.posterior, channels, 32, budget))/duration


def information_candidates(p, state, account, width=16):
    """Prefix beam ranks pending observations, NEVER utility or teacher labels.

    Partial-prefix score uses elapsed duration, complete score full duration.
    Pending observations only enter a prior-predictive score, not an update to
    the decoder belief. At most width active prefixes survive each depth.
    """
    mask = ScalableFeasibilityMask(p, budget=account)
    active, done = [()], []
    while active:
        successors = []
        for prefix in active:
            for key in mask.legal_next(prefix, state.remaining):
                kind, name = json.loads(key)
                if kind != "operation":
                    continue
                path = prefix+(name,)
                pos = mask.state(path, state.remaining)
                channels = tuple(c for n in path for c in mask.ops[n].channels)
                elapsed = state.remaining-pos.time_left
                score = dispersion(p, state, channels, elapsed, account)
                if pos.complete:
                    if channels:
                        done.append((score, path))
                else:
                    successors.append((score, path))
        successors.sort(key=lambda x: (-x[0], x[1]))
        active = [path for _, path in successors[:width]]
    done.sort(key=lambda x: (-x[0], x[1]))
    return [validate_protocol(p, path, state.remaining) for _, path in done]


def random_route(p, state, account, rng):
    mask, prefix = ScalableFeasibilityMask(p, budget=account), ()
    while True:
        choices = [json.loads(k)[1] for k in mask.legal_next(prefix, state.remaining)
                   if json.loads(k)[0] == "operation"]
        if not choices:
            return validate_protocol(p, prefix, state.remaining) if prefix else None
        prefix += (rng.choice(choices),)
        if mask.state(prefix, state.remaining).complete:
            return validate_protocol(p, prefix, state.remaining)


def bin_index(value, edges):
    edges = tuple(map(F, edges))
    if not edges[0] <= value <= edges[-1]:
        raise ValueError("public feature outside bin interval")
    return next((i for i in range(len(edges)-1) if value < edges[i+1]), len(edges)-2)


def stratum(p, state, measured, design):
    cfg = design["state_collection"]["strata"]
    gini = (1-sum(v*v for v in state.posterior))/(1-F(1, len(p.prior)))
    distance = sum(abs(a-b) for a, b in zip(state.posterior, p.prior))/2
    return (bin_index(gini, cfg["normalized_gini"]["edges"]),
            bin_index(F(state.remaining, p.budget), cfg["relative_H"]["edges"]),
            bin_index(distance, cfg["distance_from_prior"]["edges"]), min(2, measured))


def retain(pool, p, seed, design):
    bins = defaultdict(list)
    for key, row in pool.items():
        row["retention_stratum"] = min(row.pop("public_strata"))
        bins[tuple(row["retention_stratum"])].append(key)
    salt = rng_for(seed, "retention").getrandbits(256)
    queues = {b: deque(sorted(keys, key=lambda k: digest([salt, k]))) for b, keys in bins.items()}
    root = state_key(PlannerState(p.prior, p.budget, p.capacity))
    chosen = [root] if design["state_collection"]["retain_root"] else []
    for q in queues.values():
        if root in q and root in chosen:
            q.remove(root)
    while len(chosen) < design["state_collection"]["max_selected_states_per_problem"]:
        before = len(chosen)
        for b in sorted(queues):
            if queues[b]:
                chosen.append(queues[b].popleft())
                if len(chosen) == design["state_collection"]["max_selected_states_per_problem"]:
                    break
        if len(chosen) == before:
            break
    return [pool[k] for k in chosen]


def collect(p, seed, design):
    pool, logs = {}, []
    cfg = design["state_collection"]
    for source in cfg["sources"]:
        account = PlanningWorkBudget(**design["work_limits"]["collection_per_source_episode"])
        # Separate substreams by source, using the frozen state_exploration stream.
        import random
        rng = random.Random(digest([rng_for(seed, "state_exploration").getrandbits(256), source]))
        state, path, keys = PlannerState(p.prior, p.budget, p.capacity), [], set()
        policy = (OneStepVOI(p, width=16) if source == "one_step_voi" else
                  BeamBayes(p, width=16, depth=2) if source == "beam" else
                  PosteriorSampling(p, seed=rng.getrandbits(64), width=16) if source == "posterior_sampling" else None)
        start, cpu = monotonic(), process_time()
        status, reason, source_solves = "complete", None, []

        def add(s, history, kind):
            key = state_key(s)
            keys.add(key)
            row = pool.setdefault(key, dict(id=key,
                state=dict(posterior=list(map(str, s.posterior)), remaining=s.remaining, resource=s.resource),
                provenance=[], public_strata=[]))
            row["provenance"].append(dict(source=source, kind=kind, history=history))
            row["public_strata"].append(stratum(p, s, len(s.history.released), design))

        add(state, [], "root")
        try:
            for _ in range(cfg["rollout_batches_per_source"]):
                if state.remaining == 0:
                    break
                if source == "exact":
                    obj = solver(p, design["work_limits"]["exact_label"])
                    obj.max_seconds = account.watchdog_seconds
                    obj.search_budget = account
                    try:
                        obj.value(state.remaining, state.posterior)
                    finally:
                        source_solves.append(dict(states=obj.states, likelihood_branches=obj.likelihood_branches))
                    route = obj.actions[(state.remaining, state.posterior)]
                elif source == "random":
                    route = random_route(p, state, account, rng)
                elif source == "public_information_explorer":
                    candidates = information_candidates(p, state, account)
                    route = candidates[0] if candidates else None
                else:
                    route = policy.select(state, budget=account)
                    if policy.stats.get("limit_reason"):
                        status, reason = "truncated", policy.stats["limit_reason"]
                if route is None:
                    break
                route = validate_protocol(p, route.operations, state.remaining)
                branches = list(outcomes(p, state.posterior, route.channels, cfg["max_positive_outcomes"], account))
                for feedback, _, _ in branches:
                    history = path+[dict(operations=list(route.operations), feedback=list(feedback))]
                    add(advance(p, state, route, feedback), history, "reachable_feedback")
                feedback = branches[draw_index([x[1] for x in branches], rng)][0]
                path = path+[dict(operations=list(route.operations), feedback=list(feedback))]
                state = advance(p, state, route, feedback)
                add(state, path, "rollout")
        except (PlanningLimit, WatchdogExpired) as exc:
            status, reason = "truncated", str(exc)
        logs.append(dict(source=source, status=status, reason=reason, pool_states=len(keys),
                         work=account.snapshot(), teacher_solves=source_solves,
                         cpu_seconds=process_time()-cpu, wall_seconds=monotonic()-start))
    selected = retain(pool, p, seed, design)
    for log in logs:
        log["selected_memberships"] = sum(any(o["source"] == log["source"] for o in r["provenance"]) for r in selected)
    return dict(records=selected, sampling=logs, pool_states=len(pool), selected_before_labels=True)
