"""DEV-only, episode-atomic, sealed expected-risk study.

Offline exact reference construction and online selection are separate costs.
Incomplete reference solutions remain unresolved, never heuristic substitutes.
Policy/environment seeds and hashes are evaluator-only. No final test loader.
"""
import argparse
from collections import defaultdict
from fractions import Fraction as F
import hashlib
import json
import os
from pathlib import Path
import platform
import random
from time import monotonic
import fcntl
from .problem import load_problem
from .registry import RegistryV2
from .environment import Environment,PrivateTruth
from .budget import SearchBudget
from .protocols import PlanningLimit,validate_protocol
from .teachers.exact_bayes import ExactBayes,known_model_value
from .teachers.minimax_small import solve
from .policies.channel_cover import ChannelCover
from .policies.posterior_sampling import PosteriorSampling
from .policies.beam_bayes import BeamBayes
from .policies.one_step_voi import OneStepVOI


def digest(data):
    return hashlib.sha256(json.dumps(data,sort_keys=True,default=str).encode()).hexdigest()


def source_hashes():
    root = Path(__file__).parent
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*.py"))}


def save_new(path,data):
    # Atomic publication. A killed temporary write is never read as a checkpoint.
    tmp = path.with_suffix(path.suffix+".tmp")
    with tmp.open("w") as f:
        json.dump(data,f,sort_keys=True,indent=2,default=str)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp,path)


def encode_tree(tree):
    return dict(operations=None if tree.route is None else tree.route.operations,
                branches=[dict(feedback=feedback,tree=encode_tree(child)) for feedback,child in tree.branches])


def references(p,config):
    started = monotonic()
    account = SearchBudget(**config["budget"])
    result = dict(instance_hash=p.instance_hash)
    try:
        result["oracle"] = [str(known_model_value(p,h,budget=account)) for h in range(len(p.hypotheses))]
        result["oracle_status"] = "exact"
    except PlanningLimit as exc:
        result.update(oracle_status="unresolved",oracle_reason=str(exc))
    result["oracle_compute"] = account.snapshot()
    planner = ExactBayes(p,max_states=config["max_states"],max_seconds=config["max_seconds"],
                         max_protocol_nodes=config["max_protocol_nodes"],budget_limits=config["budget"])
    try:
        value = planner.value(p.budget,p.prior)
        result["exact_bayes"] = dict(status="exact",value=str(value),
            actions=[dict(remaining=t,posterior=list(map(str,b)),
                          operations=None if route is None else route.operations)
                     for (t,b),route in planner.actions.items()])
    except PlanningLimit as exc:
        result["exact_bayes"] = dict(status="unresolved",reason=str(exc))
    result["exact_bayes"]["compute"] = dict(**planner.search_budget.snapshot(),states=planner.states,
                                            likelihood_branches=planner.likelihood_branches)
    account = SearchBudget(**config["budget"])
    try:
        cert = solve(p,max_trees=config["max_trees"],max_combinations=config["max_combinations"],
                     max_seconds=config["max_seconds"],budget=account)
        result["certified_minimax"] = dict(status="exact",value=str(cert["value"]),
            risks=list(map(str,cert["risks"])),least_favorable_prior=list(map(str,cert["least_favorable_prior"])),
            mixture=[dict(weight=str(w),tree=encode_tree(tree)) for w,tree in cert["mixture"]],stats=cert["stats"])
    except PlanningLimit as exc:
        result["certified_minimax"] = dict(status="unresolved",reason=str(exc))
    result["certified_minimax"]["compute"] = account.snapshot()
    result["total_offline_seconds"] = monotonic()-started
    result["content_hash"] = digest(result)
    return result


class ReferencePolicy:
    """A public Bayes action table or prior-free minimax mixture; no truth."""
    def __init__(self,p,method,reference,seed):
        self.problem,self.method = p,method
        if method=="exact_bayes":
            self.table = {(r["remaining"],tuple(map(F,r["posterior"]))):r["operations"]
                          for r in reference["actions"]}
        else:
            mixture = reference["mixture"]
            self.tree = random.Random(seed).choices([r["tree"] for r in mixture],
                         weights=[float(F(r["weight"])) for r in mixture])[0]
        self.previous_length = None
        self.stats = {}

    def select(self,state):
        if self.method=="exact_bayes":
            names = self.table[(state.remaining,state.posterior)]
        else:
            if self.previous_length is not None:
                feedback = state.history.released[self.previous_length:]
                matches = [b["tree"] for b in self.tree["branches"]
                           if tuple(map(tuple,b["feedback"]))==feedback]
                if len(matches)!=1:
                    raise ValueError("missing/ambiguous minimax continuation")
                self.tree = matches[0]
            names = self.tree["operations"]
            self.previous_length = len(state.history.released)
        return None if names is None else validate_protocol(self.problem,tuple(names),state.remaining)


def make_policy(p,method,tier,seed,config,refs):
    if method in ("exact_bayes","certified_minimax"):
        return ReferencePolicy(p,method,refs[method],seed)
    limits = config["tiers"][tier]
    if method=="channel_cover":
        return ChannelCover(p,budget_limits=limits)
    if method=="posterior_sampling":
        return PosteriorSampling(p,seed=seed,width=config["beam_width"],budget_limits=limits)
    if method=="beam_bayes":
        return BeamBayes(p,width=config["beam_width"],depth=config["beam_depth"],budget_limits=limits)
    if method=="one_step_voi":
        return OneStepVOI(p,width=config["beam_width"],budget_limits=limits)
    raise ValueError("unknown method")


def episode(p,identity,method,tier,theta,policy_seed,noise_seed,config,refs):
    row = dict(instance_hash=p.instance_hash,group_id=identity["group_id"],method=method,tier=tier,
               theta=theta,policy_seed=policy_seed,noise_seed=noise_seed,budget=p.budget,capacity=p.capacity)
    row["id"] = digest(row)
    if method in ("exact_bayes","certified_minimax") and refs[method]["status"]!="exact":
        return dict(**row,status="reference_unresolved",reason=refs[method]["reason"])
    policy = make_policy(p,method,tier,policy_seed,config,refs)
    env = Environment(p,PrivateTruth(theta,noise_seed),crn_key=identity["group_id"])
    utility,planning_seconds,expansions,model_calls,truncated = F(0),0.,0,0,0
    protocol_lengths,measurements,limit_reasons = [],0,defaultdict(int)
    while env.time<p.budget:
        state = env.planner_state()
        start = monotonic()
        route = policy.select(state)
        planning_seconds += monotonic()-start
        stats = getattr(policy,"stats",{})
        expansions += stats.get("expansions",0)
        model_calls += stats.get("model_calls",0)
        truncated += int(stats.get("truncated",False))
        if "limit_reason" in stats:
            limit_reasons[stats["limit_reason"]] += 1
        if route is None:
            break
        out = env.execute(route.operations)
        utility += out.reward
        protocol_lengths.append(route.duration)
        measurements += len(out.feedback)
    shortfall = None if refs["oracle_status"]!="exact" else str(F(refs["oracle"][theta])-utility)
    return dict(**row,status="completed",utility=str(utility),sample_shortfall=shortfall,
                planning_seconds=planning_seconds,expansions=expansions,model_calls=model_calls,
                truncated_selections=truncated,limit_reasons=dict(limit_reasons),protocol_lengths=protocol_lengths,
                measurements=measurements,elapsed=env.time)


def read_rows(path):
    rows = []
    if path.exists():
        with path.open() as stream:
            for line in stream:
                if not line.endswith("\n"):
                    raise ValueError("incomplete JSONL tail; retain file and recover explicitly")
                rows.append(json.loads(line))
    if len({r["id"] for r in rows})!=len(rows):
        raise ValueError("duplicate episode checkpoint")
    for r in rows:
        key = {k:r[k] for k in ("instance_hash","group_id","method","tier","theta","policy_seed","noise_seed","budget","capacity")}
        if digest(key)!=r["id"]:
            raise ValueError("episode identity mismatch")
    return rows


def run(dataset,plan_path,output,resume=False,max_new_episodes=None):
    dataset,plan_path,output = Path(dataset),Path(plan_path),Path(output)
    config = json.loads(plan_path.read_text())
    if config.get("split")!="dev":
        raise ValueError("this runner authorizes only FPL DEV")
    if max_new_episodes is not None and max_new_episodes<1:
        raise ValueError("positive episode cap required")
    for name in ("noise_seeds","policy_seeds","methods"):
        if not config[name] or len(set(config[name]))!=len(config[name]):
            raise ValueError("nonempty distinct axes required")
    if not set(config["methods"])<={"channel_cover","posterior_sampling","beam_bayes","one_step_voi"} or not config["tiers"]:
        raise ValueError("invalid method/tier")
    for limits in config["tiers"].values():
        SearchBudget(**limits)
    manifest = json.loads((dataset/"manifest.json").read_text())
    registry = RegistryV2.from_dict(manifest["registry"])
    selected = {k:v for k,v in registry.entries.items() if v["split"]=="dev"}
    if not selected:
        raise ValueError("no DEV instances")
    seal = dict(schema="fpl-dev-study-v1",config=config,manifest_hash=digest(manifest),sources=source_hashes(),
                python=platform.python_version(),platform=platform.platform(),selected=selected)
    if resume:
        if json.loads((output/"seal.json").read_text())!=seal:
            raise ValueError("source/config/data seal changed; resume refused")
    else:
        output.mkdir(parents=True,exist_ok=False)
        save_new(output/"seal.json",seal)
    with (output/"writer.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        rows = read_rows(output/"episodes.jsonl")
        done = {r["id"] for r in rows}
        added,total = 0,0
        with (output/"episodes.jsonl").open("a") as stream:
            for digest_p,identity in sorted(selected.items()):
                p = load_problem(dataset/f"{digest_p}.json")
                if p.instance_hash!=digest_p or p.split_id!="dev":
                    raise ValueError("problem hash/split mismatch")
                ref_path = output/f"reference-{digest_p}.json"
                if not ref_path.exists():
                    save_new(ref_path,references(p,config["references"]))
                refs = json.loads(ref_path.read_text())
                if refs.get("instance_hash")!=digest_p or refs.get("content_hash")!=digest({k:v for k,v in refs.items() if k!="content_hash"}):
                    raise ValueError("reference checkpoint mismatch")
                pairs = [(m,t) for m in config["methods"] for t in config["tiers"]]
                pairs += [("exact_bayes","offline_reference"),("certified_minimax","offline_reference")]
                for method,tier in pairs:
                    for theta in range(len(p.hypotheses)):
                        for ps in config["policy_seeds"]:
                            for ns in config["noise_seeds"]:
                                total += 1
                                key = dict(instance_hash=digest_p,group_id=identity["group_id"],method=method,tier=tier,
                                           theta=theta,policy_seed=ps,noise_seed=ns,budget=p.budget,capacity=p.capacity)
                                if digest(key) in done:
                                    continue
                                if max_new_episodes is not None and added>=max_new_episodes:
                                    return dict(status="paused",rows=len(done),added=added)
                                row = episode(p,identity,method,tier,theta,ps,ns,config,refs)
                                stream.write(json.dumps(row,sort_keys=True)+"\n")
                                stream.flush()
                                os.fsync(stream.fileno())
                                done.add(row["id"])
                                added += 1
                print(json.dumps(dict(instance=digest_p,rows=len(done),bayes=refs["exact_bayes"]["status"],
                                      minimax=refs["certified_minimax"]["status"])),flush=True)
        if len(done)!=total:
            raise ValueError("unexpected checkpoint rows")
        result = dict(status="complete",rows=len(done),added=added)
        save_new(output/"completion.json",result)
        return result


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",required=True)
    parser.add_argument("--plan",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--max-new-episodes",type=int)
    args = parser.parse_args()
    print(json.dumps(run(args.dataset,args.plan,args.output,args.resume,args.max_new_episodes)))
