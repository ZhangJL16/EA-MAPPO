"""Final pre-B2 study. Frozen public DEV-v2 plus separate legacy-policy audit."""
import argparse
from copy import deepcopy
from fractions import Fraction as F
import fcntl
import json
import os
from pathlib import Path
import platform
from time import monotonic
from .difficulty import tasks
from .generator import problem_json
from .problem import load_problem
from .evaluation import digest,source_hashes,save_new,ReferencePolicy
from .teachers.exact_bayes import ExactBayes
from .teachers.exact_policy import ExactPolicyEvaluator
from .work import PlanningWorkBudget,WorkPolicy,WatchdogExpired
from .protocols import PlanningLimit
from .environment import Environment,PrivateTruth
from .policies.channel_cover import ChannelCover
from .policies.posterior_sampling import PosteriorSampling
from .policies.beam_bayes import BeamBayes
from .policies.one_step_voi import OneStepVOI
from .policies.belief_mcts import BeliefMCTS


def make_policy(p,method,seed,limits,scope,plan):
    if method=="channel_cover":
        agent = ChannelCover(p)
    elif method=="posterior_sampling":
        agent = PosteriorSampling(p,seed=seed,width=plan["beam_width"])
    elif method=="beam_bayes":
        agent = BeamBayes(p,width=plan["beam_width"],depth=plan["beam_depth"])
    elif method=="one_step_voi":
        agent = OneStepVOI(p,width=plan["beam_width"])
    elif method=="belief_mcts":
        agent = BeliefMCTS(p,seed=seed,exploration=plan["mcts_exploration"])
    else:
        raise ValueError("unknown fixed policy")
    episode = {k:v*plan["episode_multiplier"] for k,v in limits.items()} if scope=="episode" else None
    return WorkPolicy(p,agent,limits,episode,watchdog_seconds=60)


def known_diagnosis(p,task_duration):
    """Analytic oracle, checked against structural conditions, not true-label access."""
    for op in p.operations:
        if op.channels:
            if op.utility is None or op.utility.observation_weights or any(op.utility.by_hypothesis):
                raise ValueError("analytic diagnosis oracle inapplicable")
        if op.utility and any(op.utility.by_hypothesis):
            if op.source!=p.reset or op.target!=p.reset or op.duration!=task_duration or max(op.utility.by_hypothesis)>1:
                raise ValueError("non-unit task in analytic oracle")
    for h in range(len(p.hypotheses)):
        if not any(o.source==o.target==p.reset and o.energy<=p.capacity and o.duration==task_duration
                   and o.utility and o.utility.by_hypothesis and o.utility.by_hypothesis[h]==1 for o in p.operations):
            raise ValueError("known model cannot execute matching task")
    return [str(F(p.budget//task_duration))]*len(p.hypotheses)


def exact_reference(p,plan,task_duration):
    result = dict(instance_hash=p.instance_hash,oracle=known_diagnosis(p,task_duration),attempts=[])
    for conf in plan["reference_budgets"]:
        planner = ExactBayes(p,max_states=conf["max_states"],max_seconds=conf["watchdog_seconds"],max_protocol_nodes=200000)
        planner.search_budget = PlanningWorkBudget(**{k:v for k,v in conf.items() if k!="max_states"})
        start = monotonic()
        try:
            value = planner.value(p.budget,p.prior)
            attempt = dict(status="exact",value=str(value))
        except (PlanningLimit,WatchdogExpired) as exc:
            attempt = dict(status="unresolved",reason=str(exc))
        attempt.update(states=planner.states,likelihood_branches=planner.likelihood_branches,
                       work=planner.search_budget.snapshot(),seconds=monotonic()-start,limits=conf)
        result["attempts"].append(attempt)
        if attempt["status"]=="exact":
            result.update(status="exact",value=str(value),bayes_risk=str(sum(w*F(o) for w,o in zip(p.prior,result["oracle"]))-value))
            break
    else:
        result["status"] = "unresolved"
    return result


def add_risks(result,p,oracle):
    if result["status"]!="exact_conditional_policy":
        return result
    risks = [F(o)-F(v) for o,v in zip(oracle,result["utility_by_hypothesis"])]
    result.update(risk_by_hypothesis=list(map(str,risks)),bayes_risk=str(sum(w*r for w,r in zip(p.prior,risks))),
                  worst_hypothesis_risk=str(max(risks)),
                  expected_expansions=str(sum(w*F(x) for w,x in zip(p.prior,result["expansions_by_hypothesis"]))),
                  expected_model_calls=str(sum(w*F(x) for w,x in zip(p.prior,result["model_calls_by_hypothesis"]))),
                  expected_latency=sum(float(w)*x for w,x in zip(p.prior,result["measured_latency_by_hypothesis"])))
    return result


def mc_fallback(p,policy,oracle,seeds,group):
    samples = []
    for theta in range(len(p.hypotheses)):
        for seed in seeds:
            env,agent = Environment(p,PrivateTruth(theta,seed),crn_key=group),deepcopy(policy)
            utility = F(0)
            try:
                while env.time<p.budget:
                    route = agent.select(env.planner_state())
                    if route is None:
                        break
                    utility += env.execute(route.operations).reward
                samples.append(dict(theta=theta,noise_seed=seed,status="completed",utility=str(utility)))
            except WatchdogExpired as exc:
                samples.append(dict(theta=theta,noise_seed=seed,status="watchdog_invalid",reason=str(exc)))
    if any(r["status"]!="completed" for r in samples):
        return dict(status="unresolved",samples=samples)
    means = [sum(F(r["utility"]) for r in samples if r["theta"]==h)/len(seeds) for h in range(len(p.hypotheses))]
    risks = [F(o)-v for o,v in zip(oracle,means)]
    return dict(status="mc_fallback_only",samples=samples,risk_by_hypothesis=list(map(str,risks)),
                bayes_risk=str(sum(w*r for w,r in zip(p.prior,risks))),worst_hypothesis_risk=str(max(risks)))


def build_jobs(plan,legacy_dataset=None,legacy_study=None):
    problems,metadata,legacy_refs,jobs = {},{},{},[]
    for p,meta in tasks(plan):
        key = p.instance_hash
        problems[key],metadata[key] = p,dict(meta,cohort="fresh_dev_v2")
        settings = [(tier,"episode",limits) for tier,limits in plan["tiers"].items()]
        tier = plan["selection_only_tier"]
        settings.append((tier,"selection",plan["tiers"][tier]))
        for method in plan["methods"]:
            for seed in plan["policy_seeds"]:
                for tier,scope,limits in settings:
                    jobs.append(dict(instance_hash=key,method=method,seed=seed,tier=tier,scope=scope,limits=limits))
    if legacy_dataset is not None:
        seal = json.loads((Path(legacy_study)/"seal.json").read_text())
        for key,meta in seal["selected"].items():
            if meta["split"]!="dev":
                raise ValueError("legacy audit only authorizes old DEV")
            p = load_problem(Path(legacy_dataset)/f"{key}.json")
            if p.instance_hash!=key or p.split_id!="dev":
                raise ValueError("legacy source mismatch")
            problems[key],metadata[key] = p,dict(meta,cohort="legacy_dev_deterministic_policy_audit")
            ref = json.loads((Path(legacy_study)/f"reference-{key}.json").read_text())
            if ref.get("content_hash")!=digest({k:v for k,v in ref.items() if k!="content_hash"}):
                raise ValueError("legacy reference hash mismatch")
            legacy_refs[key] = ref
            for method in seal["config"]["methods"]:
                for seed in (seal["config"]["policy_seeds"] if method=="posterior_sampling" else seal["config"]["policy_seeds"][:1]):
                    for tier,old in seal["config"]["tiers"].items():
                        limits = {k:v for k,v in old.items() if k!="max_seconds"}
                        jobs.append(dict(instance_hash=key,method=method,seed=seed,tier=tier,scope="selection",limits=limits))
            if ref["exact_bayes"]["status"]=="exact":
                jobs.append(dict(instance_hash=key,method="cached_bayes",seed=0,tier="reference",scope="offline",limits={}))
    for row in jobs:
        row["id"] = digest(row)
    return problems,metadata,legacy_refs,jobs


def run(plan_path,output,resume=False,max_new=None,legacy_dataset=None,legacy_study=None):
    plan_path,output = Path(plan_path),Path(output)
    plan = json.loads(plan_path.read_text())
    if max_new is not None and max_new<1:
        raise ValueError("positive checkpoint cap required")
    if (legacy_dataset is None)!=(legacy_study is None):
        raise ValueError("legacy dataset and study must be supplied together")
    problems,metadata,old_refs,jobs = build_jobs(plan,legacy_dataset,legacy_study)
    seal = dict(schema="fpl-necessity-v4",plan=plan,sources=source_hashes(),metadata=metadata,jobs=jobs,
                legacy_reference_hashes={k:digest(v) for k,v in old_refs.items()},python=platform.python_version())
    if resume:
        if json.loads((output/"seal.json").read_text())!=seal:
            raise ValueError("source/plan/problem seal changed")
    else:
        output.mkdir(parents=True,exist_ok=False)
        save_new(output/"seal.json",seal)
        (output/"problems").mkdir()
        for key,p in problems.items():
            (output/"problems"/f"{key}.json").write_text(problem_json(p)+"\n")
    with (output/"writer.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path = output/"results.jsonl"
        records = []
        if path.exists():
            with path.open() as f:
                for line in f:
                    if not line.endswith("\n"):
                        raise ValueError("incomplete checkpoint tail")
                    row = json.loads(line)
                    if row.get("content_hash")!=digest({k:v for k,v in row.items() if k!="content_hash"}):
                        raise ValueError("checkpoint content mismatch")
                    records.append(row)
        done = {r["job"]["id"] for r in records}
        if len(done)!=len(records) or not done<={j["id"] for j in jobs}:
            raise ValueError("unexpected/duplicate checkpoints")
        added = 0
        with path.open("a") as stream:
            for job in jobs:
                if job["id"] in done:
                    continue
                if max_new is not None and added>=max_new:
                    return dict(status="paused",completed=len(done),scheduled=len(jobs))
                key = job["instance_hash"]
                p,meta = problems[key],metadata[key]
                refpath = output/f"reference-{key}.json"
                if key in old_refs:
                    oracle = old_refs[key]["oracle"]
                else:
                    if not refpath.exists():
                        ref = exact_reference(p,plan,meta["axes"]["task"])
                        ref["content_hash"] = digest(ref)
                        save_new(refpath,ref)
                    ref = json.loads(refpath.read_text())
                    if ref.get("content_hash")!=digest({k:v for k,v in ref.items() if k!="content_hash"}):
                        raise ValueError("reference checkpoint mismatch")
                    oracle = ref["oracle"]
                policy = (ReferencePolicy(p,"exact_bayes",old_refs[key]["exact_bayes"],0) if job["method"]=="cached_bayes"
                          else make_policy(p,job["method"],job["seed"],job["limits"],job["scope"],plan))
                result = ExactPolicyEvaluator(p,**plan["evaluation"]).evaluate(policy)
                add_risks(result,p,oracle)
                if job["method"]=="cached_bayes" and result["status"]=="exact_conditional_policy":
                    value = sum(w*F(v) for w,v in zip(p.prior,result["utility_by_hypothesis"]))
                    if value!=F(old_refs[key]["exact_bayes"]["value"]):
                        raise AssertionError("fixed-policy replay disagrees with cached Bayes optimum")
                if result["status"]=="unresolved":
                    result["fallback"] = mc_fallback(p,policy,oracle,plan["fallback_noise_seeds"],meta["group_id"])
                row = dict(job=job,metadata=meta,oracle=oracle,result=result)
                row["content_hash"] = digest(row)
                stream.write(json.dumps(row,sort_keys=True)+"\n")
                stream.flush()
                os.fsync(stream.fileno())
                done.add(job["id"])
                added += 1
                print(json.dumps(dict(completed=len(done),scheduled=len(jobs),cohort=meta["cohort"],
                    condition=meta.get("condition"),method=job["method"],tier=job["tier"],scope=job["scope"],status=result["status"])),flush=True)
        result = dict(status="complete",completed=len(done),scheduled=len(jobs))
        save_new(output/"completion.json",result)
        return result


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--max-new",type=int)
    parser.add_argument("--legacy-dataset")
    parser.add_argument("--legacy-study")
    args = parser.parse_args()
    print(json.dumps(run(args.plan,args.output,args.resume,args.max_new,args.legacy_dataset,args.legacy_study)))
