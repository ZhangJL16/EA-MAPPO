"""Descriptive, paired DEV analysis; no steps-as-samples or max-sample risk."""
import argparse
from collections import defaultdict
from fractions import Fraction as F
import json
from pathlib import Path
from statistics import mean,stdev
from math import sqrt
from .evaluation import read_rows,save_new,digest
from .problem import load_problem


def summarize_cell(rows,prior):
    if any(r["status"]!="completed" for r in rows):
        return dict(status="unresolved",attempts=len(rows))
    bytheta = defaultdict(list)
    for row in rows:
        bytheta[row["theta"]].append(row)
    if set(bytheta)!=set(range(len(prior))):
        raise ValueError("missing hypotheses")
    risks,utilities = [],[]
    for h in range(len(prior)):
        group = bytheta[h]
        if any(r["sample_shortfall"] is None for r in group):
            return dict(status="oracle_unresolved",attempts=len(rows))
        risks.append(mean(float(F(r["sample_shortfall"])) for r in group))
        utilities.append(mean(float(F(r["utility"])) for r in group))
    # Average policy seeds within each noise block BEFORE estimating MC error.
    # This is conditional on the frozen policy-seed set, not a training-seed CI.
    bynoise = defaultdict(list)
    for r in rows:
        bynoise[r["noise_seed"]].append(r)
    noise_risks = []
    for group in bynoise.values():
        noise_risks.append(sum(float(prior[h])*mean(float(F(r["sample_shortfall"]))
                           for r in group if r["theta"]==h) for h in range(len(prior))))
    return dict(status="estimated",episodes=len(rows),risk_by_hypothesis=risks,
                utility_by_hypothesis=utilities,bayes_risk=sum(float(w)*r for w,r in zip(prior,risks)),
                worst_hypothesis_risk=max(risks),average_utility=sum(float(w)*u for w,u in zip(prior,utilities)),
                conditional_noise_mc_se=(stdev(noise_risks)/sqrt(len(noise_risks)) if len(noise_risks)>1 else None),
                planning_seconds=mean(r["planning_seconds"] for r in rows),
                expansions=mean(r["expansions"] for r in rows),model_calls=mean(r["model_calls"] for r in rows),
                measurements=mean(r["measurements"] for r in rows),
                truncated_selections=mean(r["truncated_selections"] for r in rows),
                protocols=mean(len(r["protocol_lengths"]) for r in rows),
                mean_protocol_duration=(mean(d for r in rows for d in r["protocol_lengths"])
                                        if any(r["protocol_lengths"] for r in rows) else 0))


def analyze(dataset,study):
    dataset,study = Path(dataset),Path(study)
    seal = json.loads((study/"seal.json").read_text())
    if digest(json.loads((dataset/"manifest.json").read_text()))!=seal["manifest_hash"]:
        raise ValueError("dataset seal mismatch")
    completion = json.loads((study/"completion.json").read_text())
    rows = read_rows(study/"episodes.jsonl")
    if completion["status"]!="complete" or len(rows)!=completion["rows"]:
        raise ValueError("study incomplete")
    config = seal["config"]
    cells = defaultdict(list)
    for row in rows:
        if row["instance_hash"] not in seal["selected"]:
            raise ValueError("unexpected instance")
        cells[(row["instance_hash"],row["method"],row["tier"])].append(row)
    expected_cells = {(key,method,tier) for key in seal["selected"]
                      for method,tier in ([(m,t) for m in config["methods"] for t in config["tiers"]]
                                          +[("exact_bayes","offline_reference"),("certified_minimax","offline_reference")])}
    if set(cells)!=expected_cells:
        raise ValueError("missing/unexpected method cells")
    summaries,refs = [],{}
    for (key,method,tier),records in sorted(cells.items()):
        p = load_problem(dataset/f"{key}.json")
        if p.split_id!="dev" or p.instance_hash!=key:
            raise ValueError("non-DEV/mutated problem")
        expected_keys = {(h,ps,ns) for h in range(len(p.hypotheses))
                         for ps in config["policy_seeds"] for ns in config["noise_seeds"]}
        if len(records)!=len(expected_keys) or {(r["theta"],r["policy_seed"],r["noise_seed"]) for r in records}!=expected_keys:
            raise ValueError("missing/duplicated evaluation axis")
        info = seal["selected"][key]
        result = dict(instance_hash=key,group_id=info["group_id"],distribution_id=info["distribution_id"],
                      method=method,tier=tier,capacity=p.capacity,horizon=p.budget,channels=len(p.channels))
        result.update(summarize_cell(records,p.prior))
        if key not in refs:
            refs[key] = json.loads((study/f"reference-{key}.json").read_text())
        ref = refs[key]
        if result["status"]=="estimated" and ref["exact_bayes"]["status"]=="exact" and ref["oracle_status"]=="exact":
            exactrisk = sum(float(w)*float(F(v)) for w,v in zip(p.prior,ref["oracle"]))-float(F(ref["exact_bayes"]["value"]))
            result.update(exact_bayes_risk=exactrisk,mc_excess_over_exact_bayes=result["bayes_risk"]-exactrisk)
        summaries.append(result)
    # Paired method contrasts: same instance, true model, policy seed, CRN index.
    paired = []
    for row in summaries:
        key,method,tier = row["instance_hash"],row["method"],row["tier"]
        if method in ("exact_bayes","certified_minimax") or row["status"]!="estimated":
            continue
        ref_records = cells.get((key,"exact_bayes","offline_reference"),[])
        if not ref_records or any(r["status"]!="completed" for r in ref_records):
            continue
        lookup = {(r["theta"],r["policy_seed"],r["noise_seed"]):float(F(r["sample_shortfall"])) for r in ref_records}
        p = load_problem(dataset/f"{key}.json")
        gaps = defaultdict(list)
        for r in cells[(key,method,tier)]:
            gaps[r["theta"]].append(float(F(r["sample_shortfall"]))-lookup[(r["theta"],r["policy_seed"],r["noise_seed"])])
        paired.append(dict(instance_hash=key,group_id=row["group_id"],method=method,tier=tier,
                           paired_mc_bayes_gap=sum(float(p.prior[h])*mean(gaps[h]) for h in gaps)))
    # Equal weight per root; capacity/horizon cells are repeated conditions.
    group_cells = defaultdict(list)
    for row in summaries:
        if row["status"]=="estimated":
            group_cells[(row["group_id"],row["method"],row["tier"])].append(row)
    grouped = [dict(group_id=g,method=m,tier=t,conditions=len(rs),
                    bayes_risk=mean(r["bayes_risk"] for r in rs),
                    planning_seconds=mean(r["planning_seconds"] for r in rs),
                    expansions=mean(r["expansions"] for r in rs),
                    model_calls=mean(r["model_calls"] for r in rs))
               for (g,m,t),rs in sorted(group_cells.items())]
    # Pareto status is descriptive within each instance, never across different
    # horizons/distributions. Exact references are excluded (offline cost).
    frontier = []
    for key in sorted(seal["selected"]):
        points = [r for r in summaries if r["instance_hash"]==key and r["status"]=="estimated"
                  and r["tier"]!="offline_reference"]
        for r in points:
            dominated = any(s["bayes_risk"]<=r["bayes_risk"] and s["planning_seconds"]<=r["planning_seconds"]
                            and (s["bayes_risk"]<r["bayes_risk"] or s["planning_seconds"]<r["planning_seconds"])
                            for s in points)
            frontier.append(dict(instance_hash=key,method=r["method"],tier=r["tier"],empirical_nondominated=not dominated))
    return dict(schema="fpl-dev-analysis-v1",episode_rows=len(rows),root_groups=len({v["group_id"] for v in seal["selected"].values()}),
                cells=summaries,root_aggregates=grouped,paired_to_bayes=paired,empirical_frontier=frontier,
                references=refs,episodes_sha256=__import__("hashlib").sha256((study/"episodes.jsonl").read_bytes()).hexdigest(),
                limitations=["DEV only; no neural training; no final-test access",
                  "Root graph is the sampling unit. Capacity/horizon variants are repeated conditions.",
                  "Four noise replicates and two fixed policy seeds: descriptive pilot, not population significance.",
                  "Worst-hypothesis risk is max of estimated means; finite-replicate upward selection bias remains.",
                  "MC SE is conditional on fixed policy seeds and task, not an instance-generalization interval.",
                  "Reference-unresolved cells remain visible; compare Bayes gaps only on explicitly solved subsets.",
                  "Empirical Pareto status is noise/timing dependent; wall limits are cooperative."])


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",required=True)
    parser.add_argument("--study",required=True)
    parser.add_argument("--output",required=True)
    args = parser.parse_args()
    out = Path(args.output)
    if out.exists():
        raise ValueError("refuse to overwrite analysis")
    result = analyze(args.dataset,args.study)
    save_new(out,result)
    print(json.dumps(dict(rows=result["episode_rows"],groups=result["root_groups"],cells=len(result["cells"]))))
