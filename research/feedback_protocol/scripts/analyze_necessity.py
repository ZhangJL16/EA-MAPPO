"""Frozen-rule exact-gap/compute analysis. No policy optimization or new data."""
import argparse
from collections import Counter,defaultdict
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from fpl.evaluation import digest,source_hashes
from fpl.problem import load_problem


def analyze(study,output):
    study,output = Path(study),Path(output)
    seal = json.loads((study/"seal.json").read_text())
    completion = json.loads((study/"completion.json").read_text())
    if seal["sources"]!=source_hashes():
        raise ValueError("runtime source changed")
    rows = [json.loads(s) for s in (study/"results.jsonl").read_text().splitlines()]
    if completion["status"]!="complete" or len(rows)!=len(seal["jobs"]):
        raise ValueError("incomplete study")
    if {r["job"]["id"] for r in rows}!={j["id"] for j in seal["jobs"]}:
        raise ValueError("schedule mismatch")
    if any(r.get("content_hash")!=digest({k:v for k,v in r.items() if k!="content_hash"}) for r in rows):
        raise ValueError("checkpoint hash mismatch")
    references = {}
    for key,meta in seal["metadata"].items():
        if meta["cohort"]=="fresh_dev_v2":
            ref = json.loads((study/f"reference-{key}.json").read_text())
            if ref["content_hash"]!=digest({k:v for k,v in ref.items() if k!="content_hash"}):
                raise ValueError("reference hash mismatch")
            references[key] = ref
    compact = []
    for row in rows:
        item = {**row["job"],**row["metadata"]}
        result = row["result"]
        item.update({k:v for k,v in result.items() if k not in ("action_map","fallback")})
        p = load_problem(study/"problems"/f"{item['instance_hash']}.json")
        item["horizon"] = p.budget
        if result["status"]=="exact_conditional_policy":
            if row["job"]["scope"]=="episode":
                factor = seal["plan"]["episode_multiplier"]
                if (result["worst_episode_expansions"]>row["job"]["limits"]["max_expansions"]*factor or
                    result["worst_episode_model_calls"]>row["job"]["limits"]["max_model_calls"]*factor):
                    raise AssertionError("episode budget violated")
            if item["instance_hash"] in references and references[item["instance_hash"]]["status"]=="exact":
                ref = references[item["instance_hash"]]
                gap = F(result["bayes_risk"])-F(ref["bayes_risk"])
                if gap<0:
                    raise AssertionError("exact fixed policy beats exact optimum")
                item.update(exact_bayes_risk=ref["bayes_risk"],exact_gap=str(gap),normalized_gap=str(gap/p.budget))
        elif "fallback" in result:
            item["mc_fallback"] = {k:v for k,v in result["fallback"].items() if k!="samples"}
        compact.append(item)
    fresh = [c for c in compact if c["cohort"]=="fresh_dev_v2"]
    rule = seal["plan"]["decision_rule"]
    witnesses = defaultdict(list)
    for key,ref in references.items():
        if ref["status"]!="exact":
            continue
        candidates = [c for c in fresh if c["instance_hash"]==key and c["scope"]=="episode"]
        lows = [c for c in candidates if c["tier"]==rule["low_tier"] and c["method"] in rule["strong_methods"]]
        highs = [c for c in candidates if c["tier"]==rule["high_tier"] and c["method"] in rule["strong_methods"]]
        low_ok = len(lows)==len(rule["strong_methods"])*len(seal["plan"]["policy_seeds"]) and all(
            "normalized_gap" in c and F(c["normalized_gap"])>=F(rule["minimum_normalized_low_gap"]) for c in lows)
        if not low_ok:
            continue
        for hi in highs:
            if "normalized_gap" not in hi or F(hi["normalized_gap"])>F(rule["maximum_normalized_high_gap"]):
                continue
            lo = next(c for c in lows if c["method"]==hi["method"] and c["seed"]==hi["seed"])
            if F(hi["expected_model_calls"])>=rule["minimum_work_increase"]*max(F(1),F(lo["expected_model_calls"])):
                witnesses[hi["condition"]].append(dict(root=hi["group_id"],method=hi["method"],seed=hi["seed"],
                        low_gap=lo["normalized_gap"],high_gap=hi["normalized_gap"],
                        model_call_ratio=str(F(hi["expected_model_calls"])/max(F(1),F(lo["expected_model_calls"])))))
    growth = []
    for root in sorted({m["group_id"] for m in seal["metadata"].values() if m["cohort"]=="fresh_dev_v2"}):
        rootrefs = {seal["metadata"][k]["condition"]:r for k,r in references.items() if seal["metadata"][k]["group_id"]==root}
        base = rootrefs["base"]
        if base["status"]!="exact":
            continue
        basecalls = sum(a["work"]["model_calls"] for a in base["attempts"])
        for axis in ("long_horizon","redundant_channels","deeper_hierarchy"):
            r = rootrefs[axis]
            if r["status"]=="exact":
                calls = sum(a["work"]["model_calls"] for a in r["attempts"])
                if calls>=rule["minimum_work_increase"]*max(1,basecalls):
                    growth.append(dict(root=root,axis=axis,model_call_ratio=calls/max(1,basecalls)))
    quality = any(len({w["root"] for w in ws})>=rule["minimum_roots"] for ws in witnesses.values())
    compute = len({w["root"] for w in growth})>=rule["minimum_roots"]
    unresolved_scale = any(r["status"]!="exact" and seal["metadata"][k]["condition"] in
                           ("long_horizon","redundant_channels","deeper_hierarchy") for k,r in references.items())
    verdict = "GO" if quality and compute else ("HOLD" if unresolved_scale else "STOP")
    summary = dict(schema="fpl-necessity-v4-analysis",decision=verdict,
        decision_scope="Frozen DEV operational criterion; not a proof that neural learning is necessary/sufficient.",
        quality_condition=quality,compute_condition=compute,quality_witnesses=dict(witnesses),reference_growth=growth,
        counts=dict(Counter(c["status"] for c in compact)),fresh_counts=dict(Counter(c["status"] for c in fresh)),
        reference_counts=dict(Counter(r["status"] for r in references.values())),cells=compact,references=references,
        plan=seal["plan"],seal_sha256=hashlib.sha256((study/"seal.json").read_bytes()).hexdigest(),
        limitations=["Two fresh root groups; axis variants are repeated conditions, not independent roots.",
          "Exact expected values integrate environment feedback only, conditional on each frozen algorithm seed.",
          "MCTS/PS here do not integrate the distribution over all internal random seeds.",
          "Wall watchdog invalidates a result; it never selects an action.",
          "MC fallbacks remain MC and cannot satisfy the exact-gap GO criterion.",
          "No new B2 gate, neural training, final test or external task is executed."])
    output.mkdir(parents=True,exist_ok=False)
    (output/"summary.json").write_text(json.dumps(summary,sort_keys=True,indent=2)+"\n")
    lines = ["# Final pre-B2 necessity study", "",f"Decision under frozen operational rule: **{verdict}**.","",
             f"Quality condition: {quality}; compute-growth condition: {compute}.",
             f"Exact-reference statuses: `{summary['reference_counts']}`.",
             f"Fixed-policy evaluations: `{summary['counts']}` (fresh subset `{summary['fresh_counts']}`).","",
             "## Fresh DEV: large-tier episode-budget exact risk", "",
             "All numbers integrate feedback exactly conditional on policy seed. U = unresolved (not zero).", "",
             "| Root | Condition | Bayes optimum | Cover | PS | Beam | VOI | MCTS |","|---|---|---:|---:|---:|---:|---:|---:|"]
    for key,meta in seal["metadata"].items():
        if meta["cohort"]!="fresh_dev_v2":
            continue
        ref = references[key]
        values = []
        for method in seal["plan"]["methods"]:
            c = next(c for c in fresh if c["instance_hash"]==key and c["method"]==method and c["tier"]=="large" and c["scope"]=="episode")
            values.append(f"{float(F(c['bayes_risk'])):.6f}" if c["status"]=="exact_conditional_policy" else "U")
        optimum = f"{float(F(ref['bayes_risk'])):.6f}" if ref["status"]=="exact" else "U"
        lines.append(f"| {meta['group_id']} | {meta['condition']} | {optimum} | "+" | ".join(values)+" |")
    lines += ["","## Legacy DEV deterministic-policy re-evaluation","",
              "This is the new deterministic-work contract, not a claim to reproduce every clock-dependent old action.","",
              "| Root | B | H | Method | Tier | Seed | Exact risk | Status |","|---|---:|---:|---|---|---:|---:|---|"]
    for c in compact:
        if c["cohort"]=="fresh_dev_v2":
            continue
        p = load_problem(study/"problems"/f"{c['instance_hash']}.json")
        value = f"{float(F(c['bayes_risk'])):.6f}" if c["status"]=="exact_conditional_policy" else "U"
        lines.append(f"| {c['group_id']} | {p.capacity} | {p.budget} | {c['method']} | {c['tier']} | {c['seed']} | {value} | {c['status']} |")
    lines += ["","## Boundaries",""]+[f"- {x}" for x in summary["limitations"]]
    (output/"RESULTS.md").write_text("\n".join(lines)+"\n")
    files = [(p,str(p.relative_to(study))) for p in sorted(study.rglob("*")) if p.is_file() and p.name!="writer.lock"]
    files += [(ROOT/"src/fpl"/name,f"runtime/fpl/{name}") for name in seal["sources"]]
    inventory = {}
    with zipfile.ZipFile(output/"raw_evidence.zip","x",compression=zipfile.ZIP_DEFLATED) as archive:
        for path,name in files:
            data = path.read_bytes()
            info = zipfile.ZipInfo(name,date_time=(2026,9,19,0,0,0)); info.compress_type=zipfile.ZIP_DEFLATED
            archive.writestr(info,data)
            inventory[name] = dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
    (output/"inventory.json").write_text(json.dumps(dict(files=inventory,archive_sha256=hashlib.sha256((output/"raw_evidence.zip").read_bytes()).hexdigest()),sort_keys=True,indent=2)+"\n")
    print(json.dumps({k:summary[k] for k in ("decision","quality_condition","compute_condition","counts","fresh_counts","reference_counts")}))


if __name__=="__main__":
    p = argparse.ArgumentParser(); p.add_argument("--study",required=True); p.add_argument("--output",required=True)
    a = p.parse_args(); analyze(a.study,a.output)
