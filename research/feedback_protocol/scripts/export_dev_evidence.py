"""Package frozen DEV evidence and independently replay public cached policies.

No policy optimization, new instances, new seeds or final-test evaluation.
This is not a human/external proof audit.
"""
import argparse
from collections import Counter
from fractions import Fraction as F
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from fpl.problem import load_problem
from fpl.protocols import validate_protocol
from fpl.utility import protocol_vector
from fpl.belief import outcomes
from fpl.evaluation import digest,source_hashes


def feedback_probability(p,theta,feedback):
    value = F(1)
    for c,y in feedback:
        mu = p.hypotheses[theta][p.channels.index(c)]
        value *= mu if y else 1-mu
    return value


def replay(p,ref):
    n = len(p.hypotheses)
    result = {}
    if ref["exact_bayes"]["status"]=="exact":
        actions = {(a["remaining"],tuple(map(F,a["posterior"]))):a["operations"] for a in ref["exact_bayes"]["actions"]}
        @lru_cache(None)
        def value(t,b):
            names = actions[(t,b)]
            if names is None:
                return (F(0),)*n
            route = validate_protocol(p,tuple(names),t)
            vector = list(protocol_vector(p,route))
            for feedback,_,post in outcomes(p,b,route.channels):
                tail = value(t-route.duration,post)
                for theta in range(n):
                    vector[theta] += feedback_probability(p,theta,feedback)*tail[theta]
            return tuple(vector)
        vector = value(p.budget,p.prior)
        weighted = sum(w*v for w,v in zip(p.prior,vector))
        if weighted!=F(ref["exact_bayes"]["value"]):
            raise ValueError("cached Bayes policy does not attain certified value")
        result["exact_bayes"] = dict(status="rational_policy_replay_checked",utility_by_hypothesis=list(map(str,vector)),
            risk_by_hypothesis=[str(F(o)-v) for o,v in zip(ref["oracle"],vector)],weighted_utility=str(weighted))
    if ref["certified_minimax"]["status"]=="exact":
        def value_tree(tree,t):
            if tree["operations"] is None:
                return (F(0),)*n
            route = validate_protocol(p,tuple(tree["operations"]),t)
            vector = list(protocol_vector(p,route))
            expected = {fb for fb,_,_ in outcomes(p,(F(1,n),)*n,route.channels)}
            actual = [tuple(map(tuple,b["feedback"])) for b in tree["branches"]]
            if set(actual)!=expected or len(actual)!=len(expected):
                raise ValueError("incomplete tree feedback support")
            for branch,feedback in zip(tree["branches"],actual):
                tail = value_tree(branch["tree"],t-route.duration)
                for theta in range(n):
                    vector[theta] += feedback_probability(p,theta,feedback)*tail[theta]
            return tuple(vector)
        mixture = ref["certified_minimax"]["mixture"]
        if sum(F(x["weight"]) for x in mixture)!=1:
            raise ValueError("mixture not normalized")
        total = [F(0)]*n
        for item in mixture:
            vec = value_tree(item["tree"],p.budget)
            for i in range(n):
                total[i] += F(item["weight"])*vec[i]
        risks = tuple(F(o)-v for o,v in zip(ref["oracle"],total))
        if risks!=tuple(map(F,ref["certified_minimax"]["risks"])) or max(risks)!=F(ref["certified_minimax"]["value"]):
            raise ValueError("minimax mixture replay disagrees")
        result["certified_minimax"] = dict(status="rational_policy_replay_checked",risk_by_hypothesis=list(map(str,risks)))
    return result


def export(dataset,study,output):
    dataset,study,output = Path(dataset),Path(study),Path(output)
    seal = json.loads((study/"seal.json").read_text())
    if seal["sources"]!=source_hashes():
        raise ValueError("runtime sources differ from frozen study")
    analysis = json.loads((study/"analysis.json").read_text())
    output.mkdir(parents=True,exist_ok=False)
    replayed = {}
    for key in seal["selected"]:
        p = load_problem(dataset/f"{key}.json")
        if p.instance_hash!=key or p.split_id!="dev":
            raise ValueError("non-DEV or changed instance")
        ref = json.loads((study/f"reference-{key}.json").read_text())
        replayed[key] = replay(p,ref)
    (output/"policy_replay.json").write_text(json.dumps(replayed,sort_keys=True,indent=2)+"\n")
    compact = {k:v for k,v in analysis.items() if k!="references"}
    compact["references"] = {key:{
        "oracle_status":ref["oracle_status"],"oracle":ref.get("oracle"),"oracle_compute":ref["oracle_compute"],
        "total_offline_seconds":ref["total_offline_seconds"],
        **{m:{k:v for k,v in ref[m].items() if k not in ("actions","mixture")}
           for m in ("exact_bayes","certified_minimax")}}
        for key,ref in analysis["references"].items()}
    (output/"summary.json").write_text(json.dumps(compact,sort_keys=True,indent=2)+"\n")
    files = [(study/"seal.json","study/seal.json"),(study/"episodes.jsonl","study/episodes.jsonl"),
             (study/"completion.json","study/completion.json"),(study/"analysis.json","study/analysis.json"),
             (dataset/"manifest.json","dataset/manifest.json")]
    files += [(dataset/f"{key}.json",f"dataset/{key}.json") for key in seal["selected"]]
    files += [(study/f"reference-{key}.json",f"study/reference-{key}.json") for key in seal["selected"]]
    # Include frozen runtime source so a future code edit cannot orphan evidence.
    files += [(ROOT/"src/fpl"/name,f"runtime/fpl/{name}") for name in seal["sources"]]
    inventory = {}
    with zipfile.ZipFile(output/"dev_evidence.zip","x",compression=zipfile.ZIP_DEFLATED) as archive:
        for path,name in files:
            data = path.read_bytes()
            info = zipfile.ZipInfo(name,date_time=(2026,9,19,0,0,0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info,data)
            inventory[name] = dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
    archive_hash = hashlib.sha256((output/"dev_evidence.zip").read_bytes()).hexdigest()
    (output/"inventory.json").write_text(json.dumps(dict(archive_sha256=archive_hash,files=inventory,
        raw_dataset=str(dataset),raw_study=str(study)),sort_keys=True,indent=2)+"\n")
    refs = compact["references"]
    rows = [json.loads(line) for line in (study/"episodes.jsonl").read_text().splitlines()]
    counts = Counter(r["status"] for r in rows)
    lines = ["# Frozen DEV quality–compute pilot", "", 
             f"{len(rows)} scheduled records; statuses: `{dict(counts)}`.",
             "Four root groups, 12 repeated capacity/horizon conditions; no final test or training.",
             "", "## Per-condition comparison", "",
             "Risk below is Monte Carlo Bayes risk (not exact expected risk). Exact reference risk",
             "comes from rational Bayes value. Approximate methods use the LARGE shared budget.",
             "", "| Root | B | T | Exact Bayes risk | Beam MC | VOI MC | PS MC | Cover MC |", 
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    keys = sorted(seal["selected"],key=lambda k:(seal["selected"][k]["group_id"],
                  next(c["capacity"] for c in compact["cells"] if c["instance_hash"]==k),
                  next(c["horizon"] for c in compact["cells"] if c["instance_hash"]==k)))
    for key in keys:
        cs = [c for c in compact["cells"] if c["instance_hash"]==key]
        c = cs[0]
        exact = next((r.get("exact_bayes_risk") for r in cs if "exact_bayes_risk" in r),None)
        scores = [next(r["bayes_risk"] for r in cs if r["method"]==m and r["tier"]=="large")
                  for m in ("beam_bayes","one_step_voi","posterior_sampling","channel_cover")]
        lines.append(f"| {c['group_id']} | {c['capacity']} | {c['horizon']} | {'unresolved' if exact is None else f'{exact:.4f}'} | "+" | ".join(f"{s:.4f}" for s in scores)+" |")
    lines += ["", "## Reference solve costs", "",
              "| Root / B / T | Bayes status | States | Offline seconds | Minimax status |", "|---|---|---:|---:|---|"]
    for key in keys:
        c = next(c for c in compact["cells"] if c["instance_hash"]==key)
        b,m = refs[key]["exact_bayes"],refs[key]["certified_minimax"]
        lines.append(f"| {c['group_id']} / {c['capacity']} / {c['horizon']} | {b['status']} | {b['compute']['states']} | {b['compute']['seconds']:.6f} | {m['status']} |")
    lines += ["", "## Interpretation limits", ""]+[f"- {x}" for x in compact["limitations"]]
    lines += ["- A zero empirical MC SE from four draws is not zero true uncertainty.",
              "- Root aggregates with unresolved references cover different numbers of conditions; do not compare mismatched averages.",
              "- No parameter/seed/horizon/policy changes were made after outcome inspection.",
              "- This is internal rational replay, not external independent expert validation.","",
              f"Raw evidence archive SHA-256: `{archive_hash}`.",""]
    (output/"RESULTS.md").write_text("\n".join(lines))
    print(json.dumps(dict(records=len(rows),checked_policy_instances=sum(len(x) for x in replayed.values()),
                          archive_bytes=(output/"dev_evidence.zip").stat().st_size,archive_sha256=archive_hash)))


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",required=True)
    parser.add_argument("--study",required=True)
    parser.add_argument("--output",required=True)
    args = parser.parse_args()
    export(args.dataset,args.study,args.output)
