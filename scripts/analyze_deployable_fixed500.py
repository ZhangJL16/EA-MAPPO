#!/usr/bin/env python3
"""Paired, fixed-benchmark v2 audit; conditional task uncertainty, not seed inference."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest

from scripts.resume_recovery_sac_ppo_stratified import BUCKETS, TASK_SHA256, load_tasks


def read(path: Path):
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def describe(x: np.ndarray) -> dict:
    return {"n": len(x), "mean": float(x.mean()), "sd": float(x.std(ddof=1)),
            "median": float(np.median(x)), "q25": float(np.quantile(x, .25)),
            "q75": float(np.quantile(x, .75)), "q95": float(np.quantile(x, .95)),
            "max": float(x.max())}


def wilson(k: int, n: int) -> tuple[float, float]:
    p, z = k / n, 1.959963984540054
    center = (p + z*z/(2*n)) / (1+z*z/n)
    radius = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return center-radius, center+radius


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, default=ROOT/"artifacts/recovery_sac_ppo_scratch_20260906_v1/sac")
    parser.add_argument("--candidate", type=Path, default=ROOT/"artifacts/deployable_observation_v2_fixed500_20260907_v1")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    tasks, seed = load_tasks()
    task_map = {r["source_task_index"]: r for r in tasks}
    paths = [args.baseline/"evaluation_stratified.json", args.candidate/"evaluation_stratified.json"]
    summaries = [read(args.baseline/"summary.json"), read(args.candidate/"summary.json")]
    assert summaries[0]["task_source_sha256"] == TASK_SHA256
    assert summaries[1]["contract"]["task_sha256"] == TASK_SHA256
    assert summaries[1]["contract"]["formal"]
    assert sha(Path(summaries[1]["contract"]["checkpoint"])/"model.zip") == summaries[1]["contract"]["model_sha256"]
    raw = [read(p) for p in paths]
    for rows in raw:
        assert len(rows) == 500 and len({r["source_task_index"] for r in rows}) == 500
        for r in rows:
            t = task_map[r["source_task_index"]]
            assert r["seed"] == seed+r["source_task_index"] and r["distance_bucket"] == t["distance_bucket"]
            assert r["safe_goal"] == (bool(r["goal_reached"]) and r["collision_count"] == 0)
            assert 0 <= r["collision_count"] <= r["steps"] <= 4000
            assert np.isclose(r["straight_line_distance"], t["straight_line_distance"])
    a, b = [sorted(rows, key=lambda r: r["source_task_index"]) for rows in raw]
    for x,y in zip(a,b,strict=True):
        assert np.isclose(x["start_distance"], y["start_distance"])
    fields = ["goal_reached", "safe_goal", "collision_count", "energy"]
    arrays = [{k: np.array([r[k] for r in rows], dtype=float) for k in fields} for rows in (a,b)]
    rng = np.random.default_rng(20260907)
    strata = [np.array([i for i,r in enumerate(a) if r["distance_bucket"] == bucket]) for bucket in BUCKETS]
    samples = np.concatenate([rng.choice(ids, (5000,len(ids)), replace=True) for ids in strata], axis=1)
    effects = {}
    for field in fields:
        old,new = arrays[0][field], arrays[1][field]
        delta = new-old
        lo,hi = np.quantile(delta[samples].mean(axis=1), [.025,.975])
        effects[field] = {"baseline": describe(old), "v2": describe(new),
                          "delta_mean": float(delta.mean()), "delta_ci95": [float(lo),float(hi)]}
        if field in fields[:2]:
            lost = int(((old==1)&(new==0)).sum())
            gained = int(((old==0)&(new==1)).sum())
            effects[field].update(baseline_only=lost, v2_only=gained,
                both_success=int(((old==1)&(new==1)).sum()),
                mcnemar_exact_p=float(binomtest(gained,lost+gained,.5).pvalue) if lost+gained else 1.)
    order = sorted(fields[:2], key=lambda k: effects[k]["mcnemar_exact_p"])
    previous = 0.
    for rank,k in enumerate(order):
        previous = max(previous, min(1., (2-rank)*effects[k]["mcnemar_exact_p"]))
        effects[k]["holm_p"] = previous
    common = np.array([x["goal_reached"] and y["goal_reached"] for x,y in zip(a,b,strict=True)])
    shared = {}
    for field in ("energy", "success_path_ratio"):
        old = np.array([a[i][field] for i in np.flatnonzero(common)],float)
        new = np.array([b[i][field] for i in np.flatnonzero(common)],float)
        shared[field] = {"baseline": describe(old), "v2": describe(new), "delta_mean": float((new-old).mean())}
    buckets = {}
    for bucket,ids in zip(BUCKETS,strata,strict=True):
        buckets[bucket] = {label:{k:float(values[k][ids].mean()) for k in fields}
                           for label,values in zip(("baseline","v2"),arrays,strict=True)}
    tail = []
    for rows in (a,b):
        total = sum(r["collision_count"] for r in rows)
        timeouts = [r for r in rows if r["termination"] == "deadline"]
        tail.append({"timeouts":len(timeouts), "contact_timeouts":sum(r["collision_count"]>0 for r in timeouts),
            "contacts_in_timeouts":sum(r["collision_count"] for r in timeouts), "total_contacts":total,
            "tasks_at_least_1000_contacts":sum(r["collision_count"]>=1000 for r in rows),
            "top10_contact_share":sum(r["collision_count"] for r in sorted(rows,key=lambda r:r["collision_count"],reverse=True)[:10])/max(total,1)})
    stats = {"n_tasks":500,"training_runs_per_method":1,"training_seed":0,
             "source_sha256":{str(p):sha(p) for p in paths}, "task_sha256":TASK_SHA256,
             "effects":effects,"common_arrivals":shared,"distance_buckets":buckets,
             "tail_baseline_v2":tail,"bootstrap_replicates":5000,"bootstrap_seed":20260907,
             "inference_scope":"conditional paired task/scene inference for these frozen models; not across training seeds"}
    out = args.output_dir.resolve()
    out.mkdir(parents=True,exist_ok=False)
    figures = out/"figures"
    figures.mkdir()
    (out/"statistics.json").write_text(json.dumps(stats,indent=2,allow_nan=False)+"\n")
    plt.rcParams.update({"font.size":10,"pdf.fonttype":42,"axes.spines.top":False,"axes.spines.right":False})
    colors = ["#0072B2","#D55E00"]
    fig,axes = plt.subplots(1,2,figsize=(10,4.1),layout="constrained")
    for ax,field,title in zip(axes,fields[:2],["Goal arrival","Collision-free arrival"],strict=True):
        for j,(label,values,color) in enumerate(zip(["Prior SAC","V2"],arrays,colors,strict=True)):
            ys=np.array([values[field][ids].mean() for ids in strata])
            cis=np.array([wilson(int(values[field][ids].sum()),len(ids)) for ids in strata])
            ax.errorbar(np.arange(5)+(j-.5)*.13,ys*100,
                        yerr=np.maximum(0,np.stack([ys-cis[:,0],cis[:,1]-ys]))*100,
                        fmt="o-" if j==0 else "s--",color=color,label=label,capsize=3)
        ax.set(xticks=np.arange(5),xticklabels=BUCKETS,ylim=(0,105),ylabel="Tasks (%)",title=title,xlabel="Distance (m)")
        ax.tick_params(axis="x",labelrotation=25)
        ax.grid(axis="y",alpha=.2)
        ax.legend()
    fig.savefig(figures/"figure-01-distance-comparison.pdf")
    fig.savefig(figures/"figure-01-distance-comparison.png",dpi=180)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4),layout="constrained")
    thresholds=np.arange(4001)
    for label,values,color,style in zip(["Prior SAC","V2"],arrays,colors,["-","--"],strict=True):
        ys=np.array([(values["collision_count"]>=v).mean() for v in thresholds])
        ax.plot(thresholds,ys*100,color=color,linestyle=style,label=label)
    ax.set(xscale="symlog",xlim=(0,4000),ylim=(0,100),xlabel="Unified contacts per task (threshold; symlog)",ylabel="Tasks at or above threshold (%)")
    ax.legend(); ax.grid(alpha=.2)
    fig.savefig(figures/"figure-02-contact-tail.pdf")
    fig.savefig(figures/"figure-02-contact-tail.png",dpi=180)
    plt.close(fig)
    table="| Metric | Prior SAC | V2 | V2−prior |\n|---|---:|---:|---:|\n"
    for k in fields:
        e=effects[k]; table+=f"| {k} | {e['baseline']['mean']:.6f} | {e['v2']['mean']:.6f} | {e['delta_mean']:+.6f} |\n"
    (out/"analysis-report.md").write_text(
        "# Paired fixed500 audit\n\nQuestion: did the final v2 retain navigation on the registered fixed tasks?\n\n"+table+
        "\nBoth frozen models use one training seed0;500 paired task instances are not500 training runs. "
        "The benchmark has been used before, so it is not a fresh holdout.\n\n"
        "Both compared500-task runs have HOCBF disabled; separate raw/HOCBF return experiments are not this baseline. "
        "V2 regressed on both arrival endpoints; do not promote it as an improved energy-aware navigator. "
        "Collision tails and common-arrival energy/path comparisons are in statistics.json. "
        "All timeouts remain included. Energy is synthetic realized cost, not a sustainability certificate.\n\n"
        "## Claim candidates\n\n"
        "- Claim: this frozen v2 has lower arrival and collision-free arrival on fixed500.\n"
        "  - Evidence: paired rows, risk differences, stratified bootstrap, exact McNemar with Holm across two endpoints.\n"
        "  - Allowed: regression for these checkpoints on this benchmark.\n"
        "  - Forbidden: all single-channel policies are worse, or deleting flags caused the loss.\n"
        "  - Uncertainty: one training seed; multiple input changes versus prior SAC; repeated benchmark.\n"
        "  - Next check: keep v2 context and restore only a distance-derived internal hit channel.\n"
        "  - Decision: keep bounded empirical claim; causal attribution withheld.\n\n"
        "- Claim: extra battery inputs have not established energy management.\n"
        "  - Evidence: task-only training, reset budget, budget-independent reward/termination.\n"
        "  - Allowed: this study does not test energy-sustainable mission decisions.\n"
        "  - Forbidden: energy safety solved or disproved by navigation-only evaluation.\n"
        "  - Uncertainty: future energy-task training remains untested.\n"
        "  - Next check: explicit budget/return objective after navigation representation is resolved.\n"
        "  - Decision: retain limitation.\n")
    (out/"stats-appendix.md").write_text(
        "# Statistical audit\n\n"+table+
        "\nSee statistics.json for means, sample SDs, medians, quartiles,95th percentiles, maxima, "
        "paired raw-scale effects and95% conditional task bootstrap intervals. Resample pairs within "
        "each of five strata (100 each),5000 replicates, seed20260907. Two co-primary binary endpoints "
        "use exact McNemar (binomial on discordant pairs), Holm-adjusted together; no normality assumption "
        "or t-test. Heavy-tailed zero-inflated counts are retained without trimming; continuous effects "
        "are descriptive/exploratory, not extra significance claims. Common-arrival path/energy effects "
        "are explicitly conditional on both models arriving, not total task utility.\n\n"
        "McNemar assumes independent task pairs and endpoint exchangeability under the null. "
        "Independent scenario seeds support a scene-level interpretation, but do not remove shared "
        "training or repeated-benchmark adaptivity. Intervals concern a hypothetical same-stratum "
        "task distribution for these fixed models, not uncertainty in the exact finite benchmark counts "
        "and not algorithm-level variation across training seeds. No across-seed significance claim.\n\n"
        "Figure1 uses per-bucket Wilson95% score intervals (100 tasks), not run SD; descriptive bucket "
        "breakdowns have no independent post-hoc significance stars. Figure2 is exact empirical survival, "
        "with no uncertainty band or causal implication.\n")
    (out/"figure-catalog.md").write_text(
        "# Figure catalog\n\n## figure-01-distance-comparison.pdf\n\n"
        "Purpose: locate navigation degradation across distance strata. Source: paired immutable500, "
        "100 tasks per bucket, two frozen seed0 models. Points show rates; error bars Wilson95% task-level "
        "intervals, not training-run variability. Notice whether long-range safe arrival is weaker and "
        "that arrival and safety are distinct. Implication: do not approve the new interface based on "
        "training arrival alone. No per-bucket causal/significance claim; inspect exact rates and pairing.\n\n"
        "## figure-02-contact-tail.pdf\n\nPurpose: reveal repeated-contact tails hidden by median0. "
        "Source: all500 unified policy-step counts; empirical fraction>=threshold; symlog x axis, "
        "no smoothing or trimming, no error band. Notice the extended high-count tail of v2. "
        "Implication: inspect failed recovery, not merely mean path of successful tasks. This does "
        "not identify collision cause or justify changing frozen collision semantics.\n")
    print(json.dumps({"effects":effects,"common_arrivals":shared,"tails":tail,"output_dir":str(out)},indent=2))


if __name__ == "__main__":
    main()
