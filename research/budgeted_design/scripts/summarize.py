"""Descriptive paired DEV report; no best checkpoint/seed/horizon selection."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import zipfile
import numpy as np
import torch


def stats(values):
    return {"mean": float(np.mean(values)), "sd_across_training_seeds": float(np.std(values, ddof=1)),
            "per_seed": [float(x) for x in values]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--reference", required=True)
    args = p.parse_args()
    data, output = Path(args.data), Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((data/"manifest.json").read_text())
    cfg = manifest["binding"]["config"]
    rows = json.loads((data/"evaluation.json").read_text())
    lookup = {(r["kind"],r["seed"],r["intervention"],r["H"]):r for r in rows}
    expected = len(cfg["seeds"])*len(cfg["eval_horizons"])*(len(cfg["methods"])+2)
    assert len(rows) == len(lookup) == expected
    summary = {"scope":"first public-task DEV prototype; not full DAD reproduction", "aggregate":[], "pairs":[], "costs":[]}
    for kind in cfg["methods"]:
        interventions = ["normal", "query_constant_4", "query_zero"] if kind == "budget_attention" else ["normal"]
        for intervention in interventions:
            for horizon in cfg["eval_horizons"]:
                selected = [lookup[kind,s,intervention,horizon] for s in cfg["seeds"]]
                summary["aggregate"].append({"kind":kind,"intervention":intervention,"H":horizon,
                    "lower":stats([r["lower_mean"] for r in selected]),
                    "upper":stats([r["upper_mean"] for r in selected])})
    for control, intervention in [(x,"normal") for x in ["pool","fixed_attention","pool_wide"]] + [("budget_attention","query_constant_4"),("budget_attention","query_zero")]:
        for horizon in cfg["eval_horizons"]:
            samples = np.array([np.array(lookup["budget_attention",s,"normal",horizon]["lower_samples"]) -
                np.array(lookup[control,s,intervention,horizon]["lower_samples"]) for s in cfg["seeds"]])
            summary["pairs"].append({"comparison":f"budget_attention - {control}/{intervention}","H":horizon,
                "delta_spce":stats(samples.mean(1)),
                "paired_MC_SE_conditional_on_all_checkpoints":float(samples.mean(0).std(ddof=1)/np.sqrt(samples.shape[1]))})
    for kind in cfg["methods"]:
        costs = [json.loads((data/f"cost_{kind}_{s}.json").read_text()) for s in cfg["seeds"]]
        summary["costs"].append({"kind":kind,"parameters":costs[0]["parameters"],
            "total_training_seconds":sum(c["training_seconds"] for c in costs),
            "total_simulated_measurements":sum(c["measurements"] for c in costs),
            "total_likelihood_calls":sum(c["likelihood_calls"] for c in costs),
            "median_CPU_decision_ms":float(np.median([c["cpu_latency"]["median_ms"] for c in costs])),
            "median_GPU_decision_ms":float(np.median([c["training_device_latency"]["median_ms"] for c in costs]))})
    source = Path(args.reference)
    ref_files = ["README.md","LICENSE","location_finding.py","location_finding_eval.py","neural/modules.py","contrastive/mi.py"]
    provenance = {"author_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=source,text=True).strip(),
        "author_files":{f:hashlib.sha256((source/f).read_bytes()).hexdigest() for f in ref_files},
        "python":platform.python_version(),"torch":torch.__version__, "numpy":np.__version__,
        "cuda":torch.version.cuda,"gpu":torch.cuda.get_device_name(),"CPU":platform.processor(),
        "reference_test":"selected original Torch AST forward functions matched; original Pyro runner not run",
        "CUBLAS_WORKSPACE_CONFIG":":4096:8", "teacher_cost":0,
        "remaining_work":"Step-DAD; native T=30 full-budget reproduction; second task; broader repeats; final confirmation"}
    for filename,obj in [("summary.json",summary),("provenance.json",provenance),("manifest.json",manifest)]:
        (output/filename).write_text(json.dumps(obj,indent=2,allow_nan=False)+"\n")
    report = ["# Public Source Location Finding: budget-reading DEV pilot", "", "## Material Passport", "",
        "Mode: run. Evidence: actual local GPU training and paired simulation evaluation.",
        "Status: first bounded prototype completed; not paper-score reproduction or confirmation.",
        "No FPL/UAV data, exact teachers, test/OOD roots or private CONFIRM used.", "",
        "## Protocol", "",
        f"{len(cfg['seeds'])} seeds × {len(cfg['methods'])} models; {cfg['steps']} updates each. "
        f"Train outer={cfg['batch']}, inner={cfg['contrasts']}; mixed H={cfg['train_horizons']}. "
        f"Native DAD reference uses fixed H={cfg['native_horizon']}.",
        f"Evaluation: {cfg['eval_rollouts']} independent rollouts/checkpoint/H, L={cfg['eval_contrasts']}; "
        "latent/noise/contrastive draws paired across methods and training seeds. Final checkpoint only.",
        "H=6/12 are developer interpolation/extrapolation probes, not final unseen confirmation.", "",
        "## sPCE lower-bound estimate (nats, mean ± SD across 3 training seeds)", "",
        "| Method | H=4 | H=6 | H=8 | H=12 |", "|---|---:|---:|---:|---:|"]
    for kind in cfg["methods"]:
        selected = [r for r in summary["aggregate"] if r["kind"]==kind and r["intervention"]=="normal"]
        report.append("| "+kind+" | "+" | ".join(f"{r['lower']['mean']:.4f} ± {r['lower']['sd_across_training_seeds']:.4f}" for r in selected)+" |")
    report += ["", "## Paired candidate-minus-control lower-bound differences", "",
        "Positive favors budget-conditioned reading. Each cell lists all three training-seed differences.","",
        "| Control | H | seed differences | mean | conditional paired MC SE |",
        "|---|---:|---|---:|---:|"]
    for row in summary["pairs"]:
        report.append(f"| {row['comparison']} | {row['H']} | "+", ".join(f"{v:.4f}" for v in row["delta_spce"]["per_seed"])+
            f" | {row['delta_spce']['mean']:.4f} | {row['paired_MC_SE_conditional_on_all_checkpoints']:.4f} |")
    report += ["", "## Estimator interval diagnostic (means, NOT a confidence interval)","",
        "| Method | H | lower | upper | upper − lower |", "|---|---:|---:|---:|---:|"]
    for row in summary["aggregate"]:
        if row["intervention"]=="normal":
            lo,up=row["lower"]["mean"],row["upper"]["mean"]
            report.append(f"| {row['kind']} | {row['H']} | {lo:.4f} | {up:.4f} | {up-lo:.4f} |")
    report += ["", "## Costs", "", "| Method | parameters | total train seconds (3 seeds) | CPU ms/decision | GPU ms/decision |", "|---|---:|---:|---:|---:|"]
    for c in summary["costs"]:
        report.append(f"| {c['kind']} | {c['parameters']} | {c['total_training_seconds']:.2f} | {c['median_CPU_decision_ms']:.4f} | {c['median_GPU_decision_ms']:.4f} |")
    total = sum(c["total_training_seconds"] for c in summary["costs"])
    report += ["", f"Total synchronized training wall time on GPU: {total:.2f}s ({total/3600:.4f} device-hours, not GPU kernel-active hours).",
        "Training includes random-input transfer and optimization; excludes checkpoint serialization and evaluation. "
        "Native fixed-H control uses a different horizon workload; see exact simulated measurement/likelihood counts in summary.json.",
        "The first invalid evaluation briefly overlapped the tail of training; training seconds are accounting, not isolated speed benchmarks. "
        "Corrected inference latencies were measured after training completed.",
        "Latency: batch=1, four public history pairs, remaining=4, 20 warm-ups + 100 synchronized calls, one CPU thread. "
        "Not whole-deployment latency or hardware-independent FLOP equivalence.", "",
        "## Interpretation limits", "",
        "This small optimization budget tests implementation and an initial mechanism hypothesis, not converged optimal performance. "
        "Three training repeats do not support a population superiority claim. The 512 rollouts are not 512 independently trained models. "
        "Training-seed SD and conditional MC SE are distinct; steps/horizons are not independent replicates. No p-values or best-seed selection.",
        "sPCE is a finite-inner lower bound and sNMC an upper bound in expectation, not exact EIG. "
        "The lower bound has ceiling log(4097)=8.3180; a large lower/upper gap weakens rankings as claims about true EIG.",
        "Fixed-query attention and wider pooling are essential controls. A tiny positive number or query-intervention sensitivity alone "
        "does not establish a useful new mechanism. Any gains must survive stronger runs and independent tasks.",
        "Step-DAD, original full T=30 training, per-H specialists and external-task transfer were NOT completed. "
        "No novelty claim and no Spotlight rating follows from this run.", "",
        "## Corrected implementation error", "",
        "An initial evaluation omitted loading trained weights and evaluated initialized models. "
        "Those original evaluation files are retained with INVALID_EVALUATION_NOTICE.json and excluded here. "
        "All trained checkpoints were migrated byte-for-byte to a fresh corrected evaluation directory; "
        "a checkpoint loading regression test now checks exact tensor identity and changed predictions. "
        "No retraining, changed objective, seed replacement or performance-based selection occurred. "
        "See evaluation_manifest.json for both source bindings and checkpoint hashes.", "",
        "Sources: [DAD paper](https://proceedings.mlr.press/v139/foster21a.html), "
        "[pinned author code](https://github.com/ae-foster/dad/tree/4b1008174e1531d1f14601d83cef481c0f586f36).", ""]
    (output/"RESULTS.md").write_text("\n".join(report))
    # Full portable developer evidence, including optimizer checkpoints and per-rollout estimates.
    archive = output/"pilot_evidence.zip"
    with zipfile.ZipFile(archive,"x",compression=zipfile.ZIP_DEFLATED) as z:
        for file in sorted(data.iterdir()):
            if file.is_file() and file.suffix in {".json",".pt"}:
                z.write(file, "run/"+file.name)
        for filename in ["summary.json","provenance.json","manifest.json","RESULTS.md"]:
            z.write(output/filename,"report/"+filename)
        package=Path(__file__).resolve().parents[1]
        for folder in ["src","scripts","tests","configs"]:
            for file in sorted((package/folder).rglob("*")):
                if file.is_file() and file.suffix in {".py",".json"} and "__pycache__" not in file.parts:
                    z.write(file,"source/"+str(file.relative_to(package)))
        for filename in ["README.md","THIRD_PARTY.md","pyproject.toml"]:
            z.write(package/filename,"source/"+filename)
        for filename in ref_files:
            z.write(source/filename,"author_reference/"+filename)
    hashes = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.iterdir()) if p.is_file()}
    (output/"SHA256.json").write_text(json.dumps(hashes,indent=2)+"\n")
    print(json.dumps({"rows":len(rows),"total_training_seconds":total,"archive_sha256":hashes[archive.name]}))


if __name__ == "__main__":
    main()
