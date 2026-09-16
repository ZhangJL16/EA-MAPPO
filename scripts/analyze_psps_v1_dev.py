#!/usr/bin/env python3
"""Frozen ordered Gate O then Gate D analyzer for complete PSPS-v1 DEV."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from scripts.psps_v1_features import LATEST_NAMES, PSPS_NAMES, features, fold_for_world
from scripts.run_conditional_history_collection import atomic_json
from scripts.validate_identification_contract import file_hash
from scripts.validate_psps_v1_contract import validate_contract

CHANNELS=("latest","psps"); ALPHAS=(0.1,1.0,10.0,100.0); CLIP=(1e-6,1-1e-6)


def sigmoid(value):
    return np.clip(1.0/(1.0+np.exp(-np.clip(value,-40,40))),*CLIP)


def fit_logistic(x, successes, trials, alpha):
    mean=x.mean(0); std=np.maximum(x.std(0),1e-6); z=(x-mean)/std; design=np.column_stack((np.ones(len(z)),z)); beta=np.zeros(design.shape[1]); penalty=np.r_[0.0,np.full(z.shape[1],alpha)]
    for _ in range(100):
        p=sigmoid(design@beta); gradient=design.T@(trials*p-successes)+penalty*beta
        weight=trials*p*(1-p); hessian=design.T@(design*weight[:,None])+np.diag(penalty)+np.eye(len(beta))*1e-10
        step=np.linalg.solve(hessian,gradient); beta-=step
        if np.max(np.abs(step))<1e-10: break
    if not all(np.isfinite(v).all() for v in (mean,std,beta)): raise ValueError("non-finite logistic state")
    return {"mean":mean,"std":std,"beta":beta,"alpha":float(alpha)}


def predict(state,x): return sigmoid(np.column_stack((np.ones(len(x)),(x-state["mean"])/state["std"]))@state["beta"])


def logloss(prob, success, trials):
    rate=success/trials; return -(rate*np.log(prob)+(1-rate)*np.log1p(-prob))


def select_alpha(x,success,trials,worlds):
    folds=np.array([fold_for_world(str(w),outer=False) for w in worlds]); losses={}
    for alpha in ALPHAS:
        values=[]
        for fold in range(4):
            train,test=folds!=fold,folds==fold
            if not train.any() or not test.any(): raise ValueError("empty inner PSPS world fold")
            state=fit_logistic(x[train],success[train],trials[train],alpha); values.extend(logloss(predict(state,x[test]),success[test],trials[test]).tolist())
        losses[alpha]=float(np.mean(values))
    return min(ALPHAS,key=lambda a:(losses[a],-a))


def inner_predictions(x,success,trials,worlds):
    folds=np.array([fold_for_world(str(w),outer=False) for w in worlds]); result=np.zeros(len(x))
    for fold in range(4):
        train,test=folds!=fold,folds==fold; alpha=select_alpha(x[train],success[train],trials[train],worlds[train]); result[test]=predict(fit_logistic(x[train],success[train],trials[train],alpha),x[test])
    return result


def choose_threshold(prob,q):
    grid=np.arange(1,20)/20; values=[np.mean(q[np.arange(len(q)),(prob>t).astype(int)]) for t in grid]
    best=max(values); return float(max(t for t,v in zip(grid,values) if abs(v-best)<=1e-15))


def simultaneous(rows,seed):
    point=rows.mean(0); se=rows.std(0,ddof=1)/math.sqrt(len(rows)); rng=np.random.Generator(np.random.PCG64(seed)); centered=rows-point
    multipliers=rng.standard_normal((20_000,len(rows))); stats=(multipliers@centered/len(rows))/np.maximum(se,1e-15); critical=float(np.quantile(np.max(np.abs(stats),axis=1),0.95))
    return point,point-critical*se,point+critical*se,critical


def world_means(rows,worlds):
    identities=list(dict.fromkeys(worlds.tolist())); return np.asarray([rows[worlds==w].mean(0) for w in identities])


def load_data(contract_path,run_dir,receipt_path):
    contract=json.loads(contract_path.read_text()); receipt=json.loads(receipt_path.read_text()); validation=validate_contract(contract,contract_path.parent,receipt)
    if not validation["PSPS_DEV_READY"]: raise ValueError("invalid PSPS contract/receipt")
    results=json.loads((run_dir/"results.json").read_text()); assigned=contract["pre_h"]["master_splits"]["PSPS_DEV"]
    if results.get("completed")!=96 or [r["world_identity"] for r in results["records"]]!=assigned: raise ValueError("PSPS DEV is incomplete or unordered")
    xs={c:[] for c in CHANNELS}; worlds=[]; steps=[]; completion=[]; q=[]
    for record in results["records"]:
        jp=run_dir/"records"/record["json"]; npzp=run_dir/"records"/record["npz"]
        if file_hash(jp).removeprefix("sha256:")!=record["json_sha256"] or file_hash(npzp).removeprefix("sha256:")!=record["npz_sha256"]: raise ValueError("PSPS raw record hash mismatch")
        meta=json.loads(jp.read_text());
        with np.load(npzp,allow_pickle=False) as archive: arrays={k:archive[k].copy() for k in archive.files}
        feats=features(arrays); included=[a for a in meta["anchors"] if a.get("included")]
        if len(included)!=len(feats["psps"]): raise ValueError("PSPS anchor/feature mismatch")
        for index,anchor in enumerate(included):
            pair={r:{a:None for a in ("R","C")} for r in range(64)}
            for row in anchor["resamples"]:
                r=int(row["replicate"]); a=row["action"]; o=row["outcome"]
                expected=float(o["task_increment"]-2*o["operational_failure"]-0.25*(o["collision_count"]>0))
                if o["utility"]!=expected or o["collision_free_arrival"] is not (o["collision_count"]==0): raise ValueError("PSPS utility/collision semantics changed")
                if pair[r][a] is not None: raise ValueError("duplicate PSPS pair cell")
                pair[r][a]=row
            if any(v[a] is None for v in pair.values() for a in ("R","C")): raise ValueError("incomplete 64-pair grid")
            if any(v["R"]["disturbance_seed"]!=v["C"]["disturbance_seed"] for v in pair.values()): raise ValueError("C/R CRN mismatch")
            comp=np.zeros((2,)); qq=np.zeros((2,2))
            for half in (0,1):
                reps=[r for r in range(64) if r%2==half]; comp[half]=sum(pair[r]["C"]["outcome"]["task_increment"]>=1 for r in reps)
                for ai,a in enumerate(("R","C")): qq[half,ai]=np.mean([pair[r][a]["outcome"]["utility"] for r in reps])
            for c in CHANNELS: xs[c].append(feats[c][index])
            worlds.append(meta["world_identity"]); steps.append(anchor["anchor_step"]); completion.append(comp); q.append(qq)
    return contract,{c:np.asarray(v) for c,v in xs.items()},np.asarray(worlds),np.asarray(steps),np.asarray(completion),np.asarray(q),results


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--contract",type=Path,required=True); parser.add_argument("--freeze-receipt",type=Path,required=True); parser.add_argument("--run-dir",type=Path,required=True); parser.add_argument("--output-dir",type=Path,required=True); args=parser.parse_args()
    output=args.output_dir.resolve()
    if output.exists(): raise FileExistsError(output)
    contract,xs,worlds,steps,completion,q,results=load_data(args.contract.resolve(),args.run_dir.resolve(),args.freeze_receipt.resolve()); outer=np.array([fold_for_world(str(w),outer=True) for w in worlds]); swaps=[]
    for discovery,evaluation in ((0,1),(1,0)):
        row={"pred":{},"threshold":{},"constant_prob":np.zeros(len(worlds)),"constant_action":np.zeros(len(worlds),int)}
        for c in CHANNELS: row["pred"][c]=np.zeros(len(worlds)); row["threshold"][c]=np.zeros(len(worlds))
        for fold in range(6):
            train,test=outer!=fold,outer==fold; success=completion[:,discovery]; trials=np.full(len(worlds),32.0)
            row["constant_prob"][test]=success[train].sum()/trials[train].sum(); row["constant_action"][test]=int(q[train,discovery].mean(0)[1]>q[train,discovery].mean(0)[0])
            for c in CHANNELS:
                alpha=select_alpha(xs[c][train],success[train],trials[train],worlds[train]); state=fit_logistic(xs[c][train],success[train],trials[train],alpha); row["pred"][c][test]=predict(state,xs[c][test])
                inner=inner_predictions(xs[c][train],success[train],trials[train],worlds[train]); threshold=choose_threshold(inner,q[train,discovery]); row["threshold"][c][test]=threshold
        swaps.append(row)
    briers={c:[] for c in (*CHANNELS,"constant")}; log_losses={c:[] for c in (*CHANNELS,"constant")}
    for swap,evaluation in zip(swaps,(1,0)):
        rate=completion[:,evaluation]/32
        for c,p in (("latest",swap["pred"]["latest"]),("psps",swap["pred"]["psps"]),("constant",swap["constant_prob"])):
            briers[c].append(p*p-2*p*rate+rate); log_losses[c].append(logloss(p,completion[:,evaluation],np.full(len(worlds),32)))
    brier={c:np.mean(v,axis=0) for c,v in briers.items()}; contrasts_o=np.column_stack((brier["latest"]-brier["psps"],brier["constant"]-brier["psps"])); wo=world_means(contrasts_o,worlds); po,lo,uo,co=simultaneous(wo,contract["pre_h"]["static_spec"]["bootstrap"]["gate_o_seed"])
    prevalence=[float(completion[:,h].sum()/(32*len(worlds))) for h in (0,1)]; singular=np.linalg.svd((xs["psps"]-xs["psps"].mean(0))/np.maximum(xs["psps"].std(0),1e-6),compute_uv=False); positive=int(np.sum(singular>max(singular[0]*1e-10,1e-12)))
    counts={str(s):int(np.sum(steps==s)) for s in (256,768)}; raw_ok=len(set(worlds))>=90 and all(v>=85 for v in counts.values())
    gate_o_checks={"raw_validity":raw_ok,"nondegenerate_completion":all(0.05<=p<=0.95 for p in prevalence),"simultaneous_positive_lcbs":bool(np.all(lo>0)),"minimum_relative_brier_reductions":bool(po[0]/brier["latest"].mean()>=0.05 and po[1]/brier["constant"].mean()>=0.05),"finite_and_rank":bool(all(np.isfinite(v).all() for v in brier.values()) and positive>=8)}; gate_o_passed=all(gate_o_checks.values())
    gate_d={"interpreted":gate_o_passed,"passed":False,"checks":{},"reason":None}
    if gate_o_passed:
        value_rows=[]; disagreements=[]; mixes=[]; threshold_summary=[]
        for swap,evaluation in zip(swaps,(1,0)):
            actions={c:(swap["pred"][c]>swap["threshold"][c]).astype(int) for c in CHANNELS}; qe=q[:,evaluation]; idx=np.arange(len(worlds)); value_rows.append(np.column_stack((qe[idx,actions["psps"]]-qe[idx,actions["latest"]],qe[idx,actions["psps"]]-qe[idx,swap["constant_action"]]))); disagreements.append(actions["psps"]!=actions["latest"])
            mixes.append({a:{"fraction":float(np.mean(actions["psps"]==ai)),"worlds":len(set(worlds[actions["psps"]==ai].tolist()))} for ai,a in enumerate(("R","C"))}); threshold_summary.append({c:sorted(set(swap["threshold"][c].tolist())) for c in CHANNELS})
        values=np.mean(value_rows,axis=0); wd=world_means(values,worlds); pd,ld,ud,cd=simultaneous(wd,contract["pre_h"]["static_spec"]["bootstrap"]["gate_d_seed"]); checks={"simultaneous_positive_lcbs":bool(np.all(ld>0)),"minimum_point_gains":bool(np.all(pd>=0.05)),"action_disagreement":float(np.mean(disagreements))>=0.05,"both_actions_represented":all(row[a]["fraction"]>=0.05 and row[a]["worlds"]>=5 for row in mixes for a in ("R","C"))}; gate_d={"interpreted":True,"passed":all(checks.values()),"checks":checks,"contrasts":{"psps_minus_latest_utility":{"estimate":float(pd[0]),"lcb":float(ld[0]),"ucb":float(ud[0])},"psps_minus_best_constant_utility":{"estimate":float(pd[1]),"lcb":float(ld[1]),"ucb":float(ud[1])}},"action_disagreement":float(np.mean(disagreements)),"action_mix":mixes,"outer_fold_thresholds":threshold_summary,"simultaneous_critical":cd}
    else: gate_d["reason"]="GATE_O_FAILED_STOP_WITHOUT_GATE_D_INTERPRETATION"
    result={"schema_version":"psps-v1-dev-gates-v1","role":"BLIND_DEV_GATE_O_THEN_D","worlds":len(set(worlds)),"anchors":len(worlds),"step_anchor_counts":counts,"gate_o":{"passed":gate_o_passed,"checks":gate_o_checks,"completion_prevalence_halves":prevalence,"brier":{c:float(v.mean()) for c,v in brier.items()},"log_loss":{c:float(np.mean(v)) for c,v in log_losses.items()},"contrasts":{"latest_brier_minus_psps":{"estimate":float(po[0]),"lcb":float(lo[0]),"ucb":float(uo[0])},"constant_brier_minus_psps":{"estimate":float(po[1]),"lcb":float(lo[1]),"ucb":float(uo[1])}},"relative_reductions":{"vs_latest":float(po[0]/brier["latest"].mean()),"vs_constant":float(po[1]/brier["constant"].mean())},"psps_positive_singular_values":positive,"simultaneous_critical":co},"gate_d":gate_d,"confirmation_authorized":bool(gate_o_passed and gate_d["passed"]),"old_pai_confirm_accessed":False,"method_train_authorized":False,"input_hashes":{"contract":file_hash(args.contract.resolve()),"receipt":file_hash(args.freeze_receipt.resolve()),"results":file_hash(args.run_dir.resolve()/"results.json")}}
    output.mkdir(parents=True)
    model_arrays={}; model_rows=[]
    for channel in CHANNELS:
        for discovery in (0,1):
            success=completion[:,discovery]; trials=np.full(len(worlds),32.0); alpha=select_alpha(xs[channel],success,trials,worlds); state=fit_logistic(xs[channel],success,trials,alpha); threshold=choose_threshold(inner_predictions(xs[channel],success,trials,worlds),q[:,discovery]); prefix=f"{channel}_half{discovery}"
            for name in ("mean","std","beta"): model_arrays[f"{prefix}_{name}"]=state[name]
            model_rows.append({"channel":channel,"discovery_half":discovery,"alpha":alpha,"threshold":threshold,"prefix":prefix})
    np.savez_compressed(output/"models.npz",**model_arrays); atomic_json(output/"models.json",{"schema_version":"psps-v1-frozen-dev-models-v1","states":model_rows,"feature_names":{"latest":list(LATEST_NAMES),"psps":list(PSPS_NAMES)},"refit_on_confirm":False})
    support_arrays={}; support_json={"schema_version":"psps-v1-support-v1","channels":{}}
    for channel in CHANNELS:
        mean=xs[channel].mean(0); std=np.maximum(xs[channel].std(0),1e-6); standardized=(xs[channel]-mean)/std; _,singular,right=np.linalg.svd(standardized,full_matrices=False); components=right[:min(20,len(right))]; representation=standardized@components.T; distances=[]
        for index,world in enumerate(worlds):
            candidates=np.flatnonzero(worlds!=world); distances.append(float(np.linalg.norm(representation[candidates]-representation[index],axis=1).min()))
        radius=float(np.quantile(distances,0.99)*1.15); support_arrays[f"{channel}_mean"]=mean; support_arrays[f"{channel}_std"]=std; support_arrays[f"{channel}_components"]=components; support_json["channels"][channel]={"components":len(components),"radius":radius,"positive_singular_values":int(np.sum(singular>max(singular[0]*1e-10,1e-12)))}
    np.savez_compressed(output/"support.npz",**support_arrays); atomic_json(output/"support.json",support_json)
    atomic_json(output/"gates.json",result); atomic_json(output/"analysis.json",{"schema_version":"psps-v1-dev-analysis-v1","artifact_hashes":{name:file_hash(output/name) for name in ("gates.json","models.json","models.npz","support.json","support.npz")},"confirmation_authorized":result["confirmation_authorized"],"method_train_authorized":False}); return 0


if __name__=="__main__": raise SystemExit(main())
