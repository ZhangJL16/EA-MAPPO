"""One-time explicit migration of VALID trained checkpoints, NOT bad eval rows.

Only the known evaluator weight-loading bug is allowed as the source difference.
Model/objective/config versions must be byte-identical. Old directory preserved.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import torch
from bdr.run import binding, atomic_json, load_checkpoint_policy


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source",required=True)
    p.add_argument("--output",required=True)
    args=p.parse_args()
    source, output = Path(args.source), Path(args.output)
    manifest=json.loads((source/"manifest.json").read_text())
    old=manifest["binding"]
    cfg=old["config"]
    new=binding(cfg)
    expected_old="f41503be8014cb1ed0de4b4e5854083f371f257ad3e2f60b689f9a1baf164fcd"
    assert old["runtime"]["run.py"]==expected_old
    for k in old:
        if k!="runtime":
            assert old[k]==new[k]
    for k in old["runtime"]:
        if k!="run.py":
            assert old["runtime"][k]==new["runtime"][k]
    output.mkdir(exist_ok=False)
    hashes={}
    for seed in cfg["seeds"]:
        for kind in cfg["methods"]:
            file=source/f"{kind}_{seed}.pt"
            policy,saved=load_checkpoint_policy(file,cfg,"cpu")
            assert saved["binding"]==old
            for k,v in policy.state_dict().items():
                assert torch.equal(v,saved["state_dict"][k])
            shutil.copy2(file,output/file.name)
            hashes[file.name]=hashlib.sha256(file.read_bytes()).hexdigest()
    shutil.copy2(source/"manifest.json",output/"manifest.json")
    atomic_json(output/"evaluation_manifest.json",{"binding":new,"checkpoint_hashes":hashes,
        "reason":"Load trained state_dict before evaluation; original evaluations INVALID initialization-only",
        "original_directory":str(source),"training_unchanged":True,
        "timing_caveat":"Initial invalid evaluation overlapped end of last training seed; train latency is not isolated benchmark"})
    atomic_json(source/"INVALID_EVALUATION_NOTICE.json",{
        "scope":"ALL eval_*.json, cost_*.json and evaluation.json in THIS original directory INVALID for trained-policy claims",
        "cause":"Evaluator omitted policy.load_state_dict; reported initial networks",
        "valid_assets":"All 15 trained checkpoints and their training logs remain valid",
        "corrected_output":str(output),"no_retraining":True})


if __name__=="__main__":
    main()
