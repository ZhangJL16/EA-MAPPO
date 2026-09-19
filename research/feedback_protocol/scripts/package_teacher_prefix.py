"""Verify and package an already-completed target export; no solver access."""
from collections import defaultdict
from fractions import Fraction as F
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--data',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--design',required=True)
    args=parser.parse_args()
    data,out,design=map(Path,(args.data,args.output,args.design))
    if out.exists():raise FileExistsError('new package directory required')
    manifest=json.loads((data/'manifest.json').read_text())
    if manifest['status']!='complete':raise ValueError('incomplete export')
    for name,digest in manifest['files'].items():
        if hashlib.sha256((data/name).read_bytes()).hexdigest()!=digest:raise ValueError('file hash mismatch')
    states=[json.loads(x) for x in (data/'states.jsonl').read_text().splitlines()]
    prefixes=[json.loads(x) for x in (data/'prefixes.jsonl').read_text().splitlines()]
    ids={s['id'] for s in states}
    weights=defaultdict(F)
    for p in prefixes:
        if p['state_id'] not in ids:raise ValueError('dangling prefix')
        weights[p['state_id']]+=F(p['policy_weight'])
        if any(F(v)>0 for v in p['advantage'].values()):raise ValueError('positive advantage')
    if set(weights)!=ids or any(w!=1 for w in weights.values()):raise ValueError('state-weight mismatch')
    design_data=json.loads(design.read_text())
    if design_data['status']!='design_frozen_implementation_pending':raise ValueError('unexpected V2 stage')
    commitment=dict(file=design.name,sha256=hashlib.sha256(design.read_bytes()).hexdigest(),
                    scope='Design bytes only; no V2 implementation, problem generation or solver run')
    out.mkdir(parents=True)
    shutil.copy2(data/'manifest.json',out/'manifest.json')
    (out/'teacher_v2_design_commitment.json').write_text(json.dumps(commitment,indent=2)+'\n')
    archive=out/'teacher_prefix_v1.zip'
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for name in sorted(manifest['files']):z.write(data/name,'targets/'+name)
        z.write(data/'manifest.json','targets/manifest.json')
        z.write(Path(__file__).with_name('export_teacher_targets.py'),'scripts/export_teacher_targets.py')
        z.write(Path(__file__),'scripts/package_teacher_prefix.py')
        z.write(design,'design/'+design.name)
    audit=dict(status='passed',states=len(states),prefixes=len(prefixes),
               all_state_policy_weights_one=True,all_advantages_nonpositive=True,
               archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
               archive_bytes=archive.stat().st_size,design_commitment=commitment)
    (out/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
