"""Freeze before any production outcome; verify source, dependencies and jobs."""
import argparse
import json
from pathlib import Path
import subprocess
from single_life_rl.scripts.io_utils import digest, source_inventory, write_json
from single_life_rl.scripts.jobs import generate

ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT.parent


def dependencies():
    paths=list((REPO/'persistent_uav_throughput_v1/persistent_uav').glob('*.py'))
    # Frozen navigator has transitive simulator dependencies outside its package.
    for directory in ('envs','experiments/directional_navigation','runtime_support'):
        paths.extend((REPO/directory).rglob('*.py'))
    paths += [REPO/'runtime_support/review_bundle/safety/collision/_qp_native.so',
              REPO/'artifacts/hocbf_correction_sac_20260908_v1/sac/checkpoint_000131072/model.zip',
              REPO/'persistent_uav_throughput_v1/evidence/v1_2/calibration/frozen_regimes.json']
    return {str(p.relative_to(REPO)):digest(p) for p in sorted(set(paths))}


def freeze(config_path, output):
    output=Path(output).resolve(); output.mkdir(parents=True,exist_ok=True)
    if (output/'freeze.json').exists(): raise FileExistsError('freeze already exists; use verify/resume')
    config_path=Path(config_path).resolve(); config=json.loads(config_path.read_text())
    jobs=generate(config); write_json(output/'jobs.json',jobs)
    record=dict(config=config,config_path=str(config_path),config_sha256=digest(config_path),
                git_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
                sources=source_inventory(ROOT),dependencies=dependencies(),jobs_sha256=digest(output/'jobs.json'),jobs=len(jobs),
                safety='lifetime delta; no reset of statistical risk on regeneration',
                execution='fixed ordered suites; aggregate analysis only after all jobs; checkpoints allowed')
    write_json(output/'freeze.json',record)
    return record


def verify(output):
    output=Path(output); f=json.loads((output/'freeze.json').read_text())
    if digest(f['config_path']) != f['config_sha256']: raise RuntimeError('config changed after freeze')
    if source_inventory(ROOT) != f['sources']: raise RuntimeError('research source changed after freeze')
    if dependencies() != f['dependencies']: raise RuntimeError('physical dependency changed after freeze')
    if digest(output/'jobs.json') != f['jobs_sha256']: raise RuntimeError('job plan changed after freeze')
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()!=f['git_sha']: raise RuntimeError('Git SHA changed after freeze')
    return f


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();print(json.dumps({'jobs':freeze(a.config,a.output)['jobs']}))
