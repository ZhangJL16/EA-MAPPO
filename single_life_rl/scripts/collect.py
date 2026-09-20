import json
from pathlib import Path
from single_life_rl.scripts.freeze_protocol import verify
from single_life_rl.scripts.io_utils import write_json, digest


def collect(output):
    output=Path(output); f=verify(output)
    library_ref=json.loads((output/'library_frozen.json').read_text())
    if digest(output/'library/library.json')!=library_ref['sha256']:raise RuntimeError('library changed')
    expected={j['job_id']:j for j in json.loads((output/'jobs.json').read_text())}
    rows=[];inventory=[]
    for suite in ('binary','recursive','nuisance','random','uav'):
        folders=sorted((output/suite).glob('worker_*'))
        assert len(folders)==f['config']['workers']
        for folder in folders:
            status=json.loads((folder/'status.json').read_text())
            if status['status']!='complete':raise RuntimeError('analysis before completion prohibited')
            for path in sorted(folder.glob('j*.json')):
                row=json.loads(path.read_text())
                if row['spec']!=expected[row['job_id']]:raise RuntimeError('job mismatch')
                rows.append(row);inventory.append(dict(path=str(path.relative_to(output)),sha256=digest(path)))
    ids=[r['job_id'] for r in rows]
    assert len(ids)==len(set(ids))==len(expected) and set(ids)==set(expected)
    aggregate=output/'aggregate';aggregate.mkdir(exist_ok=True)
    write_json(aggregate/'results.json',sorted(rows,key=lambda r:r['job_id']))
    write_json(aggregate/'inventory.json',inventory)
    write_json(aggregate/'integrity.json',dict(passed=True,expected=len(expected),unique=len(ids),missing=0,duplicates=0,
                    freeze_sha256=digest(output/'freeze.json'),library_sha256=library_ref['sha256']))
