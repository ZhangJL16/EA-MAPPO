"""Package frozen admission, pilot raw data, audit, and exact runtime bytes."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import zipfile
from fpl.evaluation import digest
from fpl_v2.runtime import closure

ROOT = Path(__file__).resolve().parents[1]


def package(admission, data, audit, output):
    out=Path(output)
    if out.exists():raise FileExistsError(out)
    admission,data,audit=map(Path,(admission,data,audit))
    seal=json.loads((data/'seal.json').read_text())
    summary=json.loads((audit/'summary.json').read_text())
    if seal['source_hashes']!=closure() or summary['seal_hash']!=digest(seal):
        raise ValueError('cannot package changed runtime or wrong audit')
    entries={}
    for label,base in [('admission',admission),('data',data),('audit',audit)]:
        for p in sorted(base.iterdir()):
            if p.is_file() and p.suffix in ('.json','.md'):
                entries[f'{label}/{p.name}']=p.read_bytes()
    for pkg in ('fpl','fpl_v2'):
        for p in sorted((ROOT/'src'/pkg).rglob('*.py')):
            entries['runtime/'+str(p.relative_to(ROOT/'src'))]=p.read_bytes()
    for name in ('configs/teacher_v2_design_v1.json','configs/teacher_v2_design_v1.sha256.json',
                 'configs/teacher_v1/exclusions.json','configs/teacher_v1/admission.json',
                 'CONTRACT_T51C_V2.md','scripts/audit_teacher_v2.py','scripts/package_teacher_v2.py',
                 'scripts/export_teacher_targets.py','tests/test_teacher_v2.py'):
        entries['source/'+name]=(ROOT/name).read_bytes()
    out.mkdir(parents=True)
    archive=out/'teacher_v2_sizing_evidence.zip'
    hashes={name:sha256(content).hexdigest() for name,content in entries.items()}
    entries['FILES_SHA256.json']=json.dumps(hashes,sort_keys=True,indent=2).encode()
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for name,content in sorted(entries.items()):
            info=zipfile.ZipInfo(name,(2026,9,19,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
            z.writestr(info,content)
    for name in ('summary.json','RESULTS.md'):
        (out/name).write_bytes((audit/name).read_bytes())
    receipt=dict(schema='teacher-v2-evidence-v1',archive=archive.name,
                 sha256=sha256(archive.read_bytes()).hexdigest(),bytes=archive.stat().st_size,
                 admission_hash=summary['admission_hash'],seal_hash=summary['seal_hash'],
                 files=hashes,raw_sources=dict(admission=str(admission),data=str(data),audit=str(audit)))
    (out/'manifest.json').write_text(json.dumps(receipt,sort_keys=True,indent=2)+'\n')
    print(json.dumps({k:receipt[k] for k in ('archive','sha256','bytes','admission_hash','seal_hash')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('admission','data','audit','output'):p.add_argument('--'+k,required=True)
    a=p.parse_args();package(a.admission,a.data,a.audit,a.output)
