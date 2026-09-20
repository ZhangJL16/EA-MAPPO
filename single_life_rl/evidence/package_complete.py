"""Post-completion packaging only; never runs or selects an experiment."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tarfile

root=Path(sys.argv[1]).resolve()
out=Path(__file__).resolve().parent/'complete_v1'
status=json.loads((root/'status.json').read_text())
assert status['status']=='complete'
integrity=json.loads((root/'aggregate/integrity.json').read_text())
assert integrity['passed'] and integrity['unique']==47000
out.mkdir(parents=True,exist_ok=True)
for name in ('REPORT.md','summary.json','summary.csv','integrity.json','scaling.png','scaling.pdf'):
    shutil.copyfile(root/'aggregate'/name,out/name)
for name in ('freeze.json','startup_health.json','library_frozen.json','status.json'):
    shutil.copyfile(root/name,out/name)
shutil.copyfile(root/'library/library.json',out/'mission_library.json')
files=sorted(p for p in root.rglob('*') if p.is_file() and p.suffix not in ('.lock','.tmp'))
inventory=[]
archive=out/'raw_run.tar.gz'
with tarfile.open(archive,'w:gz',compresslevel=6) as tar:
    for p in files:
        inventory.append(dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
        tar.add(p,arcname=str(p.relative_to(root)),recursive=False)
with tarfile.open(archive,'r:gz') as tar:
    members=tar.getmembers()
    assert [x.name for x in members]==[x['path'] for x in inventory]
    for member,record in zip(members,inventory):
        assert hashlib.sha256(tar.extractfile(member).read()).hexdigest()==record['sha256']
(out/'raw_inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
(out/'archive_verification.json').write_text(json.dumps(dict(passed=True,files=len(files),bytes=archive.stat().st_size,
   sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),members_checked_individually=True),indent=2)+'\n')
(out/'SHA256SUMS').write_text(''.join(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n' for p in sorted(out.iterdir()) if p.is_file() and p.name!='SHA256SUMS'))
print(json.dumps(dict(archive_bytes=archive.stat().st_size,files=len(files),passed=True)))
