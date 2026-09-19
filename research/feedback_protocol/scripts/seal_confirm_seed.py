"""Commit future seed material without generating/opening final instances.

Local custody only: this is not independent external blinding, nor a frozen
final protocol. The method, distributions and evaluation must be frozen later
before the custodian releases this seed material. Never print the preimage.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets


def seal(private,public):
    private,public = Path(private).resolve(),Path(public).resolve()
    repo = Path(__file__).resolve().parents[3]
    if private.is_relative_to(repo) or private==public:
        raise ValueError("private seed custody must be outside repository")
    if private.exists() or public.exists():
        raise ValueError("refuse to replace commitment or private custody")
    material = dict(schema="fpl-confirm-seed-material-v1",master_seed=secrets.token_hex(32),nonce=secrets.token_hex(16))
    data = json.dumps(material,sort_keys=True,separators=(",",":")).encode()
    commitment = dict(schema="fpl-confirm-seed-commitment-v1",sha256=hashlib.sha256(data).hexdigest(),
        derivation="After method/protocol freeze: HMAC-SHA256(master_seed, axis || index), specified in final protocol.",
        axes=["iid","size","topology","cost","hypothesis"],
        status="seed material committed; NO instances generated; final protocol NOT frozen",
        custody="local user-controlled file outside repository; NOT independently externally blinded")
    fd = os.open(private,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,"wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    with public.open("x") as f:
        json.dump(commitment,f,sort_keys=True,indent=2)
        f.write("\n")
    print(json.dumps(commitment))


if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--private",required=True)
    p.add_argument("--public",required=True)
    a = p.parse_args()
    seal(a.private,a.public)
