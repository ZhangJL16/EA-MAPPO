from pathlib import Path
import hashlib
import json
import os
import pickle


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n')
    os.replace(tmp, path)


def save_pickle(path, value):
    path = Path(path); tmp = path.with_suffix('.tmp')
    data = pickle.dumps(value, protocol=5)
    tmp.write_bytes(data); os.replace(tmp, path)
    return hashlib.sha256(data).hexdigest()


def source_inventory(root):
    return {str(p.relative_to(root)): digest(p) for p in sorted(root.rglob('*'))
            if p.is_file() and p.suffix in ('.py','.md','.json')
            and not any(x in p.parts for x in ('artifacts','evidence','__pycache__'))
            and p.name != 'freeze.json'}
