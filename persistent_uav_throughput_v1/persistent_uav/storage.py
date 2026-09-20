"""Atomic, checksummed, local resume snapshots; no model or optimizer in pickle."""
import hashlib
import json
import os
from pathlib import Path
import pickle
import tempfile
from contextlib import contextmanager
import fcntl


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _atomic_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as out:
            out.write(data)
            out.flush()
            os.fsync(out.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def write_json(path, value):
    _atomic_bytes(path, (json.dumps(value, indent=2, allow_nan=False) + '\n').encode())


def snapshot(folder, state):
    """Immutable generation + atomic pointer: crash cannot pair old bytes/new digest."""
    folder = Path(folder)
    payload = pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)
    digest = hashlib.sha256(payload).hexdigest()
    name = 'checkpoint_' + digest[:20] + '.pkl'
    _atomic_bytes(folder / name, payload)
    write_json(folder / 'resume.json', dict(file=name, sha256=digest))
    # Keep the current and one previous generation, never delete the active one.
    files = sorted(folder.glob('checkpoint_*.pkl'), key=lambda p: p.stat().st_mtime, reverse=True)
    for old in files[2:]:
        if old.name != name:
            old.unlink()
    return dict(file=name, sha256=digest, bytes=len(payload))


def restore(folder):
    folder = Path(folder)
    ref = json.loads((folder / 'resume.json').read_text())
    path = folder / ref['file']
    if path.parent != folder or sha(path) != ref['sha256']:
        raise RuntimeError('snapshot path or checksum mismatch')
    # This reader is only for this project's trusted local snapshots.
    return pickle.loads(path.read_bytes())


@contextmanager
def exclusive_run(folder):
    folder = Path(folder).resolve()
    folder.parent.mkdir(parents=True, exist_ok=True)
    # Lock lives beside the output so a fresh output can remain empty.
    with (folder.parent / ('.' + folder.name + '.lock')).open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('another process is already writing this run') from None
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
