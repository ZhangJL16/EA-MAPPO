from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_hash(root: str | Path) -> str:
    root_path = Path(root)
    digest = hashlib.sha256()
    paths = sorted(
        path
        for prefix in (
            "agents",
            "envs/navigation",
            "safety",
            "experiments/new_route",
            "experiments/energy_transfer",
            "experiments/navigation_scale",
            "scripts",
        )
        for path in (root_path / prefix).rglob("*.py")
    )
    for path in paths:
        digest.update(str(path.relative_to(root_path)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def git_sha(root: str | Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
