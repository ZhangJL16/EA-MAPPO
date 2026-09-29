"""Explicit, auditable removal of noncurrent top-level research routes.

The retained active simulator and model are checked against provenance before
and after deletion. The manifest records hashes, not a backup of deleted data.
"""

import argparse
from datetime import datetime, timezone
import gzip
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import sys


ROOT = Path(__file__).resolve().parents[3]
OLD = ROOT / "persistent_uav_throughput_v1"
sys.path.insert(0, str(OLD))
from persistent_uav.provenance import provenance  # noqa: E402

OUT = ROOT / "docs/cleanup_20260929"
PLAN = OUT / "noncurrent_route_deletion_manifest.json.gz"
STATUS = OUT / "status.json"

REMOVE_TOP = (
    "agent", "archive", "calibration", "cert_runtime", "common",
    "expert_advice", "innovation", "network", "policy", "research",
    "scripts", "single_life_rl", "tests", "theory",
    "DERIVATION_PACKAGE.md", "FULL_PROBLEM_AUDIT.md",
    "main.py", "main_energy_delivery_auction.py", "runner.py",
    "LAMDA_innovation_atlas_2023_2026.html",
    "LAMDA_papers_and_innovation_methods_2023_2026.xlsx",
)
KEEP_EXPERIMENTS = {"directional_navigation", "jacobian_energy_bridge"}
KEEP_ENVS = {"UAVEnergyDelivery.py", "UAVEnergyDeliverySAC.py"}
KEEP_ARTIFACTS = {"hocbf_correction_sac_20260908_v1", "zotero_thesis_20260922"}
KEEP_DOCS = {
    "LOCKED_COLLISION_RECOVERY_PROTOCOL.md", "cleanup_20260920",
    "cleanup_20260929", "project_history_20260920",
    "energy_safety_dialogue_20260922", "learning_planning_literature_20260923",
    "literature_20260922",
}


def targets():
    items = [ROOT / name for name in REMOVE_TOP]
    for parent, keep in (("experiments", KEEP_EXPERIMENTS),
                         ("envs", KEEP_ENVS),
                         ("artifacts", KEEP_ARTIFACTS),
                         ("docs", KEEP_DOCS)):
        items.extend(child for child in sorted((ROOT / parent).iterdir())
                     if child.name not in keep)
    return [x for x in items if x.exists() or x.is_symlink()]


def hash_file(path):
    digest = sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def entries_for(root):
    if root.is_symlink():
        yield dict(path=str(root.relative_to(ROOT)), type="symlink", target=os.readlink(root))
    elif root.is_file():
        yield dict(path=str(root.relative_to(ROOT)), type="file",
                   bytes=root.stat().st_size, sha256=hash_file(root))
    elif root.is_dir():
        for current, directories, files in os.walk(root, followlinks=False):
            base = Path(current)
            for name in sorted(directories + files):
                path = base / name
                if path.is_symlink():
                    yield dict(path=str(path.relative_to(ROOT)), type="symlink",
                               target=os.readlink(path))
                elif path.is_file():
                    yield dict(path=str(path.relative_to(ROOT)), type="file",
                               bytes=path.stat().st_size, sha256=hash_file(path))


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    os.replace(temp, path)


def plan():
    if PLAN.exists() or STATUS.exists():
        raise RuntimeError("cleanup plan already exists; use --execute")
    before = provenance()
    chosen = targets()
    files = [row for target in chosen for row in entries_for(target)]
    payload = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                   selected_roots=[str(x.relative_to(ROOT)) for x in chosen],
                   files=files, file_count=sum(x["type"] == "file" for x in files),
                   total_bytes=sum(x.get("bytes", 0) for x in files),
                   retained_provenance=before,
                   recovery_limit="SHA256 list is not a backup; untracked deleted data cannot be restored from it.")
    OUT.mkdir(parents=True, exist_ok=True)
    with gzip.open(PLAN, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, ensure_ascii=False)
    atomic_json(STATUS, dict(status="planned", roots=payload["selected_roots"],
                             file_count=payload["file_count"], total_bytes=payload["total_bytes"],
                             manifest_sha256=hash_file(PLAN)))
    print(json.dumps(dict(status="planned", roots=len(chosen), files=payload["file_count"],
                          bytes=payload["total_bytes"])))


def execute():
    with gzip.open(PLAN, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    state = json.loads(STATUS.read_text())
    if state["status"] != "planned" or state["manifest_sha256"] != hash_file(PLAN):
        raise RuntimeError("cleanup manifest/status mismatch")
    if provenance() != payload["retained_provenance"]:
        raise RuntimeError("active runtime provenance changed before deletion")
    if sorted(state["roots"]) != sorted(str(x.relative_to(ROOT)) for x in targets()):
        raise RuntimeError("target set changed since plan")
    for row in payload["files"]:
        path = ROOT / row["path"]
        if row["type"] == "symlink":
            if not path.is_symlink() or os.readlink(path) != row["target"]:
                raise RuntimeError(f"symlink changed: {path}")
        elif not path.is_file() or path.stat().st_size != row["bytes"] or hash_file(path) != row["sha256"]:
            raise RuntimeError(f"file changed: {path}")
    atomic_json(STATUS, {**state, "status": "deleting"})
    for relative in payload["selected_roots"]:
        path = ROOT / relative
        if path.is_symlink() or path.is_file():
            path.unlink()
        else:
            shutil.rmtree(path)
    if provenance() != payload["retained_provenance"]:
        raise RuntimeError("active runtime provenance changed after deletion")
    atomic_json(STATUS, {**state, "status": "deleted",
                         "deleted_utc": datetime.now(timezone.utc).isoformat()})
    print(json.dumps(dict(status="deleted", roots=len(payload["selected_roots"]),
                          files=payload["file_count"], bytes=payload["total_bytes"])))


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    plan() if args.plan else execute()


if __name__ == "__main__":
    main()
