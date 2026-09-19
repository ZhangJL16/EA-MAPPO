#!/usr/bin/env python3
"""Read-only local handoff inventory for EA-MAPPO.

Does not import repository modules, install dependencies, run tests/experiments,
resume jobs, access networks, traverse artifact directories, or read artifact
contents. It hashes an explicit allowlist of source files. Evidence paths are
checked by metadata only. The sole write is a NEW JSON report outside the repo.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from datetime import datetime, timezone

AUDITED_COMMIT = "fff2b026ca3e4ea6b68cacc3964971b0749bfeb3"
SOURCE_ALLOWLIST = (
    "AGENTS.md", "README.md", "task_plan.md", "notes.md", ".gitignore",
    "requirements.uv.txt", "pytest.ini",
    "experiments/bundling_calibration_core.py",
    "scripts/run_bundling_calibration.py",
    "scripts/analyze_bundling_calibration.py",
    "scripts/diagnose_bundling_allocation.py",
    "scripts/verify_bundling_finite_budget_theory.py",
    "scripts/audit_resource_separation_bridge.py",
    "tests/test_bundling_calibration.py",
    "docs/ICML_RESOURCE_SEPARATION_PAPER_CORE_20260916.md",
)
EVIDENCE_METADATA_ONLY = (
    "artifacts/bundling_calibration_startup_v1_20260916/manifest.json",
    "artifacts/bundling_calibration_startup_v1_20260916/sealed_predictions.json",
    "artifacts/bundling_calibration_startup_v1_20260916/health_4096.json",
    "artifacts/bundling_calibration_startup_v1_20260916/analysis_4096",
    "artifacts/bundling_finite_budget_theory_20260916/strict_benefit_certificate.json",
    "artifacts/resource_separation_bridge_20260916/audit_receipt.json",
)
PACKAGES = ("numpy", "scipy", "pytest", "torch", "gymnasium", "pyro-ppl", "networkx")


def command(args: list[str], cwd: Path | None = None) -> dict:
    env = os.environ.copy()
    env["GIT_OPTIONAL_LOCKS"] = "0"
    try:
        result = subprocess.run(args, cwd=cwd, env=env, check=False,
                                capture_output=True, text=True, timeout=20)
        return {"returncode": result.returncode, "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"returncode": None, "stdout": "", "stderr": str(exc)}


def relative_metadata(root: Path, relative: str, hash_source: bool) -> dict:
    if "confirm" in relative.lower():
        raise ValueError("CONFIRM paths are forbidden")
    path = root / relative
    row: dict = {"path": relative, "contents_read": False}
    # Do not follow any symlink components, including external artifact mounts.
    current = root
    for part in Path(relative).parts:
        current = current / part
        if current.is_symlink():
            row.update(exists=None, skipped="symlink component; not followed")
            return row
    try:
        info = path.stat()
    except FileNotFoundError:
        row["exists"] = False
        return row
    except OSError as exc:
        row.update(exists=None, error=str(exc))
        return row
    row.update(exists=True, is_file=path.is_file(), is_dir=path.is_dir(),
               size_bytes=info.st_size if path.is_file() else None)
    if hash_source and path.is_file():
        sha = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                sha.update(chunk)
        row.update(sha256=sha.hexdigest(), contents_read=True)
    return row


def inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True,
                        help="Local checkout; no clone or fetch is performed")
    parser.add_argument("--output", type=Path, required=True,
                        help="New JSON path OUTSIDE the checkout; parent must exist")
    parser.add_argument("--include-hardware", action="store_true",
                        help="Optionally run read-only nvidia-smi hardware query")
    args = parser.parse_args()
    root = args.repo.expanduser().resolve(strict=True)
    if not root.is_dir():
        parser.error("--repo must be a directory")
    output = args.output.expanduser().absolute()
    if not output.parent.exists():
        parser.error("output parent does not exist; create it explicitly first")
    if output.is_symlink() or output.exists():
        parser.error("output must be a new non-symlink file; refusing overwrite")
    if inside(output.resolve(), root):
        parser.error("write report outside the repo to preserve working tree")

    git = lambda *parts: command(["git", "--no-optional-locks", "-C", str(root), *parts])
    top = git("rev-parse", "--show-toplevel")
    if top["returncode"] != 0:
        parser.error("directory is not an accessible git checkout")
    if Path(top["stdout"]).resolve() != root:
        parser.error("--repo must point to the repository root")
    head = git("rev-parse", "HEAD")
    branch = git("symbolic-ref", "--short", "-q", "HEAD")
    status = git("status", "--porcelain=v1", "--untracked-files=no")
    base_exists = git("cat-file", "-e", f"{AUDITED_COMMIT}^{{commit}}")
    difference = (git("diff", "--name-status", AUDITED_COMMIT, "HEAD")
                  if base_exists["returncode"] == 0 else
                  {"returncode": None, "stdout": "", "stderr": "audited commit not in local object store"})

    evidence = list(EVIDENCE_METADATA_ONLY)
    for cap, on in ((3, True), (4, True), (3, False), (4, False)):
        for method in ("resource_path", "cost_aware_reference", "cost_blind", "independent_ucb", "oracle_allocation"):
            key = f"B{cap}_{'on' if on else 'off'}_{method}.checkpoint.json"
            evidence.append("artifacts/bundling_calibration_startup_v1_20260916/" + key)
    versions = {}
    for package in PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    report = {
        "schema": "ea-mappo-local-handoff-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "remote_audited_commit": AUDITED_COMMIT,
        "local_head": head, "branch": branch,
        "matches_audited_commit": head.get("stdout") == AUDITED_COMMIT,
        "tracked_working_tree_status": status,
        "committed_file_changes_since_audit": difference,
        "environment": {"python": sys.version, "platform": platform.platform(),
                        "logical_cpu_count": os.cpu_count(), "package_metadata": versions},
        "source_allowlist": [relative_metadata(root, p, True) for p in SOURCE_ALLOWLIST],
        "evidence_metadata_only": [relative_metadata(root, p, False) for p in evidence],
        "scope": {
            "repository_modules_imported": False, "tests_executed": False,
            "experiments_executed": False, "network_access": False,
            "artifact_contents_read": False, "confirm_paths_accessed": False,
            "untracked_directories_traversed": False,
            "warning": "File existence and versions do not certify reproducibility or hardware capacity."
        }
    }
    if args.include_hardware:
        report["gpu_metadata"] = command([
            "nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"
        ])
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"Wrote local metadata report: {output}")
    print("No project code, tests, simulations, training or CONFIRM data were accessed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
