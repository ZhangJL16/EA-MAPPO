"""Build a current-route-only server transfer bundle from the cleaned tree."""

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import tarfile


ROOT = Path(__file__).resolve().parents[3]
PROJECT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT / "migration"
BUNDLE = OUTPUT / "persistent_uav_current_route_20260929.tar.gz"
MANIFEST = OUTPUT / "bundle_manifest.json"

ROOT_FILES = (
    "AGENTS.md", "README.md", "PROJECT_CLOSEOUT.md", "task_plan.md", "notes.md",
    "MINIMAL_PERSISTENT_THROUGHPUT_SPEC.md", "requirements.txt",
    "requirements.uv.txt", "uv.toml", ".gitignore", ".ignore",
    "envs/UAVEnergyDelivery.py", "envs/UAVEnergyDeliverySAC.py",
    "artifacts/hocbf_correction_sac_20260908_v1/sac/checkpoint_000131072/model.zip",
    "review_bundle",
)
ROOT_DIRS = (
    "persistent_uav_throughput_v1", "experiments/directional_navigation",
    "experiments/jacobian_energy_bridge", "runtime_support/review_bundle",
    "docs/cleanup_20260920", "docs/cleanup_20260929",
    "docs/energy_safety_dialogue_20260922",
    "docs/learning_planning_literature_20260923", "docs/literature_20260922",
    "docs/project_history_20260920",
    "research_projects/persistent_uav_feasibility_20260929",
)


def digest(path):
    h = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def selected_files():
    paths = [ROOT / x for x in ROOT_FILES]
    for relative in ROOT_DIRS:
        directory = ROOT / relative
        for path in directory.rglob("*"):
            parts = path.relative_to(ROOT).parts
            if ("__pycache__" in parts or ".pytest_cache" in parts
                    or "migration" in parts or path.suffix in (".tmp", ".lock")):
                continue
            if parts[:4] == ("research_projects", "persistent_uav_feasibility_20260929",
                             "runs", "full_window_pilot_v1"):
                continue  # Mutable x86 episode rows are a separate later evidence transfer.
            if path.is_file() or path.is_symlink():
                paths.append(path)
    return sorted(set(paths), key=lambda path: str(path.relative_to(ROOT)))


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if BUNDLE.exists() or MANIFEST.exists():
        raise FileExistsError("bundle already exists; do not silently replace it")
    paths = selected_files()
    entries = []
    for path in paths:
        if path.is_symlink():
            entries.append(dict(path=str(path.relative_to(ROOT)), type="symlink",
                                target=path.readlink().as_posix()))
        elif path.is_file():
            entries.append(dict(path=str(path.relative_to(ROOT)), type="file",
                                bytes=path.stat().st_size, sha256=digest(path)))
        else:
            raise RuntimeError(f"missing selected path: {path}")
    payload = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                   kind="current_route_server_transfer_static_snapshot",
                   files=entries, file_count=sum(x["type"] == "file" for x in entries),
                   bytes=sum(x.get("bytes", 0) for x in entries),
                   excluded="Active runs/full_window_pilot_v1 is not included; it continues on x86.",
                   cross_platform="ARM must generate a new manifest; x86 absolute-path/package provenance prevents direct resume.")
    MANIFEST.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    with tarfile.open(BUNDLE, "w:gz", dereference=False) as archive:
        archive.add(MANIFEST, arcname=str(MANIFEST.relative_to(ROOT)), recursive=False)
        for path in paths:
            archive.add(path, arcname=str(path.relative_to(ROOT)), recursive=False)
    print(json.dumps(dict(bundle=str(BUNDLE), sha256=digest(BUNDLE),
                          file_count=payload["file_count"], source_bytes=payload["bytes"],
                          bundle_bytes=BUNDLE.stat().st_size)))


if __name__ == "__main__":
    main()
