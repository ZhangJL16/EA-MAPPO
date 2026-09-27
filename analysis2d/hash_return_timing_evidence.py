"""Hash the complete 2D return-timing branch evidence and analysis sources."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

from .collect_return_timing_branches import collect as collect_first
from .collect_return_window import collect as collect_window


REPO = Path(__file__).resolve().parents[1]
OUTPUT = REPO / "evidence" / "return_timing_branch_cpu_20260927" / "EVIDENCE_MANIFEST.json"
SOURCES = (
    "RETURN_TIMING_BRANCH_PROTOCOL_20260927.md",
    "RETURN_TIMING_WINDOW_PROTOCOL_20260927.md",
    "RETURN_TIMING_WINDOW_AMENDMENT_20260927.md",
    "RETURN_TIMING_RESULT_20260927.md",
    "analysis2d/freeze_return_timing_roots.py",
    "analysis2d/branch_return_timing_cpu.py",
    "analysis2d/collect_return_timing_branches.py",
    "analysis2d/freeze_return_window_roots.py",
    "analysis2d/audit_return_window_roots.py",
    "analysis2d/branch_return_window_cpu.py",
    "analysis2d/branch_return_window_grouped_cpu.py",
    "analysis2d/collect_return_window.py",
    "analysis2d/hash_return_timing_evidence.py",
)


def main() -> None:
    first = collect_first()
    window = collect_window()
    if (first["primary_roots"], first["exploratory_roots"], window["root_count"]) != (48, 8, 210):
        raise ValueError("return-timing evidence is incomplete")
    paths = [REPO / source for source in SOURCES]
    for name in ("return_timing_branch_cpu_20260927", "return_timing_window_cpu_20260927"):
        paths.extend(sorted((REPO / "evidence" / name).rglob("*.json")))
    paths = [path for path in paths if path != OUTPUT]
    if len(paths) != len(set(paths)):
        raise ValueError("duplicate manifest member")
    hashes = {str(path.relative_to(REPO)): sha256(path.read_bytes()).hexdigest()
              for path in paths}
    payload = {
        "protocol": "return_timing_cpu_evidence_manifest_v1",
        "first_takeover_roots": first["primary_roots"],
        "early_voluntary_roots": first["exploratory_roots"],
        "frozen_window_roots": window["frozen_root_count"],
        "window_excluded_roots": len(window["excluded_roots"]),
        "valid_window_roots": window["root_count"],
        "member_count": len(hashes), "sha256": hashes,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print(f"hashed {len(hashes)} return-timing evidence members")


if __name__ == "__main__":
    main()
