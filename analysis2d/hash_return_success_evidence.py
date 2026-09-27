"""Hash complete successful-task negative-control evidence."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

from .collect_return_success_controls import collect


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence" / "return_success_control_cpu_20260927"
OUTPUT = EVIDENCE / "EVIDENCE_MANIFEST.json"
SOURCES = (
    "RETURN_SUCCESS_CONTROL_PROTOCOL_20260927.md",
    "RETURN_SUCCESS_CONTROL_RESULT_20260927.md",
    "analysis2d/freeze_return_success_roots.py",
    "analysis2d/branch_return_success_controls_cpu.py",
    "analysis2d/collect_return_success_controls.py",
    "analysis2d/audit_return_state_overlap.py",
    "analysis2d/hash_return_success_evidence.py",
)


def main() -> None:
    summary = collect()
    if summary["root_count"] != 97 or summary["source_parent_count"] != 34:
        raise ValueError("incomplete successful-task negative-control evidence")
    paths = [ROOT / source for source in SOURCES]
    paths.extend(sorted(path for path in EVIDENCE.rglob("*.json") if path != OUTPUT))
    if len(paths) != len(set(paths)):
        raise ValueError("duplicate evidence member")
    hashes = {str(path.relative_to(ROOT)): sha256(path.read_bytes()).hexdigest()
              for path in paths}
    OUTPUT.write_text(json.dumps({
        "protocol": "return_success_control_cpu_evidence_manifest_v1",
        "source_maps": summary["source_maps"],
        "no_completion_maps": summary["no_completion_map_count"],
        "parent_trajectories": summary["source_parent_count"],
        "branches": summary["root_count"],
        "member_count": len(hashes), "sha256": hashes,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"hashed {len(hashes)} success-control members")


if __name__ == "__main__":
    main()
