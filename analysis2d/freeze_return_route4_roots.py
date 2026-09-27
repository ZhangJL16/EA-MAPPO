"""Freeze all lag-10/20 takeover roots for the simple route-4 alternative."""

from __future__ import annotations

import json

from .collect_shield_feedback_cpu import read_json
from .audit_return_window_roots import audit
from .freeze_return_timing_roots import ROOT, digest
from .freeze_return_window_roots import OUTPUT as WINDOW_ROOTS


OUTPUT = ROOT / "evidence" / "return_route4_alternative_cpu_20260927" / "roots.json"
WINDOW_BRANCHES = WINDOW_ROOTS.parent / "branches"
LAGS = (10, 20)


def freeze() -> dict:
    source = read_json(WINDOW_ROOTS)
    valid = set(audit()["valid_root_ids"])
    roots = []
    for window_root in source["roots"]:
        if window_root["id"] not in valid or window_root["lag_decisions"] not in LAGS:
            continue
        branch_path = WINDOW_BRANCHES / f"{window_root['id']}.json"
        branch = read_json(branch_path)
        if (branch["root"] != window_root
                or branch["roots_sha256"] != digest(WINDOW_ROOTS)):
            raise ValueError(f"prior window branch provenance differs: {branch_path}")
        roots.append({
            **window_root,
            "id": "route4_" + window_root["id"],
            "window_root_id": window_root["id"],
            "window_branch_sha256": digest(branch_path),
        })
    if (len(roots) != 86 or len({r["id"] for r in roots}) != 86
            or sum(r["lag_decisions"] == 10 for r in roots) != 44
            or sum(r["lag_decisions"] == 20 for r in roots) != 42):
        raise ValueError("unexpected route-4 alternative root selection")
    return {
        "protocol": "return_route4_alternative_cpu_v1",
        "protocol_sha256": digest(ROOT / "RETURN_ROUTE4_ALTERNATIVE_PROTOCOL_20260927.md"),
        "source_window_roots_sha256": digest(WINDOW_ROOTS),
        "root_count": len(roots), "lags": list(LAGS), "roots": roots,
    }


if __name__ == "__main__":
    payload = freeze()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT.exists() and read_json(OUTPUT) != payload:
        raise ValueError("frozen route-4 roots changed")
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print(f"froze {payload['root_count']} route-4 alternative roots")
