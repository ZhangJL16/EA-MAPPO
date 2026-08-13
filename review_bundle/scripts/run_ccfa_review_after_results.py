#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    root = Path("artifacts/new_route/e1_energy")
    completed = sorted(root.glob("seed*/COMPLETED.json"))
    if len(completed) < 3:
        raise RuntimeError("results-aware CCFA review is blocked until at least three seeds complete")
    packet = {
        "status": "READY_FOR_RESULTS_AWARE_CCFA_REVIEW",
        "completed_markers": [str(path) for path in completed],
        "required_skills": ["ccf-paper-reviewer", "ccf-rebuttal-writer"],
        "inputs": [
            "docs/new_theory/CLAIM_EVIDENCE_LEDGER.md",
            "docs/new_theory/NOVELTY_MATRIX.md",
            "artifacts/new_route/e1_energy_aggregate.json",
            "artifacts/new_route/e1_energy_table.md",
        ],
    }
    Path("artifacts/new_route/RESULTS_REVIEW_PACKET.json").write_text(json.dumps(packet, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
