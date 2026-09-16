#!/usr/bin/env python3
"""Loader-only execution repair for the frozen paired-advantage DEV analyzer.

The frozen PAI analyzer imports the predecessor CMI-v3 loader, whose only
incompatible precondition is a hard-coded predecessor contract ID. This wrapper
binds every transitive analysis source, validates the PAI-v3 ID and its original
analyzer binding, adapts only that ID on an ephemeral private copy, and then
executes the byte-identical frozen PAI Gate H/I main function.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.analyze_cmi_v3_dev as predecessor
import scripts.analyze_paired_advantage_dev as frozen
from scripts.validate_identification_contract import file_hash


PAI_CONTRACT_ID = "paired-advantage-identification-20260914-v3"
PREDECESSOR_CONTRACT_ID = "conditional-history-mc-q-20260912-v3"
BOUND_SOURCES = {
    "scripts/analyze_paired_advantage_dev.py":
        "sha256:e17b1de87615df961a47ccce9eb57779a6116356b8432bab0da9bcba2df1c41e",
    "scripts/analyze_cmi_v3_dev.py":
        "sha256:2e615008de3011dba738bd15164bbcfc78138816f42391a1bdae5820712cb70f",
    "scripts/analyze_cmi_paired_advantage_exploratory.py":
        "sha256:312ea758400940bef915ef7da810f2b2b81a64d80485eb699e7caa800ab82ffc",
    "scripts/cmi_v3_models.py":
        "sha256:83386aabcc754781cb949155e030521c3023249cf294b7d9c3f9e609f5240800",
    "scripts/validate_identification_contract.py":
        "sha256:be672bde6bd32cd0b492fffd8564040de5d14fc89f78bfe5bbc74950ba79977a",
}


def validate_bound_sources() -> None:
    for relative, expected in BOUND_SOURCES.items():
        actual = file_hash(ROOT / relative)
        if actual != expected:
            raise ValueError(
                f"analysis repair source binding changed: {relative}: "
                f"expected {expected}, found {actual}"
            )


def load_pai_dev(contract_path: Path, run_dir: Path):
    """Run the predecessor loader after adapting only its obsolete ID guard."""
    validate_bound_sources()
    contract = json.loads(contract_path.read_text())
    pre = contract.get("pre_h", {})
    if pre.get("contract_id") != PAI_CONTRACT_ID:
        raise ValueError("not the frozen paired-advantage v3 contract")
    if pre.get("source_hashes", {}).get("dev_analyzer_v3") != BOUND_SOURCES[
        "scripts/analyze_paired_advantage_dev.py"
    ]:
        raise ValueError("PAI contract does not bind the frozen DEV analyzer")

    proxy = copy.deepcopy(contract)
    proxy["pre_h"]["contract_id"] = PREDECESSOR_CONTRACT_ID
    with tempfile.TemporaryDirectory(prefix="pai-loader-repair-") as temporary:
        proxy_path = Path(temporary) / "contract.json"
        proxy_path.write_text(
            json.dumps(proxy, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        data = predecessor.load_dev(proxy_path, run_dir)
    data["contract"] = contract
    return data


def main() -> int:
    frozen.load_dev = load_pai_dev
    return frozen.main()


if __name__ == "__main__":
    raise SystemExit(main())
