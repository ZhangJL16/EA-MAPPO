"""Calibrated backup CBF with verified retry on OSQP iteration exhaustion."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import numpy as np
import osqp
from scipy import sparse

from nav3d_v2.controller import SegmentTrackingController

from . import return_audit_v2, return_audit_v3


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "m100_1km_retrospective_executable_return_v4_backup_cbf_verified_qp_retry"
SOURCE_FILES = return_audit_v3.SOURCE_FILES + ("delivery_1km/return_audit_v4.py",)
_retry_count = 0


def source_hash() -> str:
    digest = sha256()
    for relative in SOURCE_FILES:
        digest.update(relative.encode() + b"\0")
        digest.update((ROOT / relative).read_bytes() + b"\0")
    return digest.hexdigest()


def build_manifest(navigation: Path, job_id: str) -> dict:
    manifest = return_audit_v3.build_manifest(navigation, job_id)
    manifest["protocol"] = PROTOCOL
    manifest["source_sha256"] = source_hash()
    manifest["qp_retry"] = {"trigger": "maximum iterations reached",
                            "max_iter": 20000, "eps_abs": 1e-7,
                            "eps_rel": 1e-7, "polishing": True}
    return manifest


class RetryController(SegmentTrackingController):
    @staticmethod
    def _solve(nominal, rows, lower, upper):
        global _retry_count
        projected, elapsed, reason = SegmentTrackingController._solve(nominal, rows, lower, upper)
        if projected is not None or reason != "maximum iterations reached":
            return projected, elapsed, reason
        _retry_count += 1
        solver = osqp.OSQP()
        solver.setup(P=sparse.eye(3, format="csc"), q=-nominal,
                     A=sparse.csc_matrix(rows), l=lower, u=upper,
                     verbose=False, polishing=True, eps_abs=1e-7,
                     eps_rel=1e-7, max_iter=20000)
        result = solver.solve(raise_error=False)
        if (result.x is None or result.info.status not in ("solved", "solved inaccurate")
                or not np.all(np.isfinite(result.x))
                or np.any(rows @ result.x < lower - 2e-5)
                or np.any(rows @ result.x > upper + 2e-5)):
            return None, elapsed, f"retry_{result.info.status}"
        return np.asarray(result.x, dtype=np.float64), elapsed, f"retry_{result.info.status}"


def simulate_return(world, position, velocity, station):
    original = return_audit_v3.SegmentTrackingController
    try:
        return_audit_v3.SegmentTrackingController = RetryController
        return return_audit_v3.simulate_return(world, position, velocity, station)
    finally:
        return_audit_v3.SegmentTrackingController = original


def audit_state(navigation: Path, output: Path, manifest: dict, step_index: int) -> dict:
    global _retry_count
    _retry_count = 0
    original = return_audit_v2.simulate_return
    try:
        return_audit_v2.simulate_return = simulate_return
        row = return_audit_v2.audit_state(navigation, output, manifest, step_index)
    finally:
        return_audit_v2.simulate_return = original
    row["return_qp_retry_count"] = _retry_count
    return row
