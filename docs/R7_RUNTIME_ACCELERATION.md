# R7 Exact-Semantics Runtime Acceleration

Date: 2026-09-01

## Scope

This change accelerates the sampled-data robust HOCBF execution layer used by
R7. It does not alter PPO, the observation or reward contract, the LiDAR
resolution, the physical substep, the HOCBF gains, the sampled-data residual,
the actuator constraints, or fallback behavior.

## Implemented path

1. Static LiDAR hit points are kept as arrays. Ordinary HOCBF terms and the
   sampled-data residual bounds are evaluated in batches, eliminating one
   `SphericalObstacle` allocation and two Python constraint loops per hit.
2. Large three-variable projection QPs first use exact constraint generation.
   A restricted optimum is accepted only after every omitted inequality is
   checked. If the restricted solve is inconclusive, execution falls back to
   the established full solver.
3. The established full Hildreth/Dykstra sweep is reproduced in a small C
   kernel. Python retains validation, matrix inversion, degenerate-row checks,
   feasibility thresholds, reason codes, diagnostics, and emergency braking.
   If the optional shared library is absent or stale, the Python reference path
   remains active.

The native kernel is rebuilt with:

```bash
uv run --no-project --python .venv/bin/python \
  python scripts/build_hocbf_native.py
```

## Exactness argument for constraint generation

Let the full feasible set be `F` and a working-set relaxation be `F_W`, so
`F` is a subset of `F_W`. If `u_W` minimizes the strictly convex projection
objective over `F_W` and the full inequality scan verifies `u_W` is in `F`,
then every `u` in `F` also belongs to `F_W`. Therefore the objective at `u_W`
is no greater than the objective at any `u` in `F`, and `u_W` is the full-QP
optimum. No heuristic top-k truncation is used.

## Verification

- Native large-QP and Python reference paths agree on feasibility,
  convergence, iteration count, active-constraint count, reason, maximum
  violation, and acceleration within `1e-11`.
- Object and array sampled-data LiDAR paths agree on the executed acceleration
  and all non-timing diagnostics.
- Collision-filter/projection suite: 80 passed.
- UAV environment and R7 trainer/checkpoint suite: 96 passed.

## Deterministic benchmark

On fixed near-obstacle states with 232--472 LiDAR constraints and four physics
substeps, the previous policy-step wall time was approximately 78--236 ms.
The optimized implementation measured approximately 8--18 ms on the same class
of states. Full robust-QP solver time fell from approximately 74--231 ms to
5--15 ms. These are deterministic engineering benchmarks, not formal training
results; end-to-end gain depends on the learned policy's state occupancy.

## Activation boundary

Already running Python worker processes retain their imported implementation.
The optimized path becomes active only in newly started/resumed workers. A run
must not be relabeled as optimized unless its process started after the native
library build and its source provenance records the new files.
