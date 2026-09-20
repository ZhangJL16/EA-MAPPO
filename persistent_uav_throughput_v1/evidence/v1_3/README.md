# v1.3 dock-idle semantics and diagnostic authorization

2026-09-20 user approved v1.2, calibration and navigation qualification for
scheduling diagnostics, conditional on this dock-idle repair.

- Dock idle: zero energy use, no automatic charging; same existing goal-radius station tolerance.
- Away idle: original hover energy, including possible depletion.
- Explicit recharge retains positive duration and net charge rate.
- 29/29 focused tests pass (`tests.txt`). Frozen physical-flight regression passes.
- Real frozen-SAC continuing trace: serve → serve → recharge → dock idle → serve.
  Dock idle t=31.85→200.0, energy=100→100, next task arrival processed, physical resets=1.
  Full outcome after midflight restore matches the uninterrupted branch. Original evidence preserved.
- The original 1000-job calibration was NOT rerun. Frozen manifest SHA256 remains
  `a2b74f9ff84c83377ea081d0cda9610b635b739e6e178a0132bc1106c85f9f19`.
- `calibration_compatibility.json` preserves historical and current provenance separately.
  `scripts/record_dock_idle_compatibility.py` verifies the only navigation AST change is
  the dock condition on stationary hover consumption. Legacy physics, model, dependencies,
  flight path and calibration engine are unchanged. Evaluation checks this compatibility
  record and still requires exact current provenance on resume.

Authorized diagnostic first stage: B4 validation, all 27 frozen regimes, four thresholds
(0.1, 0.25, 0.5, 0.75), 10 predeclared validation seeds = 1080 persistent runs.
B0–B5 evaluation is a later stage and is not auto-launched. No neural training, MPC,
OracleSafe or small reference. No final 98% kill test or 5% chance-constraint claim.
Estimator error audit remains needed before interpreting method differences.
