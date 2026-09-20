# PersistentUAVThroughput-v1

Read README.md and MINIMAL_PERSISTENT_THROUGHPUT_SPEC.md first.
This is separately scoped; the old regenerative-control project remains closed.
Legacy flight physics/model are read-only dependencies. Preserve collision and
LiDAR-only deployment contracts.

2026-09-20 latest user authorization supersedes the previous baseline hold:
- Fix dock idle: zero energy consumption and no automatic charge; away idle retains hover cost.
- Focused unit tests and updated continuing smoke; do NOT rerun 1000-job calibration.
- Preserve the frozen 27 regimes. Navigation qualification is PASS for scheduling diagnostics.
- After the fix passes, B0–B5 diagnostics are authorized. First results cannot trigger 98% kill test.
- Neural training remains prohibited. MPC, OracleSafe and small reference are deferred.
- Use existing 10 validation / 20 evaluation seeds and fixed threshold candidates.
  Empirical risk acceptance is not a 5% safety certificate.
- Check first resumable checkpoint health, then hand back; do not monitor to completion
  or automatically advance from validation to evaluation.
- Before interpreting results, estimator error audit and estimation/planning
  decomposition remain required; poor B5 results alone do not establish scheduling difficulty.
