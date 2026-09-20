# PersistentUAVThroughput-v1

Latest user review after f15de33 authorizes offline Stranding Decomposition only:
- Use the existing 270 B5 runs; no new simulations or policies, including OracleSafe.
- Decompose pre-task to post-task reserve using measured task cost and the return
  estimate at the actual endpoint. Separately account for any intervening idle cost.
- Keep predicted-infeasible fallback, point-estimator false-safe, and waiting
  mechanisms distinct. Failed-leg costs are censored, not complete energy labels.
- Do not equate nonpositive predicted return margin with true irrecoverability.
- New analysis: evidence/stranding_decomposition_20260920/. The 166 failed returns
  partition into 67 fallback, 37 task-error margin flips, 40 non-fallback waiting
  margin losses and 22 return false-safe cases. No new phase was launched.

Latest user review after c158cc07 (2026-09-20) narrows the next stage:
- B5 subsequently completed all 270 jobs at 20:59: 27/27 regimes fail empirical
  eligibility. Evidence: evidence/b5_validation_complete_20260920/. Do not rerun
  completed jobs or infer a new experiment authorization from the prior launch GO.
- B4 ARM validation is complete: none of the four preregistered candidates is
  empirically eligible in any of 27 regimes; this does not exclude all SOC thresholds.
- Only B5 reserve-SJF validation is GO: original 27 regimes × 10 validation seeds.
- Preserve the frozen estimator, parameters, strict reserve test and zero added margin.
- Log task/return predictions, starting battery, reserve margin and leg-specific
  depletion/navigation outcomes; report throughput and all three failure rates together.
- B0–B3 and evaluation are on hold. No training, MPC, Oracle or 98% kill test.
- Feasibility of tight regimes is unresolved. Separate feasibility, estimation and
  planning before interpreting B5 failure as a planning or learning opportunity.
- Necessary focused tests, one tiny real smoke, then first-checkpoint startup health
  and hand back. Do not monitor to completion or promote to another stage.
- Workspace is now /mnt/workspace/zjl-exp; historical B4 path records stay immutable.

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

Latest handoff: local B4 failed at 382/1080; v1.4 wait-grid repair is engineering-only.
Do not resume local production or import its 382 rows into ARM results. ARM is the
new execution host after audited compatibility update and focused smoke. Preserve
existing migration_20260920 records; see ARM_DIAGNOSTIC_HANDOFF.md.
