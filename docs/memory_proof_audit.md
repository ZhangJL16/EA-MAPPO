# Memory Observer–Tube–HOCBF Proof Audit

## Scope and assurance

The audit covers the explicit interval observer, finite-horizon orthotope tube,
directional robust HOCBF, sampled-data hold strengthening, and uncertainty-set
monotonicity. Same-family fresh-agent judgments are provisional. A requested
Claude cross-family overlay failed to start because the local `claude` CLI is
not installed; no cross-family acceptance is claimed.

## Round 1 — failed

Exact audited revision: working tree based on Git
`657b08b3768c90966b32813a97b708bebf7fe175`.

Fresh proof audit result: **1 FATAL, 5 CRITICAL, 4 MAJOR, 1 MINOR**.

| ID | Severity | Verified defect | Resolution |
|---|---|---|---|
| R1-01 | FATAL | `h` and `psi1` values down to `-1e-10` were accepted as certified. | Preconditions now require exact `>= tolerance` with finite nonnegative default `tolerance=0`. |
| R1-02 | CRITICAL | A Boolean association flag carried no track identity or reset-bound provenance. | Added immutable `track_id` plus explicit `TrustedBaseIntervalCertificate`; missing/mismatched identity fails closed. |
| R1-03 | CRITICAL | Positive innovation tolerance could exclude truth while retaining certification. | Certified mode now requires zero innovation tolerance and rejects nonfinite values. |
| R1-04 | CRITICAL | Documentation implied production runtime integration. | Claim narrowed to the controlled experiment runner; production collision filter is explicitly out of scope. |
| R1-05 | CRITICAL | Real-arithmetic interval proof was called a floating implementation certificate. | All theorem claims are explicitly ideal-real-arithmetic; no outward-rounded/physical certificate is claimed. |
| R1-06 | CRITICAL | Tiny leak could round the implemented contraction factor to one. | Network constructor enforces an execution-dtype contraction gap of at least `1e-6`. |
| R1-07 | MAJOR | Observer residual jerk and total true jerk were conflated. | API and derivation now distinguish `jerk_residual_bound` from `true_obstacle_jerk_bound`. |
| R1-08 | MAJOR | Negative obstacle radius was accepted. | Runtime rejects nonfinite or negative radius; theorem states `d >= 0`. |
| R1-09 | MAJOR | Regularity and feasible-set domains were vague. | Added absolute continuity, a.e. dynamics, measurable bounded inputs, compact intervals, and explicit actuator-set definition. |
| R1-10 | MAJOR | Edge tests were missing. | Added track swap, missing certificate, strict precondition, nonfinite tolerance, contraction-gap, and timestamp containment regressions. |
| R1-11 | MINOR | Documented state conflated public interval state and aggregate observer memory. | Definition now names the aggregate object and its split implementation. |

## Independent runtime audit — failed

A separate fresh audit found one additional FATAL runtime defect: the observer
advanced at the 0.2 s sensor period but its state was reused at intermediate
0.1 s controller holds. It also found that marginal split-conformal baselines
were being counted as deterministically certified.

Resolutions:

1. the analytic observer now propagates at every 0.1 s control hold and only
   corrects on scheduled 0.2 s measurements;
2. closed-loop artifacts report truth-containment under-bound steps as a
   diagnostic;
3. learned/Kalman/IMM split-conformal states are explicitly
   `certification_valid=False`; they may be run as probabilistic baselines but
   cannot contribute deterministic certified-step counts.

## Round 2 — failed

Fresh same-family audit result: **1 FATAL, 1 CRITICAL, 3 MAJOR, 1 MINOR**.

| ID | Severity | Verified defect | Resolution |
|---|---|---|---|
| R2-01 | FATAL | A cached base certificate could be reused after contradictory innovation. | Removed cached-certificate reuse. Reset certification now requires a caller-supplied same-track certificate with a strictly newer measurement epoch. |
| R2-02 | CRITICAL | A QP solution violating a constraint by solver tolerance could still be labeled certified. | Constraints are strengthened by `1e-6`; certification additionally requires convergence and at least `0.5e-6` post-solve slack in the original constraints. |
| R2-03 | MAJOR | Split-conformal radii reused validation data involved in checkpoint selection. | Radii now use a dedicated first-half test calibration partition; validation is not reused. |
| R2-04 | MAJOR | Controlled obstacle simulation used acceleration ZOH while the theorem bounded constant jerk. | Simulation, nominal alignment, and collision diagnostics now use constant realized jerk integration. |
| R2-05 | MAJOR | Runtime accepted a configured jerk bound smaller than pre-registered scenario maneuvers. | `ClosedLoopConfig` rejects `obstacle_jerk_bound < 12`; regression verifies scenario jerk is bounded. |
| R2-06 | MINOR | “Execution dtype” wording included unsupported FP16 semantics. | The implementation now rejects parameter execution outside FP32/FP64; claims are narrowed accordingly. |

### Executor-discovered post-Round-2 defect

Before the next audit, a direct scenario replay found that the controlled
obstacle velocity reached approximately 165 m/s while reset certificates still
attested a 12 m/s base bound. That made reset certification false in the
experiment even though the observer algebra was conditional. The runner now
derives a componentwise finite-horizon envelope from initial velocity, the
declared 6 m/s² acceleration bound, and the full rollout horizon, and records
any runtime envelope violation. This correction widens the analytic interval;
it does not tune the benchmark to improve performance.

## Round 3 — failed

Fresh same-family audit result: **0 FATAL, 3 CRITICAL, 1 MAJOR, 1 MINOR**.

| ID | Severity | Verified defect | Resolution |
|---|---|---|---|
| R3-01 | CRITICAL | An invalid reset erased the public certificate epoch, allowing an older same-track certificate to be replayed. | The observer now keeps a private greatest-accepted epoch ledger per track; invalid states retain their last same-track epoch, and stale replay has a regression test. |
| R3-02 | CRITICAL | The exact-state baseline used a sampled-data bound assuming constant obstacle acceleration while the simulator applied nonzero jerk. | The exact-state baseline now calls the same directional jerk-aware robust filter with zero state radii. |
| R3-03 | CRITICAL | The exact-state path bypassed convergence and positive original-constraint post-solve slack checks. | Routing exact state through the common robust filter makes the reserve and post-solve checks mandatory. |
| R3-04 | MAJOR | A 99% split-conformal label was unsupported because scale fitting and quantile calibration shared 500 samples, which cannot resolve six-coordinate simultaneous 99% coverage. | The runner now uses componentwise maximum calibration residuals and explicitly makes no 99%, distribution-free, or repeated-time claim. Baselines remain uncertified. |
| R3-05 | MINOR | FP64 network parameters were supported directly but observer features were forced to FP32. | Observer features now follow parameter dtype; FP64 regression added. |

The interrupted artifact
`artifacts/memory_closed_loop_controlled_final_20260820_000136/` is marked
`INVALID.json` and cannot support claims.

## Round 4 — passed for the scoped ideal-arithmetic claims

Fresh same-family exact-current-source audit result: **0 FATAL, 0 CRITICAL,
0 MAJOR, 0 MINOR**. The scoped test slice passed `41` tests. The final
  controlled artifact was superseded after integrity remediation by
`artifacts/memory_closed_loop_controlled_final_audited_20260820/`; all 13
recorded executed-source SHA-256 values match the current source files, the
estimator training-source hash matches its checkpoint artifact, and all
calibration/test inputs are hashed.

The acceptance is deliberately narrow:

- it validates the conditional ideal-real-arithmetic observer, tube, robust
  HOCBF, sampled-data hold, and monotonicity statements;
- it does not certify the learned/Kalman/IMM empirical boxes;
- it does not turn infeasible emergency fallback steps into certified steps;
- it does not establish floating-point soundness or physical-UAV safety;
- it does not establish novelty.

The final artifact contains `60,152` controlled transitions across `12`
matched scenarios and remains below the `100,000` transition ceiling. The
oracle exact-state path has zero simulated collisions and zero interval
under-bounds, but `2.5204%` of steps are infeasible/uncertified fallback. It is
therefore not evidence of a fully certified end-to-end trace.

Cross-family review remains unavailable because the local `claude` CLI is not
installed. The zero-defect result is a same-family provisional audit, not an
independent external certification.

## Post-audit integrity remediation

After Round 4, `closed_loop.py` changed only in development-box split choice
and source/input provenance recording: held-out test data no longer configures
closed-loop radii, and the hash manifest was expanded. Observer, tube, HOCBF,
QP, dynamics, fallback, and theorem documents were unchanged. The final
integrity reviewer closed both critical experiment issues and reported zero
unresolved fatal/critical/major/minor findings after the sole stale latency
number was corrected. The final relevant test slice is `114 passed`.

Accordingly, the Round-4 semantic proof verdict applies to the unchanged
theorem implementation modules. No claim is made that the subsequent
provenance-only runner edit received a new cross-family proof review.
