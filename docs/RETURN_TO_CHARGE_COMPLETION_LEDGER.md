# Return-to-Charge Research Completion Ledger

## Purpose

This ledger maps the expert's twelve-step research sequence to authoritative
code, tests, runtime artifacts, and explicit gates.  A row is `COMPLETE` only
when the evidence proves the full row; implementation, smoke, or a live process
does not substitute for a formal experimental result.

## Fixed Research Invariants

- Top-level task: decide when to irreversibly switch from task execution to
  charger return.
- ML object: Resource-to-Go of the executed policy--safety-filter closed loop.
- Hard collision authority: HOCBF; learned energy/reliability modules are not
  certificates.
- Current probability semantics: deterministic point Resource-to-Go plus
  epistemic uncertainty.  q90/q95 aleatoric claims are gated on a future
  clone-rollout variance audit with a physically motivated disturbance process.
- Headline utility: stranding--throughput Pareto frontier, not prediction MAE.
- Stage C and SIRP are forbidden before the Oracle headroom Gate passes.

## Expert Task Ledger

| ID | Required work | Authoritative evidence | Status | Remaining proof |
| --- | --- | --- | --- | --- |
| 1 | Freeze commit, dependencies, and seed protocol | Formal pipeline manifest records evaluator/source SHA, fixed seed 170001, exact command, fixed task count, and clean evaluator worktree | `COMPLETE` for current Gate chain | A later Stage C benchmark needs its own dependency lockfile and immutable dataset manifest |
| 2 | Pluggable `ReturnManager` with exact legacy switching regression | `review_bundle/safety/switching/return_manager.py`; seeded legacy/refactored switching test | `COMPLETE` | None for current Quantile/SOC/distance managers |
| 3 | Correct mission q95 semantics | `MissionEnergyEstimate` distinguishes direct joint mission, component sum, and union-bound coverage; component-risk regression test | `COMPLETE` | No stochastic q95 headline until probability Gate changes |
| 4 | Simulator clone Oracle for return-now and task-then-return | `ModelBasedEnergyRolloutEstimator`; clone-state, obstacle/LiDAR/HOCBF, direct joint mission, and cache regressions | `COMPLETE` as implementation | Formal decision-value evidence remains pending |
| 5 | SOC, distance, Frozen-TD, Online-TD, Oracle reserve sweep | Stage-B Oracle runner and gated TD decision runner exist | `IN PROGRESS` | Formal navigation Gate, calibration, Oracle Gate, TD readiness, and complete five-method frontier |
| 6 | Stop ICLR return line if Oracle lacks material headroom | Preregistered Gate: Oracle must improve eligible heuristic throughput by at least 5% under common stranding ceiling 0.05, with at least 100 independent cycles per point | `PENDING RESULT` | Formal Oracle artifact with `evaluable=true` and `passed=true/false` |
| 7 | Port, rather than freely reimplement, Dopamine/PCM/OfflineRL-Kit components | Protocol names source projects and fair-information contract | `GATED, NOT STARTED` | Oracle Gate PASS and pinned upstream revisions/licenses |
| 8 | MC-direct, PCM-Executed, executed-WM, ensemble under shared data/protocol | Method contracts described only | `GATED, NOT STARTED` | Implementations, tests, common source dataset, held-out evaluation |
| 9 | Direct high-level continue/return switcher | Required in protocol, no implementation | `GATED, NOT STARTED` | Oracle Gate PASS, SB3 switcher, matched training privileges and seeds |
| 10 | Complete \(2\pi\times2\Pi\) leave-one-pair-out with interface-extrapolation × horizon matching | Design only | `PENDING` | Two policies, two genuinely different safety operators, matched strata, all held-out pairs |
| 11 | Add cross-fitted occupancy-weighted reliability only if ensemble is insufficient | Theory transport term and stop rule documented | `CONDITIONAL, NOT STARTED` | Evidence that executed-WM+ensemble fails to rank long-horizon ETG risk |
| 12 | CMDP track, second safety family/domain, full paper evidence | Separate modular/end-to-end comparison design and provisional proof package | `PENDING` | Pilot PASS, fair CMDP privileges, second domain/filter, independent theory review |

## Current Formal Gate Chain

```text
500-task navigation prerequisite
    -> 500-task battery calibration
    -> 100-run battery validation
    -> 100-independent-cycle Oracle headroom comparison
    -> exact 500k Quantile-TD collection
    -> Frozen/Online TD decision comparison
```

Every arrow is fail-closed.  The downstream process must consume a named passing
artifact; it must not infer success from process exit alone or recompute a missing
Gate from an incomplete table.

## Current Evidence Snapshot

- Navigation source checkpoint: 500k JSEB SAC checkpoint from the archived 1M
  training artifact.
- Formal evaluation: 500 fixed navigation tasks, six environment workers,
  central deterministic policy inference, no policy/TD updates or replay writes.
- Formal navigation result: pending while this ledger was written.
- Battery calibration, Oracle, TD, and TD decision outputs: pending and gated.
- Proof package: five corrected theorems/propositions and two corollaries drafted;
  independent semantic proof acceptance is not available.
- Decision-interval telemetry: implementation and related tests pass; formal
  overshoot data are pending a later decision run from a commit containing that
  telemetry.

## Completion Conditions

The active research goal is not complete until one of the following evidence-
bounded outcomes is reached:

1. **Scientific continuation:** navigation, calibration, and Oracle Gates pass;
   the five-method decision frontier is complete; Stage C pilot and its stop rules
   are adjudicated; the required compositional/generalization evidence is either
   completed or honestly narrows the venue claim.
2. **Scientific stop:** a preregistered Gate fails with a valid, auditable formal
   artifact; the failure is classified as scientific rather than implementation;
   unsupported downstream work is not run; and the resulting research conclusion
   and revised route are documented.

Neither a smoke test nor a running background session satisfies either condition.
