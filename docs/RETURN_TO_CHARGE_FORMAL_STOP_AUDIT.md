# Return-to-Charge Formal Gate Stop Audit

Date: 2026-08-30  
Goal source: `ORIGINAL_GOAL.txt`  
Outcome invariant: stranding--throughput Pareto improvement  
Classification: historical scientific prerequisite stop, superseded for the active R3 conditional research contract

> **Superseded on 2026-08-30.** This file remains an immutable account of why
> the former deployment-quality chain stopped. The user subsequently selected
> R3 as the frozen baseline for conditional Energy/return research and rejected
> treating the historical success/path/collision thresholds as theorem
> assumptions. The active machine contract is
> `artifacts/r3_fixed_baseline_energy_chain/navigation_platform_contract.json`.
> It preserves R3's old Gate failure while authorizing battery validation and
> Oracle Headroom conditional on frozen R3 plus HOCBF. Statements below that the
> chain can reopen only through a new threshold-passing platform are historical,
> not current instructions.

## Decision

The original formal chain remains stopped at the 500-task navigation
prerequisite. R1--R5 each completed its fixed 500,000-transition budget and the
fixed 500-task evaluation, but none passed all preregistered navigation and
safety thresholds. Battery calibration, battery validation, the 320-independent-
cycle-per-point Oracle Decision Headroom Gate, and learned-method comparisons therefore
remain correctly unexecuted.

This is a valid scientific stop under the protocol: a named prerequisite failed
with terminal artifacts; the failure is a property of the frozen navigation/
safety platform under the fixed Gate; downstream evidence was not fabricated or
run out of order.

## Requirement audit

| Requirement | Authoritative evidence | Verdict |
| --- | --- | --- |
| Pluggable ReturnManager preserves legacy switching | Regression tests and completion-ledger row 2 | Complete |
| Mission q95/component semantics are explicit | Probability semantics tests and completion-ledger row 3 | Complete |
| Conditional probability object is audited | R5 fixed-snapshot machine artifact and protocol §3 | Complete for deterministic ETG; stochastic quantile claim forbidden |
| Simulator Oracle supports return-now and task-then-return | Stage-B implementation, clone/cache tests, smoke artifact | Implementation complete; formal evidence absent |
| Navigation prerequisite | R1--R5 `EVALUATION_COMPLETED.json` and `STOPPED_NAVIGATION_NOT_READY.json` | Formal FAIL |
| Battery calibration and 100-run validation | Requires a passing navigation artifact | Not authorized / not evaluable |
| Oracle Decision Headroom | Requires passed navigation, calibration, and validation | Not evaluable |
| Five-method stranding--throughput frontier | Requires Oracle Gate PASS | Not authorized |
| Theory mechanism | Theorems 1--26 and finite verifiers | Substantial conditional theory; not empirical Pareto evidence |

R5 is the final authorized learned-navigation repair. Its 500-task result was:

```text
overall success                  0.938   (required >= 0.98)
minimum distance-bucket success 0.860   (required >= 0.95)
mean path ratio                  1.09846 (required <= 1.10)
obstacle collision steps         2497    (required 0)
boundary-contact step rate       0.0     (required < 0.01)
```

The formal artifact is
`artifacts/jseb_navigation_repair_r5_uv_seed0_20260829_160435/R5/STOPPED_NAVIGATION_NOT_READY.json`.

## Probability-semantics diagnostic

The navigation failure does not prevent a read-only audit of what random object
the simulator defines. The standalone P0 runner loaded the wrapper-matched R5
500k checkpoint, reconstructed the original 24-obstacle sampled-data-HOCBF
platform, fixed one complete deployment snapshot, and repeated the exact Oracle
return three times. All three costs were `14.322758552613978`; variance was zero
and there was one value at the numerical tolerance. The artifact is
`artifacts/r5_return_probability_semantics_20260830/probability_semantics_audit.json`.

The runner records that no wind, transition-noise, moving-obstacle, or seeded
future-disturbance model exists. Its repetition index is not mislabeled as a
disturbance seed. Hence the current target is point Resource-to-Go with epistemic
reliability, and aleatoric q90/q95 claims remain mechanically unauthorized. The
same artifact retains `navigation_gate_passed=false` and
`downstream_stages_authorized=false`; probability identification does not unlock
calibration or Oracle Headroom.

## Non-learning replacement-platform diagnostic

To test whether navigation could be treated as a nuisance platform rather than
continued SAC repair, a separate deterministic go-to-goal controller runner was
implemented in `scripts/evaluate_fixed_navigation_platform.py`. It preserves the
R5 map, task distribution, HOCBF sampled-data mode, and formal thresholds, but
performs no policy training, updates, or replay writes.

The execution-only 10-task smoke at
`artifacts/fixed_navigation_platform_smoke_20260830_v2` completed normally:

```text
success                          9 / 10
mean path ratio                  0.996465
obstacle collision steps         0
boundary-contact step rate       0
```

This smoke is not formal evidence and cannot authorize downstream stages. Its
0.90 success rate is materially below the 0.98 formal threshold, so the candidate
is not promoted to a costly 500-task Gate. Doing so would be another low-evidence
navigation attempt contrary to the no-endless-repair rule.

## What would reopen the chain

Only an explicitly authorized new platform contract may reopen the formal chain.
It must be substantively different from R1--R5, freeze its controller and safety
operator before evaluation, declare a fresh 500-task seed, and pass the unchanged
navigation thresholds. A passing artifact would hand off to battery calibration,
then 100-run battery validation, then the paired Oracle Headroom Gate. The
current formal Oracle protocol uses 320 independent cycles per point for the
default 12-point grid. The earlier 110-cycle value only made zero-event
certification mathematically possible; at a 1% true rate it had 0.331 power.
The 320-cycle protocol allows six failures, has 0.956 per-family power, and has
a 0.912 lower bound on jointly certifying one safe Oracle and one safe heuristic
family without assuming independence.

Until that happens:

- exploratory R3 Energy data may support mechanism/theory work only;
- no Oracle headroom, Pareto, stochastic-coverage, or oral-level empirical claim
  is authorized;
- the completed mathematical design remains a conditional candidate rather than
  a substitute for the missing Gate evidence.

The Stage-B executable now preserves this distinction mechanically. Existing
but nonpassing prerequisite JSONs produce
`STOPPED_PREREQUISITES_NOT_READY.json`, exit code 4, and
`downstream_authorized=false`; they do not produce `FAILED.json`. Missing,
unreadable, or malformed prerequisite files remain implementation/input errors
and retain the failure path.

The historical R5 completion wrapper is now consumed through its actual
`final_navigation` schema rather than being mistaken for a flat metric table.
Its 500-task metrics are recovered correctly, while its explicit
`navigation_gate_passed=false`, `downstream_navigation_ready=false`, and
`downstream_stages_authorized=false` remain authoritative. Formal continuation
also requires one SHA-256 identity chain from the navigation wrapper and loaded
checkpoint through calibration and battery validation; historical artifacts
without that complete chain cannot be silently mixed with a different policy.

The current R5 artifact can now be reconstructed by the shared evaluator despite
its orchestration-only `R5`, sampled-data-HOCBF, and vector-environment flags.
A read-only checkpoint load produced the expected 2055-dimensional observation,
3-dimensional action, 8x128 LiDAR, 24 obstacles, HOCBF enabled with
sampled-data robustness, and projection geometry enabled. This verifies artifact
compatibility only; R5's failed navigation authorization remains unchanged.

No `ccfa.yaml` is present in the project, so project-state tracking remains in
the protocol and completion ledger rather than an inferred YAML update.
