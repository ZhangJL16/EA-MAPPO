# JSEB 500k Navigation Gate Result

## Verdict

`checkpoint_transition_500000.zip` completed the preregistered independent
500-task model-selection evaluation and **failed navigation readiness**.
Collision and boundary safety passed, so this is not a collision-safety failure.

## Provenance

- source artifact: `artifacts/jacobian_safety_energy_static_1m_20260825_053617`
- source Git SHA: `df566d294eb16985197e71a0cd711a2e20fa5d2b`
- checkpoint SHA256: `b831653ef71367f7a5feefb45d9ad54c7e8a88a5acfdee15a0b3adb6f5f31638`
- evaluation artifact: `artifacts/jseb500k_navigation_gate_parallel_20260826_211448`
- evaluator Git SHA: `d45548fe0b6ff9abe3cffd3c6116c9b5ccee81aa`
- task count: 500
- task seed: 170001
- policy updates / replay writes / TD updates: none

## Metrics

| Predicate | Required | Observed | Verdict |
| --- | ---: | ---: | --- |
| overall success | >= 0.98 | 0.84 | FAIL |
| minimum bucket success | >= 0.95 | 0.82 | FAIL |
| mean path ratio | <= 1.10 | 1.631250 | FAIL |
| boundary-contact step rate | < 0.01 | 0.000003331 | PASS |
| obstacle-collision steps | 0 | 0 | PASS |

Distance-bucket success rates were 0.82, 0.85, 0.84, 0.83, and 0.86 for
`100-500`, `500-1500`, `1500-2500`, `2500-4000`, and `>4000` meters.
The evaluation used 600,409 environment transitions and took 6,424.6 seconds.

## Fail-Closed Consequence

`downstream_navigation_ready = false`.  The calibration waiter wrote
`STOPPED_NAVIGATION_NOT_READY.json`; battery calibration, Oracle headroom,
Energy TD, and return-decision comparisons are not authorized for this policy.
The result activates `JSEB_NAVIGATION_GATE_REPAIR_PROTOCOL.md` without changing
the environment, obstacle distribution, HOCBF authority, fixed 500k budget, or
Gate thresholds.

## Claim Boundary

This result establishes that the current 500k policy is safe under the measured
collision predicates but not navigation-ready.  It does not establish that the
structured LiDAR encoder or corrected Jacobian bridge will succeed; those are
new controlled hypotheses requiring independent fixed-budget runs.
