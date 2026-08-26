# JSEB 500k Navigation Training Diagnostic

## Status and Evidence Boundary

This document is a **training-log diagnostic**, not the formal 500-task
navigation Gate result.  It summarizes the trajectory of the archived Phase-1
training run while the deterministic fixed-task evaluation is still running.
It must not be cited as a final success rate, path ratio, collision rate, or
readiness decision.

The formal result is authoritative only when
`artifacts/jseb500k_navigation_gate_parallel_20260826_211448/eval_checkpoint_500000.json`
exists and all preregistered predicates have been evaluated.

## Provenance

- Source artifact:
  `artifacts/jacobian_safety_energy_static_1m_20260825_053617`
- Source Git SHA: `df566d294eb16985197e71a0cd711a2e20fa5d2b`
- Evaluated checkpoint:
  `phase1_navigation/checkpoint_transition_500000.zip`
- Training curve:
  `training_curve.jsonl`
- Training environments: 8
- Training budget: exactly 500,000 environment transitions
- Last periodic training row: 496,000 transitions
- Formal evaluation task set: 500 fixed tasks, seed 170001
- Formal evaluator Git SHA: `d45548fe0b6ff9abe3cffd3c6116c9b5ccee81aa`

The source pipeline was intentionally interrupted after the 500k navigation
checkpoint, so it has no completed downstream Phase-1B/Phase-2 summary.  The
checkpoint itself exists; the independent formal Gate is evaluating it.

## Training-Curve Snapshot

The logger writes every 8,000 environment transitions.  The 100k, 300k, and
500k rows below therefore use the last logged row not exceeding the named
milestone.  Success rate is derived as cumulative naturally terminated
successful episodes divided by cumulative naturally ended episodes.  Partial
episodes are excluded.

| Nominal milestone | Logged transitions | Ended episodes | Successful | Derived success rate | Mean path ratio | Mean steps per ended task | Interval HOCBF intervention rate | Gradient updates / transition |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100k | 96,000 | 65 | 58 | 0.8923 | 1.9066 | 1429.43 | 0.3475 | 0.9478 |
| 200k | 200,000 | 144 | 126 | 0.8750 | 1.6823 | 1321.28 | 0.3536 | 0.9750 |
| 300k | 296,000 | 218 | 189 | 0.8670 | 1.7517 | 1324.56 | 0.3021 | 0.9831 |
| 400k | 400,000 | 309 | 271 | 0.8770 | 1.5958 | 1254.08 | 0.5220 | 0.9875 |
| 500k | 496,000 | 390 | 340 | 0.8718 | 1.5223 | 1239.17 | 0.3043 | 0.9899 |

The training success statistic is not directly interchangeable with the
fixed-task evaluation success rate because the training task distribution,
episode censoring, and sample weighting differ.  It is nevertheless useful as
a convergence diagnostic.

## Safety and Optimization Audit

Across the 496,000 logged transitions:

- boundary-contact transitions: 6 (`1.21e-5` of logged transitions);
- obstacle-collision transitions: 2 (`4.03e-6` of logged transitions);
- HOCBF-intervened transitions: 178,159 (`0.3592`);
- HOCBF emergency-brake transitions: 35,937;
- naturally ended failed/stuck tasks: 50;
- gradient updates at 496k: 490,992;
- gradient-update/transition ratio at 496k: `0.9899`.

The serialized 500k checkpoint additionally reports:

- actor gradient steps: 494,992;
- bridge gradient steps: 490,000 (`0.9899` of actor-gradient steps);
- mean shield-consistency loss over bridge-ready steps: `0.1086`;
- nonzero shield-loss steps: 489,059;
- mean shield valid fraction after the local trust-region mask: `0.1108`;
- Phase-1 energy bridge steps: 0, as intended for navigation-only Phase 1.

The projection-geometry dataset itself had a cumulative valid-Jacobian rate of
`0.9143` at 496k, whereas only `0.1108` of sampled bridge examples remained
valid after comparing the current actor action with the replay anchor under the
configured `0.35` local trust region.  The Jacobian bridge was therefore active
on nearly every optimizer step but supplied a nonzero actor constraint on a
small subset of each sampled batch.  A high geometry-valid rate does not imply
broad valid supervision after replay-policy drift.

| Logged transitions | Nominal-safe action rate | HOCBF intervention rate | Emergency-brake rate | Mean intervention norm | Valid-Jacobian rate |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 96,000 | 0.6099 | 0.3901 | 0.0848 | 0.2699 | 0.8918 |
| 200,000 | 0.5819 | 0.4181 | 0.1040 | 0.2913 | 0.8763 |
| 296,000 | 0.6054 | 0.3946 | 0.0922 | 0.2633 | 0.8919 |
| 400,000 | 0.6247 | 0.3753 | 0.0807 | 0.2454 | 0.9040 |
| 496,000 | 0.6408 | 0.3592 | 0.0725 | 0.2220 | 0.9143 |

This is evidence of a modest late improvement in nominal-action safety, not of
independent safe-navigation mastery.  More than one third of all actions still
required HOCBF intervention, and the navigation metrics remained far outside
the readiness thresholds.

The low contact counts show that the hard safety layer mostly prevented
physical contacts during training.  They do **not** establish navigation
readiness: HOCBF changed the nominal action on about 36% of logged transitions,
and the policy still exhibited long paths and many task timeouts.  Safety-filter
success must not be counted as actor navigation success.

The update audit does not indicate accidental `gradient_steps=1` under eight
environments.  After the learning-start delay, the cumulative update ratio
approaches one update per collected transition, as intended by
`train_freq=(1, "step")` and `gradient_steps=-1`.

## Preregistered Formal Gate

The independent 500-task evaluation must satisfy all of:

1. overall success rate at least `0.98`;
2. every distance-bucket success rate at least `0.95`;
3. mean path ratio at most `1.10`;
4. boundary-contact step rate below `0.01`;
5. zero obstacle-collision steps.

The last training diagnostic is far from the first three navigation targets:

- cumulative training success: `0.8718` versus formal threshold `0.98`;
- mean training path ratio: `1.5223` versus formal threshold `1.10`;
- mean ended-task length: `1239.17` policy steps.

This is strong evidence that the checkpoint is unlikely to pass the formal
navigation Gate, but it is **not** a substitute for the fixed-task result.  No
threshold may be lowered after observing this diagnostic.

## Interpretation if the Formal Gate Fails

A valid formal failure should be classified as a **navigation prerequisite
failure**, not an Energy-TD, battery-calibration, Oracle-headroom, or
return-manager result.  The fail-closed pipeline must then stop before those
stages.  The current evidence points to navigation inefficiency, timeout
behavior, a flat 2055-dimensional LiDAR input passed to the default SAC MLP
without a structure-aware encoder, and narrow valid Jacobian supervision after
replay drift.  It does not point to an optimizer update-ratio bug or a
collision-safety plumbing failure.  The representation and bridge-coverage
items remain root-cause hypotheses until controlled ablations test them.

The repair target would be a policy that independently passes the same fixed
500-task Gate.  It would be invalid to continue downstream by lowering the
success/path thresholds, counting HOCBF intervention as learned navigation, or
using incomplete training episodes as successful tasks.

## Interpretation if the Formal Gate Passes

If the fixed-task result unexpectedly passes every predicate, that artifact
overrides this diagnostic and unlocks battery calibration.  Calibration and
Oracle stages must still pass their own preregistered Gates before TD training
or Stage-C baselines are allowed.
