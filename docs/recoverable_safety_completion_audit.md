# Recoverable Safety Completion Audit

## Scope

This audit checks the requested diagnostic and research-decision stage. It does not redefine success as a new controller: the requested stop rules require ending a candidate when the closest work supplies a stronger theorem or online computation misses 20 Hz.

## Requirement Ledger

| Requirement | Status | Evidence |
| --- | --- | --- |
| Read current research overview and safety audits | Complete | Existing baseline and safety documents were inspected before implementation |
| Preserve frozen SAC and old safety/energy code | Complete | Only new experiment, script, test and documentation paths were added |
| Locate current sampled-data HOCBF/Candidate A | Complete | Diagnostic directly imports the existing filter and HOCBF primitives |
| Log exact pointwise `rho` | Complete | `compute_pointwise_margin` uses `joint_feasibility_margin` over the cylindrical physical input set |
| Log raw and strengthened HOCBF semantics separately | Complete | Every negative strengthened step computes `rho_true_unstrengthened` |
| Construct ten hard families | Complete | Unit test checks ten unique required families |
| Bound obstacle velocity/acceleration/jerk | Complete | `DynamicObstacleSpec` validates and clips all three quantities |
| Preserve UAV sample/hold/rate/limit semantics | Complete | Constants imported from the existing safety benchmark |
| Model dropout/occlusion explicitly | Complete | Visibility windows plus four perception modes |
| Save state, obstacles and sensing history | Complete | Schema-complete hard-state JSONL artifacts |
| Save HOCBF and actuator rows | Complete | HOCBF rows plus actual 32-facet actuator and velocity rows |
| Save relative velocity, TTC, clearance and braking | Complete | Explicit hard-state fields |
| Detect `rho_t > 0` then `rho_future < 0` | Complete | 556 Stage-2 and 451 matched Stage-3 events |
| Separate current infeasibility and negative entry | Complete | Step counts and deduplicated negative-entry records are separate |
| Classify dropout, reappearance, delay, braking and multi-obstacle causes | Complete | Headline cause plus independent flags |
| Test whether current action actively loses future feasibility | Partial but sufficient for diagnosis | Finite sampled oracle finds alternatives in 356/451 events; it is not a robust proof |
| Stable repeated failure | Complete | Independent 1,010-scenario and 1,040-rollout matched studies |
| Compare four perception treatments | Complete | Matched scenario comparison |
| Define candidate recoverability object | Complete | Robust future-feasibility margin and predecessor derivation |
| Prior-art attack before new theorem | Complete | Fourteen primary papers in review and CSV matrix |
| Select Route A/B/C | Complete | All three closed with evidence |
| Do not invent a theorem | Complete | Theory document labels definitions and conditional implications |
| Evaluate 20 Hz computation | Complete | Mean and tail latency reported; tail requirement fails |
| Record progress and collision outcomes | Complete for diagnostic stage | Net progress, collision, clearance and fallback metrics |
| Record energy/deadlock/oscillation | Not executed by design | Required only for a candidate comparison; candidate failed novelty gate before Stage 4 |
| Implement closest-work baselines | Not triggered | Instruction makes final baseline implementation conditional on a candidate passing earlier gates |
| Run formal multi-seed experiments | Correctly stopped | Closest-work and compute stop conditions triggered |
| Write all named documents | Complete | Problem, theory, counterexamples, experiments, literature review/matrix, novelty audit and final decision |

## Artifact Integrity

The schema-complete artifacts are:

- `artifacts/uav_recoverable_safety_hand_v4_20260820/`;
- `artifacts/uav_recoverable_safety_random1000_seed0_v2_20260820/`;
- `artifacts/uav_recoverable_safety_matched250x4_seed1_v3_20260820/`.

All three manifests identify the same frozen checkpoint SHA-256:

`df9a4354420a1a043b60d704292c6fe06ddd2127e937d3d9c72278b287c1ff0a`.

Every inspected hard state has 34 actuator rows, 34 velocity rows and explicit relative velocities. The matched artifact contains 451 hard-state records.

## Known Limitations

- The boundary-plus-obstacle family logs physical boundary contact but the inherited collision filter does not represent workspace boundaries as CBF rows. This family produced no negative-entry event and is retained as a negative hand case, not evidence of boundary-CBF conflict.
- The finite action oracle sees the realized next obstacle state and cannot be deployed online.
- Short diagnostics do not establish task completion, deadlock, oscillation or energy overhead.
- Zero collisions mean this study exposes loss of certified control authority, not demonstrated physical crashes.

## Audit Verdict

The requested failure-diagnosis, mathematical-object, prior-art and decision stages are complete. The candidate method stage is intentionally closed rather than left unperformed: it fails explicit novelty and real-time gates.

## Test Evidence

- New recoverability diagnostics plus existing collision-filter regression: `70 passed`.
- Root repository suite: `390 passed, 33 subtests passed`.
- `review_bundle` suite with its required package root: `58 passed`.
- The first root-suite attempt had one existing certified-runtime assertion fail; the isolated rerun passed and the complete second run passed all 390 tests. No old source file was changed to obtain this result.
