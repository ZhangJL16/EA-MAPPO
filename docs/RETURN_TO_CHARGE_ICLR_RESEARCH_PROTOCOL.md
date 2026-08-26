# Reliable Return-to-Charge under Closed-Loop Shift

## 1. Canonical Research Question

The top-level task is a one-way stopping decision:

```text
CONTINUE TASK  or  CHARGER_COMMITTED
```

The agent should delay return long enough to maximize useful task throughput, but
not so long that it exhausts energy before reaching the charger. The machine
learning problem is therefore not energy prediction in isolation:

> How can an autonomous agent estimate and trust the resource required by its
> executed closed loop when the navigation policy and hard safety operator may
> change?

The intended headline evidence is a better stranding--throughput Pareto frontier,
not a lower prediction MAE alone.

## 2. Fixed System Roles

```text
frozen or replaceable navigation policy
    -> nominal action
hard HOCBF safety operator
    -> executed action
physical dynamics + TelemetryCostModel
    -> realized energy
Resource-to-Go estimator
    -> point or conditional distribution estimate
reliability mechanism
    -> epistemic margin / abstention / adaptation
ReturnManager
    -> irreversible TASK -> CHARGER_COMMITTED decision
```

- HOCBF remains the final collision-safety layer.
- Learned navigation, Jacobian features, Energy estimators, and reliability scores
  are not safety certificates.
- Charger arrival recharges and resumes the continuous task stream.
- Energy exhaustion is a true failure terminal.

## 3. Probability Semantics Gate

The predicted object is conditioned on deployment-time information. Random task
generation or random initial states do not create conditional aleatoric uncertainty
once the current state and goal are observed.

The mandatory Stage-B audit fixes state, goal, policy, safety operator, obstacle
realization, and all observed context, then repeats simulator clones while varying
only future disturbance seeds.

Current environment result from the code-path smoke:

```text
conditional variance = 0
q50 = q90 = q95
seeded future disturbance process = absent
```

Therefore the current formal object is deterministic Energy-to-Go plus epistemic
prediction reliability. A stochastic q90/q95 headline is prohibited until a
physically motivated future disturbance process creates a stable non-degenerate
conditional return distribution.

## 4. Mission Upper-Bound Semantics

The legacy fallback

```text
task q95 + return-after-task q95
```

is not called a joint mission q95. In general,

```text
q95(X) + q95(Y) != q95(X + Y).
```

The implementation now records one of these explicit semantics:

- `direct_joint_mission_upper_bound`
- `deterministic_oracle_joint_mission_cost`
- `direct_mission_upper_bound`
- `sum_of_component_q95_not_joint_mission_q95`

For a stochastic mission, the preferred method is a direct joint rollout. A valid
component alternative must allocate risk, for example delta_task + delta_return <=
delta_mission, rather than add two nominal q95 values and claim 95% mission coverage.

## 5. Stage A: Pluggable Return Decision

The environment no longer owns one hard-coded Quantile-TD switching rule. It binds
three independent objects:

```text
navigation policy
Energy estimator, if required
ReturnManager
```

Implemented decision managers:

- `QuantileEnergyReturnManager`
- `FixedSOCThresholdReturnManager`
- `DistanceEnergyReturnManager`

The Stage-B runner can currently attach frozen/online Quantile-TD checkpoints,
the deployed compact JSEB conformal checkpoint, or the simulator Oracle to the
same Quantile return manager.

The default Quantile manager preserves the previous two-boundary behavior:

```text
remaining <= return-now requirement + reserve
or
remaining <= task-then-return requirement + reserve
    -> CHARGER_COMMITTED
```

Commitment remains absorbing until the charger is reached. SOC and distance
managers can run without an Energy estimator, which makes the decision comparison
fair and avoids unnecessary estimator calls.

The extraction is guarded by a seeded trajectory regression: a test-only copy of
the former inline two-boundary rule and the pluggable Quantile manager receive the
same initial state, task, actions, frozen estimator outputs, battery dynamics, and
seed. Their state trajectories and exact `CHARGER_COMMITTED` switching step must
match.

## 6. Stage B: Oracle Decision Headroom

Before training PCM, world models, ensembles, or a new reliability network, the
simulator must demonstrate that perfect Resource-to-Go information has decision
value.

The Oracle estimator clones the current environment and preserves:

- dynamics and physical limits;
- TelemetryCostModel;
- current position and velocity;
- current task and charger;
- static obstacles;
- LiDAR contract;
- HOCBF configuration and projection geometry.

It evaluates:

```text
RETURN NOW
FINISH CURRENT TASK -> zero service velocity -> RETURN
```

The Oracle never updates SAC or TD replay. All decision methods use the same
decision cadence. The default Stage-B cadence is every 10 policy steps plus the
first step of a newly sampled task; the old environment default remains every step.

### Smoke evidence, not a paper result

Latest code-path artifact:

```text
artifacts/return_decision_stage_b_smoke_20260826_cache_194739
```

This used a heuristic policy, a 400 m smoke map, no obstacles, two battery cycles,
and one parameter per method. It only validates the code path:

| Method | Stranding | Tasks/cycle | Tasks/simulated hour | Arrival energy fraction |
|---|---:|---:|---:|---:|
| SOC 0.20 | 0.00 | 14.5 | 226.83 | 0.140 |
| distance | 0.00 | 16.0 | 225.62 | 0.032 |
| Oracle | 0.00 | 15.5 | 230.98 | 0.084 |

The apparent Oracle headroom is encouraging but cannot support a scientific claim:
the sample has only two cycles, no trained SAC, no obstacles, and one seed.

The runner now evaluates every method/parameter on the same list of seeds, writes
seed-level and battery-cycle-level records, keeps evaluation transitions separate
from training interactions, and reports two-sided Wilson 95% intervals for
stranding. The formal gate uses the Wilson upper bound rather than the observed
failure fraction. In particular, zero failures in two smoke cycles has an upper
bound of approximately 0.658 and is correctly recorded as
`PENDING_INSUFFICIENT_CYCLES`, not as evidence of safety.

Formal seed rollouts can be executed concurrently with `--evaluation-num-envs`.
Parallelism is only across independent paired seeds; each worker reconstructs the
same frozen checkpoint, environment contract, method parameter, and cycle budget.
Aggregation remains cycle exact. The queued five-seed Gate uses five workers, so
parallel execution changes wall-clock cost but not the preregistered 100-cycle
evidence unit.

The deterministic Oracle caches the first exact task rollout and reuses suffix
energy when the live state follows that trace. It still recomputes `RETURN NOW` at
every decision. The latest smoke recorded 109 exact suffix hits and 33 misses;
208 rollout requests were needed instead of 426 requests without suffix reuse.
Caching is disabled for per-step decision cadence, where a step can invoke the
decision rule both before and after motion.

### Gate B

Run the frozen learned navigation policy with calibrated capacity, HOCBF obstacles,
paired seeds, reserve/SOC sweeps, and enough battery cycles for confidence intervals.

The preregistered first gate uses at least 100 cycles per method/parameter point,
a common 5% Wilson-upper stranding ceiling, and requires the Oracle to improve the
best eligible SOC/distance throughput by at least 5%. These are engineering pilot
thresholds for deciding whether to invest in learned baselines, not a theorem or a
paper-level safety certificate.

- If Oracle does not materially dominate SOC and distance frontiers, stop using
  return-to-charge as the ICLR headline.
- If Oracle shows stable headroom, proceed to learned estimators.

Post-Gate learned-method runs may evaluate only Frozen/Online TD or later learned
estimators, but they must explicitly inherit the formal `oracle_headroom_gate.json`.
The runner rejects TD-only formal runs without that artifact and rejects inherited
Gate files unless `status=PASS`, `evaluable=true`, and `passed=true`; it never
recomputes an empty Oracle Gate from a learned-method-only table.

## 7. Stage C: Minimal Prediction Baselines

Only after Gate B passes:

1. Frozen Quantile-TD.
2. Online Quantile-TD.
3. MC-supervised IQN-style quantile predictor.
4. PCM-Executed using the official policy-conditioning protocol.
5. Executed-action probabilistic world model.
6. Executed-action world model plus bootstrap ensemble.

All methods receive the same frozen SAC, HOCBF operator, source trajectories,
target-policy query privileges, evaluation seeds, and training budget. The generic
executed-action ensemble is the main simple-method threat.

The Quantile-TD preparation runner is:

```text
scripts/train_quantile_td_for_navigation_checkpoint.py
```

It is fail-closed: a formal run requires the 500k navigation readiness JSON and a
statistically evaluable `PASS` from the Oracle headroom gate. It then collects
exactly 500k TD transitions under the frozen 500k JSEB policy and runs ten
held-out evaluations at 50k intervals, each on the same independent 500-task set.
Held-out environments run in parallel, use central batched deterministic policy
inference, and load frozen CPU snapshots of the TD critic. The live critic's
optimizer, replay, update count, and parameters are audited before and after every
evaluation.

The current LiDAR/HOCBF contract produces a 2055D Energy-TD state (7 goal-motion
features plus 1024 ranges and 1024 validity values). A 7D checkpoint is therefore
not silently reused in this setting. Historical JSEB commands containing the old
Phase2A/2B/2C budgets are migrated explicitly to their unified summed energy
budget; all other unknown historical arguments remain errors.

The 2k runner smoke at
`artifacts/quantile_td_checkpoint_smoke_200107` reached the exact budget, preserved
the frozen policy hash, made 1,303 TD updates, retained 1,814 replay transitions,
and completed two parallel held-out evaluations. It correctly stopped with
`TD_NOT_READY` because the deliberately short smoke did not complete the far
distance bucket. This is a mechanics result, not Energy-TD evidence.

## 8. Stage D: Decision Baselines

Every estimator plugs into the same ReturnManager rule. The decision table includes:

- fixed SOC thresholds;
- distance times empirical energy per meter;
- direct high-level CONTINUE/RETURN switcher;
- frozen and online Quantile-TD;
- PCM;
- executed-action world model;
- world model plus ensemble margin;
- proposed reliability-aware estimator, only if justified;
- simulator Oracle.

CMDP methods such as CPO or SDAC belong to a separate end-to-end track because they
may retrain the navigation policy. They must not be mixed into the fixed-policy
prediction table.

## 9. Stage E/F: Closed-Loop Shift and Reliability

The decisive shift experiment is a complete 2-policy by 2-safety-operator
leave-one-pair-out design. Pair novelty and executed-interface extrapolation must be
controlled separately, and interface extrapolation must be matched against hitting
horizon.

The reliability method is implemented only if ordinary ensemble disagreement fails
to detect accumulated long-horizon ETG error. Its intended statistic is a
cross-fitted local interface error aggregated under target rollout occupancy, not an
ad hoc OOD classifier.

## 10. Primary Metrics

Mission metrics:

- stranding / energy-exhaustion rate;
- tasks per simulated hour;
- tasks per battery cycle;
- charger return success;
- pre-recharge unused energy at charger;
- premature-return rate under simulator counterfactual coupling.

Mechanism metrics:

- ETG underestimation;
- boundary-weighted underestimation;
- risk--coverage and AURC;
- horizon-stratified error;
- CRPS and q90/q95 pinball only when the probability-semantics gate passes.

Rare-event confidence is based on battery-cycle counts and paired cycle-level
bootstrap or Wilson intervals, not merely on a small number of random seeds.

## 11. Theory Obligations

The minimum aligned theory package is:

1. Executed-interface identifiability for unseen policy--filter pairs under target
   interface support.
2. Observational-equivalence impossibility outside identifiable interface support.
3. Occupancy-weighted long-horizon Resource-to-Go transport error.
4. Return-boundary decision stability: prediction error changes the decision only
   inside a corresponding boundary band.
5. Conditional return-failure implication under explicitly stated calibration and
   reserve assumptions.
6. Joint mission risk or valid component risk allocation.

No learned estimator is promoted to an unconditional hard energy-safety
certificate.

## 12. Evidence Status

```text
Stage A pluggable ReturnManager: IMPLEMENTED, TARGETED TESTS PASS
Stage B probability-semantics audit: IMPLEMENTED, CURRENT ENVIRONMENT DETERMINISTIC
Stage B Oracle runner: IMPLEMENTED, MULTI-SEED/CYCLE AUDIT SMOKE PASS
Formal Oracle headroom gate: PENDING
Learned baseline suite: PENDING
2x2 compositional shift: PENDING
Reliability method: NOT YET JUSTIFIED
CMDP decision track: PENDING
Second safety family/domain: PENDING
ICLR-level claim status: PENDING
```

The active historical JSEB/Quantile-TD long run remains a control experiment. It is
not evidence that this new return-first protocol has passed Gate B.
