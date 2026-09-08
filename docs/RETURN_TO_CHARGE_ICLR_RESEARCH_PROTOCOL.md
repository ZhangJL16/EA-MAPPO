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

Requirement-by-requirement completion evidence is tracked in
`docs/RETURN_TO_CHARGE_COMPLETION_LEDGER.md`; a live process or smoke never marks
a gated research stage complete.

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

The probability audit fixes state, goal, policy, safety operator, obstacle
realization, cost model, and all observed context, then repeats the exact Oracle
rollout. The repetition index is explicitly not described as a disturbance seed:
the current simulator has no future wind, transition-noise, or moving-obstacle
law to seed.

The read-only R5 platform diagnostic at
`artifacts/r5_return_probability_semantics_20260830/probability_semantics_audit.json`
loaded the wrapper-matched 500k checkpoint and reconstructed its 24-obstacle,
8x128-LiDAR, sampled-data-HOCBF environment. Three complete returns from the
same snapshot each consumed `14.3227585526` synthetic Energy units:

```text
conditional variance = 0
minimum = q50 = q90 = q95 = maximum = 14.3227585526
seeded future disturbance process = absent
navigation Gate = failed; downstream authorization = false
```

Therefore the current formal object is deterministic Energy-to-Go plus epistemic
prediction reliability. A stochastic q90/q95 headline is prohibited until a
physically motivated future disturbance process creates a stable non-degenerate
conditional return distribution. This diagnostic identifies the probability
object only; it does not repair navigation or establish Oracle headroom.

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

The corrected Quantile manager separates the stopping boundary from the
post-commit safety certificate:

```text
task-then-return is certified -> CONTINUE TASK
task-then-return is uncertified and return-now is certified -> CHARGER_COMMITTED
neither route is certified -> CHARGER_COMMITTED as an uncertified emergency
```

Direct-return infeasibility alone does not force commitment when completing the
task, applying the real service reset, and then returning remains certified.
The two feasible sets are not assumed nested because goal switching changes the
executed policy and task service resets velocity.

Commitment remains absorbing until the charger is reached. SOC and distance
managers can run without an Energy estimator, which makes the decision comparison
fair and avoids unnecessary estimator calls.

Nested legacy cases remain guarded by a seeded trajectory regression. New
regressions cover the non-nested viability-restoration region and require the
manager to preserve the certified task--then--return action.

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
descriptive stranding uncertainty. The formal gate instead uses simultaneous
one-sided Clopper--Pearson upper bounds, with Bonferroni error allocation over
the complete preregistered candidate grid, rather than an observed failure
fraction or a pointwise interval. In particular, zero failures in two smoke
cycles is correctly recorded as
`PENDING_INSUFFICIENT_CYCLES`, not as evidence of safety.

Formal seed rollouts can be executed concurrently with `--evaluation-num-envs`.
Parallelism is only across independent paired seeds; each worker reconstructs the
same frozen checkpoint, environment contract, method parameter, and cycle budget.
Aggregation remains cycle exact. The powered 320-seed protocol defaults to six
workers, so parallel execution changes wall-clock cost but not the preregistered
independent-cycle evidence unit. Changing worker count does not change schedules,
cycle IDs, confidence levels, or Gate sample size. One persistent worker pool is
used for the complete candidate grid, so the frozen policy is loaded at most once
per worker rather than once per worker and candidate point. For the default grid,
this reduces the maximum policy initializations from 72 to 6 without changing any
cycle or method pairing.

The launch manifest freezes all candidate points, the evaluation-seed hash, the
3840 total battery-cycle jobs, parallelism, the exact safety-power audit, and the
Oracle-shadow stopping scope before the first cycle runs. Non-Oracle shadow
rollouts are required through and including the first decision disagreement;
they stop after that event because the coupling theorem and audit deliberately
exclude the already-diverged suffix. The independently executed Oracle point
still runs for the complete cycle and supplies the paired outcome.

The deterministic Oracle caches the first exact task rollout and reuses suffix
energy when the live state follows that trace. It still recomputes `RETURN NOW` at
every decision. The latest smoke recorded 109 exact suffix hits and 33 misses;
208 rollout requests were needed instead of 426 requests without suffix reuse.
Caching is disabled for per-step decision cadence, where a step can invoke the
decision rule both before and after motion.

### Gate B

Run the frozen learned navigation policy with calibrated capacity, HOCBF obstacles,
paired seeds, reserve/SOC sweeps, and enough battery cycles for confidence intervals.

The preregistered first gate uses 320 independent evaluation seeds and exactly one
battery cycle from each seed per method/parameter point. This avoids treating 20
correlated continuous cycles from each of only five worlds as 100 independent
Bernoulli observations. The default grid contains 12 candidate points. It uses a
common 5% simultaneous stranding ceiling: each point receives a one-sided exact
Clopper--Pearson upper bound at Bonferroni level \(0.05/12\), so safety remains
95%-familywise valid after selecting the fastest eligible point. With zero
stranding, 100 cycles give upper bound 0.05333 and cannot certify the ceiling;
110 give 0.04860, but this zero-event-only design has just 0.331 certification
power when the true safe-point rate is 1%. Because the Gate needs both an Oracle
and a heuristic family to be certified, each designated safe family is powered
to at least 95%, which gives a dependence-robust joint lower bound of 90% by the
union bound. The powered design uses 320 cycles, allows at most six observed
strandings, has per-family exact-binomial power 0.956, and joint lower bound
0.912 at the preregistered 1% design rate. The Gate then
requires the Oracle to improve the best eligible SOC/distance throughput by at
least 5%. These are engineering pilot thresholds for deciding whether to invest in
learned baselines, not a theorem or a paper-level safety certificate. Multi-cycle
continuous recharge remains covered by mechanics tests and later mission runs; it
is not misused as the independence unit for this Gate.

The fixed sample size does not claim guaranteed power to distinguish exactly a
5% throughput gain without a preregistered paired-rate variance. Instead, the
selection-aware max-t interval implements a three-way decision: its lower bound
can establish headroom, its upper bound can establish insufficient headroom, and
an interval crossing 5% is scientifically inconclusive. No unobserved variance
assumption is introduced merely to promise a binary result.

The statistical guarantee is explicit. Let candidate \(j\in[K]\) have
\(X_j\sim\mathrm{Binomial}(n_j,p_j)\), and define

\[
U_j=F^{-1}_{\mathrm{Beta}(X_j+1,n_j-X_j)}(1-\alpha/K),
\]

with \(U_j=1\) when \(X_j=n_j\). Exact binomial coverage and the union bound give

\[
\Pr\!\left(\forall j\in[K]:p_j\le U_j\right)\ge 1-\alpha.
\]

Consequently, any data-dependent selected point \(\widehat j\), including the
fastest point satisfying \(U_j\le 0.05\), inherits
\(p_{\widehat j}\le U_{\widehat j}\) on the same event. No independence across
candidate methods is required. For zero events,
\(U_j=1-(\alpha/K)^{1/n_j}\); at \(K=12,\alpha=0.05\), 107 cycles are the exact
integer crossing and 110 is only a rounded minimally evaluable audit size; the
powered formal design is 320. This is a standard selection-valid evidence lemma,
not an Oral-level novelty claim.

- If the simultaneous Oracle-headroom upper bound is below the materiality
  threshold, stop using return-to-charge as the ICLR headline. If the lower and
  upper bounds straddle the threshold, record an inconclusive Gate rather than
  a scientific negative.
- If Oracle shows stable headroom, proceed to learned estimators.

A formal Gate failure is a scientific stop, not an implementation crash. An
inconclusive Gate also stops downstream work, but it is not evidence that
Oracle headroom is absent. In either case the runner writes
`STOPPED_AFTER_ORACLE_HEADROOM_GATE.json` and exits with code 4; it does not
write `FAILED.json` or return success. A passed Gate writes `COMPLETED.json`.

Post-Gate learned-method runs may evaluate only Frozen/Online TD or later learned
estimators, but they must explicitly inherit the formal `oracle_headroom_gate.json`.
The runner rejects TD-only formal runs without that artifact and rejects inherited
Gate files unless `status=PASS`, `evaluable=true`, and `passed=true`; it never
recomputes an empty Oracle Gate from a learned-method-only table.

The same distinction applies before Gate B starts. A complete prerequisite
audit that fails a fixed-platform identity, calibration-evidence, or endurance
condition is a named
scientific stop: the runner writes `STOPPED_PREREQUISITES_NOT_READY.json`, sets
`downstream_authorized=false`, and exits with code 4. It reserves `FAILED.json`
for missing, unreadable, malformed, or internally inconsistent inputs and
implementation errors.

Formal prerequisite identity is also fail-closed. Stage B accepts either a
legacy completion wrapper or the explicitly authorized
`fixed_navigation_energy_research_contract_v1`. The latter preserves the source
navigation-quality FAIL but treats success, distance-bucket, path, collision,
and boundary metrics as descriptive strata for a conditional fixed-platform
estimand. It verifies that the contract checkpoint SHA-256 equals the checkpoint
actually loaded for Stage B. Battery calibration records that checkpoint and
contract hash; battery validation records both plus the calibration-file hash.
Any missing or mismatched link produces `FAIL_GATE_B_PREREQUISITES`. A raw flat
navigation metric table remains useful for local metric tests, but cannot by
itself authorize a formal Oracle run.

The historical `0.98` overall success, `0.95` bucket success, `1.10` path ratio,
and zero-collision criteria were engineering platform-quality choices. They are
not assumptions of the stopped-risk theorems. The theorem layer instead requires
a fixed/queryable executed interface, oracle-shadow coupling through first
disagreement, a declared stopped-margin law, adequate quotient coverage/overlap,
and exponential transience. Poor navigation can worsen coverage, variance, or
external validity and can add a competing absorbing outcome, but no theorem
turns those effects into the four historical constants. The R3 analysis therefore
retains task limits, collision episodes/steps, boundary contacts, energy
exhaustion, return failure, charger arrival, and premature return as distinct
outcomes rather than conditioning them away.

Checkpoint identity is necessary but not sufficient: Stage B also reconstructs
the navigation environment from the named artifact's exact command. It reuses
the LiDAR tensor, obstacle population/radii, HOCBF coefficients and sampled-data
mode, projection geometry, task geometry, dynamics limits, and telemetry cost
configuration. Only the declared Energy-managed phase, calibrated battery
capacity, reserve sweep, and decision cadence are overridden. The R5 adapter
removes orchestration-only vectorization flags, maps its parser-compatible base
to R4, and then restores the R5 label and sampled-data-robust HOCBF bit; this is
recorded for the historical R5 route. The active R3 route reconstructs the R3
artifact's own exact command and hashes its config. Independent Stage-B CLI
defaults cannot silently replace the executed closed-loop contract.

### Historical 100k control classification

The frozen-100k control completed all 500 calibration tasks, but only 400 reached
their goals (`calibration_success_rate=0.80`; bucket rates 0.75--0.87). It therefore
fails the navigation/calibration prerequisite independently of later software
execution. Battery validation then encountered a separate implementation failure:
the long-running parent process had loaded the pre-extension worker call signature,
while newly spawned children imported the updated signature from a mutable worktree.
This hot-code mismatch is not an Energy-learning result. The artifact is retained as
a dual classification: insufficient 100k navigation and post-calibration worker
launch failure. Formal jobs run from the immutable clean worktree, and the worker's
new estimator-checkpoint argument now has a backwards-compatible default.

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

TD readiness is an engineering evidence Gate, not a stochastic-coverage claim.
The final independent 500-task evaluation must complete at least 95% overall and
95 of the 100 tasks in every distance bucket, produce finite ordered predictions,
report far-distance error, keep overall and far-distance MAE no worse than the
mean-energy scale of a zero-energy predictor, and avoid an underestimation rate
above 75%. Failure writes `STOPPED_TD_NOT_READY.json` and prevents TD decision
evaluation.

The current LiDAR/HOCBF contract produces a 2055D Energy-TD state (7 goal-motion
features plus 1024 ranges and 1024 validity values). A 7D checkpoint is therefore
not silently reused in this setting. Historical JSEB commands containing the old
Phase2A/2B/2C budgets are migrated explicitly to their unified summed energy
budget; all other unknown historical arguments remain errors.

The formal 500-task navigation checkpoint Gate supports parallel environments
through `--evaluation-num-envs` (default 6). Each worker keeps an independent
environment and task seed while one central deterministic policy call batches the
active observations. Evaluation interactions remain excluded from training
transitions and replay, task records are restored to their original fixed-set
order, and a progress JSONL is emitted every 25 completed tasks by default.

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

### 8.1 Finite-deadline Oracle semantics

The simulator Oracle is a finite-deadline completion Oracle. For hitting time
(T_g), nonnegative executed-loop cost (c_t), and the preregistered rollout
horizon (H), it returns

\[
E_H(x,g)=\sum_{t<T_g}c_t\quad\text{if }T_g\le H,
\qquad E_H(x,g)=+\infty\quad\text{otherwise}.
\]

The finite prefix accumulated up to (H) is retained only as a diagnostic when
the goal is not reached; it is not substituted for completed-goal ETG. A
a return-now deadline miss triggers emergency return only when the
task--then--return branch is also uncertified. If task completion followed by
the real service reset restores return viability, the Oracle continues the task.
A finite return-now branch with an infeasible task--then--return branch commits
immediately. Neither event asserts that the charger is unreachable at an
unbounded horizon. Formal records encode finite, infinite, and not-evaluated
states separately and count both infeasible branch types by method and seed.

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

## 10.5 Oral-Idea Contraction

### Venue lens and maturity

- **Target:** ICLR main track, theory-meets-RL/safe-autonomy family.
- **Current stage:** mature theorem exploration, empirically stopped before
  Oracle headroom.
- **Strongest honest contribution type:** a new task-specific theoretical
  synthesis and certificate for irreversible return decisions under executed-
  interface composition shift; not a new generic classifier, Bellman loss, or
  representation architecture.
- **Current weakness:** 33 individually numbered results create theorem sprawl,
  several components are classical, and the formal mission evidence is absent.
- **Development potential:** the full source-data-to-first-disagreement-to-
  Pareto chain remains conceptually coherent and testable if compressed.

### Normalized idea card

**Task.** Decide when an agent must irreversibly stop task execution and commit
to a charger when deployment combines a navigation policy and a safety operator
in a composition not labeled by source return trajectories.

**Gap.** Risk-sensitive RL/OPE can estimate transformed value objects, and
classification theory can transfer score error through a margin, but neither
piece alone certifies how source executed-interface information survives an
endogenous stopped target law, near-transient charger-hitting risk, and the
original stranding--throughput coordinates.

**Root challenge.** The target law is generated by the exact Oracle-shadow
stopping process, the executed action is a composition of policy and safety
filter, the killed exponential resolvent amplifies primitive error by the
transience gap, and only errors near the irreversible boundary change mission
behavior.

**Core insight.** The sufficient statistical object is a risk-observable
quotient of the *executed* interface plus a separate stopped-boundary law. The
quotient controls what must be transported; a dual certificate chooses between
average pushed-overlap stopped-\(L_2\) control and a faster target-active uniform
confidence route; first-disagreement coupling converts the tighter valid route
into a Pareto uncertainty rectangle.

**Mechanism.** Use independent structure, nuisance, estimation, and evaluation
folds to certify a quotient chart and killed first-passage score; count fresh
primitive transitions rather than replay draws; compute both the pushed-overlap
integrated certificate and per-active-cell confidence radius; use their valid
minimum for the manager certificate; compare every learned implementation with
matched-information risk-transformed Doob/DICE/KROPE baselines.

**Causal chain.** Certified executed-interface quotient
\(\rightarrow\) lower effective coverage/critical estimation radius
\(\rightarrow\) smaller stopped score certificate
\(\rightarrow\) fewer coupled first disagreements
\(\rightarrow\) a narrower stranding--throughput Pareto rectangle.

### Main-theorem contraction

Only the following three blocks should appear as main theoretical contributions
in the current state.

1. **Identification and equivalence boundary.** State when the executed closed-
   loop charger-hitting resource object is identifiable from source primitives,
   give the support impossibility boundary, and include matched-information Doob
   equivalence. This prevents an algorithm-name novelty claim.
2. **Quotient--transience information phase.** Combine the risk-observable
   quotient, fresh-transition martingale sample unit, Perron/transience
   amplification, and the declared local upper/lower phase. Scalar identities,
   individual perturbation lemmas, and allocation algebra move to the appendix.
3. **Dual stopped-Pareto finite-sample certificate.** Present the exact
   integrated and uniform routes, their assumptions and constants, the
   first-disagreement coupling, and the resulting mission-coordinate rectangle.
   State explicitly that the integrated route is not the classical Hölder
   minimax rate under strong density.

Two algorithmic corollaries are sufficient: (i) cross-fitted stopped-score loss
plus chart selection by the complete certificate, and (ii) cost-aware collection
of new target-active primitive transitions. Replay prioritization is an
optimization implementation detail and must not become a sample-complexity
claim.

A repaired Theorem 27 Assouad construction now supplies one common exact-chart,
source-equals-target first-passage lower family for quotient dimension,
transience, stopped margin, and Pareto disagreement. This strengthens blocks 2
and 3; it does not justify a fourth headline because the construction is a
classical strong-coverage Hölder--margin lower slice and does not cover arbitrary
pushed overlap, learned charts, or nuisance estimation. A fourth main theorem is
justified only by such a general lower result or by a declared computational/
hypothesis-class separation from matched transformed baselines. Adding another
identity or loss does not fill this gap.

### Grounding and inference boundary

Source-supported facts are: generic margin comparison and Hölder plug-in rates,
covariate-shift target regression, orthogonal/cross-fitted nuisance learning,
restricted OPE, KROPE representation stability, Doob transformation, Perron
perturbation, and Neyman allocation are established foundations. The local
literature packets record those collisions.

The optimizer inference is that the publishable unit, if any, is their
nontrivial compatibility in the executed-interface first-passage stopping
experiment and the resulting two-route mission certificate. Novelty of this
full package remains conditional until the joint lower-bound search and matched-
baseline evidence are complete.

### Overlap and evidence challenges

- **Overlap challenge:** a correct transformed OPE method can copy the chart and
  selector. **Revision:** claim no unrestricted method superiority; make the
  certificate/phase diagram the object and require matched information.
- **Evidence challenge:** theorem verifiers do not show decision value, and all
  R1--R5 navigation prerequisites failed. **Revision:** empirical promotion
  requires Oracle headroom first, then a preregistered \(2\pi\times2\Pi\)
  composition test showing certificate contraction and paired Pareto behavior.
- **Likely pivot condition:** if the Oracle upper headroom bound is below 5%, the
  learned-manager paper stops. A theory-only route then requires the missing
  common minimax lower theorem and is better aligned with COLT/AISTATS than an
  ICLR Oral empirical claim.

### Minimum convincing evidence after the Gate reopens

1. Compare against exact Oracle, SOC/distance managers, direct exponential TD,
   primitive world-model ensembles, and matched-information risk-transformed
   Doob/DICE/KROPE.
2. Hold out policy--filter compositions, report both
   \(\mathfrak C_h\) and \(p_{\min}\), and show which dual route is active before
   examining mission outcomes.
3. Sweep transience gap and quotient nuisance dimension to test the predicted
   phase rather than only average return error.
4. Ablate stopped weighting, quotient certification, uniform confidence, and
   cross-fitting; count independent cycles and fresh transitions, never replay
   draws.
5. Report first disagreement, certificate coverage/width, stranding, throughput,
   and failure cases. The central claim fails if the certificate contracts while
   paired Pareto behavior does not, or if matched transformed baselines inherit
   the same result without the declared restriction.

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
7. A simultaneous source-to-target EIRR certificate whose positive MGF intervals
   propagate through the exact two-branch ReturnManager and preserve a declared
   fraction of exact-risk stranding--throughput headroom.
8. A finite-interface attainability/limitation theorem separating exponential
   resource severity from executed-interface coverage, with primitive plug-in and
   DICE-style baselines required before any learned EIRR novelty claim.
9. A continuous risk-observable interface quotient composed with a separate
   stopped-boundary functional: exact preservation of the killed log-MGF grid and
   ReturnManager decision, approximate first-disagreement/Pareto stability, a
   quotient-complexity upper rate, and a matching representation/estimation lower
   bound. The boundary functional must not be hidden inside a one-step metric; a
   generic kernel embedding or bisimulation result does not satisfy this
   obligation.
10. A two-layer orthogonal EIRR score must keep target safety-execution
    composition and primitive killed-risk transition residuals separate. Its
    population target must be invariant when either the pair of risk-occupation
    ratios or the pair of continuation nuisances is correct, and its remaining
    bias must be the sum of two explicit nuisance-product terms. Cross-fitted
    root-\(n\) language is allowed only under iid/independent-episode validation
    or a declared trajectory CLT. This score is a DR/OPE estimator-design lemma,
    not an independent novelty claim.
11. The next novelty Gate is an information-geometry theorem, not another loss:
    derive the canonical gradient of the randomly stopped killed Feynman--Kac
    target under known \((\nu,L)\) and source primitive sampling, then compare its
    efficiency bound directly with DRL, OPRA-DR, MWL/MQL, and primitive plug-in.
    A viable oral-level route must prove a sharp upper/lower phase transition
    involving quotient coverage, exponential transience, and the stopped margin.
    If it is only the standard transformed-MDP efficient influence function, keep
    it as analysis and remove the standalone novelty claim.
12. The scalar critical theorem fixes the candidate phase variables:
    \(\Delta=1-a(1-p)\), exact risk-quotient source mass \(\mu_h\), raw-MGF
    information \(n\mu_h\Delta^4\), and deployed log-risk information
    \(n\mu_h\Delta^2\). Before promotion, replace the scalar gap by a simple
    Perron spectral gap, state a nondegenerate projected primitive-noise
    condition, prove matching upper/lower limits, and connect the resulting
    local scale to the stopped margin. A scalar Bernoulli calculation or fitted
    slope alone does not pass this Gate.
13. The multi-state result must keep three limits distinct: fixed-\(\Delta\)
    semiparametric LAN, the critical-family variance limit
    \(\Delta\downarrow0\), and a joint triangular array
    \(\Delta_n\downarrow0\). The first two do not imply the third. Any joint
    \(\Phi(-z)\) stopped-decision floor must state uniform DQM, Lindeberg, and LAN
    assumptions. A literature audit against Markov-chain conditioning, killed
    QSD perturbation/estimation, Feynman--Kac particle limits, SSP hardness, and
    margin theory is mandatory; the spectral exponents alone are not novelty.
14. The next algorithm-generating theorem is a cost-aware critical-mode source
    design. Its oracle priority must be proportional to target left-Perron
    occupation times the conditional standard deviation of the right-Perron
    continuation witness, divided by square-root acquisition cost. Since this is
    a Neyman-allocation specialization, oral-level promotion requires an adaptive
    oracle inequality that learns the quotient, Perron modes, and conditional
    projected variance near criticality. TD-error prioritization or another
    hand-designed energy loss does not satisfy this Gate.
15. Generic adaptive Neyman allocation and pilot-estimated stratum variance are
    prior art. The surviving theorem Gate is a critical-scale separation: under
    a uniformly conditioned Perron eigenprojector and nonvanishing projected
    noise, prove that
    normalized allocation learning needs \(m\to\infty\) but not
    \(m\Delta^2\to\infty\), while final log-risk resolution still pays the
    \(B\Delta^2\) information scale. Then give matching failure rates or
    impossibility when the Perron reduced-resolvent norm diverges, projected noise vanishes,
    or quotient recovery error is comparable to those scales.
16. The information-design result must reach the original irreversible mission
    criterion. Under an oracle-shadow stopped margin exponent \(\kappa\), derive
    a finite-confidence transfer from canonical variance \(V\) to
    first-disagreement/Pareto radius \(V^{\kappa/2}\) and local boundary-loss
    radius \(V^{(\kappa+1)/2}\). State whether the powers are tight, propagate
    the adaptive-oracle ratio, and distinguish a scalar-query design from one
    shared replay distribution serving many decision queries. A smaller OPE
    variance alone does not satisfy this obligation.
17. A practical replay sampler must serve a collection of stopped-boundary
    queries with one allocation. The declared objective is the margin-powered
    compound variance \(\sum_t\omega_tV_t^p\), with
    \(p=\kappa/2\) for Pareto disagreement or
    \(p=(\kappa+1)/2\) for local boundary loss. Derive and verify the global
    allocation, show how common versus query-specific critical gaps enter it,
    and then prove an adaptive guarantee while learning the query--stratum
    sensitivity matrix. Generic multiobjective optimal design or behavior-policy
    search does not satisfy the final adaptive Gate.
18. Continuous quotient learning must expose its intrinsic complexity instead
    of asserting that a neural encoder discovers it. On a declared local-mass
    Hölder quotient slice, prove the point-query rate, a nontransitive-safe
    pseudometric distortion sandwich, and the joint
    \(d_Q\)--\(\mathfrak g\)--\(\Delta\)--\(\kappa\) phase with a matching
    local testing order. KROPE is mandatory closest work: generic
    bisimulation-representation stability or Bellman completeness is not a new
    claim. Promotion beyond the declared slice requires learned-chart or
    dependent-replay control and a strict separation from Doob/DICE/KROPE.
19. Baselines must distinguish invalid risk-neutral collapse from a correct
    oracle-risk transform. An expected-reward KROPE loss may merge interfaces
    with equal mean resource and unequal exponential risk, while raw-interface
    DICE may pay nuisance coverage absent on the risk quotient. Nevertheless,
    KROPE/DICE on the true Doob-transformed chain is algebraically valid and
    mandatory. Any method claim must therefore concern learning/certifying the
    unknown transform and its stopped consequence, not the transformed Bellman
    equation itself.
20. Learning the Doob coordinates is not by itself a statistical contribution.
    With the cemetery coordinate retained, the map between the killed operator
    and transformed discounted kernel is invertible; direct and transformed
    plug-in values coincide for the same primitive estimate, and near-transient
    relative conditioning remains \(1/\Delta\). A proposed method must instead
    demonstrate lower quotient complexity, smaller decision-localized loss, or
    an explicitly computational advantage under matched information.
21. Dependent execution certificates count fresh primitive transitions, not
    replay minibatch draws. When interface eligibility is known before the
    outcome, first-hit residuals form a martingale difference sequence even
    under history-dependent policy/filter execution. The quotient radius may use
    unique chronological hit count \(m\); replaying one stored row \(R\) times
    must not replace it by \(Rm\). Retrospective outcome-selected encoders require
    a separate fold or post-selection analysis.
22. A global sup-norm value certificate is sufficient but not necessary for the
    irreversible decision. Under a stopped-margin exponent \(\kappa\) and a
    cross-fitted stopped-law score moment \(R_p=\mathbb E|E|^p\), \(p>1\), the
    required transfer is disagreement order
    \(R_p^{\kappa/(p+\kappa)}\) and boundary-loss order
    \(R_p^{(\kappa+1)/(p+\kappa)}\), with matching constructions. A proposed
    encoder/loss must establish the stopped-law moment without future leakage and
    beat matched risk-transformed Doob/DICE/KROPE there; global MAE or duplicated
    replay loss does not satisfy this Gate. This integrated route is sharp only
    given the \(L_p\) budget. It must not be called end-to-end Hölder minimax:
    under strong target-active coverage a uniform cell confidence band gives the
    faster classical plug-in boundary exponent.
23. Stopped-risk regression must measure source/target coverage after pushing
    both laws through an independently certified chart. For partition cells,
    report \(\mathfrak C_h=\sum_jq_j/p_j\), chart distortion, stopped
    pseudooutcome nuisance, intrinsic dimension, and transience gap. Raw-atom
    ratios retained after aggregation are invalid. Report the dual certificate:
    the pushed-overlap integrated route and, when valid, the target-active
    uniform route based on \(p_{\min}\) and a uniform nuisance bound. Take only
    the tighter *valid* bound and retain the margin-radius condition. A
    certified-chart estimator rate is only an intermediate Gate: Oral promotion requires a rate for
    learning the chart, or a matched-information separation showing that a
    correct Doob/DICE/KROPE baseline cannot inherit the same quotient gain.
24. Finite chart libraries may be selected by a uniform risk-witness diameter
    certificate. The certificate must report \(r_m\), true/empirical diameter
    sandwich, nonrepresentation term \(V_k\), and the
    \(m_{\min}\Delta^2/\log(ND)\) decision phase. This does not authorize an
    unrestricted superiority claim: under matched information and a retained
    cemetery coordinate, direct and transformed decision-rule classes are
    identical. Any claimed separation must predeclare the exact computational,
    hypothesis-class, regularization, or side-information restriction.
25. Oracle and Pareto promotion must consume a versioned paired evidence bundle.
    Each method decision is paired with an Oracle-shadow decision under the same
    task/obstacle schedule and is logged before the action using only available
    information. The independent uncertainty unit is a battery cycle, not a
    decision event. The bundle must retain the exact branch requirements, first
    disagreement, cycle stranding/throughput outcomes, descriptive Wilson
    interval, simultaneous exact safety upper bound, and
    all upstream Gate artifacts. The fail-closed field contract is
    `docs/RETURN_TO_CHARGE_ORACLE_PARETO_RESULT_SCHEMA.md`. R3's terminal-risk
    tilt is exploratory and cannot substitute for Oracle-shadow stopped
    occupation or paired cycle outcomes.

    Implementation note: Stage-B no longer treats equal RNG seeds as sufficient
    pairing. It generates tasks from `(evaluation_seed, cycle, task_index)`,
    because method-dependent return times otherwise alter task sampling. Every
    pre-action check now logs both decisions and marks only the stopped first
    disagreement; cycle outcomes are joined only at the same schedule and
    reserve. The smoke path validates this contract but never promotes a Gate.
26. Oracle Headroom may not pass from a throughput point estimate alone. Among
    points whose simultaneous familywise stranding statistic clears the common
    ceiling, resample the common independent schedule IDs jointly. Construct a
    studentized maximum-deviation rate band over every preregistered Oracle,
    SOC, and distance point, not only the observed safe subset. The formal gain
    lower bound is the maximum eligible Oracle lower-rate bound divided by the
    maximum eligible heuristic upper-rate bound, minus one. With confidence
    0.95, 10,000 replicates, and fixed seed 20260830, both the observed gain and
    this full-family lower bound must exceed the preregistered 5% material-
    headroom threshold. The selected-max percentile interval is descriptive
    only. Failure of the lower bound to clear 5% is not, by itself, evidence of
    insufficient headroom; it stops as `INCONCLUSIVE_ORACLE_HEADROOM` unless a
    separately constructed simultaneous upper bound is below 5%.
27. Oracle-shadow first disagreement must be backed by an independently executed
    coupled Oracle path, not only an in-place extra model query. Join events by
    keyed schedule, reserve, pre-action step, information time, and task index.
    Through and including the first disagreement, require equality of position,
    velocity, task goal, remaining Energy, exact effective requirement, and the
    Oracle commit decision; after commitment decisions diverge, do not require
    the paths to remain equal. Missing or mismatched events stop as
    `FAIL_ORACLE_SHADOW_EVIDENCE_INTEGRITY`. Downstream runs may inherit only a
    PASS artifact that explicitly records `evidence_integrity_passed=true`.
28. Safety-frontier selection must not use pointwise 95% intervals as though
    they were simultaneous. For the complete preregistered Oracle/SOC/distance
    family of size \(K\), use one-sided Clopper--Pearson upper bounds with
    per-point error \(0.05/K\). For the default 12-point grid, 107 cycles are the
    zero-event mathematical minimum and 110 are only minimally evaluable. They
    are not sufficiently powered for formal promotion. The formal design uses
    320 independent cycles per point and a 1% design stranding rate. Each of the
    two required safe families has at least 95% certification power, yielding at
    least 90% joint Gate power without assuming independence; pointwise Wilson
    intervals remain descriptive only. Any grid expansion is automatically
    included in \(K\) and triggers a new power audit.
29. Throughput inference must cover the complete candidate family before the
    random safety subset is selected. If simultaneous bands satisfy
    \(\theta_o\ge L_o\) for every Oracle point and \(\theta_h\le U_h\) for every
    heuristic point, then for arbitrary data-dependent eligible sets
    \(\widehat O,\widehat H\),
    \[
    \frac{\max_{o\in\widehat O}\theta_o}
         {\max_{h\in\widehat H}\theta_h}-1
    \ge
    \frac{\max_{o\in\widehat O}L_o}
         {\max_{h\in\widehat H}U_h}-1.
    \]
    Stage-B uses a paired schedule studentized max-t bootstrap to approximate
    this joint rate-band event. Unlike the stranding bound, this bootstrap is
    asymptotic rather than finite-sample exact; the Gate names that limitation
    and does not promote the old selected-max percentile interval.
30. Minimal evaluability is not statistical power. For \(X\sim\mathrm{Bin}(n,p)\),
    the largest failure count certified by the familywise CP rule is
    \[
    c_n=\max\{x:\Pr_{p=0.05}(X\le x)\le0.05/K\},
    \]
    and per-family power at design rate \(p_0\) is
    \(\pi_n=\Pr_{p_0}(X\le c_n)\). If \(F\) safe families are required, their
    joint certification power is at least \(1-F(1-\pi_n)\). Stage-B must record
    this exact audit and reject a formal launch whose joint lower bound is below
    0.90. With \(K=12,p_0=0.01,F=2\), 110 cycles have \(c_n=0\), per-family
    power 0.331, and a vacuous joint lower bound; 320 cycles have \(c_n=6\),
    per-family power 0.956, and joint lower bound 0.912. Changing the grid,
    ceiling, confidence, design rate, or required families requires recomputing
    the audit rather than inheriting these values.
31. Oracle Headroom uses two directional one-sided simultaneous tests, not a
    point-estimate dichotomy. In addition to obligation 29, construct bands
    \(\theta_o\le U_o\) for every Oracle point and \(\theta_h\ge L_h\) for every
    heuristic point over the full candidate family. Then
    \[
    \frac{\max_{o\in\widehat O}\theta_o}
         {\max_{h\in\widehat H}\theta_h}-1
    \le
    \frac{\max_{o\in\widehat O}U_o}
         {\max_{h\in\widehat H}L_h}-1,
    \]
    whenever the selected heuristic lower denominator is positive. Declare
    `PASS` only when the point estimate and the one-sided 95% lower gain bound
    both clear 5%; declare `FAIL_INSUFFICIENT_ORACLE_HEADROOM` only when the
    separately calibrated one-sided 95% upper gain bound is below 5%; otherwise
    declare `INCONCLUSIVE_ORACLE_HEADROOM`. These are separate directional 95%
    statements, not a claimed 95% two-sided interval. A zero heuristic lower
    denominator makes the upper bound unavailable and therefore inconclusive.

No learned estimator is promoted to an unconditional hard energy-safety
certificate.

The corrected statements, assumptions, counterexample construction, and proof
obligations are written in `docs/RETURN_TO_CHARGE_PROOF_PACKAGE.md`.  The base
transport theorem explicitly assumes measurable one-step couplings and Lipschitz
continuation laws, while the return-failure corollary requires a measured
decision-interval overshoot bound. The finite-state end-to-end certificate now
reaches the population Pareto criterion, but still assumes rather than derives
the learned density-ratio/Doob-transform nuisance rate. An attempted fresh-agent
proof audit did not return a result and is recorded as reviewer-infrastructure
failure, not proof acceptance.

The environment now records the previous decision check, the margin drop at
commitment, threshold-crossing overshoot, energy consumed between checks, change
in governing required energy, and whether the configured reserve covered the
observed interval drift.  Energy drift is logged only for managers whose margin
unit is `synthetic_simulation_energy_units`; SOC margins remain explicitly
dimensionless fractions.

## 12. Evidence Status

```text
Stage A pluggable ReturnManager: IMPLEMENTED, TARGETED TESTS PASS
Stage B probability-semantics audit: IMPLEMENTED, CURRENT ENVIRONMENT DETERMINISTIC
Stage B Oracle runner: IMPLEMENTED, MULTI-SEED/CYCLE AUDIT SMOKE PASS
Decision-interval overshoot telemetry: IMPLEMENTED, 114 RELATED TESTS PASS
Theory proof package: THEOREMS 1--33 DRAFTED/VERIFIED, INDEPENDENT REVIEW NOT COMPLETED
Historical navigation-quality Gate: R1--R5 COMPLETE, ALL FAILED (DESCRIPTIVE)
Fixed empirical platform: R3 USER-SELECTED, CONTRACT HASHED, POLICY FROZEN
R3 500-task battery calibration: ATTESTED, CAPACITY 383.35430890654663
R3 refined capacity: 304.9538842289515, HASH-BOUND TO CONTINUOUS-WORKLOAD SOURCE
R3 100-run confirmatory endurance: PASS; 100 DEPLETED, 0 CENSORED, MEAN 1668.3655 S
Formal Oracle headroom gate: PENDING CORRECTED VIABILITY-ORDER PREFLIGHT; PRIOR FORMAL RUNS INVALID
Current Quantile-TD baseline: IMPLEMENTED, WAITING ON ORACLE GATE
MC-IQN / PCM-Executed / executed-WM / ensemble suite: GATED, NOT STARTED
2x2 compositional shift: PENDING
Reliability method: NOT YET JUSTIFIED
CMDP decision track: PENDING
Second safety family/domain: PENDING
ICLR-level claim status: R3 CONDITIONAL EMPIRICAL CHAIN REOPENED, NO PARETO RESULT YET
```

The chain remains fail-closed on identity and evidence integrity, but no longer
uses the later engineering navigation thresholds as theorem assumptions. The
user selected R3 as the frozen conditional platform. The derived contract at
`artifacts/r3_fixed_baseline_energy_chain/navigation_platform_contract.json`
keeps `legacy_navigation_gate_passed=false`, binds checkpoint SHA
`02c1ecf55c24ebb2652507ba0c751ff81f7bf8d2cfbdefc39adb88d5099f910a`,
and authorizes only

```text
Oracle-vs-heuristic stranding--throughput difference | frozen R3 + HOCBF.
```

Navigation success, path ratio, and collision outcomes remain mandatory report
columns and sensitivity strata; they are not launch thresholds. No R6 repair is
authorized.

The separately declared R3 exploratory Energy run completed 500k frozen-policy
Energy transitions and 930 complete trajectories. Its 500-task calibration also
contains reusable raw measurement evidence: 475 successes, 25 retained
`task_step_limit` outcomes, unchanged policy parameters, no SAC/TD training, and
calibrated capacity `383.35430890654663`. The attestation recomputes mean power
and capacity from those records and binds every source hash without editing the
historical file. The old monolithic parallel validation ended with `EOFError`;
the replacement validation writes ten-run batches independently so completed
batches survive a later worker failure. Its risk-tilted occupation audit is stored under
`artifacts/r3_frozen_energy_exploratory_20260829_173003/`
`risk_tilted_occupancy_analysis/`. It is mechanism evidence only: the observed
energy--horizon correlation is 0.9802, so interface analysis must match horizon
rather than interpret exponential concentration as interface causality. That
within-composition audit is now complete. Under five contiguous held-out blocks,
aggregate executed-interface summaries improve total-Energy MAE over a strong
post-trajectory confounder model by 3.9041%, with paired moving-block-bootstrap
absolute improvement 0.028031 [0.006198, 0.051350] and approximate conditional-
permutation upper-tail 1/201. The highest intervention quintile has lower mean
residual but substantially larger residual variance, q95, and normalized log-MGF.
This supports a conditional tail-risk mechanism, not a positive intervention-cost
coefficient. Stage E remains locked because these future trajectory summaries are
not deployable and R3 contains no held-out policy--filter composition.

An additional historical non-learning platform diagnostic on 2026-08-30 used the existing
deterministic go-to-goal controller with the frozen R5 map and sampled-data HOCBF
contract. Its 10-task execution smoke completed 9/10 tasks with zero collision
and boundary-contact steps and mean path ratio `0.996465`. This is smoke-only and
does not define the active chain and is not promoted to another 500-task attempt.
The former stop audit is retained as historical evidence and marked superseded
by the user-selected R3 contract.
