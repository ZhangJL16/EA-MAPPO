# Notes: Return-to-Charge Research

## 2026-09-08 new-navigation return-energy research launch

Completed correction adaptation131072 and pairedfixed500. Read-only descriptive
results-analysis audit: raw484 arrivals/450safe/16timeouts/40072contacts;
HOCBF492arrivals/492safe/8timeouts/1contact. HOCBF safe gains42/losses0 on paired
tasks, but interventions22.91% (74875/326837steps); one contact in a timeout.
No matchedlambda0 adaptation or independent training seeds: do not claim the
auxiliary loss alone caused improvement, zero failure, or a safety certificate.

Continue with isolated frozen-new-policy return energy collection, actualknown
charger, fresh legal moving starts, unchanged2056 LiDAR inputs/locked collisions.
Budget500000 real envsteps explicitly announced as working500k interpretation;
not navigation or energy-optimizer update count. Save all per-step energy/contact
and every16th predictor observation. Actor still consumes everyfull sensor packet.
Suffixlabels exclude only future contacts; incomplete budget tails remaincensored
and resumable. Scene-seeded70/20/10split prevents same-scene temporal leakage.

Artifact artifacts/new_navigation_return_energy_500k_20260908_v1, PID96663.
Four focusedtests passed includingtiny8→16step real-modelcontinuation. First8192
formalcheckpoint healthPASS:16completeepisodes/397samples/1959pendingtransitions,
8fullworker snapshots,79.84MB checkpoint, finiteobservations/correctunifiedlabels,
sourcehashesmatch, processalive, noERROR, navigation_updates0. Stopmonitoring.
Collection endsat500000 and awaitsanalysis; energy-head fitting notrunning yet.

Next: meanMC, conditionalquantileMC andfactorized defective-law comparisons on
shared scene-held-outnew-policydata. Do not mislabel pinballMC asQuantileTD.
Correct survivalBellman target kills a predicted zero-contact EVENT, notphysical
episodes. PartialLiDAR is not guaranteedMarkov: exactstateBellman claims cannot
be silently imposed onobservations. New protocol documents this limitation.
No fullmission switching/batteryexhaustion claim fromreturn-onlydata.

First launcher used Path.resolve on .venv/bin/python, following symlink tosystem
Python and failing numpyimport beforecreatingoutputs; relaunched viaabsolute
venvpath. No lostdata. Error remainsinexternalconsolelog forprovenance.
Protocol docs/NEW_NAVIGATION_RETURN_ENERGY_500K_PROTOCOL.md.

## 2026-09-08 authorized fixed500 successor

User explicitly requested post-learning500-task testing then energy research
on the new navigator. Added independent evaluator and detached queue; no live
training source changed (all runtime hashes verified). Queue PID19195 healthy
WAITING_FOR_TRAINING; binds final131072 adaptation checkpoint, not latest midrun.
Training PID9399 alive,53248 adaptation transitions at handoff. Evaluation runs
the same fixed500 tasks twice (raw/HOCBF), preserves locked collisions, commits
groups of8 and resumes without rerunning prior committed groups. Queue stops
on training pause/error/changed source and after evaluation; no auto-promotion.
Six focused tests and one actual checkpoint5×2×2 smoke passed; smoke explicitly
nonformal. Evaluation output hocbf_correction_fixed500_20260908_v1, queue output
hocbf_correction_followup_20260908_v1. No formal evaluation/energy run started yet.
Energy500k meaning queried asynchronously; working plan freezes new navigation
for500000 extra energy-stage transitions. Prior JSEB pipeline is not a drop-in
replacement; old-policy labels/calibration must not be mixed with new-policy
truth. Mean/MC, Quantile-TD and defective safe-return-law are separate planned
comparisons. Collector/interface port and experimental semantics remain work.
See docs/HOCBF_FIXED500_AND_ENERGY_HANDOFF_20260908.md. Stop monitoring now.

## 2026-09-08 correction-supervised SAC launch

User explicitly superseded the previous stop with “请继续实验”.14 focused
tests pass, including nonzero auxiliary supervision, lambda0/native SAC exact
updates, real-source warm-start identity, full-state numerical continuation,
and one16-step subprocess smoke. Separate source files preserve historical code.
Detached PID9399 launched `artifacts/hocbf_correction_sac_20260908_v1` for131072
additional transitions from the raw524288 checkpoint. First learned checkpoint
8192 verified: actor/Q3192updates each, finite metrics, intact files/source hash,
no ERROR, PID alive.399auxiliary updates/395nonzero;3192teacher queries/3176valid,
1496corrected and16solver-invalid rejected. This is startup evidence, not policy
improvement evidence. Stop monitoring. No auto-evaluation, comparison arm, or
promotion launched.

## 2026-09-07 correction-supervised SAC implementation (initial code-only request)

Implement only; user explicitly stops before experiment launch. Keep the recent
524288-step raw SAC architecture/weights, not the historical R3/JSEB architecture.
Use fresh HOCBF teacher queries from the same replay observation and current
reparameterized action; reconstruct local points from all1024 sensor ranges and
hit flags, recover velocity using declared actuator/sensor constants. No layout
or cached Jacobian enters learning. Query a small subbatch periodically to bound
QP work; do not resimulate trajectories. SAC retains nominal-action semantics.
Warm start policy/critic/entropy parameters, but clear old unshielded replay and
optimizers; full subsequent checkpoints preserve optimizer/replay/RNG/workers.
Native execution uses substep HOCBF; teacher predicts the anchor-state single
projection, not mean substep actions. Float32 observation reconstruction is not
an exact identity to float64 sensor packets; quantify tolerance in focused tests.
Runtime fallback or infeasible/uncertified teacher outputs never become labels.
Keep existing effective reward: explicitly zero the dormant intervention penalty
when enabling HOCBF, so no extra reward term is silently activated.

## Final v3 fixed500 evaluation launch

User approved after v3 training completed524288. Added an isolated evaluator
with unchanged v2 sampling/parking/metric/batch-resume AST and v3-specific
protocol/model validation. One5-task×2-step final-model GPU smoke passed.
Formal output deployable_derived_hit_v3_fixed500_20260907_v1, PID20663.
Startup 200/500 committed and prefix validated, no ERROR, process alive.
HOCBF remains off. Stop monitoring; await user for full paired analysis.

## 2026-09-07 v3 restart after interruption

User explicitly requested continuing the existing task. No live trainer was
present. Matched source/config and complete393216 checkpoint restored with
model/optimizer/replay/RNG/worker state; old417792 logs preserved as
training_records_interrupted_417792_20260907.json and corresponding status copy.
Detached PID3998 advanced to397312 with392312 actor/critic updates and finite
metrics, no ERROR; first resumed block141.94s. Full checkpoint still393216,
next periodic425984. No code or experiment changes, no repeated resume gates.
Stop checking after this startup health; user will request results later.

## 2026-09-07 fixed500 complete, derived-hit ablation

Launch handoff: PID101139, deployable_derived_hit_v3_sac_20260907_v1. Committed8192
with3192 actor/critic updates, finite metrics, full checkpoint, no ERROR;1039
external dimensions and2 internal channels verified. Stop monitoring until
user asks. All current compared/navigation arms have HOCBF disabled.

v2 regressed:461/500 goal,372/500 safe,39timeouts,221.222mean contacts vs old
raw SAC496/423/4/29.436. Both HOCBF disabled; user recollection refers to
separate raw/HOCBF return-interface tests. Paired safe85 losses vs34 gains,
RD−.102, stratified paired bootstrapCI[−.144,−.060]; conditional frozen-model
task inference only, one training seed0 each.30 contact timeouts contribute
95481/110611 contacts; common458 arrivals still use more energy/path in v2.
Bundle with statistics, paired assumptions/limitations and inspected figures:
artifacts/deployable_v2_fixed500_analysis_20260907_v1/.

Next single-factor test preserves all1039 external observations and v2 context,
reconstructs hit=(normalized_range<1) inside first conv; extra channel weights
zero-init, old weights and global RNG stream preserved (whole SAC tested).
144 extra weights/encoder only; neither extra truth observations nor HOCBF.
Same scratch seed0,524288 steps, identical train/checkpoint function and env.
4 focused tests passed including one tiny16-step runner smoke. No new safety
or energy theorem claimed. See docs/DERIVED_HIT_ENCODING_ABLATION_V3.md.

## 2026-09-07 completed v2 training audit / fixed500 next step

Evaluation launch handoff: Python PID93949,
artifacts/deployable_observation_v2_fixed500_20260907_v1. Initial verification32/500
committed, immutable source-index alignment and safe-arrival semantics valid,
no ERROR, process alive. Do not monitor to completion or auto-start a follow-up.

Completed524288 transitions,128 learning blocks,753 finished training episodes;
actor and critic each519288 updates. All recorded update metrics finite;
final model/replay/state sizes and all recorded training source hashes match.
Training loop time4.399877h vs previous SAC4.676558h (~5.9% less, not a controlled
hardware benchmark and excludes checkpoint writes). Collection0.267607h;
remaining4.132270h is updates/loop overhead. No claim of2x acceleration.

Last200 training episodes:199 arrived,151 zero-contact arrivals,1timeout,
mean35.46 contacts,median0; successful path ratio1.10816. Last100:99 arrived,
76 zero-contact. Training is stochastic and on a changing policy; these are
descriptives, NOT fixed500 evaluation and NOT independent training seeds.
Previous SAC last200:200 arrived,176 zero-contact,mean11.945 contacts,path1.03955;
different completed task mix and evolving policies block a formal winner claim.
New last200 contact total7092: worst task3149 contacts/4000steps and timeout;
next1333 contacts/2104steps but eventually arrived. Top two account63.2% of
contacts; mean alone hides the long contact tail. Cause not identified yet.

All753 training episodes used task mode and initial SOC1; none exhausted the
ledger. Last200 terminal SOC ranged~.5782.. .9888. Resource objective and
termination do not depend on budget. This run does not train energy safety or
return-mode decision-making; do not propose indefinite extra navigation steps
as a way to fix missing task supervision.

Core formal evidence missing: results-analysis read-only audit mode, no p-values
or publication figure/report. Next experiment is the unchanged immutable500
five-distance task set with frozen final v2 actor, paired by source task index
to old SAC outputs. Dedicated evaluator parks finished workers, validates
resume prefix, hashes model/source/protocol, preserves training artifacts.
Three focused evaluator tests passed; one actual final-model5-task×2-step
smoke checks the new inference path, not efficacy. No preliminary performance
gate or automatic subsequent training. See docs/DEPLOYABLE_V2_FIXED500_PROTOCOL.md.

## 2026-09-06 deployable observation v2

Launch handoff: Python PID6740; output deployable_observation_v2_sac_20260906_v1.
8192-step committed checkpoint verified,3192 actor and3192 critic updates,
all metrics finite, complete model/replay/state, no ERROR, process alive.
Stop monitoring after this startup check; user will request results later.

New isolated 1039-D observation and single-channel 8x128 ordered LiDAR encoder.
Preserve all distances; remove deterministic threshold-mask channel. Add charger
relative direction/log distance, observed budget, explicit mission command, and
last policy-step unified contact. Old collectors and 2056-D policies unchanged.
Navigation agent.energy is a constant dummy; new observation uses the declared
initial budget minus existing cumulative realized telemetry, not future labels.
Budget source is the old R3 synthetic capacity, not a newly validated endurance
guarantee. Keep reward, physics and task sampling unchanged. Therefore training
is navigation with new observations, NOT task/return switching or learned energy
sustainability; return-command support exists but task-only sampling does not
train it. 13 focused tests passed including exact checkpoint continuation and
one tiny subprocess smoke. User subsequently authorized launch; check first
learned checkpoint, then hand off. See docs/DEPLOYABLE_OBSERVATION_V2.md.

## 2026-09-04 Certified Meet Fusion

- The corrected one-sided hazard Gate completed on 150 scenes.  It achieved
  Brier 0.102854 versus geometry 0.110672, fixed-0.90 dangerous false-safe
  0.03190 versus 0.03820, CRC risk 0.03478, zero dominance/monotonicity
  violations, and four of five fold wins.  CRC coverage rose only from 0.39203
  to 0.40049, failing the preregistered +0.02 requirement; no fresh confirmation
  is authorized for that branch.
- The meet control `min(q_geometry, q_direct)` achieved CRC risk 0.02825 and
  coverage 0.41897 in the same exploratory crossfit, motivating a separate
  method with exact lattice semantics rather than more hazard MLP tuning.
- Final development freeze uses 90 train, 30 validation, and 30 CRC calibration
  scenes.  On the calibration role, meet risk is 0.01826 with coverage 0.72150;
  geometry risk is 0.01825 with coverage 0.64994.  These are threshold-selection
  diagnostics, not confirmation evidence.
- Evaluator preflight on already inspected data showed why risk must be compared
  with the fixed alpha rather than forced below a more conservative baseline.
  Protocol v1 was retired before fresh data; v2 preserves the alpha=0.05
  selective-risk objective and keeps baseline risk as a reported diagnostic.
- The v2 freeze, evaluator, source optimization, and causal fork path pass 32
  focused tests.  A five-scene nominal-only/fork smoke completed 5 source
  rollouts and 50 branches with exact observation/layout match and complete
  action diversity.
- The formal v2 fresh chain is live in unified session 87525.  One-time health
  check saw 30/75 nominal source rollouts complete at 37.47 seconds with eight
  active and no failure marker.  It will then collect 3,000 fork branches and
  run exactly one frozen evaluation; do not monitor until the user asks.
- The nominal source finished 75/75 in 234.85 s.  Forking then exposed a
  float32 boundary canonicalization mismatch at scene 34 anchor 2 (z
  0.4999504 versus the legal/worker-canonical 0.5 m).  No branch existed for
  that anchor and 1,373 completed atomic branches were retained.  Anchor
  preparation now applies the same ulp-limited position sanitizer as workers;
  resume session 68753 was healthy at 1,374/3,000 branches.

## 2026-09-01 R7 Exact-Semantics Runtime Acceleration

- R3 SAC's higher GPU duty was caused by 495,000 batch-256 replay updates plus
  about 490,000 Jacobian-bridge actor updates. Its simulator remained in eight
  CPU subprocesses and used ordinary HOCBF with top-16 selection.
- R7 collected 2,000 transitions for about 73 s and then used CUDA for about
  1.55 s per update. Its sampled-data robust, no-top-k contract exposed a
  separate CPU bottleneck: Python hit-object construction and full QP sweeps.
- The robust LiDAR path now evaluates ordinary HOCBF arrays and the exact scalar
  sampled-data residual formula in a batch. Reference and optimized paths agree
  on all non-timing diagnostics.
- Large QPs use exact constraint generation with an all-row acceptance scan.
  Numerically inconclusive restricted problems retain the established full
  solver/fallback behavior.
- The unchanged three-variable full-solver loop has a C implementation built by
  `scripts/build_hocbf_native.py`; a missing/stale shared library falls back to
  Python. Native/reference outputs agree within `1e-11` with identical status,
  iteration, active-set, and reason diagnostics in the dedicated regression.
- Deterministic near-obstacle probes reduced four-substep policy-step time from
  about 78--236 ms to 8--18 ms. Verification passed 80 collision/projection and
  96 environment/R7 tests. See `docs/R7_RUNTIME_ACCELERATION.md`.

## 2026-09-01 ICLR Readiness and Story Audit

- Current submission verdict is `3/10 Reject`, while development potential is
  promising. The distinction is evidence-based: the main paper has seven
  main-text pages and a coherent executed-interface question, but every central
  empirical cell remains pending.
- The frozen R3 contract directly contradicts an unconditional “hard collision
  authority” story: `legacy_navigation_gate_passed=false`, overall success 0.96,
  mean path ratio 1.1880, three collision episodes, and 2,242 collision steps.
  Its own non-claim forbids converting conditional energy-study authorization
  into deployment-quality navigation/collision evidence.
- The best-fit collision infrastructure is a sampled-data robust HOCBF with
  bounded inputs, inter-sample compensation, safe initialization, verified
  solver/fallback behavior, and exact swept-clearance audit. Recent Composite
  CBF work is a strong quadrotor comparator, but the local `aggregate_hocbf`
  prototype is not a faithful implementation and must not be labeled as one.
- The collision layer must be frozen before energy calibration. Replacing the
  filter changes executed occupancy, path time, and resource use, so existing
  calibration remains historical evidence but cannot authorize downstream
  Oracle/learned/Pareto claims for the new stack.
- Continuous-workload endurance attempt 3 passed its declared calibration Gate
  (100/100 depletion, zero censoring, mean 1,668.3655 s, -7.313%), but formal
  Oracle Decision Headroom remains absent. The Oracle Gate remains the kill test
  before learned Resource-to-Go work.
- Existing controlled filter diagnostics support only a next-step hypothesis:
  sampled-data HOCBF observed zero collisions under exact controlled conditions,
  while dropout/combined sensing stress produced collisions and unsafe fallback
  events remained. Zero observed collision is not a certificate.
- ICLR 2027 AI disclosure is a blocking integrity issue. The current planned
  “literature/citation/reference only” statement does not match the broader
  project assistance in theory, methodology, experiment design, implementation,
  interpretation, and writing; author rewriting does not erase that history.
- Canonical outputs:
  `ccfa-review-reports/when-should-an-agent-return-iclr-review.md`,
  `docs/ICLR_ENERGY_SAFETY_STORY_BLUEPRINT.md`, and
  `literature-search-20260901-collision-safety-layer/`.

## 2026-09-01 R6 Learned-Navigation Audit

- The frozen R3 run already used a structured 2D LiDAR convolution plus a local
  Jacobian bridge, but remained an off-policy SAC system. Its 500-task result was
  0.96 success, 1.188 mean path ratio, 0.3587 HOCBF intervention-step rate, and
  0.1516 emergency-brake-step rate. It therefore does not establish that the
  nominal actor independently learned robust navigation.
- `UAVEnergyDeliverySACEnv.step` already distinguishes the policy's nominal
  action from the mean executed HOCBF action and reports progress, collision,
  fallback, and intervention diagnostics. R6 can consume this contract without
  changing R3 dynamics or safety semantics.
- There is no project model Registry/Factory. The `architecture-design` skill is
  therefore inapplicable; R6 is a standalone versioned experiment package.
- R6 uses an independent Beta action distribution on [-1,1], goal-conditioned
  attention over LiDAR rays with physical ray directions, and a GRU driven by
  previous executed action/intervention/progress feedback. No route, waypoint,
  heuristic action, PD control, or deadlock state enters the network.
- PPO likelihoods are always computed for the sampled nominal action. The
  transition is generated by the executed action, while a continuous
  intervention-weighted distillation loss moves the actor mean toward the safe
  executed action. Collision and intervention are separate CMDP costs with
  learned nonnegative dual multipliers.
- GPU is available (`NVIDIA GeForce RTX 5060 Laptop GPU`, PyTorch 2.7.1+cu128).
  Environment workers remain CPU processes with one OMP/MKL thread each; actor
  inference and every PPO minibatch update run centrally on CUDA.
- R6 smoke v1 completed its 4096 training transitions but its deliberately weak
  policy made the 10-task, 4000-step evaluation expensive. It was interrupted
  only after the training checkpoint and HEALTH Gate were complete. This is a
  retained non-scientific performance-path diagnostic, not a failed trainer.
- After setting OMP/MKL/OpenBLAS/NumExpr before numerical imports, every worker
  showed one OS thread. The thread-cap v2 smoke completed 4096 transitions in
  50.00 s; 91 R6/environment regressions passed.
- The v2 MSE critic loss rose to 60.28 and pre-clip gradient norm reached 56.36.
  Huber value regression reduced the final smoke value loss to 5.63 and maximum
  pre-clip gradient norm to 2.29 while keeping KL below 0.0037. Huber smoke v3
  completed 4096 transitions in 41.95 s with finite metrics and CUDA optimizer.
- Background R6 development run launched at 2026-09-01 13:19 Asia/Shanghai:
  `artifacts/r6_learned_navigation_development_seed6001_v1`. Contract: 262,144
  training transitions, 8 simulator workers, recurrent rollout length 256,
  50 stratified phase-end diagnostic tasks, CUDA actor/optimizer, external
  `timeout` 28,800 s (8 h). Timeout PID is 586433; Python trainer PID at the
  one-time live check was 586437. First update completed 2048 transitions in
  21.04 s total, collection throughput 153.35 transitions/s, finite Huber loss
  1.877, pre-clip gradient norm 0.8285, KL 0.00302, zero emitted collision
  steps, and HEALTH status `HEALTHY`. Eight worker processes each had one OS
  thread. Per user instruction, no further monitoring occurs until requested.

## Authoritative Decisions

### 2026-09-01 R7 CPPO-PID selection

- The user explicitly rejected a broad bespoke R7 redesign and requested the
  successful SAC modules with SAC replaced by PPO plus an established CMDP
  safety method.
- Primary sources screened: CPO (ICML 2017), FOCOPS (NeurIPS 2020), and
  PID-Lagrangian (ICML 2020). CPO requires a second-order trust-region solve and
  FOCOPS changes the PPO projection target; neither is a minimal PPO swap.
- Selected method: CPPO-PID / PID-Lagrangian PPO. The ICML 2020 paper reports
  reduced constraint overshoot and improved hyperparameter robustness; the
  current OmniSafe reference implementation combines standardized reward and
  cost advantages as `(A_r - lambda * A_c) / (1 + lambda)` and updates lambda by
  filtered P/I/D terms.
- Minimal model contract: reuse `StructuredLidarFeatureExtractor`, feed-forward
  Gaussian actor, separate critic feature extractor with exactly two scalar
  heads (`V_r`, `V_c`), standard PPO clipping, and HOCBF as the final execution
  layer. No GRU, set-attention replacement, safety distillation, or three-cost
  design is authorized in the first run.
- The single CMDP cost is a documented combination of HOCBF intervention and
  executed collision contact. Its positive limit must be calibrated against
  successful R3 behavior rather than set to zero. The immutable final Gates
  remain: overall success >= 0.98, every distance bucket >= 0.95, mean path ratio
  <= 1.10, boundary-contact step rate < 0.01, and zero obstacle-collision steps.
- Skill routing error: the nature-academic-search router prose named
  `multi-source-search.md`, while its manifest correctly mapped the workflow to
  `wf1-multi-source-search.md`; the manifest path was used.

- The user selected R3 as the conditional frozen platform despite the historical
  engineering navigation FAIL. This selection must not be rewritten as a passing
  navigation claim.
- Return-to-charge value is judged by paired stranding--throughput Pareto evidence,
  not prediction error alone.
- Oracle Decision Headroom is a kill Gate: no learned ETG/EIRR method should be
  trained or promoted before perfect information demonstrates useful headroom.

## Endurance Evidence

- V1 completed 100 runs but only 59 ended in `energy_exhausted`; 41 ended at
  `task_step_limit`. Mean reported time was therefore not an admissible estimate
  of time to true depletion.
- V2 preserves physical state and rolls failed tasks into a continuous workload.
  Unit and critical-path regressions passed 145/145 before formal launch.
- V2 attempt 1 directory:
  `artifacts/r3_fixed_baseline_energy_chain_continuous_endurance/`. It ended in
  `FAILED.json` before completing the first ten-run batch.
- Attempt 1 exposed an overly strict sampling-reference check: a physically safe
  UAV position inside the extra goal-clearance halo was rejected during task
  rollover. Candidate-goal clearance was correct and remains unchanged.
- `_sample_task_point` now validates the physical reference for finite shape and
  map interior without imposing candidate-goal obstacle clearance. A step-level
  regression verifies no teleport/recharge and a goal-clear candidate.
- The complete changed critical path passed 146/146 tests before attempt 2.
- Active v2 attempt 2 directory:
  `artifacts/r3_fixed_baseline_energy_chain_continuous_endurance_attempt2/`.
  Launch-time PID `951662` and unified session `73678` were verified live once.

## Next Decision

When the formal v2 process is terminal, audit the completed artifact against all
Gate invariants. If it passes, hand off to the already implemented formal Oracle
Headroom collector. If it fails, preserve the exact failure artifact and diagnose
the scientific or implementation cause without silently weakening the Gate.

## 2026-08-30 Continuation Audit

- Unified execution session `50967` remained open when polled once.
- OS process `940729` was present in active `Rl+` state at approximately 166 s
  elapsed. No terminal marker or first ten-run batch existed yet.
- Passing the active `RUNNING.json` to `audit_gate_b_prerequisites` produced
  `FAIL_GATE_B_PREREQUISITES`. Missing evidence included the 100-run count,
  completed v2 schema, continuous-workload completion flag, zero-censoring
  count, and observed endurance tolerance statistics.
- This verifies fail-closed behavior: a live/partial artifact cannot authorize
  Oracle Decision Headroom. No Oracle or learned-method process was started.

## Oracle Stage-B Formal Preflight

- The formal argument parser was exercised without creating an output directory,
  loading a model, or executing a battery cycle.
- First-Gate methods are exactly `soc`, `distance`, and `oracle`; learned TD
  methods are not in the default headroom run.
- Candidate family: four SOC thresholds (`0.10`, `0.20`, `0.30`, `0.40`), four
  distance reserve fractions (`0`, `0.05`, `0.10`, `0.15`), and Oracle at the
  same four reserve fractions. The design therefore contains 12 points.
- Each point uses 320 distinct seeded battery cycles with one cycle per seed,
  for 3,840 cycle jobs and six persistent worker processes.
- Evaluation-seed list SHA-256:
  `7650369517009717b6095770005d8177545a96eadefa0a46860bff0463e23935`.
- Familywise one-sided Bonferroni--Clopper--Pearson design passes: per-point
  certification power `0.9562346748821128`, joint Oracle-plus-heuristic lower
  bound `0.9124693497642256`, and up to six strandings remain certifiable at the
  declared 5% ceiling.
- Throughput uses a simultaneous paired max-t rate band across the complete
  candidate family. The fixed sample size does not assume enough effect power to
  resolve exactly 5%; `INCONCLUSIVE` is an admissible outcome.
- Non-Oracle methods are coupled to an Oracle shadow through and including the
  first disagreement, and complete cycles are joined by seed, cycle, and reserve.
- Planned formal output directory remains absent:
  `artifacts/r3_oracle_headroom_stage_b_formal/`. It must not be created until
  the v2 endurance artifact passes all prerequisites.

## Gate-B Endurance Provenance Contract

- `GateBPrerequisiteAudit` now records the validation schema, endurance estimand,
  continuous-workload flag, censored-run count, and validation-run count.
- These fields survive additional checkpoint/environment identity failures and
  flow through Stage-B `prerequisite_audit`, `summary.json`, and the Oracle/Pareto
  evidence bundle rather than being recoverable only from an external SHA.
- Ten targeted provenance/identity tests and the complete 61-test Gate/Oracle
  statistics suite pass. This code change does not alter the already running
  attempt-2 depletion process.

## 2026-08-31 Stage-B Runtime Incident

- The first formal run used six spawned processes, each loading a CUDA SAC model
  and retaining the default 24 Torch intra-op/inter-op threads. Runtime inspection
  showed about 73 OS threads, 1.6 GiB RSS, and 270--290% CPU per worker, while the
  RTX 5060 GPU was only about 9% utilized.
- Stage-B is evaluation only (`training_env_transitions = 0`): the expensive path
  is CPU environment simulation plus exact Oracle clone rollouts at each return
  decision, not gradient training.
- At least two workers were replaced after a confirmed SIGSEGV. Blocking
  `multiprocessing.Pool.map` has no atomic per-seed persistence or retry contract,
  so the partial run cannot become admissible formal evidence.
- On user instruction, process group 35431 was terminated. The partial directory
  `artifacts/r3_oracle_headroom_stage_b_formal/` is retained for audit.
- New launch constraint: projected formal wall clock must be below 10 hours; do
  not reduce the preregistered 320 independent cycles per point merely for speed.

## 2026-09-01 Bounded Stage-B Redesign

- Profiled one exact 833-step R3 rollout: 16.88 s initially, with 10.74 s in
  Python HOCBF filtering and only 0.78 s in SAC inference.
- Vectorized spherical HOCBF construction/top-k and direct static LiDAR point
  batches preserve the exact 36.974186813146325 energy prediction while reducing
  the same rollout to 3.75 s (about 4.5x).
- Formal work is now scheduled by seed across the complete candidate family.
  Exact Oracle bundles are shared only for byte-identical float32
  state/velocity/task-goal/charger keys; diverged trajectories remain misses.
- Each completed seed family is atomically written and validated. Broken process
  pools retry only missing seeds, and `--resume` requires exact launch-argument
  compatibility.
- Two inherited clone guards (`max_steps_per_task` and episode step limits) were
  found to preempt the declared 4000-step Oracle loop. All clone-local guards now
  lie strictly beyond that loop; the Oracle loop is the sole admissible budget.
- To meet the user wall-clock constraint without reducing per-point inference,
  the confirmatory family is fixed before outcomes to five points: SOC 0.20,
  distance reserves 0 and 0.05, and Oracle reserves 0 and 0.05. These values come
  from the prior smoke protocol; 320 independent cycles remain required per point.
- Critical verification after the redesign: 218 tests passed.
- Formal optimized-v2 launch at 2026-09-01 00:23 CST: five candidate points,
  320 cycles/point (1600 total), twelve one-thread CPU workers, passed design
  power and P0 audits, and a GNU timeout process-group limit of 34200 s (9.5 h).
  Timeout PID 130319, Python parent PID 130323, unified session 72799. All twelve
  workers were live at about one CPU core and 0.76 GiB RSS each when checked once.

## 2026-09-01 Optimized-v2 Terminal Failure

- The process is terminal; no Stage-B worker remains live.
- Launch was 00:23:42 CST and `FAILED.json` was written at 00:44:20 CST, for an
  elapsed wall time of about 20 minutes 38 seconds, well before the 9.5-hour
  external timeout.
- The direct cause is a `return_after` Oracle clone (by stack position) reaching
  the estimator's explicit 4000-policy-step loop boundary. This is 800 simulated
  seconds at the frozen 0.2 s policy interval; it is not one of the inherited
  task/episode guards already repaired.
- No `seed_results` file exists. The current family-level atomic write occurs
  only after all five candidates finish, and the worker exception is re-raised
  without seed/candidate/trajectory state. Therefore the exact failing seed and
  whether the trajectory is stuck, oscillatory, or merely slow are not yet
  evidenced.
- Scientific rule: do not increase the 4000-step cap or reinterpret failure as
  safe return until start/goal/final distance, best progress, recent progress,
  safety interventions, and candidate provenance are recorded and reproduced.

## 2026-09-01 Instrumented Failure Reproduction

- Added `ModelBasedRolloutError` diagnostics for the task, return-after-task,
  and return-now rollout roles. Budget failures now record start/goal/final
  states, distance/progress summaries, path length, energy, and safety events.
- A Stage-B worker now atomically writes `seed_failures/seed_<seed>.json` with
  seed, candidate index, method/parameter, completed candidates, traceback, live
  evaluation context, and rollout diagnostics before re-raising.
- Added a reproducible non-formal wrapper at
  `scripts/diagnose_return_decision_stage_b.py`; it derives all scientific
  arguments from the SHA-bound optimized-v2 launch rather than reconstructing
  them manually.
- Verification: 220/220 tests passed across the MC regression, return-decision,
  environment, and HOCBF/filter suites.
- Launched `artifacts/r3_oracle_headroom_stage_b_diagnostic_v1/` using seeds
  110001--110012, 12 one-thread workers, the same five candidates, and unchanged
  4000-step Oracle cap. The source Gate-B prerequisites passed again. Unified
  session is 83627; outer timeout is 3600 seconds. This artifact is explicitly
  marked `formal_evidence: false` and cannot support a Pareto claim.

## 2026-09-01 Failure Reproduction Result

- The diagnostic terminated normally through its failure contract rather than
  the one-hour timeout. Eleven seeds wrote structured failure files; seed 110006
  alone completed all five candidates.
- All eleven failures occur while evaluating the first SOC=0.20 point because
  its coupled Oracle shadow is evaluated before later candidates. They therefore
  block collection and are not eleven observations that SOC is inferior.
- Failure roles: 9 return-now and 2 return-after-task. All use 4000 policy steps
  (800 simulated seconds). Median final charger distance is 1161.078 m; median
  last-100-step progress is -0.0117 m; 8/11 have nonpositive recent progress,
  3/11 never improve over the start, and 10/11 achieve their best distance by
  step 600 before a long plateau.
- None of the eleven records an obstacle collision or boundary contact. HOCBF
  modifies 85.1--100% of physics-substep actions (median 99.25%). This is strong
  evidence of a persistent policy/filter interaction but not yet causal evidence
  that the filter alone is responsible.
- In seed 110006 only, Oracle and distance-reserve-0 both complete ten tasks at
  26.359 tasks/hour with zero stranding, versus SOC's eight tasks and 23.452
  tasks/hour. This one-seed +25% tasks/cycle and +12.4% rate observation is a
  diagnostic clue only; it cannot authorize a Pareto or headroom claim.

## 2026-09-01 Paired Stall-Causality Ablation Launch

- Added `scripts/analyze_return_rollout_stalls.py`. For each of the eleven
  SHA-bound failed states it reconstructs the same evaluation seed/obstacle
  realization and tests three counterfactuals: frozen SAC without HOCBF,
  goal-directed heuristic with HOCBF, and goal-directed heuristic without
  HOCBF. The observed frozen-SAC+HOCBF outcome is retained as the fourth paired
  condition rather than recomputed.
- This design isolates whether failure persists under the same learned policy
  when filtering is removed, and whether it persists under a different goal
  controller when filtering is retained. It is a causal diagnostic, not a
  formal performance comparison or authorization to change the frozen platform.
- Script compilation/help and two focused regressions passed. The run is active
  in `artifacts/r3_return_rollout_stall_ablation_v1/`, session 83215, with eleven
  one-thread workers and a 3600-second outer timeout. The Gate-B prerequisite
  audit passed and the artifact is explicitly `formal_evidence: false`.

## 2026-09-01 Paired Stall-Causality Result

- The eleven-state paired diagnostic completed in about 192 seconds. The
  observed frozen-SAC+HOCBF loop reached the charger in 0/11 states; frozen SAC
  without HOCBF reached it in 7/11, the goal heuristic with HOCBF in 8/11, and
  the heuristic without HOCBF in 9/11.
- Removing HOCBF while holding SAC fixed and replacing SAC while holding HOCBF
  fixed both rescue many states. One state is filter-specific, three are
  policy-dominant, five are rescued by every counterfactual, and one defeats all
  tested controllers. The recurrent failure is therefore an executed-loop
  SAC--HOCBF interaction, not a single-module attribution and not evidence that
  the 4000-step horizon is merely slightly too short.
- These states were selected because the original loop failed. The proportions
  are causal diagnostics on those states, not population success rates or
  formal Pareto evidence.

## 2026-09-01 Finite-Deadline Completion Semantics

- The invariant Oracle object is the operational deadline-completion resource
  E_H(x,g): the full nonnegative path cost when the executed loop hits g by H,
  and +infinity otherwise. The finite cost accumulated through H is retained as
  a diagnostic prefix but is not completed-goal ETG.
- Increasing H can only weakly decrease E_H in the extended-real order. A
  mission sum is taken in [0,+infinity], so any deadline-infeasible component
  makes task-then-return infeasible at that horizon.
- ReturnManager commits immediately when task-then-return is infeasible but
  return-now is finite. If return-now itself is deadline-infeasible, it enters
  emergency return while explicitly withholding a safe-return certificate.
  Neither branch claims infinite-horizon unreachability.
- Evidence schema v6 encodes finite, infinite, and not-evaluated separately.
  Exact finite/infinite tags and both branch-feasibility indicators participate
  in the stopped Oracle coupling audit. Post-first-disagreement events have an
  inactive shadow and null exact semantics rather than being mislabeled finite.
- Changed-path verification currently passes 160 energy/ReturnManager/schema
  tests plus the subsequent 139-test schema/environment selection.

## 2026-09-01 Deadline-Semantics Preflight Launch

- The complete critical selection across MC regression, environment,
  ReturnManager/Stage-B, schema v6, and HOCBF fast paths passes 229 tests.
- Launched the unchanged five-candidate diagnostic family for historical failure
  seed 110009 in
  `artifacts/r3_oracle_headroom_stage_b_deadline_semantics_preflight_v1/`.
  It uses the SHA-bound optimized-v2 launch, uv, one worker/Torch thread, the
  unchanged 4000-step Oracle horizon, and a one-hour external wall-clock limit.
- Unified execution session: 57093. This is non-formal preflight evidence only.
  Do not poll it until the user explicitly requests the result.

## 2026-09-01 Deadline-Semantics Preflight Result

- Session 57093 terminated with exit code 0 after about 254.8 seconds. Historical
  failure seed 110009 completed all five candidates and wrote
  `COMPLETED_DIAGNOSTIC.json`; no `ModelBasedRolloutError` recurred.
- At global step 431, the exact return-now branch was finite, while
  task-then-return missed the operational horizon. The effective requirement was
  encoded as null plus `is_infinite=true`; Oracle committed for reason
  `task_then_return_deadline_infeasible` and returned successfully.
- This one seed yielded zero stranding for all candidates. SOC completed 10 tasks
  at 29.431/h; distance reserves 0 and 0.05 completed 12 at 30.684/h and 11 at
  29.480/h; both Oracle reserves completed one task at 20.414/h. This is a
  semantic/runtime preflight, not a headroom estimate or method comparison.
- The seed result passes strict JSON parsing and contains no NaN/Infinity token.
  The formal v3 design remains frozen independently of these outcomes: five
  points, 320 cycles per point, 1600 total jobs, 12 one-thread workers, seed SHA
  `7650369517009717b6095770005d8177545a96eadefa0a46860bff0463e23935`, and
  passed exact stranding-power audit with joint lower bound 0.9673943.

## 2026-09-01 Formal v3 Oracle Headroom Launch

- Launched `artifacts/r3_oracle_headroom_stage_b_formal_deadline_v3/` through
  uv in unified session 14411 with an outer process-group limit of 34200 seconds
  (9.5 hours).
- The execution command is reconstructed from the SHA-bound optimized-v2 launch
  and changes only the fresh output directory; current code writes the new v6
  evidence contract and finite-deadline manifest.
- Frozen evidence design: SOC 0.20; distance reserves 0 and 0.05; Oracle reserves
  0 and 0.05; 320 independent seeded cycles per point; 1600 total cycle jobs;
  12 workers; one Torch thread each; unchanged 4000-step Oracle horizon.
- The preflight result did not change the candidate set, Gate thresholds, sample
  size, seed list, or statistical method. Do not poll until the user asks.

## 2026-09-01 Formal v3 Stop and Termination-Semantics Repair

- Formal v3 was stopped by interrupt (exit 130) to avoid accumulating invalid
  evidence. It had written eight complete seed families and three failures
  (110004, 110012, 110013). All partial files are retained, but the directory is
  marked inadmissible, non-resumable, and non-citable.
- Root cause: Stage-B labeled every environment truncation as a Phase-2
  emergency guard. In the environment, a TASK segment can instead hit its
  4000-step task limit. Moreover, the inherited 20000-step global guard was
  smaller than the physical exhaustion bound:
  `ceil(304.953884 / 0.012) = 25413` policy steps.
- The environment now offers Stage-B continuous task workload semantics. A
  timed-out TASK goal is recorded and replaced by the next keyed task while UAV
  position, battery, simulation clock, path history, and battery-cycle identity
  remain continuous. A committed charger return cannot use this rollover.
- The Stage-B global guard is now derived from the nonnegative telemetry model:
  `G = max(G_source, cycles*ceil(C/e_min)+1)`. For R3, `e_min=0.012` and
  `G=25414`. The guard remains fail-closed but cannot pre-empt physical battery
  exhaustion under the frozen energy contract.
- Unexpected truncation failures now preserve the exact environment
  `end_reason`, remaining energy, guard value, policy steps, and rollover count.
  Four related test modules pass 163 tests. The next allowed execution is a
  non-formal three-seed preflight on 110004, 110012, and 110013; formal v4 is
  contingent on that result.

## 2026-09-01 Task-Rollover Preflight Result and Hybrid Viability Inversion

- The three-seed preflight completed in 1752.863 seconds with exit code zero.
  Seeds 110004, 110012, and 110013 each completed all five candidates. Across
  fifteen cycles there were ten continuous task rollovers, zero emergency
  guards, eight charger arrivals, and seven true energy-exhaustion endpoints.
  All six JSON artifacts parse strictly and no failure file exists.
- This establishes only that the task-rollover/global-guard repair works. It
  does not establish headroom: the seeds were selected from historical failures
  and each method has only three cycles.
- Oracle reserve 0 and 0.05 each stranded on seeds 110012 and 110013. At seed
  110012, the prior check at global step 1191 certified both routes, with direct
  return requirement 27.539 and mission requirement 34.401 at remaining energy
  254.647. At step 1201, direct return became deadline-infeasible while the
  task--then--return route remained feasible with requirement 33.998 and
  remaining energy 254.244. The old manager committed and later exhausted.
- Seed 110013 repeats the mechanism: at step 1391 direct return was feasible
  with requirement 17.913 and mission requirement 42.976; at step 1401 direct
  return became deadline-infeasible while mission remained feasible at 42.459
  against remaining energy 248.618. The old manager committed and exhausted.
- The feasible sets are non-nested because changing TASK to CHARGER changes the
  goal-conditioned SAC--HOCBF closed loop, while completing a task applies a
  hybrid service reset that zeros velocity. Thus direct return can be
  uncertified even when completing the task restores certified return.
- Corrected invariant: \(C_M=\mathbf1\{b>m+E_M\}\) controls the binary stopping
  action. If \(C_M=1\), continue. If \(C_M=0,C_R=1\), commit safely. If both
  are zero, enter an explicitly uncertified emergency return. \(C_R\) is a
  certificate, not a second stopping boundary.
- QuantileEnergyReturnManager, decision audit semantics, formal manifest,
  derivation/proof/protocol documents, EIRR certificate propagation, and the
  Oracle/Pareto schema were migrated accordingly. Schema v7 requires
  task_then_return_stopping_boundary; v6 evidence cannot be mixed with it. The
  corrected related suite passes 165 tests and the theory verifier suite passes
  7 tests.

## 2026-09-01 Viability-Order Preflight v2 Infrastructure Failure

- The attempted corrected-rule preflight at
  `artifacts/r3_oracle_headroom_stage_b_viability_order_preflight_v2/` is
  terminal and non-formal. It persisted only seed 110004; seeds 110012 and
  110013 are absent, so it does not satisfy Phase 6b.8.
- `FAILED_DIAGNOSTIC.json` reports exhausted worker retries, but the wrapper had
  explicitly forced zero retries and produced no structured per-seed exception.
  Kernel evidence records a `python3` segmentation fault at 11:24:11 during the
  run. This classifies the attempt as native worker/infrastructure failure, not
  a ReturnManager or Oracle scientific result.
- The diagnostic wrapper now supports one-worker execution and explicit
  whole-pool retries while retaining atomic completed seed-family files. The
  next admissible action is a fresh one-worker/two-retry run on exactly seeds
  110004, 110012, and 110013 after the currently active R6 evaluation releases
  the simulator resources.
- A combined uv regression over `test_uav_energy_delivery_sac.py`,
  `test_energy_mc_regression.py`, `test_oracle_pareto_records.py`, and
  `test_return_decision_research.py` passes 166 tests. This covers the corrected
  non-nested ReturnManager, probability/deadline tri-state semantics, v7 evidence
  records, Stage-B inference, and the new diagnostic worker override.
- The completion ledger and protocol evidence snapshot were corrected to consume
  the already authoritative attempt-3 endurance artifact: capacity
  `304.9538842289515`, 100/100 true depletion, zero censoring, mean
  `1668.3655 s`, relative error `-7.313%`. This changes the upstream endurance
  status to PASS but does not promote Oracle Headroom, which remains pending
  Phase 6b.8 and a fresh formal run.
- A first single-worker restart was immediately rejected by the launch health
  check: PID 647978 had 47 OS threads despite the manifest's one-thread request.
  The cause was import order—NumPy/Torch were imported before the ProcessPool
  initializer changed environment variables. The run was interrupted before any
  seed-family result and marked `STOPPED_INVALID_WORKER_PARALLELISM`.
- `run_return_decision_stage_b.py` now sets OMP, MKL, OpenBLAS, NumExpr, BLIS,
  and VecLib caps before numerical-library imports and writes
  `stage_b_worker_health_v1`. Targeted recovery/thread tests pass.
- Phase 6b.8 v4 is active in unified session 83662 under timeout PID 649352 and a
  10,800-second hard limit. The worker health file reports PID 649414 with
  `os_thread_count=1`, Torch intra-op/inter-op both one, and every native thread
  environment equal to one. It uses exactly the three mechanism seeds, five
  candidate points, one worker, two atomic retries, and unchanged v7 semantics.
  Per the user experiment rule, do not poll or continue the Gate until requested.
- At 2026-09-01 18:00 Asia/Shanghai the user explicitly requested that active
  experiments be interrupted before closing the terminal. Phase 6b.8 v4 had
  completed zero atomic seed-family files after about ten minutes. Its dedicated
  process group was stopped and
  `artifacts/r3_oracle_headroom_stage_b_viability_order_preflight_v4_threadcapped_single_worker/STOPPED_BY_USER.json`
  records the restart scope. No scientific result was obtained from v4.
- The concurrent R6 run had already completed all 262,144 training transitions
  and saved `checkpoint_transition_000262144.pt`; training remained descriptive
  failure evidence with 0/64 recent successes. Its 50-task evaluation had run
  for about 3 h 20 min without an atomic result when stopped by user. The trainer
  wrote `INTERRUPTED.json` and `checkpoint_interrupted_000262144.pt`; any restart
  must load this checkpoint for bounded evaluation only rather than retrain.

## 2026-09-01 R7 Collision-Safety Literature Selection

- The collision module is now explicitly treated as mature infrastructure, not
  the paper's core innovation. A source-backed survey is recorded in
  `docs/R7_COLLISION_SAFETY_MODULE_SURVEY.md`.
- Safe-RL CMDP algorithms and runtime filters solve different problems:
  CPPO-PID controls expected episodic cost, whereas zero collision at each step
  requires an always-on statewise safety filter. The selected stack is therefore
  CPPO-PID plus robust sampled-data HOCBF-QP, not CPPO-PID alone.
- The project is an unusually direct match for HOCBF: double-integrator
  translational dynamics, three-dimensional acceleration actions, explicit
  actuator limits, 20 Hz physics filtering, and local LiDAR geometry.
- Current R7 incorrectly hard-codes ordinary continuous-time HOCBF. The older
  controlled 1,000-rollout safety artifact showed ordinary HOCBF at 18.4%
  collision-rollout rate versus zero for sampled-data HOCBF, while the latter
  retained 100% success, mean path ratio 1.0388, and about 1.005 ms mean filter
  time. This artifact is prototype selection evidence, not a substitute for the
  immutable R7 formal evaluation.
- For on-policy filtered training, buffer the nominal PPO action for the
  likelihood ratio, apply the safe projected action to the simulator, and score
  the physical reward on the executed transition. R7 already follows this
  action-buffer convention.
- Replace binary HOCBF activation cost by squared normalized correction
  magnitude. This is supported by the 2024 quadrotor safety-filter training
  study and carries more information about unsafe action severity. Collision,
  unsafe fallback, deadline miss, and boundary contact remain fail-closed gates.
- Before any real-world guarantee, add calibrated measurement/discretization
  and dynamics-disturbance bounds. Simulator zero-collision evidence does not
  establish real-world forward invariance.

## 2026-09-01 R7 One-Hour Pilot and Conditional 500k Launch

- Fixed evaluation selection to round-robin across requested distance buckets.
  A 25-task five-bucket evaluation now contains exactly five tasks per bucket;
  the earlier prefix selection would have tested only the shortest ordered
  bucket and could inflate development success.
- Added auditable continuation from a checkpoint: model, actor and critic
  optimizers, PID-Lagrangian state, global/update counters, simulator seed
  cursor, NumPy generator, and CPU/CUDA Torch random states are restored. A
  two-stage CUDA smoke completed from 256 to 512 total transitions.
- Locked “effect is very good” before seeing the pilot: training successes > 0,
  overall deterministic success >= 0.80, every distance bucket >= 0.60, mean
  path ratio <= 1.50, zero obstacle collision and boundary steps, and sampled-
  data HOCBF intervention rate <= 0.20.
- Launched `artifacts/r7_sampled_hocbf_pilot_then_500k_seed7002_v1` through uv
  as a detached pipeline. Pilot budget is 196,000 transitions with 8 x 250
  rollouts, 70-minute total stage timeout, and 25 stratified evaluation tasks.
  If the locked Gate passes, it resumes from transition 196,000 to exactly
  500,000 under a separate nine-hour timeout; otherwise it stops.
- First update at 2,000 transitions was healthy: actor and critic losses finite,
  CUDA actor active, collection throughput 175.835 transitions/s, collision
  step rate 0, mean squared action correction 0.008313, and HOCBF intervention
  rate 0.0205.
- At 118,000 transitions the user authorized replacing the insufficient
  70-minute pilot timeout. SIGINT produced the atomic checkpoint
  `checkpoint_interrupted_000118000.pt`; the old wrapper then terminated as
  expected. No transition replay from the 100k periodic checkpoint was needed.
- Added `--pilot-resume-checkpoint` to the conditional pipeline and made the
  health Gate trigger on the first update of each process, including resumed
  runs. Ten focused tests pass after this change.
- Launched
  `artifacts/r7_sampled_hocbf_pilot_then_500k_seed7002_resume118k_v2` with a
  three-hour pilot timeout. Its first resumed update completed at 120,000 total
  transitions and was healthy: zero collision steps, 11.75% intervention,
  finite losses, CUDA actor, and one successful 248-step episode. The 500k
  promotion policy is unchanged.
- The user then explicitly removed all wall-clock limits and required resumable
  intermediate evidence instead. The v2 run was interrupted at exactly 128,000
  transitions and saved `checkpoint_interrupted_000128000.pt`.
- Added exact source-index task selection and
  `scripts/evaluate_r7_formal_500_resumable.py`. Formal evaluation uses the
  immutable 500-task source (100 tasks in each of five distance buckets), runs
  in batches of 20, and atomically writes `PROGRESS.json` after each batch. A
  restart validates checkpoint/task hashes and evaluates only missing source
  indices. There is no wall-clock timeout.
- Extended the conditional runner with `--no-timeouts` and
  `--formal-evaluation-500`. The active v3 pipeline resumed at 128,000, will
  run the 25-task promotion Gate, conditionally continue to 500,000, and then
  automatically launch the resumable formal 500-task evaluation. Its first
  resumed update at 130,000 was healthy with zero collision steps and 8.8%
  HOCBF intervention.

## 2026-09-02 R7 accelerated restart handoff

- Old v3 reached 196,000 transitions and atomically wrote
  `artifacts/r7_sampled_hocbf_pilot_500k_formal500_resume128k_unbounded_v3/pilot_196k/checkpoints/checkpoint_transition_000196000.pt`.
  Full `torch.load` validation found 31 model entries, both optimizer states,
  PID state, NumPy/Torch/CUDA RNG state, update 98, and seed cursor 720133. Its
  SHA-256 is
  `bca3fe9de29462b197de5492d1dc09969cabf86bb59040881c4689d5edcc5c97`.
- Old process group 89423 was terminated only after that checkpoint became
  visible; no old v3 process remained.
- The pipeline now has a mutually exclusive `--promoted-resume-checkpoint`
  recovery path. Its decision artifact is labeled recovery, not a newly
  observed pilot Gate. The trainer restores SIGINT for detached jobs and hashes
  `filter.py`, `hocbf.py`, `_qp_native.c`, and `_qp_native.so` for provenance.
- Eleven focused R7 tests, Python compilation, and `git diff --check` passed.
- Active v5: PGID 120325, trainer PID 120345, output
  `artifacts/r7_sampled_hocbf_promoted500k_formal500_accelerated_resume196k_v5`.
  All eight simulation workers mapped `_qp_native.so`. The health Gate passed
  at 198k; by the one-time validation it had reached 206k with zero collision
  step rate and 9/9 recent successes. Collection improved from 70--85 s per 2k
  transitions to 2.1--3.1 s (643--960 transitions/s). It has no timeout and
  automatically starts the resumable formal 500-task evaluation after 500k.
  Do not monitor until the user asks.

## 2026-09-02 accelerated R3 SAC reproduction request

- User requested a fresh-from-zero reproduction of the prior R3 SAC result
  using the current accelerated HOCBF implementation, followed by a formal
  500-task evaluation.
- Treat this as an implementation-equivalence reproduction: no checkpoint
  resume and no algorithm/reward/hyperparameter tuning are authorized.
- Planned durable report: `docs/R3_ACCELERATED_REPRODUCTION.md`.
- Frozen original artifact:
  `artifacts/jseb_navigation_repair_r3r4_uv_seed0_20260828_193904/R3`.
  Original git SHA `7e30d8f5b6928bbb505b9eaa3873c1791590400b`, checkpoint
  SHA-256 `02c1ecf55c24ebb2652507ba0c751ff81f7bf8d2cfbdefc39adb88d5099f910a`.
- Immutable source-failure evaluation SHA-256 is
  `ca4ae3a11b129948a4c4b22a27973da74e8ad006c69a08b3c6343b2e14172a6e`;
  immutable selection-task SHA-256 is
  `ead4b79839f0af425896ef27d455d4d6ea78afd60cda7e28e6f3ffbef1cd07c9`.
- Original formal result: 500 tasks, 96% overall success, bucket successes
  99/98/94/97/92%, mean path ratio 1.188026, 2,242 obstacle-collision steps,
  and four boundary-contact episodes. It was `STOPPED_NAVIGATION_NOT_READY`
  because the safety gate failed; this is the reproduction target, not a claim
  that original R3 passed all gates.
- Current uncommitted code had removed the original SubprocVecEnv/forkserver
  training path and forced DummyVecEnv. That is not an acceptable reproduction
  factor and must be restored while retaining the accelerated filter internals.
- Restored the original optional SubprocVecEnv/forkserver training path and its
  one-thread-per-worker audit fields. Added an explicit accelerated-reproduction
  mode that permits the necessarily dirty acceleration sources only after all
  baseline factors and two immutable input hashes pass, and records SHA-256 for
  twelve code/native artifacts.
- Contract preflight: 13/13 checks true. Focused suite: 110 passed. CUDA smoke:
  exact 2,000 transitions plus five-task evaluation completed; checkpoint and
  terminal sentinel were written. The zero smoke success rate is expected at
  2k and is not scientific evidence.
- Formal reproduction launched detached at PGID 204721 / trainer PID 204724:
  `artifacts/r3_sac_accelerated_reproduction_seed0_500k_formal500_20260902_v1`.
  All eight forkserver workers loaded `_qp_native.so`. The first 8k log has
  finite metrics, 2,992 gradient updates, 1.034665 mean step reward, zero
  collision/boundary contacts, and 0.167875 HOCBF intervention rate. Stop
  monitoring until requested.

## 2026-09-02 Energy continuation after accelerated R3 reproduction

- The accelerated reproduction completed all 500,000 transitions and the fixed
  500-task evaluation.  Its formal navigation metrics match the actual frozen
  R3 formal artifact exactly (96% success, mean path ratio
  1.188026281808251, 2,242 obstacle-collision steps, and four boundary-contact
  episodes).  The separate 84% file is a historical source/selection
  evaluation and must not be mislabeled as the frozen R3 formal result.
- Formal evaluation wall clock fell from 4,336.140 s to 1,941.420 s while the
  scientific records remained unchanged.  This validates navigation-level
  accelerated semantics but does not by itself attest finite-energy switching
  and cumulative telemetry.
- The next Gate is a paired one-cycle reference-versus-accelerated energy
  trajectory on frozen R3.  It must compare executed states/actions, safety
  decisions, per-step/cumulative realized energy, task schedule, and terminal
  semantics before any Phase 6b.8 launch.
- Terminal Phase 6b.8 v4 remains stopped by user with zero completed seed
  families.  A passing energy equivalence check authorizes a fresh v5 run on
  seeds 110004, 110012, and 110013, with one worker, two retries, and unchanged
  v7 candidate/decision semantics.
- Focused acceleration, telemetry-energy, R3 contract, and Stage-B worker
  regressions pass 6/6.  The paired equivalence run is active in unified
  session `73738` at
  `artifacts/r3_energy_runtime_equivalence_seed110004_20260902_v1`.
  Conditional unified session `2985` will launch
  `artifacts/r3_oracle_headroom_stage_b_viability_order_preflight_v5_accelerated`
  only after a passing equivalence sentinel.  It enforces one worker, two
  retries, seeds 110004/110012/110013, and a 34,200-second (9.5-hour) limit.
  Stop monitoring until the user requests an audit.
- E2 v1 ended after 58.3 minutes with both trajectories reaching the charger,
  two tasks completed, and zero collision/boundary events, but it failed exact
  trace equality (12,463 versus 12,952 steps).  The first numerical divergence
  occurred at step 50 after the diagnostic itself rescaled the objective and
  changed both constraint and QP paths.  This is invalid isolation evidence,
  not a scientific acceleration failure.  The conditional shell also failed
  parsing and no Phase 6b.8 v5 directory was created.
- User authorized corrected experiments.  E2 v2 must (A) compare full
  finite-energy trajectories with the array HOCBF path fixed and only the
  Python/native QP backend varied, and (B) compare object/array constraint
  construction on fixed snapshots with the same native solver.  Only a single
  terminal PASS may authorize fresh Phase 6b.8 execution.
- Corrected E2 v2 passed Python compilation, six focused regressions, and a
  12-snapshot frozen-contract preflight.  The snapshot preflight held the native
  QP fixed and observed exactly zero action delta and zero diagnostic mismatch.
  The long run will use a fresh v2 directory and directly chain to v5 on exit
  code zero, avoiding the failed nested shell watcher.
- The corrected fail-closed chain is active in unified session `65130`.  At
  2026-09-02 22:22 Asia/Shanghai, E2 v2 had a live Python process, a RUNNING
  sentinel, the frozen model/environment loaded successfully, and no error
  output.  Phase 6b.8 v5 remains conditional on E2 exit code zero.  Per user
  instruction, do not monitor further until asked.

## 2026-09-03 learned safe and energy-sustainable navigation theory pass

- User asked to continue research toward a neural policy that itself learns
  safe and energy-sustainable navigation.  The active Phase 6b.8 process was
  deliberately not inspected, altered, or restarted.
- Exact-overlap search changed the novelty boundary.  RC-PPO (NeurIPS 2024)
  already solves deterministic minimum-cost reach-avoid.  Stochastic
  Minimum-Cost Reach-Avoid RL / RAPCPO (author record says accepted ICML 2026)
  already imposes a probabilistic reach-avoid condition while minimizing
  expected cumulative cost.  Therefore “safe arrival plus lower energy” is a
  covered claim.
- Formal-methods search found that consumption MDPs (CAV 2020) already model a
  finite battery, non-negative transition consumption, atomic reload states,
  and almost-sure repeated Büchi objectives.  Battery state augmentation and
  recharge-cycle semantics cannot be claimed as new.
- The remaining candidate object is an extended-real, finite-horizon
  resource-to-recharge random variable: actual energy if the charger is reached
  before collision/deadline, and `+infinity` otherwise.  Its finite-threshold
  CDF is exactly the joint probability of safe, timely recharge within the
  remaining battery.  This avoids survivor bias and arbitrary scalarization.
- Derived an exact finite-horizon distributional Bellman recursion for the full
  CDF, explicitly avoiding an invalid additive recursion for static VaR.  Also
  derived the joint battery certificate, calibrated approximation margin,
  conditional budget-monotonicity result, and a no-independence lifecycle union
  bound across regenerative recharge cycles.
- A fixed nonzero per-cycle failure allowance cannot justify infinite-lifecycle
  sustainability.  A summable schedule such as
  `delta_k = 6*delta_total/(pi^2*k^2)` gives a valid total-risk bound, but may be
  infeasible under irreducible stochasticity; this is a testable limitation.
- Minimal architecture consequence: retain the existing structured navigation
  encoder and SAC task critics; add battery/deadline/charger/cycle-risk inputs
  and one conceptual distributional feasibility critic with an explicit
  failure atom.  A learned latent TASK/RETURN head is optional.  There is no
  theorem-level reason for three unrelated safety critics.
- Durable outputs:
  `docs/LEARNED_SUSTAINABLE_NAVIGATION_DERIVATION.md` and
  `literature-search-20260903-learned-safe-energy-sustainable-navigation/`.
- Current verdict: mathematically coherent after reframing, but oral-level
  novelty is not yet claimable.  Full equation-level comparison with RAPCPO,
  RC-PPO, QCRL, and consumption-MDP theory plus calibrated empirical evidence
  is the next Gate.
- RAPCPO equation-level inspection completed: its exact object is a lower bound
  on stochastic reach-avoid probability; its practical compensation-normalized
  quantity is explicitly non-certified, and its optimizer does not enforce
  feasibility at every update.  Nevertheless, any fixed-budget query of the
  proposed resource CDF is exactly a reach-avoid query after augmenting the
  state with remaining resource and declaring resource exhaustion unsafe.
  Therefore the CDF is retained for implementation but demoted as standalone
  novelty.
- Added an approximate Bellman-subsolution theorem: if the learned lower score
  has one-sided residual at most `epsilon_h`, true joint success probability is
  lower-bounded by the score minus `sum_h epsilon_h`.  This yields an explicit
  horizon-dependent training/certification margin, but a distribution-shift-
  aware finite-sample bound is still needed for a strong paper.
- Further search found regret-free model-free LTL learning for unknown finite
  MDPs (ICML 2025) and adaptive-conformal learned safety filters (L4DC 2026).
  These close two easy rescue routes.  The remaining hard target is quantitative
  consumption-Büchi learning under continuous unknown dynamics with raw-policy
  evidence; absent a stronger approximation theorem, it is an integration paper.

## 2026-09-03 Phase 6b.8 audit and chained resource-CDF exploration

- At the requested audit, Phase 6b.8 v5 was healthy and incomplete.  Seed
  110004 had atomically completed at 23:40:47; the single worker remained at
  approximately 100% of one CPU core on the pending seeds.  No failure or
  terminal sentinel existed.
- On seed 110004, SOC 0.20 and distance reserves 0/0.05 each completed two tasks,
  reached the charger, and had zero stranding/collision.  Their task rates were
  2.780, 2.645, and 2.850 per simulated hour.  Oracle reserves 0/0.05 instead
  returned from the initial charger state in one step with zero tasks and about
  0.99997 unused battery fraction because the task-then-return 4,000-step
  deadline was infeasible.  With only one selected mechanism seed this is
  descriptive and cannot decide the three-seed Gate.
- The existing decision logs stop Oracle-shadow labeling after the first
  disagreement.  Seed 110004 therefore contains only the duplicated initial
  labeled state for the non-Oracle trajectories and cannot support a neural
  learnability claim.  Blind offline distillation from this artifact was
  rejected.
- Added `scripts/explore_resource_cdf_learnability.py` and
  `tests/test_resource_cdf_learnability.py`.  The experiment constructs
  full-LiDAR/task/charger/deadline features, labels finite task-then-recharge
  resource bins plus an explicit collision/deadline failure atom, holds out
  whole task seeds in three folds, and compares CDF calibration/discrimination
  against constant and geometric baselines.  Each task is saved atomically and
  `--resume` skips complete tasks.
- Pure and Oracle regression tests pass 28/28.  A five-task/20-step fail-closed
  smoke terminated correctly as uninformative, and a separate near-charger R3
  rollout exercised the finite-success branch and wrote three finite horizon
  examples.
- Unified session 13597 is waiting on Phase 6b.8 PID 847760.  It will launch the
  30-task GPU-backed pilot only after the upstream completion sentinel; an
  upstream failure or missing process writes `BLOCKED_UPSTREAM.json` instead.
  The pilot has no wall-clock timeout and is not formal evidence.

## 2026-09-03 Phase 6b.8 terminal result and CDF-pilot restart

- Phase 6b.8 v5 completed all three selected seed families at 01:35:34 after
  about 2 h 20 min of Stage-B execution.  It wrote fifteen complete candidate
  cycles, no failure sentinel, and terminal status
  `COMPLETED_WITHOUT_REPRODUCING_FAILURE`.
- Across the three descriptive seeds, Oracle reserve 0/0.05 each returned on
  3/3 cycles with zero stranding and thirteen tasks.  SOC and both distance
  variants each stranded on seed 110012, returned on 2/3 cycles, and completed
  16, 17, and 17 tasks respectively.  No candidate recorded an obstacle
  collision.  The three seeds were selected for known mechanisms, so no
  superiority or significance claim is allowed.
- The original chained CDF pilot reached its upstream completion but exited
  before collection because bare `cuda` was passed to `torch.cuda.set_device`,
  which requires an explicit index.  This is a launcher bug, not an experimental
  outcome.  Device selection now maps bare `cuda` to index zero; focused tests
  pass 29/29.
- The same preregistered 30-task pilot was launched afresh in unified session
  74337 at 07:07.  `RUNNING.json` hash-binds the frozen R3 checkpoint and the
  completed Phase 6b.8 sentinel.  The process is compute-active; rollout
  collection is CPU-bound and the distribution-head fit will use the GPU.

## 2026-09-03 Resource-CDF snapshot repair and resume

- The first pilot attempt stopped after twelve atomic tasks because a stored
  float32 trajectory velocity had horizontal norm `20.0000024591` against a
  physical limit of `20.0`.  The `2.46e-6` excess arose when an executed state
  was revalidated for a diagnostic snapshot; it was not a training or dynamics
  divergence.
- Snapshot construction now projects only float32-scale excess back to the
  physical horizontal/vertical limits.  The tolerance is derived from machine
  epsilon, and a material excess such as `20.01` remains fail-closed.  The
  original task-12/index-400 case now produces a finite 2066-dimensional
  feature vector.
- Focused resource-CDF and Oracle regressions pass 31/31.  The prior failure
  sentinel was moved to `failure_history`, and the pilot resumed from its twelve
  complete task files in unified session 61304.  The first health audit found
  fourteen tasks complete, including the formerly failing task 12, with the
  process compute-active and no new failure sentinel.

## 2026-09-03 Boundary-balanced resource-CDF scaling launch

- The 30-task pilot terminated scientifically (not operationally) with AUROC
  0.434 versus 0.725 for the geometry baseline and only 12 negative initial
  budget queries.  It therefore could not distinguish undertraining from poor
  boundary coverage and an unsuitable raw representation.
- Added a preregistered 300-task scaling experiment spanning eight battery
  fractions, four horizons, all five distance buckets, three whole-task folds,
  raw/frozen-R3/compact representations, and 25/100/400 epoch checkpoints.  The
  fixed primary is the frozen dual-goal R3 encoder at 400 epochs.
- Focused regression and smoke validation pass 36/36.  The deliberately tiny
  smoke run completed fail-closed and is engineering evidence only.
- The full run is active in unified session 39036 and writes to
  `artifacts/r3_resource_cdf_boundary_balanced_scaling_300tasks_seed240001_20260903_v1`.
  It has no wall-clock timeout and is resumable from atomic task files and
  completed representation/fold outputs.  The one-time health check observed
  three completed task files, a live compute-active process, and no failure
  sentinel.  Do not poll again until the user requests an audit.

## 2026-09-03 Resource-aware encoder Gate implementation

- The completed scaling study rules out simple epoch undertraining under its
  fixed design: `frozen_encoder@400` degraded to Brier 0.169 and ECE 0.160,
  while `frozen_encoder@25` had Brier 0.135.  The geometry baseline remained
  stronger at Brier 0.101.
- Locked `docs/RESOURCE_AWARE_ENCODER_GATE_PROTOCOL.md`: each outer test fold
  has 100 untouched tasks; the remaining tasks are bucket-stratified into
  selection-training, early-stopping validation, and separate temperature-
  calibration partitions.  No R3 actor parameter is updated.
- Added `scripts/run_resource_aware_encoder_gate.py`.  It trains an R3-
  initialized encoder copy, aligns finite CDF atom edges exactly with every
  queried budget, combines normalized categorical NLL with task-weighted query
  Brier loss, selects epochs on validation tasks, refits, and calibrates a
  temperature on separate task seeds.
- The frozen-CE and trainable-CE variants isolate encoder adaptation and the
  calibration objective.  Promotion also requires improvement over geometry on
  at least two outer folds, preventing a pooled-query-only pass.
- The focused learnability/scaling/new-Gate suite passes 21/21 under uv.
- The 75-task two-epoch smoke subsequently completed all three outer folds and
  wrote checkpoints, fold records, predictions, and a terminal engineering-only
  result.  Its scores are not scientific evidence.
- The full 300-task run is active in unified session 47585 at
  `artifacts/r3_resource_aware_encoder_gate_seed270001_20260903_v1`, with no
  wall-clock timeout.  Its first atomic unit, `frozen_ce` fold 0, selected epoch
  30 and stopped at epoch 180 after the registered validation patience.  The
  process remained compute-active with no failure sentinel.  Resume skips each
  completed variant-fold.  Do not poll again until requested by the user.

## 2026-09-03 Identifiable defective resource-CDF repair

- The resource-aware categorical Gate completed 9/9 units but did not promote:
  primary Brier 0.124 versus geometry 0.101, with every outer fold worse.  All
  primary folds selected the minimum epoch 25, so added epochs are not the
  missing mechanism.
- Dataset audit found 3,777 finite and 439 failure rows.  Finite energy is
  strictly positive (0.0190--0.3686 capacity).  At initial states, 268/300 tasks
  have one repeated finite energy value across feasible horizons and 32/300
  never become finite.  This supports a failure hurdle plus continuous energy
  location, not 72 freely estimated categorical atoms.
- `docs/DEFECTIVE_RESOURCE_CDF_DERIVATION.md` reframes the state-dependent scale
  as non-identifiable from one outcome per exact state.  The executable model
  uses a learned global smoothing scale and separate task-held-out calibration.
- Added `scripts/run_defective_resource_cdf_gate.py` with compact, encoder,
  fusion-no-CDF, and full-fusion variants.  The primary objective combines
  failure BCE, finite log-energy Huber, and twice-weighted budget Brier loss.
- Formula, loss-gradient, all-failure minibatch, calibration, selection, split,
  and prior resource-CDF regressions pass 29/29 under uv.
- The 75-task, two-epoch real-data smoke subsequently completed all three outer
  folds and exercised the R3 encoder, hurdle loss, refit, three-parameter
  calibration, checkpoints, and terminal aggregation.  Its score is engineering
  evidence only.
- The full four-variant, three-fold Gate is active in unified session 19196 at
  `artifacts/r3_defective_resource_cdf_gate_seed290001_20260903_v1`.  The first
  atomic unit, `compact_hurdle` fold 0, selected epoch 10 and stopped at epoch
  160.  The process remained compute-active with no failure sentinel.  Resume
  skips completed variant-folds.  Do not poll again until requested.
- 2026-09-03 defective continuous resource-CDF formal Gate completed 12/12
  units at `artifacts/r3_defective_resource_cdf_gate_seed290001_20260903_v1`.
  The primary fusion hurdle failed promotion (Brier 0.120253, AUROC 0.898915,
  ECE 0.029666) against geometry (0.101266, 0.923326, 0.051447); all three test
  folds were worse.  Compact hurdle was the best learned variant (Brier
  0.110691), while fusion without the CDF was better than fusion with it.
  Failure-atom AUROC was only 0.656 for the primary.  Conclusion: stop tuning
  the supervised head on this observational dataset and create grouped
  same-scene action interventions before changing the actor.
- Counterfactual follow-on contract: scene seed is the independent unit;
  repeated actions/horizons/budgets remain grouped.  First Gate within-scene
  action/outcome/energy diversity and held-out conditional predictability, then
  and only then run joint safe-energy policy learning.
- Implemented `GROUPED_COUNTERFACTUAL_SAFE_ENERGY_DATA_GATE_V1`: frozen R3
  actions are perturbed by keyed temporally correlated residuals at five fixed
  settings; task and return legs reuse an identical worker-resident obstacle
  layout.  Per-step records include frozen/behavior/executed actions, physical
  energy, compact goal state, projection slacks, contacts, and leg identity.
  The formal data Gate compares the same MLP trained on geometry+intervention
  versus frozen R3 embedding+intervention under scene-disjoint three-fold tests.
- Focused counterfactual/environment suite passes 90/90.  Smoke v2 completed
  25 rollouts and 7,109 transitions, reached the task goal in five rollouts,
  exercised all five return legs, reproduced matched initial observations
  exactly, and resumed without recollection.  It is engineering evidence only.
- Launched the resumable 150-scene x 5-intervention formal data Gate at
  `artifacts/r3_grouped_counterfactual_safe_energy_gate_seed310001_20260903_v1`
  with eight environment workers and batched CUDA R3 inference.  One-time
  health check: 15/750 rollouts complete, eight active, no failure sentinel.

## 2026-09-03 pivot to action-conditioned rechargeability

- The grouped counterfactual run is terminal at 750/750 rollouts.  Its data
  diversity Gate passed, but the learned trajectory outcome model did not beat
  the geometry baseline.  Together with the earlier categorical, resource-aware,
  and defective-CDF failures, this rejects further passive-head tuning.
- The operational target is now a pre-action, budget-conditioned
  rechargeability critic.  Its event jointly includes charger reach, collision
  avoidance, deadline compliance, and cumulative executed energy; failure is
  not removed by conditioning on successful returns.
- Existing rollout files cannot reconstruct both task-relative and
  charger-relative state at every task-leg step because absolute positions and
  both goals were not stored.  A fresh, smaller enhanced collection is required;
  relabeling the existing arrays would silently answer a weaker question.
- Leakage boundary: realized energy, current-step HOCBF outcome/slack,
  terminal outcome, and suffix totals are targets/audits only.  Critic inputs
  stop immediately before the proposed action.
- The first experiment is a learnability Gate, not actor training.  It compares
  full action-conditioned Bellman learning against geometry-only, no-action,
  and no-Bellman controls under scene-disjoint folds.  A failed Gate prevents a
  long policy experiment.
- Deterministic task regeneration plus inversion of the stored active-goal
  direction/log-distance recovered the missing physical position.  The smoke
  cache reconstructed all 750 rollouts with maximum initial-position error
  0.0009766 m, avoiding an unnecessary repeat of roughly 1.1 million simulator
  transitions.  This does not recover per-step LiDAR and therefore does not
  remove the collision-tail limitation.
- Added the monotone-knot critic, exact terminal Bellman target, dependency-free
  tied-rank AUROC, grouped nested split, four preregistered variants, checkpoint
  resume, and atomic result semantics.  Focused tests pass 7/7; the two-epoch
  3,000-transition smoke completed all 12 units on CUDA in 2.5 seconds after
  cache construction.  Its scores are engineering-only and cannot promote.
- The formal Gate was launched in unified session 12077 at
  `artifacts/rechargeability_critic_gate_seed350001_20260903_v2`.  Its 750
  source rollouts were converted into a 24,000-transition immutable cache;
  `RUNNING.json` is present, PID 21638 is compute-active and visible to CUDA,
  and there is no failure sentinel.  Do not poll it again until requested.

## 2026-09-04 forked-action follow-on decision

- The formal action-conditioned Gate actually completed 12/12 units in 10.92 s
  after cache construction.  Full Brier 0.05275 lost to geometry 0.04232; the
  no-action ablation was better than full at 0.04988.  Bellman consistency did
  help residual (0.001804 versus 0.002355), and matched action sensitivity was
  nonzero, but it was not predictive enough to change the actor.
- Diagnosis: whole-trajectory intervention is not a fixed-state action fork.
  Its action is confounded with later states and continuation behavior, while a
  single initial action has little leverage on a thousand-step outcome.  Five
  collision rollouts and missing per-step LiDAR make collision-tail inference
  additionally unsupported.
- Locked repair: reuse nominal R3 trajectories only to choose anchor states;
  regenerate the deterministic obstacle layout; reconstruct the exact anchor;
  store its full 2055-dimensional observation once; then run ten paired raw
  eight-step macro-actions followed by the identical frozen-R3/HOCBF
  continuation.  Scene remains the independent unit.
- The short raw prefix is essential: collecting only post-HOCBF actions would
  teach the critic the rule-based projection rather than the neural actor's
  unsafe action boundary.  The filtered continuation keeps the contrast local
  and reduces variance/cost.
- Implemented a worker-resident fork reset, per-step temporary HOCBF bypass,
  analytic low-clearance anchor selection, ten registered actions, atomic
  per-branch resume, and the paired-data Gate.  The corrected focused suite
  passes 16/16.
- Smoke output `artifacts/lidar_forked_action_data_gate_smoke_20260904_v1`
  completed 50 branches in 2.87 s.  All anchor observation and obstacle hashes
  match; full 2055-dimensional observations are stored; source-prefix error is
  5.96e-8.  All branches are deliberately four-step censored, so its 100%
  unsafe label is engineering-only and must not be interpreted scientifically.
- Launched the preregistered 15-scene forked-action pilot in unified session
  93955 at `artifacts/lidar_forked_action_data_gate_15scenes_20260904_v1`.
  One-time health check: all 60 full-LiDAR anchors persisted; 24/600 branch
  records completed, eight were active, the CUDA process was registered, and
  no failure sentinel existed.  Do not poll again until the user asks.
- The pilot subsequently stopped with 73 branch files (progress sentinel 72)
  when worker 0 strictly revalidated a float32 horizontal speed about 1.6e-6
  above 20 m/s.  This is the same snapshot-rounding mechanism previously
  diagnosed in the resource-CDF run.  The repair projects only excess within
  64 float32 epsilons times the physical scale; 20.01 m/s remains an error.
- The repaired suite passes 17/17.  Resume migrated eight not-yet-started
  numerical-limit anchors, preserved all existing branch files, and restarted
  in unified session 22307.  One-time health check observed 102/600 complete,
  eight active, 490 pending, and no failure sentinel.
- The resumed forked-action pilot completed 600/600 branches in 925.5 s and
  wrote `PROMOTE_TO_LIDAR_ACTION_CRITIC_GATE`.  All registered checks passed.
  Unsafe prevalence was 7.83% (collision 3.67%, boundary 1.50%); 46.67% of
  anchors showed paired safety or >=5% energy variation, while only 8.33%
  showed both safe and unsafe branches.
- The action mechanism is directionally sensible: `toward_obstacle` had the
  lowest safe-recharge rate (86.67%) and highest collision rate (11.67%);
  `away_from_obstacle` had zero collisions and 91.67% safe recharge.  The
  >4000 m bucket concentrated most failures (65.83% safe versus >=97.5% in
  every shorter bucket), so later splitting must keep complete scenes and
  report bucket sensitivity rather than pooling it away.

## 2026-09-04 large-scale LiDAR residual critic Gate

- Locked a 75-scene, 300-anchor, 3,000-branch follow-on.  It compares a strong
  geometry/action critic, LiDAR without current action, direct LiDAR+action,
  and a zero-initialized LiDAR/action residual over frozen geometry logits.
  The residual construction exactly recovers geometry at epoch zero and
  preserves monotonicity in battery budget by construction.
- The critic consumes the exact anchor observation through the frozen R3
  structured-LiDAR encoder.  Inputs exclude realized energy, future executed
  actions, collision labels, terminal reasons, and candidate names.  Scene is
  the independent split unit; all actions, anchors, and budgets remain grouped.
- Initial critic smoke failed before optimization because sparse bucket-coded
  scene IDs created an empty validation set under direct integer modulo.  The
  split now uses stable unique-scene ordinal rank; a regression test covers the
  exact sparse pattern.  The repaired suite passes 14/14 and all four variants
  complete on CUDA.
- The three-epoch/15-scene smoke produced finite metrics and zero architectural
  budget-monotonicity violations.  Its residual did not beat geometry, which is
  neither a pass nor a scientific failure because epoch count and sample size
  were deliberately truncated.  Formal promotion remains locked to the
  preregistered 75-scene result.
- Launched the bounded formal chain in unified session 26254.  Collection is at
  `artifacts/lidar_forked_action_data_gate_75scenes_20260904_v1`; a passing data
  Gate automatically starts critic fitting at
  `artifacts/lidar_residual_critic_gate_75scenes_20260904_v1`.  One-time health
  check found 300 full-observation anchors saved, 16/3,000 atomic branches
  complete, eight branches active, 2,976 pending, PID 42220 alive and registered
  with CUDA, and no failure sentinel.  Stop monitoring until the user asks.
- Collection subsequently completed 3,000/3,000 branches in 5,336.1 s and
  passed all six data checks: 9.63% unsafe, 5.70% collision, 0.30% boundary,
  100% action-diverse anchors, and 57.33% paired safety/resource variation.
  `toward_obstacle` produced 19.67% collisions versus 0.33% for
  `away_from_obstacle`, validating the intended fixed-state directional signal.
- The first critic run was invalidated by a globally off-by-one anchor metadata
  index and stopped after fold 0 when unstable `log(expm1(x))` overflowed inside
  the residual inverse-softplus.  Collection is unaffected: its branches use
  anchor objects directly, and metadata/list observation hashes align 300/300.
  The repaired loader ignores legacy numeric indices, binds by anchor ID and
  canonical row, and verifies all hashes.  Stable inverse-softplus removes the
  overflow; focused tests pass 16/16.
- Formal repaired run `artifacts/lidar_residual_critic_gate_75scenes_20260904_v2`
  completed all three held-out folds in 6.47 s on CUDA with finite checkpoints.
  Residual scene-Brier was 0.085337 versus geometry 0.093516 (8.75% lower) and
  AUROC 0.90518 versus 0.88687.  It tied geometry in fold 0 by selecting epoch
  zero and won folds 1--2.  However dangerous false-safe was 4.06% versus 3.92%,
  violating the locked non-inferiority requirement; ECE also worsened from
  0.05576 to 0.06043.  Final decision: do not modify the R3 actor.
- The preregistered control is scientifically informative: direct
  LiDAR+action achieved Brier 0.087060 and dangerous false-safe 3.833%, whereas
  LiDAR without current action was worst at Brier 0.102364 and essentially zero
  matched-action sensitivity.  This supports action-conditioned perception but
  does not post-hoc authorize the direct model as the primary method.

## 2026-09-04 independent LiDAR/action confirmation

- Froze `lidar_action_direct` as the new primary hypothesis and reserved the
  untouched offsets 15--29 in every source distance bucket as a disjoint
  75-scene confirmation set.  All three development-fold weights,
  normalizations, temperatures, the 0.90 declaration threshold, and metrics are
  immutable before seeing confirmation outcomes.
- Added joint dangerous false-safe, false-safe conditional on infeasibility,
  unsafe-among-accepted, declaration coverage, and deterministic 10,000-draw
  paired-scene bootstrap intervals.  Promotion requires both predictive
  superiority and safety non-inferiority; scene is the sole resampling unit.
- Added bucket-offset collection and corrected future `anchor_index`
  provenance.  The changed-path suite passes 19/19.  A five-scene/50-branch
  smoke used exactly scenes 15, 45, 75, 105, and 135, wrote canonical indices
  0--4, and completed without failure; it is not evidence.
- Launched the resumable confirmation chain in unified session 58114.  Formal
  collection is `artifacts/lidar_forked_action_confirmation_75scenes_20260904_v1`;
  a passing data Gate automatically invokes frozen evaluation at
  `artifacts/independent_lidar_action_confirmation_20260904_v1`.  One-time
  health check: 300 anchors persisted, 39/3,000 branches complete, eight active,
  2,953 pending, PID 120833 alive and registered with CUDA, no failure sentinel.
  Stop monitoring until the user asks.
- Confirmation collection stopped after 2,493 branch files because two wholly
  unstarted scene-137 anchors reconstructed 25.7 and 51.0 micrometers below the
  legal 0.5 m altitude.  A bounded position sanitizer now projects only errors
  within eight float32 ulps at world scale (about 3.8 mm); a 1 cm test violation
  remains rejected.  Migration changes only anchors with no saved branches and
  recomputes their observations, hashes, R3 actions, and candidate actions.
- The expanded focused suite passes 20/20.  Resume in unified session 48369
  migrated exactly two positions and zero velocities, retained all completed
  files, and advanced to 2,513/3,000 with eight active, 479 pending, no failure
  sentinel, and PID 152338 registered with CUDA.  Frozen independent evaluation
  remains chained after a passing collection Gate.  Stop monitoring until the
  user asks.
- Confirmation collection subsequently completed 3,000/3,000 in 2,039.4 s of
  resumed wall time and passed all data checks.  The held-out half has 13.4%
  unsafe, 8.2% collision, 1.27% boundary contact, 100% action diversity, 50.67%
  paired outcome/resource variation, and 23.67% mixed safe/unsafe anchors.
- The independent frozen evaluation is terminal at
  `artifacts/independent_lidar_action_confirmation_20260904_v1`.  Direct
  LiDAR/action ensemble Brier 0.085261 improves geometry 0.091324 by 6.64%, and
  each frozen member improves Brier.  However its paired 75-scene Brier
  difference CI [-0.023109, +0.016800] includes zero; scene wins/losses are
  48/27 with high dispersion.
- More importantly, direct dangerous false-safe is 4.016% versus geometry
  2.984% (difference +1.031 percentage points; paired scene 95% CI
  [-0.651,+3.596] points).  Conditional on infeasibility it is 15.23% versus
  11.32%; unsafe among accepted is 6.45% versus 5.42%.  The direct model also
  declares safe more often (62.27% versus 55.04%).
- Failure concentrates in the >4,000 m bucket: direct vs geometry Brier
  0.19358/0.17125 and joint dangerous false-safe 11.92%/7.17%.  It improves
  Brier in 500--1,500, 1,500--2,500, and 2,500--4,000 m, showing useful
  perception conditional on regime but unsafe global extrapolation.  Frozen
  0.90-threshold confirmation therefore rejects actor integration; a post-hoc
  direct threshold near 0.985 would match only the point dangerous rate while
  reducing coverage to 48.68%, and is exploratory rather than evidence.

## 2026-09-04 certified-meet confirmation and RCPS repair

- Fresh nominal collection completed 75/75, and paired fixed-state collection
  completed 3,000/3,000 branches. The paired-data Gate passed with 5.033%
  unsafe branches, 2.8% collision, 1.133% boundary contact, 100% action
  diversity, and 53.667% paired safety/resource variation.
- The frozen certified meet improved selective coverage from geometry's
  66.136% to 72.231% (paired scene bootstrap difference 95% CI
  [4.253, 7.853] points), with zero probability-dominance and budget-
  monotonicity violations. Its empirical scene false-safe risk was 6.486%, so
  the locked 5% Gate correctly rejected actor modification. Geometry itself was
  5.932%, indicating threshold-transfer failure rather than a meet-only ranking
  failure.
- A diagnostic threshold sweep found that meet reaches about 5% empirical risk
  near 0.815 while retaining 66.10% coverage, versus geometry's 61.42% at its
  own about-5% threshold. These values are post-hoc and are not evidence.
- The official RCPS Hoeffding--Bentkus--empirical-Bennett construction gives a
  95% upper mean-risk bound of 6.0604% for 75 zero-loss scenes. Therefore the
  prior 75-scene design was mathematically incapable of certifying risk <=5%,
  even under perfect observed outcomes.
- Added endpoint-safe HBB bounds and nested RCPS threshold selection in
  `experiments/rechargeability_safety/risk_control.py`, plus frozen calibration
  and confirmation evaluators. The changed suite passes 19/19.
- New nominal rollouts save exact simulator pre-action positions and velocities
  alongside compact observations. A five-scene/50-branch smoke recovered the
  source prefix exactly (maximum error 0.0), required no state migration, and
  produced valid anchor hashes. Legacy compact-state decoding remains only for
  old artifacts.
- Frozen next protocol: 450 new calibration scenes, then only after a passing
  non-vacuous HBB Gate, 150 new untouched confirmation scenes. The already
  inspected fresh 75 scenes are development-only.
- Launched the detached chain with PGID 219370. One-time health validation at
  28 s found 25/450 calibration-source rollouts persisted, eight active, 417
  pending, exact-state snapshot format declared, CUDA registered, and no
  failure/result sentinel. Stop monitoring until the user asks.
- At 15,952/18,000 branches, changed only orchestration parallelism from 8 to
  12 workers. The machine has 24 physical cores; measured private memory made
  12 safe while 16 was rejected as unnecessarily close to the 15 GiB RAM
  ceiling. TERM stopped the old process group after atomic persistence; the
  resume loader retained all 15,952 files and restarted only uncommitted work.
  New PGID 301762 is healthy with 12 active branches, about 7.3--7.6 GiB total
  RAM used, zero swap, and no failure sentinel. Scientific semantics and frozen
  seeds are unchanged.
- User-requested recoverable pause at 2026-09-04 21:19:54 CST stopped PGID
  301762. Exactly 16,692/18,000 branch JSON files are preserved; the 12 active
  but uncommitted branches were not counted, so resume has 1,308 remaining.
  No relevant Python worker remains. `PAUSED.json` records anchor hashes, the
  ordered branch digest, and the exact 12-worker resume command.
- User resumed the chain. Anchor JSON/NPZ hashes, ordered digest, and all 16,692
  saved branch files matched the pause manifest. New PGID 2915 has 12 live
  workers; `PAUSED.json` was cleared by the resume path, with no failure
  sentinel and zero swap usage. Stop monitoring until requested.
- Calibration completed and passed all RCPS Gates: meet threshold 0.9315,
  empirical scene risk 2.8194%, HBB 95% upper bound 4.9922%, and coverage
  48.9844%; geometry threshold 0.9540, bound 4.9673%, coverage 45.2641%.
  Meet coverage gain is 3.7204 points. The original chain then stopped because
  `uv run --python .venv/bin/python -c` parsed `-c` as a uv option; no
  confirmation data had been touched.
- Fixed both guard invocations and added a confirmation-only fail-closed chain.
  It verified calibration protocol/model/risk-curve hashes and launched exactly
  the frozen 150-scene confirmation under PGID 13216 with 12 workers. Initial
  health was 5/150 nominal rollouts persisted, 12 active, no failure sentinel,
  zero swap. Calibration data are not rerun.
- Confirmation nominal source completed 150/150, but the first fork stopped at
  3,069/6,000 on `scene_077_anchor_00`. Its first projection left horizontal
  speed 6.22e-7 m/s above 20; the second altered velocity by 1.91e-6 m/s and
  changed two compact-observation floats (max 1.19e-7), correctly tripping the
  exact hash.
- Added a strict, bitwise-idempotent float32 norm projection and regression over
  the real vector plus 721 directions. The first resume then exposed a partially
  completed legacy anchor whose old snapshot must not be recanonicalized.
  Fork workers now restore registered hash-bound velocity bits after physical
  validation; preparation/migration remains the single canonicalization point.
- Focused suite passes 21/21. Direct worker reproductions of both failing
  formal anchors have identical 2055-dimensional arrays, hashes, and max error
  zero. Second resume PGID 29193 retained all files and advanced to 3,071 with
  12 active workers, no failure sentinel, and zero swap.

## 2026-09-05 independent confirmation and closed-loop transfer

- The 150-scene untouched confirmation completed all 6,000 branches and passed
  its seven preregistered checks. Certified Meet at threshold 0.9315 obtained
  2.0133% empirical scene false-safe risk and 50.1456% coverage; geometry at
  0.9540 obtained 1.9465% risk and 46.4089% coverage. The paired coverage gain
  was 3.7367 points (95% bootstrap CI 2.9722--4.4723), while the risk difference
  CI included zero. Meet's advantage is selective coverage under the same risk
  cap, not globally superior calibration and not a worst-case guarantee.
- Added `geometry_action_features` for deployment. Against the confirmation
  dataset, all 6,000 x 26 values are bitwise identical to the offline critic
  inputs (max absolute difference zero). Related feature/state tests pass
  11/11; the full fork patch suite remains 21/21.
- Froze `docs/BOUNDED_RCPS_CLOSED_LOOP_PILOT_PROTOCOL_V1.md`: 75 new cases,
  five distance buckets, five balanced available-energy levels, and paired
  ungated/geometry/Meet arms. A rejection is an absorbing return commitment;
  actor and HOCBF are unchanged. The protocol explicitly treats this as
  adaptive-distribution mechanism evidence, not as extension of the RCPS
  certificate.
- The five-case smoke completed all 15 atomic records, exercised both Gate
  acceptance and early return, and passed an all-complete resume reopen. Formal
  output is `artifacts/rcps_closed_loop_pilot_75cases_20260905_v1`, PGID 111248.
  One-time health: 11/225 complete, 12 active, 202 pending, no failure marker,
  zero swap. Stop monitoring until requested.

## 2026-09-05 bounded closed-loop pilot result

- Formal pilot completed 225/225 uncensored paired rollouts in 625.54 s; case,
  protocol, checkpoint, budget, world, and obstacle-layout provenance validate.
  The stale `PROGRESS.json` status remains `RUNNING` despite active=0 and
  complete=225; matching terminal `RESULT.json` and `COMPLETED.json` plus no
  live process make this a telemetry-only defect.
- Frozen result is `DO_NOT_PROMOTE`: Meet has one more energy-exhaustion event
  than geometry and therefore fails `meet_unsafe_not_worse_than_geometry`.
  Meet vs geometry is 5 vs 3 task completions, 71 vs 72 safe returns, and 2 vs
  1 unsafe terminals. None of these paired differences establishes Meet
  superiority (task exact McNemar p=.625; safety p=1.0).
- Ungated R3 completes 72/75 tasks but has 10/75 unsafe terminals. Meet reduces
  that to 2/75 (paired bootstrap difference -10.67 points, 95% CI
  [-18.67,-2.67]), but completion collapses to 5/75. The unadjusted unsafe
  McNemar p=.0215 becomes .1289 after the eight-comparison Holm family.
- Meet commits at step zero in 56/75 cases and eventually commits in 70/75;
  geometry is 58/75 and 72/75. Meet completes no task at >=1500 m. This is
  mission abandonment rather than useful safe navigation.
- The causal semantic flaw is explicit. The critic label is task action + task
  continuation + return; low feasibility was incorrectly mapped to immediate
  return. Case 23 rejects then exhausts; cases 42/57 reject then time out;
  case 72 Meet returns at step 0 and exhausts while geometry moves four task
  steps then returns safely. Pointwise RCPS calibration also does not cover an
  adaptively repeated, absorbing stopping rule.
- Next design: separate `V_continue(s,a,B)` and `V_return(s,B)`. Return only
  when the return head is viable; if neither is viable, choose a recovery
  action maximizing next-state return viability. Calibrate the whole stopping
  trajectory with time-uniform or persistent/hysteretic control, and add an
  absolute task-utility Gate. Analysis bundle:
  `artifacts/rcps_closed_loop_pilot_75cases_20260905_v1/analysis-output`.

## 2026-09-05 dual-viability correction

- Literature audit confirms that recovery policies, feasible/recoverable sets,
  reach-avoid return filters, battery-augmented state, and reload-state MDPs are
  prior art. `Back to Base` (L4DC 2025) is especially close: it explicitly
  motivates a drone returning to a charging target and constructs a reach-avoid
  VB-CBF safety filter. A plain return critic is not an oral-level contribution.
- Corrected objects are now frozen: `V_R^kappa` for return-now viability and
  `Q_OP^kappa` for one executed action followed immediately by frozen recovery.
  The old completion-then-return label is neither an upper nor lower bound for
  either object; two deterministic counterexamples and the 75-case pilot both
  establish this.
- Added a conditional finite-horizon theorem: if every adaptive task action
  leaves the next state in the recovery-viable level set with conditional error
  `epsilon_t`, then total failure is at most `delta_R + sum epsilon_t`.
  Pointwise scene RCPS does not supply the conditional premise; learned
  closed-loop claims require whole-trajectory calibration/evaluation unless a
  verified successor set is available.
- Frozen `research-question-card.md`, focused literature review, and the
  development-only 3,410-rollout protocol before observing corrected outcomes.
- First smoke exposed that reset-based retargeting rejects a float32 velocity a
  few ulps above the nominal limit after a legal simulator step. Added
  `WorkerInPlaceRetarget`, which changes only goal/horizon bookkeeping and
  asserts bitwise preservation of position and velocity. No projection or
  zero-velocity shortcut is used.
- The resumed 22-rollout smoke completed with 22 atomic records, no failure
  marker, and terminal progress. Focused suite: 14 passed. This validates
  orchestration and invariants only; smoke records are censored and are not
  scientific evidence.
- Launched the full development diagnostic detached under PGID 12322 at
  `artifacts/dual_viability_counterfactual_diagnostic_150scenes_20260905_v1`.
  The single startup health check found 46/3,410 complete, 12 active, 3,352
  pending, all workers alive, and no `FAILED.json`. It is intentionally not
  monitored further; wait for the user to request result analysis.
- Full diagnostic completed: 3,410/3,410, zero censoring, 5,797.6 s. Strict
  anchor-clustered analysis is in
  `artifacts/dual_viability_counterfactual_diagnostic_150scenes_20260905_v1/analysis-output`.
- R3 return-now succeeds at only 275/310 anchors (88.71%, Wilson 95% CI
  84.70-91.77): 17 timeout, 16 collision, two boundary failures. It is useful
  as a baseline but invalid as a high-reliability terminal certificate.
- Corrected option outcomes are primarily state-dependent: 2,766/3,100 safe,
  only 21/310 structurally mixed anchors, and at most 22/310 mixed at any
  energy budget. Ten candidate safe rates span 1.61 pp; paired Cochran
  Q=5.633, p=.776. Energy has Friedman p=3.33e-7 but negligible Kendall
  W=.017. Low discrimination is not generally HOCBF action collapse because
  300/310 anchors retain ten distinct executed actions.
- Historical completion and corrected option labels are not interchangeable:
  historical mean energy is 13.18% capacity versus 7.93% for option. At the
  10.86% budget, positive rates are 36.48% versus 81.45%, an anchor-clustered
  difference of +44.97 pp (95% CI +39.10 to +50.74; Holm p<1e-4).
- Decision: no new binary `Q_OP(x,a)` training from these labels. First repair
  R3 recovery failures; then learn `V_R(x)` with separate structural and
  continuous energy heads and evaluate actions via a successor model. Fresh
  scenes and multiple policy seeds are required for later confirmation.

## 2026-09-05 minimal R3 collision-safe RL selection

- Exact audit found the collision-learning mismatch: R3 terminates on goal or
  energy exhaustion, not collision; task completion pays +100, first obstacle
  contact costs -1.2, and repeated contact costs -0.42. In the 500-task R3
  evaluation, three collision episodes produced 2,242 contact steps, including
  one 2,240-step repair/recontact loop. This is an objective-definition defect,
  not evidence that another network block is missing.
- R3's old Jacobian bridge also has a measured coverage bottleneck. Although
  86.44% of sampled bridge records passed the geometry precheck, only 24.61%
  remained inside the actor-to-anchor trust region during training. The final
  actor still needed HOCBF on 35.87% of evaluation steps and emergency braking
  on 15.16%.
- Primary-source screening covered CPO, PID-Lagrangian, ET-MDP, CaT,
  SoloParkour's off-policy CaT, Recovery RL, SAILR, reach-avoid RL, RCRL,
  Lyapunov projection, sampling-based safe RL, HOCBF, and sampled-data CBF.
  Recovery/intervention methods require an extra policy/critic; expected-CMDP
  methods add a cost critic/dual and do not express binary path safety; learned
  CBFs duplicate known project geometry.
- Selected first candidate is R3-RACT: replace, rather than augment, the old
  bridge and mixed reward critic with a nonnegative constraint-terminated
  reach-avoid Bellman target. Existing twin Q networks estimate discounted
  safe goal reach. The artificial survival weight is
  'q=(1-first_contact)*exp(-kappa*d_projection^2*dt)', where projection
  distance is measured against the sampled-data robust HOCBF feasible set. It
  is not called a collision probability. No new neural network is added.
- A direct CaT multiplication of the current R3 reward is rejected: rewards are
  mixed-sign and ordinary SAC's differential-entropy continuation can be
  negative, so termination need not be monotonically bad. The chosen recursion
  keeps a nonnegative reach-avoid critic and treats entropy only as a temporary
  actor regularizer, annealed to zero.
- The theorem package now proves contraction, the random-absorption
  interpretation, and conservatism relative to physical discounted reach-avoid.
  It explicitly does not call the hazard a collision probability or the learned
  actor a certificate. Runtime safety remains conditional on independently
  valid sampled-data HOCBF assumptions.
- Canonical design: `docs/R3_MINIMAL_COLLISION_SAFE_RL_DESIGN.md`. Literature
  packet: `literature-search-20260905-r3-collision-safe-rl/`. No training was
  started in this design phase.
- A fixed-policy random finite-MDP numerical check reproduced the
  killed-kernel potential identity to '1.15e-14' maximum error and found zero
  monotonicity violation when increasing 'kappa'. This validates the algebra,
  not neural training or physical safety.

### 2026-09-05 implementation correction and actual experiment launch

- The preliminary RACT design required correction, not direct implementation:
  normalized horizontal actions undergo radial clipping, so raw normalized
  changes are not shield corrections. The implemented signal measures physical
  acceleration differences, normalized by 2*sqrt(5^2+3^2), squared/integrated
  over every 0.05s substep. Emergency output is a proxy, not a projection proof.
- Hard zero survival now means actual obstacle/boundary contact only. Emergency
  and fallback are not automatically terminal. Nonterminal initial-state goal
  indexing uses gamma^(tau_goal-1), matching immediate goal payoff one.
- Kappa uses median whole-success-trajectory positive correction integral,
  avoiding the preliminary per-second-half-life scale's long-horizon collapse.
  Full-state exact Bellman identities are not neural convergence guarantees;
  single-frame partial observation and LiDAR point-ball/cylinder mismatch are
  explicit limitations. This is SAC-derived, not unchanged soft SAC.
- Four objective unit tests passed. Ten 8-step smoke episodes completed with
  paired observation/layout hash checks and CUDA inference; their deliberate
  timeouts say nothing about navigation ability.
- Started `artifacts/r3_collision_interface_audit_50pairs_20260905_v1` at
  12:59:03 local. This is 50 paired development tasks, ordinary vs robust HOCBF,
  frozen R3 accelerated-reproduction 500k checkpoint, 8 workers and central GPU.
  Startup confirmed RUNNING, 49/100 records in 59.4 seconds, Python PID 67750.
  Completed episodes are atomic/resumable; an interrupted in-flight episode
  restarts. No automated policy-training promotion and no continuous monitoring.
- New implementation files: `experiments/r3_collision/core.py`,
  `scripts/run_r3_collision_interface_audit.py`, and objective unit tests.
  Original simulator and R3 checkpoint are untouched.

### 2026-09-05 matched reach-avoid trainer implemented and launched

- Follow-up user authorization: implement the next step after interface audit.
  Audited dataset has 49/50 safe goals under both filters and zero contacts.
- Pretraining hazard diagnostic found 7/49 robust safe trajectories had weight
  below .01 under old kappa=.372132. Revised calibration explicitly BEFORE
  short training: all-safe-trajectory H90=15.1510, kappa=.0457492043,
  min observed whole-trajectory weight=.277313, gamma=.9988772156 unchanged.
  This is an acknowledged design revision, not unchanged preregistration.
- `experiments/r3_collision/training.py`: standalone environment wrapper,
  replay storing beta/phi/q/correction-integral, and ReachAvoidSAC. Exactly two
  critics, original actor architecture/input. Physical contact terminates only
  the wrapper episode after the original repair; no physical environment edits.
  Timeout uses terminal observation bootstrap, not reset observation.
- Both arms: 50k transitions, stochastic frozen R3 actor for first 10k;
  complete-path shaped MC return initializes reset critic; no timeout-as-zero
  MC labels. After warmup, TD-only critic and actor updates; no added MC loss.
  Warm-start completion-selection bias is explicit. Actor LR3e-5, critic3e-4,
  batch256, one update per transition, tau=.005; actor entropy .001 anneals
  to zero by transition30k, never enters critic target.
- Eight tests passed, plus real CUDA smoke: 5k+5k transitions and 30 paired
  evaluation rollouts. Both arms had 3274 MC and 1000 actor updates, identical
  initial actor/critic hashes, finite losses. Smoke shield-off safe goals stayed
  2/5 for frozen R3 and both new arms: no demonstrated autonomous safety gain.
  This smoke validates execution, not efficacy or significance.
- Job: `artifacts/r3_reach_avoid_pair_50k_each_20260905_v1`. Two sequential
  training arms, then 50 tasks x shield on/off x three models =300 development
  evaluations, seeds710001/720001 separated from smoke seeds610001/620001.
  No automatic scale-up. Eight environments single-threaded, CUDA central.
- Save every10k: model/optimizers+replay+RNG in atomically published checkpoint
  directory. Resume restarts unfinished environment episodes and clears pending
  MC trajectories; it is not bitwise simulator-state continuation. PAUSE_REQUEST
  or SIGTERM requests save/exit at next1k chunk boundary.
- Batch256 benchmark on smoke copy: .03617s/update. Expected roughly1–2h for
  full development job including evaluations, subject to scene/runtime variation.

## 2026-09-05 — failure-driven from-scratch safe navigation review

- The reach-avoid pair completed and failed: shield-on safe goals hard7/50,
  continuous1/50 versus frozenR3 47/50. Do not resume/scale this branch.
- Read-only audit on 1,024 fixed old replay states, seed930001: unshaped
  min-twinQ values outside [0,1] reach38.77%/45.31% at50k. Necessary-range
  violation is confirmed; causal uniqueness and probability calibration are not.
- User permits random initialization and treats R3 only as experience/reference.
  Literature packet screens17/retains14 with reading-depth caveats. FOCOPS,
  C-TRPO, SafeMPO and author implementations examined; no new experiment started.
- Prefer first-order on-policy FOCOPS as first new baseline, matched with plain
  PPO under the same finite-task contract. Prior R7 PID-PPO used correction cost
  plus contact, defaultbudget300, not an episode collision-probability constraint.
- Added FH-CMDP exact event identity and finite-horizon recurrence; these are
  standard modeling facts, not novel or neural-safety theorems. Undiscounted
  event probability cannot be replaced by gamma_c=.99 cost without disclosure.
- Next implementation protocol in docs/SAFE_NAVIGATION_FROM_SCRATCH_REDESIGN.md.
  Skill use influenced evidence-depth labels, read-only failure analysis, and
  separation of mathematical identities from approximate training guarantees.

## 2026-09-05 — critical forward-citation evidence

- User requested downstream usage and criticism, not an automatic new training
  run. Packet: literature-search-20260905-safe-rl-forward-citations/.
- Safety-Gymnasium Table1 FOCOPS CarGoal normalized R/C: authorimplementation
  .79/2.45 versus SafePO .52/.93. Cost improvement is not a Pareto dominance
  claim; task/implementation matter. PointGoal SafePO cost1.32 still overbudget.
- ICML2024 Langevin Table2 (2.5Msteps, budget25): FOCOPS PointGoal cost38.08,
  CarGoal20.09. PPOLag16.48/19.50, respectively. Does not establish universal
  ranking; supports a matched simple Lagrangian control.
- IROS2023 multiplicative-value paper evaluates FOCOPS in rawLiDAR/Gazebo and
  reports context-dependent sample efficiency; SAC better on some navigation.
  New multiplicative variants also have timeout/overconservatism limitations.
- SafeMPO ICLR2026 Table1 at10M: B25 mean costs32.23/30.87/32.92, all over25.
  Downgraded to theory reference; own Section5 acknowledges boundary shift.
- CRAX preprint2026 supports FOCOPS on some tasks but Goal costs exceed25 on
  levels1/3. Main text500M vsappendix100M budget conflict recorded; H100/2048env
  timing not transferable. ProSh is verified AAMAS2026extendedabstract with
  arXivlongversion, not verifiedICLR2026 despite a university news grouping.
- SafeOR-Gym gives resource-domain negative evidence but not UAV applicability.
  FCSRL/CAL/new-method independent forward evidence not established this pass.
- Updated design: PPOLag vsFOCOPS samephysicaleventcost; plainPPOlearningcheck.
  No preselected winner, no module stack, no training/code edits. pypdf used
  ephemerally through uv to read public PDFs in memory after browser errors.

## 2026-09-05 existing-asset audit: directionality is lost at final pooling

- User explicitly encourages critical reuse, especially structured LiDAR.
- Main-agent code review: `experiments/jacobian_energy_bridge/features.py`
  has two circular-azimuth Conv2d layers followed by global mean/max. Goal
  features fuse only afterwards. Thus z(g,roll(L))=z(g,L) for fixed g, any
  weights in exact arithmetic. This is a representation limitation shared by
  R3/R7, not a PPO-only problem; cannot establish sole historical failure cause.
- Frozen accelerated R3 500k, actual `_update_lidar` calls: position
  (2000,2000,200), velocity(10,0,0), goal(2500,2000,200), one radius50 cylinder
  in each cardinal direction at center distance70. Final embeddings identical;
  nominal deterministic action differences <=5.97e-8. Ordered2x16 mean/max
  readout of the same convolution maps differs by2.0341. No training/steps or
  checkpoint writes. Reproduction command and hashes in audit document.
- Existing tests validate intermediate equivariance, not downstream bearing
  identifiability; all56 encoder/R7/reachavoid tests plus5 relevantphysical
  environment tests passed. No nativefullsuite rerun this turn.
- Reuse full1024rays, hit/range channels, localconvolutions, NumPy ray batches,
  parallel workers, physicalrepair, atomiccheckpoint pattern, scenehash/splits.
  Do not reuse globalpool unchanged, correctioncost/PID as eventCMDP, oldgate
  thresholds as theoremrequirements, or unfinishedsimulatorresume as bitwise.
- Independentenergyconfirmation: Brier .08526vs.09132 but sceneCI crosses0;
  dangerousjointfalse-safe4.0156%vs2.9844%, promotionfalse. Reuse diagnostics,
  data and policy-conditionedestimand; not evidence of a safe deployablecritic.
- Design updated: diagnose/repair directionalreadout first; ordinaryPPO
  learningcheck then matchedPPOLag/FOCOPS. 2x16 readout onlycandidate, adds
  1,015,808 parameters across actor/critic firstlayers; require runtime/capacity
  accounting. No policy source changes or new training launched.
- Deliverable: docs/SAFE_NAVIGATION_ASSET_REUSE_AUDIT.md.

## 2026-09-05 directional implementation and first plain-PPO pilot

- Implemented isolated directional_navigation/features.py and environment.py;
  no mutation of old R3/R7 extractor/checkpoints. Ordered2x16 readout, full
  1024 rays, extra remaining-time scalar. Actor/value extractors separate.
- Supervised direction probe v2:48train/24heldout scenes x4 rotations,
  equal139570parameters/300updates. Globaltiled MSE.500098/90deg versus ordered
  .000258/.75deg. Diagnostic targets NOT used for navigation training.
- New script train_directional_ppo.py uses stock installedSB3PPO, gamma1,
  GAE.95,Gaussian preclip likelihood, rewardscale.01, firstphysicalcontact true
  terminal,deadline true terminal. Contact-step repair/goal reward suppressed.
  No costcritic,multiplier,PID,energyhead,shield or routeplanner.
- Keyed taskcursor, optimizer+RNG checkpoints and per-checkpoint evaluation
  files verified with128→256mechanicalresume. Unfinished tasks restart at next
  keyed task, not bitwise simulatorresume. StartupSB3seedqueue explicitlyclear.
- 153tests passed;12new rerun aftertypingfixes. Ruff/Pyrightclean,compilepass.
  Pyright originally18frameworktypingfindings fixed without mathchanges.
- Old fullscene preflight8192steps ~13s, steady4096 ~5.807s; loaddependent.
  Finalv3 detached pilot launched via processgroup157451, fromscratch262144,
  eightenvs24obstacles, then50heldoutdevelopmenttasks. No auto500k promotion.
  Artifact: artifacts/directional_ppo_navigation_pilot_20260905_v3.
- Report: docs/DIRECTIONAL_PPO_PILOT.md. Only initialhealth verification then
  handoff; no continuing outcome monitoring requested.

## 2026-09-05 complete-task PPO-Lagrangian comparison

- Prior plainPPO finished262144/50dev,29safe20contact1timeout,363.18s.
  Frozen replay this turn matches all50outcomes/lengths, distinguishes18obstacle
  and2boundary contacts. Near-boundary terminal states show approachingvelocity,
  but no sole-cause or extra independentperformance claim.
- Verified OmniSafe PPOLag combinedadvantage and projectedLagrange source via
  officialGitHub. Implemented finite-taskadaptation in new lagrangian.py/cohort.py.
  One257parametercosthead sharescritictrunk; actorunchanged. Botharms trainboth
  values; onlylambda0vsadaptive differs. NoPID,shield,energyhead,worldmodel.
- Fullfixed-policy cohorts avoid completed-onlyshorttrajectorycostbias; idle
  slots parked(noextra physics). MCcostgamma1/lambda1 includeslatecontact,
  rewardGAE.95, jointadvnormalization, uniform rewardscale.01. Nonlinear deep
  approximation not a safetyguarantee; costVrange recorded.
- Firstallfailurecohort constantcostadv is removedbycentering, so lambda rise
  doesnotmanufacture safebehavior information; noted as limitation. Preflight
  lambda0→.95→1.9, control0; costVsecondrange~.95–1.029. Sixinitialtests plus
  twoextraexactCMDP/partialcohorttests,161regressiontotalpassed. Ruff/Pyright0.
- Preflightresume1→2cohorts restored optimizer/RNG/lambda; initialweightsmatch.
  Matchedseed schedule comparison is at fixedtaskgroups, NOT equaltransitions.
- Main protocol64cohorts×8tasks=512trainingtasks perarm; max2,048,000steps/arm.
  Lagrangianthencontrol sequential; finaldet50 andstoch50 separated. δ.05only
  exploratory. First3devtasks were inspected inmechanicalpreflight; notformal
  untouchedholdout. Noautopromotion oroutcomemonitoring.
- Detachedlaunch processgroup183290:
  artifacts/directional_lagrangian_pair_64cohorts_20260905_v1.
  Protocol: docs/DIRECTIONAL_PPO_LAGRANGIAN_PROTOCOL.md.

## 2026-09-05 completed-pair failure diagnosis and minimal intervention

- BotharmsCOMPLETE2512.26s; lagtraining11safe420contact81timeout versus
  control220safe275contact17timeout. Deterministic0/44/6vs25/24/1;
  stochastic0/45/5vs23/26/1. Singleinit,512matchedtasks/64cohorts notmatchedsteps.
- Newread-onlydiagnosticscript replays1/2/8/16/32/64frompre-updateRNGcheckpoints
  botharms;96tasks matchoriginaloutcomes/steps; nooptimizerstep orweightchange.
- Lag16/32/64 weightedcost/rewardgradientnorm10.79/18.88/43.01; currentbatchcost
  R²-.594/-.344/-.332. Control64R²-.317 too; criticerroralone notdifferentialcause.
  Gradientratio is4096fixedsamplepre-update proxy, notAdamstep ortruegradient.
- AllfailureMCcentering equalsnegativecenteredbaseline, demonstratedalgebraically;
  no claimMCgenerallybiased. Jointstandardization cancelscommon(1+lambda)scale.
  Lag81/92zero-costtasks timeout; notsafegoal, nor evidenceofsafehoverpolicy.
- PrimaryliteraturePID-LagICML2020§7 supports reward/dualunitrelationship;
  GAEICLR2016supports bias/variancedistinction; CRPOwouldnotcreateabsentcostsignal.
- Evidencebundle: artifacts/directional_lagrangian_failure_analysis_20260905_v1
  containsreport,statsappendix,catalog,2vectorfigures,summary,12replayJSONs.
- Nextsinglefactor: existingrunner--dual-lr.01instead1, allotherchoicesfixed,
  samerandominitialization/schedule64cohortsplusdet/stoch50each, repeatcontrolfor
  reproducibility. This isdiagnostic, notguaranteedcorrectsteporlearnedsafety.
- Launchedslowdualpair processgroup202678/trainer202681; firstcohort2174steps,
  lambda.0095,checkpoint0001,finiteupdates.24relatedtests,Ruff/compile/Pyright
  pass. Finalhandoff:noongoingmonitoring,waitforuserrequest.

## 2026-09-05 timeout research after slowdual completion

- SlowdualCOMPLETE2668.36s,det26/13/11,stoch22/16/12; zero-lambdacontrol
  exactlyreproducesold25/24/1and23/26/1. Goalimprovementnotestablished;
  contactMcNemarrawp.035/.041,4contrastHolm.139. Noformal500claims.
- Newtracewrapperreplays200evaltaskswithfullresolutionmetrics,sampledplots;
  allseed/outcomes/steps/rawreturnmatchoriginal,weightsunchanged.
- Det10/11timeoutstail100sprogress<50m;2persistentlow-speedstalls,7inefficient
  movementflags. Stoch12/12inefficient,8everwithin60m,4within5—6mnotwithin5m.
  Somefarstallsandzmisses,notuniformnear-goalbug. Nocontactorphysicsrepair.
- Rewardidentityverifiedto2.1e-4rawrewardfloatprecision;positivevelocitybonus
  mayrewardinefficientmotionbutmanyplateaurewardsnegative,notsolecause.
- ReviewedOmniSafegae-rtgbranch/PPOLagcosttrace.95andGAEpaper. Implemented
  isolatedcost_trace.pyactoradvantagechangeonly,criticMCtargets/dualunchanged.
  Correctedvaluerangediagnosticsbecauselegacyrc-acidentityrequiresMCadv.
- Newrunnerforksslowdualcheckpoint64withoptimizer/RNG/lambda.26675into
  gae95andmc_control(adaptivedualBOTH),additional32cohorts256taskseach.
  Percohortrecordscommittedwithcheckpoint,logscanberegeneratedonresume.
- Protocol docs/DIRECTIONAL_COST_TRACE_PROTOCOL.md;analysisbundle
  artifacts/directional_slowdual_timeout_audit_20260905_v1.29testsandRuffpass;
  sixloggingtypefindingsfixedwithdictcheck,Pyright0. Preflightresumeinprogress.
- Preflightresume65→66passedbotharms,16uniquetasksperarm,matchedforkweights,
  nologduplicates. Main32extracohortpairlaunchedinbackgroundwithdet/stoch50
  perarm;noautomaticpromotion. Artifactdirectional_cost_trace_pair_20260905_v1.

## 2026-09-05 collision contract overrides earlier launch

- Cost-trace v1 paused/quarantined after update reproducibility discrepancy;
  v2 restored original CuDNN deterministic flags, two preflights completed.
  No v2 main launch: user rejects first-contact episode termination.
- User freezes first -1.2 / consecutive fixed -0.42 reward, reset after one
  clean policy step. Cost remains 1 per contacted policy step. Later explicitly
  requests only unified collision counts, no boundary/obstacle split statistics.
- New RecoveryCohort and LockedRecoveryPlant preserve old source files, enforce
  reward constants, zero all velocity at repair, keep same goal after contact.
  New evaluation schema omits collision type and distinguishes goal after
  contact from zero-contact arrival. Legacy outcome exists only inside reused
  evaluation collector and is removed from saved records.
- Frozen-policy recovery audit registered, two checkpoint64 policies × two
  action modes ×50 developer scenes. No optimizer updates or automatic training.
  New repeat-cost targets require cumulative counts, not binary terminal labels;
  old .05 probability budget cannot silently become .05 contact-count budget.
- Four-task smoke paused via SIGTERM at complete group, resumed remaining modes,
  no duplicate saved tasks and frozen weights unchanged. Not efficacy evidence.
- Static check first found5 typing errors (base Env type and optional shape),
  fixed with cast/assert; added finite record validation. Patch hunk initially
  failed after formatting; reapplied against current source. Ruff import order
  fixed mechanically. No historical environment/reward source changed.
- Final34tests pass, Ruffclean, Pyright0; original slowdual source hashes all
  unchanged. Main recovery audit processgroup38284/trainer38287 detached;
  first16tasks committed by46.1s, weights unchanged. No monitoring beyond initial
  healthcheck. Artifact directional_recovery_audit_20260905_v1; user will ask later.

## 2026-09-06 recovery PPO baseline implementation

- User approved training continuation; asks original R3 collision semantics.
  Verified git7e30d8f config collision_terminal=False and environment termination
  excludes collision. Original R3 != later RACT/first-contact directional PPO.
- No locked environment changes. New recovery_ppo.py audits true count-to-go
  and performs reward-only PPO with inactive binary cost head. Old MC collector
  already accumulates repeated costs; old probability dual/trace assumptions
  are not reused. No arbitrary safety budget introduced; this is baseline work.
- New train_recovery_ppo.py resumes weights/optimizer/RNG, atomic group records,
  fixed keyed development evaluation. Source control64;32extra cohorts256tasks,
  det/stoch50each after training; no automatic promotion or monitoring.
- 38tests passed; Ruff/Pyrightclean. Added float32 two-ULP tolerance for MC label
  arithmetic, not relaxed physical/success/count rules. One documentation patch
  hunk failed before writing, reapplied against exact text; test import sorted.
- GPU preflight a one group + resume second group vs independent b two groups:
  models/optimizers bitwise equal at65/66; all episodes and nontiming stats exact.
  No duplicated seeds/logs. Existing recovery audit source hashes unchanged.
- Main background launch artifacts/recovery_ppo_baseline_20260906_v1. Awaiting
  first checkpoint health only, no efficacy conclusion from preflight.
- Main first checkpoint65 now verified bitwise matches preflight;16786newsteps,
  finiteupdate24.13s, noerror. Detachedprocessgroup54052/trainer54055.
  Hand off without monitoring;32additionalcohorts then100developer evaluations.

## 2026-09-06 recovery PPO credit-assignment follow-up

- Regression diagnosis replayed cohorts65/81/89/96 exactly with no weight
  mutation. Weak GAE/MC alignment motivates a controlled test; it is not proof
  that MC will work. A simple update-count explanation was not supported.
- New actor-only trace wrapper preserves the reward critic's original GAE(.95)
  targets. GAE control checkpoint65 reproduces the old baseline model and
  optimizer bitwise; MC checkpoint differs as intended. MC actor advantage has
  much higher variance (~6.999 versus ~0.448 in cohort65), so failure remains a
  real possibility and must be decided by paired outcomes.
- Focused23tests pass; Ruff clean; Pyright0. Signal pause at complete checkpoint
  and `--resume` completion were verified with no replayed saved cohort.
- Formal development pair is running at
  `artifacts/recovery_reward_credit_pair_20260906_v1`, PID81355. It uses the
  same checkpoint64, matched seeds,16cohorts/128tasks per arm and100 eval tasks
  per arm. First MC checkpoint65 matches preflight exactly, metrics are finite,
  and no error file exists. Do not monitor until user asks.
- User froze lean experiment startup: only necessary changed-code checks, one
  tiny smoke for a new path, first-checkpoint health, and resume testing only
  when that path changed. No large preliminary gate chains without explicit
  approval and a concrete risk/stopping rule.

## 2026-09-06 deeper reading and critic-budget continuation

- User approved a longer experiment and full-paper reading. Read9 main papers
  at the methods/theorem/experiment sections listed in
  literature-search-20260906-ppo-critic-training/papers.md; PDF originals and
  page-tagged text are local. PPG AppendixC/D and official Spinning Up value
  loop motivate a critic-only completion, not full PPG or a new safety theorem.
- KL-coupled value training ran only1.3896 equivalent passes on average in
  the latest GAE arm and.2065 in the MC arm. The planned value budget was10.
  New wrapper executes the original PPO update, then only fits the unused
  value budget; frozen lambda-return labels and all actor parameters unchanged
  during the extra phase. Independent shuffle avoids changing the actor RNG.
- Nine focused tests passed, including forced KL stop, unchanged actor and
  control update, optimizer-state restoration and eval resume; Ruff/Pyright
  clean. Old storage/restore functions reused; no separate long resume gate.
- Background PID95809, artifacts/recovery_critic_completion_20260906_v1.
  Four scheduled arms: two optimizer variants × two matched continuation random
  streams, all fromcontrol64.96cohorts/768tasks each; checkpoint160 is final.
  Every32cohorts50development deterministic tasks, then500new deterministic
  scenarios and50development stochastic tasks per final arm. No evaluation
  result changes the schedule. Estimated4—7h and resumable at each saved group.
- Initial health confirmed through checkpoint66 after61.76s session time:
  actor12steps, critic740steps/10equivalent passes, critic-only phase7.47s,
  target MSE .12043→.04009, actor unchanged and no error sentinel. No inference
  about safe navigation efficacy from these startup groups; wait for user.

## 2026-09-06 scratch comparison prepared, user holds startup

Latest critic completion did not robustly improve navigation: two continuation
streams give opposite signs. Historical SAC R3 is shield-assisted, not directly
comparable to unshielded PPO. New protocol is
docs/RECOVERY_SAC_PPO_SCRATCH_PROTOCOL.md; runner
scripts/run_recovery_sac_ppo_scratch.py. No formal experiment launched.

User asks whether PPO slow updates were fixed: drafted native PPO collects
512steps×8workers rather than waiting for full cohorts, bootstraps nonterminal
rollout edges, max10epochs with KL stopping. Actual optimizer steps are logged.
This repairs update cadence, not proven learning efficacy or SAC-like replay.

User questions gamma1: finite4000-step tasks admit undiscounted returns, but
gamma1 need not be best for optimization. New paired runner gamma=.99 for BOTH
algorithms; old evidence files are not edited. Gamma and GAE lambda are distinct.

User explicitly pauses BEFORE formal startup. Keep the preparation, wait for
fresh authorization. Unit tests7pass/1bad start-position fixture; both native
SAC/PPO full CPU continuation tests pass. Ruff command missing. New runner
execution smoke/static checks are still pending. No background experiment exists.

## 2026-09-06 scratch comparison started after user release

- Fixed only the invalid test start point.8focused tests and py_compile pass;
  Ruff/Pyright unavailable. One real CUDA256-step two-arm smoke completed.
- Formal independent-session process Python PID4713, artifact
  artifacts/recovery_sac_ppo_scratch_20260906_v1. First ordinary nohup attempt
  was reaped before Python startup and left only a warning log; no state lost.
- First formal SAC checkpoint8192 committed. Warmup block4096 took2.03s; next
  block took109.83s with3192actor/critic updates, finite losses, GPU active.
  Replay3.294GB plus exact worker/RNG state331MB. Rough SAC training time near5h;
  evaluation/PPO add uncertainty. Do not watch completion.
- Pre-outcome interpretation and next-branch ledger:
  docs/RECOVERY_SAC_PPO_DECISION_LEDGER.md. PPO update cadence is repaired, but
  replay exposure and effective GAE horizon remain genuinely different. Current
  pair is a one-seed complete-configuration diagnostic, not causal attribution.

## 2026-09-06 restored immutable five-distance formal evaluation

User caught that prior formal500 was stratified:100tasks in each of100–500,
500–1500,1500–2500,2500–4000,>4000m. Verified the actual preserved JSON rather
than relying on memory: seed170001, exact counts100each, actual max distance
5012.325m, SHA-2566169ef6e...e00692d. The initial scratch runner's ordinary
random500 would not have been comparable.

User confirmed correction. SAC checkpoint-paused at520192 before any evaluation;
new runner preserves all training state and old audit files, binds the immutable
task hash, fixes world seed to170001+source index, and interleaves buckets only
for balanced resumable prefixes. Collision execution remains current locked
recovery, not the historical split/termination contract.10 focused tests pass.
Resume completed SAC524288 with519288 actor/critic updates and began stratified
evaluation; first16/500 rows persisted. Do not monitor further until user asks.

## 2026-09-06 independent-review reconciliation begins

- Scratch SAC/PPO is COMPLETE on the same immutable 500 tasks. SAC:496 goals,
  423 zero-contact goals,4 timeouts,path ratio1.063,mean unified contacts29.436.
  PPO:374/297/126/1.213/859.132. Both use one initialization, so this is not a
  multi-seed algorithm-ranking result.
- Read all four top-level independent-review records. Its central ordering
  claim survives: safe navigation is an experimental platform, not the paper's
  oral-level novelty, and an arbitrary 98%/zero-contact gate must not freeze
  resource-return research. Its old claim that the recovery reward provides
  essentially no avoidance signal is now too strong: scratch SAC learned 84.6%
  zero-contact arrival without a shield under the frozen reward/recovery rule.
- The old action-conditioned option route remains rejected by its own3410
  rollouts (return-now275/310; matched-action sensitivity p=.776; energy Kendall
  W=.017). The most direct next question is whether the new SAC can repair the
  return-policy bottleneck on those frozen anchors before state-level
  resource-to-go/first-passage work proceeds.
- Created a strict paired-analysis bundle for scratch SAC/PPO. Key paired task
  differences: arrival+24.4pp[20.8,28.0],safe arrival+25.2pp[21.0,29.6]. SAC's
  contact mean is tail-dominated: median0,p95~2,five tasks create90.0% of all
  contacts. PPO target-KL is not the simple cause: median10/10 epochs,92/128
  full blocks, value explained variance median.978. Actual actor sample
  presentations differ ~28.6x (SAC132.94M,PPO4.64M). Therefore no target-KL or
  critic-completion branch is authorized now.
- Reconciled the independent review in
  docs/INDEPENDENT_REVIEW_RECONCILIATION_20260906.md. Retain platform-vs-novelty,
  state V_R/defective first-passage and oral-evidence criticisms; revise its
  obsolete no-avoidance-signal/R3-only claims; reject another immediate PPO
  variant and an immediate full2x2 matrix.
- Implemented current-contract scratch-SAC anchor return evaluation: raw policy
  and standard HOCBF on the same310 preserved task anchors. Contact repairs and
  continues, zeros velocity, counts one unified contact/step, and never saves
  boundary/obstacle split metrics.7 focused tests pass, compile passes, and a
  two-anchor CUDA smoke completed both arms without contact.
- Formal resumable diagnostic launched at
  artifacts/sac_anchor_return_interface_310_20260906_v1, launcher82149/python
  82152. Startup health check found34 raw rows committed,eight active,no error.
  Do not monitor to completion; wait for user.

## 2026-09-06 executed-interface result and stochastic return-law phase

- The310x2 anchor diagnostic completed in678.26s with no error. Raw SAC:
  309arrival/254zero-contact/1timeout,mean energy18.833. HOCBF:
  292arrival/292zero-contact/18timeouts,mean energy22.583. Contact is nonterminal
  in both arms; no split contact statistics were saved.
- Strict inference clusters310 anchors within150 source scenes. HOCBF-minus-raw
  scene effects: safe return+13.17pp[8.67,18.00],arrival-3.50pp[-6.33,-1.00],
  energy+3.579[1.096,6.459],steps+145.7[59.9,241.4]. HOCBF's18timeouts have
  median remaining distance1381m and median intervention fraction1.0;13 contain
  repeated contacts. This is a safety-throughput-energy composition shift, not
  a goal-radius artifact. One policy seed/development anchors bound the claim.
- Froze the new object as a defective resource law Y=finite energy only on
  timely zero-contact charger arrival and +infinity otherwise. Identity
  P(Y<=b|x,I)=p_safe(x,I)F_finite(b|x,I,safe) is a modeling decomposition, not
  yet a novelty theorem. The research must later show an identifiability or
  decision-risk consequence and beat a properly calibrated mean/capped model.
- Added a physically declared post-controller AR(1) execution-error process:
  sigma.04 normalized action,rho.95,component clip.12,constant across four
  physics substeps. It is applied after raw/HOCBF, drives existing motion and
  energy integration, and is seeded by anchor+replicate identically across
  interfaces. It is controlled simulator stress, not real-UAV calibration.
- Ten focused tests and compile pass. A2-anchor×2-replicate CUDA smoke completed
  both arms; atomic rows contain only unified contacts and distinct physical
  disturbance seeds/RMS. Formal4,960-trajectory collection launched at
  artifacts/stochastic_defective_return_310x8_20260906_v1, launcher97592/python
  97595. First health check:8/2480 raw committed,eight active,no error. No
  automatic predictor fit; do not monitor until user asks.
