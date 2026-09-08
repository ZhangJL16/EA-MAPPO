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
  epistemic uncertainty. The R5 fixed-snapshot audit has zero variance across
  three exact Oracle returns; q90/q95 aleatoric claims remain gated on adding
  and auditing a physically motivated disturbance process.
- Headline utility: stranding--throughput Pareto frontier, not prediction MAE.
- Fixed empirical platform: user-selected R3 checkpoint, conditional estimand
  given frozen R3 plus HOCBF. Its historical navigation-quality Gate remains
  failed; navigation success, path ratio, and collision telemetry are descriptive
  strata rather than theorem assumptions or downstream launch thresholds.
- Stage C and SIRP are forbidden before the Oracle headroom Gate passes.

## Expert Task Ledger

| ID | Required work | Authoritative evidence | Status | Remaining proof |
| --- | --- | --- | --- | --- |
| 1 | Freeze commit, dependencies, and seed protocol | Formal pipeline manifest records evaluator/source SHA, fixed seed 170001, exact command, fixed task count, and clean evaluator worktree | `COMPLETE` for current Gate chain | A later Stage C benchmark needs its own dependency lockfile and immutable dataset manifest |
| 2 | Pluggable `ReturnManager` with exact legacy switching regression | `review_bundle/safety/switching/return_manager.py`; seeded legacy/refactored switching test | `COMPLETE` | None for current Quantile/SOC/distance managers |
| 3 | Correct mission q95 semantics | `MissionEnergyEstimate` distinguishes direct joint mission, component sum, and union-bound coverage; component-risk regression test | `COMPLETE` | No stochastic q95 headline until probability Gate changes |
| 4 | Simulator clone Oracle for return-now and task-then-return | `ModelBasedEnergyRolloutEstimator`; clone-state, obstacle/LiDAR/HOCBF, direct joint mission, and cache regressions | `COMPLETE` as implementation | Formal decision-value evidence remains pending |
| 5 | SOC, distance, Frozen-TD, Online-TD, Oracle reserve sweep | Stage-B Oracle runner and gated TD decision runner exist; frozen R3 identity, refined capacity, and disjoint-seed continuous-workload endurance are hash-bound | `R3 ENDURANCE GATE PASS / ORACLE PREFLIGHT PENDING` | Complete corrected viability-order Phase 6b.8, then run a fresh formal Oracle/SOC/distance headroom family before adding Frozen/Online TD |
| 6 | Stop ICLR return line if Oracle lacks material headroom | Preregistered Gate: Oracle must improve eligible heuristic throughput by at least 5% under common stranding ceiling 0.05. The default 12-point grid uses 320 independent cycles per point and 1% design stranding. Each required safe family has at least 95% certification power, giving at least 90% joint power without independence. The gain point estimate and paired full-family max-t 95% lower bound must both clear 5% for PASS; only a separately calibrated upper bound below 5% establishes insufficient headroom; crossing is inconclusive | `PENDING RESULT / UPSTREAM ENDURANCE PASS` | Corrected-rule three-seed preflight, followed by a formal R3 Oracle artifact with an explicit PASS, scientific FAIL, or INCONCLUSIVE decision |
| 7 | Port, rather than freely reimplement, Dopamine/PCM/OfflineRL-Kit components | Protocol names source projects and fair-information contract | `GATED, NOT STARTED` | Oracle Gate PASS and pinned upstream revisions/licenses |
| 8 | MC-direct, PCM-Executed, executed-WM, ensemble under shared data/protocol | Method contracts described only | `GATED, NOT STARTED` | Implementations, tests, common source dataset, held-out evaluation |
| 9 | Direct high-level continue/return switcher | Required in protocol, no implementation | `GATED, NOT STARTED` | Oracle Gate PASS, SB3 switcher, matched training privileges and seeds |
| 10 | Complete \(2\pi\times2\Pi\) leave-one-pair-out with interface-extrapolation × horizon matching | Design only | `PENDING` | Two policies, two genuinely different safety operators, matched strata, all held-out pairs |
| 11 | Add cross-fitted occupancy-weighted reliability only if ensemble is insufficient | Theory transport term and stop rule documented | `CONDITIONAL, NOT STARTED` | Evidence that executed-WM+ensemble fails to rank long-horizon ETG risk |
| 12 | CMDP track, second safety family/domain, full paper evidence | Separate modular/end-to-end comparison design and provisional proof package | `PENDING` | Pilot PASS, fair CMDP privileges, second domain/filter, independent theory review |
| 13 | Finite-state executed-interface risk-resolvent theory | Nine foundational statements plus a finite-interface primitive plug-in transfer bound and matching coverage--risk lower order | `DRAFT PROOFS + NUMERICALLY VERIFIED` | Continuous/neural interfaces, misspecification, arbitrary replay dependence, and general-state or explicitly finite-state paper scope |
| 14 | Theorem-level closest-work audit | 35-source base audit; a refreshed 17-paper continuous-interface audit including 2025 KROPE/risk-total-reward collisions; a DR/risk-OPE audit; and a 17-paper mathematical audit covering Markov-chain conditioning, killed/QSD perturbation and CLTs, Feynman--Kac limits, SSP hardness, margin theory, adaptive Neyman allocation, and trajectory stratification | `COMPLETE FOR CURRENT STATEMENTS AND NEXT TARGET` | Refresh before submission; no individual generic regression, representation-stability, spectral, or margin statement may be sold as novel |
| 15 | Irreversible first-disagreement and Pareto stability | Oracle-shadow stopped-occupation theorem, bounded-metric coordinate rectangle, and population Oracle-headroom preservation condition | `PROVED + FINITE-LAW VERIFIED` | Formal paired mission trajectories, Wilson/statistical uncertainty, and non-vacuous estimator error rates |
| 16 | Cross-fitted EIRR certificate to conservative exact-risk headroom | Simultaneous finite-query certificate, positive MGF interval propagation through EVaR, task--then--return commitment boundary, separate direct-return certificate, one-sided first-disagreement rate, and Pareto preservation condition | `PROVED AFTER NON-NESTED-ACTION REFRAME + CORRECTED FINITE-LAW VERIFIED` | Use the tabular plug-in radius or derive a stricter learned-Doob/continuous-interface radius; establish exact-risk comparator headroom under the corrected formal Gate chain |
| 17 | Finite-interface source-to-target transfer and Pareto lower rate | Empirical-Bernstein primitive-cell radii, queryable target composition, robust resolvent certificate, matching \(\Omega(e^{\lambda L}/\mu)\) estimation order, and \(3/8\) irreversible Pareto-decision error floor | `PROVED + FINITE-LAW VERIFIED` | Empirical mechanism evidence; continuous/neural extension; prove any claimed advantage over primitive plug-in and DICE baselines |
| 18 | R3 horizon-matched executed-interface mechanism audit | Five-fold contiguous/interleaved cross-fitting, strong post-trajectory controls, paired moving-block bootstrap, coarsened matching, approximate conditional interface permutation, and residual exponential-tail stratification | `COMPLETE EXPLORATORY / FORMAL GATE FALSE` | Repeat on the predeclared \(2\pi\times2\Pi\) held-out compositions with deployable pre-decision features; current summaries use future trajectory information |
| 19 | Continuous risk-observable quotient plus stopped-boundary functional | Restricted-CME/IPM equivalence, exact quotient factorization, encoded residual resolvent, log-risk/grid propagation, fixed-linear minimax theorem, and Theorem 27's continuous local-mass Hölder query rate with stopped transfer | `DECLARED CONTINUOUS HÖLDER SLICE PROVED + VERIFIED / UNKNOWN-CHART NEURAL RATE OPEN` | Extend to learned intrinsic charts and dependent witnesses; stop the oral claim if the joint phase reduces exactly to restricted OPE/KROPE on the Doob chain |
| 20 | Two-layer orthogonal EIRR score | Theorem 17 separates queryable target-execution and primitive killed-risk residuals; proves exact block double robustness, an exact two-product bias identity, and its L2 bound; finite population verifier exercises both robust blocks and joint misspecification | `POPULATION IDENTITY PROVED + FINITE VERIFIED / LEARNED RATE OPEN` | Audit closest DRL/marginalized-ratio theorems; learn four nuisances under positivity and dependent sampling; prove a structure-dependent rate/lower bound or retain this only as an estimator-design lemma |
| 21 | Stopped-EIRR canonical gradient and information limit | Theorem 18 differentiates the killed Feynman--Kac fixed point in the conditional primitive law, identifies the canonical gradient and efficiency variance, and gives a least-favourable local irreversible-decision floor; finite perturbation verifier checks the derivative and information norm | `IID SEMIPARAMETRIC SLICE PROVED + FINITE VERIFIED / PHASE TRANSITION OPEN` | Determine whether quotient support, approach to transience, and the stopped margin yield a genuinely new sharp phase transition; otherwise classify Theorem 18 as transformed-MDP standard EIF theory |
| 22 | Exact quotient--transience information phase transition | Theorem 19 proves in a one-state stopped-risk slice that raw-MGF information scales as \(n\mu_h\Delta^4\), log-risk/requirement information as \(n\mu_h\Delta^2\), and local irreversible-decision resolution as \((n\mu_h\Delta^2)^{-1/2}\); verifier checks Fisher/delta identities and exponents | `ONE-STATE PHASE TRANSITION PROVED + FINITE VERIFIED / GENERALIZED BY THEOREM 20` | No remaining scalar proof; multi-state scope and novelty boundary are tracked in row 23 |
| 23 | Multi-state Perron-mode information phase split | Theorem 20 proves the finite critical-family raw/log efficiency exponents, identifies the exact quotient-coverage/right-mode noise coefficient, and separates a projected-noise-degenerate phase; a non-symmetric two-state verifier checks both regimes | `FINITE CRITICAL FAMILY PROVED + VERIFIED / UNIFORM LAN CONDITIONAL` | The focused mathematical audit shows the generic spectral ingredients are occupied; any joint \(\Delta_n,n\) decision floor requires explicit uniform triangular-array LAN, and learned quotient/dependent replay remain open |
| 24 | Cost-aware critical-mode information design | Theorem 21 solves the exact cost-constrained stratum allocation, gives its Perron critical limit, and proves a finite pilot relative-efficiency bound; verifier checks the critical constant, oracle optimum, and plug-in bound | `EXACT DESIGN THEOREM PROVED + VERIFIED / KNOWN-QUOTIENT ADAPTATION IN THEOREM 22` | Generic oracle and adaptive Neyman algebra is closed; learned quotient and collapsing-phase issues are tracked in row 25 |
| 25 | Singularity-free adaptive critical design | Theorem 22 proves a high-probability pilot priority rate and adaptive oracle inequality under known quotient, bounded primitives, bounded Perron reduced resolvent, and projected-variance floors; its key separation permits \(m\Delta^2\to0\) while allocation efficiency tends to one | `KNOWN-QUOTIENT ADAPTIVE ORACLE PROVED + VERIFIED / FAILURE PHASES OPEN` | Generic adaptive Neyman allocation is prior art; prove sharp rates or impossibility when effective Perron separation, projected noise, or quotient separation collapses, and then connect the gain to stopped loss |
| 26 | Conditioned collapse phase and stopped-Pareto transfer | Theorem 23 gives the joint sufficient phase and a locally matching effective-separation/anisotropy lower slice; Theorem 24 transfers canonical variance and the adaptive-oracle ratio through the stopped margin, with an exact Gaussian plug-in witness for the \(V^{\kappa/2}\) and \(V^{(\kappa+1)/2}\) powers | `JOINT PHASE + STOPPED TRANSFER PROVED AND VERIFIED / LEARNED QUOTIENT + SHARED MULTI-QUERY DESIGN OPEN` | Classical perturbation, adaptive stratification, variance concentration, and generic margin conversion are not novelty; next resolve learned quotient rates and whether one shared replay design has a new multi-query optimum |
| 27 | Shared stopped-margin Perron design | Theorem 25 proves global convexity for every margin power \(p>0\), derives the exact shared-allocation fixed point and \(p=1\) closed form, and separates common-critical cancellation from query-specific critical emphasis | `SHARED ORACLE DESIGN PROVED + VERIFIED / ADAPTIVE SENSITIVITY-MATRIX LEARNING OPEN` | Multi-characteristic stratified allocation, compound optimal design, multiple-logger OPE, and behavior-policy search are prior art; novelty requires a sharp adaptive phase or lower bound while learning stopped occupation and Perron sensitivities |
| 28 | Finite learned risk quotient | Theorem 26 recovers the exact finite quotient from stacked first/second risk-witness moments at order \(B_W^2\gamma^{-2}\log(ND/\delta)\), proves the matching \(m\gamma^2=O(1)\) error floor, and specifies the neural moment-reconstruction target | `FINITE RATE + LOWER BOUND PROVED AND VERIFIED / CONTINUOUS SLICE IN THEOREM 27` | Concentration, separated clustering, and Le Cam testing are classical; unknown-chart neural recovery remains open |
| 29 | New fixed-navigation platform audit | Deterministic go-to-goal runner preserves the frozen R5 environment/HOCBF contract without training; 10-task execution smoke completed with 9/10 success, path ratio 0.996465, and zero collision/boundary steps | `EXECUTION PATH VALID / NOT PROMOTED TO FORMAL GATE` | Smoke success 0.90 is below the unchanged 0.98 formal threshold; no costly 500-task attempt is authorized under the no-endless-repair rule |
| 30 | Continuous quotient--spectral--stopping phase | Theorem 27 proves the \(\widetilde O(n^{-\alpha/(2\alpha+d_Q)})\) point-query radius, \(2\varepsilon_n\) metric sandwich, and joint \((d_Q,\mathfrak g,\Delta,\kappa)\) phase. Its lower claim now uses a distributed strong-density Hölder--margin Assouad hypercube mapped through the exact killed first-passage inverse, rather than incorrectly transferring one point-query Le Cam bump directly to a random stopped law | `IID LOCAL-MASS UPPER + EXACT-CHART SOURCE=TARGET ASSOUAD PARETO LOWER PHASE PROVED/VERIFIED` | The common lower slice closes \((d_Q,\Delta,\kappa)\) and Pareto orders but not arbitrary pushed overlap, learned charts, nuisance estimation, or the separate \(\mathfrak g\) priority family. Generic ingredients are occupied and KROPE closes the stability headline |
| 31 | Risk-neutral KROPE failure and oracle-Doob baseline boundary | Theorem 28 gives an equal-mean/unequal-exponential-risk interface pair, a raw-versus-quotient coverage gap, and the exact oracle transformed-chain equivalence | `EXACT SEPARATION/EQUIVALENCE BOUNDARY PROVED + VERIFIED` | Risk-neutral KROPE and raw DICE are insufficient, but correctly risk-transformed KROPE/DICE remain mandatory; novelty must come from learning the unknown transform and stopped certificate |
| 32 | Unknown-Doob reparameterization audit | Theorem 29 proves the Doob map is invertible, direct and transformed plug-in values coincide exactly, minimax risk is coordinate-invariant, and relative transform conditioning retains the \(1/\Delta\) critical scale | `EXACT NEGATIVE THEOREM + VERIFIED` | Learned-Doob alone is not a rate contribution; only quotient complexity reduction, decision localization, or algorithm-specific computational gains remain viable |
| 33 | Predictable-interface martingale quotient | Theorem 30 replaces iid transitions by first-\(m\)-hit martingale averages under adaptive policy/filter execution, preserves the continuous joint phase, and proves replay duplicates cannot increase certificate counts | `DEPENDENT CHRONOLOGICAL SLICE PROVED + VERIFIED` | Query eligibility must be predictable and outcomes fresh; retrospective outcome-selected representations need post-selection control, and replay affects optimization only |
| 34 | Sharp stopped-occupation \(L_p\) localization | Theorem 31 converts integrated score error into first-disagreement/Pareto power \(\kappa/(p+\kappa)\) and boundary-loss power \((\kappa+1)/(p+\kappa)\), with an explicit construction attaining both powers | `DECISION-LOCALIZATION THEOREM PROVED + VERIFIED` | A cross-fitted learned estimator must still attain lower stopped-law \(L_p\) risk than matched Doob/DICE/KROPE; generic margin conversion is not standalone Oral novelty |
| 35 | Cross-fitted stopped-quotient regression | Theorem 32 bounds target stopped \(L_2\) risk by chart distortion, pseudooutcome nuisance, intrinsic Hölder error, and the pushed-forward overlap coefficient \(\sum_jq_j/p_j\); it propagates the rate through transience and Theorem 31 and proves an exact raw nuisance factor \(K\) | `CERTIFIED-CHART ESTIMATOR PROVED + VERIFIED` | Both laws must be pushed through the chart before ratios are formed; learned-chart distortion remains an assumed certificate, and matched transformed OPE can inherit the same quotient gain |
| 36 | Finite-library chart learning and matched-information equivalence | Theorem 33 uniformly certifies all candidate partition diameters, proves the selector oracle inequality and \(m_{\min}\Delta^2/\log(ND)\) decision phase, then proves direct and unrestricted transformed manager-decision classes are identical under matched information | `FINITE CHART RATE PROVED + VERIFIED / UNRESTRICTED SEPARATION CLOSED` | Arbitrary continuous neural-chart generalization remains open; any strict method comparison must declare computational, hypothesis-class, regularization, or side-information asymmetry |
| 37 | R3 stopped-quotient certificate proxy | A three-fold trajectory-level audit fits bounded risk witnesses on 310 structure trajectories, estimates ordinary source masses on 310 disjoint trajectories, and evaluates terminal exponential-Energy tilts on 310 disjoint target trajectories. At beta 1 the target ESS fraction is 0.01074; the four-atom interface family has radius 0.50338 (`WIDE_EXPLORATORY`), while the eight-atom goal-sensitive family has minimum cell count 3 and radius 2.18397 (`VACUOUS_AT_MAX_DIAMETER`) | `EXPLORATORY DIAGNOSTIC COMPLETE / FORMAL GATES FALSE` | The target law is a terminal-risk tilt, not Oracle-shadow stopped occupation; nuisance, transience, stopped-margin, first-disagreement, stranding, and throughput evidence are absent. Do not select or promote a chart from the partial objective |
| 38 | Paired Oracle-shadow decision and cycle evidence path | Stage-B now generates method-invariant tasks keyed by evaluation seed, cycle, and task index; evaluates the exact Oracle branches through and including the stopped first disagreement; records its direction; joins each cycle to the complete Oracle run with the same keyed schedule and reserve; and validates a versioned fail-closed evidence bundle | `IMPLEMENTED + END-TO-END SMOKE VERIFIED / ENDURANCE PASS / FORMAL RESULT PENDING` | The smoke produced 369 decision events, five cycles, three first disagreements, and complete Oracle cycle joins, while retaining `SMOKE_ONLY_NOT_FORMAL_EVIDENCE` and `formal_eligible=false`. Formal evidence still requires corrected viability-order preflight and 320 independent cycles per point for the default grid |
| 39 | Selection-aware paired Oracle-headroom inference | A paired schedule bootstrap resamples independent cycle IDs jointly and builds directional studentized maximum-deviation rate bands across every preregistered Oracle/SOC/distance point. After the random safety subset is formed, lower and upper selected-frontier gain bounds follow from the corresponding rate extrema. Formal CLI defaults fix confidence 0.95, 10,000 replicates, and seed 20260830 | `IMPLEMENTED + SYNTHETICALLY VERIFIED / FORMAL RESULT PENDING` | The old selected-max percentile interval is descriptive only. The bootstrap bands are asymptotic, whereas stranding coverage is finite-sample exact. Actual bounds remain absent until the formal 320-cycle run |
| 40 | Oracle-shadow pathwise coupling audit | For every method/schedule/reserve group, the audit joins the independently executed Oracle event at the same step, information time, and task index; verifies position, velocity, task goal, remaining Energy, exact requirement, and Oracle decision through and including the stopped first disagreement; and deliberately ignores post-disagreement paths | `IMPLEMENTED + END-TO-END SMOKE VERIFIED / FORMAL RESULT PENDING` | The keyed smoke audited three method/schedule pairs and 214 pre-disagreement events, found three first disagreements, and obtained exactly zero position and requirement error. It remained non-formal. Missing/mismatched coupling now forces `FAIL_ORACLE_SHADOW_EVIDENCE_INTEGRITY`, and legacy PASS artifacts without this audit cannot unlock downstream stages |
| 41 | Selection-valid simultaneous stranding safety | A one-sided exact Clopper--Pearson upper bound is computed for every preregistered Oracle/SOC/distance point at Bonferroni level \(0.05/K\), then the safe frontier is selected. For \(K=12\), 107 cycles are only the zero-event minimum; the 320-cycle design allows six failures | `IMPLEMENTED + SYNTHETICALLY VERIFIED / FORMAL RESULT PENDING` | Pointwise Wilson intervals remain descriptive. Grid expansion increases \(K\) automatically and triggers a new power audit. Formal values remain absent until the upstream Gate chain passes |
| 42 | Safety-set-valid simultaneous throughput headroom | A paired studentized max-t band is constructed over the complete candidate family before the safe subset and fastest points are selected. On the joint rate-band event, the selected Oracle/heuristic extreme-rate ratio is bounded below by the ratio of selected lower/upper band extrema | `IMPLEMENTED + SYNTHETICALLY VERIFIED / FORMAL RESULT PENDING` | A 60-schedule synthetic selection example reduced the old percentile lower bound from 0.02687 to the valid full-family bound 0.01176. Formal R3 evidence awaits endurance validation and the paired run; bootstrap coverage is asymptotic and must be reported as such |
| 43 | Exact safety-certification power audit | The Bonferroni CP rejection event is inverted through \(P_{p=0.05}(X\le x)\le0.05/K\). At \(K=12\), 110 cycles certify only zero failures and have 0.331 per-family power at a 1% true rate. The 320-cycle design certifies up to six failures, has 0.956 per-family power, and has a 0.912 joint lower bound for one safe Oracle plus one safe heuristic family. Exact minima are 314 cycles at 1% and 694 at 2% for the required 95% per-family power | `IMPLEMENTED + EXACTLY VERIFIED / FORMAL RESULT PENDING` | Formal CLI rejects any initial Oracle design below its preregistered 90% joint lower bound. The 1% design point follows the original target; sensitivity at 2% is reported rather than silently claimed powered |
| 44 | Three-way Oracle-headroom identification Gate | Full-family directional max-t rate bands now provide both a lower and an upper selected-frontier gain bound. PASS requires the point estimate and lower bound to clear 5%; a scientific insufficient-headroom result requires the upper bound below 5%; any crossing is explicitly inconclusive | `IMPLEMENTED + SYNTHETICALLY VERIFIED / FORMAL RESULT PENDING` | Separate one-sided 95% tests control the directional claims. A missing/nonpositive heuristic lower denominator cannot produce a scientific FAIL. Machine evidence is v6 with tagged finite-deadline completion requirements; legacy PASS artifacts without both ordered finite bounds or deadline tags cannot unlock Stage C |
| 45 | Scientific prerequisite-stop contract | Stage-B distinguishes a complete but nonpassing contract/calibration/validation audit from missing or malformed inputs. The former writes `STOPPED_PREREQUISITES_NOT_READY.json`, exits 4, records failed integrity/endurance conditions, and forbids downstream execution without emitting `FAILED.json` | `IMPLEMENTED + INTEGRATION VERIFIED + FIXED-BASELINE MODE ADDED` | Legacy R1--R5 deployment-quality verdicts remain unchanged; the new R3 mode authorizes only the conditional energy-research estimand and still fails closed on identity, taxonomy, calibration, or endurance errors |
| 46 | Frozen-policy prerequisite identity chain | The prerequisite adapter consumes legacy completion wrappers or the fixed-baseline research contract, verifies the checkpoint SHA against the policy loaded by Stage B, and chains contract, calibration, validation, and artifact-config hashes. Raw flat metrics remain diagnostic-only | `IMPLEMENTED + R3 CONTRACT/ATTESTATION VERIFIED` | R3's historical navigation Gate remains false; its separately hashed research contract records the user selection and performance-as-descriptive scope |
| 47 | Executed closed-loop environment identity | Formal Stage-B reconstructs environment kwargs from the named navigation artifact rather than independent CLI defaults, then records the exact base contract and the limited Energy-managed overrides. R5 orchestration-only flags are normalized while sampled-data robust HOCBF is restored explicitly | `IMPLEMENTED + ACTUAL R5 CHECKPOINT LOAD VERIFIED / FORMAL GATE STILL FAILED` | Read-only R5 verification produced observation 2055, action 3, LiDAR 8x128, 24 obstacles, HOCBF/sample-data robustness/projection all enabled. This proves compatibility, not navigation PASS or Oracle headroom |
| 48 | Frozen Stage-B workload and bounded Oracle compute | The launch manifest freezes the 12 candidate points, 320-seed hash, 3840 cycle jobs, exact safety power, and pairing contract. A persistent six-process pool cuts maximum policy loads from 72 to 6. Non-Oracle shadow evaluation stops only after the audited first disagreement, while the paired Oracle cycle remains complete | `IMPLEMENTED + END-TO-END SMOKE/REGRESSION VERIFIED / FORMAL RESULT PENDING` | The smoke skipped 15 post-disagreement shadow evaluations while the stopped coupling audit passed all 214 compared events with zero error. Formal R3 wall time remains unmeasured. Throughput uses a valid three-way max-t Gate; 320 cycles are not advertised as guaranteed 5% effect power without a paired-rate variance assumption |
| 49 | R5 platform probability-semantics P0 | A standalone read-only runner loads the wrapper-matched R5 500k checkpoint, reconstructs the 24-obstacle sampled-data-HOCBF platform, hashes a fixed snapshot, and repeats the exact return Oracle without pretending repetition indices are disturbance seeds | `DETERMINISTIC POINT ETG CONFIRMED / DOWNSTREAM STILL STOPPED` | Three costs were all `14.322758552613978` with variance `0`; the artifact forbids aleatoric q90/q95 and retains navigation/downstream authorization false. A distributional target requires a physically specified and audited future disturbance law |
| 50 | Contracted stopped-Pareto dual certificate | Step 35 composes Theorems 31--33 into an exact finite-sample Pareto rectangle and separates two noninterchangeable regimes: pushed-overlap stopped-\(L_2\) localization for weak/average coverage and a faster target-active uniform confidence band under strong per-cell coverage. Exact constants, margin-radius clipping, probability clipping, both rate exponents, and route selection are verified in `artifacts/stopped_pareto_contraction_20260830/` | `DUAL UPPER CERTIFICATE VERIFIED / STRONG-COVERAGE LOWER PHASE REPAIRED / GENERAL MINIMAX WITHHELD` | The \(L_2\)+Markov chain is slower than the classical Hölder plug-in boundary rate under strong density. Theorem 27 now replaces its unsupported point-query-to-margin jump with one distributed Assouad killed-first-passage family attaining the uniform-route \((d_Q,\Delta,\kappa)\) and Pareto lower orders. Arbitrary pushed overlap, learned charts, and nuisance-bearing source--target shift remain without a matching lower bound; no new theorem number or downstream experiment is authorized |
| 51 | R3 conditional energy-research platform contract | `navigation_platform_contract.json` preserves `legacy_navigation_gate_passed=false`; refined capacity `304.9538842289515` is confirmed by 100 disjoint seeds under continuous task workload | `CONTRACT + CALIBRATION + ENDURANCE GATE COMPLETE` | Attempt 3 has 100/100 true depletion, zero censoring, mean `1668.3655 s`, relative error `-7.313%`, and a hash-bound `COMPLETED.json`; no remaining proof before Oracle preflight |

## Current Formal Gate Chain

```text
user-selected frozen R3/HOCBF platform contract
    -> attested 500-task battery calibration
    -> 100-run battery validation
    -> 320-independent-cycle Oracle headroom comparison for the default grid
    -> exact 500k Quantile-TD collection
    -> Frozen/Online TD decision comparison
```

Every arrow is fail-closed.  The downstream process must consume a named passing
artifact; it must not infer success from process exit alone or recompute a missing
Gate from an incomplete table.

## Current Evidence Snapshot

- The archived flat-MLP 500k JSEB checkpoint failed the fixed 500-task
  navigation Gate: overall success `0.84`, minimum bucket success `0.82`, and
  mean path ratio `1.63125`; it nevertheless had zero obstacle-collision steps.
- Repair R1 (structured LiDAR, bridge off) completed 500k transitions and the
  same 500-task evaluation.  It improved path ratio to `1.05972` but reached
  only `0.90` success and accumulated `3151` obstacle-collision steps, so it
  failed navigation and safety readiness.
- Repair R2 (structured LiDAR plus the original uniform-replay bridge) also
  completed the fixed budget.  It reached `0.888` success, path ratio
  `1.10867`, and `17984` obstacle-collision steps, so it failed every downstream
  authorization condition.
- R3 and R4 both completed their exact 500k budgets and fixed 500-task
  evaluations. R3 reached `0.96` overall success, bucket minimum `0.92`, path
  ratio `1.18803`, and `2242` obstacle-collision steps. R4 reached `0.904`
  overall success, bucket minimum `0.82`, path ratio `1.35285`, and `11373`
  collision steps. Both wrote authoritative `STOPPED_NAVIGATION_NOT_READY.json`
  artifacts with downstream authorization false.
- R5 also completed its exact 500k budget and fixed evaluation. It reached
  `0.938` overall success, bucket minimum `0.86`, path ratio `1.09846`, and
  `2497` obstacle-collision steps, and likewise stopped fail-closed. In accordance
  with the no-endless-repair rule, no R6 navigation refinement is authorized by
  this ledger.
- The user selected R3 as the fixed baseline for conditional Energy/return
  research. This does not rewrite its old navigation-quality FAIL. The existing
  500-task calibration contains 475 successful trajectories and 25 retained
  `task_step_limit` outcomes, uses frozen SAC and no TD training, and gives
  capacity `383.35430890654663`. The new attestation recomputes its mean power and
  capacity from all raw task records and binds the R3 checkpoint, source wrapper,
  artifact config, platform contract, and calibration with SHA-256. A resumable
  replacement validation completed all ten batches and 100 runs without worker
  failure. It recorded 59 `energy_exhausted` and 41 right-censored
  `task_step_limit` outcomes. Mean observed terminal time was `1657.9 s`
  (`-7.894%` from 30 minutes), but treating censored times as depletion is biased;
  the strict endurance Gate therefore stopped. This is a validation-continuation
  problem, not permission to retune R3 navigation.
- The exploratory risk-tilted audit shows strong concentration: at normalized
  `beta=1`, terminal ESS is `0.0128` of 930 paths and the top 5% hold `0.5202` of
  terminal tilted mass; risk-occupation ESS is `0.0120` of 370,572 visits and the
  top 5% of paths hold `0.5601` of visit mass. Energy--horizon correlation is
  `0.9802` and a linear horizon model explains `0.9609` of total-energy variance.
  After linear horizon adjustment, Energy--intervention partial correlation is
  `-0.1412`, while Energy-rate--intervention correlation is only `0.0034`.
  Horizon matching is therefore mandatory, and the current evidence does not
  support an unsigned-intervention-causes-energy claim.
- The completed horizon-matched R3 audit uses five contiguous trajectory blocks
  as its primary held-out scheme. Relative to a strong post-trajectory control
  containing duration, realized path geometry, goal/distance strata, and nominal
  action effort, executed-interface summaries reduce total-Energy MAE by `3.9041%`;
  the paired absolute MAE improvement is `0.028031` with moving-block-bootstrap
  95% interval `[0.006198, 0.051350]`, and the approximate conditional-permutation
  upper-tail probability is `1/201`. This is a within-composition association,
  not a deployable predictor or causal effect. Energy-rate MAE is essentially
  unchanged and subgroup effects have mixed signs.
- The mechanism is tail/heteroskedastic rather than a positive mean penalty. The
  highest intervention quintile has strong-model residual mean `-0.1108` but
  residual standard deviation `1.3199`, q95 `2.0459`, and normalized beta-one
  log-MGF `0.7384`; the lowest quintile has mean `0.1744`, standard deviation
  `0.7137`, q95 `1.1372`, and log-MGF `0.4270`. Beta-one residual-risk tilting
  enriches mean intervention by `1.3538x`. Thus the R3 evidence supports studying
  conditional exponential residual risk, not adding a signed intervention-count
  penalty.
- The original censored validation is retained as invalid evidence. The refined
  continuous-workload confirmation now has 100/100 true depletion endpoints,
  zero censoring, and mean endurance `1668.3655 s` (`-7.313%`), so its Gate
  passes. Formal Oracle Headroom remains absent: two earlier formal attempts are
  invalid implementation/termination artifacts, and the corrected viability-
  order three-seed preflight is not yet complete. TD decision outputs remain
  fail-closed behind an Oracle Headroom PASS.
- Proof package: five corrected base theorems/propositions, two mission
  corollaries, nine finite-state risk-resolvent statements, and the sequential
  irreversible-decision bridge are drafted. The
  finite-state verifier reports spectral radius `0.3`, perturbation-identity
  error `3.43e-16`, residual-identity error `0`, and status
  `FINITE_STATE_IDENTITIES_VERIFIED`. Its Doob reduction has row-sum error `0`,
  matrix reconstruction error `5.55e-17`, and resolvent reconstruction error
  `2.22e-16`; this verifies explicit examples, not
  neural consistency or oral-level novelty. Independent semantic proof
  acceptance is not available.
- The irreversible-boundary verifier uses four explicit coupled path atoms. Its
  exact first-disagreement probability equals the stopped boundary-occupation
  bound (`0.6`); actual stranding-coordinate deviation `0.4` is below `0.6`, and
  actual throughput deviation `5.0` is below `6.0`. This is a tight algebraic
  example, not formal Oracle or Pareto evidence.
- The corrected end-to-end EIRR verifier propagates simultaneous positive MGF
  intervals through finite-grid EVaR, selects the task--then--return branch as
  the commitment boundary, and keeps direct return as a separate certificate.
  Its effective requirement gap exactly equals its interval bound
  (`0.5108256`), and its one-sided first-disagreement probability exactly
  equals its stopped-boundary bound (`0.4`). A regression with an arbitrarily
  larger direct-return requirement confirms that it cannot override the
  certified mission stopping boundary.
- The finite-interface transfer verifier composes empirical-Bernstein cell
  radii into an infinity-norm resolvent certificate. Its exact value error is
  `0.0005715`, below the computable empirical-Bernstein bound `0.0110030`; its hard family has
  \(e^{\lambda L}=20\), source coverage \(\mu=0.1\), and minimax error floor
  `0.35625` plus decision-error floor `0.375` through `50` source transitions. This verifies the finite algebra
  and matching order, not neural or continuous-interface consistency.
- The risk-observable quotient verifier separates the one-step conditional-law
  IPM from the nonlinear stopped-boundary functional. Its exact zero-distance
  quotient preserves the finite MGF value; in the approximate slice, interface
  diameter `0.07` produces maximum value error `0.01822`, below the positive-
  resolvent uniform bound `0.11376`, and requirement error `0.02261`, below the
  pointwise bound `0.02279`. This verifies the algebra and a constructed boundary
  disagreement, not a learning rate or representation novelty.
- The fixed-linear quotient theorem proves that a target risk functional is
  identifiable exactly when its risk-occupation feature mean lies in the source
  design range. Its verifier gives a singular raw-support example with quotient
  coefficient `1`, exact minimax MSE `0.008` at `n=20`, a collapsed-feature
  indistinguishable lower bound `4.0`, and a least-favourable irreversible
  decision error `Phi(-1/2)=0.30854`. The result assumes fixed nuisances and
  Gaussian linear witnesses; it is not a neural/dependent-trajectory theorem.
- The two-layer orthogonal EIRR verifier has killed-operator spectral radius
  `0.20714`. Both the exact-ratio/arbitrary-model score and the
  exact-model/arbitrary-ratio score equal target `1.0369960`. Under joint
  misspecification, actual bias `0.04955025` exactly equals the two product terms
  and is below the L2 bound `0.09152815`. This verifies a population identity,
  not learned nuisance rates, dependent replay inference, or oral-level novelty.
- The canonical-gradient verifier has spectral radius `0.59236`. Its analytic
  pathwise derivative `-0.0105134550` matches a central conditional-law
  perturbation to absolute error `2.66e-11`. The squared gradient norm and
  conditional-variance efficiency formula both equal `0.0481111947`; the
  least-favourable derivative matches its square root to `6.75e-11`, and the
  marginal-design tangent inner product is `6.25e-17`. This validates a finite
  example, not the general LAN regularity or a new phase transition.
- The critical-phase verifier fixes quotient coverage `0.2` and checks gaps
  `0.2, 0.1, 0.05, 0.025`. After removing the bounded Bernoulli factors, halving
  the gap multiplies raw-MGF efficient variance by exactly `16` and log-risk
  variance by exactly `4`; fitted log--log slopes are `-4` and `-2`. At
  `n=50,000`, the local requirement scale grows from `0.07469` to `0.84382` as
  the gap shrinks. This is an exact scalar slice, not yet a multi-state spectral
  theorem.
- The Perron-phase verifier uses a non-symmetric two-state positive-operator
  family. In the nondegenerate regime, raw/log tail slopes are `-4.044/-2.039`
  and limiting-constant relative errors are `1.30%/1.15%`; exact projected-noise
  degeneracy changes the slopes to `-2.023/-0.017`. This verifies Theorem 20's
  finite construction, not uniform triangular-array LAN or learned replay.
- The critical-design verifier reports `1.18%` relative error for the predicted
  cost-aware Perron limit. None of 2,000 random feasible allocations beats the
  oracle. With at most `10%` pilot sensitivity error, the second-stage variance
  ratio is `1.0068` against analytic upper bound `1.2222`; a `5%` pilot budget
  gives full-budget ratio `1.0598`. This does not prove the required pilot
  estimation rate.
- The adaptive critical-design verifier uses
  \(m_\Delta=\lceil20/\Delta\rceil\) pilot draws per interface. While
  \(m_\Delta\Delta^2\) falls from `1.0` to `0.0625`, the median oracle variance
  ratio improves from `1.00228` to `1.00009`; every q90 ratio is below `1.011`.
  This verifies the finite scale-separation construction under a known quotient,
  not the collapsing-eigengap or learned-quotient cases.
- The collapsing-design verifier checks the exact local derivative
  \((1+\chi)/\mathfrak g\) to normalized error below \(10^{-6}\). At
  \(m=(1+\chi)^2/\mathfrak g^2\), bounded-pilot KL stays near `0.02` while
  log-priority separation stays near `0.20`. Its Gaussian/Bernoulli comparison
  verifies that vanishing projected variance has no distribution-free exponent.
- The stopped-Pareto verifier checks margin exponents
  \(\kappa\in\{0.5,1,2\}\). Gaussian plug-in disagreement has power \(\kappa\),
  boundary-weighted loss has power \(\kappa+1\), and the exact leading constants
  match to relative error below `2e-4`. This is a mathematical slice, not mission
  Pareto evidence.
- The shared-margin verifier checks convex powers down to \(p=0.4\), matches the
  KKT fixed point to maximum relative error `1.63e-7`, and finds no better point
  in 1,000 random feasible allocations per power. Scaling all sensitivities by a
  common critical gap changes normalized allocation by at most `5.78e-8`, while
  query-specific gaps `0.05/0.20` receive allocation `80/20` in the diagonal
  \(p=1\) slice.
- The finite-quotient verifier recovers all three quotient classes in 300/300
  seeded bounded-interface trials above the Hoeffding separation threshold. At
  \(m\gamma^2\approx0.05\), exact Bernoulli KL stays below `0.11` and the Pinsker--
  Le Cam minimax error lower bound remains above `0.38`.
- Decision-interval telemetry: implementation and related tests pass; formal
  overshoot data are pending a later decision run from a commit containing that
  telemetry. On 2026-08-30, the current uv environment passed the combined
  ReturnManager/Oracle/Gate, risk-tilt, quotient, orthogonal-score, canonical-
  gradient, critical-phase, and finite-state theorem verifier selection (`80
  passed`, 19
  deprecation warnings). Adding the historical Phase-2 final-runner file produced
  `80 passed / 2 failed`; both failures are artifact-dependent and caused by the
  absent historical `uav_energy_risk_v5_20260818_0330/COMPLETED.json`, not a
  ReturnManager or theory failure. That completion marker was not fabricated.
  After adding Theorems 27--33 and the fixed-platform audit, the current clean
  theory/ReturnManager/navigation-audit subset is `102 passed, 19 warnings` under
  the uv environment. All 19 theory JSON artifacts parse, the fourteen
  critical-design/Perron/quotient/martingale verifier scripts compile, and
  `git diff --check` is clean.
- The R3 stopped-quotient proxy uses complete trajectories as the statistical
  unit and three disjoint contiguous folds. At beta 1, its target-fold ESS
  fraction is `0.0107381`. The interface-only family is wide (uniform witness
  radius `0.5033795`); adding goal type creates cells as small as three and a
  vacuous radius `2.1839743`. Raw pushed overlap is `3.13898` and horizon-
  coarsened overlap is `1.90494`, but the partial Theorem-32 proxy omits nuisance,
  transience, and actual stopped-occupation terms, so its numerical minimum is
  not a method-selection result. The artifact explicitly leaves Oracle Decision
  Headroom and the Pareto Gate `NOT_EVALUABLE`.
- A fail-closed Oracle/Pareto record contract now fixes paired pre-decision,
  independent-cycle, aggregate, and theory-diagnostic fields. Its validator
  accepts an empty exploratory template but rejects formal promotion without
  passing upstream Gates and nonempty paired records. The targeted schema plus
  R3 tests pass (`7 passed`). The broader ReturnManager/environment, navigation,
  R3 mechanism, and complete EIRR theorem selection passes under uv (`183
  passed`, 187 third-party deprecation warnings).
- Reusing an `evaluation_seed` alone was found insufficient for paired causal
  interpretation because method-dependent return times change task-sampling
  position and RNG consumption. Stage-B now keys every task by
  `(evaluation_seed, battery_cycle, task_index)` and records an Oracle-shadow
  decision at every pre-action check. The revised smoke validates all five cycle
  joins and the machine evidence bundle, but is explicitly non-formal. After
  this change and the selection-aware paired throughput inference, the broader
  related selection passes (`194 passed`, 187 third-party deprecation warnings).
- The independent Oracle execution is now checked against every method's
  Oracle-shadow through the stopped first disagreement. In the dedicated smoke,
  all 214 compared decision events across three method/schedule groups matched
  exactly in state, remaining Energy, exact effective requirement, and Oracle
  decision; maximum position and requirement errors were both `0.0`. The smoke
  remains `formal_eligible=false`. A formal coupling mismatch, incomplete cycle
  join, or inherited legacy PASS without `evidence_integrity_passed=true` stops
  the pipeline before downstream comparison.
- The Oracle safety frontier now uses selection-valid simultaneous inference.
  Pointwise Wilson intervals are retained only as descriptive columns. For the
  default 12-point family, the one-sided Bonferroni--Clopper--Pearson zero-event
  upper bound is `0.053332` at 100 cycles and `0.048603` at the preregistered
  110 cycles. That value is now classified as minimally evaluable but
  underpowered; the jointly powered default is 320 cycles. The machine evidence
  contract is upgraded to `return-to-charge-oracle-pareto-v6`, and inherited
  pointwise-only, underpowered, or percentile-throughput-only PASS files are
  rejected. The expanded ReturnManager, environment, Oracle-record, Energy,
  and theorem regression selection previously passed under uv (`316 passed`,
  187 third-party deprecation warnings); the prior expanded regression passed
  `219` tests. After the workload/shadow changes, the changed-path regression
  passes `137` tests with `187` third-party warnings, and the new five-test
  design/parallelism/shadow selection also passes.

## Completion Conditions

The active research goal is not complete until one of the following evidence-
bounded outcomes is reached:

1. **Scientific continuation:** the selected frozen-platform identity,
   calibration, endurance, and Oracle Gates pass;
   the five-method decision frontier is complete; Stage C pilot and its stop rules
   are adjudicated; the required compositional/generalization evidence is either
   completed or honestly narrows the venue claim.
2. **Scientific stop:** a preregistered Gate fails with a valid, auditable formal
   artifact; the failure is classified as scientific rather than implementation;
   unsupported downstream work is not run; and the resulting research conclusion
   and revised route are documented.

Neither a smoke test nor a running background session satisfies either condition.
