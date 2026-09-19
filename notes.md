# Current research notes

## 2026-09-19 P1.1: task screening removed

P1 passed user review. This bounded correction moves C/R to task-free post-delivery
D-states; C draws only after commitment, R draws/discards nothing. Charged H has
no C/R choice and forces its first fresh IID task. New PostDeliveryUAV inherits
unchanged P1 navigation/service; historical source/evidence are preserved.
15 tests (9 original + 6 new), real one-step smoke, checkpoint resume and raw audit
pass. Three single-seed 1,000-second windows: after-1 11 deliveries / 11 recharges,
after-2 13/6, threshold-30 13/4; no depletion/contact, exactly one initial reset
each. All policies consume the same task-sequence prefix, with no R-induced RNG
advance. Final fractions <.05s are censored on the physics grid. This is semantic
sanity evidence, not reserve insufficiency or a benchmark. P2-A remains deferred;
no oracle/census/branching, energy critic, q95 fitting or neural learning.
Report: `docs/REGENERATIVE_CONTROL_P11_20260919.md`; portable raw traces:
`research/regenerative_control/evidence/p11_20260919`. Nothing left running.

## 2026-09-19 P1 continuing regenerative environment complete

User cancelled P0-B critic qualification; P0-A PASS for restricted startup.
No old critic recovery or training. New code preserves original navigation/source
checkpoint and implements pre-task C/R, whole pickup/dropoff, paid station capture
and charge, continuing IID task stream, exact physical state across goals.
Final v2: 128/128 macro smoke successes, zero contact; 1,000-second trace has
14 deliveries, six recharge completions and one initial reset, no depletion.
Nine tests and raw/source/energy/regeneration audits pass. Initial v1 retained;
cutoff guard fixed in v2 with independent same-budget evidence.
Regeneration is conditional on fixed map and renewing controller state. Grouped
quantiles reflect initial-velocity sampling, not independent navigation noise;
no joint mission quantile or complete finite Markov kernel is claimed. P2 remains
unstarted and requires the user's next authorization. Portable raw evidence is in
`research/regenerative_control/evidence/p1_20260919`; report:
`docs/REGENERATIVE_CONTROL_P1_20260919.md`.

Earlier P0 stopping language below is historical and superseded by this amendment.

## 2026-09-19 persistent regenerative control: P0 not passed

The user replaced open-ended research with ordered P0/P1/minimal P2 and no new
SAC or high-level training. Original navigator checkpoint SHA256
fb79482ae7d832b40137ff1a080ee84912eabb95867e118aca80c846f6ae62be is frozen.
18 target-role legs (100/300/600 distance, 4/24 obstacles): 18 reached, 0 contacts.
One scene per cell is an engineering check, not broad reliability evidence.
Retained energy heads are return-only mean/q95/defective, not a validated joint
pickup/dropoff/return four-quantile predictor. Runtime energy estimator defaults
to None. Coverage remains unavailable, not an observed calibration failure.
Stop before P1/P2; user confirmed the intended critic probably no longer exists. No method comparison,
ΔQ plot or sign-flip result exists. Preserve ≥.98 simple/oracle kill criterion.
See `docs/REGENERATIVE_CONTROL_P0_20260919.md`; raw outputs in
`artifacts/regenerative_p0_20260919`. Earlier entries retain historical provenance.

## 2026-09-19 control adaptation startup

User selected pretrained continuous-control adaptation/retention as the new
problem, not a predeclared novel mechanism. Budget-reading paused; no AFPP/V3
or full-vector escalation. New `research/control_adaptation` leaves old runtime intact.
Seed 5101 startup paused at 1,573 transitions / 617 updates, finite parameters.
Five tests pass. Native parity requires serial clients: upstream changeDynamics
omits physicsClientId. Actual trainer uses one client.
Further diagnostic: Quadrotor overwrites supplied init randomization after its
base constructor; perturbations are additive. Native z=1 plus U(.1,1.5) can start
above z=2 termination boundary. Seed 8101 starts at z=2.203167; both controllers
terminate after one step. Preserve this failure, no seed replacement or quiet fix.
This is startup evidence only, not a competent base or adaptation comparison.
Details: `docs/CONTROL_ADAPTATION_STARTUP_20260919.md`.

## 2026-09-19 public-task prototype: mechanism hypothesis not supported yet

The user withdrew theorem-only escalation and authorized an actual low-cost
method experiment. New isolated `research/budgeted_design` uses the DAD 2D,
two-source continuous task, not FPL roots/teachers or resource constraints.
Fifteen 600-update training runs completed. Paired 512-rollout evaluations at
four horizons, L=4096, include three seeds and query-only interventions.

Budget-query minus pool mean sPCE differences at H=4/6/8/12:
-0.0308 / -0.0940 / -0.0869 / -0.1450 nats. Against fixed attention:
-0.0084 / -0.0400 / -0.0698 / -0.0458. Query interventions do not show a robust
positive effect. Do not claim superiority, convergence or a universal negative
result; this was a short DEV comparison. No automatic larger run started.

An initial evaluator forgot `load_state_dict` and evaluated initialized
networks. All initial evaluation rows are INVALID and preserved with a notice.
The trained checkpoints themselves were valid; corrected evaluation uses identical
checkpoint bytes in a new directory and passes loading/end-to-end regression.
Training-tail timing overlapped that initial invalid eval; training times are
accounting, not isolated speed evidence. Corrected inference latency is separate.
No original Pyro-stack/full-budget DAD reproduction or Step-DAD experiment is
claimed. Details, raw estimates and portable checkpoints are linked in
`docs/BUDGET_READING_PILOT_20260919.md`.

## 2026-09-19 residual decomposition: Route C

All 4,684 census decision probes and 128 exact episode decompositions completed.
Large VOI has zero fresh-state and episode gaps on all 32 pilot variants. Small
VOI has 9/1,171 fresh nonzero residuals, yet 20/32 positive episode gaps due to
the actual cumulative compute contract. 79.72% of its pooled gap is explicitly
autopilot; 82.56% is at nonmeasuring-optimal states; all positive loss accompanies
work-limit hits. No evaluated policy visits non-prior measurement-required
states, so that subset contributes zero occupancy-weighted gap. This qualifies
the old curriculum focus and supports stopping the current learned-planner line,
not invalidating previous DEV findings or claiming amortization is impossible.
No new roots/DEV/training; 452 supplemental evaluation-only states on the same
problems. 63 tests pass. See `docs/FPL_T51E_RESIDUAL_20260919.md`.

## 2026-09-19 bounded census qualifies the Case C diagnosis

Depth≤2 measurement-only closure (including joint batches) completed for the
same 32 pilot variants: 1,171 exact states / 1,115 non-prior / 6 adaptive;
zero unresolved. Five adaptive states were missed by the old collector, across
three structures/two families. Train has 2 adaptive states in one structure;
validation has 4 in two structures. All adaptive states are depth 1, H_initial12.
This demonstrates a collection gap (qualified C1) while coverage remains sparse.
It does not prove economics is the sole cause or that arbitrary V2 histories
lack adaptive states. Prefer V2.1 measurement-closure collection design before
V3; no new roots, design changes, full run or training executed. See
`docs/FPL_T51D_CENSUS_20260919.md`. Old source/evidence and mask remain frozen.

## 2026-09-19 V2 sizing: Case C, not a solver failure

All 32 structural identities / 128 paired variants were admitted before labels;
the authorized 8 groups / 32 variants yield 591 exact states, 358 non-prior,
41 measurement-required, but only **one** non-prior required-sensing state,
in validation (train zero). All 591 pooled states fit below the retention cap;
the quota did not discard candidates. Information explorer adds 150 non-prior
memberships and the sole adaptive state, yet does not fix continuation coverage.
Nine Beam source collections hit episode caps; label solves had zero failures.
Catalogue-free joint time/energy/count DP matches all 5,383 V1 and 6,849 V2
prefixes. 56 tests pass. Stop full V2 expansion and AFPP training under Case C;
do not retune frozen seeds/folds/caps. See `docs/FPL_T51C_V2_SIZING_20260919.md`.

## 2026-09-19 prefix supervision and V2 design freeze

Pure audited-Q export yields 5,383 prefixes from 273 states, with 2,394 decision
prefixes and 82 tied next-action sets. END rows carry no policy weight; each
state totals weight one. Root STOP differs from decoder END. No new solves.
Metadata-only structural holdout gives 145 train/128 val states over 6/2 groups,
but non-prior measurement-required states are 0/2. This split is not retuned;
prefix expansion does not cure continued-sensing coverage. V2's 32-group design
and 8-group sizing pilot are prospective, not implemented or executed. 49 tests
pass. No network/test/CONFIRM/push. See `docs/FPL_T51B_EXPORT_V2_DESIGN_20260919.md`.

## 2026-09-19 Teacher V1 complete census

The authorized unchanged 23-root continuation completed. Full audit passed:
273 exact, zero unresolved; 147 non-prior states; 37 measurement-optimal states,
but only five in their intersection, all hierarchy-dense. 24 roots remain only
eight structures. 58 states have tied optima. Label CPU totals 4.273 seconds;
whole-shard CPU 7.574 seconds. No selected-state solver failures observed, not
proof against small/easy-task sampling bias. Exact first-operation changes with
belief/horizon exist but do not establish broad adaptive-continuation coverage.
Full evidence: `docs/FPL_TEACHER_V1_COMPLETE_20260919.md`. No prefix export,
Teacher V2, neural training, test/OOD or confirmation run in this task.

## 2026-09-19 T5.1: exact teacher data startup

TRAIN inputs admitted 24 roots across eight predefined strata; one structural
clone rejected before labels. First root: seven exact V/Q labels, two updated
belief states, no optimal measurement at this short budget. Kept without tuning.
Mixed histories use public predictive feedback, not hidden truth. Status/caps
remain explicit; arithmetic replay is not an independent optimality proof.
43 tests pass. Remaining 23 root shards and broad coverage are pending under
the first-checkpoint handoff rule. No network/training/test/CONFIRM. See
`docs/FPL_T5_1_STARTUP_20260919.md` for the sealed continuation command.

## 2026-09-19 fourth FPL batch: exact fixed-policy evaluation

371 fixed-policy expectations are exact, conditional on owned internal RNG state;
no MC fallback needed. The old sparse-T12 estimate 3.0 becomes exactly 5.25 for
Beam/VOI/cached Bayes. Fresh hierarchical DEV solves Bayes 10/12. Frozen operational
GO rule passes: two redundant-channel configurations show low-work exact gaps
closed by higher-work Beam; same-model H12→H18 reference work rises >6x even
excluding failed smaller-cap attempts. Cheap VOI remains a serious competitor.
Dimension changes also alter prior/sparse edges in this generator; do not claim
an isolated causal redundancy effect. MCTS is a custom fixed-seed baseline.
No neural training, no new pre-B2 gate, no final-test execution. Local seed
commitment is not independent external blinding. See FPL_FOURTH_BATCH_20260919.md.

## 2026-09-19 third FPL batch: bounded scientific comparison

RegistryV2 separates distribution/template/clone identity. Global selection
budgets include recursive calls. Rational minimax bases retain independent
primal and dual supports. Paired-CRN DEV study completed 1,920 planned records,
retaining 144 unresolved-reference attempts; no test evaluation or training.
Bayes exact 11/12; minimax exact 4/12; cached policy values replayed rationally.
The sparse T=12 MC estimate (3.0) falls below exact Bayes risk (5.25): it is
small-sample optimism, not beating the oracle, even with empirical SE zero.
The sole unresolved Bayes condition hit a configured 2,000-state cap. This does
not establish the quality–compute gap required for a neural method. No B2 promotion.
See `docs/FPL_THIRD_BATCH_20260919.md` and its portable DEV evidence archive.

## 2026-09-19 second FPL batch

Utility is separate from sensing; hidden-model task payoff is evaluator-only and
does not silently enter the posterior. Template grouping plus name-invariant
structural fingerprints prevents configured split leakage. Rational minimax
certificates are available for tiny policy-tree games. Beam and posterior-sampling
baselines avoid complete catalogue enumeration. Twelve measured scaling cells
show topology-dependent catalogue growth and a distinct belief-tree cost; low
engineering caps must not be advertised as fundamental computational failure.
See docs/FPL_SECOND_BATCH_20260919.md. Quality/compute comparisons remain open.

## 2026-09-19 isolated feedback-protocol implementation

First batch from expert_advice is executable: committed reset-to-reset protocols,
separate duration/energy, exact public belief, channel-coverage heuristic and
rational finite-budget Bayes reference. The new comparator is matching known-model
finite-budget value, not silently T*rho. Metadata-only inventory found every
allowlisted old evidence path. 13 tests and two debug smoke episodes pass;
no algorithm superiority inferred. See docs/FPL_FIRST_BATCH_20260919.md.

## 2026-09-16 completed AI relevance layer

Feasible acquisition protocol belongs to the comparison object, not just an
implementation detail. Report joint experiments, acquisition/reset costs,
feedback timing and conditional continuation, alongside the two exact summaries.
“Not determined by two summaries” is not probabilistic independence or a claim
that geometry alone determines risk. ReAct/AgentBench support context only;
no LLM application of the theorem or benchmark-failure claim. Negative T4096
calibration retained. See docs/ICML_AI_RELEVANCE_LAYER_20260916.md and its audit.
No research or experimental expansion.

## 2026-09-16 completed reusable comparison bridge

Diagnostic strictness needs new-policy benefit at EVERY old least-favorable prior,
not one prior with improved continuation. A public old Bayes lower l and one
new mixture with worst risk U<l supply coverage without locating those priors.
Asymptotic neutrality separately needs a surviving old maximizing dual.
Automated screening batches give an exact non-navigation history transfer, not
independent validation. New normalized-posterior/vector checker reproduced the
T12 and terminal/static controls exactly; 4 tests pass. Figure 1 v2 preserves
certificate data and original exports. External independent audit remains
unclaimed. See docs/RESOURCE_SEPARATION_BRIDGE_AUDIT_20260916.md.

## 2026-09-16 completed paper-core assembly

Main result first: resource-realizable nesting can preserve two specific summaries
while strictly changing finite-budget minimax risk. Full Introduction, Theorem 1,
Discussion and source-bounded related-work table assembled; Figure 1 uses exact
anchor lower/upper certificates, not exact risk values or invented curves.
Maximum coefficient equality is not full geometry equality. Negative T4096
calibration retained. See docs/ICML_RESOURCE_SEPARATION_PAPER_CORE_20260916.md
and docs/ICML_RESOURCE_SEPARATION_ASSEMBLY_AUDIT_20260916.md. No new research run.

## 2026-09-16 theorem-section architecture (completed)

Put the robust resource-realizable strict separation first, existing duality
criteria second as explanation, controls third. The main claim compares two
summaries: execution and maximal fixed-instance C. It does not preserve the
whole asymptotic allocation geometry: A's coefficient changes. Thus neither
"statistically indistinguishable" nor "neutral asymptotic geometry" is justified.
ML takeaway is insufficiency of those summaries for finite-budget risk comparison,
not a new sensing principle. See docs/ICML_THEOREM_SECTION_OUTLINE_20260916.md.
Writing only; negative calibration and all certificates unchanged.

## 2026-09-16 Introduction scientific interpretation (completed)

Resource augmentation can alter feedback-dependent allocation of future calendar
budget while preserving execution and maximal fixed-instance C. Do not claim a
timing-only mechanism: the high witness still commits to its entire sortie and
adapts between completed excursions. Nor does asymptotic neutrality require an
inactive new dual row: AB is tight at the surviving q price. The introduction
states the robust separation inside controlled sensing, with qualified execution
equality and negative T4096 calibration intact. See
docs/ICML_INTRODUCTION_FIRST_PAGE_20260916.md. No new proof or run.

## 2026-09-16 completed controlled-sensing positioning

Exact representation in controlled experiments is accepted, not evaded. Cost,
adaptivity, posterior continuation, GL allocation and the broad finite/asymptotic
distinction all have strong precedents. Wu et al. 2015 explicitly discuss zero
logarithmic coefficient with substantial finite-time difficulty; Adusumilli 2025
uses decision risk beyond rate rankings. Retain the narrower robust conjunction
of resource realization, qualified execution equality, equal maximal fixed-instance
C and strict finite-budget minimax risk. Named prior theorems do not directly
supply that certificate; this is not proof of worldwide priority. Positioning
and comparison are complete, without new math or runs. See
docs/CONTROLLED_SENSING_REDUCTION_POSITIONING_20260916.md and its evidence audit.

## 2026-09-16 abstraction — distinct asymptotic and finite bottlenecks

Nested experiments: v1-v2=min_pi[(v1-b1(pi))+(V2(pi)-V1(pi))]. Strict finite
benefit requires improvement at EVERY old least-favorable prior, not just more
KL at lower shared cost. Equal maximal fixed-instance C requires an old
maximizing hypothesis's optimal KL-price dual to remain feasible on new
experiments. The existing q dual survives AB exactly while the finite witness
certifies a uniform gap. This yields a general dual-criterion theorem and the
current open family as corollary. It does NOT establish new foundational
controlled-sensing theory; all duality tools are classical and novelty remains
open. No new game/horizon/learner/experiment. See
docs/RESOURCE_ENABLED_FINITE_BUDGET_PRINCIPLE_20260916.md.

## 2026-09-16 open-family resource separation — mathematical audit only

Four structural parameters (r,u,v,w) vary openly around(.05,.05,.30,.95),
with hypotheses(u,r),(v,r),(v,w) and common dock mean r. A uniform policy
coupling gives risk Lipschitz constant T(T+3)/2, so the old T12 certificate
survives throughout radius1e-4 with gap>.038509475. Equal execution and equal
maximal fixed-instance asymptotic coefficients across capacities are proved
throughout this box. Exact coefficient equality uses the stated B-baseline=dock
sharing; it is NOT open in an unrestricted reward matrix. No span equality
throughout the box is claimed. Primary-source audit finds substantial controlled
sensing/structured-bandit overlap; novelty remains uncleared. No sampled run,
new horizon, policy change or T4096 reinterpretation. See
docs/ROBUST_RESOURCE_BUNDLING_SEPARATION_20260916.md.

## 2026-09-16 strict finite-budget benefit — exact theory certificate, NOT a run

The current three-hypothesis class has unchanged maximal C=.498691947 at theta_q;
the primary truth's C reduction must not be used as a minimax coefficient reduction.
The requested F fixed-point lower is valid, but cannot exceed the prior high
frozen-learner upper at any T>8. Exact finite-game Bellman certificates on the
SAME class instead yield Rstar_3(12)>=2173/8000=.271625 and
Rstar_4(12)<=8604621/40000000=.215115525, gap>=.056509475. Every integer T=12..24
has an exactly verified gap>1/20. Low lower includes primitive feedback adaptation
and arbitrary legal terminal states; high witnesses are committed and end at q.
Known-model charger-terminal execution at12/24 is equal at every hypothesis;
capacity-only off catalogue is exactly equivalent at all T. Feedback-free high
charger-terminal risk at12 is32/115, above the adaptive witness. This is a
handwritten/computer-assisted theorem certificate, not independent formal proof,
novelty adjudication, or an empirical result. No T=4096 trajectory rerun or retuning.
See docs/STRICT_FINITE_BUDGET_CAPACITY_BENEFIT_20260916.md.

## 2026-09-16 first scientific bundling calibration — no primary ordering

Same one paired seed resumed from 128 to4096, all twenty traces valid and sealed
sources unchanged. Primary resource-path/cost-aware/cost-blind: P_low,on=4.10,
P_high,on=4.30, P_low,off=P_high,off=4.10. Requested ordering did NOT appear;
negative control is exact. Complete path costs 4.00 versus4.40, final partial
remainders +.10 versus-.10. B-only remains8 in both; high/on adds8 AB (7 forced),
so finite forced-coverage overhead dominates rather than replacing B-only cost.
Residual P-C logT is3.315 split versus4.143 bundle; no asymptotic-fit claim.
Privileged allocation and weak UCB have favorable ordering but cannot substitute
for failed primary calibration. No tasks/hour, seed sweep, horizon extension,
learner changes or descriptor library. Stop and preserve valid negative result.
See docs/BUNDLING_CALIBRATION_RESULT_4096_20260916.md.

## 2026-09-16 measurement-bundling validation proposal

Stop theorem expansion; plan an isolated idealized 2D known-route inspection robot.
Exact finite Theta={(.30,.05),(.30,.95),(.05,.05)}, known dock reward mean .05,
singletons take3 and bundled AB takes4. Optima A/B/dock are unchanged under EVERY
hypothesis at capacities3/4 and bundling-on/off. At primary truth (.30,.05),
gain=.1, reachable bias span=.2; only theta_B is a costly confusing alternative.
With a=kl(.05,.95)=2.6499951, C_split=.25/a=.0943398 and C_bundle=.05/a=.0188680.
Bundling-off restores split cost at the SAME high capacity. These are a new
calibration of the existing theorem, NOT a new theory claim or observed gain.

Proposed routing library: distinct physical route/cost signatures, not coordinate
reskins; seal C and world selection before learning outcomes. Evaluate pseudo-regret
privately because realized optimal-route reward noise grows sqrt(T) and can swamp
the log(T) signal. Do not claim an 80% coefficient change gives 80% tasks/hour gain.
Oracle allocation is a privileged scheduling reference, not known-model-optimal
execution. Cost-aware structured baseline should share the same leading coefficient;
cost-blind can tie too, especially in calibration. Finite horizons cannot certify
asymptotic slope equality or automatically refute it. No implementation/run,
no DEV/CONFIRM access. Proposal choices and startup authorization remain pending.
See docs/MEASUREMENT_BUNDLING_EMBODIED_VALIDATION_PLAN_20260916.md.

## 2026-09-16 class-wide resource-path attainability draft

Finite known observable reward hypotheses, common finite positive supports,
unique optimal safe path types; deterministic known dynamics/cost, one recharge,
decreasing. Shared row parameters and unequal durations are allowed. One learner
uses finite-model likelihood certificates, forced query counts sqrt(j), exact
model-selected LP quotas (1+eta_j)lambda f(t), and zero-cost optimal-path quotas
for nonconfusing alternatives. eta_j=j^(-1/8); f(t)=log t+2 log log t+O(1).
False exploitation is summable; uniform-in-count row likelihood events fail with
summable probability O(j^(1/4)exp(-a j^(1/4))) at exploration index j. Query count
is O(log T) in expectation, forced cost O(sqrt(log T)), bad-query cost O(1).
Thus Reg/log T -> C_B at EVERY fixed hypothesis. Handwritten, audit pending.

This closes class-wide attainability beyond the symmetric five-state example.
It also proves within-excursion adaptivity cannot improve the leading coefficient
in this subclass: committed paths attain the all-adaptive lower. It does NOT
extend the continuous example's upper to general continuous Theta or optimal ties.
OSSB explicitly discusses vector semi-bandit feedback; claiming otherwise would
be wrong. Its displayed per-round upper does not literally match unequal-duration
calendar cost, but this is a weighted structured-experiment allocation, and broader
Graves--Lai/recurrent-control prior art prevents declaring a novel principle.
No implementation, training, UAV or DEV/CONFIRM access. See
docs/RESOURCE_PATH_CATALOGUE_ATTAINABILITY_20260916.md.

## 2026-09-16 general resource-coupled lower theorem

Generalized the five-state result to finite deterministic, single-reload,
decreasing safe Consumption MDPs with shared unknown bounded reward laws.
Theta can be finite or continuous and is frozen before capacity is chosen.
Safe complete path counts generate a cone; G_B is their minimum centered cost
for a prescribed primitive information allocation. It is polyhedral and may
be nonseparable. Confusing alternatives agree on all rows of optimal paths
and share no optimal path with truth. Low regret implies discrimination via
gamma=min_p (c_theta(p)+c_theta'(p))/ell_p>0 and a pathwise two-slack test.
This proves liminf Reg/log T>=C_B, without unique optimum or IID adaptive cycles.
Use predictable primitive-row KL, never Wald on reward-selected path counts.

The previous coupled coefficients and learner are Example 1 / matching special
case, not a general upper. The generalization is a handwritten result, not an
independent audit or novelty certification: Graves--Lai and structured bandits
remain direct prior art. No unknown-transition or growing-B minimax claim.
No implementation or experiment. See
docs/GENERAL_RESOURCE_COUPLED_INFORMATION_LOWER_BOUND_20260916.md.

## 2026-09-16 concrete coupled frontier result

Five base states q,h0,hA,hB,hF, every primitive cost1, known deterministic
transitions, q baseline reward .2, and two IID Bernoulli means shared across
measurement rows. At B=3 only singleton sorties (3 steps) are safe; B=4 permits
AB in one sortie (4 steps). Both means were identifiable already at B=3. At
the SAME true means (.1,.1), gain=.2 and canonical span=.2 for both capacities;
p0=H_R=1, decreasing and primitive class fixed.

Projected exploration cost is G3=.5(xA+xB), G4=.5(xA+xB)-.4 min(xA,xB).
Decision alternatives at B=4 additionally include nuA+nuB>.8, requiring a
joint KL constraint. Symmetry/convexity solve the allocation exactly:
C3=1/kl(.1,.6)=1.8159985; C4=.3/kl(.1,.4)=1.3257374 (26.997% reduction).
One theoretical certify-or-reveal learner uses the CHECKED Kaufmann--Koolen
2021 Theorem 7 sum-KL calibration and safe revealing blocks; it attains these
log-regret coefficients on the unique-optimal fixed class. This is a handwritten
lower+upper draft, not a minimax theorem or an independent/formal proof audit.
Actual safety remains known; learning does not alter the safety kernel.

This provides a feasible coupled result, not just another negative/check. But
the catalogue is still a duration-aware structured semi-bandit: coupling alone
is not a new principle. NeurIPS 2018 DEL cannot be blindly applied (every-policy
irreducibility fails under q baseline). No implementation or experiment.
See docs/COUPLED_CONSUMPTION_FRONTIER_RESULT_20260916.md.

## 2026-09-16 narrow consumption-bandit attainability draft

Frozen independent deterministic sorties have energy e_i, duration ell_i and
terminal Bernoulli mean mu_i. Capacity gives F_B={i:e_i<=B}; gain is max mu_i/ell_i.
Low regret under mu and a single-arm alternative v>ell_i rho* implies a testing
event, hence E N_i/log T>=1/kl(mu_i,ell_i rho*) for competitive suboptimal arms.
The allocation solves C_B=sum(ell_i rho*-mu_i)/kl(mu_i,ell_i rho*).
One calendar-time KL-UCB rate learner achieves Reg/log T->C_B using the checked
predictable-sampling bound from Garivier--Cappe 2011 Theorem 10. Endpoint arms
with ell_i rho*>=1 and terminal incomplete sorties are explicitly accounted for.

This is matching INSTANCE-dependent asymptotic optimality among uniformly
efficient learners, not a gap-uniform minimax log T theorem. Attainability passes
for this narrow class, but its known feasible-arm gating and independent cycles
decouple the geometry; no new resource-RL result is established. Theoretical
policy only, no implementation or experiment. Independent proof audit pending.
See docs/CONSUMPTION_BANDIT_ATTAINABILITY_THEOREM_20260916.md.

## 2026-09-16 positive feasibility-gated theory foundation

Stop negative example search. The corrected statistical object is Theta/~_B,
not a capacity-dependent subset of true parameters. Equivalence of all safe
transcript laws equals equality of queryable base outcome rows, and preserves
EVERY safe-policy objective. Thus incomplete identifiability alone is not a
regret obstruction. Virtual budgets give E_B1 subset E_B2 for B_1<=B_2.

At fixed theta, exact time-indexed safe flow constraints characterize attainable
expected occupancies. For one adaptive learner, transcript KL is sum of shared
base-row expected counts times row KL. Regret is expected augmented occupation
times Bellman slack plus a boundary of magnitude at most the S0 span H.
These give a testing-constrained information-cost program C_B and the necessary
bound Reg>=C_B-H. This is NOT matching minimax: low regret must imply the chosen
tests, and one unknown-model learner must attain compatible allocations across
Theta. E_B alone loses query frequency, transit/recovery cost and comparator
changes. Existing Graves--Lai/structured-RL information constraints are direct
prior art; no new principle or rate is claimed from these identities alone.
See docs/CONSUMPTION_FEASIBILITY_GATED_INFORMATION_REGRET_20260916.md.

## 2026-09-16 S1 universal equivalence obstruction

Universal multiplicative/O(1)-additive equivalence across ALL viable capacities
fails under S0 assumptions. A fixed two-state base class has q idle (cost1,
reward0), two unknown Bernoulli reward actions q->w (cost2), and w->q
(cost1,reward0). At B=2 the reward actions are not viable, so R_2(T)=0.
At every B>=3 they are viable, and a direct adaptive testing argument plus
an existing two-arm bandit upper bound give R_B(2n)=Theta(sqrt(n)).
The class, costs, support and reward laws are frozen; p_0=1, H_R=1,
single reload, decreasing, and span<=3/8. This exposes feasibility-gated
information, not rare transitions or growing bias. All B>=3 in THIS example
are exactly equivalent, so it does not disprove eventual equivalence generally.
Failure of universal equivalence does not imply Omega(B) hardness. Distinguish
'no additional leading B factor' from 'same minimax risk across capacities'.
Handwritten proof; independent audit remains open. No new algorithm or run.
See docs/CONSUMPTION_COUNTER_STATISTICAL_EQUIVALENCE_KILL_TEST_20260916.md.

## 2026-09-16 S0 capacity-free span under base regularity

The handwritten S0 attempt closes for the single-reload decreasing reachable
subclass with uniform safe recovery H_R. Upward-safe coupling permits deletion
of every positive-cost repeated-base-state subwalk. A maximal reachable battery
representative consequently has a safe supported base-simple access path of
length L<=S-1 and probability alpha>=p_min^(S-1). Attempt the path once;
recover after a deviation, and execute the canonical greedy excursion after
success. Nonpositive centered reward of this complete charger cycle implies
max h<=rho [L+(1-alpha)H_R]/alpha. Together with min h>=-rho H_R:
span(h)<=rho (L+H_R)/alpha<=(S-1+H_R)/p_min^(S-1).

Uniform p_min>=p_0>0 and B-independent H_R give a class-uniform B-free bound.
This kills capacity-growing canonical span under those assumptions, NOT every
capacity-dependent minimax effect. A fixed three-state class p in (0,1) has
span=2(B-2)/(2+p(B-2)); fixed p gives a bounded span, whereas choosing p=1/B
within the same class produces linear growth with a vanishing base probability.
438 exact Fraction checks passed; these validate example equations, not the
general theorem. Independent proof audit, theorem-level novelty, and S1 minimax
consequences remain open. No algorithm, experiment, training or CONFIRM ran.
See the appended S0 derivation package.

## 2026-09-16 S-minus-1 canonical reachable excursion

The active S0 class is now fixed-base, single-reload, decreasing, with normalized
charger q_B and safe domain D_B consisting ONLY of states reachable with positive
probability along a finite safe history from q_B. All safe supported successors
of D_B stay in D_B. A nonreload lifted cycle would project to a zero-consumption
base cycle, so the nonreload support is a DAG. For fixed B every safe policy has
tau_q<=|D_B minus q_B| and tau_q^+<=1+|D_B minus q_B|. This is not B-free mixing.

Given scalar optimal gain and an AROE solution, all safe greedy selectors are
proper, zero slack is attained, terminal bias is zero, and h(q_B)=0 uniquely
determines h as the maximal centered excursion reward. Within the SAME B,
virtual-budget coupling gives h(s,b_2)>=h(s,b_1) for b_2>=b_1, when both states
belong to D_B. Maxima use the largest REACHABLE battery level, not automatically
B; reachable battery sets need not be intervals. Across-capacity monotonicity
is not proved because gain changes.

Uniform recovery to q_B still gives h>=-H_R, hence only the positive canonical
excursion remains in S0. For general models the fallback is minimum span among
AROE solutions on the chosen domain; in the active subclass normalized bias is
unique, so that optimization is unnecessary. Reward in the proof explicitly
means r_t=r_theta(S_t,A_t); realized-reward versions require bounded samples.
These are handwritten foundational proofs, not B-free-span/minimax results
and not independent or machine verification. See the appended S-minus-1 package.

## 2026-09-16 fixed-base stopped-bias lemma

The active minimax object is sup over a single B-independent Theta of
Regret(Lift_B(M_theta)); B changes only the counter capacity. The main sequence
is S0 -> S1 -> S2; exact-planning hardness is now a nonblocking side question.

The handwritten proof gives an exact stopped Bellman identity: h(x) equals
expected centered reward up to an integrable reload hitting time, plus terminal
bias, plus accumulated nonnegative Bellman slack. Uniform recovery alone gives
h(x)>=min_reload h-rho H_R>=min_reload h-H_R. Normalize min_reload h=0 for the
claimed negative-side bound; an arbitrary single charger reference is insufficient
when recovery is only to a multi-charger set.

Excursion equality requires zero infimum total slack among integrable hitting
policies (e.g. a proper Bellman-greedy policy). A two-state zero-cost example is
communicating and uniformly recoverable but has multiple Bellman biases and
fails equality for some of them. The proof therefore does not assert iid
regeneration, unconditional h=G, or a B-free full span. Next S0 must control
positive centered excursion AND reload-boundary bias oscillation, or explicitly
specialize to a common single reload reference. Details are appended to
docs/CONSUMPTION_MDP_REDUCTION_KILL_TEST_20260916.md. No machine proof or
independent proof review is claimed.

## 2026-09-16 Consumption-MDP statistical-complexity reframe

The central question is now whether a known deterministic replenishing counter
can change minimax regret despite adding no unknown transition parameters.  The
generic lift proves only `d_lifted = d_base`; capacity may still enter through
the optimal bias span or another effective-horizon quantity.  Span growth alone
is not enough: it must be converted into a subclass-specific information-theoretic
lower bound, with base dynamics, support, rewards, costs, and mixing quantities
controlled independently of B.

The obvious long-sortie witness fails exactly: a deterministic cycle with B
unit-reward work steps and two zero-reward reload/return steps has gain
`B/(B+2)` but bias span `2B/(B+2) < 2`.  Long cycle length alone therefore does
not imply a B-dependent learning horizon; a valid lower-bound family must create
B-scale cumulative reward imbalance without hiding B in the base dynamics.

General compact exact planning is provisionally demoted.  Exact quantitative
reachability in binary-encoded Consumption MDPs is NP-hard, and a
target-to-absorbing-reward-one construction appears to transfer this to exact
expected mean-payoff planning.  The reduction still requires proof that making
the target safely absorbing does not enlarge the source feasible-policy set; it
also does not establish hardness of approximation or hardness in a safely
communicating subclass.  Compact approximate planning is deferred until a
statistical result survives.

The serial gates are now P0 (formal planning-hardness reduction), S0 (clean
capacity/span characterization), S1 (minimax statistical lower bound or B-free
characterization), S2 (matching upper bound), then C0 (tractable compact
planning subclass).  A restricted exploratory enumeration of deterministic
two-state/two-action templates found no linear span growth; this is only a
diagnostic that the obvious construction is nontrivial, not a theorem.

## 2026-09-16 Consumption-MDP reduction audit

The proposed “unknown Consumption MDP + hard non-depletion + average-reward
regret” direction does not yet survive as stated. With known transition support
and known costs, safety can be compiled into a greatest robust viability kernel.
The safe augmented transition model is an exact linear mixture with the same
unknown transition parameter dimension as the base model. Conditional on
Bellman optimality and a known bias-span bound, existing bounded-span
linear-mixture RL is therefore close to a direct statistical reduction.

Known support does not imply Bellman optimality: safe action restriction can
create irreversible end components with different gains, and unknown rewards
then yield linear regret from a single trajectory. Conversely, fully unknown
safety-relevant support conflicts with exact zero-violation exploration. The
remaining plausible contribution is not generic safe learning, but either a
replenishment-specific span/lower-bound result or compact optimistic planning
that avoids the explicit S(B+1) computation. Full derivation and kill gates are
in `docs/CONSUMPTION_MDP_REDUCTION_KILL_TEST_20260916.md`.

## 2026-09-16 v1a prospective amendment startup

The v1 startup revealed two analysis-contract blockers before any scientific
outcome was inspected: Gate M used energy-failure incidence rather than
failure-time accounting, and Gate N held the LOO selector fixed in bootstrap
inference. v1a preserves the interventions and worlds, adds simulated task
completion timestamps, freezes common-alive survival accounting, and replaces
the Gate N primary family with simultaneous M4-M0/M1/M2/M3 contrasts. LOO is
descriptive only.

The v1a freeze, 17 tests, excluded smoke, and first-DEV five-method startup grid
all passed integrity checks. v1 outputs remain startup-only and are excluded
from v1a analysis. No Gate analysis was run, no tasks/hour result was inspected,
CONFIRM remains untouched, and the remaining 23 DEV worlds remain unauthorized.

## 2026-09-16 empirical-route decision

The recurrence abstraction, lumpability, and finite-tail certification theorem
stories are closed. The only approved next gate is a controlled intervention
test of whether productive stuckness causally limits long-run tasks/hour.

Repository audit found that repeated recharge, fixed-horizon throughput,
unlimited task clocks, the frozen SAC navigator, and exact simulator
snapshot/restore already exist. The paused PSPS lineage cannot be repurposed:
it has 27 committed and 35 accessed DEV worlds and remains a distinct frozen
experiment.

The new plan uses 24 fresh paired worlds and five prespecified methods: SOC40,
fixed-timeout return, legal no-progress return, legal no-progress replan, and a
privileged task-start completion oracle that probes 4,000 future policy steps,
restores the exact state, and intervenes only in the executed counterfactual
run. The evaluation plant removes the 12,000-step branch guard. Gates separately
test oracle throughput headroom, unproductive-dwell mechanism, and whether
simple heuristics capture the oracle gain. No training is authorized. See
`docs/ORACLE_PRODUCTIVE_STUCKNESS_INTERVENTION_PLAN_20260916.md`.

Implementation/startup update: the isolated M0--M4 runner, local freeze and
identity validator, integrity-only startup auditor, and registered paired
analyzer are implemented. Nine focused tests passed. One excluded 120-second
smoke world and exactly one formal DEV world completed all five methods. Both
startup audits passed with zero CONFIRM access; formal analysis was not run and
no scientific outcome was inspected during the startup decision. The runner is
stopped before the remaining 23 worlds.

## 2026-09-15 requested best-world trajectory GIF

The requested visualization must use only already accessed PSPS DEV evidence.
It is a post-hoc qualitative diagnostic, not a formal generalization result.
PSPS Gate O/D has not run, so no PSPS-driven deployment selector exists yet;
the GIF must state which existing return trigger is used and must not imply that
the compact predictive state already controls C/R decisions.

The environment has an unbounded keyed task stream rather than a finite “all
tasks” list. The selected visualization is therefore one complete registered
battery-sortie R branch: best current complete DEV world by mean utility and
return tie-breaks, step-768 anchor, median-duration successful R replicate 27.
The CUDA replay exactly reproduced the saved 210-step return outcome. It
completed task 1 at step 360, committed R during task 2 at step 768, returned at
step 978 with zero collisions and 337.6823 pre-recharge energy. The 181-frame
GIF passed first/last-frame QA. See
`docs/PSPS_V1_BEST_WORLD_RETURN_GIF_20260915.md`.

## 2026-09-15 PSPS-v1 3×8 execution amendment

The user authorized reducing only outer world concurrency from 4 to 3 because
the resumed service approached its 20 GiB cgroup limit. Before the amendment,
the earlier interruption was proven to be a Python 3.12 SIGSEGV rather than an
OOM kill. The amendment must preserve all scientific design and all existing
committed, staged, and partial outputs; it must remain DEV-only with automatic
analysis, confirmation, and training disabled.

The 4×8 service paused normally at 15:48:55 with generation 20, 22 staged
records, 6,377 partial replicate files and 8,285,985 committed branch steps. A
hash snapshot protects 6,444 immutable evidence files. The 3×8 execution record
is `sha256:222c85c0358fd974cae76aefb1f90473b907ce2f763859116f626ff0e43ed191`;
it preserves the original manifest and changes only outer world concurrency.
Four focused tests passed before resume. See
`docs/PSPS_V1_EXECUTION_AMENDMENT_3X8_20260915.md`.

The amended `psps-v1-dev-resume-3x8.service` started at 15:52:41 and committed
generation 21 at 15:52:54. Post-resume amendment validation passed all 6,444
pre-existing file hashes; the new record hash/shape checks also passed. Current
and peak memory were approximately 4.6 GiB with zero swap at the first-checkpoint
handoff. DEV access remained 26; PSPS CONFIRM and old PAI CONFIRM remained zero.
No automatic analysis, confirmation, or training started. Monitoring stopped.

## 2026-09-15 PSPS-v1 prospective freeze

User authorized a new independent Progress/Stall Predictive State experiment.
The frozen state is 20 legal scalars: a matched 10-scalar latest prefix plus 10
fixed progress/stall trends. Gate O predicts C task completion before the branch
guard by cross-world binomial logistic ridge and must beat latest and constant
Brier baselines. Only then Gate D maps completion probability through an
outer-training-only threshold and must beat latest and best-constant utility.
Both use parity-split 32/32 discovery/evaluation swaps and simultaneous
whole-world max-t inference. Any DEV failure stops; old PAI CONFIRM is permanently
denied and METHOD-TRAIN false.

`artifacts/psps_v1_20260915` contains 2 excluded smoke, 96 fresh DEV and 128
unopened CONFIRM worlds. The 226 identities have zero overlap with all retained
contract registries. The preaccess record and freeze head were externally
anchored in Rekor entry
`108e9186e8c5677a706d07aedd1abe32c27b4481872395b4c03c96f28f86e55bf3f6109e9406423b`.
Contract validation passes with DEV access 0 and all CONFIRM access 0. Fifteen
focused current tests passed before freeze.

Excluded smoke then passed on 2 worlds / 4 anchors / 8 branches / 4,306 policy
steps. Formal `psps-v1-dev.service` started and reached the first durable world:
2 anchors, 256 branches, 126,923 policy steps, zero failures/contacts, exact
64-pair CRN grids and finite 10-D/20-D features. Validator reports five ordered
DEV accesses (four initial plus the newly scheduled successor), PSPS CONFIRM=0,
old PAI CONFIRM=0. Service current/peak memory was 8.0/8.4 GiB under 20 GiB.
Stop monitoring after this startup-health check. No automatic Gate analysis or
promotion is enabled. See `artifacts/psps_v1_20260915/startup_health_psps_v1.md`.

## Scientific state

Two earlier estimator-adequacy DEV studies did not establish stable legal-
history prediction of conditional action value. The current experiment instead
targets paired advantage `U(C)-U(R)` directly. Existing exploratory evidence was
used only to design Gate H/I and is not retained as formal evidence or pooled.

Gate H requires held-out selector value over the cross-fitted best constant
action, both action signs with sufficient world coverage, and positive noise-
corrected heterogeneity. Gate I additionally requires full history to beat
latest and cross-fitted constant advantage/action baselines under simultaneous
world-level inference. Full/latest action disagreement must be at least 5%, and
both fixed model families must agree in direction.

## Active lineage

PAI v1 was invalidated at zero formal access because a spawned child bypassed a
wrapper. PAI v2 was invalidated after 15 DEV accesses because its diagnostic
RMS summed per safety-filter invocation but divided by policy steps; collision
repair can invoke the filter more than once in one step. Individual disturbance
vectors remained bounded, but v2 cannot be repaired or pooled after access.

PAI v3 changes only the diagnostic denominator to the actual invocation count.
Its 226 physical worlds are disjoint from CMI v1-v4 and PAI v1/v2. Startup
health passed at 2/96 worlds with 512 formal rows, 253,893 branch steps, zero
unified collisions, zero operational failures, valid hashes, exact paired CRN
seeds, and zero CONFIRM access.

## Runtime dependencies retained by cleanup

- Active v3 contract, worlds, receipt, smoke, records, and checkpoints.
- Minimal old contracts/checkpoint head/acceleration decision required by the
  validator's disjointness and parent-hash checks.
- `new_navigation_energy_heads_20260908_v1/{latest.pt,features.pt}`.
- `new_navigation_energy_global_scale_repair_20260908_v1/repair.json`.
- `hocbf_correction_sac_20260908_v1/sac/checkpoint_000131072/model.zip`.
- `runtime_support/review_bundle/{envs,safety}`, imported through the root
  `review_bundle` compatibility symlink without changing environment/method code.

Cleanup reduced the workspace from approximately 204 GiB to 7.4 GiB, including
the untouched 6.8 GiB `.venv`; artifacts fell from approximately 197 GiB to
331 MiB. Full details: `docs/REPOSITORY_MINIMIZATION_20260914.md`.

## 2026-09-15 DEV v3 interruption

The initial `paired-advantage-dev-v3.service` stopped at 84/96 committed worlds.
The coordinator reported `BrokenProcessPool`; the kernel log identifies a Python
3.12 child SIGSEGV at 05:21:46 CST. There was no OOM-kill event, the service peak
was 17.4 GiB under its 20 GiB limit, and swap usage was zero. The durable state
contains an ordered 84-world prefix, 85 staged tickets, and 89 registered DEV
accesses. Contract revalidation passed with `PAI_DEV_READY=true`, no errors, and
zero CONFIRM access. Resume must use the unchanged collector and `--resume`; no
analysis is permitted until all 96 worlds and the raw audit complete.

The original `ERROR.json` was preserved as
`pai_dev_raw_v1/incidents/ERROR_20260915T052202_python_sigsegv.json`. At 08:19 CST
the unchanged collector resumed under `paired-advantage-dev-v3-resume.service`.
The resume checkpoint and status refreshed at generation 84, four world children
were active, memory was 5.1 GiB, and no new active error artifact was present.

## 2026-09-15 DEV completion and raw audit

Collection completed 96/96 with 96 durable tickets, 96 ordered DEV access
events, 192 anchors, 24,576 branch records and 30,154,281 branch policy steps.
The independent raw audit passed all record hashes, exact 64-pair C/R grids,
paired CRN seeds, utility calculations, unified collision accounting, finite
NPZ arrays, legal feature construction and access-chain checks. It observed two
branches with contact (2,719 unified policy-step contacts total), 1,938
operational failures and 22,638 returned outcomes; these are integrity summaries,
not Gate results. See `docs/PAIRED_ADVANTAGE_RAW_AUDIT_20260915.md`.

The audit exposed a frozen analysis-path compatibility bug. The hash-bound
`analyze_paired_advantage_dev.py` imports `load_dev` from the predecessor CMI-v3
analyzer, whose first guard requires contract ID
`conditional-history-mc-q-20260912-v3`; the formal PAI contract ID is
`paired-advantage-identification-20260914-v3`. The original analyzer would
therefore fail before computing a Gate. Raw data is unaffected. Preserve the
frozen analyzer and create a hash-linked, loader-only repair that retains all
original gate calculations and thresholds before any Gate result is emitted.

## 2026-09-15 Gate H/I result

The loader-only repair was frozen locally before Gate execution and bound the
parent contract/receipt, complete raw results, raw audit, original PAI analyzer,
predecessor loader and transitive model/hash helpers. It changes no scientific
calculation. The repaired execution completed and all output hashes revalidated.
The repair was not separately Rekor-anchored, which must be disclosed.

Gate H passed all checks. Selector gain was 0.1595866 with world-bootstrap 95%
CI [0.0764974, 0.2566752]; noise-corrected variance was 0.6692155 with CI
[0.3524630, 1.0013675]. Both C and R were represented across anchors/worlds.

Gate I failed. Full-history MSE 1.3175968 was worse than latest 0.8029238 and
constant 0.6840744. Full-minus-latest MSE improvement was -0.5146730; both
LINEAR_RIDGE and RFF_RIDGE estimates were negative. Full-minus-latest selected
utility was 0.0059408 with a simultaneous interval crossing zero, and
full-minus-best-constant utility was -0.0350749. Although full/latest decisions
disagreed on 10.9375% of anchors, those changes did not improve held-out value.

Frozen decision: stop. Do not access CONFIRM, design/promote an information-
utilization method from this lineage, or launch METHOD-TRAIN. Full report:
`docs/PAIRED_ADVANTAGE_DEV_RESULT_20260915.md`.

## 2026-09-15 latent-factor direction

User instruction: do not train a larger history model. First identify the latent
factor that actually controls C/R advantage and test whether legal sensor history
can observe it. The active hypothesis separates recoverability margin from task-
opportunity margin, but neither is yet claimed as true. Existing Gate H/I results
support the question, not the answer. The first phase is decomposition and
measurement audit on existing DEV data only; simulator geometry is permitted
only as a labeled privileged diagnostic. CONFIRM access and all actor/history-
model training remain forbidden.

## 2026-09-15 latent-factor exploratory result

Existing DEV outcomes were decomposed without accessing CONFIRM or launching
training. Operational-failure penalty accounts for 70.46% of C/R advantage
heterogeneity; task increment accounts for 29.54% and unified-contact penalty is
negligible. The dominant failure mode is C reaching the 12,000-step branch guard
before completing any task: 939 such branches, all with task increment zero.
This mode contributes -0.15283 of the total -0.18262 mean failure penalty.

Compact legal history has suggestive but inconclusive out-of-world value. Its
full-advantage MSE is 0.62142 versus 0.64480 for legal latest and 0.68407 for a
constant; history-minus-latest MSE improvement CI is [-0.04029, 0.08786], and
selected-utility gain CI is [-0.00297, 0.07308]. Legal stall/progress signatures
show the largest descriptive differences. Adding simple privileged exact-
geometry summaries does not improve held-out utility or failure prediction.
Current interpretation: controller-conditioned task reachability/stuckness is
the leading latent factor, but its legal observability is not established.
Report: `docs/LATENT_FACTOR_OBSERVABILITY_DEV_RESULT_20260915.md`.
