> **LEGACY / SUPERSEDED:** Historical research provenance only. This file is not an active method, theorem, implementation, test, or README authority. The active route is in docs/new_theory/.

# Red-Team Log

## Round 1 — Builder → Skeptic → Repair (complete; acceptance FAIL)

### Builder

The strongest coherent candidate is task-independent recursive recoverability with a continuous SAC-family policy, a separately certified frozen recovery controller, finite first-passage recovery energy, verified continuous task support, and an explicit charging/departure/resume lifecycle.

### Skeptic findings

| ID | Severity | Issue / failure mechanism | Counterexample or witness | Repair | Status |
| --- | --- | --- | --- | --- | --- |
| R1-01 | BLOCKER | The paper method was not uniquely aligned with evidence | Generator-SAC theorem versus clean Standard SB3 SAC experiments | Select Generator-SAC primary; demote SB3 to baseline/intervention | CLOSED |
| R1-02 | HIGH | Charging theorem certified a state-dependent hold but execution recorded zero | Residual velocity can require nonzero acceleration | Execute/log exact certified hold and account flight energy | CLOSED-SOFTWARE |
| R1-03 | HIGH | Persistent epoch omitted fresh runtime support identity | Static manifest could group changed snapshot supports | Composite runtime epoch plus independent manifest | CLOSED-SOFTWARE |
| R1-04 | HIGH | Dual-field contribution is decorative in current implementation | No learned collision or energy estimator exists | Define minimal proposal role, derive heterogeneity result, implement later only after theory closure | OPEN |
| R1-05 | BLOCKER | Code's `R_RL` is only a topology/cell-ID graph 1-core | It omits energy, full action-set verification, and explicit charge seed | Rename implementation candidate kernel; specify ideal energy-augmented `Phi` | CLAIM REPAIRED; CODE OPEN |
| R1-06 | HIGH | Current Generator advantage can be explained by center/runtime/data differences | Center-only matches Generator-SAC on current fixtures | Use a matched, non-saturated residual-learning experiment | OPEN |
| R1-07 | MEDIUM | 2× certificate identity is tested, but numerical cost domination is not | A matching hash can still bind a numerically insufficient bound | Add full-domain/property test linking 2× realized costs to certificate upper bounds | OPEN |
| R1-08 | MEDIUM | Teacher “benefit” bundles replay prefill and actor imitation | Guided condition changes both mechanisms | Run 2×2 factorial | OPEN |
| R1-09 | MEDIUM | `R`, `A_rec`, `A_cont` exist mostly as boolean certificates/private IDs | Code cannot directly expose theorem objects for audit | Add typed theorem-facing views or document exact sufficient certificate predicates | OPEN |
| R1-10 | BLOCKER | Endpoint-safe actions can cross an obstacle | One-dimensional step from `-1` to `1` crosses obstacle at `0` | Add current-step swept-tube predicate to `A_safe` | OPEN |
| R1-11 | BLOCKER | Energy envelope clips negative true successors | `e=0.25`, realized cost `1` gives `e+=-0.75`, excluded by `[0,inf)` clipping | Remove clipping or add precondition `e_lower >= c_bar` | OPEN |
| R1-12 | BLOCKER | E3 ignores successor increase in state-dependent reserve | Current margin holds while successor reserve increases by one | Use constant executable reserve or recurse on `E_bar+m` | OPEN |
| R1-13 | HIGH | FREE promotion lacks nearest-return/false-negative soundness | A missed nearer obstacle with a valid farther return can mark occupied space FREE | Add ray-soundness calibration premise | OPEN |
| R1-14 | HIGH | Publication/bus failure is described as kappa takeover | Failed fallback write can leave the previous task command active | Separate publisher failure as fail-closed/unresolved | CLAIM REPAIR OPEN |
| R1-15 | HIGH | Charging changes the transition kernel | Base flight envelope cannot contain energy-increasing successor | Define and verify `Post_hat_charge` | OPEN |
| R1-16 | HIGH | Old invariance quantifier exceeds certificate lifetime | Certificate expires at finite `T` | State theorem only to `tau_valid` | CLAIM REPAIR OPEN |

### Repair state

The primary method, hold execution, and composite epoch repairs are landed. Targeted regressions passed (43 tests). The canonical master now distinguishes exact versus rectangular recovery energy, `A_rec` versus `A_safe`, ideal state-level `Phi` versus the implemented graph kernel, and true `Post` versus `Post_hat`. Round 1 still fails because foundational theorem blockers remain.

### Round-1 score (provisional; 0–5)

| Axis | Score | Reason |
| --- | ---: | --- |
| Mathematical coherence | 2 | Exact/upper objects can be repaired, but old statements still conflict |
| Proof correctness | 0 | Three global counterexamples invalidate headline invariance |
| Assumption transparency | 4 | Existing docs are unusually explicit about synthetic/calibration limits |
| Method/implementation alignment | 3 | Charging and epoch software mismatches repaired; atlas fixed point still mismatched |
| Method/experiment alignment | 2 | Method selected, but learned residual remains unsupported |
| Dual-field necessity | 0 | No theorem, counterexample, or implementation |
| Novelty differentiation | 1 | Closest-work search incomplete |
| Falsifiability | 4 | Many matched tests can be stated concretely |
| Paper simplicity | 2 | Existing package contains multiple overlapping theory routes |
| Claim discipline | 4 | Most artifact docs already avoid physical overclaiming |

Acceptance: **FAIL / critical_gap**. Same-family fresh proof audit found 3 FATAL, 5 CRITICAL, and 5 MAJOR issues; no counterexample was found for the frozen constrained soft-operator contraction or affine-tanh density algebra under their explicit hypotheses.

## Round 2 — Recoverability and lifecycle Builder → Skeptic → Repair (complete; conditional reframe)

### Builder

Use the pending-task-indexed hybrid invariant `I_q`, exact joint robust first-passage cost `J^kappa`, a versioned upper certificate `E^kappa`, lower battery energy `e^-`, pure predecessor `A_rec`, full `A_safe`, an ideal energy-augmented viability operator, and a separate task-state invariant.

### Skeptic and repair

| ID | Severity | Counterexample / finding | Repair | Status |
| --- | --- | --- | --- | --- |
| R2-01 | BLOCKER | Zero-cost self-loop satisfies `E=E` for every finite value but never reaches charging | Require properness; non-hitting paths cost infinity | CLOSED-IN-SPEC |
| R2-02 | HIGH | Outcomes `(10,g)` and `(0,h)` with `E(h)=100` give exact joint value 100 but separated upper 110 | Separate exact `J^kappa` from rectangular upper `E^kappa` | CLOSED-IN-SPEC |
| R2-03 | BLOCKER | Nominal battery 5 with radius .1 falsely passes requirement 5 | Define `R` with lower energy `e^-` | CLOSED-IN-SPEC |
| R2-04 | BLOCKER | Cell-ID kernel admits an exact-energy-boundary state without positive-volume normal support | Rename implementation candidate domain | CLAIM CLOSED; CODE OPEN |
| R2-05 | HIGH | Support transitions wholly to `G_charge` but has no cell edge; graph pruning removes it | Include terminal seed in ideal `Phi` | CLOSED-IN-SPEC |
| R2-06 | HIGH | Hashed terminal certificate can carry nonzero recovery energy and still be `.valid` | Validate finite exact zero | CODE OPEN |
| R2-07 | HIGH | Position/velocity hold check can pass while energy exits charging set | Verify complete hybrid successor | OPEN |
| R2-08 | HIGH | Safety authority has no task ID; scheduler can replace task at station | Prove and log `q^+=q` separately | OPEN |

### Score and verdict

Ideal-spec mathematical coherence: 4/5. Method/code equivalence: 2/5 because the atlas implements only a weaker graph kernel. Verdict: **COHERENT AFTER REFRAMING; verbatim package rejected**.

## Round 3 — Three-role hostile ICLR review (complete; reject)

Independent theory, novelty, and evidence reviewers each scored the current bundle **3/10 Reject**.

| Lens | Fatal axis | Repair adopted now | Remaining score gate |
| --- | --- | --- | --- |
| Theory | graph kernel is not state-level `Phi`; overlapping-cell minimum rank invalid; hybrid lifecycle open | selected-certificate rank; terminal zero-energy validation; energy-preserving charge-set check | state-level refinement or remove `R_RL` from theorem; complete `I_q` proof |
| Novelty | generator masking and backup recoverability already exist | closest-work matrix expanded; primitive novelty wording forbidden | non-assembly regeneration theorem/counterexample |
| Evidence | no learned residual/field/teacher contribution; train/eval provenance mixed | ablation script no longer overwrites evaluation rows; gate renamed artifact completeness | matched non-saturated Generator study and completed factorial |

Round-3 verdict: **not ICLR-ready**. The audit succeeded in preventing unsupported submission claims; it did not manufacture evidence or declare unresolved proofs complete.

## Round 4 — Fresh canonical-theorem blind audit (repair recorded; closure in Round 5)

A fresh same-family ultra reviewer read only the canonical theorem and proof dependencies. The pre-repair statement received **FATAL 1 / CRITICAL 8 / MAJOR 4 / MINOR 1** and therefore invalidated the earlier “no known internal FATAL/CRITICAL” disposition.

| ID | Severity | Counterexample / gap | Repair now in `PAPER_THEOREM.md` | Status |
| --- | --- | --- | --- | --- |
| R4-01 | FATAL | `x in R\\G_ch` paired with initial mode `CHARGING` satisfied the old product-domain quantifier | Mode-compatible domains `I_N`, `I_B`, `I_C` | REPAIRED; RE-REVIEW PENDING |
| R4-02 | CRITICAL | Selected rank did not quantify every robust successor, committed child, terminal seed, or child persistence | Finite selected proof DAG with exhaustive `sigma_nu(x+)`, strict child rank, and rank-zero subset of `G_ch` | REPAIRED; RE-REVIEW PENDING |
| R4-03 | CRITICAL | An arbitrary supersolution need not satisfy `E_i <= i c_max` | Removed the false upper and retained only `J^kappa <= E_nu` | REPAIRED; RE-REVIEW PENDING |
| R4-04 | CRITICAL | Realized cost and successor lower energy were not connected to the stored recursion | Added `d_t<=d_bar`, lower-energy propagation, and the explicit three-line reserve inequality | REPAIRED; RE-REVIEW PENDING |
| R4-05 | CRITICAL | Kappa, charge, and hold lacked complete current-tube coverage | Every covered branch now includes complete collision and energy tubes | REPAIRED; RE-REVIEW PENDING |
| R4-06 | CRITICAL | Endpoint charge can hide an intra-step energy deficit | Complete charge schema requires a nonnegative lower-energy prefix | REPAIRED; RE-REVIEW PENDING |
| R4-07 | CRITICAL | Departure mode reset and noncompletion/backup task transitions were asserted but not defined | Exhaustive `RUN/KAPPA/CHARGE/HOLD/DEPART` relation with atomic departure reset | REPAIRED; RE-REVIEW PENDING |
| R4-08 | CRITICAL | Certificate expiry could precede finite recovery and a latent lost premise was treated as an observable stopping time | Pre-publication full-interval `Covered(t)`, horizon `tau_cov`, and conditional `r_0`-step arrival | REPAIRED; RE-REVIEW PENDING |
| R4-09 | CRITICAL | “valid/verified/covered” was circular | Primitive finite `Verify_t` schema separated from external physical soundness | REPAIRED; RE-REVIEW PENDING |
| R4-10 | MAJOR | Dependency graph omitted terminal seed, selected child, energy recursion/tubes, mode/task, and timing edges | Rewritten proof skeleton and claim dependency graph | REPAIRED; RE-REVIEW PENDING |
| R4-11 | MAJOR | Notation drift used a state-dependent reserve and treated departure as a mode | Constant `m_e`; `DEPART` is only a transition label | REPAIRED; RE-REVIEW PENDING |

At the end of Round 4 the theorem remained `PROVISIONAL`; Round 5 below records the required independent re-review. Software regressions and synthetic manifests cannot substitute for semantic review or physical calibration.

## Post-review repair disposition

The canonical theorem excludes `R_RL`, targets certified departure into `R`, uses a selected-child descent DAG, requires exact-zero terminal recovery energy, includes complete charge/hold tube coverage, and stops at the complete-step horizon `tau_cov`. Round 5 confirms the repair under the displayed premises.

This does not reverse the paper decision. The full bundle remains **3/10 Reject / not ICLR-ready** because the primary learned Generator residual, the dual proposal fields, autonomous charging, and the recovery-teacher mechanism lack completed matched evidence. The optional energy-augmented state-level `R_RL` capability is also unimplemented and cannot be advertised.

## Round 5 — Formula normalization and final proof closure (complete)

### Builder

Use a single selected-node invariant: finite nonnegative realized recovery consumption `d_t`, its node upper `d_bar_nu`, the outward-rounded stored upper `E_nu`, lower battery energy `e^-`, and directly verified complete action supports bound to certificate records `Xi`.

### Skeptic and repair

| ID | Severity | Counterexample / finding | Repair | Status |
| --- | --- | --- | --- | --- |
| R5-01 | LOW | Allowing negative “consumption” makes `J_cov` leave its declared nonnegative codomain | Type `d_t`, `d_bar_nu`, and `E_nu` as finite nonnegative scalars | CLOSED |
| R5-02 | LOW | Reusing `C` for both certificate object and action support makes `Post_hat_C(x,C)` ambiguous | Rename certificate records `Xi` and supports `S` | CLOSED |
| R5-03 | LOW | A graph arrow could be read as constructing `E` from `J_cov` | Show stored recursion and exact cost as parallel objects feeding the domination lemma | CLOSED |

The fresh final blind pass tried 1-D obstacle crossing, negative/zero cost, truncated coverage, overlapping children, unsafe full-battery departure, endpoint-only charge safety, task mutation, and unsafe-full-support/safe-singleton cases. No counterexample survived the explicit premises. All Round-4 FATAL/CRITICAL/MAJOR rows are therefore closed by re-review; their historical “pending” labels record the state at the end of that round.

### Round-5 score (0–5)

| Axis | Score | Reason |
| --- | ---: | --- |
| Mathematical coherence | 5 | One canonical narrow theorem and stable object semantics |
| Proof correctness | 5 | Zero open proof-checker issues under declared premises |
| Assumption transparency | 5 | Physical calibration/publication premises and non-claims are explicit |
| Method/implementation alignment | 4 | Runtime/replay contracts pass; optional state-level `R_RL` remains intentionally unimplemented |
| Method/experiment alignment | 2 | Main learned Generator/field benefits remain unevidenced |
| Dual-field necessity | 3 | Distinct operators and constructive ordering loss are specified; empirical benefit is open |
| Novelty differentiation | 3 | Primitive novelty is disclaimed; regenerative composition remains vulnerable |
| Falsifiability | 5 | Every retained claim has a failure criterion or artifact plan |
| Paper simplicity | 4 | Canonical theorem is narrow; historical archive remains large but superseded |
| Claim discipline | 5 | Levels 0–4 and strict non-claims are maintained |

Round-5 internal-theory verdict: **PASS, VERIFIED-CONDITIONAL, same-family/provisional**. Full ICLR-readiness remains unresolved because evidence and novelty gates are independent of proof correctness.

## Round 6 — Post-repair hostile panel, publication race, and closest-work attack (repair complete; final re-review pending)

### Builder

The post-Round-5 candidate binds every positive authority claim to the command and complete certificate actually valid at atomic publication, not to a preview decision. Its current-step action certificate includes both the complete swept hull and the within-step lower-energy prefix, and departure evaluates the same lower battery quantity used by recoverability.

### Skeptic findings and repairs

| ID | Severity | Counterexample / finding | Repair | Status |
| --- | --- | --- | --- | --- |
| R6-01 | BLOCKER | A valid `KAPPA_BACKUP` preview followed by certificate loss in `step_recovery` executed an uncertified emergency brake while preserving `command_source=kappa` and stale KAPPA authority | Derive authority from final `covered_at_publication`; publish the fallback as `uncertified_emergency_brake`, record `FAIL_CLOSED`, terminate, use zero bootstrap, exclude kappa metrics and child commitment | CLOSED-SOFTWARE; FINAL INDEPENDENT RE-REVIEW PENDING |
| R6-02 | HIGH | The theorem required current swept-tube containment, but the executable action certificate hash bound only successor membership | Add a complete current swept-hull inclusion predicate and a thin occupied-gap counterexample with safe endpoints | CLOSED for the synthetic straight-segment collision contract |
| R6-03 | HIGH | Endpoint lower energy could be safe while a within-step lower prefix was unsafe | Add a hash-bound within-step energy-prefix predicate and a direct prefix counterexample | CLOSED for the synthetic transition contract |
| R6-04 | HIGH | Departure used nominal plant energy even though `R` and the theorem use certificate lower energy | Evaluate the route gate with `e^- = e-rho_e` and add a nominal-pass/lower-bound-fail regression | CLOSED-SOFTWARE |
| R6-05 | BLOCKER for broad novelty | Persistification already combines battery-state augmentation, safety-constrained nominal control, charging regions, and long-duration task execution; LDCBF work combines finite-duration safety learning with charger abandonment/hold | Remove all first-persistent-task/charging wording and narrow the candidate to robust complete-support certificate plus publication-authority/task-identity composition | CLOSED-AS-CLAIM-REPAIR; NOVELTY SCORE GATE REMAINS |
| R6-06 | BLOCKER for paper evidence | All three hostile reviewers still scored the bundle 3/10 because learned Generator, dual-field, teacher, and autonomous-charging benefits are not established | Retain these as hypotheses/supporting mechanisms and require matched studies before empirical promotion | OPEN EXTERNAL-EVIDENCE GATE |

### Panel disposition before the R6 repairs

The safe-control reviewer found R6-01 and reproduced it with an injected preview/recheck race. The off-policy reviewer found no fatal flaw in the narrow affine-tanh Generator-SAC density or branch targets, but rejected the submission because the executed learned method and contribution evidence remain weak. The skeptical generalist found no fatal internal flaw in the canonical conditional theorem, but rejected the broad novelty framing and requested the two closest persistent-autonomy comparisons. Each reviewer assigned **3/10 Reject**.

### Round-6 score after repair, before final independent re-review (0–5)

| Axis | Score | Reason |
| --- | ---: | --- |
| Mathematical coherence | 5 | Publication-time authority, complete action predicates, and lower energy now use one invariant |
| Proof correctness | 4 | Canonical theorem remains coherent, but its audit hash is stale after ledger edits |
| Assumption transparency | 5 | Synthetic versus physical tube, publication, and calibration premises are explicit |
| Method/implementation alignment | 4 | The reproduced fatal race is repaired and the 436-test current snapshot passes; independent final re-review is pending |
| Method/experiment alignment | 2 | No matched learned-contribution result was created |
| Dual-field necessity | 3 | H0/H1 support heterogeneous outputs, not separate-network necessity or benefit |
| Novelty differentiation | 2 | Broad lifecycle novelty is refuted; the narrow composition remains vulnerable to an assembly objection |
| Falsifiability | 5 | Race, tube, prefix, energy, authority, and causal claims have explicit tests or failure criteria |
| Paper simplicity | 4 | One primary method and one narrow theorem remain, but optional mechanisms still burden the story |
| Claim discipline | 5 | Unsupported first/benefit claims are explicitly forbidden |

Round-6 disposition: **internal repair implemented and full regression passed, but final review is not yet closed**. The current snapshot passes 436 tests; the first current-hash proof pass found only two cosmetic documentation issues, which were repaired and sent to a fresh hash-closure audit. Three independent final hostile reviews are still required. The full paper remains not ICLR-ready while learned evidence and novelty differentiation are open.

## Round 7 — Complete publication/commit state machine (repairs implemented; full regression and final panel pending)

### Builder

Positive software authority is now derived from the final published source plus `covered_at_publication`. Preview authority only selects a candidate path. Proof-child commitment is permitted exactly for finally covered task or kappa publication; preclassified fail-closed and unavailable-hold paths bypass recovery evaluation.

### Skeptic findings and repairs

| ID | Severity | Counterexample / finding | Repair | Status |
| --- | --- | --- | --- | --- |
| R7-01 | BLOCKER | An `RL_GENERATOR` preview followed by final recovery-certificate loss executed the stale fallback, reported KAPPA, remained nonterminal, and advanced the selected child | Reclassify every uncovered final source as `FAIL_CLOSED`, terminate, exclude kappa metrics, and gate commitment on final covered task/kappa source | CLOSED by targeted injection and 440-test regression; re-review pending |
| R7-02 | BLOCKER | Direct `step_recovery` correctly reported fail-closed after recheck loss but still called `commit_execution(..., False)`, advancing `active_cell_id` | Commit only when the final recovery decision is certified | CLOSED by active-cell/no-call and 440-test regressions; re-review pending |
| R7-03 | BLOCKER | A preview already classified `FAIL_CLOSED` called `step_recovery`; a newly valid kappa could resurrect positive authority and commit a child before the outer label was overwritten | Add a dedicated `step_fail_closed` path that never evaluates or commits recovery | CLOSED by valid-kappa-under-fail-closed and 440-test regressions; re-review pending |
| R7-04 | HIGH | An unavailable charger hold returned incomplete task/goal/energy metadata and could enter replay inconsistently | Route it through dedicated fail-closed execution and emit the complete persistent transition contract | CLOSED by runtime→transition→replay→update and 440-test regressions; re-review pending |
| R7-05 | BLOCKER | Certificate version could change during candidate construction; watchdog rejected the task bundle but published the old recovery action as certified kappa and retained its hash | Treat a live snapshot mismatch as uncovered publication, execute the emergency action, clear the recovery hash, terminate, and commit no child | CLOSED by watchdog, direct-runtime, acceptance, persistent-wrapper, and 440-test regressions; re-review pending |
| R7-06 | MEDIUM | Acceptance logic called every fallback `kappa`, and generic authority failure erased the concrete sensing/energy/version cause | Record `command_source`/coverage explicitly and preserve the concrete `fallback_reason` separately from fail-closed authority category | CLOSED by the complete failure matrix and 440-test regression |
| R7-07 | LOW | The audited theorem ledger still contained prose saying its own hash audit was stale | Replace time-dependent prose with a stable pointer to exact hash-specific audit provenance | CLOSED in source; fresh proof hash audit pending |
| R7-08 | HIGH | The formal persistent trainer discarded `next_context` on time-limit truncation, so `bootstrap_on_truncation=True` was multiplied by a synthesized `FAIL_CLOSED` successor and zeroed rather than using the real post-step authority branch | Capture the real successor certificate/authority context before reset for every nonterminal transition; drop it only on true termination | CLOSED by focused transition/Bellman regression; complete-suite rerun and final off-policy re-review pending |
| R7-09 | MEDIUM | `GeneratorSACConfig` and the theory selected normalized-coordinate temperature as primary, but the formal training CLI silently defaulted to physical coordinates | Derive the CLI default directly from `GeneratorSACConfig().temperature_coordinate`; retain physical coordinates only as an explicit ablation | CLOSED in source and focused temperature tests; final off-policy re-review pending |
| R7-10 | BLOCKER | A KAPPA preview could be followed by a version-changing successful reprepare; the new recovery action was published with the old snapshot tag and its child was committed | Require `prepared_snapshot == initial_snapshot == live_snapshot` before certified KAPPA publication; otherwise publish the uncovered emergency action, terminate, and commit nothing | CLOSED by the exact successful-reprepare mutation counterexample; complete-suite rerun and final panel pending |
| R7-11 | BLOCKER | `step_nominal_action` had the same old-tag/new-certificate race and could publish `task`, report coverage, and commit after a version-changing reprepare | Apply the same three-way snapshot equality gate to both nominal acceptance and recovery fallback | CLOSED by direct nominal mutation injection; complete-suite rerun and final panel pending |
| R7-12 | BLOCKER | The charger-hold branch verified a hold and then unconditionally reported `covered_at_publication=True`; a version change between preview and motion had no final binding check | Bind the preview epoch to initial/live full snapshots immediately before motion and fail closed before calling charging dynamics on any mismatch | CLOSED by a hold-verifier mutation injection with zero charger motion; complete-suite rerun and final panel pending |
| R7-13 | BLOCKER | The main RL watchdog used `state.snapshot` as its live callback. Geometry changes were visible through a mutable reference, but copied calibration/kappa `bound_versions` were frozen in that old state; an actor-time energy-version mutation still published task authority and committed | Rebuild the complete certificate state inside every watchdog freshness callback | CLOSED by parameterized sensor/dynamics/tracking/energy/terminal/kappa mutations, all fail-closed with no commit; complete-suite rerun and final panel pending |
| R7-14 | BLOCKER | Watchdog publisher-stage conflict returned KAPPA before `snapshot_is_current()`. A pre-staged emergency plus stale live version therefore became certified kappa without running the producer | Treat every stage conflict as uncovered, check freshness before choosing the reason, and publish the supplied emergency action; an already-committed publisher is also returned as uncovered rather than positive authority | CLOSED by the exact stale emergency-stage conflict test; complete-suite rerun and final panel pending |
| R7-15 | BLOCKER | An accepted normal `RUN` ending inside the station returned mode `CHARGING`, but the exhaustive theorem required every `RUN` successor to remain `NORMAL` | Add `RUN-ARRIVE`: a live successor-bound station certificate proves `x+ in R intersection G_ch`; the mode update is zero-duration, uses the RUN task rule, and receives no same-step charge | REPAIRED in runtime/theorem/tests; fresh proof audit and final panel pending |
| R7-16 | BLOCKER | Open-gate `I_C` without a Generator, or rank-zero `I_N` after refusal, was routed to a fictitious KAPPA command and could charge/commit | Require positive rank and a non-charging source for κ; closed charging admits only verified CHARGE/HOLD, open charging only verified DEPART, otherwise fail-closed; plumb this rule through actor-invalid, early fallback, and watchdog paths | REPAIRED by pure and executable attacks; complete suite/final panel pending |
| R7-17 | BLOCKER | Collector stored cached pre-step epoch/`c,G`, while runtime could publish under a different re-prepared support; returned `o'` was also built before persistent time/mode/charge updates | Merge actual publication `action_context` into every transition, validate `candidate=c+G tanh(u)`, refresh successor context before rebuilding `o'`, and keep current versus next margins separate | REPAIRED; focused publication/normal/KAPPA/hold/truncation tests pass; complete suite pending |
| R7-18 | BLOCKER | The early task-closure-failure branch published certified κ without initial/prepared/live equality | Apply the same full-snapshot freshness gate before early fallback; stale execution publishes uncovered emergency, terminates, and commits no child | CLOSED by exact mutation attack and independent reviewer reproduction; complete suite pending |
| R7-19 | BLOCKER | A calibration version or hash changed before all snapshots, so snapshot equality held while the atlas recovery cell was stale | Bind manifest-cache keys and every recovery evaluation to construction/live dependency versions and calibration fingerprints; reject stale cells/gates before κ authority | REPAIRED by pre-step energy-version and same-version hash-replacement regressions; complete suite pending |
| R7-20 | BLOCKER | Certificate snapshots omitted validity time; actor-time timestamp expiry left the task command labelled covered | Add timestamp to `CertificateStateSnapshot` freshness while excluding elapsed time from replay grouping; expiry mutation now fails closed before publication | REPAIRED by exact mid-cycle expiry regression; complete suite pending |
| R7-21 | BLOCKER | `RUN-ARRIVE` checked a station hold after motion, then entered charging after an unbound dependency drift; the old `Covered(t)` definition also required this postinterval classifier before command publication | Put station verification, final snapshot readback, and mode commit in one transaction; split `CmdCovered_t` from `HybridCovered_t`, with `INVALID` ending only the hybrid continuation at that index | CLOSED by exact dependency-drift/no-charge regression and independent theorem reread; complete suite/proof audit pending |
| R7-22 | BLOCKER | A positive-rank κ child whose state overlapped the nominal terminal was discarded: the task wrapper entered charging and atlas refresh cleared the committed child, violating strict selected-rank descent | Remove geometry-only backup arrival; hash-bind and read back the declared child, preserve every positive-rank commitment under terminal overlap, and enter charging only for the exact committed level-zero child plus fresh hold | CLOSED by positive-overlap and level-one-to-zero executable regressions; complete suite pending |
| R7-23 | BLOCKER | Recovery commitment identity was absent from snapshots, so `active_cell_id` could change after a valid readback without changing the physical snapshot | Include recovery-active flag and active edge/cell/rank/hash in `CertificateStateSnapshot`; validate realized membership and child hash, then compare and commit the successor mode transactionally | CLOSED by active-child-clearing injection; complete suite pending |
| R7-24 | BLOCKER | Watchdog and direct branches checked freshness and then separately wrote the command register, leaving a check-to-publication drift window | Add a shared-lock compare-and-publish primitive with pure double snapshot readback; route task/κ/nominal/direct-recovery publication through it and serialize all normal in-scope certificate writers | CLOSED at scoped single-controller software level by exact check/commit mutation test; hardware/RTOS/bus atomicity remains external |
| R7-25 | BLOCKER | Certified charger HOLD performed its final snapshot comparison and then executed motion/charging outside the shared transaction | Execute hold verification, live readback, physical hold, and charging update inside the certificate-state transaction | CLOSED by source inspection and independent reviewer replay; complete suite pending |

### Round-7 score after targeted repairs, before final gates (0–5)

| Axis | Score | Reason |
| --- | ---: | --- |
| Mathematical coherence | 5 | Canonical theorem is unchanged; authority/commit semantics now implement its stopping boundary |
| Proof correctness | 5 | Round-8 exact-hash review returned zero findings and stable pre/post hashes |
| Assumption transparency | 5 | Uncovered emergency execution remains explicitly outside the positive theorem |
| Method/implementation alignment | 4 | All reproduced authority races pass targeted and 440-test regressions; final hostile review remains |
| Method/experiment alignment | 2 | Learned Generator/field/teacher benefits remain unevidenced |
| Dual-field necessity | 3 | Structural heterogeneity is defensible; empirical value remains open |
| Novelty differentiation | 2 | The narrow composition still faces the known-parts objection |
| Falsifiability | 5 | Preview, recheck, version, commit, replay, metric, and metadata failures are injected directly |
| Paper simplicity | 4 | One final authority law replaces branch-specific implicit behavior |
| Claim discipline | 5 | No safety is claimed for fail-closed emergency motion or synthetic tests |

Round-7 disposition: **the publication/commit repairs passed the complete 440-test snapshot; the later truncation/default repair passed a focused 38-test suite, and KAPPA, nominal, charger-hold, stage-conflict, plus six copied-bound-version races pass exact focused injections. A new complete-suite rerun is still required. Two fresh 2× full-domain audits are byte-identical and match the versioned artifact. Internal closure remains pending three post-repair hostile reviews.** External learned-evidence and novelty gates remain open regardless of software closure.
