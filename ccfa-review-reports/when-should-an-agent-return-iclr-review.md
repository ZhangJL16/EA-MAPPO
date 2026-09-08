# ICLR 2027 Readiness Review — *When Should an Agent Return?*

## 1. Report Metadata

- Review date: 2026-09-01
- Target venue/year/track: ICLR 2027 Conference Track
- Paper title: *When Should an Agent Return? Reliable Resource-to-Go under Closed-Loop Composition Shift*
- Input materials reviewed: `paper/iclr2027/main.tex`, section/appendix sources,
  `paper/iclr2027/build/main.pdf`, build log, result sheets, repository evidence
  ledgers, and collision-filter artifacts
- Search basis: manuscript sources and appendix; frozen platform and completion
  ledgers; claim--evidence matrix; current build log; controlled collision-filter
  diagnostics; current primary literature and official ICLR 2027 policies
- Report file: `ccfa-review-reports/when-should-an-agent-return-iclr-review.md`
- Reviewer mode: standard full scientific, writing, compliance, and evidence audit
- Manuscript version / comparison range: current 2026-08-28 build and current
  2026-09-01 repository evidence; prior 2026-08-28 review used only as history
- Frozen comparison contract ID: `fixed_navigation_energy_research_contract_v1`
- Development potential and current submission readiness are scored separately.
  Red `RESULT PENDING` markers are treated as missing evidence, not as negative
  empirical results.

## 2. Desk Rejection Assessment

| Check | Finding | Verdict |
| --- | --- | --- |
| Page limit | The existing build has seven main-text pages, below the ICLR 2027 nine-page submission limit. | Pass |
| Scope | Reliable decision making, uncertainty, control, and embodied learning fit ICLR. | Pass |
| Anonymity | No author identity was found in the reviewed main source/build. Final supplementary files and repository metadata still need a release audit. | Provisional pass |
| Minimum reviewability | Every central empirical result slot is pending: navigation, Oracle headroom, learned estimator, uncertainty, Pareto, replication, and CMDP comparison. | Fail if submitted now |
| Claim integrity | The draft assigns collision authority to HOCBF, while the frozen R3 contract records 2,242 collision steps in three episodes and explicitly withholds deployment-quality collision authorization. | Fail if unchanged |
| AI-use policy | The planned statement says AI only assisted literature/citations/references. Repository history shows broader assistance in theory, methodology, experiment design, implementation, and manuscript work, all of which ICLR 2027 requires or recommends disclosing. Author rewriting does not erase that history. | Fail if unchanged |
| Prompt injection | No adversarial instructions were found in the paper materials. | Pass |
| Ethics/reproducibility | The limitations are directionally responsible, but safety wording and artifact admissibility are not yet aligned. | Major repair |

**Desk-risk conclusion:** high if submitted in the present state. The page and
scope checks pass, but absence of the core evidence and a materially incomplete
AI-use statement make the manuscript non-submittable. ICLR 2027 requires
double-blind submission, a nine-page main-text limit, and an AI-use statement;
identity leakage can trigger desk rejection
([Author Guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines)). Its
author policy expressly includes AI assistance with theory, claims, proofs,
hypotheses, methodology, experiments, implementation, interpretation, code,
artifacts, and writing in the disclosure boundary
([AI Policy for Authors](https://iclr.cc/Conferences/2027/AIPolicyForAuthors)).

## 3. Paper Summary And Contribution Map

The paper studies an irreversible continue/return decision for a resource-limited
agent. Its central proposal is to estimate Resource-to-Go at the **executed
interface**—after a downstream controller and safety filter have transformed the
nominal action—rather than at the nominal policy interface. It combines a
conditional occupancy-shift argument, executed-interface estimators, ensemble
uncertainty, and a thresholded ReturnManager. The intended empirical endpoint is
lower stranding at matched useful throughput under composition shift.

The logical chain is:

1. an action-transforming safety layer changes closed-loop occupancy and energy;
2. nominal-interface resource prediction can therefore be miscalibrated;
3. executed-interface prediction should reduce this mismatch;
4. reliability-aware switching should convert better prediction into a favorable
   stranding--throughput frontier;
5. collision safety remains the responsibility of a fixed external filter.

This is a coherent ICLR story. At present, however, steps 3--5 have no admissible
end-to-end evidence, and step 5 is contradicted by the selected platform record.

- Claimed problem: irreversible continue/return decisions under finite resource.
- Claimed gap: nominal policy variables need not identify resource demand after
  an action-transforming controller/safety filter.
- Method/contribution map: identification analysis; executed-interface
  Resource-to-Go estimators; epistemic reliability; one-way ReturnManager.
- Evidence package: local theory/proofs and detailed experiment contracts, but no
  promoted empirical headline result.
- Stated limitations: Oracle headroom may fail; SIRP may be deleted; claims may
  remain domain-specific; the learned module does not certify collision safety.

## 4. Search And Related-Work Basis

- Queries used: sampled-data robust HOCBF quadrotor; composite control barrier
  function LiDAR quadrotor; predictive sampled-data safety filter; learned CBF
  input limits; black-box safety filter.
- Sources searched: IEEE/DOI publisher records, PMLR proceedings, arXiv for
  labeled preprints, official author implementation pages, and ICLR 2027 pages.
- Closest works found: Xiao--Belta HOCBF; Breeden et al. sampled-data CBF;
  Oruganti et al. robust sampled-data CBF; Harms et al. Composite CBF;
  Wabersich--Zeilinger predictive safety filtering.
- Unverified related-work risks: the 2025 predictive sampled-data quadrotor paper
  is a preprint; no universal SOTA ordering is supportable across sensing/model
  regimes; no faithful local Composite CBF implementation was found.
- Source-quality screening status: complete for the collision-layer selection
  question; primary sources retained, snippet-only/low-authority records excluded.

The novelty and positioning check used official ICLR 2027 guidance and a current
primary-source search recorded in
`literature-search-20260901-collision-safety-layer/`. The closest clusters are:

- high-order CBFs for relative-degree constraints;
- sampled-data and robust CBFs for sample-and-hold execution;
- recent composite/LiDAR CBFs for dense quadrotor obstacle constraints;
- predictive and learned safety filters for uncertain or black-box dynamics.

The strongest fit to the current known double-integrator, bounded-acceleration,
fixed-timestep simulator is a sampled-data robust HOCBF, not a new learned
collision method. High-order CBF foundations are established by Xiao and Belta
([IEEE TAC](https://doi.org/10.1109/TAC.2021.3105491)); sampled-data conditions
are treated by Breeden et al.
([IEEE L-CSS](https://doi.org/10.1109/LCSYS.2021.3076127)); bounded uncertainty
is treated by Oruganti et al.
([preprint](https://arxiv.org/abs/2309.08050)). A strong recent quadrotor point of
comparison is Harms et al.'s Composite CBF with dense onboard LiDAR and flight
evaluation ([ICRA 2025](https://doi.org/10.1109/ICRA55743.2025.11127368)). The
repository's `aggregate_hocbf` is not a faithful implementation of that work and
must not be presented as such.

## 5. Expected Review Outcome

**Current verdict: Reject (3/10), high confidence.**

- Expected outcome: Reject in current form.
- Main accept signal: a crisp executed-interface energy-safety question with a
  falsifiable Oracle kill test and potentially useful conditional theory.
- Main reject signal: absent central experiments, contradicted collision contract,
  and noncompliant planned AI disclosure.
- Confidence: 5/5 on current readiness; 3/5 on eventual novelty.

**Development verdict:** promising research direction, but not an evidence-complete
submission. The most persuasive accept case would be a narrow paper showing that
a frozen, adequately audited collision layer induces an executed-action occupancy
shift, that executed-interface Resource-to-Go improves calibrated resource-risk
prediction, and that this improvement yields a statistically defensible
stranding--throughput Pareto gain. The current reject case is decisive: no central
experiment is populated; the frozen collision evidence contradicts hard-safety
wording; formal Oracle headroom has not passed; and the AI disclosure is
incomplete.

ICLR reviewers are asked whether the paper identifies a concrete problem, is
grounded in literature, supports its claims, and creates sufficiently significant
new knowledge; state-of-the-art numerical performance is not itself mandatory
([Reviewer Guidelines](https://iclr.cc/Conferences/2027/ReviewerGuidelines)).
Thus the paper does not need to win collision benchmarks, but it does need honest,
complete evidence for its own energy-safety claims.

## 6. Strengths And Weaknesses

### Strengths

1. **Sharp problem primitive.** The irreversible continue/return decision is easy
   to understand and matters in embodied systems with resource limits.
2. **Useful interface distinction.** Separating nominal from executed actions is a
   plausible and general way to expose safety-filter-induced composition shift.
3. **Claim-aware drafting.** Pending cells, limitations, ablation sheets, and
   theorem conditions make unsupported promotion harder.
4. **Potentially valuable negative result.** If Oracle headroom is absent, the
   project has a principled stop rule instead of forcing a learned-method claim.
5. **Compact manuscript structure.** The source is readable and currently within
   the venue's page budget.

### Weaknesses

1. **Weakness:** no empirical paper yet.

   **Evidence basis:** all discriminating result cells are `RESULT PENDING`.

   **Reviewer deduction:** claims about accuracy, reliability, headroom, and the
   Pareto frontier are unevaluable.

   **Required fix:** complete the Gate-ordered, paired evaluation and promote only
   admissible artifacts.
2. **Weakness:** collision contract is false for the selected platform.

   **Evidence basis:** ordinary HOCBF was not sampled-data robust by default in
   R3, which recorded three collision episodes and 2,242 collision steps.

   **Reviewer deduction:** the stated systems hierarchy is internally invalid.

   **Required fix:** freeze and audit a passing layer or condition/weaken the
   safety claim.
3. **Weakness:** theory is conditional on the key object.

   **Evidence basis:** exact or controlled executed-interface agreement is central
   to the stated bound.

   **Reviewer deduction:** the theorem may be non-discriminating or vacuous in the
   observed regime.

   **Required fix:** independently audit the proof and report measured interface
   errors, constants, and boundary mass.
4. **Weakness:** algorithmic novelty is fragile.

   **Evidence basis:** direct/TD prediction, ensembles, and threshold switching
   are established ingredients; SIRP is optional.

   **Reviewer deduction:** an implementation combination alone is unlikely to
   clear ICLR novelty.

   **Required fix:** isolate the executed-interface mechanism with matched
   counterfactuals and decision-relevant evidence.
5. **Weakness:** scope is too wide for the remaining evidence.

   **Evidence basis:** the plan includes multiple estimators, mechanism factors,
   Pareto analysis, replication, direct switchers, planning, and CMDP controls.

   **Reviewer deduction:** high risk of incomplete or underpowered comparisons.

   **Required fix:** retain the minimal discriminating baseline set and one
   bounded shift.
6. **Weakness:** compliance narrative is inaccurate.

   **Evidence basis:** the planned narrow AI statement conflicts with documented
   broader assistance.

   **Reviewer deduction:** material integrity and desk-risk concern.

   **Required fix:** disclose actual assistance under the ICLR 2027 policy.

## 7. Potentially Missing Related Work

**Work:** Breeden et al., *Control Barrier Functions in Sampled-Data
Systems*, and Oruganti et al., *Robust Control Barrier Functions for
Sampled-Data Systems*.

**Status:** searched.

**Why relevant:** they address the continuous-time versus sample-and-hold gap in
the implemented controller.

**Overlap:** direct overlap with the declared HOCBF execution contract, not the
energy-learning contribution.

**Needed comparison:** cite both, instantiate their assumptions, and compare
ordinary with sampled-data robust HOCBF under identical nominal actions.

**Work:** Harms et al., *Safe Quadrotor Navigation Using Composite Control
Barrier Functions* (ICRA 2025).

**Status:** searched.

**Why relevant:** strong recent dense-LiDAR quadrotor evidence.

**Overlap:** collision-constraint composition and QP filtering.

**Needed comparison:** position as a current external comparator; do not claim
that the local aggregate prototype is equivalent.

**Work:** Wabersich and Zeilinger, *A Predictive Safety Filter for
Learning-Based Control of Constrained Nonlinear Dynamical Systems*.

**Status:** searched.

**Why relevant:** backup/terminal-set alternative to local QP projection.

**Overlap:** modular runtime safety layer, not the return decision.

**Needed comparison:** explain why its model, terminal-set, and online-planning
cost is unnecessary for the current known static model
([Automatica](https://doi.org/10.1016/j.automatica.2021.109597)).

**Work:** Lavanakul et al. black-box discriminating-hyperplane filters and Liu et
al. neural CBFs under input limits.

**Status:** searched.

**Why relevant:** learned alternatives when model/barrier feasibility is unknown.

**Overlap:** filter learning and safe action constraints.

**Needed comparison:** state that the current known-model setting does not
justify this extra learning/verification burden.

**Work:** resource-aware autonomy, SOC/distance heuristics, direct switching,
planning, and CMDP baselines already cited/planned in the manuscript.

**Status:** manuscript-provided; coverage is broad but the closest exact
first-passage comparison remains a positioning risk.

**Why relevant:** reviewers must see what executed-interface prediction adds.

**Overlap:** the same final continue/return decision.

**Needed comparison:** one fair-information factorial rather than a catalog of
loosely matched baselines.

The paper should avoid “state of the art” as a singular label. There is no
universally best collision filter across known versus unknown maps, continuous
versus sampled execution, perception assumptions, and hardware constraints.
“Most applicable under the declared model and sensing contract” is the defensible
criterion.

## 8. Claim-Evidence Audit

| Claim | Where stated | Evidence provided | Strength | Reviewer deduction | Required fix |
| --- | --- | --- | --- | --- | --- |
| Nominal/executed interfaces can be non-identifiable from nominal data alone | Abstract, problem, theory | Constructive argument and appendix proof | Local/conditional | Plausible theoretical contribution, pending independent audit | Audit proof and expose scope conditions |
| Executed-interface error admits an occupancy-weighted bound | Theory and appendix | Coupling/Lipschitz derivation | Conditional | May be vacuous without measured constants/error | Report constants, boundary mass, and empirical non-vacuity |
| Executed-interface estimation is more accurate/calibrated | Abstract result slot; experiment setup | No promoted result | Unsupported | No method advantage established | Paired fixed-seed comparison with intervals |
| Better prediction improves return decisions | Method, experiments, conclusion | Oracle Gate incomplete; no learned result | Unsupported | Surrogate may have no decision value | Establish Oracle headroom, then learned advantage |
| Method improves stranding--throughput tradeoff | Abstract, introduction, results | Entire Pareto sheet pending | Unsupported | Main empirical claim absent | Predeclared paired frontier and uncertainty analysis |
| HOCBF retains hard collision-safety authority | Abstract, related work, limitations | R3 records three collision episodes and 2,242 collision steps | Contradicted | Systems contract invalid in current evidence | Replace/freeze/audit layer or condition/weaken claim |
| Result generalizes beyond one simulator | Experiment setup and conclusion | Replication pending | Unsupported | Significance remains domain-specific | Minimal second shift/domain or bounded claim |
| AI assisted only literature/reference work | AI-use statement | Repository record shows broader assistance | Contradicted | Compliance/integrity risk | Truthful ICLR disclosure |

## 9. Experiment / Benchmark / Reproducibility Audit

The repository demonstrates serious experimental governance, but governance is
not a substitute for a completed study. The continuous endurance attempt 3 is
admissible for its declared calibration test: 100/100 true-depletion endpoints,
zero censoring, calibrated capacity 304.953884, mean endurance 1,668.37 seconds,
and relative error -7.31%. It is not evidence for a learned return manager, a
Pareto improvement, or collision safety. Formal Oracle Decision Headroom remains
absent; therefore downstream learned-method evidence is not yet authorized.

The controlled safety prototype is useful diagnosis, not a certificate. It
reports zero observed collisions for sampled-data HOCBF under exact controlled
conditions, but also unsafe fallback/violation events and severe failures under
obstacle dropout and combined sensing stress. The project must declare whether
dropout is excluded by assumption or handled by a valid backup. Unit tests
(68 passed for the collision-energy filter suite) establish implementation
regression coverage only.

Minimum reproducibility repairs:

1. freeze collision code, configuration, timestep, sensing contract, initial set,
   solver/fallback behavior, and exact swept-clearance metric;
2. recalibrate energy after that freeze, because changing the filter changes
   occupancy, flight time, and energy use;
3. complete the Oracle headroom Gate on unseen seeds before training/promoting the
   learned estimator;
4. report paired seeds, sample sizes, confidence intervals, censoring, solver
   failures, fallback events, collisions, stranding, and useful throughput;
5. publish an anonymous, hash-bound artifact or provide enough pseudocode and
   configuration to reconstruct all decision rules.

## 10. Multi-Reviewer Panel

**Reviewer:** R1

**Expertise:** method and theoretical soundness

**Likely score:** 3/10

**Confidence:** 4/5

**Main positive signal:** clean nominal/executed interface decomposition.

**Main negative signal:** conditional theory has no empirical non-vacuity audit.

**Evidence basis:** theorem/appendix plus empty mechanism table.

**Score-change condition:** audited proof and measured bound terms that predict
held-out failures.

**Reviewer:** R2

**Expertise:** experiments and statistical evidence

**Likely score:** 2/10

**Confidence:** 5/5

**Main positive signal:** careful Gate and result-sheet design.

**Main negative signal:** no central experiment is admissible.

**Evidence basis:** every headline result is pending; Oracle Headroom absent.

**Score-change condition:** paired, uncertainty-aware mechanism and frontier
results after all upstream Gates.

**Reviewer:** R3

**Expertise:** novelty and related-work positioning

**Likely score:** 4/10

**Confidence:** 3/5

**Main positive signal:** interface-aware energy stopping could be distinct.

**Main negative signal:** collision, estimator, ensemble, and switcher components
are established.

**Evidence basis:** current collision-safety literature search and manuscript
method inventory.

**Score-change condition:** isolate one new, decision-relevant energy mechanism
against the closest matched alternatives.

**Reviewer:** R4

**Expertise:** robotics and collision safety

**Likely score:** 2/10

**Confidence:** 5/5

**Main positive signal:** separating collision from resource decisions is good
systems design.

**Main negative signal:** the frozen R3 record contradicts hard-safety language.

**Evidence basis:** three collision episodes, 2,242 collision steps, and an
explicit platform non-claim.

**Score-change condition:** frozen sampled-data robust layer with audited
assumptions, swept clearance, feasibility, and fallback.

**Reviewer:** R5

**Expertise:** ablation and causal evidence

**Likely score:** 3/10

**Confidence:** 4/5

**Main positive signal:** pair/interface/horizon factors are conceptually useful.

**Main negative signal:** proposed factorial and baseline menu are oversized and
empty.

**Evidence basis:** experiment setup and result sheets.

**Score-change condition:** one predeclared matched nominal-versus-executed
factorial plus a direct switcher.

**Reviewer:** R6

**Expertise:** reproducibility and artifacts

**Likely score:** 4/10

**Confidence:** 4/5

**Main positive signal:** strong ledgers, hashes, fail-closed Gates, and unit
regressions.

**Main negative signal:** no final result-complete anonymous artifact.

**Evidence basis:** repository ledgers and pending formal runs.

**Score-change condition:** promoted immutable artifacts with reconstruction
instructions and final seed/statistics audit.

**Reviewer:** R7

**Expertise:** writing and novice readability

**Likely score:** 4/10

**Confidence:** 4/5

**Main positive signal:** accessible motivation and compact structure.

**Main negative signal:** placeholders and an oversized results table prevent a
finished narrative.

**Evidence basis:** current PDF/source and visual inspection.

**Score-change condition:** result-driven rewrite with a single contribution
spine and readable tables.

**Reviewer:** R8

**Expertise:** ethics, compliance, and AC perspective

**Likely score:** 2/10

**Confidence:** 5/5

**Main positive signal:** limitations already reject some overclaims.

**Main negative signal:** planned AI disclosure is materially incomplete.

**Evidence basis:** AI-use statement, repository history, and ICLR policy.

**Score-change condition:** accurate disclosure and corrected safety claims.

**Agreement:** the direction is promising, but the present submission lacks
decision-relevant evidence.

**Disagreement:** eventual novelty of the conditional theory after results.

**Decisive positive axis:** a clean executed-interface energy mechanism.

**Decisive negative axis:** absent evidence plus integrity/safety-contract defects.

**Unresolved evidence:** Oracle headroom, estimator calibration, frontier gain,
collision contract, and bounded transfer.

**AC stance:** reject now; reassess after the minimal Gate sequence is complete.

## 11. Concerns Table

| ID | Severity | Concern | Evidence basis | Affected criterion | Fix class | Required action | Owner skill | Score-change condition |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C1 | fatal | All main empirical cells are pending | Results source and sheets | evidence | experiment | Populate only after declared Gates pass | `ccf-experiment-designer` | Complete paired mechanism/frontier evidence |
| C2 | fatal | Incomplete AI-use statement | AI statement versus ICLR policy/project history | ethics | ethics/limitations | Write truthful disclosure | `ccf-integrity-auditor` | Compliance audit passes |
| C3 | major | Hard-safety claim contradicts R3 | 3 collision episodes; 2,242 collision steps | soundness | method/soundness | Freeze a passing layer or condition the claim | `ccf-experiment-designer` | C0 contract passes |
| C4 | major | Oracle headroom absent | Completion ledger | significance | experiment | Run predeclared kill test | `ccf-experiment-designer` | Oracle lower bound clears threshold |
| C5 | major | Collision change invalidates calibration | Filter changes executed occupancy/energy | reproducibility | reproducibility | Recalibrate after freeze | `ccf-experiment-designer` | New C1 Gate passes |
| C6 | major | Theory may be non-discriminating | Conditional bound; no measured terms | soundness | method/soundness | Audit proof and report non-vacuity | `ccf-integrity-auditor` | Bound predicts observed error regime |
| C7 | moderate | Scope/baseline grid is oversized | Experiment setup | clarity/evidence | compression | Narrow to minimal discriminating suite | `ccf-paper-writer` | All retained comparisons are powered |
| C8 | minor | Appendix table is undersized | PDF visual audit | clarity | writing | Split/reformat decision table | `ccf-visual-composer` | Readable at normal zoom |
| C9 | minor | Main results table wraps poorly | PDF visual audit | clarity | writing | Redesign once metrics are final | `ccf-visual-composer` | No severe wrapping |

## 12. AC / Meta-Review

**Reviewer consensus:** worthwhile problem and potentially useful interface
distinction; current evidence is insufficient.

**Reviewer disagreement:** whether the theory becomes novel enough once its
conditions are shown non-vacuous.

**Decisive acceptance axis:** a paired, reproducible energy-decision frontier gain
caused specifically by the executed interface.

**Decisive rejection axis:** missing empirical support, contradicted collision
contract, and inaccurate disclosure.

**AC stance:** reject in current form.

**Discussion risks:** reviewers may over-credit the detailed protocol as evidence
or, conversely, dismiss the energy idea because the collision stack is invalid;
the revision must separate those two layers cleanly.

The reviewers would likely agree that the problem is worthwhile and that the
nominal-versus-executed interface distinction could support an interesting ICLR
contribution. They would also agree that the current artifact is not reviewable
as a completed scientific paper. No central empirical claim is supported, one
system claim is contradicted by the frozen evidence, and the planned disclosure
does not match venue policy. Disagreement may concern whether the conditional
theory is sufficiently novel once experiments arrive; that question cannot be
resolved until the authors show non-vacuous, decision-relevant gains. The AC
recommendation is reject now, encourage resubmission after a narrowed and fully
executed evidence program.

## 13. Quantitative Scores

### Scientific scores

| Dimension | Score / 5 | Deduction and repair condition |
| --- | ---: | --- |
| Novelty | 3 | Promising interface framing, but standard estimator/filter/switcher ingredients. Repair with one sharply isolated new mechanism and closest-work comparisons. |
| Soundness | 2 | Conditional theory and false current safety contract. Repair with proof audit, non-vacuity measurements, and a frozen collision contract. |
| Evidence | 1 | Central result slots are empty. Repair requires completed Oracle, predictor, reliability, and Pareto Gates. |
| Significance | 3 | Important problem but no demonstrated effect size or generality. Repair with a useful frontier gain and bounded transfer claim. |
| Clarity | 4 | Coherent and readable; pending cells and scope breadth interrupt the final argument. |
| Reproducibility | 3 | Strong internal ledgers, but no final anonymous executable artifact or result-complete protocol. |
| Ethics/limitations | 1 | AI-use disclosure is inaccurate and safety terminology exceeds evidence. Repair both before submission. |

- Overall recommendation: **3/10 — Reject**
- Reviewer confidence: **5/5** on current readiness; **3/5** on eventual novelty
- Rank within a typical ICLR batch: lower quartile in current form because the
  core evidence is absent, despite above-average research organization.

### Writing scorecard

Weighted score: **3.4/5 (moderate-high revision risk)**.

| Dimension | Score / 5 |
| --- | ---: |
| Story and problem framing | 4 |
| Contribution clarity | 4 |
| Paragraph/section organization | 4 |
| Claim--evidence discipline | 3 |
| Method readability | 4 |
| Experiment narration | 2 |
| Related-work positioning | 3 |
| Terminology consistency | 4 |
| Prose quality | 4 |
| LaTeX/render quality | 3 |
| Reviewer-risk control | 2 |

Static checks found no unresolved citations, references, duplicate labels,
overfull boxes, or TODO markers; the existing log contains eight underfull-box
warnings. The prose checker found no em-dash issue and one semicolon-density
advisory. The main visual problems are a heavily wrapped pending-results table
and an undersized appendix decision table.

### Compact score summary

- Quality: 2/5
- Clarity: 4/5
- Significance: 3/5
- Originality: 3/5
- Soundness: 2/5
- Evidence: 1/5
- Reproducibility: 3/5
- Ethics / Limitations: 1/5
- Overall: 3/10 — Reject
- Confidence: 5/5 on present readiness
- Score-change conditions: accurate disclosure; passing frozen collision
  contract; recalibration; Oracle headroom; paired estimator and Pareto evidence;
  independently audited non-vacuous theory.

## 14. Questions For Authors

1. Which exact collision filter, sensing assumptions, sample time, safe initial
   set, input bounds, solver behavior, and backup policy will be frozen before
   energy calibration?
2. If the Oracle headroom Gate fails, will the authors stop the learned-manager
   claim rather than search for a favorable downstream metric?
3. What is the one result that distinguishes executed-interface estimation from
   a direct switcher given the same state, action, battery, and filter signals?
4. Are perception dropouts outside the claimed operating domain, or will the
   system include a verified backup for them?
5. Will the AI-use statement disclose the actual assistance in theory,
   methodology, experiments, implementation, interpretation, and writing?

## 15. Score Revision Criteria

A score of 3/10 means the submission has identifiable merit but lacks the
evidence and integrity conditions required for conference acceptance. This is
not a prediction that the research direction will fail. It is a judgment that
the present PDF is closer to a disciplined preregistered research plan than to a
completed ICLR paper. Passing the page limit, having clean prose, or following a
sound Gate process cannot compensate for missing central evidence.

**Raising the score would require:** truthful disclosure; a valid frozen collision
contract; post-freeze calibration; Oracle headroom; non-vacuous theory; and
paired decision-frontier evidence that survives matched baselines and uncertainty
analysis.

**Lowering the score would be triggered by:** promotion of diagnostic/smoke
artifacts as formal results, unconditional safety/SOTA wording, hidden selection
over seeds or thresholds, or incomplete disclosure in the submitted version.

**Concerns unlikely to change before submission:** collision filtering and common
estimator/ensemble/switcher ingredients will remain prior art; generality will
remain bounded unless a second independent setting is completed.

## 16. Action Plan And CCFA Handoffs

**Priority:** P0

**Action:** replace the planned narrow AI statement with an accurate ICLR 2027
disclosure and audit anonymity.

**Owner skill:** `ccf-integrity-auditor` then `ccf-submission-checker`

**Input needed:** full human/AI contribution record and final submission bundle

**Expected output:** compliant disclosure and file-level anonymity report

**Handoff required:** yes

**Priority:** P0

**Action:** freeze sampled-data robust HOCBF infrastructure with explicit input,
sensing, inter-sample, initialization, solver, fallback, and swept-clearance
contract. Treat Composite CBF as an external comparator.

**Owner skill:** `ccf-experiment-designer`

**Input needed:** chosen filter/configuration, task set, sensing domain, and proof
assumptions

**Expected output:** immutable C0 contract and paired safety audit

**Handoff required:** yes

**Priority:** P0

**Action:** rerun battery/endurance calibration after the collision freeze.

**Owner skill:** `ccf-experiment-designer`

**Input needed:** passing C0 artifact

**Expected output:** hash-bound post-freeze C1 calibration

**Handoff required:** yes

**Priority:** P0

**Action:** run Oracle Decision Headroom as a kill test; stop the learned claim if
the Gate fails.

**Owner skill:** `ccf-experiment-designer`

**Input needed:** passing C0/C1 artifacts and unseen paired seed schedule

**Expected output:** formal PASS, scientific FAIL, or INCONCLUSIVE decision

**Handoff required:** yes

**Priority:** P1

**Action:** retain one executed estimator, one matched nominal estimator, a direct
switcher, SOC/distance rules, and at most one fair constrained-control comparator;
delete SIRP unless it wins its declared Gate.

**Owner skill:** `ccf-idea-optimizer` then `ccf-experiment-designer`

**Input needed:** Oracle PASS and frozen information/compute contract

**Expected output:** minimal powered comparison matrix

**Handoff required:** yes

**Priority:** P1

**Action:** prove decision relevance through calibrated first-passage prediction
and paired stranding--throughput outcomes, including censoring and failure logs.

**Owner skill:** `ccf-experiment-designer`

**Input needed:** trained estimators and preregistered paired protocol

**Expected output:** mechanism table and uncertainty-aware Pareto figure

**Handoff required:** yes

**Priority:** P2

**Action:** add one bounded composition/layout shift or explicitly restrict the
claim, then perform the result-driven manuscript and visual rewrite.

**Owner skill:** `ccf-paper-writer` and `ccf-visual-composer`

**Input needed:** promoted evidence only

**Expected output:** claim-evidence-aligned ICLR manuscript and readable tables

**Handoff required:** yes

**Checks run:** manuscript/source/result scan; frozen JSON and ledger audit;
current primary literature search; official ICLR policy check; unit-test replay
(68 passed); existing PDF/log visual/static audit; prose check; Markdown diff
check.

**Checks skipped:** fresh LaTeX compilation (no TeX executable installed), formal
proof re-derivation, and any new formal experiment.

**Unresolved risks:** collision C0 is not passed, post-freeze calibration is
absent, Oracle headroom is absent, every central empirical slot is pending, and
the current AI statement is not compliant.

The accompanying executable redesign is
`docs/ICLR_ENERGY_SAFETY_STORY_BLUEPRINT.md`.
