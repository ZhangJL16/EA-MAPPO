# ICLR 2027 Cynical Review: When Should an Agent Return?

## 1. Report Metadata

Review date: 2026-08-28  
Target venue/year/track: ICLR 2027, conference paper  
Paper title: *When Should an Agent Return? Reliable Resource-to-Go under Closed-Loop Composition Shift*  
Input materials reviewed: full pre-results TeX manuscript, proofs, experimental protocol, result sheets, BibTeX, rendered PDF, claim/evidence and pending-result ledgers, repository research/proof protocols, and the user-provided Keogh SIGKDD 2009 research tutorial  
Search basis: official ICLR 2027 rules; official PMLR, NeurIPS, OpenReview, and IEEE/DOI records; targeted nearest-work search  
Report file: `ccfa-review-reports/2026-08-28-when-should-an-agent-return-iclr-conference-review.md`  
Reviewer mode: standard full scientific review of an explicitly pre-results draft; scores reflect evidence present now, not promised evidence

## 2. Desk Rejection Assessment

- Paper length: PASS. Main text ends on page 7; the initial-submission limit is 9 pages.
- Topic compatibility: PASS. The problem fits RL, world models, uncertainty, and autonomous decision making.
- Minimum quality: PASS for reviewability, but not for acceptance. The manuscript is coherent and technically inspectable.
- Policy/anonymity/compliance: BLOCKED only by the unconfirmed AI-use wording; static anonymity, template, PDF metadata, and font checks pass.
- Prompt injection and hidden manipulation detection: PASS. No hidden reviewer instructions, adversarial text, or manipulation was found.
- Ethics and reviewability: PASS with an important caveat: all authors must verify the AI disclosure and the eventual artifact licenses/provenance.

## 3. Paper Summary And Contribution Map

The paper studies when a finite-resource autonomous agent should irreversibly stop productive work and return for replenishment. Its central chain is policy--safety-operator composition shift, executed-interface extrapolation, Resource-to-Go error, return-boundary error, and then either stranding or premature return. The formal contribution separates novelty of a component pair from identification of the executed state--action interface, gives a conditional Wasserstein error-accumulation result with an SSP truncation term, and localizes threshold-decision changes to a boundary band. The proposed empirical framework compares direct prediction, policy-conditioned models, executed-action world models, ensembles, classical planning, resource-safety heads, and end-to-end constrained control under a common one-way ReturnManager. Collision authority remains with HOCBF. Every empirical result is presently an empty, explicitly marked evidence slot.

Claimed problem: decide when to return under resource constraints when a hard safety operator can rewrite the task policy's actions.  
Claimed gap: policy/filter identity coverage need not equal coverage of the executed primitive interface, and prediction quality must be judged at an irreversible decision boundary.  
Method/contribution map: structural identification/non-identification results; finite-horizon transport plus SSP tail; return-boundary stability; modular predictor/reliability/manager protocol; gated SIRP candidate.  
Evidence package: complete local proofs and a detailed preregistered experiment design, but no completed empirical result.  
Stated limitations: exact interface agreement is an assumption, transport regularity can fail at filter discontinuities, deterministic simulation does not justify physical tails, SIRP must be deleted if ensembles suffice, and broad claims require replication.

## 4. Search And Related-Work Basis

Queries used: policy-conditioned environment models; executed-action safety filters; future-resource safety networks; UAV return/recharge planning; density-constrained charging RL; world-model uncertainty; selective prediction and conformal shift.  
Sources searched: PMLR, NeurIPS proceedings, OpenReview publication records, IEEE Xplore/DOI, and DBLP for discovery with primary-page verification.  
Closest works found: Guo & B{\"u}rger's Predictive Safety Network; Chen et al.'s PCM; Qin et al.'s DCRL; Choudhry et al.'s UAV energy-risk model; Mathew et al.'s persistent-UAV recharge planning; executed/compositional world models and Deep Ensembles.  
Unverified related-work risks: broader battery-aware model-predictive control and field-robot return-to-base literatures may contain closer system-level comparators; this search was targeted, not a systematic review of every robotics venue.  
Source-quality screening status: PASS for all 23 cited entries; guessed and snippet-only records were excluded.

## 5. Expected Review Outcome

Expected outcome: reject at the current evidence state.  
Main accept signal: a crisp and unusually honest structural distinction between pair identity, executed-interface support, long-horizon error, and the downstream stopping boundary.  
Main reject signal: the paper has no empirical evidence for the mechanism or utility that motivates its ICLR submission, while the identification theorem is close to a formalization of an exact-agreement assumption.  
Confidence: 5/5 about the current draft; uncertainty concerns future results, not the present evidence gap.

## 6. Strengths And Weaknesses

### Strengths

- The manuscript does not convert failed pilots, live runs, or smoke tests into results.
- The probability semantics are disciplined: deterministic Resource-to-Go plus epistemic reliability, no fabricated physical q95/q99 interpretation.
- The safety boundary is clear: HOCBF remains collision authority and the learned module is not a certificate.
- The protocol contains strong falsification gates: Oracle headroom, pair/interface/horizon controls, risk--coverage, common ReturnManager, direct switching, constrained control, and second-setting replication.
- The first-page figure and main result shells make the causal chain and missing evidence visible.

### Weaknesses

Weakness: no empirical result exists.  
Evidence basis: Abstract, Results, Conclusion, and all appendix result sheets are explicitly marked pending.  
Reviewer deduction: the central claims of practical relevance, mechanism, and frontier improvement are untested; promises receive no acceptance credit.  
Required fix: complete the gate chain and insert provenance-backed multi-seed/multi-cycle results, including failures.

Weakness: the positive identification theorem risks being read as tautological.  
Evidence basis: Section 4 assumes every observationally compatible primitive agrees with the true primitive on every target executed interface and concludes the target law is identified.  
Reviewer deduction: the theorem cleanly exposes the right object, but exact target-interface agreement is already almost the definition of identification; novelty must not rest on the proof alone.  
Required fix: motivate why this criterion changes experimental design, provide a checkable finite-sample proxy or impossibility boundary, and ensure empirical factorial results distinguish pair identity from interface risk.

Weakness: the confirmed algorithmic novelty is weak if the gated SIRP candidate is deleted.  
Evidence basis: Section 6 otherwise combines direct/TD prediction, executed-action world models, Deep Ensembles, and a threshold manager.  
Reviewer deduction: if executed-WM plus ensemble succeeds, the contribution becomes theory plus benchmark/protocol rather than a new method; that can still work, but the paper must be sold accordingly.  
Required fix: precommit to the paper category implied by results and do not preserve SIRP for novelty theater.

Weakness: rare-event and domain claims are expensive and currently unresolved.  
Evidence basis: one primary deterministic UAV simulator, pending second setting, and no calibrated cycle count yet.  
Reviewer deduction: a visually attractive Pareto curve from 100 cycles would not establish low stranding probability or broad autonomy relevance.  
Required fix: preregister precision-driven sample sizes, run a materially different filter/domain, and report domain-specific conclusions if replication fails.

## 7. Potentially Missing Related Work

Work: Predictive Safety Network for Resource-constrained Multi-agent Systems.  
Status: searched and added.  
Why relevant: it already combines future-resource prediction with a hierarchical resource-safety policy above task policies.  
Overlap: closest practical architecture.  
Needed comparison: adapted predictive-safety head under the same ReturnManager and information contract; now present in the planned baseline matrix.

Work: Density Constrained Reinforcement Learning.  
Status: searched and added.  
Why relevant: it contains remaining-energy and charging-station constraints in an electric-vehicle setting.  
Overlap: direct end-to-end resource-aware constrained control.  
Needed comparison: include when density semantics can be matched; otherwise state the incompatibility explicitly.

Work: Multirobot Rendezvous Planning for Recharging in Persistent Tasks.  
Status: searched and added.  
Why relevant: establishes classical persistent-UAV recharge scheduling.  
Overlap: replenishment timing and throughput under limited endurance.  
Needed comparison: at minimum a matched receding-horizon planner with the declared plant/resource primitive.

Work: broader battery-aware MPC and field-robot return-to-base literature.  
Status: unverified systematic coverage.  
Why relevant: may supply stronger classical baselines or weaken claims of practical novelty.  
Overlap: online energy sufficiency and return timing.  
Needed comparison: systematic robotics search before final related-work freeze.

## 8. Claim-Evidence Audit

| Claim | Where stated | Evidence provided | Strength | Reviewer deduction | Required fix |
| --- | --- | --- | --- | --- | --- |
| Pair overlap is unnecessary under target executed-interface agreement | Sec. 4, App. A | local path-law proof | partial/conditional | correct-looking structural statement, but exact-agreement assumption is strong and nearly definitional | independent proof audit plus empirical consequence |
| Outside interface support, target Resource-to-Go is not uniformly identified without structure | Sec. 4, App. A | two-world construction | partial/conditional | useful negative boundary | independently audit model-class and law-separation details |
| Local primitive error accumulates with occupancy and horizon | Sec. 5, App. A | conditional coupling/Lipschitz proof | partial/conditional | not implied by coverage; manuscript says so correctly | verify assumptions or present as explanatory lens only |
| Bounded error changes a threshold decision only near its boundary | Sec. 5, App. A | algebraic proof | strong locally | valid but elementary; value lies in experimental weighting | show boundary-weighted metrics predict mission outcomes |
| Interface risk, not pair identity, predicts error | Intro/RQ2 | none | unsupported | central empirical mechanism absent | controlled factorial result with matched horizons |
| Reliable Resource-to-Go improves stranding--throughput | Abstract/RQ4/Conclusion | none | unsupported | decisive practical claim absent | paired frontier with intervals and strong alternatives |
| Generality beyond one simulator/filter | RQ5/Discussion | none | unsupported | broad framing must remain conditional | second domain or materially different filter family |
| SIRP is needed | Method | none; explicitly gated | not claimed | correct restraint | delete if ensemble explains residual error |

## 9. Experiment / Benchmark / Reproducibility Audit

- Baselines: the revised plan includes SOC, distance--energy, receding-horizon planning, predictive-safety head, direct switcher, direct/TD Resource-to-Go, PCM variants, executed-WM, ensemble, gated SIRP, Oracle, and matched CPO/CVPO/SDAC/DCRL tracks. This is strong on paper but entirely unexecuted.
- Ablations: pair seen/unseen, interface low/high, horizon short/long, nominal versus executed action, ensemble versus support/horizon features, and manager-matched prediction comparisons are planned. The needed interaction analysis must be explicit, not only marginal bar charts.
- Datasets/benchmarks: one deterministic UAV simulator is fixed; the second setting is unspecified. Without the second setting, the work is simulation-specific.
- Metrics: MAE alone is correctly insufficient; boundary underestimation, risk--coverage/AURC, stranding, unused arrival resource, tasks/cycle, and paired frontiers are appropriate.
- Statistical rigor: Wilson and paired bootstrap intervals are planned. The revised protocol correctly treats 100 cycles as a floor and requires precision-driven rare-event sample sizes.
- Robustness/failure cases: fail-closed gates and deletion criteria are strong. Required failure reports include active-set discontinuities, long horizons, out-of-support interfaces, and unsuccessful replication.
- Implementation details: information access and provenance fields are specified, but architecture, optimization, compute, and hyperparameters necessarily remain incomplete before implementation.
- Artifacts and reproducibility: no anonymous code/data/model/environment package exists. The current appendix is a contract, not a reproducible experiment.
- Limitations: unusually candid and mostly complete; the main missing limitation is that the exact interface-agreement theorem may offer limited algorithmic guidance without a finite-sample support diagnostic.

## 10. Multi-Reviewer Panel

### Reviewer A: RL / CMDP

Reviewer: A  
Expertise: constrained RL, hierarchical policies, CMDPs  
Likely score: 3/10  
Confidence: 5/5  
Main positive signal: the modular-vs-end-to-end question is posed fairly and HOCBF authority is not conflated with a learned cost critic.  
Main negative signal: there is no evidence that prediction plus a hand-designed irreversible manager beats a direct switcher or a matched resource-aware constrained policy.  
Strongest objection: why should a separately learned Resource-to-Go module and threshold be preferable to learning the continue/return decision or optimizing the resource constraint end to end?  
Fatal concern if any: fatal for the current submission—every direct-control comparison is pending.  
Missing citation: the initial draft omitted Qin et al.'s DCRL charging/resource setting; this is now repaired.  
Missing baseline: DCRL where task-equivalent, plus CPO/CVPO/SDAC and the direct switcher at matched interaction/compute budgets; now required by the protocol but not run.  
Overclaim: calling the modular hierarchy a contribution before showing Oracle headroom or direct-switcher inferiority.  
Unclear sentence/figure: “All model-based methods receive the same target-component queries” needs an eventual exact information-budget table with query counts and compute, not only yes/no access.  
Evidence basis: Sections 6--8 and Table 2.  
Score-change condition: audited Oracle headroom plus a matched direct-control track could raise this to 5--6; failure of Oracle headroom keeps it at 3 or below.

### Reviewer B: World Models / Uncertainty

Reviewer: B  
Expertise: model-based RL, distribution shift, epistemic uncertainty  
Likely score: 3/10  
Confidence: 5/5  
Main positive signal: pair identity versus executed-interface support is a clean way to describe safety-filter-induced shift, and the paper avoids false conformal/tail claims.  
Main negative signal: executed-WM plus Deep Ensemble may already solve the proposed empirical problem, leaving no new method; moreover, exact target-interface agreement makes the positive theorem structurally unsurprising.  
Strongest objection: where is the nontrivial bridge from the theorem's exact support/agreement condition to a learnable reliability score that outperforms a world-model ensemble?  
Fatal concern if any: current absence of the pair/interface/horizon factorial and risk--coverage results is fatal; ensemble success would not be fatal if the paper is reframed honestly as analysis/benchmark work.  
Missing citation: the initial draft omitted Predictive Safety Network, the closest resource-prediction hierarchy; this is now repaired and elevated to a baseline.  
Missing baseline: executed-WM plus Deep Ensemble, kNN interface distance, predictive-safety head, and horizon-only reliability, all calibrated without target leakage.  
Overclaim: “an unseen pair can remain identifiable when its target executed interface is covered” is mathematically scoped, but readers may mistake coverage for a finite-data certificate; every occurrence must preserve the identification/estimation distinction.  
Unclear sentence/figure: “interface risk” is not yet operationally defined in the main paper; the final method must specify the metric, representation, neighborhood/kernel, and calibration split.  
Evidence basis: Sections 4--7, Appendix A/B.  
Score-change condition: a controlled interaction result showing that interface extrapolation and horizon predict severe underestimation beyond pair identity and ensemble disagreement could raise this to 6.

### Reviewer C: Robotics / Energy

Reviewer: C  
Expertise: persistent autonomy, UAV energy, safety filters  
Likely score: 2/10  
Confidence: 4/5  
Main positive signal: the problem is practically recognizable, the irreversible cycle semantics are explicit, and early-return cost is measured alongside stranding.  
Main negative signal: the current evidence is a deterministic, simulation-specific battery manager with no battery aging, stochastic wind, sensing error, charger availability, docking failure, or hardware validation.  
Strongest objection: why is this a broadly relevant autonomy contribution rather than a simulator-specific threshold manager wrapped around a navigation stack?  
Fatal concern if any: broad robotics claims are fatal without a second domain/filter and meaningful perturbations; a carefully scoped simulation claim could survive.  
Missing citation: the initial draft omitted Mathew et al.'s persistent-UAV recharge planning; this is now repaired.  
Missing baseline: receding-horizon energy-aware planning using the same known primitive, plus the Oracle and tuned distance/SOC heuristics.  
Overclaim: the title's generic “Agent” is broader than the planned evidence unless replication succeeds.  
Unclear sentence/figure: Resource-to-Go needs explicit physical units, battery accounting, charger geometry, decision period, reserve, and docking/arrival semantics in the final experimental appendix.  
Evidence basis: Problem Setup, RQ1/RQ4/RQ5, Limitations, Appendix B.  
Score-change condition: two materially different settings with realistic perturbations and a planner comparison could raise this to 5; one-simulator success caps the score near 4.

### Best-Justified Reviewer

Reviewer: Best-justified accept case  
Expertise: theory-guided empirical RL  
Likely score: 4/10  
Confidence: 4/5  
Main positive signal: the paper organizes a messy deployed-control issue into an auditable sequence with sharp non-claims and unusually strong falsification gates.  
Main negative signal: the sequence is not yet empirically instantiated.  
Evidence basis: Intro, Sections 4--7, Limitations, appendices.  
Fatal concern if any: no accept case exists without results.  
Score-change condition: a clean negative or positive mechanism study with complete artifacts could make the analysis valuable even if SIRP is deleted.

### Writing / Clarity Reviewer

Reviewer: Writing and clarity  
Expertise: ML paper structure and visual communication  
Likely score: 5/10 on clarity alone  
Confidence: 5/5  
Main positive signal: the causal chain, safety scope, and pending evidence are recoverable quickly; the first-page figure is self-contained.  
Main negative signal: dense theorem language and repeated qualifiers may obscure the simple stopping problem for a broad ICLR audience.  
Evidence basis: Abstract, page-1 figure, Sections 3--5.  
Fatal concern if any: none.  
Score-change condition: add one concrete trajectory example and define the operational interface-risk score without increasing jargon.

### Ethics / Reproducibility Reviewer

Reviewer: Ethics and reproducibility  
Expertise: artifact audit and responsible ML  
Likely score: 4/10 at present  
Confidence: 5/5  
Main positive signal: no human/private data issue is apparent; provenance fields, failure retention, and no-invention rules are explicit.  
Main negative signal: the intended narrow AI-use statement is not yet accurate for the current draft because the author rewrite is incomplete, and no anonymous artifact package exists.  
Evidence basis: Appendix B, pending-result ledger, AI-use section.  
Fatal concern if any: inaccurate AI disclosure or identity leakage could become desk-rejection/ethics issues.  
Score-change condition: completed author rewrite, a disclosure checked against the frozen final text, and frozen anonymous code/data/model/environment artifacts.

### Novice Advocate Reviewer

Reviewer: Novice advocate  
Expertise: broad ICLR readership  
Likely score: 4/10  
Confidence: 4/5  
Main positive signal: “too early versus too late” and the executed-action chain are intuitive from page 1.  
Main negative signal: identification, occupancy, Wasserstein continuation regularity, and SSP arrive before a concrete worked example.  
Evidence basis: pages 1--4.  
Fatal concern if any: none.  
Score-change condition: one running example linking filter intervention to extra return energy and a boundary flip.

### AC / Meta-Reviewer

Reviewer: Area Chair synthesis  
Expertise: cross-area RL/robotics adjudication  
Likely score: 3/10  
Confidence: 5/5  
Main positive signal: coherent, falsifiable thesis with disciplined scope.  
Main negative signal: empirical evidence is entirely absent and theoretical novelty alone is unlikely to carry the paper.  
Evidence basis: consensus of A/B/C and claim--evidence audit.  
Fatal concern if any: present evidence state.  
Score-change condition: results must validate both mechanism and downstream decision utility against the newly strengthened closest baselines.

Agreement: all reviewers agree that the problem and causal framing are intelligible, but that zero empirical evidence is decisive now.  
Disagreement: the best-case reviewer sees a potentially valuable analysis/benchmark paper even if no new method survives; the domain reviewer doubts that simulation-only evidence can justify the generic framing.  
Decisive positive axis: mechanism clarity plus disciplined evidence gating.  
Decisive negative axis: missing empirical mechanism and utility evidence.  
Unresolved evidence: Oracle headroom, factorial mechanism, ensemble/SIRP gate, decision Pareto frontier, direct constrained-control comparison, and replication.  
AC stance: reject current draft; invite reevaluation only after the complete gate chain produces auditable evidence.

## 11. Concerns Table

| ID | Severity | Concern | Evidence basis | Affected criterion | Fix class | Required action | Owner skill | Score-change condition |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C1 | fatal | no empirical result supports practical/mechanism claims | all result slots pending | evidence, significance | experiment | complete gated evaluation with provenance | experiment pipeline | mechanism + decision results present |
| C2 | major | identification theorem may be viewed as exact-agreement tautology | Sec. 4/App. A | novelty, soundness | method/soundness | independent proof/novelty audit and operational finite-sample bridge | proof checker | theorem yields testable nontrivial consequence |
| C3 | major | direct constrained control may dominate | RQ4/CMDP slot | significance | experiment | matched direct switcher and CPO/CVPO/SDAC/DCRL | experiment designer | modular track wins or exposes distinct tradeoff |
| C4 | major | executed-WM ensemble may make SIRP unnecessary | Sec. 6/RQ3 | novelty | experiment | run strong ensemble; delete SIRP if gate fails | result-to-claim | residual mechanism remains after ensemble |
| C5 | major | one deterministic simulator cannot support generic agent framing | RQ5 | generality | experiment | second domain/filter or narrow title/claims | experiment designer/paper writer | replication passes or scope narrows |
| C6 | moderate | rare-event precision not yet fixed | statistics paragraph | evidence | experiment | preregister ceiling and interval-driven cycle count | experiment designer | interval resolves declared ceiling |
| C7 | moderate | interface-risk score is unspecified | Appendix B | reproducibility | method/soundness | define representation, metric, split, and leakage controls | paper writer | score can be independently reproduced |
| C8 | moderate | anonymous artifacts absent | submission audit | reproducibility | reproducibility | freeze code/data/models/env/license/README | submission checker | third party can recreate tables |
| C9 | fatal for upload | narrow AI-use disclosure depends on an incomplete author rewrite | AI-use section | policy/ethics | ethics/limitations | independently replace non-literature AI-assisted content, then confirm final disclosure | author | disclosure accurately matches frozen submission |

## 12. AC / Meta-Review

Reviewer consensus: the manuscript has a good problem decomposition, but the submission is currently a theory/protocol shell.  
Reviewer disagreement: whether a strong negative result plus the structural analysis would be sufficient for ICLR; this depends on how decisive and general the study becomes.  
Decisive acceptance axis: show that executed-interface extrapolation and horizon explain failure beyond pair identity and standard ensemble scores, then translate that into a better or more interpretable stranding--throughput frontier.  
Decisive rejection axis: no Oracle headroom, no advantage over direct switching/planning/constrained control, or only one favorable simulator.  
AC stance: reject now. Do not average the evidence score upward because the plan is strong.  
Discussion risks: theoretical contribution may be called obvious/assumption-defined; applied reviewers may reject simulation scope; method reviewers may delete SIRP and ask what remains.

## 13. Quantitative Scores

## Scorecard

| Dimension | Score (1-5) | Confidence (1-5) | Evidence basis | Deduction / score-change condition |
|:---|:---:|:---:|:---|:---|
| Novelty | 3 | 4 | Sec. 4--6 and closest-work search | executed-interface framing is distinct, but the positive theorem is structurally close to its assumption and confirmed method is mostly standard; decisive mechanism evidence can raise it |
| Soundness | 3 | 4 | Sec. 3--5, App. A | local arguments are scoped and plausible, but no independent proof audit and no finite-sample bridge |
| Evidence | 1 | 5 | Sec. 8 and result sheets | all empirical cells are empty; only completed audited results can raise this |
| Significance | 3 | 4 | Intro, RQ1/RQ4/RQ5 | important stopping problem, but utility and generality are untested |
| Clarity | 4 | 5 | Abstract, first-page figure, claim gates | causal story is clear; operational support metric and worked example are missing |
| Reproducibility | 2 | 5 | Appendix B | protocol/provenance contract is strong, actual artifacts absent |
| Ethics / Limitations | 4 | 5 | Problem semantics, Limitations, AI section | honest non-claims; AI statement awaits factual confirmation |

**Overall:** 3/10  | **Scholarly Confidence:** 5/5

**Recommendation:** reject  
**Verdict:** mechanism plus decision evidence against executed-WM ensemble, direct switching, classical planning, and constrained control could raise the score by 2--3 points; failed Oracle headroom or unreplicated simulation-specific effects keep it at 3 or lower.

Quality: 3/5  
Clarity: 4/5  
Significance: 3/5  
Originality: 3/5  
Soundness: 3/5  
Evidence: 1/5  
Reproducibility: 2/5  
Ethics / Limitations: 4/5  
Overall: 3/10  
Confidence: 5/5  
Score-change conditions: complete evidence package, independent proof audit, and scoped generality.

| Change | Condition | Likely affected dimensions | Expected movement |
| --- | --- | --- | --- |
| Raise score | Oracle headroom plus controlled pair/interface/horizon mechanism and matched Pareto advantage | Evidence, significance, novelty | +2 overall |
| Raise score | second setting and reproducible artifact package | Significance, reproducibility | +1 overall |
| Lower score | Oracle has no practical headroom or simple planner/direct switcher matches all gains | Significance, novelty | -1 overall or terminate paper line |
| Lower score | independent proof audit finds invalid identification/transport step | Soundness | fatal for theory claims |
| No quick change | realistic cross-domain validation and rare-event statistics | Evidence, significance | requires new experiments |

## 14. Questions For Authors

1. What finite-sample, source-only quantity operationalizes target executed-interface risk without using target outcomes?
2. If executed-WM plus Deep Ensemble wins the risk--coverage gate, will the authors remove SIRP and submit the work as an analysis/benchmark paper?
3. What declared stranding ceiling and confidence precision determine the cycle count at each frontier point?
4. Can DCRL's density constraints be made semantically equivalent to the resource/return objective; if not, what is the fairest direct constrained-control alternative?
5. What second domain or safety-operator family is fixed before seeing primary results?
6. Has every non-literature AI-assisted part of the current draft been independently replaced rather than merely copyedited, so the narrow disclosure accurately describes the submitted version?

## 15. Score Revision Criteria

Raising the score would require: completed gated experiments; strong matched baselines; a testable link from interface support/horizon to failure; downstream paired mission utility; and at least one replication setting.  
Lowering the score would be triggered by: no Oracle headroom, ensemble/planner/direct-switcher equivalence, target leakage, underpowered rare-event estimates, invalid proof assumptions, or broad claims from one simulator.  
Concerns unlikely to change before submission: hardware validation and full real-world battery/stochastic coverage unless the project scope expands materially.

## 16. Action Plan And CCFA Handoffs

Priority: P0  
Action: complete the independent author rewrite, then factually confirm the narrow ICLR AI-use statement against the frozen submission.  
Owner skill: author + submission checker  
Input needed: author-rewritten manuscript and final literature/reference workflow  
Expected output: accurate finalized disclosure in paper and submission form  
Handoff required: yes

Priority: P0  
Action: independently audit the four formal results and assumptions.  
Owner skill: proof checker  
Input needed: frozen theorem/proof TeX  
Expected output: gap ledger and repaired proof package  
Handoff required: yes

Priority: P0  
Action: execute the navigation-to-Oracle-to-mechanism-to-decision gate chain without bypassing failed gates.  
Owner skill: experiment pipeline  
Input needed: frozen code/checkpoints/configs and preregistered sample sizes  
Expected output: provenance-complete raw artifacts and failure reports  
Handoff required: yes

Priority: P1  
Action: compare against predictive-safety head, receding-horizon planner, direct switcher, executed-WM ensemble, and matched constrained-control track.  
Owner skill: experiment designer  
Input needed: information/compute/interaction contracts  
Expected output: fair baseline matrix and audited results  
Handoff required: yes

Priority: P1  
Action: convert completed evidence into only the claims it supports and replace result shells within the two-page reserve.  
Owner skill: result-to-claim + paper writer  
Input needed: audited result package  
Expected output: evidence-bounded final Results/Abstract/Conclusion  
Handoff required: yes

Checks run: full manuscript/proof/protocol read; current literature repair; claim/numeric/citation/anonymity/build/font/page/visual audits; three cynical domain reviews; AC synthesis.  
Checks skipped: independent proof-agent review, experimental artifact verification, systematic all-robotics-venues review, author rewrite verification, and final AI disclosure confirmation.  
Unresolved risks: empirical emptiness, theory novelty, direct-control competitiveness, generality, rare-event power, artifact readiness, author rewrite completeness, and AI-use factual accuracy.
