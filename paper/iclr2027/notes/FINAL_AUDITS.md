# Pre-Results Integrity and Submission Audits

Audit date: 2026-08-28  
Target: ICLR 2027 conference submission  
Manuscript: `paper/iclr2027/main.tex` and `build/main.pdf`  
State: evidence-bounded pre-results draft; not submission-ready

## Integrity audit

Mode: full  
Artifacts checked: all manuscript and appendix TeX sources, overview TikZ, baseline and result-slot tables, `references.bib`, compiled PDF/log/auxiliary files, repository research blueprint/protocol/completion ledger/derivation and proof packages, literature-search artifacts, and the user-provided Keogh SIGKDD 2009 tutorial.  

### Claim--evidence matrix

| Claim class | Evidence checked | Status | Integrity decision |
| --- | --- | --- | --- |
| Executed-interface identification under queryable target components, shared primitive, target-interface agreement, and proper SSP | theorem statement and complete local appendix proof | supported locally, independent proof audit pending | retain with every assumption visible; do not call it a finite-sample guarantee |
| Non-identification outside source interface support over the declared structure-free kernel class | proposition and two-world construction | supported locally, independent proof audit pending | retain with model-class qualifier |
| Occupancy-weighted finite-horizon transport and SSP truncation | theorem and recursive coupling proof | supported only under one-step coupling and continuation-law Lipschitzness | retain; coverage alone is explicitly rejected |
| Threshold-decision disagreement is confined to a prediction-error band | theorem and algebraic proof | supported locally, independent proof audit pending | retain; do not equate disagreement with stranding |
| Pair identity is empirically weaker than executed-interface extrapolation | no completed factorial result | unsupported at present | marked `RESULT PENDING`; no empirical claim allowed |
| Oracle headroom, selective reliability, Pareto improvement, generality, and CMDP competitiveness | no completed audited result | unsupported at present | all remain visible pending slots |
| HOCBF supplies hard collision-safety authority | system design/protocol plus cited HOCBF literature; no new paper-level empirical safety claim | scoped design statement | learned Resource-to-Go is explicitly not a certificate |

Numeric consistency findings:

- No experimental measurement is reported in the abstract, main results, conclusion, tables, or figures.
- The 5% Oracle headroom value is a preregistered practical-relevance gate, not a measured gain or significance threshold; the text now requires the continuous effect, interval, and sensitivity analysis.
- `100 cycles` is now only a floor. The manuscript requires the final count to be chosen from the precision needed at the declared stranding ceiling and explicitly says that 100 cycles cannot resolve a one-percent rare-event claim by itself.
- The appendix's 90% lower bound from two marginal q95 statements is a union-bound derivation, not a simulator result; it correctly denies a 95% joint claim.
- Metric directions agree across the protocol, baseline contract, main result slots, and appendix result sheets.

Citation metadata findings:

- 23 unique citation keys are used and 23 unique BibTeX entries are present; there are no unused or duplicate keys.
- All entries were checked against official proceedings, publisher/DOI, or OpenReview publication records.
- The cynical-review search added three material predecessors that were absent from the first draft: Predictive Safety Network, Density Constrained Reinforcement Learning, and persistent-UAV recharge rendezvous planning.

Citation-context findings:

- Predictive Safety Network is now identified as the closest practical resource-prediction hierarchy; the paper limits its novelty delta to queryable policy--operator composition, executed-interface support, and irreversible-boundary analysis.
- PCM remains the closest target-policy-occupancy modeling risk and is not portrayed as an executed-interface result.
- Distributional and conformal citations are used only for method/baseline motivation; the manuscript denies physical q95/q99 and unconditional shift-coverage claims in the deterministic simulator.
- HOCBF/safety-filter citations support collision-authority separation, not an energy guarantee.

Severity: no fabricated result or citation found; major unresolved evidence gap because every empirical claim is pending; moderate independent-proof risk.  
Safe edit suggestions applied: closest-work repair, rare-event sample-size qualification, practical-gate qualification, stable theorem-name ledger, and float/readability repairs.  
Next CCFA owner: experiment design/execution and independent proof review, then result-to-claim and paper writer.  
No-invention status: PASS. Empty result cells remain empty and historical pilot/failure values were not promoted.

## Submission audit

Mode: full  
Venue and rule freshness: official ICLR 2027 Author Guidelines and AI Policy for Authors rechecked on 2026-08-28 (`https://iclr.cc/Conferences/2027/AuthorGuidelines`, `https://iclr.cc/Conferences/2027/AIPolicyForAuthors`). The official rules require double-blind anonymity, at most 9 main-text pages for initial submission, references excluded, appendices after references, and a mandatory AI-use section outside the page limit.  
Files checked: official style/bibliography/sample bundle, `main.tex`, every included TeX/TikZ/table file, BibTeX, build log/auxiliary files, and compiled PDF. No `ccfa.yaml` exists in this paper directory, so project-state tracking is unavailable.

### Pass/fail checklist

| Gate | Status | Evidence |
| --- | --- | --- |
| Official ICLR 2027 template | PASS | local style file SHA-256 `797deef41724e93761426ac0cbcca46279a91cc650dd1f0ce76a4f08d2098ea6`; downloaded bundle was previously matched byte-for-byte |
| PDF build | PASS | Windows TeX Live 2025 `latexmk -pdf` exits 0 |
| Main-text page limit | PASS for current draft | main text ends and references begin on page 7; limit is 9 pages |
| References/appendix order | PASS | references begin on page 7; appendix begins on page 10 |
| Double-blind text | PASS for local static scan | anonymous author block; no name, affiliation, email, local path, or repository URL found in sources/PDF |
| PDF metadata | PASS | Title and Author metadata fields are empty |
| Fonts | PASS | every font reported by `pdffonts` is embedded and subsetted |
| Page size | PASS | US Letter, 612 x 792 points |
| Citations/references | PASS | no undefined citations/references; 23/23 key match |
| LaTeX diagnostics | PASS | no fatal error, undefined control, overfull box, or LaTeX/package warning in the final log |
| First-page overview | PASS | rendered page inspected; readable without color and no overlap/clipping |
| Main/appendix result tables | PASS as shells | result status is visibly pending; appendix cells remain empty |
| Mandatory AI-use section | BLOCKED | the intended narrow disclosure is recorded, but it becomes accurate only after the author independently replaces the remaining AI-assisted non-literature content |
| Empirical submission evidence | FAIL/PENDING | all required navigation, Oracle, prediction, decision, replication, and CMDP artifacts are pending |
| Anonymous artifact package | FAIL/PENDING | no frozen release package, environment lock, models/data manifest, licenses, or anonymous artifact README exists yet |

Build/package issues: none in the final build; total PDF length is 13 pages because references, proofs, experiment design, result sheets, and AI disclosure follow the 7-page main text.  
Anonymity/page/font/metadata issues: no current static issue; author identities and self-citations require a fresh human check immediately before upload.  
Length budget status: 2 main-text pages remain for audited final results; result-slot prose must be replaced rather than simply appended.  
Artifact/reproducibility issues: the appendix gives a provenance schema and fairness contract, but actual code/data/checkpoint/environment/license artifacts cannot exist until experiments finish.  
Required fixes: complete the author rewrite gate, then convert the AI disclosure to past tense and confirm it against the frozen submission; complete and audit experiment gates; independently audit proofs; insert only provenance-backed results; build anonymous artifacts; rerun page, citation, anonymity, font, and policy checks at the frozen submission revision.  
Next CCFA owner: author rewrite now; later experiment pipeline, proof checker, result-to-claim, integrity auditor, and submission checker.

## Checks run

- Official 2027 policy freshness and template check.
- Windows TeX Live compile and log scan.
- PDF page/metadata/page-size/font inspection.
- Source and PDF anonymity regex scan.
- Citation-key, BibTeX-key, unused-key, duplicate-label, and undefined-reference checks.
- Claim-verb, numeric-token, pending-marker, metric-direction, and result-slot scans.
- Rendered visual inspection of page 1, the main Results page, and appendix result/AI-use pages.
- `git diff --check`.

Checks not yet possible: experimental artifact verification, code/data/model license checks, independent proof audit, final author/self-citation anonymity review, author rewrite completion, and finalized AI-use accuracy.
