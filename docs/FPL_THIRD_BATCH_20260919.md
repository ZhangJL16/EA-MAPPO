# FPL third batch: scientific comparison completed, no B2 promotion

Base commit: `3ecc0351c473387aef3efd919bcaa2ac1d82098e`, master.
This batch is a local implementation and bounded DEV study, not a new trained
method or complete A/B/C. No commit/push was performed by this execution.

Post-implementation note (2026-09-19): this third batch was subsequently committed
as `f557b3c47e4e8b1ccdb5623526405e3167acf4f0`. The original execution-time
statement above remains unchanged as a historical record.

## Deliverables

| Requested item | Implemented evidence | Boundary |
|---|---|---|
| T3.1 Scientific Dataset V1 | RegistryV2; independently seeded random graph/parameter generator; frozen 44-config manifest | Same-generator clone-disjoint IID; explicit OOD axes, not final-test results |
| T3.2 Certified Tiny Minimax | Independent primal/dual rational basis search; unequal-support tests; 60 seeded random matrices vs SciPy LP | Capped tree/LP search can remain unresolved; runtime stdlib only |
| T3.3 Compute budget | Shared SearchBudget across complete select, recursively including candidates/outcomes/tails | Cooperative wall limit and implementation work units; not equal FLOPs |
| T3.4 Evaluator | Channel-indexed CRN, all truths, separate policy/noise seeds, episode-atomic resume, expected-risk estimates | Four root groups; Monte Carlo estimates, not population guarantees |
| T3.5 DEV pilot | Six method types, four online methods at two tiers, offline exact references | No neural training; no final-test/OOD performance evaluation |

Contract: [CONTRACT_V03](../research/feedback_protocol/CONTRACT_V03.md).
Commands: [package README](../research/feedback_protocol/README.md).

## Frozen protocol and completion

- 12 DEV conditions from four root groups: two IID roots with paired capacities
  3/5 and horizons 4/8; one dense and one sparse size-5 root, capacity 7,
  horizons 8/12. Scale roots are not claimed as IID test data.
- All two hypotheses × policy seeds `{11,12}` × noise seeds `{31,32,33,34}`.
- ChannelCover, PosteriorSampling, BeamBayes, OneStepVOI each use SMALL/LARGE:
  `(300 expansions, 3000 model charges, .2 s)` and
  `(3000 expansions, 30000 model charges, .5 s)` **per complete selection**.
  Episode totals can exceed these limits because an episode has several decisions.
- ExactBayes and CertifiedMinimax use explicitly separate offline reference
  budgets. Bayes state cap 2,000; minimax unique-vector cap 300. Their fast
  cached-policy execution is not an online planning advantage claim.
- 1,920 scheduled records: **1,776 completed executions, 144 reference-unresolved
  records**. All 1,536 online-baseline episodes completed; no unresolved result
  was substituted with zero or an approximate optimum.
- Oracle 12/12 resolved; ExactBayes 11/12; CertifiedMinimax 4/12. Unresolved
  attempts remain in every schedule and report. Minimax solved the four T=4 IID
  conditions; all other minimax conditions hit the unique-vector cap.

One first-record checkpoint was checked for DEV-only selection, legal elapsed
budget, identity and compute fields; the same sealed run resumed to completion.
There was no outcome-guided change to methods, seeds, hypotheses or horizons.
No original bundling calibration/CONFIRM data or frozen scripts were modified.

## What the data do and do not establish

The full [per-condition result table](../research/feedback_protocol/provenance/dev_pilot_v1/RESULTS.md)
and [machine-readable summary](../research/feedback_protocol/provenance/dev_pilot_v1/summary.json)
include risks, conditional Monte Carlo error, measurements, protocol lengths,
compute, paired Bayes comparisons and empirical Pareto status.

**Question A — distance to an exact reference:**

- All eight small IID conditions have exact Bayes solutions. At several
  conditions the simple policies already have MC Bayes risk equal to the exact
  Bayes risk. This does not require a learned protocol generator.
- For IID root 0, T=8, the exact Bayes risk is 3.5. Large Beam and VOI both
  estimate 3.625, with conditional noise MC SE .625; PS gives 4.0. The finite
  sample does not establish population superiority. On the same CRN trajectories,
  VOI and the cached Bayes policy have equal measured risk.
- Sparse T=12 is an important **estimation warning**: exact Bayes risk is 5.25,
  but the Bayes, large Beam and large VOI trajectories all estimate 3.0. Their
  four noise-block values are identical, giving an empirical SE of zero. They
  have NOT beaten the optimum. A small sample can be optimistic and its sample
  variance can miss real uncertainty. We retained this result, did not add
  favorable seeds, and do not use it as a scientific performance certificate.
- Bayes and minimax are not interchangeable: in the solved T=4 conditions a
  Bayes tie-break can have worst-hypothesis risk 4 while the certified randomized
  minimax risk is 2; both have Bayes risk 2. Both estimands are reported.

**Question B — computational limits:**

- Exact Bayes solve time on the eight IID conditions was approximately
  0.00047–0.0178 s. Sparse T=12 solved in .0672 s at 847 states. Thus these
  conditions do not demonstrate a need for amortized neural planning.
- Dense T=8 solved at 1,008 states in .1635 s. Dense T=12 reached the specified
  2,000-state cap after .3796 s and remained unresolved. This is a configured
  reference limit, not proof that exact planning is hardware-infeasible.
- At that unresolved dense T=12 condition, large VOI estimates risk 5.0 (noise
  MC SE 1.0), versus 6.0 for Beam/PS. Since the optimum is unresolved and the
  sample is small, this does not establish distance from the best attainable
  value or a fundamental inability of ordinary planning.
- Counts and per-instance empirical Pareto points expose budget sensitivity;
  they do not turn one timing/noise sample into a general Pareto theorem.

**Decision: do not launch B2 neural training.** This pilot has not established
the required conjunction “exact planning genuinely costly + ordinary planners
reliably far from attainable value.” It also does not prove that simple planning
is sufficient at every scale. Keep the nonlearning planning line and the frozen
DEV evidence; no expansion of the algorithm claim is justified by this batch.

## Correctness and statistical checks

30 focused tests pass, including the original 21. New tests cover independent
primal/dual supports, random LP cross-checks, clone-disjoint split semantics,
global budgets, no-enumeration VOI, channel-query CRN, minimax execution after
feedback, max-of-means risk and strict pause/resume/seal behavior.

After the study, cached public policies were replayed by a rational
hypothesis-wise vector evaluator, without reoptimizing or drawing new samples:
all 11 completed Bayes policies attain their stored weighted values, and all
four completed minimax mixtures reproduce their certified risk vectors. See
[policy replay](../research/feedback_protocol/provenance/dev_pilot_v1/policy_replay.json).
This is implementation-diverse internal verification, not independent human
proof review.

Statistical reporting follows the nature-statistics skill's unit-of-analysis
checks: root groups, repeated conditions and Monte Carlo replicates are separate;
no p-values, significance or population generalization is asserted. Minimax is
max of per-hypothesis estimated means, not worst observed trajectory. Root
summaries with missing references cover different conditions and must not be
compared as if matched. The raw per-condition results remain primary.

## Reproducible evidence and provenance

- Dataset: `/home/zjl/fpl_scientific_v1_20260919`
- Frozen study: `/home/zjl/fpl_dev_pilot_v1_20260919`
- Portable [DEV evidence archive](../research/feedback_protocol/provenance/dev_pilot_v1/dev_evidence.zip):
  227,920 bytes. Includes raw episode JSONL, all reference attempts, the seal,
  manifest, 12 evaluated public configurations, full analysis and frozen runtime
  source. It includes no final-test trajectories.
- SHA-256: `b634bf71a2e05125c42a7e525ea220881be075f61e6f39b89d1380fd02614f7a`.
- [Inventory](../research/feedback_protocol/provenance/dev_pilot_v1/inventory.json)
  supplies per-file hashes. An output directory alone is not called a remote
  reproducibility package; the portable archive is provided in the repository.

The first-batch post-hoc commit note already existed. This batch appended the
second-batch note for `3ecc035...` without rewriting its historical statement.
Full teacher datasets, learned decoder/critic, external likelihood adapters and
final IID/OOD evaluations remain unimplemented. No GPU training was started.
