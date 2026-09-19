# FPL fourth batch — exact policy evaluation and final B2 decision

Base: `f557b3c47e4e8b1ccdb5623526405e3167acf4f0`, master.
This batch is currently local/uncommitted. No push, neural training, external
task execution or final-test evaluation was performed.

## Outcome

**The frozen operational rule returns GO for a later amortized-planning study.**
This is a bounded research decision, not proof of neural necessity, ML novelty,
classical-planner optimality failure in general, or a Spotlight claim. No fifth
batch has been launched, and no additional pre-B2 gate is proposed.

All **371 fixed-policy evaluations succeeded exactly** over environmental
feedback: 240 fresh DEV-V2 cells and 131 legacy-DEV audit cells. None required
MC fallback. New Bayes references solved 10/12 configurations; both deeper
K=8, H=18 configurations remain unresolved at the 12,000-state cap.

## T4.1 — the old MC ambiguity is resolved

The previous sparse T=12 result was a four-noise-block sample, not expected risk.
Under the new deterministic work contract:

| Method / setting | Exact Bayes risk | Exact worst-hypothesis risk |
|---|---:|---:|
| Cached ExactBayes | 5.25 | 6.60 |
| Beam LARGE | 5.25 | 6.60 |
| OneStepVOI LARGE | 5.25 | 6.60 |
| Beam SMALL | 6.00 | 12.00 |
| OneStepVOI SMALL | 6.00 | 12.00 |
| PS, either frozen seed | 6.00 | 7.00 |

The apparent `3.0 < 5.25` disappears without adding noise seeds. In the legacy
dense T=12 condition, whose old Bayes optimization was unresolved, fixed-policy
evaluation still resolves LARGE Beam at 6 and LARGE VOI at 5.6. Neither value
is asserted to be the unknown optimum.

The evaluator branches full policy state, including RNG, history and remaining
episode balance, rather than merging equal beliefs. Exactness is conditional
on the internal algorithm seed. Eleven cached Bayes policies reproduce their
stored values exactly; all fresh fixed-policy risks with resolved references
are at least their exact Bayes optimum.

## T4.2–T4.4 — implemented contracts and planning baseline

- Deterministic selection budgets: SMALL 300/3,000; MEDIUM 1,200/12,000;
  LARGE 4,800/48,000 expansion/model-call charges. Episode pools cap each
  at four times its selection allowance. MEDIUM selection-only controls are
  additional, separately labeled cells. Wall time only invalidates abnormal
  runs; it cannot choose a fallback action.
- Public hierarchy: coarse and specialized sensors, explicit costs, shared
  trajectories, redundant-channel and K=8 variants. Two new seeds, six conditions
  per root; all frozen before execution, with no performance filtering.
- Added a generative belief-UCT baseline with progressive-widened prefix
  proposals, batch-end feedback and posterior-only rollout decisions. It is not
  an official POMCP reproduction or a claim to be the strongest possible MCTS.
- Five method types, three episode work tiers, a medium selection-only control,
  one fixed internal seed for fresh stochastic policies. No training seeds exist.

Details: [v0.4 contract](../research/feedback_protocol/CONTRACT_V04.md).

## T4.5 — exact quality–computation evidence

The frozen rule tests a normalized low-budget gap of at least 0.05 H for all
three planning baselines, a high-budget gap at most 0.02 H, and a realized model
call increase of at least fourfold at both roots in the same condition. It also
requires increased reference work on a scale axis. These cutoffs are an explicit
operational research choice, not an ICML standard.

The redundant-channel configurations satisfy the two-root quality condition:

| Root | Bayes optimum | SMALL Beam | SMALL VOI | SMALL MCTS | LARGE Beam | Beam expected-call ratio |
|---|---:|---:|---:|---:|---:|---:|
| 2201 | 7.140000 | 9.473684 | 10.015789 | 9.789474 | 7.140000 | 10.19× |
| 2202 | 6.641538 | 7.384615 | 8.653846 | 8.076923 | 6.641538 | 12.63× |

These are rational fixed-policy expectations, not MC fluctuations. LARGE VOI
is also effective: risk 7.357895 at root 2201 and 6.96 at root 2202. This is
important evidence **for keeping ordinary planning as a serious competitor**,
not a reason to omit it from the eventual method comparison.

For the same-model horizon shift H=12→18, successful Bayes solve work grows
from 33,356→235,364 model charges (7.06×) at root 2201 and
24,780→158,196 (6.38×) at root 2202. Thus growth remains over fourfold even
excluding the abandoned 2,000-state attempts. Including all actual reference
attempts, the sealed analysis records about 11× at each root.

Successful solve times were approximately .071→.504 seconds and .054→.356
seconds, respectively. These are measurable increases, **not a claim that all
classical planning is operationally unaffordable**. LARGE Beam's exact-quality
redundant-condition runs take about .195/.241 seconds of expected episode
planning; LARGE VOI is about .030/.029 seconds and near optimal. Whether a
learned planner is worthwhile at a particular deployment latency still has to
be demonstrated by that method, including offline training cost.

Deeper K=8 references remain unresolved. That region supplies no certified
optimal-quality gap and is not used to pass the GO quality test. Fixed-policy
values there are nevertheless exact, not oracle values.

## Evidence limitations that must stay visible

1. Two roots support a within-study replicated phenomenon, not a population or
   OOD superiority claim. Stochastic planners use one fixed internal seed in
   fresh DEV; exact environmental integration does not remove seed dependence.
2. The hierarchy generator changes priors and some sparse edges when dimension
   changes, because it consumes a sequential RNG stream. Redundancy/depth are
   joint configurations, not clean one-factor causal interventions. No generator
   change was made after results. The GO quality comparison uses the identical
   configuration across methods; the compute-growth comparison uses a horizon
   change that preserves the root model/graph.
3. The SMALL budgets are deliberately limited. Gaps concern the actual planning
   implementations and their fallback/scheduling rules, not a fundamental lower
   bound on every classical algorithm. The MCTS baseline is custom, return-guarded
   and untuned, so its poor performance cannot eliminate the entire MCTS family.
4. Several conditions are solved or nearly solved by cheap VOI. GO means a scoped
   amortization research opportunity, not evidence of a new ML paradigm.
5. Timing is a single-host measurement integrated over exact feedback branches,
   not an exact or hardware-independent latency law. Work counters have declared
   implementation units. Offline reference solve cost is reported separately.

## Reproduction, figures and checks

- Study: `/home/zjl/fpl_necessity_v4_20260919` (source/plan seal, 371 atomic rows,
  separate reference attempts and completion marker).
- [Full results](../research/feedback_protocol/provenance/necessity_v4/RESULTS.md),
  [machine-readable summary](../research/feedback_protocol/provenance/necessity_v4/summary.json).
- [Model-call curve](../research/feedback_protocol/provenance/necessity_v4/figures/expected_model_calls.pdf),
  [expansion curve](../research/feedback_protocol/provenance/necessity_v4/figures/expected_expansions.pdf),
  [latency curve](../research/feedback_protocol/provenance/necessity_v4/figures/expected_latency.pdf).
  All 240 fresh cells appear in each plot; all 371 rows are exported in source
  CSV. Legacy audit is not pooled into fresh-root plots. Dashed lines are solved
  Bayes optima; the two unresolved references are explicitly marked.
- [Portable raw evidence](../research/feedback_protocol/provenance/necessity_v4/raw_evidence.zip),
  189,004 bytes; SHA-256
  `5f4e9193e3ed60aed1a9f359a8552bd015632a259ce353a260af5090111f92b3`.
  Includes raw results, public problems, references, seal and frozen runtime.
- 37 focused tests pass, including original 30; pause/resume, unequal state
  branches, owned RNG preservation, shared episode budgets, MCTS safety/no
  catalogue, and watchdog-invalid semantics are checked.
- Figure preflight: 12 PASS, zero FAIL. Two warnings accepted: report PNG is
  300 dpi and no TIFF is supplied; editable PDF/SVG are the primary outputs.
  These are DEV diagnostic figures, not a claim of journal-specific submission
  compliance. Statistical and figure skills enforce exact/MC separation,
  root-level interpretation, full row accounting and nonfabricated uncertainty.

## Confirmation custody

Published v1 test/OOD seeds are now explicitly public benchmark fixtures.
A new [seed-material commitment](../research/feedback_protocol/provenance/confirm_v1_seed_commitment.json)
exists; no final instance was generated. The preimage is stored at
`/home/zjl/fpl_confirm_v1_seed_custody_20260919.json`, outside the repo with mode
0600. It has not been printed, loaded for analysis or bundled in evidence.
This is local custody, not independent external blinding. The final method and
full confirmation protocol are still unfrozen; seed commitment alone does not
authorize final evaluation. Old CONFIRM was not accessed.
