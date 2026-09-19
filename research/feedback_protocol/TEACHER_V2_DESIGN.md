# Teacher V2: frozen expansion design, not implemented data

## Material Passport

- Scope: T5.1c design only, frozen before generation or solver outputs.
- Basis: user's four TRAIN families, need for structure/continued-sensing diversity.
- Evidence status: prospective engineering defaults; no V2 roots or labels generated.
- Workflow: academic-research-suite experiment planning, inline, no external review claim.

Machine-readable design: [teacher_v2_design_v1.json](configs/teacher_v2_design_v1.json).
Its exact bytes are SHA256 committed in `configs/teacher_v2_design_v1.sha256.json`.
This freezes design decisions, **not an executable runtime, problem manifest or
seed-to-instance implementation**. Implementation must be tested and source/inputs
sealed before any future pilot. Any material design revision gets a new version.

## What changes from V1

V1 stays intact. V2 keeps K=4/depth=2, 3–5 channels and small exact budgets, while
varying actual transit connectivity and recovery structure rather than merely
cost/prior reskins. Four families: adaptive hierarchy, random resource graph,
cost/horizon tradeoff, decoy/redundancy. Direct and shared-cleanup returns have
genuine different time/energy; no unreachable padding or zero-cost graph inflation.

Independent named RNG streams prevent changing sensor count from silently
changing the prior/topology. Each admitted root has paired H=9/12 and B=3/5
variants with one fixed model/graph. Variants are not independent structures.
Public feasibility requires possible sequential measurement then task at H=12;
this does not require or predict sensing optimality.

## Admission and sizing

Target 32 distinct structural groups (8/family), within the requested 24–40 range.
Try at most 128 public candidates/family, in predetermined order. Reject reserved
DEV/test/OOD and V1 structures/root identities and within-V2 duplicates **before
labels**, using the conservative name-invariant hash. Keep every admitted root
regardless of sensing labels or solver status. Record exhausted strata explicitly.
Never access the private confirmation preimage.

Freeze all admitted identities and metadata-only student folds before labels.
The sizing pilot then labels the first 2 admitted groups/family (8 groups, four
H/B variants each). Report complete selected-state yield, time, unresolved and
coverage; do not automatically run the remaining groups or train a network.
Final scale depends on measured cost/coverage and user decision, not venue norms.

## Better state coverage without outcome filtering

Keep five V1 sources and add public_information_explorer, solely for training
state coverage. It scores legal measurement candidates by expected squared
posterior change divided by duration, ignoring task payoff. Exact rational
dispersion avoids floating-point entropy ties. The explorer is neither a claimed
new algorithm nor a benchmark baseline.

Consider all positive one-step feedback branches of selected routes. Union
source pools, retain provenance, and select at most 48 unique states/problem by
round-robin over public-feature bins: normalized Gini uncertainty, H fraction,
distance from prior and released measurement count. Use label-independent hash
priority within bins. This is a rational concentration proxy instead of entropy;
it was chosen for deterministic retention. Solver outcome, optimal action and
measurement-required labels are forbidden selection features. Retention itself
must be sealed before label generation.

The target is broader post-feedback states, not simply more PS quota. Collection
caps/truncations remain reported, and exact-label failure never produces a
partial target or an automatic higher-cap retry. Reuse V1's audit principles in
a separate V2 namespace/runtime; do not patch the old source seal.

## Internal readiness reporting, not admission filters

After data exist, report groups containing non-prior **measurement-required**
states, not merely a tied measuring optimum. User's engineering target is at
least six such groups; this design asks for at least four in train and two in
structural validation. The metadata-only fold assignment is fixed first and is
not adjusted to hit those counts. If targets are missed, retain the data and
report the shortfall; do not silently drop/replace roots or launch formal training.
This is not another B2-necessity test and is not an ICML acceptance threshold.

Exact selected-state rate, sampling truncations and condition/H/source-specific
coverage must accompany these counts. Zero unresolved does not prove absence of
small-problem or state-retention bias. V1's adaptive coverage remains five
non-prior measuring-optimal states in one family; prefix multiplication cannot
improve that fact.

## Later training remains separate

State/prefix losses, normalized value and advantages follow T5.1b. Optional
rare-class oversampling is a future explicit TRAIN-only training choice; it
must not alter evaluation distributions or root admission. No PPO, no GPU run,
no learned baseline, no model overfit test or DEV evaluation occurs in this task.
