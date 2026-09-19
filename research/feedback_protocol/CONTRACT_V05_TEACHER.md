# V0.5: exact TRAIN teacher data, not a learned planner

This extension leaves the V0.4 environment, references and frozen studies
unchanged. It implements T5.1 infrastructure only. B2 necessity is already GO;
this is not another necessity gate. No network, optimizer or RL training is run.

## Target and information boundary

The target is the **public-prior Bayes** value under committed reset-to-reset
protocols, reset-by-budget terminals and cumulative task utility. It is not a
minimax label or a known-truth best action. At public state `(problem, b, H, B)`:

`Q*(b,H,p) = expected_utility(b,p) + sum_y P_b(y|p) V*(b_y,H-duration(p))`.

Stopping is a separate candidate with value zero. Every successful label stores
the full legal protocol catalogue, exact rational Q and V, all tied maximizers,
one deterministic teacher action, and one-step outcome/continuation witnesses.
The public problem appears once per shard and is linked by its hash. It includes
the complete operation graph, public hypothesis likelihood table, task utility,
costs, capacity and legal restrictions. `legal_first_operations` contains only
first operations admitting a complete legal protocol; it is not yet a decoder.

No `PrivateTruth`, environment instance, future RNG stream or realized utility
is used to build labels or sampling states. Root/seed/source metadata is lineage,
not a future student input. Public prior-predictive feedback is sampled exactly
using rational integer weights. Realized task utility never becomes feedback.

## TRAIN admission before labels

`configs/teacher_v1/plan.json` and its hash-bound exclusion file fix 24 root
slots over eight conditions, depths 1/2, K 2/4, channels 1–5, horizons 4–12,
qualities 0.6/0.7/0.8/0.9, capacity and acquisition-cost variation, sparse/dense.
This is a small structured design, **not a full factorial or population sample**.

The exclusions were exported from the existing public scientific registry
metadata and already-used DEV-V2 specification. No public test/OOD problem files
or performance results are loaded. Root seeds 2201/2202 and all registered
scientific seed identities are forbidden. Conservative name-invariant structural
hashes of DEV/test/OOD are forbidden even if means/costs/names differ. Rejection
is before any planning/reward analysis. A bounded seed search may replace only a
structural clone. Exhausted strata remain visible, never silently redesigned.

Training roots sharing a structural hash remain **one structural group**; a
future student split must not randomly divide their records or reskins. Neither
24 root seeds nor 24 files means 24 independent graph structures. The use of
existing hierarchical generation preserves its coupled RNG stream behavior:
changes of dimension can also change prior/edges; no isolated causal axis claim.

The private confirmation preimage is neither read nor tested. Admission cannot
prove disjointness from as-yet ungenerated confirmation instances. At authorized
future confirmation generation, apply the then-frozen duplicate/custody protocol;
do not inspect the preimage now. No final test has been authorized here.

## State distribution and failures

Each root uses five independent, seed-owned public-history sources: exact Bayes,
random legal protocols, one-step VOI, Beam Bayes, posterior sampling. For each
selected protocol all positive-probability one-step feedback successors enter a
candidate pool, including branches not taken on the sampled rollout. At most
four batches per source are followed. At most six states per source are retained
by value-independent hash priority, with the initial state retained explicitly.

Thus **all one-step branches are considered, not all reachable histories saved**.
This is bounded augmentation, not exhaustive belief coverage or DAgger. Selected
sufficient states `(b,H,B)` merge for the history-independent Bayes teacher only;
their action/feedback histories and source memberships are retained and replayed.
This does not change ExactPolicyEvaluator's full-policy-state semantics.

Sampling planner caps are logged as truncation. Each state receives a fresh
bounded exact solve. State, catalogue, outcome, work or watchdog exhaustion
produces `unresolved` with reason/work, and **no V/Q/action label**. No approximate
fallback, output-based retry, success-only root filtering or imputed zero label.
All selected states, successful or not, remain in the shard. Later training must
explicitly filter `status == exact` and report the resulting selection bias.

## Accounting, reproducibility and stopping

Each root is one atomic, hash-checked checkpoint containing problem, provenance,
labels, sampling failures, work counters, process CPU and wall seconds. The seal
binds plan, exclusions, admitted identities and all FPL source hashes. Single
writer locking, checksum verification and source-drift rejection protect resume.
An interrupted root is not a valid shard; interrupted/repeated work is not in
completed-shard CPU totals and must be separately accounted for if it occurs.

Recorded work separates state collection from label solving. Enumeration,
likelihood and planner calls are charged; bookkeeping/serialization/audit work
is not an expansion/model call. Whole-shard CPU/wall includes in-process label
arithmetic audits but excludes admission and the separate export audit. No GPU
time or trained parameter count exists. Zero network calls is not a speed claim.

`audit_teacher.py` replays admission, hashes, history reachability, complete
candidate sets, exact likelihoods, Q arithmetic and maximizing sets. It does not
independently prove that every stored continuation value is optimal: that rests
on ExactBayes and its tests. Do not call it blind external proof validation.

Following AGENTS.md, the default run writes one first checkpoint and exits.
Continuation is explicit `--resume --max-new N`; existing checkpoints are not
recomputed. Do not modify FPL source after sealing and then silently resume.

The coverage report is descriptive: label status, source membership, non-prior
states, beliefs/horizons, optimal-measurement states, ties, structural groups and
CPU/work. It does not assert that a student will generalize or that coverage is
sufficient. T5.2–T5.4 remain unimplemented.
