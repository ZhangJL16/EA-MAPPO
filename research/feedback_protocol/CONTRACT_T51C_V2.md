# T5.1c: Teacher V2 sizing and decoder feasibility

## Authority and frozen inputs

This implements the user-approved eight-group sizing study, not AFPP training.
`configs/teacher_v2_design_v1.json` is unchanged (SHA256
`d07ea15a226b5e6a072695dda6abbf003e26a3f6d4e6f0e7bb3d6c6b17df4a8e`).
All 32 admitted groups and 128 public H/B variants, rejections and folds are
persisted before collection/labeling. Only the first two accepted groups per
family (32 variants) can enter this runner. No full-run CLI is provided.

## Implementation choices fixed before admission/labels

- Canonical RNG seed input uses compact JSON `[namespace, seed, stream]`,
  SHA256 big-endian integer. Source substreams derive from the state-exploration
  stream plus source name. Each family receives a disjoint 128-seed interval,
  starting at 300001, in frozen family order.
- The detailed shared-cleanup definition takes precedence over the informal
  adaptive-family phrase “singleton total 3”: direct return costs 3 total;
  shared return costs 4. Shared recovery is not free or unreachable padding.
- `cost_horizon` independent transit-edge probability, unspecified numerically
  in that entry, is 1/2. Extras/type choices use topology RNG; per-channel polarity
  and one root quality use their independent streams.
- Width 16 applies to coverage planners and information-explorer prefix beam;
  Beam depth is 2. Random coverage samples legal primitive continuations.
  Explorer partial prefixes are ranked by predicted squared posterior change
  divided by elapsed prefix duration, completed candidates by full duration.
  It never ranks task utility; no observations are released during construction.
- The exact collection source inherits the 3000-state/10000-protocol-node
  per-decision structural caps, but shares the **collection** episode work and
  15-second watchdog; labels use their separate frozen 5-second limits.
- Fold order is SHA256 of `[salt, clone_hash]` using existing FPL `digest`.
  Admission globally forbids clone duplicates and repeated roots, so each
  connected component is exactly one admitted group; all four variants stay
  together. No fold reassignment after coverage is known.

## State collection and persistence

Six public sources collect prior-predictive histories plus every positive-mass
one-step branch. A joint episode budget charges planner recursion, explorer
likelihood branches and completion-mask expansions. Public sufficient states
merge while preserving all source/history provenance. Label-independent bins
use rational Gini, relative horizon, distance from prior and released count;
smallest bin wins across duplicate histories. Root is retained, then occupied
bins are traversed round-robin with retention-stream hash order, maximum 48.
The `.states.json` checkpoint precedes all label solves for that variant.

Exact labels reuse unchanged V1 rational Q/V code. Failed solves remain
unresolved, without partial targets, retries, replacements or cap increases.
Collection truncation is reported separately from label unresolved status.
An interrupted variant retains its start marker and refuses automatic recovery;
elapsed work without a completed receipt is unknown, never reported as zero.
Completed shards have content digests and source/admission seal references.

## Catalogue-free mask

`ProtocolDecoderState` and `ScalableFeasibilityMask` consume only the public
problem, prefix and remaining horizon. Joint Boolean DP state is
`(node, time_left, energy_left, channel_counts)`. Positive duration makes the
recursion acyclic. A next operation is allowed exactly if it is executable and
some completion reaches the **first** reset without violating any constraint.
Debit precedes reload. STOP is permitted only at the empty prefix; END only at
a completed batch. Posterior is not updated during decoding.

No protocol catalogue, reward oracle or teacher mask enters this implementation.
Separate shortest-time/shortest-energy minima would be insufficient. Exact DP
avoids materializing paths, but its worst-case state space is still
pseudo-polynomial in budgets and exponential in channel count. This is **not**
a claim to solve arbitrary constrained routing in polynomial time. State/work
cap raises `PlanningLimit`; an unresolved mask is not an empty legal set.

Regression compares every prefix of every exact V1/pilot candidate with the
saved exhaustive legal-next set, including STOP/END. Tests include conflicting
time/energy minima, repeated measurement limits and forbidden catalogue calls.

## Reports and boundaries

Report family × fold groups/variants/selected/exact/unresolved/non-prior/
measurement-required/intersection/adaptive groups; source memberships overlap
and are not independent examples. Report collection and solve work separately.
Required sensing means **all** tied-optimal protocols contain measurements;
optimal sensing means at least one does. The full-corpus 6/4/2 targets are not
automatically applied to this eight-group pilot. No root is selected by labels.

No neural training, DEV/test/OOD execution, private-confirm read or remaining
24-group solve is authorized by this contract. V1 and its historical folds
remain frozen. This is a Bayes teacher corpus, not a minimax algorithm theorem.
