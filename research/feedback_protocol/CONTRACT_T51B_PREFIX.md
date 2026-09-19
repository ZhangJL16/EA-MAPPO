# T5.1b: exact protocol-prefix training targets

This is a **pure derivation** from the audited Teacher V1 archive. It does not
call ExactBayes, enumerate new protocols, sample feedback or modify any V1 file.
Q correctness is inherited from the source teacher/audit, not independently proved.

## Semantics

For a proper nonempty prefix u, Q_prefix(u) is max Q(p) over complete protocols
extending u. Q_next(u,a) is max Q(p) over protocols extending u+a. The empty-prefix
decision also includes STOP with Q=0; its maximum equals source V*. Complete
protocol prefixes have only END, with the original complete-protocol Q, not zero.

STOP ends deployment at reset. END ends protocol **construction**: the full
protocol is then executed, its batch feedback released, and a fresh deployment
decision may follow. They are distinct tagged action types, never operation names.
No STOP is legal inside a sortie. All exported operation choices admit at least
one complete legal protocol in the exhaustive source catalogue. A merely local
edge into a dead end is not in the target mask. This is teacher mask supervision;
a scalable deployment mask/decoder has not yet been implemented.

Prefix position records source node, elapsed time, remaining H, energy after
debits, pending channel multiplicities and resource after reset on completion.
Energy is checked before reload. Belief remains the **pre-sortie posterior**
throughout decoding; pending observations are not available. This does not add
within-sortie feedback adaptivity.

Q_prefix is the value from the **original decision state**, including the
whole protocol's utility and later feedback-conditioned deployment utility.
It is not the residual value at the physical prefix endpoint. Normalize all
stored prefix values by max(original H,1), not prefix residual H. State critic
targets retain both V and V/max(H,1). Advantage is Q_next-max Q_next, exactly zero
for all optimal actions and nonpositive for the rest. All values are rational strings.

## Ties and loss contract

Store every optimal next action. Future policy loss is
`-log(sum_{a in optimal_next} pi(a | public state, prefix))`, with a legal-action
mask. Never penalize a different tied teacher-optimal operation. Future advantage
regression is averaged over legal actions within a prefix, not summed by count.

Uniformly sample a state **within its assigned fold**, then a nonterminal prefix
inside that state. Each nonterminal prefix carries weight 1/N_nonterminal; complete
END rows are retained for consistency but have policy weight zero. Each state
therefore has total prefix policy weight exactly one. Its value target is trained
once with weight one, not once per prefix. H=0 has one STOP-only decision and value
zero. This contract prevents route-rich states from gaining weight merely by
having more rows. No class-balancing or training loader/network is implemented yet.

## Information and split boundary

`problems.json` holds public graph/likelihood/utility only. `states.jsonl` separates
inputs, supervision targets and lineage. `prefixes.jsonl` references the state.
IDs, root, condition, fold, structural hash, sources, teacher outputs, outcome
witnesses and solver cost must never become inference features. ID is a join key,
not a learned embedding. Full protocol/Q/immediate/successor witnesses are retained
under supervision targets, not deployment inputs.

The internal student diagnostic split is metadata-only: salted-hash order of
structural groups, first floor(N/4) groups held out (minimum one if N>=2). Same
structure reskins and all states remain together. Export fails if a root spans
several structural groups rather than silently separating it; a future general
dataset must use connected-component grouping for that case. These are TRAIN-only
student folds, not previously reserved DEV/test/confirmation. The eight V1 groups
remain insufficient generalization evidence. Never reassign folds after looking
at adaptive-label availability.

## Integrity

The configuration pins the complete audited archive SHA256. Each source shard's
payload, public problem identity and TRAIN membership is checked. Every exported
prefix is cross-checked by a direct scan of saved complete candidates, separate
from trie aggregation; unit state weight and complete prefix coverage are checked.
Deterministic output hashes, exporter/config hashes, exclusions and fold summaries
are recorded. A manifest marked complete is written last. Existing output paths
are refused; interrupted output without a manifest is not usable training data.
No new exact solves are hidden in verification. V1 runtime remains hash-identical.
