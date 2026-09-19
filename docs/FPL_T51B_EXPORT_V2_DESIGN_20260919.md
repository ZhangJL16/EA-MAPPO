# T5.1b completed; Teacher V2 design frozen, not executed

## Material Passport

- Base commit: `3f000599cb086eea903bcd689b26cd71e2114b21`.
- Workflow: academic-research-suite, inline experiment engineering; evidence before claims.
- Scope: pure target export and prospective V2 design. No training or new teacher solve.
- Verification: six new focused tests and full 49-test suite passed; full prefix scan and packaging checks passed.

## Deliverables

- [Prefix export contract](../research/feedback_protocol/CONTRACT_T51B_PREFIX.md).
- [Exporter](../research/feedback_protocol/scripts/export_teacher_targets.py) and
  [frozen export settings](../research/feedback_protocol/configs/teacher_prefix_export_v1.json).
- [Manifest](../research/feedback_protocol/provenance/teacher_prefix_v1/manifest.json),
  [integrity audit](../research/feedback_protocol/provenance/teacher_prefix_v1/audit.json),
  [portable targets](../research/feedback_protocol/provenance/teacher_prefix_v1/teacher_prefix_v1.zip).
- [Teacher V2 design](../research/feedback_protocol/TEACHER_V2_DESIGN.md),
  [machine-readable specification](../research/feedback_protocol/configs/teacher_v2_design_v1.json),
  [SHA256 commitment](../research/feedback_protocol/configs/teacher_v2_design_v1.sha256.json).

V1 data, source, exclusions, limits and evidence are untouched. The exporter is
outside the sealed FPL runtime and imports no FPL, solver, environment or neural
framework. Input is the SHA-pinned full audit archive, not a new problem generator.

## Export result

| Item | Count |
|---|---:|
| Exact source states | 273 |
| Exported prefix rows | 5,383 |
| Nonterminal decision prefixes | 2,394 |
| Complete-protocol END rows, policy-loss masked | 2,989 |
| Decision prefixes with tied optimal next operations | 82 |
| Excluded unresolved states | 0 |
| New solver/environment calls in export | 0 |

This is **not 5,383 independent training examples**. They derive from 273
root–belief–budget states and eight structures. Every state has total prefix
policy weight one; value supervision is once per state. All advantages are
nonpositive, optimal advantages are exactly zero, and each prefix's targets were
checked against a separate direct scan of the stored complete-protocol candidates.

Outputs retain state V and V/max(H,1), full Q/immediate/outcome witnesses, prefix
Q_next, advantages and tied action sets, physical prefix position, source/work
metadata and structural fold. IDs and label/work/provenance metadata are not
inference features. The future set-policy loss sums probability over tied optima.
No network, loss optimizer or training sampler was implemented in this task.

STOP is permitted only at the empty prefix and ends deployment. END only closes
the completed protocol decoder. Prefix Q is total value from the original reset
decision, not a residual physical-state value. Decoding sees the original belief;
pending observations do not release feedback or update posterior mid-protocol.

## Metadata-only internal structural holdout

| Fold | Structures | States | Non-prior AND measurement-required states |
|---|---:|---:|---:|
| student_train | 6 | 145 | 0 |
| student_validation | 2 | 128 | 2 |

All variants/states of one structure remain together. The split uses fixed
salted-hash ordering without looking at targets; it was not revised after these
counts appeared. This is internal TRAIN-corpus validation, not a DEV or final-test
run. It supplies **no evidence that a student can learn continued sensing**;
indeed the required subset is absent on its train side. Merely deriving more
prefix rows cannot repair the V1 coverage limitation. Do not start formal AFPP
training on this export.

The test suite checks unequal Q, tied next operations, local dead-end exclusion,
STOP-only zero horizon, END's nonzero inherited Q, debit-before-reload energy,
unit state weight, structural grouping, output overwrite refusal and deterministic
full-corpus export equality. Full-prefix arithmetic validation is not independent
optimality verification of source Bayes continuations.

## Teacher V2: what is frozen versus pending

The design fixes four TRAIN families, K=4/depth=2, 3–5 channels, genuine randomized
transit/recovery structure, separate RNG streams and paired H/capacity variants.
Its target is 32 distinct structural groups, with a first sizing pilot of eight
(two per family). These are engineering assumptions, not measured capacity or
venue requirements. Candidate admission/exclusion occurs before labels.

State collection will add a public-information explorer and label-independent
stratified retention using rational uncertainty, distance from prior, H and
released measurement count. The design explicitly separates prospective goals
(at least six groups with non-prior measurement-required states, including four
train/two validation) from admission: no label-based root selection, holdout
reassignment or replacement on solver failure is allowed.

No V2 generator, candidate roots, admitted instance manifest, sizing-pilot data
or readiness result exists yet. The SHA commitment binds design bytes only;
future implementation and inputs need their own source/manifest seal before
execution. No automatic full-corpus or training promotion is authorized by JSON.

## Reproduction and evidence

From `/home/zjl/mappo`, export to a **new** path:

```bash
python3 research/feedback_protocol/scripts/export_teacher_targets.py \
  --archive research/feedback_protocol/provenance/teacher_v1_complete/teacher_v1_complete_evidence.zip \
  --audit research/feedback_protocol/provenance/teacher_v1_complete/audit.json \
  --config research/feedback_protocol/configs/teacher_prefix_export_v1.json \
  --output /home/zjl/fpl_teacher_prefix_v1_replay
```

Actual target directory: `/home/zjl/fpl_teacher_prefix_v1_20260919`.
Manifest is the last completion marker; partial output without it is invalid.
The portable ZIP includes targets, exporter, packaging script and V2 design.
Archive SHA256: `c78bb29b97088c60a483ae54705eff2928301c565ce5a16e7e5d876067c04add`.
Design SHA256: `d07ea15a226b5e6a072695dda6abbf003e26a3f6d4e6f0e7bb3d6c6b17df4a8e`.

No neural/GPU training, public DEV/test/OOD evaluation, confirmation preimage
access, Git commit or push. The next implementation task is V2 generation and
label-independent state collection under this versioned design, not T5.2.
