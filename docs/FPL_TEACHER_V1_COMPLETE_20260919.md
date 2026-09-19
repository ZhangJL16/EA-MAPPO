# Teacher V1 completed: full audit, no neural training

## Material Passport

- Workflow: academic-research-suite / experiment-agent, inline execution and descriptive collection.
- Base commit: `65c60ac187809303077eabd4e7faf19384a46894`.
- Status: both explicitly authorized commands exited successfully.
- Scope: complete sealed V1 and census its data; not T5.2, prefix export or V2.

## Execution and integrity

Executed unchanged from `/home/zjl/mappo`:

```bash
PYTHONPATH=research/feedback_protocol/src python3 -m fpl.teacher_data \
  --plan research/feedback_protocol/configs/teacher_v1/plan.json \
  --exclusions research/feedback_protocol/configs/teacher_v1/exclusions.json \
  --output /home/zjl/fpl_teacher_v1_20260919 \
  --resume --max-new 23

python3 research/feedback_protocol/scripts/audit_teacher.py \
  --data /home/zjl/fpl_teacher_v1_20260919 \
  --output /home/zjl/fpl_teacher_v1_completed_audit
```

All 23 remaining shards were completed on this continuation. No crash, automatic
retry, raised limit or approximate label substitution. Frozen FPL runtime,
plan and exclusions match the seal. The original startup shard and seal match
their archived startup bytes. The old startup evidence was not overwritten.
No runtime/test code was edited; the only new code is a post-run descriptive
report script, outside the sealed FPL runtime. It never solves or samples.

## Full evidence

- [Complete condition/source/work tables](../research/feedback_protocol/provenance/teacher_v1_complete/RESULTS.md).
- [Machine-readable census](../research/feedback_protocol/provenance/teacher_v1_complete/summary.json).
- [Per-state CSV](../research/feedback_protocol/provenance/teacher_v1_complete/states.csv).
- [Exact action-change witnesses](../research/feedback_protocol/provenance/teacher_v1_complete/action_change_witnesses.json).
- [Unchanged auditor's result](../research/feedback_protocol/provenance/teacher_v1_complete/audit.json).
- [24 raw shards + sealed runtime archive](../research/feedback_protocol/provenance/teacher_v1_complete/teacher_v1_complete_evidence.zip).
- [Derived reporting script](../research/feedback_protocol/scripts/report_teacher_v1.py).

Archive: 312,327 bytes; SHA256
`f489508a0bcc30ee9a27b882bff0894a595ed96dd033428248fb30baffa64377`.
The original auditor still names its output `teacher_startup_evidence.zip`;
that historical name was not patched. The portable copy above has a completion
name and identical bytes/hash, containing all 24 shards.

## Outcome and limits

- 24 roots, **8 structural groups**, 273 exact labels / 273 selected states.
- Zero unresolved labels; all 120 source-collection runs report complete.
- 147 non-prior states, 100 distinct numeric belief vectors, remaining H 0–12.
- 37 states admit an optimal measuring protocol; only 27 require measurement
  in every optimal protocol. Ties are retained in 58 states.
- Crucially, **only five non-prior states have a measurement-optimal protocol**.
  All five are in hierarchy-dense, across roots 101201/101301/101401, at H 7 or 9.
  No other condition provides this intersection in the retained state corpus.
- Four conditions have zero measurement-optimal states: binary-short,
  binary-redundant, hierarchy-cost, hierarchy-four-channels.
- Source memberships overlap: exact 131, random 132, VOI 127, Beam 125, PS 120.
  PS contributes no non-prior states here. These counts are not independent samples.

Within a fixed public instance, disjoint optimal first-operation sets occur in
102 same-H/different-belief pairs (16 roots), and 29 same-belief/different-H pairs
(6 roots). Of the first class, 12 pairs involve a measurement-optimal state;
all 29 of the second do. Pairs are dependent census witnesses, not significance
tests. Shared preparation actions make this a conservative first-operation test;
it is not a full characterization of adaptive protocol planning.

The frozen sampling quotas/short rollouts restrict coverage even with no solve
failures. No observed unresolved concentration exists, but this does not show
the solver would be unbiased on a harder distribution. V1 is still a pipeline
validation corpus, not evidence of graph-generalizing student competence.
The observed five-state non-prior/sensing intersection does not justify calling
adaptive continuation coverage broad. Do not promote directly to AFPP training.

## Compute

Label-solve median / maximum:

- Solver states: 27 / 779.
- Expansions: 85 / 1,312.
- Model calls: 700 / 142,228.
- CPU seconds: 0.002055111 / 0.267866736.

Total label CPU: 4.272652064 seconds. Recorded whole-shard CPU: 7.574426173 seconds
(sampling + labeling + in-process audit, including the preserved startup shard).
Total label calls: 2,199,006; sampling calls: 1,699,992. Separate full-audit,
admission and reporting/export costs are not included in those shard totals.
No GPU training or network inference occurred. These tiny-corpus timings do not
establish an amortization benefit.

## Stopping point

Full audit delivered first, as requested. No prefix training export, student
split, Teacher V2 sizing pilot, network, test/OOD or confirmation execution.
The next proposed stage remains pure derived prefix supervision, followed by
structural/continuation-coverage expansion before any formal AFPP training.
No Git commit or push was performed.

Reproduce the descriptive report into a new directory without new solves:

```bash
python3 research/feedback_protocol/scripts/report_teacher_v1.py \
  --data /home/zjl/fpl_teacher_v1_20260919 \
  --audit /home/zjl/fpl_teacher_v1_completed_audit \
  --output /home/zjl/fpl_teacher_v1_report_replay
```
