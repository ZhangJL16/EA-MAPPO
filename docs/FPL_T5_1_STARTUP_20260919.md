# T5.1 exact teacher pipeline: implemented, first checkpoint only

## Material Passport

- Origin skill: academic-research-suite / experiment-agent, inline execution.
- Origin date: 2026-09-19.
- Verification status: focused tests and completed-checkpoint replay passed.
- Scope: implementation/startup evidence, not completed fifth batch or scientific superiority.
- Base commit: `dceb2675a9b4cbb3186b955840e5786743813d87`.

The user authorized B2, beginning with the teacher dataset, not a network.
The experimental workflow keeps inputs/evidence frozen and claims bounded.
The repository startup rule stops execution after first-checkpoint health;
there is no background collector or neural training running.

## Delivered

- [Teacher contract](../research/feedback_protocol/CONTRACT_V05_TEACHER.md).
- [Exact label / mixed-history / resume implementation](../research/feedback_protocol/src/fpl/teacher_data.py).
- [Frozen TRAIN plan](../research/feedback_protocol/configs/teacher_v1/plan.json),
  [exclusions](../research/feedback_protocol/configs/teacher_v1/exclusions.json),
  [admission](../research/feedback_protocol/configs/teacher_v1/admission.json).
- [Preparation script](../research/feedback_protocol/scripts/prepare_teacher.py),
  [replay/export script](../research/feedback_protocol/scripts/audit_teacher.py).
- [Tests](../research/feedback_protocol/tests/test_teacher_data.py).
- [Startup audit](../research/feedback_protocol/provenance/teacher_v1_startup/audit.json)
  and [portable raw checkpoint + sealed runtime](../research/feedback_protocol/provenance/teacher_v1_startup/teacher_startup_evidence.zip).

All 24 planned root slots admitted. One candidate was rejected for a reserved
structural clone, before label solving. No exhausted strata. Existing DEV,
public test/OOD and original scientific runtime/evidence were not modified.
Only public registry metadata was used for exclusions; no private confirmation
seed material, test instance contents or test outcomes were opened.

## Measured startup, not full coverage

| Quantity | Observed |
|---|---:|
| Completed root shards | 1 / 24 |
| Exact labels | 7 |
| Unresolved labels | 0 |
| Non-prior posterior states | 2 |
| Distinct beliefs | 3 |
| Remaining horizons | 0, 1, 2, 3, 4 |
| Optimal measurement states | 0 |
| Graph structural groups | 1 |
| Recorded shard CPU seconds | 0.005905 |

The short binary root does not need optimal sensing. It remains in the dataset;
there is no outcome-driven replacement. Source memberships overlap after
sufficient-state deduplication, so their counts must not be added as independent
samples. All seven labels passed arithmetic/likelihood/history replay. Full suite:
**43 tests passed**, including six new tests and temporary-directory pause/resume.

This establishes implementation health only. The remaining 23 roots are not
generated into teacher shards yet, so graph/hypothesis/cost coverage and dataset
sufficiency remain unmeasured. T5.1 is not declared fully complete; T5.2 learned
architecture, learned baselines, ablations and DEV comparison have not started.

## Resume exactly this sealed dataset

Run from `/home/zjl/mappo` without changing FPL runtime/config:

```bash
PYTHONPATH=research/feedback_protocol/src python3 -m fpl.teacher_data \
  --plan research/feedback_protocol/configs/teacher_v1/plan.json \
  --exclusions research/feedback_protocol/configs/teacher_v1/exclusions.json \
  --output /home/zjl/fpl_teacher_v1_20260919 \
  --resume --max-new 23
```

`--max-new 1` instead advances one root. The prepared limits allow up to 30
retained states/root (usually fewer due to overlap), 5 seconds per exact label
solve and bounded collection solves. These are caps, not a wall-time estimate or
a measured full-generation bill. No retries automatically increase them.

After an authorized continuation, audit into a **new** directory:

```bash
python3 research/feedback_protocol/scripts/audit_teacher.py \
  --data /home/zjl/fpl_teacher_v1_20260919 \
  --output /home/zjl/fpl_teacher_v1_completed_audit
```

Do not overwrite the startup evidence. Complete coverage/status/work must be
reviewed before using these labels for a student. This is data-quality review,
not a renewed B2-necessity gate. No commit or push was performed in this task.
