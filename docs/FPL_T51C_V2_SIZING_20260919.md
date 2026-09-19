# T5.1c completed: Teacher V2 sizing and catalogue-free decoder mask

## Outcome

**Implementation and exact-mask regression pass. Teacher coverage does not.**
The frozen sizing pilot falls in the user's **Case C**: exact labels are cheap
and complete, yet post-feedback required sensing is almost absent. Do not run
the remaining 24 structures and do not start AFPP training. This does not undo
the earlier scoped B2-necessity GO; it rejects promotion of this teacher design.

| Pilot statistic | Result |
|---|---:|
| Structural identities frozen before labels | 32 |
| H/B variants admitted | 128 |
| Pilot groups / completed variants | 8 / 32 |
| Selected / exact / unresolved | 591 / 591 / 0 |
| Non-prior exact states | 358 |
| Measurement-optimal / required | 50 / 41 |
| Non-prior + measurement-required | **1** |
| Adaptive structural groups | **1** |
| Train / validation adaptive states | **0 / 1** |
| V1 / V2 exact prefix checks | 5,383 / 6,849 |
| Mask mismatches | **0** |

### Family coverage

| Family | Groups | Exact | Non-prior | Required | Non-prior + required | Adaptive groups |
|---|---:|---:|---:|---:|---:|---:|
| adaptive_hierarchy | 2 | 134 | 78 | 10 | 0 | 0 |
| random_resource_graph | 2 | 213 | 157 | 29 | 1 | 1 |
| cost_horizon | 2 | 99 | 39 | 0 | 0 | 0 |
| decoy_redundancy | 2 | 145 | 84 | 2 | 0 | 0 |

The metadata-only fold assignment produced 5 train and 3 validation pilot
groups (the full admitted universe is 24 train / 8 validation). No reassignment
was made. Train has 317 states, validation 274. Their adaptive counts remain
0 and 1. The full-corpus 6/4/2 target is not being mechanically applied to an
eight-group pilot: the decisive shortfall is concentration in one validation
structure and **zero** training examples of required continued sensing.

## What the explorer changed—and what it did not

Information explorer contributed 184 final-state memberships, including 150
non-prior states and 128 states exclusive to that source. It supplied the only
adaptive state (`root=300130, H_initial=12, B=5`, remaining H=8, posterior
`[6/13,6/13,1/39,2/39]`). It therefore improves posterior coverage; it has not
solved adaptive-supervision coverage. Source memberships overlap and do not
represent additional independent samples.

All union-pool states were retained: sum pool size = 591, maximum per variant
= 31 < frozen cap 48. Thus **the retention quota did not remove adaptive
states from these collected pools**. The study does not exhaust all reachable
belief states, so absence in the corpus is not an impossibility theorem.
Two sequential sensing batches being physically feasible does not imply that
their task value repays acquisition/recovery time. The next design discussion
must examine this distinction rather than double the root count or swap folds.

## Compute and unresolved semantics

- Exact-label CPU: 2.472 s total; median 0.00146 s; maximum 0.105 s.
- Label work: 64,532 expansions, 1,142,672 model calls.
- Label solver states: median 17, maximum 713 (cap 3,000).
- Collection CPU: 4.730 s; 336,252 expansions; 2,591,792 model calls.
- 183/192 source collections completed without a limit; 9 Beam collections
  truncated at the frozen episode work caps (4 expansion, 5 model-call).
  Their previously collected states were retained. This is distinct from
  **zero label failures**. No retries or cap changes were performed.
- Timings are one local CPU run, not population estimates or hardware claims.

## Decoder correctness and complexity

New code is isolated in `src/fpl_v2`; frozen V1 source bytes were not changed.
Joint return DP includes time, energy and channel counts. It does not load a
teacher catalogue, and catalogue calls are patched to fail during mask tests.
All 12,232 saved exact prefixes match, including STOP and END. Tests also
cover incompatible shortest-time/minimum-energy returns, measurement caps,
independent RNG streams, paired variants, utility-free explorer ranking,
admission/fold isolation, resume and explicit interruption refusal.

DP avoids materializing complete paths; it is not a polynomial-time solution
to arbitrary resource-constrained routing. Its general state space can still
grow exponentially with channels. Work-limit failure is explicit unresolved,
never a silently pruned legal action.

## Provenance and boundaries

Implementation choices were documented in
[CONTRACT_T51C_V2.md](../research/feedback_protocol/CONTRACT_T51C_V2.md) before
admission/labels. The original design JSON and SHA256 remain unchanged.
Detailed shared-cleanup costs take precedence over its informal “total 3”
phrase; cost-horizon uses p=1/2 for the unspecified transit probability.
The complete admission is sealed before any scientific teacher collection.
All selected-state checkpoints precede label generation.

Full audit includes public-history replay, likelihood/Q arithmetic, exhaustive
candidate completeness, source/admission/shard/selection hashes, and mask
regression. It is **not an external independent proof of teacher optimality**.
The first audit invocation encountered a boolean `loss_mask` parsing bug in
the new report script; that report-only line was fixed before any audit output
was written. No collection, label, frozen runtime or limit was changed or rerun.

56 focused tests passed. Old tests use temporary synthetic fixtures; they do
not execute the formal DEV/test/OOD/confirmation schedules. No neural training,
new DEV evaluation, private-confirm access, commit or push was performed.

## Commands and evidence

Working directory: `/home/zjl/mappo`; `PYTHONPATH=research/feedback_protocol/src`.

```bash
python3 -m fpl_v2.runtime admit \
  --design research/feedback_protocol/configs/teacher_v2_design_v1.json \
  --admission /home/zjl/fpl_teacher_v2_admission_20260919 \
  --exclusions research/feedback_protocol/configs/teacher_v1/exclusions.json \
  --v1-admission research/feedback_protocol/configs/teacher_v1/admission.json
python3 -m fpl_v2.runtime run \
  --design research/feedback_protocol/configs/teacher_v2_design_v1.json \
  --admission /home/zjl/fpl_teacher_v2_admission_20260919 \
  --output /home/zjl/fpl_teacher_v2_pilot_20260919 --max-new 1
# Startup arithmetic replay passed; authorized unchanged continuation:
python3 -m fpl_v2.runtime run \
  --design research/feedback_protocol/configs/teacher_v2_design_v1.json \
  --admission /home/zjl/fpl_teacher_v2_admission_20260919 \
  --output /home/zjl/fpl_teacher_v2_pilot_20260919 --resume --max-new 31
python3 research/feedback_protocol/scripts/audit_teacher_v2.py \
  --admission /home/zjl/fpl_teacher_v2_admission_20260919 \
  --data /home/zjl/fpl_teacher_v2_pilot_20260919 \
  --output /home/zjl/fpl_teacher_v2_pilot_audit_20260919
```

Portable evidence is under `research/feedback_protocol/provenance/teacher_v2_sizing/`:
full admission/public TRAIN problems, 32 raw shards, prelabel selections,
source logs, per-state/per-work audit, exact runtime source snapshot and hashes.
The top-level `summary.json` and `RESULTS.md` provide family × fold/source tables.
The remaining 96 variants have admission records only, not labels.

**Next authorization is a revised teacher-family design discussion, not a full
V2 run, fold adjustment, neural training or a new B2-necessity experiment.**
