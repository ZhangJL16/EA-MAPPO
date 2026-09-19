# T5.1d completed — collection gap exists, adaptive coverage remains sparse

## Result

The depth≤2 measuring-only census completed for **all 32 existing pilot
variants**, without changing roots, graphs, objectives, folds or solver limits.
There are **1,171 exact states, zero unresolved, and no incomplete closures**.

| Quantity | Old collector | Depth-two census |
|---|---:|---:|
| Unique states within variants | 591 | 1,171 |
| Non-prior states | 358 | 1,115 |
| Non-prior + required sensing | 1 | **6** |
| Adaptive structural groups | 1 | **3** |
| Adaptive train states / groups | 0 / 0 | **2 / 1** |
| Adaptive validation states / groups | 1 / 1 | **4 / 2** |

These sets are **not nested**: the old collector also visits task-interleaved
and longer histories. Their intersection is 164 states. The census adds 1,007
states and recovers **five adaptive states missed by the original collector**.
All six adaptive states arise at depth 1, with initial horizon 12. None of the
depth-2 states requires a further measurement.

| Family | Census states | Non-prior | Adaptive states | Adaptive groups |
|---|---:|---:|---:|---:|
| adaptive_hierarchy | 345 | 330 | 2 | 1 |
| random_resource_graph | 312 | 297 | 4 | 2 |
| cost_horizon | 81 | 71 | 0 | 0 |
| decoy_redundancy | 433 | 417 | 0 | 0 |

## Diagnosis and next branch

**Qualified C1: collector failure is demonstrated, but it is not the whole
coverage problem.** Multiple V2 structures really contain required continued
sensing, and the old collector missed most of those census states. Therefore
the previous Case C must not be promoted into a claim that the V2 family has
essentially no such states under all relevant histories.

However, this is six records—not dozens—and four of them are capacity-paired
copies from two roots. Only one training structure has adaptive examples.
The study does not establish sufficient supervision for a transferable graph
planner, nor rule out task economics as a contributor to scarcity.

The evidence-supported next branch is to **design V2.1 measurement-closure
collection on the already frozen structures, before redesigning the family**.
Do not immediately generate Teacher V3, expand the remaining groups or train
AFPP. This task implements and executes the census only. It does not silently
convert diagnosis into authorization for another run.

## Exact witnesses

All six have strictly positive exact Q gaps against the best nonmeasuring
first action, allowing that comparator optimal later continuation.

- Root 300001, H12, B3/B5: a specialist outcome leads to required coarse sensing.
- Root 300129, H12, B3/B5: another specialist outcome likewise requires coarse
  sensing. This is the only adaptive training structure.
- Root 300130, H12, B5: two different post-observation states require a specialist;
  the old information explorer found one and missed the other.

Full histories, posterior fractions, optimal protocols and strict margins are
in the [witnesses](../research/feedback_protocol/provenance/teacher_v2_census/adaptive_witnesses.json)
and [margins](../research/feedback_protocol/provenance/teacher_v2_census/adaptive_margins.json).
The observed specialist→coarse cases also caution against assuming every useful
adaptive sequence must follow the intuitive coarse→specialist order.

## Scope, verification and cost

Depth counts completed measurement batches, including legal joint routes.
Every positive-probability outcome is expanded. Deduplication is per sufficient
state per variant, with separate expansion for each reachable depth. The census
contains no intervening task/idle batch and stops at depth 2. Therefore it does
not prove absence over longer or task-interleaved histories, nor over unrun
roots. Counts are reachable-set counts, not occupancy-weighted policy risks.

The prelabel closure is checkpointed. A separately implemented normalized-belief
BFS reconstructs all state/depth memberships and incoming edges. All labels
undergo reachability, candidate-completeness and rational arithmetic replay.
All 164 previously labeled intersections agree on exact V and optimal sets.
These checks share the public model implementation; they are not an external
independent proof of Bayes optimality.

Closure CPU totals 0.074 seconds; label CPU 2.358 seconds, 1,051,232 label model
calls. This is one local run, not a statistical latency comparison. **59 tests
pass**, including full-history DFS versus deduplicated closure, joint batches,
cross-depth arrivals, caps and unchanged resume. No retries, raised caps,
root replacement, fold changes, neural training or formal DEV/test/CONFIRM use.

## Reproduction and evidence

Working directory `/home/zjl/mappo`; set
`PYTHONPATH=research/feedback_protocol/src`.

```bash
python3 research/feedback_protocol/scripts/census_teacher_v2.py run \
  --archive research/feedback_protocol/provenance/teacher_v2_sizing/teacher_v2_sizing_evidence.zip \
  --plan research/feedback_protocol/configs/teacher_v2_census_v1.json \
  --output /home/zjl/fpl_teacher_v2_census_20260919 --max-new 1
# Startup closure + label replay passed; finish the authorized bounded census:
python3 research/feedback_protocol/scripts/census_teacher_v2.py run \
  --archive research/feedback_protocol/provenance/teacher_v2_sizing/teacher_v2_sizing_evidence.zip \
  --plan research/feedback_protocol/configs/teacher_v2_census_v1.json \
  --output /home/zjl/fpl_teacher_v2_census_20260919 --resume --max-new 31
python3 research/feedback_protocol/scripts/census_teacher_v2.py analyze \
  --archive research/feedback_protocol/provenance/teacher_v2_sizing/teacher_v2_sizing_evidence.zip \
  --data /home/zjl/fpl_teacher_v2_census_20260919 \
  --output /home/zjl/fpl_teacher_v2_census_audit_20260919
```

[Full tables and interpretation](../research/feedback_protocol/provenance/teacher_v2_census/RESULTS.md),
[machine-readable summary](../research/feedback_protocol/provenance/teacher_v2_census/summary.json),
and portable archive `teacher_v2_census_evidence.zip` are under
`research/feedback_protocol/provenance/teacher_v2_census/`.
Archive SHA256: `fe6cb38bc8f6ec8b04dd9284da6123f55b0714c4e767a7635c544abd4e6f6807`.
It contains closure edges, all exact labels, original problem bytes, source
snapshot, contract, plan and per-file hashes. Old V1/V2 source and evidence
remain unchanged. Work is local; no commit or push was performed in this task.
