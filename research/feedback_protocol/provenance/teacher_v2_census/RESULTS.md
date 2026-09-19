# T5.1d — depth-two adaptive-state census

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-09-19
- Verification Status: bounded closure and label arithmetic checked
- Version Label: teacher_v2_census_v1

## Decision

**A collector gap is demonstrated; broad adaptive coverage is not.** Do not kill the V2 family on an absence claim. Prioritize a versioned measurement-closure collection design (V2.1), not an immediate V3 redesign. No next-stage collection or training is executed here.

The old collector found 1 of the 6 depth-two adaptive states; it missed 5. They span 3 structures / 2 families, but only 1 training structure. This is qualified C1 evidence, not the hypothesized discovery of dozens of adaptive states, nor proof that task economics is unproblematic.

## Exact census

| Family | Fold | Groups | States | Non-prior | Required sensing | Adaptive | Adaptive groups | Missed adaptive |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ALL | ALL | 8 | 1171 | 1115 | 19 | 6 | 3 | 5 |
| ALL | student_train | 5 | 441 | 407 | 6 | 2 | 1 | 2 |
| ALL | student_validation | 3 | 730 | 708 | 13 | 4 | 2 | 3 |
| adaptive_hierarchy | ALL | 2 | 345 | 330 | 6 | 2 | 1 | 2 |
| adaptive_hierarchy | student_train | 1 | 58 | 52 | 0 | 0 | 0 | 0 |
| adaptive_hierarchy | student_validation | 1 | 287 | 278 | 6 | 2 | 1 | 2 |
| cost_horizon | ALL | 2 | 81 | 71 | 0 | 0 | 0 | 0 |
| cost_horizon | student_train | 2 | 81 | 71 | 0 | 0 | 0 | 0 |
| decoy_redundancy | ALL | 2 | 433 | 417 | 1 | 0 | 0 | 0 |
| decoy_redundancy | student_train | 1 | 165 | 156 | 0 | 0 | 0 | 0 |
| decoy_redundancy | student_validation | 1 | 268 | 261 | 1 | 0 | 0 | 0 |
| random_resource_graph | ALL | 2 | 312 | 297 | 12 | 4 | 2 | 3 |
| random_resource_graph | student_train | 1 | 137 | 128 | 6 | 2 | 1 | 2 |
| random_resource_graph | student_validation | 1 | 175 | 169 | 6 | 2 | 1 | 1 |

32/32 closures complete; 1,171/1,171 exact labels; zero unresolved. All states were selected before labeling. No roots, objective, fold, limits or mask changed.

Depth 1: 238 states, 6 adaptive. Depth 2: 909 states, zero required sensing. Depth 0: 32 roots, 13 required sensing. Depth memberships overlap: their sum is not the unique-state total.

Old collector/census intersection: 164 states. Census adds 1,007. The census is not a superset of all 591 collector states, because the latter also includes task-interleaved and longer histories.

## Adaptive witnesses

All six occur after one measuring batch at initial H=12. Four records are two capacity-paired copies each at roots 300001 and 300129: specialist feedback is followed by required coarse sensing. The other two occur at root 300130/B5: coarse or specialist feedback is followed by another specialist. Six records are not six independent graphs or six independent mechanisms.

Every witness has a strictly positive Q gap against the best nonmeasuring **first** action, whose Q itself allows optimal later continuation. Thus this comparison is stronger than merely beating an execute-forever heuristic. Exact fractions and full positive-probability histories are in adaptive_margins.json and adaptive_witnesses.json.

## Boundaries

These are set counts, not occupancy probabilities or expected policy performance. The census exhausts at most two measuring-only batches including joint routes; it does not exhaust arbitrary task-interleaved or longer histories, and says nothing about unrun roots. It therefore cannot prove global family impossibility or uniquely identify a causal economics failure.

Closure reconstruction uses a separate normalized-belief BFS, sharing the public model/likelihood implementation. Label audits check candidate completeness, reachability and rational Q arithmetic; they are not an external independent proof of optimality.

## Work

Closure CPU: 0.074 s; label CPU: 2.358 s. Label model calls: 1,051,232. Single local run; no hardware-independent latency claim. 59 focused tests passed, including DFS/BFS equality, joint batches, cross-depth deduplication, caps and resume. No neural/DEV/test/CONFIRM run.
