# Teacher V2 sizing pilot

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-09-19
- Verification Status: arithmetic/reachability and mask regression checked
- Version Label: teacher_v2_sizing_v1

32/32 pilot variants completed; 32 identities / 128 variants admitted before labels. The remaining 24 structures / 96 variants were not labeled. No neural training or DEV/CONFIRM access.

## Coverage (exact-state categories; unresolved kept)

| Family | Fold | Groups | Variants | Selected | Exact | Unresolved | Non-prior | Required | Non-prior + required | Adaptive groups |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | ALL | 8 | 32 | 591 | 591 | 0 | 358 | 41 | 1 | 1 |
| ALL | student_train | 5 | 20 | 317 | 317 | 0 | 168 | 10 | 0 | 0 |
| ALL | student_validation | 3 | 12 | 274 | 274 | 0 | 190 | 31 | 1 | 1 |
| adaptive_hierarchy | ALL | 2 | 8 | 134 | 134 | 0 | 78 | 10 | 0 | 0 |
| adaptive_hierarchy | student_train | 1 | 4 | 38 | 38 | 0 | 10 | 0 | 0 | 0 |
| adaptive_hierarchy | student_validation | 1 | 4 | 96 | 96 | 0 | 68 | 10 | 0 | 0 |
| cost_horizon | ALL | 2 | 8 | 99 | 99 | 0 | 39 | 0 | 0 | 0 |
| cost_horizon | student_train | 2 | 8 | 99 | 99 | 0 | 39 | 0 | 0 | 0 |
| decoy_redundancy | ALL | 2 | 8 | 145 | 145 | 0 | 84 | 2 | 0 | 0 |
| decoy_redundancy | student_train | 1 | 4 | 63 | 63 | 0 | 30 | 0 | 0 | 0 |
| decoy_redundancy | student_validation | 1 | 4 | 82 | 82 | 0 | 54 | 2 | 0 | 0 |
| random_resource_graph | ALL | 2 | 8 | 213 | 213 | 0 | 157 | 29 | 1 | 1 |
| random_resource_graph | student_train | 1 | 4 | 117 | 117 | 0 | 89 | 10 | 0 | 0 |
| random_resource_graph | student_validation | 1 | 4 | 96 | 96 | 0 | 68 | 19 | 1 | 1 |

## Source membership (overlaps, not independent samples)

| Source | Fold | Selected | Exact | Non-prior | Non-prior + required | Adaptive groups |
|---|---|---:|---:|---:|---:|---:|
| beam | ALL | 211 | 211 | 77 | 0 | 0 |
| beam | student_train | 136 | 136 | 27 | 0 | 0 |
| beam | student_validation | 75 | 75 | 50 | 0 | 0 |
| exact | ALL | 238 | 238 | 98 | 0 | 0 |
| exact | student_train | 144 | 144 | 28 | 0 | 0 |
| exact | student_validation | 94 | 94 | 70 | 0 | 0 |
| one_step_voi | ALL | 237 | 237 | 91 | 0 | 0 |
| one_step_voi | student_train | 144 | 144 | 28 | 0 | 0 |
| one_step_voi | student_validation | 93 | 93 | 63 | 0 | 0 |
| posterior_sampling | ALL | 224 | 224 | 0 | 0 | 0 |
| posterior_sampling | student_train | 140 | 140 | 0 | 0 | 0 |
| posterior_sampling | student_validation | 84 | 84 | 0 | 0 | 0 |
| public_information_explorer | ALL | 184 | 184 | 150 | 1 | 1 |
| public_information_explorer | student_train | 98 | 98 | 76 | 0 | 0 |
| public_information_explorer | student_validation | 86 | 86 | 74 | 1 | 1 |
| random | ALL | 234 | 234 | 77 | 0 | 0 |
| random | student_train | 148 | 148 | 53 | 0 | 0 |
| random | student_validation | 86 | 86 | 24 | 0 | 0 |

## Mask and audit

V1: 5383 prefixes; V2 pilot: 6849 prefixes; **0 mismatches**. Catalogue calls are patched to fail during mask checks. Only audit/reference paths enumerate.

Joint DP avoids complete path materialization, not worst-case combinatorial complexity. Work/state caps are explicit unresolved conditions, not false infeasibility.

The exact label audit replays public histories, likelihoods, candidate completeness and Q arithmetic. It does not independently prove successor optimality.

## Execution limits and interpretation

Exact: 591/591; unresolved: 0; adaptive structures: 1/8. No cap increases, root replacements or retries.

Full-corpus targets 6 total / 4 train / 2 validation are not mechanically applied to this sizing pilot. Full run requires a separate decision; no automatic continuation. See source_status and work in summary.json for collection truncation, solve cost and selected-state coverage. Source truncation and label failure are distinct.

Targets remain Bayes values. Hierarchical Bernoulli-family coverage is not general experimental-planning evidence.
