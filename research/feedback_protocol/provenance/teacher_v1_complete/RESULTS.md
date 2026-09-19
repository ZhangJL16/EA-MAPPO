# Teacher V1 full coverage audit

## Material Passport

- Origin skill: academic-research-suite / experiment-agent, inline run and descriptive collection.
- Verification: sealed full audit passed; this report performs no new exact solves.
- No independent optimality proof, population inference or student-performance claim.

## Condition coverage

Counts are root–state pairs. Non-prior, sensing, ties and beliefs below refer to exact labels.
Measurement-optimal means **at least one** optimal protocol measures; measurement-required means all do.
Belief counts are distinct numeric vectors, not necessarily the same posterior decision problem.

| Condition | Roots | Exact | Unresolved | Non-prior | Measurement-optimal | Ties | Unique belief vectors | Structural groups |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| binary-short | 3 | 17 | 0 | 2 | 0 | 4 | 4 | 1 |
| binary-long | 3 | 36 | 0 | 20 | 5 | 1 | 10 | 1 |
| binary-redundant | 3 | 30 | 0 | 12 | 0 | 0 | 12 | 1 |
| hierarchy-sparse | 3 | 37 | 0 | 21 | 3 | 20 | 14 | 3 |
| hierarchy-dense | 3 | 55 | 0 | 40 | 16 | 4 | 24 | 1 |
| hierarchy-cost | 3 | 20 | 0 | 5 | 0 | 9 | 5 | 1 |
| hierarchy-four-channels | 3 | 25 | 0 | 9 | 0 | 5 | 9 | 1 |
| hierarchy-five-channels | 3 | 53 | 0 | 38 | 13 | 15 | 27 | 1 |

Structural groups overlap across conditions; do not sum that column. Global counts:

```json
{
  "roots": 24,
  "exact": 273,
  "unresolved": 0,
  "nonprior": 147,
  "measurement_optimal": 37,
  "measurement_required": 27,
  "nonprior_measurement": 5,
  "ties": 58,
  "unique_belief_vectors": 100,
  "structural_groups": 8
}
```

## Source memberships

| Source | Final unique states | Exact | Unresolved | Non-prior | Measurement-optimal |
|---|---:|---:|---:|---:|---:|
| exact | 131 | 131 | 0 | 55 | 15 |
| random | 132 | 132 | 0 | 70 | 17 |
| voi | 127 | 127 | 0 | 35 | 15 |
| beam | 125 | 125 | 0 | 25 | 24 |
| posterior_sampling | 120 | 120 | 0 | 0 | 32 |

Memberships overlap; they are not independent samples and must not be summed as dataset size.

## Label work (all selected states)

| Metric | Median | Maximum | Total |
|---|---:|---:|---:|
| solver_states | 27 | 779 | 23576 |
| expansions | 85 | 1312 | 58340 |
| model_calls | 700 | 142228 | 2199006 |
| cpu_seconds | 0.002055111 | 0.267866736 | 4.27265206 |
| wall_seconds | 0.00205493 | 0.267885208 | 4.27261104 |

Whole-shard recorded CPU: 7.57442617 s; includes sampling and in-process auditing.
Label CPU above is only exact V/Q generation. Separate full-audit/report/export cost is excluded.
Root-state work distribution is descriptive, not a sample of independent task difficulties.

## Action-change witnesses

Within one fixed public instance, compare H>0 states with either identical H and different belief,
or identical belief and different H. Count only **disjoint optimal first-operation sets**;
mere tie-breaking changes are not counted. These pairs are dependent and not population evidence.

```json
{
  "belief": {
    "pairs": 102,
    "roots": 16,
    "measurement_pairs": 12
  },
  "horizon": {
    "pairs": 29,
    "roots": 6,
    "measurement_pairs": 29
  }
}
```

Full pairs are in action_change_witnesses.json. CSV retains optimal full protocols and input identities.

## Selection and limitations

Sampling status: {'complete': 120}. Truncation detail and yield by horizon/channel/source are in summary.json.
With no unresolved selected labels, there is no observed solver-status selection bias within this corpus.
This does not rule out sampling caps, hash-priority selection or restriction to small/easy tasks.
Unresolved concentration in hypothetical harder/more informative states cannot be inferred from zero failures.
Measurement-optimal states need not require deep adaptive sensing; first-operation witnesses are a conservative
diagnostic, not proof of graph-generalizing planning competence. Eight structures remain insufficient evidence
for that claim. No network, prefix export, Teacher V2, test/OOD/confirm execution or new gate was run.

## Evidence

Archive SHA256: `f489508a0bcc30ee9a27b882bff0894a595ed96dd033428248fb30baffa64377`.
The archive has the historical internal filename convention from the unchanged audit script but contains all 24 shards.
The new portable filename is teacher_v1_complete_evidence.zip; startup evidence is untouched.
