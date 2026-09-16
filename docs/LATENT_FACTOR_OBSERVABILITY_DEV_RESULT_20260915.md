# C/R latent-factor and legal-observability DEV result

## Bottom line

The completed 96-world DEV evidence does not support “remaining battery” as the
main hidden variable.  The dominant factor is **whether continuing can finish
the current task under the fixed low-level controller without entering a
long-lived stuck trajectory**.  Operational return risk remains relevant, but
energy exhaustion is a small part of the observed C/R difference.

Legal sensor history contains descriptive signatures of this task-reachability
factor—lack of task-distance progress, low velocity toward the task, and low
overall speed—but the present low-capacity, out-of-world diagnostic does **not**
yet establish that those signatures are sufficient for reliable C/R decisions.
Exact static obstacle geometry also fails to add held-out predictive value in
the simple straight-path summaries tested here.  This is an exploratory result,
not authorization for CONFIRM access or method training.

Machine-readable result:
`artifacts/paired_advantage_identification_20260914_v3/latent_factor_observability_exploratory_v3.json`.

## What determines the observed advantage

For each anchor, the paired advantage is exactly

`Delta U = Delta task - 2 Delta operational-failure - 0.25 Delta any-contact`.

Across 192 anchors, C was better at 172 and worse at 20.  Allocating advantage
heterogeneity by `Cov(component, Delta U) / Var(Delta U)` gives:

| Additive component | Mean contribution | Heterogeneity share |
|---|---:|---:|
| Task increment | +0.91846 | 29.54% |
| Operational-failure penalty | -0.18262 | 70.46% |
| Any-contact penalty | -0.00004 | 0.001% |

The operational-failure difference is itself highly structured:

| C-minus-R failure mode | Probability difference | Mean utility contribution |
|---|---:|---:|
| `branch_guard` | +0.07642 | -0.15283 |
| `return_deadline` | +0.00960 | -0.01921 |
| `energy_exhausted` | +0.00529 | -0.01058 |

Thus `branch_guard` accounts for 83.7% of the mean operational-failure penalty.
All 939 C branches ending at the guard completed zero new tasks.  Among the 20
negative-advantage anchors, R returned in all 1,280 resamples; C produced 909
guard terminations, 186 return deadlines, 52 energy exhaustions, and only 133
returns.  The empirical factor is therefore not generic “risk” but a
controller-conditioned task-completion reachability or stuckness factor.

This is a result decomposition, not yet a causal-state proof: termination and
task increment are observed after the branch and cannot be deployed as inputs.

## Can legal history observe it?

The audit compared three small linear-ridge channels, always training on one
32-pair half, evaluating on the other half, swapping halves, and cross-fitting
by physical world:

1. 20 legal latest-frame core variables;
2. 35 legal variables adding hand-designed history trends; and
3. the same 35 legal variables plus 12 privileged exact-geometry summaries.

No recurrent network, large history encoder, actor, or critic was trained.

| Target | Constant MSE | Latest legal MSE | Legal-history MSE | Privileged-augmented MSE |
|---|---:|---:|---:|---:|
| Failure contribution | 0.34782 | 0.33298 | 0.31776 | 0.32375 |
| Task contribution | 0.06768 | 0.06373 | 0.06282 | 0.06228 |
| Full advantage | 0.68407 | 0.64480 | 0.62142 | 0.63011 |

Legal history improved full-advantage MSE over legal latest by 0.02337, but its
world-bootstrap 95% interval was [-0.04029, 0.08786].  Its selected-utility gain
over latest was 0.03097 with interval [-0.00297, 0.07308].  Against the
cross-fitted constant action, legal history gained 0.03870 with interval
[-0.00073, 0.08557].  These are promising point estimates whose intervals cross
zero; legal observability is therefore **suggested, not established**.

The strongest descriptive differences between negative and positive anchors
were legal dynamic signatures:

- energy decreased more slowly (+1.21 pooled SD), consistent with stalled rather
  than energetically demanding motion;
- the fraction of non-decreasing task-distance samples was higher (+1.07 SD);
- mean velocity toward the task was lower (-1.03 SD);
- current horizontal speed was lower (-0.94 SD); and
- current velocity toward the task was lower (-0.80 SD).

By comparison, remaining battery differed by only -0.16 SD and distance to the
charger by -0.17 SD.  These comparisons are descriptive and were inspected
after seeing DEV outcomes; they are not confirmatory feature selection.

## What the privileged control says

Exact obstacle centers and radii were used only to compute diagnostic
straight-segment clearance/count summaries for current-to-task,
current-to-charger, and task-to-charger paths.  Adding them to legal history
worsened full-advantage MSE by 0.00869 on average; the 95% interval for privileged
minus legal-history improvement was [-0.02777, 0.01184].  It also failed to
improve the failure target.

This does not prove that the legal observation is sufficient.  It says only
that these simple static straight-line geometry summaries are not the missing
oracle.  The relevant state may instead depend on controller-induced local
minima, three-dimensional maneuverability, or a longer predictive response to
actions.  That interpretation is consistent with treating partial observability
through a belief or predictive state rather than assuming a raw history vector
is automatically sufficient; see the foundational POMDP treatment by
Kaelbling, Littman and Cassandra (1998) and predictive-state formulation by
Littman, Sutton and Singh (2001).

## Research decision

Current best hypothesis:

> The sign-changing latent factor is the probability that the current task can
> be completed and followed by a successful return under the fixed controller;
> in this DEV population its dominant failure mechanism is pre-task-completion
> stuckness, not battery exhaustion.

Current observability verdict:

> Legal history contains a plausible progress/stall signal, but the completed
> DEV analysis does not yet show that it is decision-sufficient out of world.

The next experiment, if explicitly authorized later, should be one minimal
fresh-world diagnostic targeted at predicting **task completion before the
branch guard** from a prespecified progress/stall state.  It should compare a
latest-frame state against a compact legal predictive state and retain an
explicit privileged rollout oracle only as a diagnostic ceiling.  It should not
train a larger generic history model or a new actor, and it must not reuse the
unopened CONFIRM split from the stopped Gate-I lineage.

## Scope and integrity

- Existing DEV data only: 96 worlds, 192 anchors, 64 paired resamples/action.
- Raw JSON/NPZ hashes were rechecked before analysis.
- Collision semantics were unchanged; contact contributed at most once per
  policy step through the existing unified count.
- Contract validation remained valid; CONFIRM access count stayed zero.
- No training service was started and no automatic promotion occurred.
- The v1/v2 JSON files in the same artifact root are intermediate executions;
  v3 is the current result because it adds failure-mode and channel-contrast
  accounting after the same underlying computations.

## Literature used for interpretation

- Leslie P. Kaelbling, Michael L. Littman, and Anthony R. Cassandra,
  “Planning and Acting in Partially Observable Stochastic Domains,”
  *Artificial Intelligence* 101 (1998), 99–134.
  https://www.cassandra.org/arc/papers/aij98.pdf
- Michael L. Littman, Richard S. Sutton, and Satinder Singh, “Predictive
  Representations of State,” *NeurIPS 14* (2001).
  https://proceedings.neurips.cc/paper_files/paper/2001/hash/1e4d36177d71bbb3558e43af9577d70e-Abstract.html
