# T5.1e completed — Route C on the current evidence

## Decision

**Stop the current learned-planner main line.** Do not start full AFPP, a learned
residual selector, V2.1 expansion or a V3 curriculum. Preserve the resource
separation theory and FPL planning/evaluation infrastructure.

This is a scoped research decision, not a mathematical impossibility result
about amortization. The prior fourth-batch DEV observations are not invalidated;
they are also not sufficient to establish a learned method's necessity on this
teacher family. This task did not run DEV 2201/2202 or claim to decompose their
specific gaps.

## Exact results

All 1,171 census states were probed with four frozen planners, yielding 4,684
exact decision-residual rows. All 128 original-prior episode integrations passed
both rational Bellman telescoping and the unchanged hypothesis-wise
ExactPolicyEvaluator cross-check. No unresolved result, retry or cap increase.

| Planner | Fresh nonzero residuals | Episode gap > 0 | Mean episode gap |
|---|---:|---:|---:|
| Small VOI | 9 / 1,171 | 20 / 32 | 0.754581 |
| Large VOI | **0 / 1,171** | **0 / 32** | **0** |
| Small Beam | 203 / 1,171 | 28 / 32 | 1.238922 |
| Large Beam | 0 / 1,171 | 1 / 32 | 0.00625 |

The mean equally weights the 32 paired H/B variants for description, with eight
independent structural groups. It is a Bayes utility gap, not minimax regret
or a population superiority estimate.

### What explains Small VOI's deployment gap?

- **100%** of its positive residual contribution accompanies a work-limit hit.
- **79.7229%** comes from explicitly flagged score-blind `WorkPolicy` autopilot.
- **82.5589%** occurs at states where a nonmeasuring first action is optimal.
- **17.4411%** occurs at measurement-required states, all with the prior belief.
- The **non-prior + measurement-required subset contributes exactly zero**:
  none of the four policies' actual trees visits such a state.

These are separate overlapping classification dimensions, not additive shares.
Limit flags are evidence of the tested implementation's truncation context, not
a proof that one particular engineering repair uniquely eliminates the loss.
All positive-residual chosen actions are task actions. Small Beam similarly
has 100% limit-associated loss and 70.82% at nonmeasuring-optimal states.

Thus the six adaptive census states are real, but they do not explain this
study's deployed policy gaps. Increasing their curriculum count would optimize
the wrong proxy for these results.

### Fresh work versus actual work remaining

Small VOI makes only nine fresh-pool census errors (eight prior-belief, one
non-prior), but suffers larger episode losses after consuming its episode pool.
Large Beam has zero fresh census errors yet a 1/5 episode gap on one variant.
This is why `pi(b,H)` cannot be used without recording internal remaining work
when decomposing these budgeted policies.

Large VOI already achieves exact value at 9,480.87 mean model calls per episode,
versus Small VOI's 5,087.74. Single-run mean summed planning latency was roughly
16.6 ms versus 9.0 ms. These are local diagnostics, not hardware-independent
claims. The data do not justify building a learned selector to fix a loss
dominated by fallback/execution behavior, while this ordinary planner is exact.

## Scope and implementation

No roots, objective, graph, folds, old mask or frozen planner code changed.
The same fourth-batch small/large limits, width 8, Beam depth 2 and 4× episode
budget were fixed in `planning_residual_v1.json` before running.

Fresh-state probes and actual occupancy are explicitly separate. Feedback
branches deep-copy the owned policy state; the episode pool is not reset.
STOP has Q=0 and residual V*. Every exact episode satisfies:

`sum_context occupancy * residual = V*(root) - V_policy(root)`.

Actual trajectories leave the measurement-only census through task-interleaved
histories. Accordingly 452 evaluation-only supplemental V/Q states were solved
on the **same 32 problems**, all exact under unchanged limits. They are logged
separately and were not turned into a new teacher corpus.

State-level occupancy aggregation retains context IDs because the same
sufficient state can have distinct actions under different remaining compute.
The unweighted small-minus-large decision residuals are not advertised as
episode gains; episode gap differences use each policy's own occupancy.

## Deliverables

Under `research/feedback_protocol/provenance/planning_residual_v1/`:

- [state_residuals.jsonl](../research/feedback_protocol/provenance/planning_residual_v1/state_residuals.jsonl)
- [occupancy_decomposition.json](../research/feedback_protocol/provenance/planning_residual_v1/occupancy_decomposition.json)
- [small_vs_large.json](../research/feedback_protocol/provenance/planning_residual_v1/small_vs_large.json)
- [RESULTS.md](../research/feedback_protocol/provenance/planning_residual_v1/RESULTS.md)
- `summary.json`, `adaptive_contribution.json`, `supplemental_labels.json`,
  `decision.json`, and `planning_residual_evidence.zip`.

Archive SHA256:
`c260a6600bf3365c7dc46410b7e979793dcd951400bfce67e5560be74efe9572`.
The archive includes raw per-variant receipts, runtime bytes, full result files,
plan, contract, tests and per-file hashes. An earlier report-only package is
retained outside the repository; adding the explicit adaptive-subset summary
did not rerun or modify any experiment.

**63 tests passed.** Full experiment execution was startup-checked, completed
unchanged, and source-sealed. No neural training, V2.1/V3, formal DEV/test/OOD,
CONFIRM, commit or push occurred.

## Commands

Working directory `/home/zjl/mappo`; set
`PYTHONPATH=research/feedback_protocol/src`.

```bash
python3 research/feedback_protocol/scripts/planning_residual.py \
  --archive research/feedback_protocol/provenance/teacher_v2_census/teacher_v2_census_evidence.zip \
  --plan research/feedback_protocol/configs/planning_residual_v1.json \
  --output /home/zjl/fpl_planning_residual_20260919 --max-new 1
# First variant: four identities/cross-checks passed; authorized continuation:
python3 research/feedback_protocol/scripts/planning_residual.py \
  --archive research/feedback_protocol/provenance/teacher_v2_census/teacher_v2_census_evidence.zip \
  --plan research/feedback_protocol/configs/planning_residual_v1.json \
  --output /home/zjl/fpl_planning_residual_20260919 --resume --max-new 31
python3 research/feedback_protocol/scripts/analyze_planning_residual.py \
  --data /home/zjl/fpl_planning_residual_20260919 \
  --output /home/zjl/fpl_planning_residual_analysis_20260919
```
