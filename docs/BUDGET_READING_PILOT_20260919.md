# Public-task prototype: first actual budget-reading comparison

## Outcome

The requested three-way learning comparison has been implemented and run.
**This short DEV experiment does not support budget-conditioned history reading
as an improvement over output-side budget concatenation or fixed-query attention.**
This is not a claim about converged models or all possible budget-aware mechanisms.
No follow-up training, extra synthetic curricula or theorem escalation was started.

| Method | H=4 | H=6 | H=8 | H=12 |
|---|---:|---:|---:|---:|
| Sum history + output budget | 2.6573 | 3.6859 | 4.1748 | 5.1472 |
| Fixed-query attention + output budget | 2.6350 | 3.6319 | 4.1578 | 5.0480 |
| Budget-query attention + output budget | 2.6265 | 3.5919 | 4.0879 | 5.0022 |

Entries are mean sPCE lower-bound estimates, nats, across three trained seeds,
not exact mutual information. Higher is better. H=4/8 were in training; H=6/12
are developer interpolation/extrapolation probes, NOT final confirmation.
Seed SD, paired seed differences, conditional MC SE, upper-bound estimates,
near parameter-matched wider control, native-architecture reference and costs
are in the [complete generated report](../research/budgeted_design/provenance/pilot_v1/RESULTS.md).

Query-only constant/zero interventions retain the correct output-side budget.
The candidate's mean improvement over zero query-budget is only
0.0014 / -0.0042 / -0.0136 / -0.0031 nats across the four horizons.
This does not provide a stable beneficial mechanism signal. No p-value or
superiority claim is made from three training seeds.

## Delivered

- Isolated [Torch package](../research/budgeted_design/README.md), public model,
  reparameterized sequential sPCE objective, history-set policies, resume and evaluation.
- 5 models × 3 seeds × 600 updates = 15 actual trained checkpoints. Four
  budget-aware controls share an alternating 4/8 horizon schedule, optimizer,
  simulated source/noise/contrastive streams, and common initial weights where
  shapes match. Wider pool has 1,372 parameters versus candidate 1,364; not an
  exact parameter match. Fixed-H native DAD is a separate restoration reference.
- 84 evaluation cells (60 normal + 24 query interventions), 512 independent
  rollouts per cell, 4,096 independent contrastives per rollout. All seeds kept.
- Nine tests: author network/model parity, source permutation, history invariance,
  initial equality, query intervention, likelihood/gradient checks, training
  resume and checkpoint-based evaluation/resume. All passed.
- Exact operation-count accounting, synchronized training wall time (~150.6s),
  CPU/GPU batch=1 inference latency. No teacher compute; no online search.

## Restoration boundary

The [DAD author code](https://github.com/ae-foster/dad/tree/4b1008174e1531d1f14601d83cef481c0f586f36)
defines the two-source 2D standard-Normal prior, unrestricted 2D designs, and
Normal(log intensity, 0.5) observations used here. Pure Torch architecture and
forward-map definitions were executed for numerical parity tests.

The original Pyro 1.6/MLflow stack was NOT executed. This is a source-matched
Torch-only reimplementation, not a published-score replication. Original
training used T=30, hidden=256, 50,000 updates and 2,000/2,000 outer/inner
samples. Our hidden=64, 600 updates and 64/128 samples are deliberately small.
Mixed-budget training and added query are explicit experimental changes.

The full Step-DAD comparison, per-H specialist networks, second public task,
noise generalization, full-budget DAD reproduction and final confirmation are
**not completed**. This first round cannot be used to claim a strong-baseline
win, new methodology, or readiness for any venue.

## Disclosed error and recovery

The initial evaluator instantiated the right model but forgot to load checkpoint
weights. Its evaluations describe initialization and are **invalid** for trained
model comparison. They remain at `/home/zjl/budget_reading_pilot_v1_20260919`,
with `INVALID_EVALUATION_NOTICE.json`; they are not in the result table.

Training was valid. All 15 checkpoint files were copied byte-for-byte to
`/home/zjl/budget_reading_pilot_v1_corrected_20260919`. The evaluator now calls
`load_state_dict(strict=True)`. Migration checks the exact known old runtime
hash, unchanged model/objective/config, complete checkpoints and tensor identity.
New end-to-end tests compare saved evaluation with direct trained-policy output.
No retraining, favorable seed replacement or best-checkpoint selection occurred.

Initial invalid evaluation overlapped the end of the last training seed, so
training times are cost accounting, not controlled speed evidence. Corrected
inference latencies were measured after training. These are small batch=1
microbenchmarks, not whole-episode or FLOP-equivalent comparisons.

## Evidence and handoff

- [Machine-readable summary](../research/budgeted_design/provenance/pilot_v1/summary.json)
- [Source provenance](../research/budgeted_design/provenance/pilot_v1/provenance.json)
- [Portable checkpoint/evaluation/source package](../research/budgeted_design/provenance/pilot_v1/pilot_evidence.zip)
- [Checksums](../research/budgeted_design/provenance/pilot_v1/SHA256.json)

Archive SHA256:
`12dcdf82c58b511fb49f262a180e4e967a3dee9d32a19d884670a9522ecf6916`.
It includes all 15 trained optimizer checkpoints, per-rollout estimates,
old training/new evaluation source bindings, executable sources, tests and
reviewed author-code references with MIT license. The archive's README is the
pre-handoff version; this document adds interpretation, not new run data.

Reproduction commands are in the package README; use a fresh output directory.
The existing raw training directory intentionally rejects a changed source hash.
No new dependency installation, Git commit/push, FPL runtime/evidence changes,
old DEV/test/CONFIRM reads, new teacher roots, or neural AFPP run occurred.

## Research decision

Keep this as the first real empirical test of a simple mechanism. Do not use
the small numerical improvements over selected weaker controls to sell a budget
reasoning mechanism: the two closest controls explain or exceed the performance.
Also do not turn an undertrained negative pilot into a theorem that the idea
cannot work. Any continuation should test a concrete optimization or task
limitation, with strong baselines, not add modules or fabricate hard examples
solely to rescue the story. The old separation theory remains frozen support.
