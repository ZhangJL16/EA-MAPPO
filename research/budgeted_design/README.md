# Budget-conditioned history reading: public-task DEV prototype

Status: first bounded training comparison completed. [Full results](provenance/pilot_v1/RESULTS.md)
do not support a budget-query advantage in this short DEV regime.
See [handoff and limitations](../../docs/BUDGET_READING_PILOT_20260919.md).
This is a new public-task hypothesis, not a revival of stopped AFPP.
FPL/UAV runtimes, evidence, private confirmation material remain untouched.

## Task fidelity

Source: [DAD](https://proceedings.mlr.press/v139/foster21a.html) and pinned
[author code](https://github.com/ae-foster/dad/tree/4b1008174e1531d1f14601d83cef481c0f586f36).
Two independent 2D standard-Normal source locations. Unrestricted 2D designs.
Observation Normal(log(0.1 + sum_k 1/(0.0001 + squared_distance_k)), 0.5).
No reward replacement, resource constraints, graphs, teacher or inference model.
Optimize sequential prior contrastive estimation (sPCE) end-to-end through
reparameterized observations. Contrastives evaluate the actual realized designs,
not counterfactual re-planned histories. All policy inputs are public history
and (except explicitly original-architecture reference) remaining experiment count.

Published configuration is T=30, hidden=256, encoding=16, 50,000 updates,
2,000 outer and 2,000 inner samples, lr=5e-5, decay=.98 per 1,000 updates.
Our deliberately reduced DEV configuration is **not paper reproduction**:
600 updates, outer=64, inner=128, hidden=64, constant lr=3e-4, gradient norm
clip=10, three training seeds. Mixed T=4/8 alternates with equal steps.
Native-architecture DAD is trained at fixed T=8 and receives no budget feature;
it is a restoration/reference, NOT the budget-information-matched causal control.

## Mechanism and controls

- `pool`: shared pair encoder, sum pooling, output-side remaining/8.
- `fixed_attention`: budget-independent learned query, same output-side budget.
- `budget_attention`: query = learned base + (remaining/8) * learned slope,
  same output-side budget. Intervention changes ONLY query budget.
- `pool_wide`: hidden=66 rather than 64, near parameter-matched capacity control.
- `dad_native`: original sum encoder and linear emitter without budget.

Attention keys/values are the same pair encodings; no extra projection layers.
Attention output is multiplied by history length, so uniform attention equals
sum pooling. Queries start at zero. Main three models have identical initial
common weights and predictions. Fixed and conditional queries add 16 and 32
parameters respectively; wider pool adds 40. These are close, NOT exact matches.
Permutation invariance is tested. No claim this simple query is novel.

## Comparison protocol

Same three seeds, iid source/noise/independent contrastive streams coupled across
methods, optimizer/update counts and equal horizon schedule for all four budget
variants. Native fixed-T reference uses more simulator/likelihood work, reported
separately. Final predetermined checkpoint only. No best-on-eval selection.
Budget 4/8 seen; 6 interpolation and 12 extrapolation are **development probes**,
not blinded confirmation. No old FPL DEV/test/CONFIRM access.

Evaluate 512 independent rollouts per policy/H with L=4096 independent inner
particles per rollout. Report BOTH sPCE lower-bound and sNMC upper-bound
estimates, in nats. These are biased estimators of EIG, not exact MI or a confidence
interval enclosing true MI per rollout. Lower bound can saturate at log(L+1).
MC SE conditional on a checkpoint is not training-seed uncertainty. Paired method
differences use identical latent/noise/contrastive blocks. Three seeds cannot
establish population superiority. No significance/Spotlight claim.

All checkpoints store optimizer/state/step/work/log. Per-update RNG is keyed by
seed+step, so pause/resume is tested against uninterrupted execution. Config and
runtime hashes must match on resume. Evaluation shards resume without re-running
completed cells. The run is sequential on one GPU; there is no hidden teacher cost.
Inference timing includes synchronized GPU and single-thread CPU batch=1 calls;
zero online likelihood calls does not mean inference is free.

## Commands (repository root)

```bash
DAD_REFERENCE=/home/zjl/dad_public_reference_20260919 PYTHONPATH=research/budgeted_design/src python3 -m unittest discover -s research/budgeted_design/tests -v
CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH=research/budgeted_design/src python3 -m bdr.run train --config research/budgeted_design/configs/pilot_v1.json --output /home/zjl/budget_reading_pilot_v1_reproduction --device cuda
CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH=research/budgeted_design/src python3 -m bdr.run evaluate --config research/budgeted_design/configs/pilot_v1.json --output /home/zjl/budget_reading_pilot_v1_reproduction --device cuda
```

The first command can omit DAD_REFERENCE but then author-source parity is skipped.
Use `--max-updates 5` to test a resumable startup. No old dependency installation
or edits required. See THIRD_PARTY.md for source attribution.

The original `/home/zjl/budget_reading_pilot_v1_20260919` holds valid trained
checkpoints but INVALID initial evaluation (a missing `load_state_dict`). Do not
use its evaluation metrics. Corrected evaluation lives in
`/home/zjl/budget_reading_pilot_v1_corrected_20260919`, with byte-identical trained
checkpoints and explicit old/new source bindings. The narrowly scoped migration
script refuses all but the known original runtime hash. New runs use the fixed
runtime directly. A test evaluates a trained checkpoint end-to-end against a
direct computation and checks evaluation resume.

## Explicitly not delivered by this first round

Official full-budget DAD numerical reproduction; Step-DAD training/test-time
refinement; budget-specialist models for every H; second native task; OOD noise
study; formal confirmation; novelty validation. These must not be counted as
completed baselines/evidence. Whether to deepen THIS mechanism depends on actual
matched comparisons, not a new theoretical prerequisite.
