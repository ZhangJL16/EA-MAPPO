# EA-MAPPO research workspace

Final user decision (2026-09-19): **the entire regenerative-control research branch
is CLOSED to new research**, including a third statistical/finite-sample pivot.
Standalone theory assessment: **Level 1/5, workshop/supporting theory**; no current
ICML Spotlight/Oral central result. Preserve the environment, frozen SAC, P0–P2
negative evidence, exact tree and certificate as research assets. Do not expand,
retrain, change the task family, or repackage this branch to rescue a paper.
The next research activity is independent PhD problem selection, not another
method/theorem extension here. No new project or experiment has been started.

Latest (2026-09-19): [theorem-reduction novelty audit: KILL](docs/VIABILITY_BOUNDARY_NOVELTY_AUDIT_20260919.md).
**Both the regenerative-learning and current standalone theory-paper stories are CLOSED.**
Prop. 1 reduces to Wald/renewal monotonicity; Prop. 3 to differential advantage
verification; Prop. 4 to cycle performance difference plus prefix-mass domination.
No irreducible new technical step was found under the user's novelty kill test.
The derivation and all evidence remain unchanged as supporting analysis.
No new theorems, UAV experiments, training, or automatic statistical-certificate pivot.
Earlier theory-novelty-pending statements below are historical and now resolved.

Latest (2026-09-19): [confirmed KILL and theory-only Pivot A](docs/VIABILITY_BOUNDARY_THEORY_20260919.md).
**The high-level regenerative-learning main story is CLOSED.** Completed safe-tree
VB/Oracle = 0.9913422775 >= .98; one distinct early-return physical state.
No neural training, new UAV runs, random census, altered task distribution,
uncertainty or multi-agent rescue. All P0–P2 assets/evidence are retained.
[Theory derivation](DERIVATION_PACKAGE.md) gives conditional VB optimality and
near-optimality proofs; existing-tree VB-only certificate >=98.6185%.
Mathematical coherence is established under stated assumptions; novelty remains
unverified. Older startup/pending statements below are historical and resolved.

Latest (2026-09-19): [P2-A2 safe-set stopping kill-test startup](docs/REGENERATIVE_CONTROL_P2A2_20260919.md).
P2-A1 passed diagnostic review. Exact task-prefix enumeration and renewal-ratio DP
are implemented; VB is the sole baseline, with the fixed >=.98 kill criterion.
Seven focused tests and actual-SAC startup/resume checks pass. At handoff, 13
nodes were saved and 12 remained pending; **no ratio/kill conclusion yet**.
The resumable authorized run continues; no monitoring to completion or neural
training promotion. Historical P2 holds below are superseded only for P2-A2.

Latest (2026-09-19): [P2-A1 reachable-state branching](docs/REGENERATIVE_CONTROL_P2A1_20260919.md).
P1/P1.1 passed user review. Bounded census: 885 natural D observations, 65 distinct
physical keys (target 200 not reached), 260 exact task-type/R branches. No RNG
peeking or new training. No systematic comparable-state local sign reversal found;
only 4–5 comparable pairs, 48.44% exact D-successor closure, sparse abstraction
support and residual navigation timeouts prevent a scientific conclusion.
**Stop at P2-A1; no full oracle/learning promotion.** Earlier P2-hold statements
below are historical; this limited census alone was authorized.

Latest (2026-09-19): [P1.1 post-delivery C/R](docs/REGENERATIVE_CONTROL_P11_20260919.md).
P1 passed user review. P1.1 removes task screening: task-free D decisions;
C draws then executes a job, R draws/discards nothing, and charged H forces its
first IID job without C/R choice. Frozen navigation/physics/service are unchanged.
Only three 1,000-second semantic sanity traces are authorized; P2 remains deferred.
Earlier P1 pre-task rejection semantics below are historical, not the current task.

Latest (2026-09-19): [continuing regenerative control P1](docs/REGENERATIVE_CONTROL_P1_20260919.md).
**P0-A passes for restricted research startup; P0-B legacy energy-critic
qualification is cancelled, not failed.** P1 uses frozen SAC macro rollouts:
single UAV, pre-task C/R decisions, paid station service, IID task stream, and
explicit regeneration. No critic recovery/training, P2 oracle or high-level
learning is authorized in this stage. Prior research-line labels below are history.

Latest (2026-09-19): [control adaptation startup](docs/CONTROL_ADAPTATION_STARTUP_20260919.md).
The active question is few-interaction pretrained control adaptation and nominal
capability retention. Public 2D quadrotor interface and resumable SAC startup are
implemented; the first checkpoint is paused. An upstream reset-distribution issue
must be resolved before long training. No adaptation comparison is complete.
Budget-reading is paused; AFPP/V3 and full-vector escalation remain stopped.

Previous (2026-09-19): [public-task budget-reading prototype](docs/BUDGET_READING_PILOT_20260919.md)
completed 15 actual training runs and 84 paired evaluation cells on DAD source
location finding. In this short DEV run, budget-conditioned reading does **not**
outperform output-side budget concatenation or fixed-query attention on average.
This is a new empirical hypothesis, not revived AFPP or a theorem escalation.
See the isolated [implementation and results](research/budgeted_design/README.md).
No published DAD-score reproduction, Step-DAD comparison or final confirmation
is claimed; an initial evaluation bug is retained and explicitly corrected.

Previous (2026-09-19): [T5.1e residual decomposition](docs/FPL_T51E_RESIDUAL_20260919.md)
selects **Route C: stop the current learned-planner main line**. Large VOI is
exact on all 32 pilot variants; Small VOI losses are dominated by work-limited
fallback/nonmeasuring execution. Continued-sensing states contribute zero to
the actual occupancy gap. No AFPP/selective learner, V2.1 or V3 is launched.
The theory and planning/evaluation infrastructure remain separate assets.

Frozen supporting line (2026-09-19): [Feedback Protocol Lab](research/feedback_protocol/README.md)
implements the bounded planning/evaluation stages of the [A/B/C plan](expert_advice/EA_MAPPO_ABC_RESEARCH_PLAN.md).
Its public problem contract and classical comparison baselines are separate
from the frozen [resource-separation paper](docs/ICML_RESOURCE_SEPARATION_PAPER_CORE_20260916.md).
See [fourth-batch status](docs/FPL_FOURTH_BATCH_20260919.md) for exact-policy
results and limits, and [T5.1 startup](docs/FPL_T5_1_STARTUP_20260919.md) for the
TRAIN-only exact teacher pipeline and resume instructions. Teacher V1 is now
[complete and fully audited](docs/FPL_TEACHER_V1_COMPLETE_20260919.md): 273 exact
labels, eight structures, limited non-prior sensing coverage. No neural FPL
method has been trained. [T5.1b export and V2 design](docs/FPL_T51B_EXPORT_V2_DESIGN_20260919.md)
are available: exact prefix targets are derived without new solves.
[T5.1c sizing and decoder-mask checks](docs/FPL_T51C_V2_SIZING_20260919.md)
are complete: 591 exact labels and zero mask mismatches. The subsequent
[depth-two census](docs/FPL_T51D_CENSUS_20260919.md) finds 6 adaptive states over
3 structures, including 5 missed by the collector. This demonstrates a collection
gap, but train still has only one adaptive structure. No full V2 expansion,
V3 generation or AFPP training has started.

## Historical PAI / PSPS entry (retained)

The following records the earlier research line and its then-current runtime
status; it is not an instruction to resume those experiments.

This repository is currently scoped to one research question: whether the
paired C/R advantage has stable decision heterogeneity (Gate H), and only then
whether full legal history adds predictive value beyond the latest legal frame
(Gate I).

## Completed experiment

The active formal lineage is
`artifacts/paired_advantage_identification_20260914_v3`. It uses 96 fresh DEV
worlds, 64 paired-CRN resamples per action and anchor, four world processes, and
eight branch workers per world. The initial service stopped after a child Python
SIGSEGV at 84/96; the unchanged resume completed all 96 worlds.

The complete raw audit passed. Gate H passed, establishing stable paired C/R
advantage heterogeneity on DEV. Gate I failed: the tested full legal history did
not improve out-of-world prediction or selected utility over latest/constant
baselines. The frozen stopping rule keeps CONFIRM unopened and METHOD-TRAIN
unauthorized. See `docs/PAIRED_ADVANTAGE_DEV_RESULT_20260915.md`.

Protocol and machine contract:

- `docs/PAIRED_ADVANTAGE_IDENTIFICATION_PROTOCOL_20260914.md`
- `docs/PAIRED_ADVANTAGE_IDENTIFICATION_CONTRACT_TEMPLATE.json`
- `docs/LOCKED_COLLISION_RECOVERY_PROTOCOL.md`

## Current research direction

Larger history-model training is stopped. An existing-DEV-only factor audit
finds that C/R sign changes are dominated by controller-conditioned task
reachability/stuckness rather than energy exhaustion or contact. Compact legal
progress/stall history is suggestive but not yet proven decision-sufficient;
simple privileged straight-path geometry adds no held-out value. See
`docs/LATENT_FACTOR_OBSERVABILITY_DEV_RESULT_20260915.md`. No new training or
CONFIRM access has been started.

The user subsequently authorized the independent PSPS-v1 prospective test.
Its 20-D compact legal progress/stall state, Gate O then Gate D rules, 96 fresh
DEV worlds and 128 unopened CONFIRM worlds are externally frozen under
`artifacts/psps_v1_20260915`. The formal resumable DEV service passed its first
checkpoint health check and is running with automatic analysis, confirmation,
and training disabled. See `docs/PSPS_V1_PROSPECTIVE_PROTOCOL_20260915.md` and
`artifacts/psps_v1_20260915/startup_health_psps_v1.md`.

Startup audit:
`artifacts/paired_advantage_identification_20260914_v3/startup_health_pai_v3.md`.

## Repository layout

- `scripts/`: only the active collector/validator/analyzer/freezer closure and
  generic confirmation helpers.
- `tests/`: focused current runtime/model/coordinator tests.
- `artifacts/`: active v3 evidence, minimal provenance contracts, and the exact
  frozen model files required at runtime.
- `envs/`, `experiments/`, `agent/`, `common/`, `network/`, `policy/`,
  `calibration/`, `cert_runtime/`: environment and method implementations,
  intentionally untouched by repository cleanup.
- `runtime_support/review_bundle`: byte-identical runtime safety/environment
  support moved out of the old review bundle. The root `review_bundle` symlink
  preserves unchanged imports for the active frozen run.

## Health commands

```bash
.venv/bin/python scripts/validate_paired_advantage_contract.py \
  artifacts/paired_advantage_identification_20260914_v3/contract.json \
  --receipt artifacts/paired_advantage_identification_20260914_v3/freeze_receipt.json
```

Do not run the DEV analyzer until collection reaches 96/96 with a complete raw
audit. Repository minimization details are in
`docs/REPOSITORY_MINIMIZATION_20260914.md`.
