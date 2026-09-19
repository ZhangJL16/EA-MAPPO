# Feedback Protocol Lab — finite Bernoulli testbed

Current: [T5.1 teacher contract](CONTRACT_V05_TEACHER.md) and
[implementation/startup report](../../docs/FPL_T5_1_STARTUP_20260919.md).
TRAIN-only exact V/Q dataset builder and mixed-history sampling are implemented;
one of 24 root checkpoints is generated and audited. The report contains the
resume command. No neural architecture or training yet; full coverage is pending.

## Completed fourth batch

Fourth batch: [contract](CONTRACT_V04.md),
[final pre-B2 report](../../docs/FPL_FOURTH_BATCH_20260919.md).
Exact fixed-policy expectations replace MC where resolvable; deterministic
episode work accounting, hierarchical DEV and belief-MCTS are implemented.
The frozen operational criterion returns a **scoped GO for later amortized
planning research**, not an implemented/trained neural method.

```bash
PYTHONPATH=research/feedback_protocol/src python3 -m fpl.necessity \
  --plan research/feedback_protocol/configs/difficulty_dev_v2.json \
  --output /tmp/fpl_necessity_new --max-new 1
PYTHONPATH=research/feedback_protocol/src python3 -m fpl.necessity \
  --plan research/feedback_protocol/configs/difficulty_dev_v2.json \
  --output /tmp/fpl_necessity_new --resume
```

Optionally pass both `--legacy-dataset` and `--legacy-study` to separately audit
the old DEV policies; pass the same arguments on resume. Exact evaluation is
conditional on owned RNG/policy state, not all algorithm seeds. Watchdog expiry
invalidates evaluation instead of choosing an action. Current old published
test/OOD seeds are public fixtures, not blinded confirmation.

## Historical third-batch scope

Third batch: [v0.3 contract](CONTRACT_V03.md), scientific clone-disjoint datasets,
independent rational minimax bases, global compute budgets, OneStepVOI and a
sealed/resumable DEV-only expected-risk evaluator. No neural training.

Completed study: [third-batch report](../../docs/FPL_THIRD_BATCH_20260919.md),
[results](provenance/dev_pilot_v1/RESULTS.md),
[portable raw evidence](provenance/dev_pilot_v1/dev_evidence.zip).
The pilot does not establish sufficient motivation to start B2.

```bash
python3 -m unittest discover -s research/feedback_protocol/tests -v
PYTHONPATH=research/feedback_protocol/src python3 -m fpl.scientific_dataset \
  --plan research/feedback_protocol/configs/families/scientific_v1.json \
  --output /tmp/fpl_scientific_new
PYTHONPATH=research/feedback_protocol/src python3 -m fpl.evaluation \
  --dataset /tmp/fpl_scientific_new \
  --plan research/feedback_protocol/configs/dev_pilot_v1.json \
  --output /tmp/fpl_dev_new --max-new-episodes 1
# Explicitly resume the same frozen study:
PYTHONPATH=research/feedback_protocol/src python3 -m fpl.evaluation \
  --dataset /tmp/fpl_scientific_new \
  --plan research/feedback_protocol/configs/dev_pilot_v1.json \
  --output /tmp/fpl_dev_new --resume
PYTHONPATH=research/feedback_protocol/src python3 -m fpl.analyze_dev \
  --dataset /tmp/fpl_scientific_new --study /tmp/fpl_dev_new \
  --output /tmp/fpl_dev_new/analysis.json
```

Existing datasets/studies are not overwritten. Analyze only a completed study.
Reference unresolved is not a numeric bound. Exact offline construction costs
are separate from the online planning comparison. SciPy is optional, test-only.

## Historical second-batch scope

Second batch: [v0.2 contract](CONTRACT_V02.md), [implementation and profiling results](../../docs/FPL_SECOND_BATCH_20260919.md).
Includes utility/observation separation, template-grouped task generation, a split
registry, tiny exact minimax reference, prefix-beam Bayes planning, independent
posterior sampling, and resumable bounded scaling profiling. No neural method is
implemented or validated.

```bash
PYTHONPATH=research/feedback_protocol/src python3 -m fpl.dataset \
  --plan research/feedback_protocol/configs/families/splits.json \
  --output /tmp/fpl_tasks_new
PYTHONPATH=research/feedback_protocol/src python3 -m fpl.profile \
  --config research/feedback_protocol/configs/families/profile.json \
  --output /tmp/fpl_profile_new.jsonl
```

Existing output is rejected unless `--resume` is explicitly passed to the profiler.
The debug CLI below also accepts `beam_bayes` and `posterior_sampling`, with an
independent `--policy-seed`. Expected-risk evaluation was future work at that
batch; use the third-batch runner above for the new DEV study.

## First-batch commands and historical scope

Implemented: config-driven deterministic operation graph, public/private interfaces, debit-before-reload safety, batch-end feedback, exact finite-family posterior, evaluator-only finite-budget oracle, rational Bayes reference and independent channel-coverage heuristic. Read [CONTRACT.md](CONTRACT.md) for objective and terminal semantics.

No external Python runtime dependencies or project installation are required. Run from repository root:

```bash
python3 -m unittest discover -s research/feedback_protocol/tests -v
PYTHONPATH=research/feedback_protocol/src python3 -m fpl.cli \
  --config research/feedback_protocol/configs/fixture.json \
  --policy exact_bayes --budget 6 --truth 0 --noise-seed 0 \
  --output /tmp/fpl_debug_new.json
```

Output must not exist. Use `--policy channel_cover` for the independent heuristic. `--capacity 3` and/or `--bundling-off` configure the other cells. This command runs one debug episode and finishes; it is not a multi-instance evaluation or resumable training runner.

At the first batch the next stages were bounded planning, generation/registry,
checkpointing and teacher datasets. The first three now have bounded implementations.
Teacher data infrastructure is now implemented, with full generation pending.
Neural protocol decoding and DAD-style external adapters remain unimplemented.
This is not a completed A/B/C method study.

First-batch provenance and local evidence inventory: [implementation report](../../docs/FPL_FIRST_BATCH_20260919.md).
