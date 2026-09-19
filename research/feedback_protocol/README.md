# Feedback Protocol Lab — finite Bernoulli testbed

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
independent `--policy-seed`. Scientific expected-risk evaluation remains future work.

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

The next planned stages are a bounded approximate planner, systematic instance generator/split registry, pending-protocol checkpoint runner and teacher dataset. Neural protocol decoding, DAD-style adapters, scientific sweeps and formal comparisons remain unimplemented.

First-batch provenance and local evidence inventory: [implementation report](../../docs/FPL_FIRST_BATCH_20260919.md).
