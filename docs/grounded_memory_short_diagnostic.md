# Grounded Memory Short Diagnostic

## Protocol

- Artifact: `artifacts/grounded_memory_short_diagnostic_20260820/diagnostic.json`
- Artifact SHA-256: `010fa690bfffd39a50bf092ef27b018577b02ad266a37ec8401285217687070d`
- Training seed: `20260820`
- Dataset: 600 train / 200 validation / 200 test synthetic trajectories, with trajectory-disjoint generation seeds.
- History length: 16.
- Short budget: 10 epochs; this is diagnostic, not a paper result.
- Candidate: object GRU with separate Motion and Safety grounding heads.
- Baseline: same-size generic GRU with one joint output head.
- Safety labels available: current closing speed and one-second minimum-distance proxy computed from simulator state.
- Route labels: unavailable.
- Safety-filtered energy-overhead labels: unavailable.

## Representation diagnostic

| Method | Motion MAE | Safety-proxy MAE | Closing-speed MAE | Future-min-distance MAE |
|---|---:|---:|---:|---:|
| Typed object/safety heads | 1.01744 | 5.04398 | 2.97298 | 7.11499 |
| Generic joint GRU | 1.06889 | 5.03985 | 2.98154 | 7.09816 |

The typed heads slightly improve motion MAE but do not improve the safety proxy. The differences are too small and the run has one training seed, so no architecture claim is supported.

## Routing diagnostic

The fixed-routing graph was evaluated under 2,000 random/adversarial learned-message draws.

- all projected actions feasible: true;
- maximum hard-constraint violation: `0.0`;
- certified constraint bytes invariant to learned messages: true;
- learned messages changed nominal action by mean norm `0.61893`;
- mean filter intervention: `1.13346`;
- intervention with larger uncertainty: `1.06066`;
- intervention with smaller nested uncertainty: `0.84853`;
- nested-set intervention nonincrease: true.

This validates the **implementation contract**: learned route/energy messages can change proposals but cannot alter certified constraints. It does not establish theoretical novelty, because the behavior is exactly the modular safety-filter separation principle.

## Missing evidence

- no real route-grounding labels such as corridor choice, safe trajectory, or detour decomposition;
- no safety-filtered energy trajectories and therefore no energy-overhead labels;
- no proof that a learned object memory yields a smaller valid uncertainty set;
- no multi-seed optimization analysis;
- no matched long closed-loop comparison.

## Diagnostic decision

The short diagnostic confirms semantic correctness but gives no evidence that the grounded modular architecture outperforms a generic recurrent state. It therefore cannot justify a five-seed long experiment.
