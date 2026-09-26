# Research findings

## 2026-09-26: deterministic full-map learning claim

Intended claim: a learned policy outperforms a strong MPC on the static,
deterministic, full-map 2D collision-and-return-energy task.

**Result-to-claim verdict: no (same-family reviewer, provisional).** The completed
GPU H4/H8 run compares two nonlearning proposal planners, not learning with MPC.
On frozen test maps 40–55, H4 completed 138 targets (8.625/map) and H8 completed
85 (5.3125/map), with 12 H4 wins and four ties. Both had zero recorded contact
and depletion under the shared exact CPU safety layer. The previous PPO results
are validation-only and show no learning advantage; matched frozen-test PPO and
a demonstrably strong planner are missing. This verdict does not prove learning
can never outperform planning.

The H4/H8 contrast is specific to the implemented approximate candidate search.
The GPU proposal ranks only actions 0–8 in flight; `BatchedRolloutMPC.act()`
does not choose the explicit return action 9 in flight, and uses a fixed docked
charge heuristic. It therefore does not test joint return/charge optimization.
The terminal score is approximate; its relative ranking can change with the
lookahead length. Existing event logs show the policies diverge on all 16 test
maps, but do not isolate a causal reason for H8's lower throughput.

Immediate research constraint: preserve this negative result and its original
logs. Do not use it to claim that short horizons are generally better or that
learning is needed. New baseline development informed by maps 40–55 requires a
fresh untouched confirmation set for submission-facing comparisons. See
`NEXT_PHASE_PLAN_20260926.md`.

Evidence: `GPU_MPC_V6_FINAL_RESULT_20260926.md`, local 48-job summaries under
`artifacts/gpu_mpc_parallel_v6_8workers_20260926/`, and
`gpu2d/rollout_mpc.py`. The reviewer found no root `EXPERIMENT_AUDIT.json`, so
this is not a full experiment-integrity audit.
