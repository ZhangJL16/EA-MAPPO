# Stage A TD Pre-Fix Diagnosis

Date: 2026-08-14

This document records the read-only diagnosis made before changing B2/B3 training. It uses the immutable validation artifact `artifacts/energy_transfer/stage_a_validation_seed0_20260814_v2/`.

## Bellman Chain

The implemented object is consistently action-conditioned:

```text
features_t      = concat(state_t, action_t)
next_features_t = concat(next_state_t, actual_action_{t+1})
Q_target        = cost_t + 1{not charger_hit_t} Q_target(next_features_t)
```

For a terminal transition, `next_action` is a zero placeholder but the bootstrap mask is zero, so the placeholder cannot affect the target. For a non-terminal transition, `next_action` is the next action actually selected by the frozen charger-goal SAC trajectory. Training and inference both query `Q(s,a)`; no post-action `V(s')` interface is used.

## Ten-Point Audit

1. **Terminal target:** scalar and quantile targets equal the measured last-step cost exactly.
2. **Post-terminal bootstrap:** none; `charger_hit=True` multiplies the next prediction by zero.
3. **Next action:** taken from the next recorded frozen-SAC return transition, so it matches the behavior policy.
4. **Value semantics:** consistently `Q(s_t,a_t)` in dataset construction, training, saved checkpoints, and inference.
5. **Inference alignment:** the 77-state/3-action input contract matches training.
6. **Charger hit:** computed from `distance_to_goal_after <= goal_radius`; `EnergyReturnEpisode` enforces that it appears only on the final transition and equals `completed`.
7. **Terminal dilution:** present. Train has 90 terminal versus 3,322 non-terminal transitions: terminal fraction `2.6377%`.
8. **Target update:** hard copy every 20 full-batch updates allows a positive initialization error to be repeatedly propagated under gamma=1 before sparse terminal anchors correct it.
9. **Quantile pairing:** pairwise QR loss assigns equal mass to target atoms at nonuniform levels `(0.50, 0.90, 0.95, 0.99)`. That does not represent equal probability mass and makes the lower predicted quantiles chase an upper-tail-heavy empirical atom distribution.
10. **Monotone parameterization:** default softplus initialization produces mean initial outputs near `(0.69, 1.38, 2.06, 2.75)` before observing labels, while true held-out returns are below one. Positive cumulative increments amplify the bias.

## Observed Drift

The original held-out B2 error grows with horizon:

| Horizon | MAE | Mean prediction | Mean truth |
|---|---:|---:|---:|
| 1 | 0.07056 | 0.08274 | 0.01218 |
| 2 | 0.06233 | 0.08670 | 0.02437 |
| 3 | 0.05497 | 0.09156 | 0.03659 |
| 6-10 | 0.04752 | 0.13407 | 0.09806 |
| 11-20 | 0.09949 | 0.26407 | 0.19066 |
| 21-40 | 0.16935 | 0.47328 | 0.36330 |
| >40 | 0.17985 | 0.58177 | 0.56912 |

B3 starts wrong even at the terminal and remains grossly high at every horizon:

| Horizon | Median MAE | Mean q0.50 | Mean truth |
|---|---:|---:|---:|
| 1 | 4.31169 | 4.32386 | 0.01218 |
| 2 | 4.33956 | 4.36393 | 0.02437 |
| 3 | 4.36783 | 4.40442 | 0.03659 |
| 11-20 | 5.03004 | 5.22071 | 0.19066 |
| 21-40 | 5.13713 | 5.50043 | 0.36330 |
| >40 | 4.97200 | 5.54112 | 0.56912 |

A controlled rerun of the original B3 objective showed train Monte-Carlo MAE increasing from `0.511` at update 0 to `3.538`, `3.880`, `4.159`, and `4.410` at updates 200, 400, 600, and 800 while its Bellman loss remained finite. This is TD semantic drift: declining/finite Bellman loss is not evidence of correct return-energy scale.

## Deterministic Countercheck Before Repair

With the original implementation and a ten-step deterministic trajectory with cost `0.1` per step:

- B2 recovers the exact scale after 800 updates (MAE about `0.00013`), showing that the scalar target formula itself is sound;
- B3 remains high after 800 updates (median MAE about `0.198`) and predicts q0.50 `1.434` instead of `1.0` at horizon 10.

At 200 updates, B3 predicts q0.50 `10.447` instead of `1.0` at horizon 10. This isolates the B3 failure from UAV data complexity.

## Pre-Fix Root-Cause Decision

`LABEL_OR_TERMINAL_MASK_BUG = FALSE`

`Q_V_SEMANTIC_MIXING = FALSE`

`TERMINAL_ANCHOR_DILUTION = TRUE`

`B2_RUNAWAY_GAMMA_1_DRIFT = FALSE`

`B2_BOOTSTRAP_ANCHOR_PROPAGATION_PROBLEM = TRUE`

`B3_RUNAWAY_GAMMA_1_DRIFT = TRUE`

`B3_SOFTPLUS_INITIAL_SCALE_BIAS = TRUE`

`B3_NONUNIFORM_ATOM_WEIGHTING_BUG = TRUE`

B2 is primarily an optimization/anchoring failure under full-batch gamma=1 bootstrapping and function approximation; its controlled MC error decreases rather than running upward. B3 combines the same weak anchoring with a much larger initialization bias and an incorrect equal-mass interpretation of nonuniform target quantiles, producing a genuine upward semantic drift.
