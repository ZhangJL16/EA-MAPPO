# Final Rejection: Structured Memory as Paper Core

## Decision

**REJECT STRUCTURED MEMORY AS PAPER CORE.**

`CLAIM_LEVEL = ENGINEERING_ONLY`

`LONG_EXPERIMENT = NOT_LAUNCHED`

## Why the Last Exploration Fails the Gate

### Bridge 1: grounding to certified uncertainty

The bridge exists only through trusted set-membership constraints:

\[
\mathcal C_{\mathrm{ground}}
=\mathcal C_{\mathrm{base}}\cap\bigcap_i\mathcal C_i.
\]

The 500-snapshot held-out audit gives 100% containment and major contraction. However:

- C2 all-history has the tightest set;
- C1 fixed history is already much tighter than learned proposals;
- C3 generic GRU, C4 FOGM, and C5 oracle are numerically identical;
- FOGM saves only about 1.15 ms versus all-history while producing a much larger set;
- the verifier still needs independently trusted association and sensor bounds.

Thus the learned memory does not create the certificate. It only proposes constraints for a standard set-membership estimator.

### Bridge 2: true Route/Energy grounding to control value

The new 100,122-transition dataset provides real executed-trajectory labels. The outcome is negative:

- FOGM route accuracy is worse than generic GRU in 3/3 seeds;
- FOGM Safe Energy-to-Go MAE is worse than generic GRU in 3/3 seeds;
- the distance/velocity energy baseline is best;
- FOGM closed-loop path and energy are worse than no route memory;
- explicit side hysteresis has lower intervention, fewer switches, and lower energy than FOGM.

FOGM does improve short-horizon safety-energy overhead MAE in 3/3 seeds, but this narrow auxiliary effect has no demonstrated downstream consequence.

## Prior-Art Equivalence That Cannot Be Rebutted

The strongest defensible method decomposes into established components:

1. set-membership uncertainty contraction and its convergence analysis;
2. constraint pruning for tractable set over-approximation;
3. learned active-constraint selection;
4. independently checked proposal/certificate architectures;
5. measurement-robust CBFs and modular safety filters;
6. approximate information states for compressed control history.

Closest primary work includes:

- [Li et al., ICML 2024](https://proceedings.mlr.press/v235/li24ci.html), which gives non-asymptotic set-membership uncertainty-set convergence;
- [Tang et al., L4DC 2024](https://proceedings.mlr.press/v242/tang24a.html), which explicitly studies constraint pruning and tight tractable over-approximations of set-membership sets;
- [Dean et al., CoRL 2020/2021](https://proceedings.mlr.press/v155/dean21a.html), which connects valid perception-error bounds to robust CBF safety;
- [Misra et al., 2018](https://arxiv.org/abs/1802.09639), which learns relevant active constraint sets for repeated constrained optimization;
- [Jackson et al., 2021](https://arxiv.org/abs/2104.06178), which assigns proposal search to complex software and certificate checking to a small trusted monitor.

The exact object/route/energy node names do not create a theorem-level separation from this composition.

## Final Novelty Score

| Dimension | Score | Maximum | Reason |
|---|---:|---:|---|
| New mathematical object | 4 | 10 | verified history intersection is a standard feasible set |
| Separation from closest work | 3 | 10 | proposal-verifier and constraint pruning are covered |
| Nontrivial provable consequence | 4 | 10 | containment/non-interference/monotonicity are supporting lemmas |
| **Total** | **11** | **30** | below both theory and algorithmic gates |

## What Remains Useful

- The typed certified/learned routing graph is a good engineering boundary.
- The real 100k trajectory dataset and label derivation are reusable for negative-study or appendix analysis.
- The verified set-membership updater can be reused as a non-neural baseline.
- Explicit route hysteresis is a useful systems component.
- The short-horizon safety-energy overhead target may be an auxiliary diagnostic, but not a paper core without a causal downstream effect.

## What Must Stop

- Do not launch the five-seed 200k-optimizer-step structured-memory experiment.
- Do not continue GRU/LSTM/SSM architecture search for this route.
- Do not claim theoretical novelty, certified learned contraction, or a Route/Energy control gain.

## Recommended Research Direction

Use the verified set-membership and explicit hysteresis components as fixed baselines in a robotics/autonomous-systems study. Let future work be driven by a reproducible physical failure phenomenon, not by another memory architecture rebranding.

