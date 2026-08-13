# Reviewer A — Theory

## Summary

The observation model is coherent and the two-environment constructions correctly show non-identification of never-executed actions and never-selected continuation decisions.

## Scores

- Theory/soundness: **7/10**
- Confidence: **4/5**

## Positive Signal

The manuscript distinguishes proposal from executed action and correctly treats commitment as a stopping time. The minimax $1/2$ estimation lower bound follows from identical observed laws.

## Major Concerns

1. **The theorem is generic support failure.** The proof changes an outcome at a history-action pair with zero occupancy. No specifically two-level step is used.
2. **The missingness language can mislead.** If the filter is fully observed and deterministic, the central issue is positivity, not an unidentified missingness propensity.
3. **The stopping result is an action result in disguise.** At the commitment history, “continue task” has zero action probability.

## Exact Counterexample to the Claimed Separation

Use a one-state, two-action contextual bandit. Action $a$ is always selected; $b$ is never selected. Two reward laws agree on $a$ and differ on $b$. This reproduces the claimed observational equivalence without trajectories, energy, or safety.

## Fatal Concerns

- None for mathematical correctness.

## Score-Change Condition

Theory could reach 8/10 if the authors prove a dynamic lower bound in which the safety requirement itself constrains the probability of acquiring the distinguishing observation, rather than only assuming fixed zero support.
