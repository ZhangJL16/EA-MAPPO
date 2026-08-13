# Reviewer C — Methodology

## Summary

The proposed observational audit is useful for robotics logging, but there is no valid empirical protocol yet for verifying counterfactual labels that the safety mechanism deliberately avoids.

## Scores

- Methodology: **5/10**
- Confidence: **4/5**

## Positive Signal

The logs distinguish proposals, executed actions, intervention decisions, and realized outcomes. This is necessary for any future empirical audit.

## Major Concerns

1. **Oracle-label problem.** Evaluating the safety boundary for rejected actions may require dangerous physical execution or simulator ground truth.
2. **Causal estimand ambiguity.** Collision outcome after one action depends on the subsequent policy and horizon; this must be fixed before evaluation.
3. **Application wrapping risk.** A UAV battery merely adds a familiar resource constraint unless an experiment isolates a new mechanism.
4. **No falsification protocol.** The packet does not specify how to distinguish safe information acquisition from ordinary safe active learning.

## Exact Counterexample

A simulator can expose all forbidden labels and make every estimator look identifiable, while the deployed physical system still has zero support. Such a benchmark would test supervised interpolation, not the claimed deployment problem.

## Fatal Concerns

- None at this theory-exploration stage.

## Score-Change Condition

Methodology could reach 6/10 with a matched-budget protocol that separates deployable labels from retrospective oracle labels and includes SafeMDP/ActSafe-style baselines.
