# Blind Reviewer B — Novelty (Round 4)

Materials read: canonical theory and closest-work matrix only; no prior reviews or rebuttals.

## Score

**3/10, Reject. Confidence: 5/5.**

## Fatal Novelty Concern

The work combines established frameworks without a new central theorem:

- budget-conditioned policies/values: https://arxiv.org/abs/1903.01004
- multiple distributional constraints: https://arxiv.org/abs/2301.10923
- quantile-constrained cumulative cost: https://proceedings.neurips.cc/paper_files/paper/2022/hash/2a07348a6a7b2c208ab5cb1ee0e78ab5-Abstract-Conference.html
- policy-coupled conformal decisions: https://arxiv.org/abs/2607.02206
- off-policy return CDF/risk estimation: https://proceedings.mlr.press/v151/huang22b.html
- selective-label exploration/utility tradeoff: https://proceedings.mlr.press/v139/wei21a.html

## Strongest Replacement

Represent collision as a budgeted cost, energy as a distributional return, filter execution as the target policy, and charger commitment as a logged intervention. Then apply BMDP/SDAC/QCRL, distributional OPE, and selective-label exploration. The defective absorbing outcome and one-way latch are bookkeeping choices.

## Why the Candidate Results Do Not Rescue Novelty

- C1 is a two-action illustration of why budgeted policies depend on budgets.
- C4 is maximal coupling; C6 is the definition-level robustness property of TV.
- C5 is a domain-specific support-change corollary.
- I1--I3 are standard positivity/ignorability identification facts.
- I4 is weaker than existing off-policy CDF bounds.
- I5 is a standard exploration-versus-utility tradeoff.

## Score-Change Condition

No textual revision suffices. A new theorem must solve a harder problem not already implied by these frameworks, or empirical evidence must support a systems/application paper with theory demoted. That evidence does not exist yet.
