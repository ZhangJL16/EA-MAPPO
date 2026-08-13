# Reviewer B — Novelty (Round 3)

## Score

**5/10, Reject/Borderline. Confidence: 5/5.**

## Reject-First Findings

- **FATAL NOVELTY:** unresolved. The main theorem is a specialization of maximal-coupling policy perturbation plus standard coverage degradation under distribution shift.
- **CRITICAL:** the only specific term—collision-admission boundary visitation—is an elementary way to upper-bound where two hard masks differ.
- **MAJOR:** filtered-action critic consistency is already explicit in recent safety-filtered actor-critic work: https://arxiv.org/abs/2605.26452.

## Strongest Replacement

Use a standard BMDP/SDAC learner, log the executed filtered policy, bound target-policy shift with occupancy/kernel TV, and apply robust or off-policy conformal prediction. The proposed transport gate is then a domain-specific monitoring rule rather than a new learning framework.

## Closest Sources

- BMDP: https://arxiv.org/abs/1903.01004
- policy-coupled prediction: https://arxiv.org/abs/2607.02206
- conformal coverage under joint shift: https://arxiv.org/abs/2501.13430
- filtered critic target: https://arxiv.org/abs/2605.26452

## Eight-Question Verdict

The route now answers why an independent energy critic is wrong, but not why standard budgeted/distributional RL plus generic shift control cannot implement the corrected object. No ICLR-level new operator or learning theorem remains.

## Score-Change Condition

Raise to 6 only if the project establishes an identification/sample-complexity result tied to repeated-sortie data generation that generic perturbation analysis does not provide.
