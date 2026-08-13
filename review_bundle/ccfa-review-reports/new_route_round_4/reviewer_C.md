# Blind Reviewer C — Methodology (Round 4)

Materials read: canonical theory and `EXPERIMENT_REQUIREMENTS.md`; no prior reviews or rebuttals.

## Score

**6/10. Confidence: 4/5.**

## Reject-First Audit

- **FATAL:** none in falsifiability.
- **CRITICAL:** the implemented/running E1 experiment does not instantiate budget conditioning, missing-mass learning, transport, or randomized commitment probes.
- **MAJOR:** a full empirical program now requires strong BMDP/SDAC/QCRL/OPE/selective-label baselines and equal probe/recollection budgets; this is substantially larger than the current implementation.

## Strongest Counterexample

Random probes improve prediction only because they produce more charger trajectories than the baseline. Without equalizing completed/censored charger labels, the claimed self-supervision mechanism is unidentifiable experimentally.

## Verdict

The future program can falsify each claim, but no existing result supports the primary mechanism. Methodology is adequate as a plan, not as paper evidence.

## Score-Change Condition

Raise only after implementing the actual final object and preregistering equal-label/equal-environment comparisons. Never use the running E1 as coupled-method evidence.
