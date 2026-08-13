# Reviewer C — ML Methodology (Round 2)

## Decision

**Score: 5/10 (borderline design, no evidence). Confidence: 4/5.**

The proposed empirical program can falsify the narrowed mechanism, but the running E1 study only validates the collision-free special case and cannot test the coupled claim.

## Reject-First Audit

### FATAL

None in the proposed methodology if the matched BMDP/SDAC/QCRL baselines and failure-retaining data protocol are implemented.

### CRITICAL

No implemented learner currently estimates success mass or conditions energy returns on `(b,eta)`. Therefore the current code does not instantiate the canonical Loop-2 method.

### MAJOR

1. **Censoring:** timeouts need survival/censoring treatment, not binary failure or truncated return labels.
2. **Version overlap:** recollection after filter updates may be expensive; naive version conditioning without overlap is extrapolation.
3. **Capacity confound:** adding `b,eta` inputs increases model capacity; comparisons require matched parameter/data budgets and budget-stratified targets.
4. **Calibration unit:** split by sortie, not correlated transition.
5. **Baseline strength:** independent critics alone are insufficient; include BMDP, SDAC, QCRL/CVaR, off-policy conformal, and geometric RTH.

## Strongest Counterexample

Suppose collisions and empty supports occur in only the difficult 5% of routes. Training quantiles on successful returns gives excellent 95% energy coverage, but the system fails on exactly those difficult routes. Without a separately evaluated success-mass head, the method looks calibrated while operational failure is unchanged.

## Score-Change Conditions

- Raise to 6: specify a concrete target-drift-aware algorithm, immutable data schema, censoring estimator, and matched causal tests before results.
- Raise above 6 only after results; theory documents alone cannot establish trainability or usefulness.
- Lower to 2: use E1 as evidence for collision coupling or discard failed trajectories.
