# Blind Reviewer A — Theory (Round 4)

Materials read: canonical `docs/new_theory/` theory only; no prior reviews or rebuttals.

## Score

**8/10. Confidence: 5/5.**

## Reject-First Audit

- **FATAL:** none.
- **CRITICAL:** none.
- **MAJOR:** T4 calibration, T5 selected-hazard validity, and I3 probe ignorability remain strong assumptions that implementation must instantiate. C4--C6 require a valid kernel-drift estimate before operational use.
- **MINOR:** the many conditional layers may be too cumbersome for a main-paper theorem narrative.

## Strongest Counterexamples

1. A learned stopping policy commits precisely on easy returns; completed-return calibration is optimistic. I1--I3 correctly require randomized probes or justified selection correction.
2. A tiny hard-mask change opens a new state region unseen under the reference version; one-direction transport is insufficient. The canonical text correctly restricts the claim.

## Verdict

The definitions and conditional proofs are sound. No deep-TD, universal reachability, arbitrary-shift, or physical-safety theorem is fabricated.

## Score-Change Condition

Lower if routine logs are treated as randomized probes or if failed outcomes disappear from the defective law.
