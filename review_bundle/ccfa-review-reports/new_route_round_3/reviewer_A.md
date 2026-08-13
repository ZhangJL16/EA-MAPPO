# Reviewer A — Theory (Round 3)

## Score

**8/10. Confidence: 4/5.**

## Reject-First Findings

- **FATAL: none.**
- **CRITICAL: none.**
- **MAJOR:** C5 is exact only for support changes isolated to the learned bound; continuous proposal/allocator changes require C4's kernel-TV form. Estimating a high-confidence `rho_bar` remains outside the theorem.
- **MINOR:** report directional and bidirectional transport separately.

## Strongest Counterexample

The new filter admits a region never visited under the old version. An old-direction occupancy audit reports zero boundary mass, yet the new policy spends substantial time there. The documents correctly avoid a symmetric claim and recommend bidirectional audit; a paper must not silently use one direction as two-way validity.

## Soundness Judgment

C4 follows by maximal coupling, C5 by support agreement before first boundary visit, and C6 by the event characterization of TV. These are valid conditional results. The package does not claim that parameter distance controls kernel drift or that `rho_bar` is easy to estimate.

## Score-Change Condition

Lower to 5 if C5 is generalized to arbitrary continuous filters without a kernel-drift premise. Raise only with a finite-sample estimator for `rho_bar` under sequential dependence.
