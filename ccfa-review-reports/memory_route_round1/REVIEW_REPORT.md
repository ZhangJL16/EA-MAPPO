# Memory Route Hostile Review

## Scope

Blind-style review of the exact-current observer/tube/HOCBF theory, 5,000
matched-trajectory benchmark, 200k counterexample slice, hidden-size ablation,
and 60,152-transition controlled closed-loop artifact. This is a same-family
provisional review; an external Claude overlay was unavailable.

## Reviewer A — theory and soundness

**Score:** 5/10. **Confidence:** 4/5.

The conditional ideal-arithmetic interval, tube, robust directional HOCBF, and
set-inclusion proofs are now internally sound under their declared assumptions.
However, the guarantee is not recurrent-memory-specific. The neural nominal can
be removed without changing the residual-bound theorem. The oracle exact-state
trace also contains `2.5204%` infeasible/uncertified fallback, so it is not a
fully certified end-to-end execution.

**Fatal concern:** none for the scoped conditional propositions.

**Major concern:** no theorem ties the learned recurrent state to a strictly
smaller valid physical error set.

**Score-change condition:** prove and independently validate a recurrent-
dependent bound that is strictly tighter than a non-recurrent observer under
the same assumptions.

## Reviewer B — novelty

**Score:** 3/10. **Confidence:** 5/5.

The closest direct work, Matias and Silvestre 2026, already couples guaranteed
finite-horizon obstacle-set estimation and sampled-interval set flow to CBF-QP
navigation for uncertain obstacle dynamics. Robust sampled-data high-order CBF
and interval-CBF papers separately cover the remaining ingredients. Orthotopes,
constant jerk, and directional support are specializations. Hidden-state
contraction is also established prior art.

**Fatal concern:** theorem-level difference from closest work does not survive.

**Exact counterexample:** delete the recurrent nominal while retaining the same
declared residual bound; every deterministic safety theorem remains true.

**Score-change condition:** identify a conclusion unavailable to guaranteed
set-flow/observer CBF frameworks and make recurrence necessary to its proof.

## Reviewer C — experiments and methodology

**Score:** 4/10. **Confidence:** 4/5.

The experiment discipline is strong: disjoint 3k/1k/1k data splits, classical
baselines, source hashes, invalid-artifact quarantine, bounded budgets, failure
regimes, and explicit uncertified labels. The negative result is nevertheless
decisive. Ego L16 MLP is the best predictor; recurrent tubes do not improve the
coverage-width frontier; learned boxes under-bound; and the deterministic
interval freezes every rollout. No equal-certification intervention or energy
benefit is demonstrated.

**Fatal concern:** none for reporting a negative benchmark result.

**Major concern:** zero simulated collisions for uncertified methods cannot be
used as safety evidence.

**Score-change condition:** show a useful deterministic or clearly scoped
probabilistic tube with matched-coverage intervention/energy gains across more
than the 12 controlled scenarios.

## Area Chair synthesis

**Overall score:** 3/10 — Reject. **Confidence:** 5/5.

The work has become careful and falsifiable, but it does not meet the requested
theory-contribution gate. Correct conditional proofs remain, yet the recurrent
part is proof-irrelevant and empirically inferior to a fixed-window baseline.
The closest 2026 prior covers the same guaranteed set-estimation-to-CBF control
object more generally. The appropriate result is a rigorous negative closure,
not a paper novelty claim.

`FATAL_UNRESOLVED_THEORY_DEFECTS = 0`

`CRITICAL_UNRESOLVED_THEORY_DEFECTS = 0`

`NOVELTY_SCORE = 9/30`

`ACCEPT = FALSE`
