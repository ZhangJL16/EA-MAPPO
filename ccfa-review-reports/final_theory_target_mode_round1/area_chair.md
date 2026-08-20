# Area Chair Synthesis

## Scores

| Dimension | Score / 10 | Confidence |
|---|---:|---:|
| Theory, conditional lemmas | 6.0 | 0.95 |
| End-to-end theorem | 2.0 | 0.95 |
| Novelty | 2.5 | 0.90 |
| Methodology / evidence | 2.5 | 0.95 |
| Overall paper-core readiness | 2.5 | 0.95 |

## Decision

**REJECT AS PAPER-CORE THEORY.**

The investigation is scientifically valuable because it found and repaired a
metric-induced false positive. The original 100% endpoint-containment result
did not establish current-state containment. Under abrupt motion, the proposed
fast set remains feasible while true-state coverage collapses to 5%, 29%, and
24%. The corrected artifact records those cases as uncertified.

## Fatal Candidate-Level Concern

The proposed online gate cannot infer that the true motion history obeyed the
narrow jerk model. LP feasibility is existential and can be witnessed by a
different latent trajectory. Therefore adaptive history cannot safely tighten
the present-state set using this gate. Robust future jerk inflation does not
repair an invalid initial set.

## Surviving Results

- fixed-model history-set and interval-hull contraction;
- conditional bounded-jerk future propagation;
- tube-to-clearance transfer with ego error included;
- independent full-hold safety implications;
- pointwise future-jerk impossibility lower bound;
- exact finite-candidate and candidate-set energy statements from prior work.

All are conditional, standard, or negative. None meets the 22/30 novelty gate.

## Why More Closed-Loop Runs Were Not Authorized

The adaptive candidate fails a necessary theorem premise before integration.
Running 1,000--5,000 closed-loop rollouts would measure an uncertified heuristic
and could not repair theorem novelty. The existing Candidate A and B3--B5
artifacts remain valid engineering baselines, but no result may be transferred
to the rejected adaptive selector.

## Score-Change Condition

A future theory attempt would need a new computable multi-obstacle
recoverable/invariant object that certifies action existence under actual
sampled dynamics and perception uncertainty, or a nontrivial causal minimax
tube theorem with an implementable optimal construction. Merely adding a
change detector, terminal set assumption, or more UAV trials is insufficient.

## Final Claim Level

`NO_DEFENSIBLE_THEORY_CONTRIBUTION`

