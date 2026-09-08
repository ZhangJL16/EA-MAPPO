# Search Notes

## Safe Queries Used

- `minimum-cost reach-avoid stochastic dynamics reinforcement learning distributional`
- `chance-constrained reach-avoid reinforcement learning cumulative cost`
- `resource-to-go reinforcement learning charging station stochastic`
- `replenishable resources Markov decision process battery charging`
- `Control Barrier Functions for Energy Sufficiency`
- `simultaneous task and energy planning deep reinforcement learning`
- `quantile energy budget reachability reinforcement learning`
- `robot return charging stochastic energy reinforcement learning safety`

## Sources Checked

- Official NeurIPS proceedings for RC-PPO, QCRL, SDAC, and RESPO.
- PMLR for Sauté RL, RCRL, almost-sure constraints, DCRL, and Back-to-Base.
- Springer CAV record and full HTML for consumption MDP definitions and the
  almost-sure Büchi theorem.
- IEEE Xplore for Recovery RL.
- Author arXiv records for RAPCPO, energy-sufficiency CBF, and CC-SSP.
- Elsevier publisher record for STEP.
- PMLR/ICML 2025 for regret-free model-free LTL learning in unknown finite
  MDPs, and PMLR/L4DC 2026 for adaptive-conformal learned safety filters.

## Excluded Sources

- All MDPI results were excluded by source-quality policy.
- ResearchGate, review aggregators, news/blog summaries, and untraceable PDFs
  were used at most for discovery and are not evidence in the final table.
- Application-specific EV grid-charging work was excluded when it did not
  concern physical navigation or finite-battery reachability.
- Pure energy minimization without a reach/avoid or recharge constraint was
  excluded from the final 15.

## Unknowns

- RAPCPO full text was inspected through its author arXiv HTML.  It learns a
  reach-avoid critic, expected-cost critic, and successful-hit compensation
  factor; the normalized quantity used in the practical optimizer is explicitly
  described as a surrogate rather than a certified probability bound.  A final
  proof-line audit remains prudent, but the main separation is now verified.
- It remains unknown whether a less visible formal-methods paper already uses
  an extended-real energy-to-reload variable with a quantile objective.
- The energy-sufficiency CBF paper's peer-review/venue status was not established
  from the primary record; treat it as a technically relevant preprint.
- No published benchmark directly matches continuous LiDAR navigation with
  repeated task/recharge cycles and raw-policy calibration.

## Handoff Notes

- For writing: position RC-PPO, RAPCPO, and consumption MDPs before presenting
  the gap; omitting them would make the novelty discussion indefensible.
- For idea optimization: the candidate primitive is the CDF of a stopped
  extended-real resource-to-recharge variable, not another weighted reward;
  however a fixed-budget slice is equivalent to augmented-state reach-avoid and
  cannot be sold as a new principle.
- For direction scouting: search formal verification terms `consumption MDP`,
  `energy MDP`, `reload state`, and `Buchi`, not only safe-RL keywords.
- For experiment design: compare against fixed-threshold R3, oracle switching,
  QCPO/SDAC-style risk critics, RC-PPO/RAPCPO, and raw versus HOCBF-filtered
  deployment.
- For review: reject any infinite-horizon safety claim based only on a fixed
  per-cycle empirical failure rate.
