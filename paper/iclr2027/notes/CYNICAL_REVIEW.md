# Cynical Review Handoff

Full report: `../../../ccfa-review-reports/2026-08-28-when-should-an-agent-return-iclr-conference-review.md`

Current expected outcome: **3/10, reject at the present evidence state**. This is not an acceptance-probability estimate; it scores the manuscript as it exists with every empirical cell empty.

## Decisive objections

- RL/CMDP: no evidence yet that Resource-to-Go plus a fixed one-way manager beats a direct switcher, classical planner, or matched constrained-control policy.
- World models/UQ: executed-WM plus Deep Ensemble may explain all residual error, and the positive identification theorem may be viewed as close to its exact-agreement assumption.
- Robotics/energy: one deterministic UAV simulator cannot support the generic-agent framing or rare-event return claims.

## Revisions applied from the review

- Added and fairly positioned Predictive Safety Network, DCRL, and persistent-UAV recharge rendezvous planning.
- Added a predictive-safety head, DCRL-compatible track, and receding-horizon planner to the required baseline contract.
- Reframed 5% Oracle headroom as a practical gate rather than a significance claim.
- Made 100 cycles a floor and required precision-driven rare-event sample sizes.
- Preserved SIRP as a deletion-gated candidate and all empirical outcomes as `PENDING`.

## Issues requiring new evidence

Oracle headroom; pair/interface/horizon factorial; ensemble-versus-reliability gate; paired return Pareto frontier; direct-control and planning comparisons; second-domain/filter replication; anonymous reproducibility package; independent proof audit.
