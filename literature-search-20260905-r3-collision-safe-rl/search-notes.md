# Search notes

Date: 2026-09-05  
Question: Which single safe-RL mechanism best repairs collision learning in the
existing R3 structured-LiDAR SAC without replacing SAC or stacking recovery,
planning, and extra-critic modules?

## Search boundary

- Primary-source venues and author manuscripts only: PMLR, NeurIPS
  proceedings, RSS proceedings, IEEE paper/author manuscript, and arXiv when no
  open proceedings copy was available.
- Method clusters: expected-cost CMDP optimization, early/stochastic
  termination, recovery/intervention, reachability/state-wise safety,
  Lyapunov/CBF action projection, and sampled-data collision constraints.
- Compatibility filters: off-policy continuous control; direct use of R3
  observations/actions; no safe demonstrations; no certified recovery policy;
  no online MPC; no second learned policy; at most one new learned safety
  object; later compatibility with energy-return research.
- Local experimental facts were inspected only from the repository and were
  never sent as literature-search queries.

## Queries and screening logic

Queries combined the phrases `safe reinforcement learning`, `collision
avoidance`, `state-wise constraints`, `reachability`, `recovery policy`,
`early terminated MDP`, `constraints as terminations`, `SAC`, `control barrier
function`, and `sampled-data`. Citation chasing was then limited to the closest
mechanism papers named by the primary sources.

Papers were retained only when they changed one of four decisions:

1. whether collision should be an expected budget or a binary trajectory event;
2. whether an additional critic/policy is necessary;
3. how constraint violation enters the Bellman target;
4. whether continuous-time barrier values match the simulator's sampled hold.

## Reproducibility notes

- CaT was checked at equation and algorithm level from arXiv:2403.18765 and its
  IROS DOI record. Its objective assumes lower-bounded rewards and treats
  termination as removal of future learning reward, not necessarily a simulator
  reset.
- SoloParkour was checked from the CoRL proceedings PDF. It explicitly extends
  CaT to an off-policy DDPG-family learner and stores violations for recomputing
  termination probabilities; its ten-critic REDQ implementation is not part of
  the transferable CaT primitive and is rejected here.
- ET-MDP was checked from arXiv:2107.04200. Its equivalence needs a sufficiently
  low termination reward and hard termination can restrict state visitation.
- Repository evidence was read from the exact R3 config, environment reward and
  termination code, 500-task evaluation records, training diagnostics, and the
  completed dual-viability diagnostic.

## Search limitation

No paper proves that a neural UAV policy using finite LiDAR observations has
zero real-world collision probability. The selected literature can justify a
training objective and a conditional sampled-data shield theorem; it cannot turn
finite simulation results into an unconditional certificate.
