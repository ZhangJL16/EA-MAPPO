# Idea-grounding packet

## Decision

The best current fit is **sampled-data constraint-termination SAC**: retain the
R3 structured-LiDAR actor and SAC architecture, replace the local Jacobian
shield auxiliary loss with a CaT-style Bellman continuation hazard computed
from the existing sampled-data robust HOCBF projection, and retain that same
HOCBF only as the final execution guard.

This is an adaptation of prior mechanisms, not yet a novelty claim.

## Grounded gap

Two independent evidence clusters support the gap:

1. Local R3 evidence: physical collision is nonterminal; the first collision
   costs 1.2, repeated contacts cost 0.42, task completion pays 100, and one
   failed evaluation trajectory accumulated 2,240 collision steps after the
   simulator repeatedly projected it back to a legal surface. The optimized
   return therefore does not equal first-collision-free task success.
2. Safe-RL evidence: expected-cost methods need extra value/dual machinery;
   hard termination is sparse; CaT supplies a dense termination signal without
   new networks; sampled-data CBF theory says the constraint must match the
   held controller rather than continuous time only.

## Mechanism primitive versus project inference

- Source-supported primitive: constraint magnitude may reduce future learning
  return through stochastic termination without an extra critic (CaT), and the
  construction has been used with off-policy DDPG-family learning
  (SoloParkour).
- Project inference: use distance from the R3 nominal action to its
  sampled-data robust HOCBF feasible action set as the *only* continuous
  collision constraint. This replaces hand-tuned collision/intervention reward
  penalties and the stale local-Jacobian actor loss.
- Train a nonnegative reach-avoid value with killed-kernel potential shaping.
  The goal-distance term telescopes under the same survival-dependent discount,
  so it supplies dense progress feedback without altering within-state action
  rankings or reintroducing an independently weighted navigation reward.
- Existing-theory boundary: the runtime sampled-data HOCBF can supply a
  conditional forward-invariance statement; the learned policy and CaT
  objective alone cannot.

## Causal chain

Nonzero nominal-to-safe-set distance
→ lower SAC continuation probability at that exact state/action
→ unsafe nominal actions lose downstream goal value in the reward critic
→ actor moves toward the zero-distance feasible set without differentiating
through the QP
→ fewer HOCBF interventions/emergency brakes and fewer first-contact failures,
while the final filter still handles residual learning error.

## Rejected branches

- SAC-Lagrangian/PID: adds a cost critic and dual dynamics; optimizes average
  budget rather than the exact binary failure semantics.
- Recovery/SAILR: adds a policy/critic and assumes a reliable backup that R3
  does not possess.
- Reachability/RCRL: semantically stronger but adds a learned safety value; keep
  for a later joint return-energy formulation.
- Learned CBF or another projection layer: dynamics and obstacle geometry are
  already explicit, while the existing problem is actor dependence on that
  projection.
- Plain first-contact termination: keep as an ablation; it is too sparse to be
  the sole selected signal under a strong online shield.

## Protocol anchors

- Foundation checkpoint: R3 500k actor/structured encoder.
- No new neural module; reset the two existing reward critics because the
  Bellman operator changes; disable the Jacobian bridge.
- One matched hard-termination control and one selected soft-termination arm;
  no PPO, MPC, recovery policy, third critic, or energy loss in this phase.
- Evaluate paired task success/path efficiency, nominal constraint violation,
  HOCBF intervention/emergency use, and first physical collision at episode
  level. Report shield-on deployment and shield-off actor diagnostics
  separately.

## Open risks

- CaT's termination probability is an optimization device, not a calibrated
  collision probability.
- Signed reward handling must preserve the base task ordering; simply
  multiplying the current mixed-sign R3 reward by a hazard is not justified.
- Off-policy replay must store the nominal action and constraint quantity and
  recompute or version the hazard consistently.
- A conservative sampled-data shield can create infeasible/deadlock states;
  those transitions must fail closed and remain separately visible rather than
  being called successful safe navigation.

## Independent overlap and evidence challenge

An independent GPT-6 theory review challenged the first draft in two places.

1. **Overlap challenge.** CaT already supplies stochastic termination and
   SoloParkour already extends it off policy. Therefore neither “CaT for SAC”
   nor projection-distance termination alone is claimed as novelty. The honest
   research question is the measurable action-aliasing failure caused by a
   safety projection and whether a constraint-terminated reach-avoid value
   removes shield dependence without another safety critic.
2. **Evidence challenge.** Changing ordinary to robust HOCBF, disabling the
   bridge, resetting critics, and changing the target in one arm would be
   unidentifiable. The protocol therefore first evaluates the frozen interface,
   then compares two otherwise identical fine-tunes: \(\kappa=0\) versus one
   preregistered \(\kappa>0\).

The review also rejected direct CaT multiplication of R3's mixed-sign reward
and standard SAC soft continuation. The final construction uses a nonnegative
killed reach-avoid value and killed-kernel potential shaping; entropy is only a
temporary actor regularizer.
