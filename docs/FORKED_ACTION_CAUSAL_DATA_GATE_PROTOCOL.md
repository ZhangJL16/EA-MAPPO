# LiDAR Forked-Action Causal Data Gate

Date: 2026-09-04  
Protocol: `LIDAR_FORKED_ACTION_CAUSAL_DATA_GATE_V1`

## Claim tested

At an identical UAV state and obstacle scene, do different proposed actions
produce enough measurable variation in collision-safe rechargeability and
physical energy to identify an action-conditioned neural critic?

This protocol tests data identifiability only.  It neither trains the actor nor
claims deployment safety.

## Independent unit and paired intervention

The independent unit is the scene seed.  Each scene supplies correlated anchor
states; each anchor supplies nine paired action branches.  All anchor/action
rows from one scene remain together in every later split.

At an anchor, the branches share position, velocity, task goal, charger,
remaining deadline, obstacle layout, LiDAR observation, and continuation
policy.  The registered macro-action is the only changed variable.

## Anchor selection

Fifteen source scenes are selected equally across the five locked distance
buckets.  Four anchors are selected per scene from the completed nominal R3
trajectory, balancing task and return legs where available.  Within each leg,
one anchor prioritizes minimum analytic obstacle clearance and one represents
the temporal interior.  Final terminal states are excluded.

The source state is reconstructed from the saved normalized velocity, active
goal direction, and log-distance.  The static obstacle scene is regenerated
from the locked world seed.  Full LiDAR is recomputed before any branch.

## Registered action set

Each candidate is repeated for eight policy steps (1.6 simulated seconds):

1. frozen R3 nominal action;
2. velocity-cancelling brake action;
3--8. nominal action plus/minus 0.45 on x, y, and z;
9. acceleration toward the nearest obstacle;
10. acceleration away from the nearest obstacle.

After clipping, duplicate candidates remain recorded but count against the
executed-action diversity Gate.  The actual implementation contains ten
branches because the six signed perturbations are separate; this count is
authoritative over informal earlier references to nine.

## Execution semantics

HOCBF is disabled only during the repeated candidate prefix so raw unsafe
actions can produce labels.  Collision or boundary contact during any step
makes the branch unsafe and ends it early.  If the prefix remains safe, the
unchanged HOCBF and frozen deterministic R3 policy are restored.  A task-leg
branch must finish the task and subsequently reach the charger; a return-leg
branch must reach the charger.  Per-leg deadlines inherit the remaining source
deadline, with a full return deadline after task completion.

The target event is

\[
S=\{\tau_C<\tau_U,\ \tau_C\le h\},
\]

and physical energy is accumulated on every executed transition.  Budget
queries are created later from the paired stopped-energy outcomes; they are not
used to select anchors or actions.

## Leakage and stored fields

The anchor record stores full pre-action observation, physical state, both
goals, leg, remaining horizon, obstacle layout digest, nominal/candidate action,
and nearest-obstacle geometry.  Branch labels store first executed action,
energy, steps, collision/boundary flags, task success, charger success, and end
reason.  No branch outcome is fed back into another candidate.

## Pilot Gate

- 15 scenes, three per distance bucket;
- four anchors per scene;
- ten branches per anchor, at most 600 branches;
- at least five distinct first executed actions (L2 separation 0.05) for 80% of
  anchors;
- unsafe prevalence in [2%, 40%];
- at least 20% of anchors show either safe/unsafe variation or at least 5%
  relative energy range among successful branches;
- exact common anchor observation and obstacle digest across every paired
  branch;
- every branch terminal or explicitly censored; no partial branch is scored.

If class support or paired variation fails, modify anchor/action sampling before
training.  If it passes, the next Gate compares geometry+action,
LiDAR-without-action, and LiDAR+action critics on held-out scenes.
