# Research Question Card: Energy Option Preservation

Date: 2026-09-05  
Status: active mechanism study; novelty not established

## Problem

The frozen R3 navigation policy can usually finish a task, but an energy-aware
controller must ensure that its next task action does not destroy the ability to
reach the charger safely with the remaining battery. The failed RCPS pilot
learned a different event: candidate action followed by continued task
execution and only then return. A low probability for that event does not imply
that immediate return is feasible, and repeated rejection produced
near-universal early mission abandonment.

## Primary question

For a battery-augmented navigation state \(x=(s,b,h)\), can a learned lower
score for

\[
Q_{\mathrm{op}}^\kappa(x,a)
=\Pr(\text{safe charger hit}\mid A_0=a,\ A_{t\ge1}\sim\kappa)
\]

identify task actions that preserve a frozen recovery policy's return option,
while retaining materially more task utility than immediate-return or
completion-feasibility gating?

Here \(a\) means the action after the unchanged HOCBF execution interface, and
\(\kappa\) is frozen R3 retargeted to the charger.

## Gap supported by current evidence

- Recovery RL and intervention methods already separate task and recovery
  policies; reach-avoid RL and reach-avoid safety filters already characterize
  safe return to a target set.
- Consumption-MDP theory already models nonnegative resource use and reload
  states.
- Therefore “use a recovery policy,” “add battery to the state,” and “predict
  return probability” are prior art, not sufficient novelty.
- The project-specific unresolved gap is semantic and statistical: learn and
  distinguish completion viability, return-now viability, and one-step option
  preservation from paired counterfactuals, then connect the third quantity to
  a recursive closed-loop risk statement without pretending pointwise RCPS
  calibration covers adaptive repeated use.

## Falsifiable hypotheses

1. At one or more operational battery levels, completion viability and option
   preservation have nontrivial paired discordance on identical anchors.
2. Some currently recoverable anchors contain candidate actions that destroy
   recoverability, so action conditioning adds information beyond a state-only
   return score.
3. A monotone-budget option-preservation model can reduce unsafe terminal
   outcomes relative to ungated R3 without the task-completion collapse of the
   prior gate.
4. Whole-trajectory calibration/evaluation will be materially less optimistic
   than branch-level calibration, quantifying the price of adaptive reuse.

## Minimal decisive study

Use the exact 310 task-leg anchors from the already inspected 150-scene
confirmation data. From every anchor, run one immediate-return rollout. For
each of its ten frozen candidate actions, execute exactly one normal HOCBF step,
preserve the resulting state and velocity, and then return under frozen R3.
Compare these 3,410 outcomes to the existing completion-then-return branches at
the same anchor/action/budget. This is development evidence only.

## Decision rule after the study

- Continue to fresh train/calibration/confirmation splits only if the new
  labels are nondegenerate across budgets and either old/new semantic
  discordance or within-anchor action sensitivity is practically material.
- If option-preservation labels are almost entirely state-only, simplify to a
  return-value critic plus an explicit one-step successor model; do not retain
  an unnecessary action head.
- If immediate-return itself frequently fails from nominal task states, repair
  the recovery policy/corridor before training a gate.
- No oral-level novelty claim is permitted until nearest-paper overlap,
  trajectory-level risk, multiple seeds, and raw-versus-HOCBF policy behavior
  are resolved.

## Principal risks

- `Back to Base` (L4DC 2025) is a very close conceptual predecessor for
  reach-avoid return to charging/reset regions.
- The one-step action may have little measurable effect at a 0.2 s control
  period, making an action-conditioned critic statistically unnecessary.
- HOCBF may erase candidate diversity; this must be measured rather than
  bypassed.
- Finite simulator rollouts yield empirical evidence, not a physical flight
  certificate.
