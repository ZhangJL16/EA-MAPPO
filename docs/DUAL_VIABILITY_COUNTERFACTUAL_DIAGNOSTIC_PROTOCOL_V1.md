# Dual-Viability Counterfactual Diagnostic Protocol V1

Frozen: 2026-09-05, before collecting any corrected outcomes  
Evidence class: inspected-data mechanism diagnostic; not confirmation

## Purpose

Test whether the failed closed-loop pilot used the wrong counterfactual target,
and whether a deployment-aligned option-preservation target is sufficiently
nondegenerate to justify fresh model training.

## Immutable source

- Exact anchors: `artifacts/rcps_meet_confirmation_fork_150scenes_20260904_v1`
- Scope: all 310 task-leg anchors from 150 already inspected scenes
- Candidate set: the ten actions already frozen in each anchor
- Navigation/recovery actor: frozen R3 500k checkpoint recorded by the source
- Obstacle layout, state, task goal, charger, scene seed, and anchor hash:
  exactly as recorded by the source
- Control interface: unchanged sampled-data HOCBF; candidate actions are not
  executed raw
- Horizon: 4,000 recovery-policy steps after retarget

## Arms

### R0: return now

Restore the exact anchor, verify its registered observation hash, immediately
retarget to the charger without changing position or velocity, and execute
frozen R3 until success, unsafe terminal, or the fixed horizon.

### R1: one action then return

Restore the same anchor; execute exactly one frozen candidate through the
normal HOCBF; retain the resulting position and velocity; retarget to the
charger; execute frozen R3 to termination.

### Historical comparator: complete then return

Read only the existing branch with the same anchor and candidate. It used an
eight-step raw prefix, continued toward the task goal, and returned only after
task completion. It is not rerun and is not treated as deployment-aligned.

## Outcomes

For every arm record collision, boundary contact, charger hit, censoring,
executed first action, HOCBF intervention, steps, and total realized energy.
For a normalized available-energy budget \(b\), define

\[
Y(b)=\mathbf1\{\text{charger hit without collision/boundary/censoring and }
E/\mathrm{capacity}\le b\}.
\]

Evaluate the 15 budget knots frozen by the existing monotone critic.

## Diagnostic summaries

1. Paired historical-completion versus option-preservation discordance at each
   budget, including both directions.
2. Paired return-now versus option-preservation discordance.
3. Fraction of anchors with mixed R1 candidate outcomes at each budget.
4. Fraction of candidate proposals changed by HOCBF and the norm of the change.
5. Return success, exhaustion-equivalent budget failure, collision, boundary,
   and censoring by distance bucket.
6. Energy distribution and option-preservation rates by candidate name.

## Decision logic

This protocol has no promotion claim and no post-hoc pass threshold. It informs
the next design as follows:

- If R0 frequently fails structurally, repair frozen recovery behavior before
  learning any gate.
- If R1 varies mainly by anchor but not candidate, prefer a state return-value
  plus learned/known successor model over an action critic.
- If R1 has useful within-anchor variation, collect entirely fresh R1 data for
  training, scene-grouped calibration, and untouched confirmation.
- If historical and R1 labels are nearly identical, the pilot failure is more
  likely dominated by adaptive repeated thresholding than target semantics;
  calibrate the whole stopping trajectory next.

All conclusions remain simulator- and distribution-conditional.
