# Goal Switching Theorem

## Two-Resource Stopping Rule

Let `z_t in {TASK, CHARGER_COMMITTED}` and `z_0=TASK`. Let `A_task_DB(x_t,e_t,b_t)` contain task actions with a dual-budget-feasible charger continuation after the action. Let `A_charger_DB` contain charger-goal actions admitted by the same collision/energy semantics.

Define

`tau_commit = inf{t: A_task_DB(x_t,e_t,b_t)=empty and A_charger_DB(x_t,e_t,b_t) is nonempty}`.

At this stopping time set `z=CHARGER_COMMITTED` and `g=g_c`. Thereafter the mode remains committed until charger arrival and completed charging start a new sortie.

## T8 — Stopping-Time Semantics

Under observable `x_t,e_t^-,b_t` and fixed version `eta`, both action sets are `F_t`-measurable, so `tau_commit` is a stopping time. It is a feasibility boundary for two noncompensable resources, not an optimal-stopping theorem and not a learned switch policy.

The rule does not prove it commits early enough if the learned/calibrated bounds are invalid. It also does not claim throughput optimality.

## T9 — No Chattering

Each sortie has at most one `TASK -> CHARGER_COMMITTED` transition.

**Proof.** Before the first transition the mode is TASK. After it, the mode is absorbing until the sortie ends. A second transition would require re-entry to TASK, contradicting absorption.

This theorem eliminates task/charger mode oscillation by construction. It says nothing about oscillatory low-level actions under the charger goal.

## Counterfactual Baselines

Experiments must compare reversible threshold switching, hysteresis, and one-way commitment at matched trigger information. Trigger time, residual `e,b`, path length, empty-set frequency, task throughput, and charger arrival must be reported to separate latch stability from conservative early stopping.

## Final Research Status

T8--T9 remain valid structural properties. The stopping-time measurability and absorbing latch are not novelty claims.
