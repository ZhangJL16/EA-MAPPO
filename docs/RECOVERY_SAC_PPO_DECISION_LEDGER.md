# Scratch SAC/PPO result interpretation and next research decision

Date: 2026-09-06. Registered after the SAC8192 startup checkpoint and before
either arm's final evaluation. Target venue lens: ICLR/ICML-style learning
mechanism evidence. The paper-level innovation remains energy viability, not
the choice between two standard navigation baselines.

## What this pair can identify

Both arms use the same locked nonterminal contact recovery, raw reward,
gamma=.99, directional structured LiDAR, task schedule family and500 held-out
seeds. Therefore their final difference describes the performance of two
complete native algorithm configurations in this environment.

It does NOT isolate a single mathematical ingredient. SAC and PPO necessarily
differ in replay, Bellman target, action distribution, entropy regulation,
critic type and optimizer exposure. With one initialization, it also cannot
establish an algorithm-family ranking.

For524288 transitions, the maximum planned PPO sample presentations are

\[
128\ \text{rollouts}\times10\ \text{epochs}\times4096
=5{,}242{,}880,
\]

before optional KL early stopping. SAC makes approximately519288 post-warmup
actor and critic updates, each using256 replay entries, or about132.9million
sample presentations to each loss. The ratio is about25.4. Matching environment
interactions is therefore not matching compute or data reuse; both are reported.
Repeating PPO data25times is not a valid repair because the likelihood ratio
and advantages become stale as the policy moves.

PPO now updates every512steps/worker, rather than waiting for completed4000-step
tasks. It bootstraps rollout cuts using V(s). Its advantage trace still weights
a TD residual k steps away by

\[
(\gamma\lambda)^k=(.99\times.95)^k=.9405^k,
\]

which is approximately.0197 at64steps and3.89e-4 at128steps. Thus cadence is
repaired, while long-range credit still depends strongly on critic accuracy.

## Pre-registered diagnostics

Primary outcomes on exactly the same500deterministic scenes:

1. arrival rate and collision-free arrival rate;
2. unified contacts per task and timeout rate;
3. successful-path ratio, always with successful sample count;
4. energy usage as a descriptive outcome, not a trained objective;
5. transitions, wall time, actual optimizer steps and PPO epochs retained
   before KL stopping.

Mechanism diagnostics, computed without selecting a checkpoint on test results:

- deterministic action norm, learned exploration scale and goal-direction dot
  product on common states;
- final-distance distribution for timeouts, distinguishing near-goal failure
  from no-traversal failure;
- PPO approximate KL, clipping fraction, explained variance and actual joint
  updates; SAC entropy coefficient and Q loss;
- progress/contact timing traces on a fixed small diagnostic scene set. These
  explain behavior; they are not additional success gates.

## Outcome-conditioned next single change

### SAC succeeds, PPO fails

First test the observed failure mechanism rather than add a safety module.

- If PPO repeatedly stops before most of10epochs and its deterministic actions
  remain small, run one PPO-only optimization control: remove the optional
  target-KL early-stop while retaining clip=.2. This is the native clipped PPO
  objective; report realized KL rather than assume clipping enforces a trust
  region. If KL becomes destructive, the result rejects this repair.
- If PPO uses most epochs but timeouts cluster near the goal or behind obstacles,
  do not change epoch count. Test one credit-horizon change (lambda.95→.99),
  holding gamma=.99 and architecture fixed.
- If the actor has substantial goal-aligned magnitude but oscillates because
  stochastic scale dominates, test adaptive exploration separately. Do not
  mix it with the credit-horizon run.

### Both succeed

The old PPO failures are attributable to their complete coupled systems, not to
PPO being unable to learn this navigation task. Choose the baseline using
replicated seeds and compute/safety trade-offs, then add exactly one established
collision-cost method without a runtime shield: SAC-Lagrangian if SAC is chosen,
or FOCOPS/PPO-Lagrangian if PPO is chosen. This collision method is an enabling
baseline, not the energy contribution.

### Both fail

Algorithm replacement is not the supported lever. Audit whether the locked
reward and finite observation distinguish globally useful detours. The first
single intervention is then a navigation curriculum or state-memory test, chosen
from fixed-trajectory diagnostics; no constrained-RL head is added yet.

### PPO succeeds, SAC fails

Inspect SAC reward-temperature units and replay Q extrapolation. Run a fixed
entropy-coefficient or reduced update-to-data control only if the logged entropy
or Q diagnostics support it. Do not infer on-policy superiority from one seed.

Any branch requires at least three scratch seeds before a method-level claim.
The current500 scenes measure evaluation uncertainty conditional on one learned
model; they do not measure training-seed uncertainty.

## Relation to the energy research question

Energy safety is not low average consumption. In an augmented state z=(x,b,h),
with charger C, physical-contact set U, deadline h and remaining battery b,
define the stopped resource-to-recharge variable

\[
Y_h^\pi(z)=\begin{cases}
\sum_{t<\tau_C}e_t,&\tau_C\le h,\ \tau_C<\tau_U,\\
+\infty,&\text{otherwise}.
\end{cases}
\]

Then

\[
\Pr(Y_h^\pi(z)\le b-r)
\]

is exactly safe, timely return within expendable battery b-r. This makes energy
safety a resource-augmented reach-avoid/viability problem: collision safety
avoids an immediate unsafe set, while energy safety avoids states from which the
charger is no longer recoverable. Expected-energy regression does not represent
the failure atom and is insufficient for this event.

The strongest residual idea is one action-conditioned, budget-monotone neural
return-option value Q_op(z,b,a), used during training to teach the actor which
actions preserve rechargeability. It should not be stacked onto an unresolved
navigation learner. The current pair supplies the prerequisite: a defensible,
unshielded base policy and its real executed-action distribution.

Novelty remains uncertain. Recovery RL, reach-avoid RL, persistent-safety RL,
Back-to-Base and consumption MDPs already cover broad backup-policy, viability,
return-set and resource-state primitives. A publishable contribution would need
a new error-to-closed-loop-risk result or a genuinely discriminating adaptive-
trajectory calibration mechanism, plus lower task abandonment—not a renamed
energy predictor or a collection of standard modules.

## Challenges survived and open risk

Overlap challenge: safe return plus a learned viability value is already close
to Back-to-Base/Recovery RL. Revision: the paper claim cannot be merely a return
controller; it must concern the resource distribution/failure atom and a
quantified approximation-to-lifecycle-risk link.

Evidence challenge: a successful SAC/PPO baseline says nothing about energy
viability. Revision: after base navigation and collision-cost training, the
energy method needs direct held-out tests of dangerous underestimation, safe
return probability at multiple budgets, calibration, task utility and adaptive
closed-loop failure—not only energy MAE/RMSE.

Source-supported algorithm primitives: PPO paper sections3–5; SAC paper
sections4.2 and5; ICLR2020 *Implementation Matters in Deep Policy Gradients*;
the primary sources listed in docs/DUAL_VIABILITY_LITERATURE_REVIEW.md.
All outcome claims in this ledger remain unmeasured until the registered run
and independent training seeds complete.
