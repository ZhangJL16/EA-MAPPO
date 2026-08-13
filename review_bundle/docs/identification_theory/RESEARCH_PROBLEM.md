# Research Problem

## Candidate Question

Consider a persistent resource-constrained agent whose deployed safety mechanism controls which physical outcomes can ever enter the training data. At each task step, a nominal policy proposes an action, an action-level filter may reject it, and a trajectory-level stopping rule may irreversibly commit the system to a charger. The candidate question is:

> Which counterfactual safety and continuation quantities are identifiable from the resulting logs, and what extra structure is necessary to acquire informative labels without exceeding collision and energy budgets?

This is an identification question before it is an algorithm-design question.

## Candidate Phenomenon

Two observation channels are suppressed:

1. **Action-level suppression.** A rejected task action has no observed physical transition, collision outcome, or energy cost.
2. **Trajectory-level suppression.** After charger commitment, the task-continuation suffix is not realized.

The safety mechanism is adaptive: it depends on the current history and learned uncertainty. Therefore it changes the future occupancy measure and the set of labels that can be collected.

## Candidate Contribution Hypotheses

The exploration initially tested four hypotheses:

1. two-level safety censoring creates a new non-identifiability class;
2. conservative filtering can create a dynamic information-collapse fixed point;
3. boundary-directed safe probes can restore identification;
4. probe energy consumption creates a new resource-aware sample-complexity problem.

These are hypotheses, not claims.

## Prior-Art Gate Result

The closest-work attack in `CLOSEST_WORK.md` rejects the first-pass novelty interpretation:

- action-level non-observation is selective labels, bandit feedback, or zero action support;
- post-commitment suffix non-observation is zero history-action occupancy after an absorbing stopping decision;
- safe-set expansion and impossibility without smoothness/seed structure are central to SafeOpt and SafeMDP;
- information-seeking safe trajectories are central to ActSafe;
- physical query budgets can be represented by budgeted MDPs or bandits with knapsacks;
- adaptive model-induced data distributions are covered by performative learning and feedback-covariate-shift frameworks.

The remaining task is therefore adversarial: determine whether a theorem-level separation survives these reductions. No algorithm may be justified merely by the UAV story or by combining the existing components.

## Unit of Analysis

A sortie begins in task mode and ends after charger commitment and completed charging. The identification analysis focuses on the pre-commitment task process and treats the charger mode as an observed absorbing mode for the missing task suffix. It does not introduce a recovery controller, recoverability object, certificate, or persistent authority lifecycle.

## Success Criterion

The direction is promising only if there is a precise target and theorem that cannot be obtained as a direct instance of:

- no-overlap or partial-identification OPE;
- selective labels or partial monitoring;
- SafeOpt/SafeMDP/ActSafe safe-set expansion;
- constrained or budgeted exploration;
- MNAR identification with added structure;
- performative policy-induced data collection.

Descriptive coexistence of action censoring, stopping, and battery consumption is insufficient.
