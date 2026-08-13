# Safe Probing Gate

## Decision

No new probing algorithm is proposed in this round.

## Candidate Optimization

A generic boundary-directed probe would choose a policy or action $q_t$ by

$$
\max_{q_t}\;\mathcal{I}_t(q_t)
$$

subject to

$$
\Pr(\text{collision under }q_t\mid\mathcal{H}_t)\le\delta_t,
$$

$$
\Pr(\text{charger energy shortfall after }q_t\mid\mathcal{H}_t)\le\alpha_t,
$$

$$
C_t^E(q_t)\le B_t^E,
$$

where $\mathcal{I}_t$ measures information about a declared boundary target.

## Why This Is Not Yet a New Algorithmic Principle

- maximizing information subject to unknown safety constraints is SafeOpt/safe active learning;
- choosing informative safe trajectories is ActSafe;
- safe-set expansion with returnability is SafeMDP;
- information-gain selection is information-directed sampling or active experimental design;
- explicit energy/risk budgets are BMDP/BwK state or resource constraints;
- immediate commit after a probe is a safe reset/return-to-base action.

Changing the application labels to collision, battery, and charger does not produce an irreducible method.

## Conditions Required Before Algorithm Design

An algorithm would be justified only after proving at least one of:

1. a lower bound involving the interaction of action rejection and absorbing commitment that cannot be represented as an augmented safe-exploration problem;
2. an identification result that uses repeated charger-origin sorties in a way unavailable to generic episodic resets;
3. a resource-information rate distinct from a standard sample-complexity bound under a knapsack budget;
4. a falsifiable separation from SafeMDP, Active Learning with Safety Constraints, and ActSafe under matched assumptions.

No such result is currently available.

## Forbidden Shortcut

The system may not execute actions labeled forbidden by hidden simulator ground truth merely to generate a paper benchmark. Any empirical study must distinguish:

- labels obtainable during safe deployment;
- oracle labels used only for retrospective evaluation;
- simulator labels with a documented sim-to-real assumption;
- deliberately hazardous physical trials, which require independent safety and ethics review.

## Empirical Value Despite Theory Block

As a robotics methodology study, one could still compare conservative filtering, uncertainty-aware probing, and no-probing policies under equal physical interaction budgets. Such a study would test engineering tradeoffs, not claim a new identification theory.
