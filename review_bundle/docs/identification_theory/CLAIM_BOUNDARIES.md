# Claim Boundaries

## Supported Claims

1. A rejected action's physical outcome is absent from the observed log.
2. A deterministic charger commitment removes task-continuation occupancy after the stopping history.
3. Without support or structural coupling, the corresponding counterfactual targets are not point identified.
4. A uniformly safe learner cannot reliably distinguish a safe untested action from an observationally identical catastrophic one without paying failure probability or using additional structure.
5. A coordinate-separable conservative filter can remain permanently inside a strict subset of the true safe set.
6. Physical probe energy can be represented as a consumable resource in the sequential state.

## Unsupported Novelty Claims

The package does not claim that any supported statement is new. In particular, it does not claim novelty for:

- safety-induced missing labels;
- no support after filtering;
- informative stopping;
- pessimism-induced reduced exploration;
- safe-set expansion;
- safe information acquisition;
- information gain under a safety constraint;
- energy-budgeted probing;
- task abort followed by return to base.

## Non-Claims

- no universal identification of forbidden actions;
- no safe boundary recovery without additional structure;
- no globally safe exploration in arbitrary unknown environments;
- no minimax-optimal sample complexity for UAV probing;
- no new algorithm;
- no physical UAV safety guarantee;
- no guarantee under model misspecification or distribution shift;
- no claim that two censoring indicators define a new statistical class;
- no conclusion from the running E1 experiment;
- no ICLR readiness claim.

## Terminology Guardrails

- Use **structural zero support**, not “hidden label” when the filter deterministically prevents execution.
- Use **counterfactual task suffix**, not “censored trajectory” alone, because the observed charger trajectory continues.
- Use **uniform safety–identification lower bound**, not “safety paradox.”
- Use **diagnostic information collapse**, not “new collapse theorem,” unless closest-work separation is established.
- Use **application-specific two-level decomposition**, not “new information structure.”

## Novelty Burden

To reopen the theory direction, a future result must state a theorem-level difference from each of:

- Khan et al. 2024 no-overlap partial identification;
- SafeMDP safe-set expansion and impossibility without regularity;
- Active Learning with Safety Constraints;
- ActSafe safe information-seeking trajectories;
- Budgeted MDP/BwK resource constraints;
- risk-sensitive abstention with catastrophic feedback.

Without that difference, the correct disposition is an empirical robotics/systems framing.
