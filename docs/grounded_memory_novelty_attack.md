# Hostile Novelty Attack: Control-Grounded Structured Memory

## Reviewer A — theory

**Score: 6/10. Confidence: 5/5.**

The formal statements are mostly correct because they are deliberately conditional. The main weakness is that the architecture's learned grounding does not establish the premises needed for certified uncertainty contraction. Proposition 1 is a triangle inequality. Proposition 2 is the standard safety-filter separation argument. Proposition 3 is set monotonicity. The candidate does not prove that its memory is an approximate information state, only that it would be useful if it were one.

Fatal-to-theory concern: no theorem links finite data and grounding losses to a valid smaller uncertainty set under track swaps, abrupt maneuvers, or dropout.

Score-change condition: a new coverage/contraction result under weaker and verifiable assumptions than measurement-robust CBF or conformal-set methods.

## Reviewer B — novelty

**Score: 3/10. Confidence: 5/5.**

The candidate is a direct composition of:

- object slots/SlotFormer;
- RIM-style modular recurrence;
- MASIA/ExpoComm-style auxiliary grounding;
- receiver-specific routing;
- approximate information states and control-centric representation bounds;
- measurement-robust/output-feedback CBFs;
- modular safety filters.

The exact five-node graph is application-specific. Naming its fields Safety/Route/Energy does not create a new learning or control problem. The proposed non-interference theorem is already the purpose of a safety filter.

Score-change condition: identify a theorem that fails for a monolithic approximate information state but holds because of the typed routing graph, with a non-vacuous control consequence.

## Reviewer C — methodology

**Score: 5/10. Confidence: 4/5.**

The requested evaluation is substantially better than MAE-only work, but current data do not contain complete labels for route-message sufficiency or safe-trajectory energy overhead. A short synthetic motion benchmark cannot validate the full architecture. The closed-loop benchmark contains only a small scenario family and its learned uncertainty boxes were previously uncertified. A long neural experiment would therefore test an under-specified candidate and risk rediscovering estimator ranking.

Score-change condition: produce matched labels for all grounding heads, demonstrate at least one module-message ablation with a robust closed-loop effect, and preregister a training/evaluation protocol before long training.

## Area Chair synthesis

**Overall: 4/10 (reject as theory contribution).**

The architecture is sensible and potentially publishable as a robotics systems design after stronger experiments. The current theorem novelty does not meet the requested `22/30` gate.

## Numeric novelty score

| Dimension | Score | Maximum |
|---|---:|---:|
| New mathematical object | 3 | 10 |
| Separation from closest work | 3 | 10 |
| Nontrivial provable consequence | 4 | 10 |
| **Total** | **10** | **30** |

## Gate

- theorem object clear: yes;
- proof draft valid: yes for conditional propositions;
- known counterexample to claimed propositions: no, after claim narrowing;
- novelty score at least 22/30: **no**.

Therefore the long-experiment theory gate fails.
