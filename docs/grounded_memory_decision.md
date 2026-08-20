# Control-Grounded Structured Memory Decision

> **FINAL UPDATE (2026-08-20):** The last real-label and verified-contraction exploration is complete. See `docs/grounded_memory_final_rejection.md`. Structured memory is rejected as the paper core; no five-seed long experiment was launched.

## Exploration status

- Required ExpoComm paper read in full: complete.
- Primary-paper matrix: 71 papers.
- Architecture semantics: complete.
- Formula derivation and proof audit: complete.
- Hostile novelty attack: complete.
- Short typed-grounding/routing diagnostic: complete.
- Long candidate freeze: not performed because the gate failed.
- Five-seed long experiment: not launched.

## Best architecture found

**Fixed-Routing Object-Centric Grounded Memory (FOGM)** is the best semantically coherent architecture:

- per-object recurrent slots;
- explicit ego state;
- separate Safety/Route/Energy messages;
- fixed physics-informed routing;
- certified and learned channels separated;
- hard filter consumes only certified state;
- learned messages affect proposals/ranking only.

It is a good systems architecture but is not a defensible new theory contribution in its current form.

## Gate assessment

| Gate | Result |
|---|---|
| clear mathematical object | pass |
| valid proof draft | pass for narrowed conditional claims |
| no counterexample to narrowed claims | pass |
| novelty score at least 22/30 | **fail: 10/30** |
| short diagnostic distinguishes candidate | **fail** |
| route and energy grounding labels available | **fail** |

## Why the candidate is rejected for long training

1. Object slots, modular recurrence, grounding, sparse routing, and certified safety separation all have direct prior art.
2. The strongest theorem candidates are compositions of approximate information states, robust-set monotonicity, and modular safety filters.
3. The only potentially new link—learned grounding causing sound certified-set contraction—is unproved and contradicted by track-swap/dropout failure modes without additional assumptions.
4. The short diagnostic shows no meaningful advantage over a generic GRU on available labels.
5. Current data cannot test Route or Energy messages without inventing labels.

## Recommended next research move

Do not run the preregistered five-seed neural study yet. If this route is revisited, first collect a small matched dataset containing:

- explicit route/corridor decisions and detour decomposition;
- safety-filtered realized energy and nominal-vs-safe overhead;
- track association and uncertainty-set validity labels;
- adversarial ID-swap, new-obstacle, and dropout episodes.

Then ask a narrower systems question: whether typed routing improves robustness and interpretability at fixed safety, not whether it creates a new general memory theorem.

## Final status

`GROUNDED_MEMORY_LONG_CANDIDATE = NONE`

`LONG_EXPERIMENT_LAUNCHED = FALSE`

`FORMAL_SAFETY_500K_STARTED = FALSE`
