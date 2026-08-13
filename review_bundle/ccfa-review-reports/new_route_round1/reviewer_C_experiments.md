# Reviewer C — Experiments / Causal Evidence / RL Methodology

Mode: experiment-plan review
Materials: requested E0--E6 ladder, retained baseline artifact, no new results

## Summary and Score

The proposed causal ladder is appropriate, and freezing the navigation policy first is essential. However, no data currently support the central mechanism. The protocol must prevent policy/data leakage, calibration-on-evaluation, unequal candidate-action budgets, and survivor bias from charger-completed returns.

**Overall: 2/10 (strong reject as a paper today). Confidence: 5/5.**

## Scorecard

| Dimension | Score | Confidence | Evidence basis | Deduction / score-change condition |
| --- | ---: | ---: | --- | --- |
| Novelty | 2/5 | 3/5 | Novelty depends on experiments not yet run | Requires causal result |
| Soundness | 3/5 | 4/5 | Design is plausible | Censoring and off-policy target protocol unresolved in code |
| Evidence | 1/5 | 5/5 | No new-route results | Complete E1--E6 with 3--5 seeds |
| Significance | 4/5 | 3/5 | Stranding/collision tradeoff matters | Must show effect under shifts |
| Clarity | 4/5 | 4/5 | Metrics and stages are explicit | Add exact estimands/configs |
| Reproducibility | 2/5 | 5/5 | No implementation/tests yet at review time | Immutable run bundles and hashes required |
| Ethics / Limitations | 4/5 | 4/5 | Simulation-only boundary stated | Report compute and failure retention |

## Fatal Concerns

No fatal design flaw if the following are implemented. Missing all of them would be fatal for causal interpretation.

## Major Concerns and Exact Counterexamples

1. **Completed-return survivor bias.** Training only on successful charger arrivals removes high-energy/failed returns and makes upper bounds optimistic. Include censored handling and failure strata; never label truncation as completed return.
2. **Unequal action search.** A method evaluating 64 candidates has more control capacity and more selection bias than a one-action baseline. Equalize candidate sets and inference budget.
3. **Calibration leakage.** Reusing held-out seeds for tuning margins invalidates final coverage. Separate train, calibration, validation, and final evaluation streams.
4. **Policy drift leakage.** Joint fine-tuning changes `pi_c` while replay contains old returns. Freeze policy first; later stratify by policy hash and recollect/reweight.
5. **Stopping confound.** One-way commitment can reduce chatter but also commit earlier. Report trigger time, remaining energy, path length, tasks, and paired counterfactual thresholds.

## Score-Change Conditions

- Raise to 5: E1--E3 complete with five seeds, fixed held-out streams, confidence intervals, and no leakage.
- Raise to 6: E4--E6 show robust, causal, data-efficient gains and disclose failures.
- Lower to 1: smoke tests or partial runs are reported as paper evidence.
