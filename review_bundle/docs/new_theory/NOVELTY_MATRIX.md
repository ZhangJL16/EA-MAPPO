# Novelty Matrix — Loop 2

## Gate Decision

**NOVELTY GATE: FAIL FOR ICLR THEORY READINESS.**

The reformulation eliminates the false “two independent critics” semantics, but the broad operator remains a specialization/product of known Budgeted MDP, multi-constraint distributional RL, quantile-constrained RL, and policy-coupled calibration machinery. The residual distinction is precise and experimentally falsifiable, but not yet enough for a `>=6/10` novelty score without a stronger learning theorem or evidence.

| Candidate claim | Closest coverage | Decision |
| --- | --- | --- |
| remaining-risk state augmentation | Carrara 2019; Castellano 2022 | NOT NOVEL |
| budgeted Bellman operator | Carrara 2019 | NOT NOVEL |
| multiple distributional safety critics | Kim et al. 2023 | NOT NOVEL |
| cumulative-cost quantile constraint | Jung et al. 2022 | NOT NOVEL |
| learned/calibrated collision chance | Mao et al. 2024; MaxSafe 2025 | NOT NOVEL |
| policy-coupled selected-action coverage | Zheng and Jin 2026 | NOT NOVEL |
| conformal target-policy return interval | Zhang et al. 2023 | NOT NOVEL |
| safe return to base/charger | Back to Base 2025; UAV planning work | NOT NOVEL AS A GOAL |
| defective charger-return law retaining failure mass | no exact match in screened work | PLAUSIBLE REPRESENTATION CONTRIBUTION |
| collision-filter version induces energy-target discontinuity | no exact theorem match found | PLAUSIBLE STRUCTURAL OBSERVATION |
| C1 budget-omission lower bound | follows a simple two-route construction; related to BMDP necessity | CORRECT BUT TOO ELEMENTARY ALONE |
| C3 margin-stability result | simple support invariance theorem | CORRECT BUT TOO ELEMENTARY ALONE |
| repeated-sortie learning of the versioned defective law | no exact match found | EMPIRICAL/LEARNING HYPOTHESIS |
| two-resource one-way stopping | optimal stopping and budgeted CMDP cover general structure | APPLICATION-SPECIFIC, NOT THEORY NOVELTY |

## Reviewer-B Questions

**Q1. Why not two obvious safety critics?** The canonical predictor is one defective return law indexed by collision budget/version, plus its inducing local hazard. C1 proves an independent budget-agnostic energy critic is misspecified in some environments. This defeats the old semantics but does not by itself establish venue-level novelty.

**Q2. Why can ordinary CMDP not express it?** It can express the augmented process and expected constraints. Therefore expressibility is not a difference.

**Q3. Why is ordinary multi-cost CMDP insufficient?** Expected-cost critics do not identify the finite-energy tail or missing return mass, but distributional multi-constraint RL and quantile-constrained RL substantially close that gap.

**Q4. Why are energy-aware UAV planners insufficient?** They generally use planned-route or learned energy models rather than the selected shared policy's action-conditioned defective return law. This is a domain distinction, not yet a broad theoretical advance.

**Q5. Why is distributional RL plus threshold insufficient?** Without `b,eta` it can estimate the wrong target; with `b,eta` it becomes very close to a budgeted distributional RL specialization.

**Q6. Why is collision avoidance plus RTH threshold insufficient?** It ignores target change induced by the collision filter and missing return mass, but this must be shown causally.

**Q7. Why is goal-conditioned policy plus optimal stopping insufficient?** Generic stopping can express the latch; it does not supply the learned calibrated continuation law. The latch is not novel.

**Q8. What is the new object?** The narrow object is the versioned, budget-conditioned defective telemetry-energy return law induced by a learned selected-action collision envelope. No strong theorem beyond representation necessity and support stability has yet been established.

## Fatal Replacement Test

If an SDAC/QCRL/BMDP baseline conditioned on the same risk budget and trained on the same failure-retaining data matches the proposed method, the claimed theory contribution collapses. If policy-coupled/off-policy conformal calibration directly supplies the required selected-action coverage with no new sequential result, calibration novelty also collapses.

## Current Allowed Claim

“We identify and evaluate a versioned defective charger-return distribution whose target is induced by a learned collision-risk budgeted policy, and test whether modeling this endogenous coupling matters for repeated-sortie stopping.”

No stronger novelty wording is authorized.

## Redesign-2 Transport Verdict

| Transport claim | Closest coverage | Decision |
| --- | --- | --- |
| induced kernel changes critic target | filtered executed-action critic work, generic off-policy RL | KNOWN PRINCIPLE |
| path-law TV bounded by cumulative local kernel TV | standard maximal coupling / occupancy perturbation | KNOWN TECHNIQUE |
| coverage loses at most path-law TV | standard probability-metric robustness | NOT NOVEL |
| hard admission-boundary visitation controls return-law drift | no exact match found | NARROW SPECIALIZATION |
| transport-or-recalibrate gate | direct algorithmic consequence | PLAUSIBLE DESIGN, NEEDS EVIDENCE |

**Updated gate: FAIL.** The transport reframing supplies a useful diagnostic theorem but does not yet escape the product of known perturbation and conformal-shift results.

## Final-Redesign Identification Verdict

| Claim | Closest work | Decision |
| --- | --- | --- |
| outcomes observed only after a decision | selective-label literature | KNOWN PROBLEM |
| no identification without positivity | causal/OPE support condition | KNOWN RESULT |
| no identification under hidden outcome-dependent selection | MNAR/hidden-confounding literature | KNOWN RESULT |
| randomized commitment logging | randomized exploration/logging policy | KNOWN DESIGN |
| inverse-propensity CDF identification | off-policy risk/CDF estimation | KNOWN METHOD |
| finite-sample CDF concentration | OPRA/OPE literature gives stronger results | NOT NOVEL |
| label collection versus task utility | online selective-label/active collection | KNOWN TRADEOFF |
| charger-specific commitment-censored defective return | no exact UAV match found | APPLICATION SPECIALIZATION |

**Final novelty gate: FAIL.** After three major reformulations, the remaining contribution is a careful application and experiment design, not a defensible ICLR-level new theory or learning problem.
