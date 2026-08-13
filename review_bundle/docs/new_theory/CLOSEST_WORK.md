# Closest Work — Coupled Dual-Budget Route

Search date: 2026-08-13. Only public primary papers/official proceedings are used for novelty decisions. Search is evidence of screening, not proof of absence.

## Fatal-Overlap Tier

### Carrara et al., Budgeted Reinforcement Learning in Continuous State Space, NeurIPS 2019

Primary source: https://arxiv.org/abs/1903.01004

| Axis | Prior work | Current route |
| --- | --- | --- |
| Problem | optimize reward under an adjustable expected-cost budget | evaluate charger continuation under a collision-probability ledger and physical energy |
| State | augmented by budget | augmented by remaining collision budget `b` |
| Safety signal | cumulative cost | calibrated swept-collision hazard plus failure mass |
| Time scale | sequential discounted MDP | action hazard plus undiscounted charger SSP |
| Learned object | budgeted reward/cost Q and budgeted policy | defective energy-return law indexed by learned-filter version and `b` |
| Guarantee source | Budgeted Bellman results; deep approximation empirical | conditional SSP/risk/calibration implications |
| Energy semantics | generic cost | actual telemetry energy to a designated charger |
| Collision semantics | generic cost constraint | selected-action one-step hazard with stopped risk spending |
| Goal switching | not central | irreversible task-to-charger commitment |
| Data assumptions | batch interaction with unknown dynamics | repeated sorties, explicit failures/censoring/versioning |
| Difference | **Budget augmentation and budget-dependent policy are already covered.** Only the filter-induced defective target, support discontinuity, charger-specific estimand, and calibration protocol remain distinguishable. |

This paper kills any claim that `b`-augmentation or a budgeted Bellman operator is new.

### Kim et al., Trust Region-Based Safe Distributional RL for Multiple Constraints, NeurIPS 2023

Primary source: https://arxiv.org/abs/2301.10923

| Axis | Prior work | Current route |
| --- | --- | --- |
| Problem | risk-averse safe RL with multiple constraints | collision-risk ledger plus charger-return energy sufficiency |
| State | ordinary CMDP state | history/state plus residual risk budget |
| Safety signal | several cumulative cost distributions | local collision hazard and defective charger-energy law |
| Time scale | discounted episodic costs | atomic collision hazard and undiscounted goal hitting |
| Learned object | multiple distributional cost critics | success mass plus conditional energy distribution induced by filtered charger policy |
| Guarantee source | trust-region/gradient integration theory | conditional sequential composition and calibration |
| Energy semantics | energy can be one generic constraint | post-action telemetry energy to charger |
| Collision semantics | generic cumulative collision cost | swept selected-action hazard with explicit spend |
| Goal switching | absent | one-way commitment |
| Data assumptions | replay/TD(lambda) | versioned repeated sorties, censoring retained |
| Difference | **Multiple distributional critics are already covered.** The current route must beat an SDAC-style matched baseline and cannot claim multi-constraint distributional RL. |

### Jung et al., Quantile Constrained Reinforcement Learning, NeurIPS 2022

Primary source: https://proceedings.neurips.cc/paper_files/paper/2022/hash/2a07348a6a7b2c208ab5cb1ee0e78ab5-Abstract-Conference.html

| Axis | Prior work | Current route |
| --- | --- | --- |
| Problem | constrain outage probability of cumulative cost | constrain energy insufficiency after charger commitment |
| Learned object | cumulative-cost quantile/tail via distributional RL | budget-conditioned charger-return success mass and finite-energy quantiles |
| Guarantee source | quantile constraint and policy-gradient analysis | conformal selected-action coverage plus risk ledger |
| Difference | **Distributional cost plus threshold is already covered.** Charger hitting semantics, missing mass, endogenous filter version, and stopping context are the only differences. |

### Zheng and Jin, Prediction Sets for Counterfactual Decisions, 2026

Primary source: https://arxiv.org/abs/2607.02206

| Axis | Prior work | Current route |
| --- | --- | --- |
| Problem | predictions induce actions and therefore realized counterfactual outcome | bounds filter actions and thereby change charger-return data/target |
| Learned object | policy-coupled prediction sets | policy/filter-coupled defective return law |
| Guarantee source | policy-coupled conformal construction | must reuse or specialize such selection-valid machinery, not claim it |
| Difference | sequential SSP, collision budget, censoring, and repeated-sortie target drift are not the paper's focus; policy-coupled coverage itself is already claimed. |

This paper kills any novelty claim based only on “calibration after action selection.”

## Strong-Overlap Tier

### Zhang and Weng, Safe Distributional Reinforcement Learning, 2021

Primary source: https://arxiv.org/abs/2102.13446

General constrained distributional RL already supports probability/CVaR-style safety definitions. It does not supply the charger-specific budgeted defective law, but it covers the broad distributional-safe-RL framing.

### Fan et al., Safety-Polarized and Prioritized RL (MaxSafe), ICML 2025

Primary source: https://proceedings.mlr.press/v267/fan25i.html

Chance-constrained bi-level safety and learned action masks directly overlap collision-based action restriction. Current novelty cannot rest on learning a safe mask.

### Mao et al., Calibrated Prediction of Safety Chances, L4DC 2024

Primary source: https://proceedings.mlr.press/v242/mao24c.html

Calibrated learned safety chances from image observations overlap the collision side. Current work must treat this as an ingredient.

### Zhang et al., Conformal Off-Policy Prediction, AISTATS 2023

Primary source: https://proceedings.mlr.press/v206/zhang23c.html

Prediction intervals for target-policy return under policy shift overlap return calibration and show that off-policy target distributions already have conformal treatment. Current work still must solve selected sequential action/filter versioning and censoring, but cannot claim return conformalization generally.

### Castellano et al., RL with Almost-Sure Constraints, L4DC 2022

Primary source: https://proceedings.mlr.press/v168/castellano22a.html

State augmentation by a scalar safety budget and Bellman-like minimal-budget fixed points are established. This reinforces that remaining-budget state is not itself novel.

### Begzadic et al., Back to Base, 2025

Primary source: https://arxiv.org/abs/2501.02620

Reach-avoid safety filtering for autonomous return to a base/charging region is a close semantic competitor. It uses a reach-avoid value and safety filter rather than data-driven energy-return distributions, but it directly attacks “return to charger while avoiding unsafe states” as a novelty claim.

## Domain Context

- Persistent robot charging is explicitly formulated in https://arxiv.org/abs/2409.00572.
- Autonomous drone charging/mission planning with learned energy models predates this route: https://arxiv.org/abs/1703.10049.

These works prevent novelty claims based on persistence, charging, or learned UAV energy prediction alone.

## Closest-Work Conclusion

The general mathematical envelope is already occupied by BMDPs, multi-constraint distributional RL, quantile-constrained RL, and policy-coupled conformal prediction. No exact public match was found for the full tuple

`learned selected-action collision envelope -> induced shared charger-policy support -> defective budget-conditioned telemetry-energy law -> one-way persistent-sortie stopping`.

That remaining tuple is a narrow problem/learning-object distinction, not yet an established algorithmic breakthrough. It survives literature screening only as a falsifiable hypothesis.

## Redesign-2 Transport Attack

### General policy/occupancy perturbation

Policy-induced occupancy and return shifts are routinely bounded through total-variation or related divergences. A recent explicit example bounds occupancy shift by policy TV in Partial Action Replacement: https://ojs.aaai.org/index.php/AAAI/article/download/39402/43363. C4 is a standard maximal-coupling path-law argument in this family.

### Coverage under distribution shift

Coverage degradation under joint distribution shift is already bounded using TV/Wasserstein/other probability metrics. A direct recent source is https://arxiv.org/abs/2501.13430. Therefore C6's additive TV coverage debit is not a new conformal theorem.

### Filtered critic target consistency

Robust Koopman-CBF SAC explicitly trains the critic on the executed filtered action and computes targets through the filtered next action: https://arxiv.org/abs/2605.26452. This does not give C5's boundary-visitation law, but it kills any broad claim that recognizing filter-induced critic targets is new.

### Residual difference

C5 links a learned hard collision-admission boundary to charger-return path-law drift through boundary visitation. No exact public theorem was found in the screened sources. However it is a short specialization of generic coupling/perturbation machinery, so it cannot alone carry an ICLR-level theory claim.

## Final-Redesign Identification Attack

### Wei, Decision-Making Under Selective Labels, ICML 2021

Primary source: https://proceedings.mlr.press/v139/wei21a.html

This work explicitly studies missing outcomes caused by decisions and the online tradeoff between collecting labels and future utility. It directly covers the general phenomenon behind commitment-censored return labels and the supervision/throughput tradeoff.

### Huang et al., Off-Policy Risk Assessment for MDPs, AISTATS 2022

Primary source: https://proceedings.mlr.press/v151/huang22b.html

This work estimates target-policy return CDFs and risk functionals in MDPs, provides finite-sample bounds, doubly robust estimators, and minimax lower bounds. It is substantially stronger than I3--I4's fixed-stratum inverse-propensity calculation.

### Zhang et al., Conformal Off-Policy Prediction, AISTATS 2023

Primary source: https://proceedings.mlr.press/v206/zhang23c.html

This work constructs target-policy return prediction intervals with finite-sample/asymptotic coverage under logged policy data. It covers the general calibration-after-policy-selection task.

### Safe/active policy-evaluation data collection

SaVeR studies data-collection design for safe policy evaluation: https://openreview.net/pdf?id=reB9FFAaKw. Adaptive off-policy sampling and iterative active data collection also have an established literature. The proposed randomized commitment probe is therefore an application-specific logging policy, not a new learning principle.

## Final Closest-Work Decision

I1--I3 are correct causal/missing-data identification facts; I4 is a loose standard inverse-propensity concentration bound; I5 is the expected exploration/utility tradeoff. Existing selective-label and off-policy distribution-estimation work covers these mechanisms more generally and often more strongly.

The exact UAV tuple remains uncommon, but every theoretical ingredient and each proposed correction now has a direct general prior. This is fatal to an ICLR-level theory claim for the current formulation.
