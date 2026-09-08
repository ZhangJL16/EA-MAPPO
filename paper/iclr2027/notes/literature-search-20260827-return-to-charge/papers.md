# Literature Search: Reliable Resource-to-Go under Closed-Loop Composition Shift

Date: 2026-08-27  
Search purpose: Introduction, Related Work, novelty-risk, and baseline grounding.  
Target: ICLR 2027.  
Source-quality policy: official proceedings/publisher pages preferred; MDPI and snippet-only records excluded.

## Summary

- Closest practical work: Predictive Safety Network (future-resource prediction plus hierarchical safety policy); closest modeling work: policy-conditioned environment models (PCM) and compositional world models.
- Strongest uncertainty baselines: executed-action world model plus Deep Ensemble; conformal methods are conditional candidates, not automatic shift guarantees.
- Strongest decision alternatives: direct high-level switching and constrained/distributional RL (CPO, CVPO, SDAC).
- Central open conjunction: none of the verified works jointly treats a queryable policy--hard-safety composition, executed-interface support, long-horizon charger Resource-to-Go, selective reliability, and an irreversible return boundary.
- Novelty risk: PCM already makes target-policy occupancy central. The paper must claim the narrower executed-interface and decision-boundary contributions, not generic policy-conditioned generalization.

## Screened Paper Table

| Work | Type | I | C | N | Label | Role / caution |
| --- | --- | ---: | ---: | ---: | --- | --- |
| Choudhry et al. (ICRA 2021) | pure method | 4 | 4 | 4 | A | Real UAV trajectory-energy and CVaR anchor; pre-flight path risk, not closed-loop pair shift. |
| Guo & B{\"u}rger (CoRL 2019/Proceedings 2020) | pure method | 5 | 5 | 4 | Risk | Closest resource-prediction hierarchy; multi-agent shared-resource safety policy, without the executed-interface identification question. |
| Mathew et al. (T-RO 2015) | planning | 4 | 4 | 4 | A | Persistent-UAV recharge/rendezvous anchor; schedules charging trajectories rather than learning closed-loop Resource-to-Go. |
| Achiam et al. (ICML 2017) | theory + method | 5 | 5 | 4 | A | Canonical CMDP policy optimization; expected constraints do not replace hard filter plus return prediction. |
| Qin et al. (ICML 2021) | theory + method | 5 | 5 | 5 | A | Direct resource/charging CRL comparison via density constraints; mandatory positioning for the CMDP track. |
| Kim et al. (NeurIPS 2023) | pure method | 4 | 4 | 4 | A | Distributional multi-constraint baseline; end-to-end alternative. |
| Chen et al. (ICML 2024) | theory + method | 5 | 5 | 5 | Risk | Closest target-policy shift work; must distinguish nominal policy conditioning from executed interface. |
| Lakshminarayanan et al. (NeurIPS 2017) | pure method | 5 | 4 | 5 | A | Default epistemic baseline; no formal guarantee under this composition shift. |
| Ovadia et al. (NeurIPS 2019) | benchmark/analysis | 5 | 5 | N/A | A | Shows UQ must be evaluated under shift. |
| MOPO / MOReL / COMBO (NeurIPS 2020--21) | methods | 4 | 5 | 5 | A | Strong offline model-bias baselines; address dataset support, not queryable closed-loop composition per se. |
| Bellemare; IQN; FQF | methods/theory | 5 | 5 | 5 | B | Return-distribution foundations; current simulator does not justify physical tail claims. |
| SelectiveNet (ICML 2019) | pure method | 4 | 4 | 4 | B | Risk--coverage template and abstention baseline. |
| CQR; weighted conformal (NeurIPS 2019) | theory + method | 5 | 5 | 4 | A | Coverage tools with explicit exchangeability/shift assumptions; not automatic under policy--operator shift. |
| HOWM (ICML 2022) | theory + method | 4 | 4 | 4 | B | Compositional world-model anchor in object-oriented dynamics. |
| WM3C (ICLR 2025) | theory + method | 4 | 4 | 4 | B | Recent compositional environment-model comparison; different causal/language decomposition. |
| Xiao & Belta (TAC 2022) | theory + method | 5 | 5 | 4 | A | Hard collision-safety authority; should never be conflated with energy reliability. |
| Lavanakul et al. (L4DC 2024) | pure method | 4 | 4 | 4 | B | Supports explicit performance/safety separation and reusable filters. |

Scores are literature-screening judgments (1--5), not acceptance predictions. `N/A` denotes a benchmark/analysis paper.

## Closest-Work Clusters

### Energy sufficiency and return-to-base

Trajectory-level energy models quantify consumption and risk, but verified work does not analyze the executed-action distribution induced jointly by a policy and an intervention filter. The differentiation is a decision-time, closed-loop Resource-to-Go object rather than a better battery regressor.

Predictive Safety Network is the closest architectural predecessor: it predicts future resource and supplies a centralized safety policy above task policies. The paper must credit that hierarchy directly and limit novelty to queryable policy--operator composition, executed-interface support, and boundary-localized decision error. Persistent-UAV rendezvous planning supplies the complementary non-learning recharge baseline class.

### Constrained and distributional RL

CMDP methods optimize policies subject to expected or risk-sensitive constraints. They are necessary end-to-end baselines, but the modular track asks a different question: how to decide an irreversible return while collision safety remains delegated to an independently queryable hard operator.

### Policy-conditioned and compositional world models

PCM is the highest novelty risk because it adapts a dynamics model to target-policy occupancy. The present paper must demonstrate that nominal pair identity is the wrong unit once a safety operator rewrites actions, and evaluate matched executed-interface support at matched horizons.

### Model uncertainty and selective prediction

Ensembles and reject options are strong baselines. Conformal coverage survives only under its declared exchangeability or reweighting conditions; composition shift therefore motivates empirical risk--coverage evaluation and fail-closed abstention, not unconditional coverage language.

## Opportunity Map

| Cluster | Status | Open gap | Required evidence | Risk |
| --- | --- | --- | --- | --- |
| Energy prediction | crowded but open | closed-loop composition and irreversible return | Oracle headroom + paired cycles | Medium |
| Policy-conditioned models | theory-analysis gap | executed-interface rather than nominal-policy support | leave-one-pair-out, matched support/horizon | High |
| UQ/selective prediction | mechanism gap | whether epistemic scores forecast long-horizon underestimation | risk--coverage/AURC | High |
| CMDP/safe RL | comparison gap | modular prediction vs direct end-to-end switching | common-budget CMDP track | High |
| Return decision | benchmark gap | paired stranding--throughput protocol | 100 cycles/point, Wilson + paired bootstrap | Medium |

## Handoff to Writing

- State the PCM difference explicitly in Introduction and Related Work.
- Cite distributional RL only as an extension/baseline because the present evidence supports deterministic point Resource-to-Go plus epistemic reliability.
- Do not claim conformal validity under deployment shift without estimating/declaring the required weights.
- Present SIRP only as a gated candidate after executed-WM + ensemble.
- Treat Predictive Safety Network as closest practical prior work and DCRL as a direct charging/resource constrained-control alternative.
