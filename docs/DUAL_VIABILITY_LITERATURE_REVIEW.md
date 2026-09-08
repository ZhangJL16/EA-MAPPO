# Dual-Viability and Energy Option Preservation: Focused Literature Review

Date: 2026-09-05  
Scope: primary sources relevant to the post-pilot semantic correction

## Finding

The corrected direction is well grounded, but its broad ingredients are prior
art. The publishable question is not whether a task policy should have a
recovery policy. It is whether a learned, battery-conditioned controller can
estimate and preserve a recovery option under its actual execution interface,
with an honest trajectory-level risk statement and less task abandonment than
existing switching or reach-avoid filters.

## Closest theory and algorithms

1. **Recovery RL** separates task and recovery policies and learns a safety
   critic for future constraint violations. It reports strong empirical
   results but explicitly leaves formal guarantees to future work. This
   supports the modular architecture, not a certificate claim.
   [Thananjeyan et al., RA-L 2021](https://arxiv.org/abs/2010.15920)
2. **Leave No Trace** learns a reset policy and uses its value to detect entry
   into non-reversible states. This is an early direct precedent for
   option-preservation semantics.
   [Eysenbach et al., ICLR 2018](https://arxiv.org/abs/1711.06782)
3. **Advantage-Based Intervention (SAILR)** formalizes intervention using a
   baseline/backup policy and a cost action-value. It shows that the object
   being thresholded must be the value of the action under the intervention
   semantics, rather than an unrelated downstream task event.
   [Wagener et al., ICML 2021](https://proceedings.mlr.press/v139/wagener21a.html)
4. **Reach-Avoid RL** derives a discounted contraction for reach-avoid value
   learning and retains guarantees by treating the neural approximation as an
   untrusted oracle inside supervisory control. This supports a reach-avoid
   target and warns against equating a neural score with a certificate.
   [Hsu et al., RSS 2021](https://doi.org/10.15607/RSS.2021.XVII.077)
5. **RCRL** defines persistent safety through the feasible set and its
   self-consistent worst-over-time value, rather than an expected cumulative
   constraint. It also notes that a remaining resource budget can be included
   in the state constraint.
   [Yu et al., ICML 2022](https://proceedings.mlr.press/v162/yu22d.html)
6. **Back to Base** is the nearest conceptual paper. It constructs a
   time-varying reach-avoid value, proves a viscosity-based CBF result, and
   filters a nominal controller so the system avoids failure and reaches a
   reset/charging target within a deadline. Consequently, safe return to a
   charger plus minimal nominal-action modification is not itself new.
   [Begzadic et al., L4DC 2025](https://proceedings.mlr.press/v283/begzadic25a.html)
7. **Consumption MDPs** explicitly model resource exhaustion and reload states
   and synthesize strategies for reachability or repeated reachability without
   exhausting the resource. Battery augmentation and recharge cycles are
   therefore established formal-methods objects.
   [Blahoudek et al., ATVA 2020](https://link.springer.com/chapter/10.1007/978-3-030-53291-8_22)
8. **Conformal Risk Control** controls expected monotone loss for a new
   exchangeable example. Its theorem does not automatically cover an
   adaptively visited sequence of states produced by repeatedly applying the
   selected gate.
   [Angelopoulos et al., ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/file/f3549ef9b5ff520a7e41ff3cc306ab2b-Paper-Conference.pdf)
9. **Confidence sequences** provide time-uniform inference for a stochastic
   process, but this does not turn a pointwise classifier into a closed-loop
   viability certificate. A conditional supermartingale/e-process or a
   whole-trajectory calibration unit would still have to be constructed.
   [Howard et al., Annals of Statistics 2021](https://doi.org/10.1214/20-AOS1991)

## Consequence for this project

The previous label

\[
Y_{\mathrm{complete}}(x,a)
=\mathbf1\{a;\pi_{task};\kappa\text{ completes task and returns}\}
\]

cannot authorize the fallback implication

\[
Y_{\mathrm{complete}}=0\Longrightarrow
\kappa\text{ returns safely from }x.
\]

The correct pair of objects is

\[
V_R^\kappa(x)=\Pr_x^\kappa(\tau_C<\tau_U),
\qquad
Q_{\mathrm{op}}^\kappa(x,a)
=\mathbb E[\mathbf1_{\{X_1\notin U\}}V_R^\kappa(X_1)].
\]

For energy safety, \(x\) includes remaining budget and deadline; equivalently,
constraint violation includes negative budget or missing the charger deadline.
The first value determines whether fallback is meaningful now. The second
determines whether a proposed next action preserves that fallback option.

## Novelty boundary

A credible new paper would need more than this relabeling. Candidate residual
contributions are:

- paired counterfactual identification of three often-confounded viability
  objects under the exact executed-action interface;
- a budget-monotone neural option-preservation value amortized over all battery
  budgets, including collision/deadline failure as an atom at infinity;
- a finite-horizon recursive-risk theorem and calibration protocol whose unit
  is the complete adaptive stopping trajectory; and
- empirical evidence that this avoids both exhaustion and the severe task
  abandonment observed with completion-feasibility gating.

These remain hypotheses. The diagnostic protocol must first establish that the
corrected labels contain enough nondegenerate information to learn.
