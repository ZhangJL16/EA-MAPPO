# Conservative Hazard Fusion Derivation Package

## Target

Construct an action-conditioned LiDAR critic that can learn obstacle evidence
without increasing the false-safe acceptance set of the geometry critic, then
calibrate its accept/reject operating point under a scene-level risk budget.

## Status

COHERENT AFTER REFRAMING / EXTRA ASSUMPTION

The original goal of making an unconstrained fused probability uniformly more
accurate and safer is unsupported by the independent experiment.  The coherent
target is instead a one-sided learned hazard correction plus an explicit
risk--coverage calibration problem.

## Invariant Object

For pre-action state (x), candidate macro-action (a), and available energy
budget (b), the invariant object is the safe-rechargeability event

\[
Y(x,a,b)=\mathbf 1\{S=1,\ E_{\rm stop}\le b\},
\]

where (S) includes collision/boundary avoidance, required task completion,
and charger arrival before the deadline under the declared continuation.

## Assumptions

- A geometry/action critic supplies (q_g(x,a,b)\in(0,1)), non-decreasing in
  (b).
- LiDAR and action features (z=z(x,a)) are available before action execution;
  no outcome or future-action field is included.
- The learned hazard (h_\theta(z)\) is nonnegative and does not depend on the
  queried budget.  This budget independence is restrictive but preserves the
  geometry critic's ordering in (b).
- For conformal risk control, complete scenes are exchangeable calibration
  units.  All anchors, actions, and budget queries within a scene form one
  bounded loss and are not treated as independent observations.
- Reusing the already inspected 150-scene data supports method development
  only.  A later claim requires fresh scenes with the method and risk budget
  frozen in advance.

## Notation

- (q_g): calibrated geometry/action safe-rechargeability probability.
- (h_\theta\ge0): neural hazard evidence extracted from geometry, LiDAR, and
  the proposed action.
- (q_h): conservative fused critic.
- (t\): safe-declaration probability threshold.
- (Z_i): all paired branches from independent scene (i).
- (L_i(t)\): fraction of infeasible queries in scene (i) declared safe.
- (n): number of calibration scenes; (alpha): target scene-level risk.

## Derivation Strategy

Start from the geometry log-odds, subtract only nonnegative learned hazard,
prove probability dominance and budget monotonicity, convert dominance into an
accepted-set safety statement, and finally choose the operating threshold by a
scene-level conformal risk correction while measuring retained coverage.

## Derivation Map

1. Definition: safe rechargeability is a joint reach--avoid--resource event.
2. Architecture: subtract nonnegative hazard in log-odds space.
3. Proposition 1: probability and accepted-set dominance.
4. Proposition 2: preservation of budget monotonicity.
5. Definition: scene-clustered false-safe loss.
6. Imported theorem: conformal risk control for bounded monotone loss.
7. Decision criterion: compare coverage at a common risk budget, not raw
   confidence at an arbitrary threshold.

## Main Derivation

### Step 1: one-sided neural fusion — definition

Define

\[
q_h(x,a,b)
=\sigma\!\left(\operatorname{logit}q_g(x,a,b)-h_\theta(z(x,a))\right),
\qquad h_\theta(z)=\operatorname{softplus}(r_\theta(z))\ge0.
\]

This is a model definition, not a theorem.  The neural network learns where
LiDAR/action evidence indicates extra danger; it is structurally forbidden from
turning that evidence into greater optimism than the geometry parent.

### Step 2: probability and accepted-set dominance — proposition

Because sigmoid is strictly increasing and (h_\theta\ge0),

\[
q_h(x,a,b)\le q_g(x,a,b)
\]

for every input.  Therefore, for every fixed declaration threshold (t),

\[
\mathcal A_h(t)=\{(x,a,b):q_h\ge t\}
\subseteq
\mathcal A_g(t)=\{(x,a,b):q_g\ge t\}.
\]

Multiplying both acceptance indicators by the infeasibility indicator gives

\[
\mathbf1\{q_h\ge t,Y=0\}
\le
\mathbf1\{q_g\ge t,Y=0\}.
\]

Hence the joint dangerous false-safe probability and the false-safe rate
conditional on (Y=0) cannot exceed the geometry parent's values at the same
threshold, under any evaluation distribution.  This does **not** prove that
the fraction unsafe among accepted is lower, because the accepted denominator
also changes.

### Step 3: budget monotonicity — proposition

Since (q_g(x,a,b)) is non-decreasing in (b), its logit is non-decreasing.
The budget-independent quantity (h_\theta(z)) is a constant with respect to
(b).  Subtraction preserves order, and sigmoid is increasing, so

\[
b_1\le b_2\Longrightarrow
q_h(x,a,b_1)\le q_h(x,a,b_2).
\]

Thus the learned hazard cannot create an incoherent resource CDF.

### Step 4: scene-level monotone risk — definition

For scene (i), let (mathcal N_i=\{j:Y_{ij}=0\}).  Define

\[
L_i(t)=
\begin{cases}
|\mathcal N_i|^{-1}\sum_{j\in\mathcal N_i}
\mathbf1\{q_{ij}\ge t\},&|\mathcal N_i|>0,\\
0,&|\mathcal N_i|=0.
\end{cases}
\]

This is bounded in ([0,1]) and non-increasing in (t).  It treats one scene,
not one correlated branch or budget query, as the calibration observation.

### Step 5: conformal risk correction — imported theorem specialization

Conformal Risk Control considers a bounded monotone loss and selects a
calibration parameter so that expected loss on a new exchangeable observation
is controlled.  With (B=1), use the corrected empirical risk

\[
\widehat R^+_n(t)=
\frac{n}{n+1}\left(\frac1n\sum_{i=1}^nL_i(t)\right)
+\frac1{n+1}.
\]

Among thresholds satisfying (widehat R^+_n(t)\le\alpha), choose the smallest
threshold (largest coverage).  Under the theorem's exchangeability and
monotonicity conditions, the selected rule controls expected scene-level loss
up to the stated finite-sample correction.  This step specializes, rather than
re-proves, the result of Angelopoulos et al., *Conformal Risk Control*
(arXiv:2208.02814, revised 2025).

### Step 6: utility at controlled risk — decision criterion

Define test coverage

\[
C(t)=\Pr(q(X,A,B)\ge t).
\]

The method is useful only if, at a common target (alpha), its independently
calibrated threshold retains more coverage than the geometry critic while its
fresh-scene risk remains controlled.  This follows the risk--coverage framing
of selective classification; it prevents the vacuous solution (t>1), which
rejects every action and has zero observed false-safe loss.

## Remarks and Interpretation

- The architecture is a neural safety residual, not a distance-threshold rule.
  Distance and LiDAR may influence the magnitude of learned hazard, but the
  one-sided relation to geometry holds globally.
- The previous direct fusion raised safe-declaration coverage from 55.04% to
  62.27% but also raised conditional false-safe from 11.32% to 15.23%.  The new
  object explicitly evaluates whether learned discrimination can increase
  coverage at a fixed risk rather than by increasing optimism.
- Learn Then Test offers a more general finite-sample multiple-testing route
  when the chosen loss or parameter family is non-monotone.  The present Gate
  deliberately uses the smaller monotone special case.

## Boundaries and Non-Claims

- Dominance is relative to the geometry critic; it does not make an unsafe
  geometry parent absolutely safe.
- CRC controls an expectation over exchangeable scenes, not worst-case wind,
  arbitrary map shift, or every individual trajectory.
- Cross-fitted results on the already inspected 150 scenes are feasibility
  evidence only.  They cannot confirm the final paper claim.
- A passing critic Gate authorizes only a bounded actor pilot.  It does not
  prove closed-loop navigation safety or energy sustainability.

## Open Risks

- With only 30 calibration scenes per fold and (alpha=0.05), the finite-sample
  correction (1/(n+1)=0.0323) leaves little empirical-risk budget and may
  force near-total rejection.  If so, the honest remedy is more independent
  scenes, not weakening the bound after seeing results.
- A budget-independent hazard may be too restrictive if LiDAR evidence changes
  the shape rather than only the location of the energy-feasibility CDF.
- Exchangeability across procedurally generated scenes must be stress-tested
  under new task seeds and obstacle layouts before a strong claim.

## Primary References

- Angelopoulos, Bates, Fisch, Lei, and Schuster. *Conformal Risk Control*.
  arXiv:2208.02814.
- Angelopoulos, Bates, Candès, Jordan, and Lei. *Learn then Test: Calibrating
  Predictive Algorithms to Achieve Risk Control*. Annals of Applied Statistics,
  2025.
- Geifman and El-Yaniv. *Selective Classification for Deep Neural Networks*.
  NeurIPS 2017.
