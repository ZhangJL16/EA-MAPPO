# Defective Continuous Resource-CDF: Derivation Package

## Target

Derive an identifiable neural representation of the stopped, extended-real
resource-to-safe-recharge variable from the existing R3 rollout dataset.  The
model must retain collision/deadline failure as mass at \(+\infty\), predict
finite energy continuously, and return a monotone probability for every battery
budget used by the controller.

## Status

**COHERENT AFTER REFRAMING / EXTRA ASSUMPTION.**

The defective-CDF object is coherent.  The earlier proposal to learn a
state-dependent log-normal scale is not identifiable from this dataset because
each physical task state has only one realized finite energy.  The corrected
model learns failure mass and conditional log-energy location, while treating
the distribution scale as a global smoothing/calibration parameter.  This is a
working predictive family, not a claim that physical energy is truly
log-normal.

## Invariant Object

For state/deadline context \(z=(s,h)\), retain the existing stopped variable

\[
Y(z)=
\begin{cases}
E(z), & \text{safe recharge succeeds by }h,\\
+\infty, & \text{collision, boundary violation, or deadline failure}.
\end{cases}
\]

The invariant object is its finite-threshold CDF

\[
F(b\mid z)=\Pr\{Y(z)\le b\},\qquad b<+\infty.
\]

## Assumptions

- \(E(z)>0\) on every finite example.  The observed dataset satisfies this:
  the minimum finite energy fraction is 0.0190.
- Failure labels and finite energy targets are generated under the frozen R3
  policy and therefore describe that policy only.
- Task seeds, not budget or horizon queries, are the independent evaluation
  units.
- Conditional log-energy is represented by a location family with one global
  positive scale.  This is an explicit approximation.
- Calibration and test task seeds are disjoint from gradient-training and
  early-stopping task seeds.

## Notation

- \(q_\theta(z)\in(0,1)\): predicted mass at \(+\infty\).
- \(\mu_\theta(z)\in\mathbb R\): predicted conditional location of
  \(\log E\) when the return succeeds.
- \(\sigma>0\): global smoothing scale.
- \(\Phi\): standard-normal CDF.
- \(d=\mathbf 1\{Y=+\infty\}\): failure indicator.
- \(b_k\): queried battery thresholds.

## Derivation Strategy

Decompose the extended-real law into a Bernoulli failure mass and a conditional
finite law.  Use this exact mixture identity to obtain the CDF, then introduce a
log-location approximation only for the conditional finite component.  Train
the identifiable quantities directly and calibrate the global smoothing scale
on held-out tasks.

## Derivation Map

1. Extended-real outcome -> exact failure/finite mixture.
2. Exact mixture -> defective finite-threshold CDF.
3. Positive finite energy -> log-location approximation.
4. Approximate CDF -> monotonicity and failure-mass proposition.
5. Single outcome per state -> identifiability restriction on scale.
6. Task-weighted surrogate losses -> executable estimator.
7. Separate calibration split -> three-parameter predictive calibration.

## Main Derivation

### Step 1: Exact mixture identity

Let \(q(z)=\Pr(Y=+\infty\mid z)\).  Conditional on a finite outcome, let
\(G(b\mid z)=\Pr(E\le b\mid Y<+\infty,z)\).  For finite \(b\), the law of total
probability gives the exact identity

\[
F(b\mid z)=(1-q(z))G(b\mid z). \tag{1}
\]

No independence assumption is used: \(G\) is already conditional on success.

### Step 2: Log-location approximation

Because \(E>0\), define \(L=\log E\).  Approximate its conditional CDF by

\[
\Pr(L\le \ell\mid Y<+\infty,z)
\approx \Phi\!\left(\frac{\ell-\mu_\theta(z)}{\sigma}\right),
\qquad \sigma>0. \tag{2}
\]

Substitution of \(\ell=\log b\) into (1) yields

\[
\widehat F_\theta(b\mid z)
=(1-q_\theta(z))
\Phi\!\left(\frac{\log b-\mu_\theta(z)}{\sigma}\right),
\qquad b>0. \tag{3}
\]

Equation (1) is an identity.  Equations (2)--(3) are an approximation family.

### Step 3: Valid defective-CDF property

**Proposition 1.** For fixed \(z\), if \(q_\theta(z)\in[0,1]\) and
\(\sigma>0\), Equation (3), extended by zero for \(b\le0\), is nondecreasing and
right-continuous, and

\[
\lim_{b\to\infty}\widehat F_\theta(b\mid z)=1-q_\theta(z).
\]

**Derivation.** The normal CDF is nondecreasing and continuous; multiplication
by the nonnegative constant \(1-q_\theta(z)\) preserves both properties.  As
\(b\to\infty\), its standardized log argument tends to \(+\infty\), so
\(\Phi\to1\).  The missing mass \(q_\theta(z)\) is therefore exactly assigned to
\(+\infty\).

### Step 4: Identifiability restriction

For a finite training pair \((z_i,E_i)\), a free state-dependent density scale
can shrink toward zero while \(\mu_\theta(z_i)\to\log E_i\).  With one outcome
per exact state, no repeated conditional sample distinguishes genuine
aleatoric spread from interpolation error.  Thus a state-dependent
\(\sigma_\theta(z)\) has no supported physical interpretation here.

The executable model instead learns a single bounded positive \(\sigma\), used
as a predictive smoothing parameter.  A separate calibration set may transform

\[
q^c(z)=\operatorname{sigmoid}
\left(\frac{\operatorname{logit}q_\theta(z)}{T_q}\right),\quad
\mu^c(z)=\mu_\theta(z)+\Delta,\quad
\sigma^c=M_\sigma\sigma, \tag{4}
\]

with \(T_q,M_\sigma>0\).  These three global parameters are chosen without
access to test tasks.

### Step 5: Training surrogate

For task-balanced weight \(w_i\), failure indicator \(d_i\), and finite target
\(e_i\), use

\[
\mathcal L_{\mathrm{fail}}
=\frac{\sum_i w_i\operatorname{BCE}(q_i,d_i)}{\sum_i w_i},
\]

\[
\mathcal L_{\mathrm{energy}}
=\frac{\sum_{i:d_i=0} w_i
\operatorname{Huber}(\mu_i,\log e_i)}
{\sum_{i:d_i=0}w_i},
\]

and, for exact query labels
\(r_{ik}=\mathbf 1\{Y_i\le b_k\}\),

\[
\mathcal L_{\mathrm{CDF}}
=\frac{\sum_iw_i K^{-1}\sum_{k=1}^{K}
(\widehat F_\theta(b_k\mid z_i)-r_{ik})^2}
{\sum_iw_i}. \tag{5}
\]

The registered objective is

\[
\mathcal L
=\mathcal L_{\mathrm{fail}}
+\mathcal L_{\mathrm{energy}}
+2\mathcal L_{\mathrm{CDF}}. \tag{6}
\]

This is a calibrated predictive surrogate.  It is not claimed to be maximum
likelihood for the true physical process.

## Remarks and Interpretation

- Equation (3) uses three scalar degrees of freedom at evaluation time instead
  of 73 categorical logits.  It matches the empirical target structure: 268 of
  300 initial tasks have exactly one finite energy value across their feasible
  horizons, while 32 never become finite.
- The fusion network receives the original task/charger observations and LiDAR
  embedding plus a direct low-dimensional observation skip.  The skip is an
  architectural inductive bias, not a return threshold or decision rule.
- The learned CDF remains budget monotone by construction.

## Boundaries and Non-Claims

- A successful held-out Gate would establish predictive usefulness for the
  frozen R3 data distribution, not closed-loop navigation safety.
- Normality of conditional log-energy is not asserted.
- The global scale is calibration bandwidth, not identified physical noise.
- No theorem here establishes out-of-distribution or multi-cycle coverage.
- The construction is not claimed as oral-level novelty by itself.

## Open Risks

- Three hundred tasks may still be insufficient to learn obstacle-dependent
  residual energy beyond Euclidean geometry.
- Oracle labels come from a deterministic frozen-policy rollout and may contain
  systematic model bias.
- If the fusion model cannot beat geometry, the next blocker is conditional
  information/data diversity, not output parameterization or training length.
