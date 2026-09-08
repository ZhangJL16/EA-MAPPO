# Idea Grounding: Executed-Interface Risk Resolvent

## Problem

An agent must decide when to irreversibly return to a charger. Its nominal policy
does not directly control the plant: a safety operator modifies the action, and
the resulting closed loop determines both the charger hitting time and cumulative
energy. At deployment, a new policy--filter pair may be seen without labelled
return trajectories.

Current predictors are misaligned in three ways:

1. one-step maximum likelihood weights errors by source frequency, not by their
   effect on tail return energy;
2. mean TD and sparse quantile TD do not provide a stable undiscounted
   first-passage tail theorem under the present assumptions;
3. uncertainty or conformal wrappers cannot repair target regions for which the
   tail-relevant executed interface is unidentified.

## Mathematical object

For charger set \(G\), hitting time \(T_G\), and nonnegative cost \(c_t\), define

\[
Z_G=\sum_{t<T_G}c_t,
\qquad
\psi_\lambda(x)=\mathbb E_x[e^{\lambda Z_G}].
\]

For the killed closed-loop transition, define

\[
(M_\lambda f)(x)
=\mathbb E_x[e^{\lambda c_0}f(X_1)\mathbf 1\{X_1\notin G\}],
\]

and

\[
r_\lambda(x)
=\mathbb E_x[e^{\lambda c_0}\mathbf 1\{X_1\in G\}].
\]

When \(\rho(M_\lambda)<1\),

\[
\psi_\lambda=(I-M_\lambda)^{-1}r_\lambda.
\]

This identity is a known foundation. The intended new objects are the
non-normalized risk-tilted state occupation measure and its executed-interface
disintegration

\[
\eta^X_{\nu,\lambda}=\nu(I-M_\lambda)^{-1},
\]

\[
\bar\eta_{\nu,\lambda}(dx,da,du,dx',dc)
=\eta^X_{\nu,\lambda}(dx)\pi(da\mid x)
K_\Pi(du\mid x,a)K(dx',dc\mid x,u).
\]

## Central theorem target

For a learned operator \((\widehat M_\lambda,\widehat r_\lambda)\), prove the
exact identity

\[
\psi_\lambda-\widehat\psi_\lambda
=(I-M_\lambda)^{-1}
\left[(r_\lambda-\widehat r_\lambda)
+(M_\lambda-\widehat M_\lambda)\widehat\psi_\lambda\right].
\]

After applying an initial distribution \(\nu\), the target log-MGF error is
controlled by a local Bellman/operator residual integrated against
\(\bar\eta_{\nu,\lambda}\), not ordinary source occupancy. If
\(\bar\eta_{\nu,\lambda}\ll\bar\rho_s\), define

\[
\bar w_\lambda
=\frac{d\bar\eta_{\nu,\lambda}}{d\bar\rho_s}.
\]

Then an \(L_2\) bound has the form

\[
|\nu(\psi_\lambda-\widehat\psi_\lambda)|
\le
\|\bar w_\lambda\|_{L_2(\bar\rho_s)}
\|\epsilon_{\lambda,\widehat\psi}\|_{L_2(\bar\rho_s)}.
\]

The key research claim to prove is that ordinary overlap can be insufficient
even when this risk-tilted coefficient diverges, and that without domination on
a positive risk-tilted set the target tail is not point-identifiable.

## Algorithmic consequences

The theory would prescribe:

- a positive, multi-\(\lambda\) critic \(\psi_\theta(x,\lambda)\) with terminal
  value one;
- exponential Bellman targets
  \[
  Y_\lambda=e^{\lambda c}
  [\mathbf 1_{X'\notin G}\bar\psi(X',\lambda)
  +\mathbf 1_{X'\in G}];
  \]
- a separately estimated risk-tilted executed-interface density ratio satisfying
  the adjoint moment
  equation
  \[
  \mathbb E_{\bar\rho_s}\left[
  \bar w_\lambda(X,A,U)
  \{f(X)-e^{\lambda C}\mathbf 1_{X'\notin G}f(X')\}
  \right]=\mathbb E_\nu[f(X)];
  \]
- cross-fitted weighting of a positive-value loss by \(\bar w_\lambda\), with
  Itakura--Saito used as an existing baseline rather than claimed as new;
- data acquisition targeted at large risk-tilted ratio and return-boundary mass,
  rather than at generic model disagreement.

## Decision certificate

For a finite grid \(\Lambda\), define

\[
U_\delta(x)=min_{\lambda\in\Lambda}
\frac{\log\psi_\lambda(x)+\log(1/\delta)}{\lambda}.
\]

Chernoff's inequality gives

\[
\Pr_x(Z_G\ge U_\delta(x))\le\delta.
\]

If a simultaneous learned upper bound on \(\log\psi_\lambda\) fails with
probability at most \(\alpha\), the deployed bound becomes \(\delta+\alpha\).
This certificate is a corollary, not the novelty by itself.

## Falsification tests

The idea should be abandoned or substantially revised if any of these occurs:

1. existing work already proves the same target/source Feynman--Kac density-ratio
   theorem under composition shift;
2. the risk-tilted ratio collapses to an ordinary occupancy ratio in all relevant
   resource models and supplies no sharper failure prediction;
3. the required exponential-transience assumption excludes the environments in
   which return failure matters;
4. the upper-MGF estimator is too statistically unstable to outperform direct MC
   or distributional baselines at fixed data;
5. oracle experiments show negligible decision headroom over SOC/distance
   heuristics.

## Next proof obligations

1. State the killed operator on a general measurable executed-interface space and
   prove bounded invertibility under an explicit exponential drift condition.
2. Prove the resolvent identity and its weighted-norm corollaries.
3. Give a two-kernel Le Cam construction showing non-identifiability without
   risk-tilted domination.
4. Derive finite-sample error with estimated \(\bar w_\lambda\), cross-fitting,
   and
   function approximation.
5. Prove a sequential boundary-margin result by coupling oracle and learned paths
   until their first stopping disagreement.
6. Establish an exact task--return composition theorem without independence.
