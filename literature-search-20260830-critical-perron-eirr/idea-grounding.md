# Idea-Grounding Packet: Critical Perron EIRR

## Scope and Evidence Boundary

- Target functional: stopped exponential resource-to-charger risk under a finite
  killed quotient operator and source conditional-primitive observations.
- Established in-project: exact canonical gradient, critical Perron variance
  limits, and a projected-noise-degenerate phase under explicit assumptions.
- Externally established: spectral conditioning, killed eigenfunction
  perturbation, Feynman--Kac particle variance/CLTs, QSD estimation CLTs, SSP
  hardness, and generic margin-rate theory.
- Unknown: exact prior art for their quotient-aware semiparametric composition.

## Evidence Cards

| Source cluster | Supported observation | Collision with project | Transfer condition | Confidence |
| --- | --- | --- | --- | --- |
| Meyer; Ipsen--Meyer | Nearly transient chains are spectrally ill-conditioned | Perron-gap sensitivity is not novel | Finite nonnegative matrix / stochastic-chain relation | direct |
| Rudolf--Wang | Killed principal eigenfunctions and QSDs admit perturbation bounds | Killed-spectrum perturbation is not novel | Their self-adjoint generator setting differs from the finite non-symmetric operator | direct |
| Whiteley et al.; Bérard et al.; Chan et al. | Feynman--Kac particle errors and joint limits have mature theory | Generic FK variance and joint-limit language are occupied | Particle-system law differs from iid interface primitives | direct |
| Blanchet--Glynn--Zheng | Principal-eigenvector stochastic approximation has CLTs and eigenvalue-dependent rate failures | Closest eigenmode statistical collision | Estimates QSD, not the stopped-MGF conditional-law functional | direct |
| Chen et al.; Tarbouriech et al. | Goal-reaching sample complexity depends on SSP cost/hitting scales | Random horizon alone is not novel | Risk-neutral planning/learning differs from fixed-policy risk OPE | direct |
| Audibert--Tsybakov | Margin assumptions produce fast plug-in decision rates and minimax bounds | Generic stopped-margin conversion is not novel | Need map premature return/stranding loss to a calibrated excess loss | direct |
| Greenbaum--Li--Overton; Meyer--Stewart | Simple nonnormal eigenvector derivatives are controlled by eigenprojector/group-inverse machinery | Effective Perron separation is a conditioning coordinate, not a new perturbation result | Must be composed with the critical allocation functional | direct |
| Carpentier--Munos--Antos; Carpentier--Munos | Unknown-variance adaptive stratification has oracle regret and minimax lower theory | Generic pilot allocation and its difficulty are occupied | Need the near-critical singularity cancellation and failure phase | direct |
| Maurer--Pontil | Empirical variance yields bounded, data-dependent confidence bounds | The moment-estimation term is a standard ingredient | Novelty can only lie in its coupling to spectral/noise/quotient phases | direct |
| Huddleston--Claypool--Hocking; Khan--Khan--Ahsan | One allocation can optimize several estimands through convex/nonlinear programming | Generic shared multi-query allocation is occupied | Objective must be induced by the stopped margin and critical modes | direct |
| Kallus--Saito--Uehara; Hanna et al. | Multiple-logger efficiency and behavior-policy variance search are established in OPE | Generic optimal replay/data collection for OPE is occupied | Need stopped first-passage risk and a new phase/lower result | direct |

## Surviving Research Question

Given a near-critical killed quotient operator, can normalized critical-mode
allocation be learned with a pilot below the \(m\Delta^2\) scale required to
estimate stopped log-EIRR itself, and exactly which spectral/noise/quotient phase
changes destroy that separation?

This question is stronger than adding a loss term. Its oracle solution should
identify a reusable neural/RL design primitive:

\[
\text{priority}(q)
\propto
\frac{\ell(x)L(q\mid x)}{\sqrt{c(q)}}
\sqrt{\operatorname{Var}\!left(
e^{\lambda C}\mathbf 1\{X'\notin G_C\}z(X')\mid q
\right)}.
\]

The formula is a critical-mode version of classical Neyman allocation, and
generic adaptive attainment is also prior art. The paper-level burden is now the
singularity cancellation and matching failure regimes, followed by a stopped-
decision consequence—not the closed form or plug-in consistency alone.

## Minimum Theorem Stack

1. **Oracle design theorem:** exact cost-constrained minimizer and leading
   variance constant, including zero-noise and support edge cases.
2. **Critical-scale separation theorem:** prove that pilot allocation consistency
   needs \(m\to\infty\) but not \(m\Delta^2\to\infty\) under a fixed secondary
   eigengap and projected-variance floor.
3. **Failure-phase theorem:** specify the relation among pilot size,
   \(\Delta_n\), eigengap separation, quotient recovery, and moment estimation
   needed for the oracle approximation when those floors collapse.
4. **Stopped decision theorem:** under a declared margin exponent, convert the
   information gain into premature-return/stranding excess loss; acknowledge
   classical plug-in margin theory.
5. **Algorithmic corollary:** a replay sampler or data-collection policy whose
   priority score estimates the displayed critical-mode quantity.

Theorem 23 closes the known-quotient portion of item 3 with the sufficient
coordinate

\[
\Omega=O\!\left(
\frac{(1+\chi)(m^{-1/2}+\tau_M)}{\mathfrak g}
+\sqrt{\varkappa/m}+b^2/m+\tau_\sigma+\Delta/\sigma_*^2
\right),
\]

and a locally matching spectral/anisotropy lower slice. It also proves that
small projected variance alone has no universal exponent. Learned quotient
rates and the stopped-decision conversion remain open.

## Falsification Criteria

- Drop the route if the adaptive allocation cannot beat uniform/occupancy replay
  by more than a constant already implied by ordinary Neyman allocation.
- Drop any oral claim if quotient classes are chosen using outcome residuals or
  if the uniform critical assumptions require knowing the true Perron vectors.
- Do not claim a Pareto result until the original navigation Gate can measure
  stranding and throughput on the same frozen evaluation set.
