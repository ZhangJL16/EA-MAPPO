# Recoverable Safety Theory Audit

## Target

Determine whether the pointwise margin `rho` can be extended into a novel, online-computable recoverability condition for a bounded-input double-integrator UAV under sample-and-hold control, moving obstacles, and intermittent perception.

## Status

**NOT YET COHERENT AS A NOVEL THEORY.** The candidate object is mathematically coherent after reframing as a robust predecessor, but that object and its recursive-feasibility role are established concepts in viability, predictive CBF, backup CBF, and discrete-time CBF work.

## Invariant Object

The invariant object is not the instantaneous scalar `rho`. It is a set of state/information pairs from which the controller can preserve both interval safety and non-emptiness of the successor safe-action set for every admissible uncertainty realization.

## Assumptions

- UAV state `x=(p,v)` follows the repository's bounded-acceleration, bounded-velocity sample-and-hold dynamics.
- `U` is the exact cylindrical acceleration set.
- The current obstacle information state `O` is a conservative set that contains the true obstacle state.
- `W(x,O)` contains every admissible obstacle acceleration/jerk, sensing delay, and dropout evolution used in a hard guarantee.
- `F_h(x,u,w)` is the exact or conservative one-sample successor map.
- The current sampled-data HOCBF rows are valid over the hold interval whenever their stated uncertainty assumptions hold.
- Learned predictions may rank candidates but may not shrink `O` or `W` in a hard certificate.

## Notation

- `U_sd(x,O)`: current sampled-data HOCBF action set.
- `rho(x,O)`: maximum minimum slack over `U`.
- `S_rho = {(x,O): rho(x,O) >= 0}`: pointwise-feasible state/information set.
- `Pre_W(K)`: robust controlled predecessor of set `K`.
- `Psi_h(O,w)`: obstacle-information successor.

## Derivation Strategy

The derivation separates four categories: the implemented margin identity, a one-step predecessor definition, a conditional one-step implication, and the unsupported recursive claim.

## Derivation Map

1. Pointwise feasibility follows exactly from maximization of minimum row slack.
2. Future feasibility requires propagating both UAV state and obstacle-information state.
3. A robust one-step condition is the predecessor of `S_rho` intersected with current interval-safe actions.
4. One-step preservation follows directly if the uncertainty set contains the realized successor.
5. Recursive feasibility requires invariance of the new set itself, not merely one-step membership in `S_rho`.

## Main Derivation

### Step 1 — Identity: current feasibility

For rows `A_i u >= b_i`,

\[
\rho(x,O)=\max_{u\in U}\min_i(A_i u-b_i).
\]

If `rho >= 0`, the maximizing action satisfies every row. Conversely, if any action satisfies every row, its minimum slack is nonnegative, so the maximum is nonnegative.

### Step 2 — Definition: robust predecessor

For a set `K` of state/information pairs,

\[
\operatorname{Pre}_{W}(K)=
\left\{(x,O):\exists u\in U_{sd}(x,O),\ \forall w\in W(x,O),
(F_h(x,u,w),\Psi_h(O,w))\in K\right\}.
\]

The one-step recoverable inner set is

\[
R_1=S_\rho\cap\operatorname{Pre}_W(S_\rho).
\]

This is a definition, not a new theorem.

### Step 3 — Definition: future-feasibility margin

\[
r^+(x,O,u)=
\min_{w\in W(x,O)}
\rho(F_h(x,u,w),\Psi_h(O,w)),
\]

and

\[
R(x,O)=\max_{u\in U_{sd}(x,O)}r^+(x,O,u).
\]

Then `R(x,O) >= 0` is an optimization form of membership in the robust predecessor, subject to attainment and well-defined successor sets.

### Step 4 — Conditional proposition: one-step preservation

If an action `u` satisfies the current sampled-data rows and `r^+(x,O,u) >= eta >= 0`, and if the realized uncertainty `w_real` belongs to `W(x,O)`, then

\[
\rho(F_h(x,u,w_{real}),\Psi_h(O,w_{real}))\ge\eta\ge0.
\]

This follows from the definition of the minimum. It proves one-step pointwise-feasibility preservation only.

### Step 5 — What recursive feasibility would require

One-step preservation into `S_rho` is insufficient for recursion. A recursively feasible set must satisfy

\[
R_\infty\subseteq S_\rho\cap\operatorname{Pre}_W(R_\infty).
\]

Finite iterations

\[
R_{k+1}=S_\rho\cap\operatorname{Pre}_W(R_k)
\]

are standard viability-kernel inner approximations. Computing or approximating them does not become novel merely by naming the scalar optimizer a recoverability margin.

## Remarks and Interpretation

- The 356/451 sampled alternative actions are empirical evidence that current action choice matters.
- They are not robust certificates: the search uses the realized next obstacle state and a finite action grid.
- The fact that raw rows remain feasible while strengthened rows fail shows a conservatism/recursive-feasibility tension in the current sampled-data construction.
- Bounded uncertainty propagation is meaningful only if the set actually contains the true obstacle state throughout dropout.

## Boundaries and Non-Claims

- No new recursive-feasibility theorem is claimed.
- No maximum recoverable set is computed.
- No arbitrary-duration dropout guarantee is possible without a bounded reachable set that remains inside the modeled workspace.
- No collision guarantee follows from the current diagnostic dataset.
- No learned predictor carries certificate authority.
- The sampled action-grid oracle is not deployable at 20 Hz.

## Open Risks

- A conservative robust predecessor may be empty or induce deadlock.
- Multi-obstacle uncertainty growth can make online optimization intractable.
- Prior work already supplies stronger horizon and backup-policy mechanisms.
- A new method would require a theorem/property and a runtime advantage not reducible to those methods.

