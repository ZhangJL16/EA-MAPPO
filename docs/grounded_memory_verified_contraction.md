# Verified Contraction Derivation Package

## Target

Derive the strongest honest connection from grounded history to a smaller certified obstacle-state set, and determine whether a learned structured-memory proposal yields a theorem or control consequence beyond standard set-membership estimation.

## Status

**COHERENT AFTER REFRAMING / EXTRA ASSUMPTION.**

The set-intersection result is coherent only when measurement bounds, motion bounds, and association validity are independently trusted. Neural proposal scores do not establish those premises. The result is a supporting safety lemma, not a new learning theorem.

## Invariant Object

The invariant object is the feasible obstacle-state set

\[
\mathcal C_t(S)=\{x_t=(p_t,v_t,a_t):x_t\text{ satisfies the trusted physical bounds and all constraints in }S\}.
\]

Prediction error and empirical residual width are not substituted for this set.

## Assumptions

- A1. The current measurement error is componentwise bounded by `epsilon_p`.
- A2. Obstacle velocity, acceleration, and jerk are bounded by `V`, `A`, and `J` over the retained history window.
- A3. Every accepted historical measurement belongs to the same physical track.
- A4. Every accepted measurement's declared error bound is valid.
- A5. The analytic feasibility solver is exact up to its declared numerical tolerance.
- A6. The robust HOCBF consumes only the resulting certified set and fixed physical limits.

A3-A4 are independent trust assumptions. A same-track flag supplied by a learned proposer is not evidence of A3.

## Notation

- `tau_i`: age of historical measurement `i`.
- `y_i`: measured obstacle position at time `t-tau_i`.
- `epsilon_i`: trusted componentwise sensor-error bound.
- `S_P`: constraints proposed by a learned module.
- `V(S_P)`: constraints accepted by the analytic verifier.
- `C_base,t`: current-only physically reachable set.
- `C_ground,t`: intersection of `C_base,t` and accepted historical constraints.

## Derivation Strategy

Use bounded Taylor propagation to convert each trusted historical measurement into linear inequalities over current `(p_t,v_t,a_t)`. Intersect only verifier-accepted inequalities. Separate exact containment and monotonicity from any learned-selection performance claim.

## Derivation Map

1. A1-A2 yield one certified feasible strip per historical measurement.
2. A3-A4 determine whether the strip refers to the target obstacle and contains its true state.
3. Intersecting valid strips preserves truth containment and contracts the base set.
4. The learned proposer affects which strips are checked, not their mathematical validity.
5. Any contraction-rate statement requires an additional informativeness assumption; it does not follow from proposal recall alone.

## Main Derivation

### Step 1: historical measurement strip

For one Cartesian axis and history age `tau_i`, bounded Taylor propagation gives

\[
p_{t-\tau_i}=p_t-\tau_i v_t+\tfrac12\tau_i^2 a_t+r_i,
\qquad |r_i|\le \tfrac16J\tau_i^3.
\]

With `|y_i-p_{t-tau_i}| <= epsilon_i`, every true current state satisfies

\[
y_i-\epsilon_i-\tfrac16J\tau_i^3
\le p_t-\tau_i v_t+\tfrac12\tau_i^2a_t
\le y_i+\epsilon_i+\tfrac16J\tau_i^3.
\]

Together with `|v_t|<=V` and `|a_t|<=A`, this defines a polyhedral set `C_i`.

### Step 2: verified intersection lemma

Define

\[
\mathcal C_{\mathrm{ground},t}
=\mathcal C_{\mathrm{base},t}\cap
\bigcap_{i\in V(S_P)}\mathcal C_i.
\]

If `x_t^star` belongs to `C_base,t` and to every accepted `C_i`, then

\[
x_t^\star\in\mathcal C_{\mathrm{ground},t}
\quad\text{and}\quad
\mathcal C_{\mathrm{ground},t}\subseteq\mathcal C_{\mathrm{base},t}.
\]

This is an exact set-theoretic proposition.

### Step 3: proposal non-interference

Under a sound verifier, a wrong proposal has only two outcomes:

1. it is rejected, leaving the certified set unchanged; or
2. it is accepted because it satisfies all trusted premises, in which case its constraint contains the true state.

Thus arbitrary proposal errors cannot directly remove the truth. They can reduce contraction by failing to propose informative constraints.

This statement fails if association validity or the sensor-error certificate is merely asserted by the proposer.

### Step 4: contraction gain

Define the log-volume gain

\[
G(S)=\log\operatorname{Vol}(\mathcal C_{\mathrm{base},t})
-\log\operatorname{Vol}(\mathcal C_t(S)).
\]

`G(S)>=0` and is monotone for valid added constraints. However, neither submodularity nor a uniform marginal gain follows for general polyhedral intersections. A claimed bound such as

\[
\mathbb E[G(S_P)]\ge rG(S_{\mathrm{all}})
\]

does **not** follow from proposal recall `r` without extra geometric assumptions. The diagnostic therefore measures gain directly instead of claiming this theorem.

### Step 5: robust-action monotonicity

If `C_1 subseteq C_2` and the robust HOCBF right-hand side uses a support supremum over `C`, then

\[
\mathcal U_{\mathrm{safe}}(C_1)\supseteq\mathcal U_{\mathrm{safe}}(C_2).
\]

For a fixed nominal action and projection objective,

\[
I(C)=\min_{u\in\mathcal U_{\mathrm{safe}}(C)}\|u-u_{\mathrm{nom}}\|_W^2
\]

satisfies `I(C_1)<=I(C_2)` whenever both sets are feasible. This is standard robust-optimization monotonicity and does not imply trajectory-level path or energy improvement.

## Remarks and Interpretation

- The trusted historical constraints, not the neural network, produce certification.
- A network can reduce the number of constraints checked, but simple deterministic history policies can do the same.
- Current-step support values are safety-filter-sufficient only for the current filter call. Recursive sufficiency requires retaining enough information to update the set at the next step.

## Boundaries and Non-Claims

- Feasibility of an intersection does not prove that it contains the true state.
- The simulator's oracle association flag is not a deployable association certificate.
- No finite-sample neural coverage theorem is proved.
- No global HOCBF feasibility or arbitrary-environment safety theorem is claimed.
- No new contraction-rate theorem is established.

## Open Risks

- Independent association certification remains unresolved.
- The axis-aligned bounding box can be much looser than the underlying polytope.
- LP latency is dominated by repeated bound extraction, so proposal pruning has limited measured benefit.
- Existing set-membership convergence and constraint-pruning work already covers the strongest defensible mathematical/computational positioning.

