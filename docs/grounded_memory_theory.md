# Structured Memory Theory Derivation

## 1. Setting

Let the complete observable history be

\[
H_t=(o_{0:t},u_{0:t-1}^{\mathrm{nom}},u_{0:t-1}^{\mathrm{exec}}).
\]

For module `k` in `{safe, route, energy}`, let

\[
M_t^k=\sigma_k(H_t),\qquad M_{t+1}^k=F_k(M_t^{\mathrm{pa}(k)},o_{t+1},u_t^{\mathrm{exec}}).
\]

Let `phi_k(H_t)` denote the ideal module-relevant information state. The learned/implemented memory approximates it by `hat_phi_k(M_t^k)`.

## 2. Assumptions

A1. The ideal module states are recursively updateable for the declared downstream problem.

A2. On a declared domain,

\[
d_k(\phi_k(H_t),\hat\phi_k(M_t^k))\le \epsilon_k.
\]

A3. A downstream scalar objective decomposes as

\[
J(\phi_s,\phi_r,\phi_e)=J_s(\phi_s)+J_r(\phi_r)+J_e(\phi_e)
\]

and each `J_k` is `L_k`-Lipschitz in `d_k`.

A4. The certified set `E_t` contains the true obstacle state over the declared sample/hold interval.

A5. The robust safety filter forms its hard feasible set only from `E_t`, ego state, actuator limits, and fixed robust-HOCBF parameters.

A6. The solver returns a feasible action whenever the hard feasible set is nonempty; fallback behavior is separately declared when it is empty.

These assumptions are strong. In particular, learned grounding losses do not imply A2 or A4.

## 3. Proposition 1: modular error-to-objective bound

Under A2-A3,

\[
|J(\phi_s,\phi_r,\phi_e)-J(\hat\phi_s,\hat\phi_r,\hat\phi_e)|
\le L_s\epsilon_s+L_r\epsilon_r+L_e\epsilon_e.
\]

### Proof

Add and subtract one approximate module at a time, apply the triangle inequality, and then apply the corresponding Lipschitz condition. No independence between module errors is required.

### Audit

The statement is valid but not novel. It is a direct separable Lipschitz composition and is weaker than approximate-information-state performance bounds because it does not address recursive error accumulation or policy-induced distribution shift.

## 4. Proposition 2: learned/certified non-interference

Define

\[
u^{\mathrm{nom}}=\pi(x,M^{\mathrm{learn}}),\qquad
u^{\mathrm{exec}}=\arg\min_{u\in\mathcal U_{\mathrm{cert}}(x,E)}
\|u-u^{\mathrm{nom}}\|_W^2.
\]

If A4-A6 hold, then for every learned message, including zero, random, stale, or adversarial messages,

\[
u^{\mathrm{exec}}\in\mathcal U_{\mathrm{cert}}(x,E).
\]

Consequently, any safety conclusion already implied by membership in this certified robust sampled-data action set is unchanged by learned-message errors.

### Proof

The learned message changes only the objective center `u_nom`. The constraint set is independent of that message. By A6 the optimizer returns an element of the same set for every objective center. The existing robust sampled-data safety result then applies under A4.

### Audit

Correct but existing. It specializes the modular safety-filter separation principle and measurement-robust/output-feedback CBF results. It does not make an invalid certified set valid.

## 5. Proposition 3: uncertainty contraction and feasible-set monotonicity

Let `E1 subseteq E2` be two certified obstacle uncertainty sets. Suppose every robust HOCBF constraint has the form

\[
a_i^\top u\ge b_i+\sup_{e\in E}q_i(e),
\]

with all other constraints fixed. Then

\[
\mathcal U_{\mathrm{safe}}(E_1)\supseteq
\mathcal U_{\mathrm{safe}}(E_2).
\]

### Proof

Set inclusion implies

\[
\sup_{e\in E_1}q_i(e)\le \sup_{e\in E_2}q_i(e)
\]

for every constraint. Every action satisfying the tighter `E2` constraints therefore satisfies the `E1` constraints.

### Audit

Correct but standard robust-optimization monotonicity. It requires **sound** set contraction. A learned smaller set without coverage evidence does not satisfy the premise.

## 6. Corollary: intervention optimum monotonicity

For fixed nominal action and positive semidefinite `W`, define

\[
I(E)=\min_{u\in\mathcal U_{\mathrm{safe}}(E)}
\|u-u^{\mathrm{nom}}\|_W^2.
\]

Whenever both feasible sets are nonempty, Proposition 3 gives

\[
I(E_1)\le I(E_2).
\]

This is pointwise and says nothing about trajectory-level path length or energy, because changed actions alter future states and future feasible sets.

## 7. Candidate module-specific information-state theorem

One might define a safety memory as barrier sufficient if it preserves exactly the support values needed by every robust barrier constraint and is recursively predictable. Route and energy memories can be defined analogously through their costs and transitions. Exact preservation would let each downstream module operate on its own information state without loss.

This is a valid specialization of information-state theory, not a new theorem. The approximate case inherits the same core requirement: bound the error in cost evaluation and next-information-state sets. Auxiliary reconstruction heads do not establish these conditions.

## 8. Counterexamples

1. **Object-ID swap:** low motion reconstruction error can coexist with association error; the certified interval no longer contains the true track.
2. **New obstacle:** a persistent slot can preserve stale information while failing to instantiate a new constraint.
3. **Long dropout:** a learned center remains accurate on average while certified radii grow until the robust action set is empty.
4. **Adversarial route message:** non-interference preserves hard feasibility but can cause unnecessary detour or freeze.
5. **Wrong energy message:** collision safety persists, but mission energy may degrade because energy ranking changes the nominal action.
6. **Unsound learned contraction:** a narrower predicted set expands the action set and can invalidate safety; Proposition 3 does not help because A4 fails.
7. **Nonseparable objective:** cross-terms between route and energy invalidate the simple sum-of-Lipschitz bound unless a joint Lipschitz constant is used.
8. **Closed-loop accumulation:** a one-step representation error bound need not control long-horizon value without recursive transition-error conditions.

## 9. Proof status

| Result | Validity | Novelty |
|---|---|---|
| Modular Lipschitz bound | valid under A2-A3 | low/existing composition |
| Certified-channel non-interference | valid under A4-A6 | existing safety-filter principle |
| Uncertainty-set/action-set monotonicity | valid | standard robust optimization |
| Intervention optimum monotonicity | valid pointwise | immediate corollary |
| Module-specific exact/approximate sufficiency | semantically valid | specialization of information-state theory |
| Memory grounding implies certified contraction | **not proved** | central missing result |

## 10. Theory conclusion

The structured architecture is mathematically coherent, but the current theorem package does not reach the required novelty threshold. The only potentially new bridge—control-grounded memory training producing a smaller yet still valid certified uncertainty set—requires a finite-sample or deterministic coverage theorem that the current data, losses, and observer do not provide.
