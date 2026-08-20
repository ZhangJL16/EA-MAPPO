# UAV Theory Contribution Search: Final Research Decision

## Core unsolved problem

The deployed sampled-data HOCBF filter can face states where bounded acceleration
and simultaneous obstacle rows have no common action. In such states the current
emergency action is not certified. The unresolved problem is therefore:

> construct, before deployment, a nonempty controlled-invariant or recoverable
> subset for the actual sampled UAV dynamics and perception model, and provide a
> real-time policy that never leaves it while retaining useful navigation.

This remains important. It was not solved by the candidates explored here.

## Why the current HOCBF is theoretically incomplete

Forward-invariance claims are conditional on a feasible HOCBF action at every
sample and on a valid inter-sample implementation. The earlier selected filter
had 34 infeasible/uncertified fallback steps in 125 held-out scenarios. The new
harder 334-scenario benchmark has 119 such steps for Candidate A. Zero observed
collisions cannot extend a theorem over those states.

## Main mathematical object

The search introduced the joint control-authority margin

\[
\rho(x)=\max_{u\in\mathcal U}\min_i(A_i(x)u-b_i(x)),
\]

with the exact dual

\[
\rho(x)=\min_{\lambda\in\Delta}
\sigma_{\mathcal U}(A^\top\lambda)-b^\top\lambda.
\]

For the UAV cylinder input set,

\[
\sigma_{\mathcal U}(q)=\bar a_{xy}\|q_{xy}\|_2+\bar a_z|q_z|.
\]

This is an exact, cheap pointwise feasibility diagnostic. It is not a recursive
feasibility certificate.

## Branch-by-branch outcome

### 1. Direct feasibility barrier

**Rejected.** `rho(x_k)>=0` does not imply `rho(x_{k+1})>=0`. Adding a CBF row
on `rho` can conflict with the original HOCBF row, so the construction is
circular unless an independently admissible backup/invariant controller is
already known.

### 2. Certified backup set

**No new construction found.** Minimum stopping distance supplies physical
intuition but does not resolve multiple 3D obstacle conflicts. A certified
backup flow and terminal set return directly to established backup-CBF,
reachability, viability, or MPC-terminal-set methods. No closed-form
energy-optimal backup for the multi-obstacle case was derived.

The dedicated 100,000-state physical search confirms that this is not a rare
symbolic corner: 58,059 sample-strengthened action sets were empty, including
10,225 states with both `h>=0` and `psi1>=0`.  A pointwise margin diagnoses
these states but supplies no action that can recover them.

### 3. Lexicographic safety then energy

**Mathematically valid pointwise, practically rejected.** Minimizing `u^T R_E u`
over a fixed hard set with a progress floor has pointwise optimality. The
implemented candidate reduced infeasibility from 305 to 15 steps on the hard
benchmark, but success collapsed from `333/334` to `69/334`, mean energy rose
from `4.8466` to `15.9561`, and worst-rollout P99 reached `145.24 ms`.

### 4. One-step Energy-to-Go upper objective

**Conditional theorem only; rejected for deployment.** The descent lemma yields
a convex quadratic upper bound if the composed learned value has a known useful
Lipschitz-gradient constant. The current MC estimator contains ReLU activations
and a normalized goal-direction feature; no finite useful verified global/local
constant was established. Statistical conformal coverage cannot replace a
deterministic smoothness certificate.

### 5. Energy supersolution

**Not constructed.** A verified Bellman supersolution would telescope to a
finite-horizon energy bound, but the MC regressor plus conformal residual does
not certify `Vbar>=c+Vbar(F)`. Reintroducing unstable gamma-one TD does not solve
this verification problem.

### 6. Exact inter-sample sphere clearance

**Valid diagnostic, insufficient novelty.** Under exact static-sphere and
double-integrator ZOH assumptions, the barrier is quartic and its exact minimum
is found from endpoints plus cubic stationary roots. This certifies a given
action. A convex recursively feasible synthesis set was not derived, while
generic sampling-aware/ZOCBF theory is already populated.

## Valid theorem inventory

| Statement | Status | Strength |
| --- | --- | --- |
| HOCBF affine row | Proved and symbolically checked | Existing |
| Single-row support margin | Proved; 100k numerical checks | Existing convex analysis |
| Joint primal-dual margin | Proved; 100,000 cases, max gap `9.97e-8` | Pointwise only |
| `rho` local Lipschitz on fixed finite active-set region | Proved | Existing maximum theorem consequence |
| Exact static-sphere ZOH action verification | Proved; 100k checks | Special-case verifier |
| Pointwise quadratic action-energy dominance | Proved | Definitional |
| Recursive feasibility from `rho` | **Disproved** | Counterexample |
| One-step learned Energy-to-Go upper bound | Conditional | Assumption unmet |
| Finite-horizon energy bound | Not proved | No supersolution |

## Computational result

Pointwise joint-margin diagnostic:

- mean `2.994 ms`;
- P95 `7.081 ms`;
- P99 `32.456 ms`;
- maximum `393.650 ms`, with `19/100000` deadline misses.

Two-stage online filter:

- mean `8.194 ms`;
- worst-rollout P95 `89.104 ms`;
- worst-rollout P99 `145.238 ms`;
- 16,476 50 ms deadline misses.

The diagnostic is typically compatible with 20 Hz but lacks a hard tail bound.
The full candidate is not real-time at 20 Hz.

On the targeted physical hard-state set, evaluating both continuous and sampled
certificates was substantially slower: mean/P95/P99
`27.266/119.294/311.720 ms`, with 12,518 deadline misses.  This is additional
evidence against presenting global certificate recovery as a 20 Hz online
solution.

## Strongest remaining theoretical gap

The remaining gap is not another scalar margin. It is an **explicit sampled-data
controlled-invariant domain for the full bounded-input, multi-obstacle UAV** that
is:

1. computable without an intractable global reachability solve;
2. compatible with changing sensed obstacle sets;
3. guaranteed to admit an action at every 50 ms hold;
4. noncircular under its own backup/viability conditions;
5. sufficiently large to preserve navigation.

The literature already provides generic backup, viability, reference-governor,
and sampled-data frameworks. No non-equivalent solution to this gap was found
within the current model. Continuing to rename those frameworks would not be
scientifically defensible.

## Research decision

`CLAIM_LEVEL = ENGINEERING_CONTRIBUTION_ONLY`

`IS_THIS_READY_TO_BE_A_PAPER_THEORY_CONTRIBUTION = NO`

`FORMAL_500K = NOT_STARTED`

The defensible next route is a robotics/autonomous-systems paper centered on
measured sampled-data failure modes, compute tails, sensor dropout boundaries,
and energy overhead of established filters. The exact margin can remain a
diagnostic and adversarial-state generator, but not the headline theorem.
