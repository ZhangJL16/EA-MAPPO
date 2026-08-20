# Derivation Package: Feedback-Valid History-Adaptive Kinematic Tube

## Target

Determine whether causal motion history can produce a deterministic trajectory
uncertainty tube that is both tighter than a global worst-case tube and valid
when the controller observes, certifies, executes one zero-order-hold control,
and repeats. The immediate goal is a set-membership proposition and a
sampled-feedback safety theorem, not a claim that history predicts arbitrary
future maneuvers.

## Status

**ADAPTIVE CERTIFICATION CLAIM FAILED; CONDITIONAL SET-MEMBERSHIP LEMMAS
SURVIVE.** History can contract the set of current kinematic states consistent
with a *declared* bounded-jerk model. However, feasibility of a narrow-history
model is only existential: it does not prove that the true latent trajectory
obeyed that narrow jerk bound. In abrupt-motion tests the narrow LP remained
feasible while excluding the true current state. Therefore the proposed
maximum-feasible-window rule is an estimator heuristic, not an
observation-only certificate. The bounded-jerk propagation and one-hold safety
results remain valid only when the selected present-state set is independently
known to contain the true state. Those conditional results are standard
set-membership/reachability consequences, not a paper-core contribution.

## Invariant Object

The invariant object is the set of all present obstacle kinematic states that
can explain the causal measurements under the declared physical bounds:

\[
\mathcal X_t^{(L)}
=\operatorname{proj}_{(p_t,v_t,a_t)}\left\{
(p_k,v_k,a_k)_{k=t-L+1}^{t},
(j_k)_{k=t-L+1}^{t-1}:
\begin{array}{l}
|y_k-p_k|\le \delta_z,\\
|v_k|\le \bar v,\ |a_k|\le \bar a,\ |j_k|\le \bar j,\\
(p_{k+1},v_{k+1},a_{k+1})=F_{\Delta_k}(p_k,v_k,a_k,j_k),
\quad k=t-L+1,\ldots,t-1
\end{array}
\right\}.
\]

The implementation retains only the coordinatewise interval hull

\[
\mathcal B_t^{(L)}=
\prod_q\left[
\inf_{x\in\mathcal X_t^{(L)}}x_q,
\sup_{x\in\mathcal X_t^{(L)}}x_q
\right],
\]

where \(q\) ranges over the components of \((p_t,v_t,a_t)\). Thus
\(\mathcal X_t^{(L)}\subseteq\mathcal B_t^{(L)}\), but arbitrary box corners
need not belong to the exact projected polytope. The three Cartesian axes are
solved independently.

## Assumptions

- **A1 (bounded measurement error):** each Cartesian position measurement
  satisfies \(|y_k-p_k|\le\delta_z\).
- **A2 (piecewise-constant jerk over a sensor interval):** for
  \(\Delta_k=t_{k+1}-t_k\), jerk is constant and the third-order kinematic
  update is exact over the interval.
- **A3 (physical prior bounds):** \(|v_k|\le\bar v\),
  \(|a_k|\le\bar a\), and \(|j_k|\le\bar j\) componentwise.
- **A4 (future-motion bound):** during every certification hold, the future
  obstacle jerk is bounded by a declared robust value \(\bar j_R\). History
  consistency does not replace this assumption.
- **A5 (ego prediction):** either the clipped ZOH ego rollout is exact over the
  hold, including latency and actuation, or a declared pathwise bound
  \(\eta_{ego}(\tau)\) contains true ego-to-predicted position error.
- **A6 (sensing timing):** timestamps are known and strictly increasing.
- **A7 (geometry):** obstacle geometry and the UAV body radius are conservatively
  represented in the clearance calculation.

If A1--A4 are violated, the deterministic containment claim is invalid. An
innovation detector can report a violation after it becomes observable, but it
cannot retroactively protect the preceding hold interval.

## Notation

- \(p,v,a,j\): obstacle position, velocity, acceleration, and jerk.
- \(y\): measured obstacle position.
- \(L\): number of causal frames.
- \(\mathcal X_t^{(L)}\): present-state set induced by the last \(L\) frames.
- \(e_p,e_v,e_a\): componentwise half-widths of the projected present-state
  intervals.
- \(\bar j_R\): componentwise robust future jerk bound.
- \(\epsilon_t(\tau)\): Euclidean future-position tube radius.
- \(d_{safe}\): required geometric clearance including body radii.

## Derivation Strategy

1. Express bounded-jerk kinematics and bounded measurement error as a linear
   feasibility system.
2. Project this system onto the present state by six linear programs per axis.
3. Prove that adding a consistent older measurement cannot enlarge the projected
   set.
4. Integrate the present-state interval and future jerk bound to obtain a
   componentwise trajectory tube.
5. Transfer predicted clearance to true clearance.
6. apply the transfer independently on every executed hold interval.
7. prove an indistinguishability lower bound showing what history cannot shrink.

## Derivation Map

1. **Identity:** exact discrete jerk dynamics define the feasible-history set.
2. **Proposition:** projection of a longer feasible history is contained in the
   projection of its recent suffix.
3. **Proposition:** integrating interval initial errors and bounded future jerk
   gives the analytic tube.
4. **Lemma:** clearance larger than the tube implies true clearance.
5. **Theorem:** repeated one-hold certification gives sampled-feedback safety.
6. **Impossibility proposition:** identical history followed by opposite
   admissible future inputs forces a nonzero worst-case tube term.

## Main Derivation

### Step 1. Exact discrete bounded-jerk dynamics

For one axis and one interval of length \(\Delta_k\),

\[
\begin{aligned}
p_{k+1}&=p_k+\Delta_kv_k+\tfrac12\Delta_k^2a_k
          +\tfrac16\Delta_k^3j_k,\\
v_{k+1}&=v_k+\Delta_ka_k+\tfrac12\Delta_k^2j_k,\\
a_{k+1}&=a_k+\Delta_kj_k.
\end{aligned}
\]

These are identities under A2. Together with box bounds and
\(-\delta_z\le y_k-p_k\le\delta_z\), they form a polyhedron. The implemented
estimator obtains lower and upper bounds on each component of \((p_t,v_t,a_t)\)
by linear programming.

### Theorem 1. History-consistent set contraction

Let \(L_2>L_1\ge2\), and let both feasible sets be nonempty. Under identical
physical and noise bounds for the common suffix,

\[
\mathcal X_t^{(L_2)}\subseteq\mathcal X_t^{(L_1)}.
\]

Consequently, the diameter of the longer-history present-state set cannot
exceed that of the shorter-history set in any norm. Every coordinate interval
also contracts, so
\(\mathcal B_t^{(L_2)}\subseteq\mathcal B_t^{(L_1)}\).

**Proof.** Take any present state in \(\mathcal X_t^{(L_2)}\). By definition,
there is a complete feasible latent trajectory over the longer history ending
at that state. Restrict this trajectory to its final \(L_1\) samples. All
dynamics, bounds, and measurement inequalities required by the shorter problem
remain satisfied. Therefore the same present state belongs to
\(\mathcal X_t^{(L_1)}\). Set inclusion implies non-increasing diameter.
Coordinatewise infima cannot decrease and suprema cannot increase under set
inclusion, proving the interval-hull corollary. ∎

**Boundary.** The result is non-strict. Uninformative or redundant older
measurements need not reduce the set. If a different, tighter motion bound is
used only for the longer history, the inclusion comparison is no longer valid.

### Step 2. Adaptive window and its failed certification semantics

For candidate lengths \(\mathcal L=\{2,4,8,16\}\), define

\[
L_t=\max\{L\in\mathcal L:\mathcal X_t^{(L)}(\bar j_H)\ne\varnothing\}.
\]

If no window at least four frames is feasible under the history-mode jerk bound
\(\bar j_H\), the implementation resets to the last two samples and recomputes
the set using \(\bar j_R\ge\bar j_H\). This is a deterministic consistency
heuristic. It is not a certificate that the true history obeyed \(\bar j_H\):
another latent trajectory can satisfy the narrow model and measurement boxes
even when the true latent trajectory violated the bound. Consequently,
non-emptiness of \(\mathcal X_t^{(L)}(\bar j_H)\) cannot be used as an online
proof of true-state containment.

### Theorem 2. Bounded-jerk future position tube (conditional)

Assume first that the present-state box contains the true current
\((p_t,v_t,a_t)\). Let the box have componentwise half-widths
\(e_p,e_v,e_a\). Predict from the box center with
\(\hat p(\tau)=c_p+\tau c_v+\frac12\tau^2c_a\). Let true jerk be measurable,
essentially bounded, and satisfy \(|j(s)|\preceq\bar j_R\) almost everywhere.
Under A4, for every \(\tau\in[0,H]\),

\[
|p_{true}(\tau)-\hat p(\tau)|
\preceq e_p+\tau e_v+\tfrac12\tau^2e_a+\tfrac16\tau^3\bar j_R.
\]

Therefore a valid Euclidean radius is

\[
\epsilon_t(\tau)=\left\|e_p+\tau e_v+
\tfrac12\tau^2e_a+\tfrac16\tau^3\bar j_R\right\|_2.
\]

**Proof.** Subtract the center prediction from the true third-order dynamics.
The initial-condition terms are linear. The jerk remainder is
\(\int_0^\tau \frac12(\tau-s)^2j(s)\,ds\). Apply the componentwise triangle
inequality and integrate the kernel:
\(\int_0^\tau\frac12(\tau-s)^2ds=\tau^3/6\). Taking the Euclidean norm yields
the stated radius. ∎

**Critical premise.** This theorem cannot be invoked merely because the LP is
feasible. Artifact V5 shows that the adaptive narrow-model box excludes the
true current state in most abrupt-motion cases. A large future-jerk term may
still cover a single future endpoint empirically, but that coincidence does not
repair the missing premise or establish pathwise containment.

### Lemma 1. Tube-to-clearance transfer

If for every \(\tau\in[0,H]\),

\[
\operatorname{dist}(\hat p_{ego}(\tau),\hat p_{obs}(\tau))
\ge d_{safe}+\epsilon_t(\tau)+\eta_{ego}(\tau),
\]

then

\[
\operatorname{dist}(p_{ego,true}(\tau),p_{obs,true}(\tau))\ge d_{safe}
\quad\forall\tau\in[0,H].
\]

**Proof.** Apply the reverse triangle inequality twice to subtract obstacle
prediction error and ego rollout error, then apply Theorem 2 and A5. ∎

### Theorem 3. One-hold feedback-valid safety

Suppose that at every control sample \(t_k\):

1. A1--A7 hold on \([t_k,t_{k+1}]\) and the selected present-state set
   contains the true state;
2. the selected first control is verified analytically on the entire hold, or
   by a grid with a proved interpolation margin, against the combined obstacle
   and ego tubes;
3. the verifier establishes the clearance condition in Lemma 1.

Then the true closed-loop trajectory is collision-free on every executed hold.

**Proof.** Lemma 1 proves safety on any arbitrary certified hold. Apply it
independently to every executed hold. The union of those adjacent safe intervals
is safe. This is not invariant-set induction because containment and action
existence are assumed anew at each sample. ∎

**Non-claim.** This does not prove that a certified action always exists. It is
not recursive feasibility and requires a separately certified fallback or
terminal invariant construction to eliminate uncertified fallback.

### Theorem 4. History-only future-jerk uncertainty lower bound

Assume two environments generate exactly the same complete history and present
state, including the same current acceleration. Immediately after the present
sample, one applies constant jerk \(+\bar j\) and the other \(-\bar j\) along
an axis, and both trajectories remain inside the declared acceleration and
velocity bounds on \([0,H]\). Pointwise, for every \(\tau\in[0,H]\) and every
history-measurable center prediction \(\hat p_\tau\), there exists a sign
\(\sigma\in\{-1,+1\}\) such that

\[
|p_\sigma(\tau)-\hat p_\tau|\ge\bar j\tau^3/6.
\]

**Proof.** The two jerk futures are separated by
\(2\bar j\tau^3/6\). For any center, the maximum distance to two points is at
least half their separation. ∎

An acceleration lower bound \(\bar a\tau^2/2\) is valid for a separate
second-order model in which acceleration may switch directly after the sample.
It is not asserted under the present continuous-acceleration bounded-jerk
model.

**Consequence.** A deterministic tube that removes the global future-motion
term merely because recent history is smooth is invalid. A residual spike can
trigger conservative mode only after the new behavior affects a measurement.

### Proposition 1. Tube-inclusion safe-set expansion and sampling count

Let the complete predicted obstacle tubes satisfy

\[
\hat p_1(\tau)\oplus B(0,\epsilon_1(\tau))
\subseteq
\hat p_2(\tau)\oplus B(0,\epsilon_2(\tau))
\quad\forall\tau.
\]

Then the corresponding robustly certified primitive sets satisfy

\[
\mathcal S(\mathcal T_2)\subseteq\mathcal S(\mathcal T_1).
\]

If controls are sampled IID from a fixed distribution assigning mass
\(p\in(0,1)\) to \(\mathcal S(\mathcal T)\), the probability of finding at
least one certified primitive after \(N\) samples is
\(1-(1-p)^N\). For \(\beta\in(0,1)\), the minimum integer count reaching
probability \(1-\beta\) is

\[
N_{min}=\left\lceil\frac{\log\beta}{\log(1-p)}\right\rceil.
\]

For \(p=1\), one sample suffices.

For equal centers, radius ordering is sufficient. For different centers it is
not: a sufficient scalar condition is
\(\|\hat p_1(\tau)-\hat p_2(\tau)\|+\epsilon_1(\tau)
\le\epsilon_2(\tau)\). The Bernoulli identity shows how a *verified full-tube
inclusion* can reduce the samples needed to find one safe candidate. The
adaptive history rule establishes neither this inclusion nor a computable
lower bound on \(p_\epsilon\), so it provides no search-complexity guarantee.

## Numerical and Adversarial Checks

The implementation is in
`review_bundle/safety/trajectory/adaptive_history_tube.py`; tests cover set
contraction, abrupt-history reset, pathwise tube containment, and the
indistinguishable-future lower bound.

The corrected controlled artifact
`artifacts/adaptive_history_tube_1k_20260819_v6/summary.json` contains 200 cases
for each of five regimes:

| Regime | Formula-only endpoint hit | Certified containment with valid premise | True current state inside adaptive box | History bound actually satisfied | Mean selected L | P99 latency |
|---|---:|---:|---:|---:|---:|---:|
| constant velocity | 100% | 100% | 100% | 100% | 16.00 | 29.97 ms |
| slow acceleration | 100% | 100% | 100% | 100% | 16.00 | 35.81 ms |
| sudden velocity change | 100% | 5.0% | 5.0% | 0% | 14.84 | 36.63 ms |
| sudden direction change | 100% | 29.0% | 29.0% | 0% | 10.00 | 38.87 ms |
| stop-and-go | 100% | 24.0% | 24.0% | 0% | 11.18 | 35.00 ms |

The no-history radius was 31.212 m and the fixed-L2 radius was 15.103 m. The
adaptive radius remained roughly 10 m because the robust future jerk term alone
contributes 8.66 m in Euclidean norm at one second. That broad term masks the
invalid initial box in the endpoint metric. Certified-under-declared-robust-model
rate is 0% in every regime because every selected adaptive result remains
history-informed and requires the external history-motion-bound assumption.
Infeasible results are never counted as certified. The LP timing is for one tracked obstacle;
multiple obstacles leave insufficient headroom for a full 20 Hz planner.

## Remarks and Interpretation

- Under a valid narrow motion model, steady history contracts the present-state
  set. Under abrupt motion, maximum feasible window does not reliably detect
  violation: false latent histories keep the LP feasible.
- Retaining \(\bar j_R\) controls future jerk only; it cannot certify an
  initial box that excludes the true current state.
- The result does not establish that history narrows the neighborhood of the
  optimal primitive. Parametric optimizer continuity requires separate strong
  regularity assumptions and is already standard sensitivity analysis.
- The result can improve safe-candidate availability, but the current one-second
  tube is too wide for many tight-clearance scenes.

## Prior-Art Boundary

The positive chain is already represented by set-membership state estimation,
adaptive/learning-based tube MPC, adaptive model-predictive safety
certification, set-membership NMPC search-domain reduction, adaptive conformal
motion-planning tubes, and OOD-triggered reachability fallback. The exact UAV
jerk formula and bounded implementation are useful specialization, not a new
mathematical object or guarantee.

## Boundaries and Non-Claims

- No deterministic safety under undeclared acceleration or jerk changes.
- No guarantee during the detection delay after a physical-bound violation.
- No recursive feasibility or certified fallback construction.
- No deterministic candidate coverage from random samples.
- No history-conditioned containment of the optimal control parameter.
- No mission-energy optimality.
- No multi-obstacle 20 Hz claim from the one-obstacle LP timing.

## Open Risks

1. Real perception does not directly provide bounded 3D obstacle positions with
   known associations.
2. Global jerk bounds may be difficult to validate and can dominate the tube.
3. Multiple obstacle estimates consume most of the 50 ms cycle.
4. Occlusion and new obstacle appearance require a conservative reachable-set
   fallback independent of history.
5. A terminal invariant set is still required for recursive feasibility.

## Novelty Decision

| Dimension | Score / 5 | Reason |
|---|---:|---|
| Novel mathematical object | 1 | set-membership projected state set and tube are established objects |
| New guarantee | 1 | only conditional hold safety and an elementary negative bound survive |
| Difference from closest prior | 1 | only UAV kinematics and exact implementation differ |
| Relevance to observed failure | 5 | directly explains steady-history gain and abrupt-history failure |
| Experimental falsifiability | 5 | containment, width, reset, latency, and candidate availability are measurable |
| Compute deployability | 2 | one-obstacle estimator meets 50 ms; full-stack multi-obstacle cycle does not |
| **Total** | **15 / 30** | below the required 22-point gate |

**Decision: REJECT AS PAPER-CORE THEORY; KEEP AS A PHYSICS-BASED ESTIMATION
UTILITY AND AS AN IMPOSSIBILITY BOUND ON HISTORY-ONLY SAFETY CLAIMS.**
