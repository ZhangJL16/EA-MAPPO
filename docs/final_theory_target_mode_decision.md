# Final Theory-Contribution Target-Mode Decision

## CORE PROBLEM

Construct a feedback-valid, real-time safety object for sampled 3D UAV delivery
that converts causal obstacle history into less conservative uncertainty without
losing true-state containment under abrupt motion, and that still admits a
certified control at every 20 Hz cycle.

## WHY EXISTING METHODS FAIL IN OUR SCENARIO

Pointwise HOCBF and finite-horizon verification can certify a selected feasible
action or trajectory but do not ensure the next action set remains nonempty.
Adaptive set-membership/tube methods require a valid model family. The attempted
feasibility gate cannot establish that the true trajectory obeyed its narrow
history model. Generic terminal, backup, viability, and primitive-cover methods
address the remaining classes but do not yield a new low-cost UAV-specific
construction here.

## FINAL MATHEMATICAL OBJECT

No new positive object survives. The retained conditional object is the exact
bounded-noise history set `X_t^(L)` and its coordinatewise interval hull
`B_t^(L)`. The retained negative object is the pair of history-identical,
opposite admissible future-jerk trajectories.

## CORE THEOREM

The proposed core theorem—maximum feasible narrow-history window preserves
true-state containment—is **false**. A feasible LP may be witnessed by a false
latent trajectory while the true current `(p,v,a)` lies outside its hull.

The strongest valid residual theorem is pointwise: for every horizon time and
every history-measurable center predictor, one of two admissible opposite-jerk
futures has error at least `j_bar * tau^3 / 6`.

## THEOREM NOVELTY

The positive contraction, propagation, clearance, full-hold, and sampling
lemmas are standard set-membership, reachability, robust-tightening, and IID
sampling consequences. The negative jerk bound is an elementary minimax
boundary. None supplies paper-core novelty.

## ASSUMPTIONS

- bounded, correctly associated position measurements;
- exact discrete piecewise-constant-jerk history model for fixed-model claims;
- independently valid true-current-state containment;
- bounded measurable future jerk almost everywhere;
- exact ego rollout or a valid pathwise ego-error tube;
- full-hold analytic verification or a proved interpolation margin;
- conservative obstacle/body geometry;
- a certified action exists on every hold where safety is claimed.

## PROOF STATUS

**PARTIAL for the supporting package; FAILED for the adaptive core.**

T1--T5 and the full-tube inclusion proposition are conditionally valid after
proof repair. Adaptive theorem T6 is disproved. Recursive feasibility is absent.

## HISTORY CONTRIBUTION

History is useful as conditional estimation. Under steady motion, compensated
L16 velocity MAE is 0.0440. Under abrupt motion, fixed L16 MAE rises to 1.8471.
The adaptive feasibility rule does not convert that empirical phenomenon into a
certificate.

## PHYSICAL MOTION VECTOR CONTRIBUTION

**Engineering value supported.** Ego-motion compensation reduces steady L16
velocity MAE from raw-history 2.4815 to 0.0440. This is retained as sensor
preprocessing, not theory novelty.

## ADAPTIVE TUBE

**REJECTED AS CERTIFICATE.** In V6 the formula-only one-second endpoint metric
is 100% in all regimes, but certified containment is only 5%, 29%, and 24% in
sudden-velocity, sudden-direction, and stop-and-go regimes. Robust reset rate is
0% in all three abrupt regimes.

## SEARCH REGION GUARANTEE

**FAILED.** History did not prove containment of the safe or energy-optimal
primitive in a local parameter ball. Different predicted centers also invalidate
radius-only safe-set inclusion.

## SAMPLING GUARANTEE

Only the standard IID identity `1-(1-p)^N` survives under known safe mass `p`.
No deterministic epsilon-cover or computable history-induced lower bound on `p`
was constructed.

## RECURSIVE FEASIBILITY

**NOT PROVED.** No computable multi-obstacle controlled-invariant or terminal
recoverable set was instantiated for the actual sampled dynamics and perception
model.

## TRAJECTORY SAFETY

Conditional full-hold clearance transfer is valid when both ego and obstacle
tubes are valid. Existing B5 exact verification certified 54/54 selected
non-fallback sequences on bounded state subsets. This is not adaptive-history
closed-loop safety or recursive feasibility.

## PROGRESS GUARANTEE

Candidate-level CLF/progress filtering exists. Freeze, global reachability, and
positive progress under all safe states are not proved or measured for B5.

## ENERGY GUARANTEE

Only candidate-set ordering and a conditional standard Lipschitz-cover bound
survive. There is no mission-global energy theorem and no adaptive-history
energy comparison.

## CLOSEST PRIOR WORK

- Adaptive Model Predictive Safety Certification (adaptive set membership,
  robust tube, terminal safe set, recursive safety).
- Set Membership Based Nonlinear Model Predictive Control, EJC 2023
  (set-membership search-domain reduction).
- Adaptive conformal motion planning and SODA-MPC (adaptive uncertainty and
  monitor/fallback architecture).
- Adaptive/learning-based tube MPC (online uncertainty contraction and tube
  updates).
- Motion-primitive kinodynamic planning (resolution completeness and cost
  bounds).
- Robust MPC terminal sets, backup CBF, viability, and reachability
  (recoverability/recursive feasibility).

## EXACT NON-EQUIVALENCE

The current fixed-time history-window nesting and exact third-order jerk formula
are literal specializations not written identically in the closest papers.
They do not change theorem strength. The current one-hold guarantee is weaker
than recursive safety because action existence is assumed.

## COUNTEREXAMPLE RESULTS

- Exact zero-measurement construction: a narrow `j_H=0` LP is feasible at
  `(0,0,0)` while the true state after admissible jerks `(1,0,-1)` is
  `(p,v,a)=(0,-1/4,-5/6)`.
- V6 abrupt current-state/certified containment: 5%, 29%, 24%.
- Formula-only endpoint hit: 100%, demonstrating masking by the large future
  jerk term rather than valid certification.
- History cannot remove the pointwise `j_bar*tau^3/6` future term.
- Pointwise HOCBF feasibility margin does not imply next-step feasibility.

## NOVELTY SCORE

**15 / 30** for the adaptive-history method after soundness repair; below the
required 22/30 gate. Independent reviewer novelty score: 2.5/10.

## CLAIM_LEVEL

**NO_DEFENSIBLE_THEORY_CONTRIBUTION** (`NONE` in the response template).

## CLOSED-LOOP RESULTS

### Candidate A

- collision: 0 / 334 rollouts;
- success: 333 / 334 = 99.70%;
- freeze: not reported under the new definition;
- path ratio: 1.0383;
- mean energy: 4.6659, 3.817% below standard HOCBF;
- worst-rollout P99 latency: 53.78 ms;
- uncertified fallback steps: 119.

### Random Safe MPPI (B3)

- exact positive-state recall: 49/49 ordinary and 5/5 hard;
- closed-loop collision/success/freeze/path/energy: not run in the bounded
  state-level protocol;
- compute: exact-all style, not established below 50 ms.

### Proposed Adaptive-History Candidate

- closed-loop collision/success/freeze/path/energy: **not run**, because the
  necessary true-state-containment theorem failed before integration;
- synthetic certified containment: 100% steady/slow, 5%/29%/24% abrupt;
- one-obstacle estimator P99: 29.97--38.87 ms across regimes;
- full-stack multi-obstacle 20 Hz: not established.

## UNCERTIFIED FALLBACKS

- Candidate A: 119 steps on 334 hard rollouts.
- B5 state-level exact subset: 1/50 ordinary, 0/5 hard; fallback not executed.
- Proposed adaptive closed loop: not run, so zero may not be claimed.

## SAFE CANDIDATE RECALL

- coarse 100k survey: 99.9715% against the cheap endpoint-safe reference;
- exact subsets: 100% for B3 and B5, with only 49 ordinary and 5 hard positive
  states;
- no adaptive-history recall advantage was established.

## CANDIDATES REQUIRED FOR 99.9% RECALL

**NOT IDENTIFIED.** One thousand coarse proposals achieved 99.9715% on the
cheap reference, but this is not a lower bound, an exact 99.9% guarantee, or a
minimal candidate count.

## COMPUTE SAVING FROM HISTORY

**NONE ESTABLISHED.** Coarse-to-fine verification saves compute, but history did
not outperform random proposal sampling.

## ENERGY VS CANDIDATE A

**NOT MEASURED FOR THE PROPOSED ADAPTIVE METHOD.** No energy result is
transferred from Candidate A or B5.

## IS THIS READY AS THE PAPER CORE THEORY

**NO.**

## Strongest Surviving Mathematical Gap

A computable multi-obstacle controlled-invariant/recoverable subset for the
actual sampled 3D dynamics and perception uncertainty that guarantees a
certified action every 50 ms. This gap is important but is already occupied in
general form by viability, reachability, backup-CBF, terminal-MPC, and adaptive
MPSC. Two search rounds found no non-equivalent low-complexity construction.

After the history candidate failed, deterministic coverage, terminal
recoverability, and coverage-to-energy suboptimality were each attacked. The
first is covered by resolution-complete motion primitives and suffered cover
explosion; the second returned to standard invariant/backup sets without a new
construction; the third reduced to a standard conditional Lipschitz-cover
lemma whose assumptions were not globally verified. Continuing to rename these
objects is not scientifically defensible.

## FORMAL 500K

**NOT RUN.** Controlled work stayed within the requested 100k-state / 1,000-case
limits.

