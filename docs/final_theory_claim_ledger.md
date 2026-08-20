# Final Theory Claim Ledger

## Claim Levels

- **VALID:** proved under stated assumptions and numerically checked.
- **CONDITIONAL:** proof is valid but a required object or bound is not
  instantiated for deployment.
- **FAILED:** a counterexample invalidates the statement.
- **STANDARD:** valid but theorem-level novelty is rejected by closest prior.
- **EMPIRICAL ONLY:** supported only on the declared artifact distribution.

| ID | Statement | Assumptions | Proof | Numerical check | Counterexamples searched | Closest prior theorem / structure | Exact difference | Claim level |
|---|---|---|---|---|---|---|---|---|
| T1 | nonempty bounded-noise jerk-history sets contract, and their coordinatewise interval hulls contract | identical physical/noise bounds; consistent exact discrete jerk model; common suffix | restrict any longer feasible latent trajectory to its suffix; monotonic coordinate extrema give the hull corollary | LP interval widths L16 no larger than L4 in deterministic tests | empty/inconsistent set; redundant measurements; model mismatch | set-membership estimation and adaptive MPSC parameter-set contraction | obstacle kinematic state and third-order UAV-relative geometry | VALID / STANDARD |
| T2 | future component error is bounded by \(e_p+\tau e_v+\frac12\tau^2e_a+\frac16\tau^3\bar j_R\) | true current state in projected box; declared robust future jerk bound | integrate initial-state error and jerk remainder | valid-model unit cases; V6 shows the adaptive box violates the premise in abrupt regimes | unbounded jerk; wrong association; false feasible latent history; sensor error above bound | robust tube propagation and bounded-error filtering | explicit third-order closed form | VALID CONDITIONAL / STANDARD; not established for the adaptive selector |
| T3 | predicted clearance at least \(d_{safe}+\epsilon_{obs}+\eta_{ego}\) implies true clearance at least \(d_{safe}\) | T2, conservative geometry, and exact ego rollout or a valid pathwise ego-error bound | reverse triangle inequality twice | unit and randomized checks | ego model/latency error; underestimated body/obstacle radius | generic robust tube tightening | exact sphere/UAV geometry | VALID CONDITIONAL / STANDARD |
| T4 | the union of independently certified full hold intervals is collision-free | T1--T3 at every sample; analytic intersample verification or proved grid margin; a certified action exists each cycle | apply the hold implication separately and take the union | exact hold verifier tests | endpoint-only checking; infeasible action set; delayed/new obstacle; invalid adaptive present-state box | robust receding-horizon safety | implementation has an exact static-sphere hold verifier | VALID CONDITIONAL; not recursive feasibility or invariant induction |
| T5 | pointwise, identical history cannot reduce an unannounced bounded future jerk term below \(\bar j\tau^3/6\) | opposite constant-jerk futures are both admissible on the stated horizon and share the present state | pointwise half-separation minimax argument | physically admissible same-state algebraic unit test | asymmetric future set; velocity/acceleration bound violation; side information revealing future mode | worst-case indistinguishability in robust prediction/control | explicit warning for history-conditioned safety tubes | VALID / STANDARD NEGATIVE BOUND; quantifiers are \(\forall\tau\forall\hat p_\tau\exists\sigma\) |
| T6 | adaptive maximum-feasible window preserves current-state containment | LP feasibility under the fast model is treated as evidence that the true history obeyed that model | disproved: feasibility is existential and may be witnessed by a false latent trajectory | V6 certified containment: sudden velocity 5%, direction 29%, stop-and-go 24%; robust reset 0% | exact zero-measurement latent-state counterexample and 600 abrupt cases | adaptive set-membership/MHE requires a valid model set or certified mode logic | no surviving difference | FAILED |
| T7 | residual-triggered reset permits a smaller future jerk bound | smooth recent history predicts future mode | no valid proof | detector often shortens history, but future robust bound was still required | identical smooth history followed by opposite jerk | SODA-MPC and robust fallback theory expose same limitation | none | FAILED |
| T8 | full obstacle-tube inclusion monotonically expands the certified primitive set | same primitive library and geometry; \(\hat p_1\oplus B_{\epsilon_1}\subseteq\hat p_2\oplus B_{\epsilon_2}\) pointwise | direct constraint inclusion | not separately needed | different centers invalidate radius-only ordering | robust constraint tightening | UAV candidate library | VALID / STANDARD; adaptive history did not establish the premise |
| T9 | uniform samples find a safe primitive with probability \(1-(1-p)^N\) | i.i.d. sampling; safe-set probability mass \(p>0\) | complement probability | coarse recall measured empirically | narrow zero-measure safe set | random shooting/scenario sampling | none | VALID / STANDARD |
| T10 | history places the optimal safe primitive in a local ball \(B(\hat\theta,R_t)\) | unique stable optimizer and a verified sensitivity map | missing | no history proposal advantage over random | homotopy switch and discontinuous optimizer | parametric optimization; SM-NMPC bounds | no verified difference | FAILED / UNSUPPORTED |
| T11 | an \(\epsilon\)-cover of feasible controls gives \(J(U_Q^*)-J(U^*)\le L_J\epsilon\) | compact feasible set; valid cover; globally Lipschitz cost | select cover neighbor of continuous optimum | no deterministic cover constructed | clipping, nonsmooth active sets, narrow corridors | resolution-optimal kinodynamic planning | telemetry energy objective | CONDITIONAL / STANDARD |
| T12 | terminal backup membership plus shifted sequence gives recursive feasibility | deterministic consistency; robust invariant backup set; suffix remains feasible | standard shift-and-append induction | no concrete backup set tested | dead-end horizon; moving obstacle invalidates suffix | robust MPC / backup CBF / adaptive MPSC | none | CONDITIONAL / STANDARD; not deployed |
| T13 | exact quartic minimum certifies static-sphere clearance during one ZOH hold | exact double-integrator ZOH, static known sphere | extrema occur at endpoints or cubic stationary roots | 74 endpoint-safe/interior-unsafe cases found in 100k | attitude lag; moving obstacle; model error | sampled-data/ZOCBF and trajectory verification | closed-form static sphere specialization | VALID / STANDARD UTILITY |
| T14 | joint HOCBF margin \(\rho\ge0\) iff the pointwise action set is nonempty | compact convex input set; finite affine rows | max-min definition and Sion duality | 100k symbolic/numerical audit | next-step infeasibility | multi-CBF compatibility/support duality | cylindrical UAV input support function | VALID / STANDARD DIAGNOSTIC |
| T15 | \(\rho(x_k)\ge0\Rightarrow\rho(x_{k+1})\ge0\) | none sufficient as stated | disproved | scalar counterexample | explicit next-state loss of feasibility | viability-domain theory | none | FAILED |
| T16 | Candidate A has lower mission energy than standard HOCBF | matched closed-loop task stream only | no theorem; empirical comparison | 3.817% lower mean energy over 334 rollouts | different environments/seeds; fallback frequency | performance-aware CBF | physical telemetry accounting | EMPIRICAL ONLY |
| T17 | B5 history-guided proposals reduce candidates versus random B3 | same recall target and dynamic history advantage | no proof | B3 and B5 both 100% on exact subsets | random already covers tested safe set | safe MPPI / sampling MPC | none | FAILED EMPIRICALLY ON CURRENT SET |

## Proof Audit

1. Every positive deterministic tube claim requires both true-state containment
   and the robust future jerk term.
2. LP feasibility, residuals, and confidence scores are not treated as proof of
   the true motion-bound assumption.
3. Feedback-valid hold safety is separated from recursive feasibility.
4. Full tube inclusion is separated from scalar radius ordering.
5. Sampling probability is separated from deterministic coverage.
6. Candidate-set energy ordering is separated from continuous or mission-global
   energy optimality.
7. Statistical endpoint containment is not used to prove deterministic safety.

## Core-Theorem Status

There is no surviving theorem that is simultaneously valid, deployable, and
non-equivalent to the closest prior. T1--T5 are conditional or standard
supporting results, T5 is a useful negative limit, and the proposed adaptive
certification theorem T6 is false. They do not meet the paper-core novelty gate.
