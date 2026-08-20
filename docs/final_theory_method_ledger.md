# Final Theory Method Ledger

## Decision Rule

A candidate enters the paper-core set only if it scores at least 22/30 across
mathematical object, guarantee, closest-prior difference, failure relevance,
falsifiability, and 20 Hz deployability. Correct but standard lemmas remain in
the engineering foundation and are not renamed as novelty.

| ID | Idea | Mathematical object | Intended claim | Proof status | Closest prior | Counterexample / failure | Experiment | Compute | Score | Decision | Reason |
|---|---|---|---|---|---|---|---|---|---:|---|---|
| M1 | Candidate A: energy-aware sampled-data HOCBF-QP | pointwise HOCBF safe-action polytope with energy-aware objective | collision avoidance with lower pointwise action energy | conditional HOCBF proof; no recursive-feasibility proof | HOCBF, performance-aware CBF-QP | 119 infeasible/uncertified fallback steps on 334 hard rollouts | 334 rollouts, 0 collision, 99.70% success, 3.817% less energy than standard HOCBF | worst-rollout P99 53.78 ms | 19 | KEEP ENGINEERING BASELINE | strongest current closed-loop system, but not new theory and fallback is uncertified |
| M2 | Joint feasibility margin \(\rho\) | max-min HOCBF slack and support-function dual | exact online nonemptiness and recursive-feasibility precursor | exact pointwise theorem; recursive implication false | multi-CBF compatibility, viability-domain methods | explicit scalar state where \(\rho_k\ge0\) and \(\rho_{k+1}<0\) | 100k adversarial states | small convex problem | 16 | KEEP DIAGNOSTIC | useful infeasibility witness, not a new invariant object |
| M3 | Exact one-hold sphere verifier | quartic minimum over one ZOH hold | exact inter-sample safety of a selected action | VALID under exact double-integrator/static-sphere assumptions | sampled-data CBF, ZOCBF, continuous trajectory verification | invalid under unbounded model/perception error | 74 endpoint-safe/interior-unsafe cases in 100k | O(1) roots per obstacle | 20 | KEEP CERTIFICATE UTILITY | specialized exact verifier, no recursively feasible synthesis theorem |
| M4 | Fixed long history | fixed-L16 ego-compensated motion estimate | longer history improves prediction and proposals | FAILED under regime change | temporal filtering, learned trajectory predictors | abrupt L16 MAE 1.8471 versus L4 0.6877 | 10k steady + 10k abrupt | low | 14 | REJECT | stale history is harmful |
| M5 | Adaptive set-membership history tube | projected bounded-jerk feasible kinematic set \(\mathcal X_t^{(L_t)}\) | consistent history contracts current uncertainty; robust propagation gives feedback-valid hold safety | FAILED as an observation-only adaptive certificate; conditional fixed-model lemmas remain valid | adaptive MPSC, set-membership filtering, adaptive tube MPC | a false narrow-model latent history can remain feasible while the true current state lies outside the projected box | V6: abrupt certified containment 5% / 29% / 24% despite 100% formula-only endpoint hits | one obstacle P99 35.0--38.9 ms | 15 | KEEP DIAGNOSTIC UTILITY / REJECT THEORY | LP feasibility does not verify the true motion-bound assumption; positive remainder is standard conditional reachability |
| M6 | Confidence-gated fast/robust hybrid | history-consistency mode and robust base mode | stale history automatically falls back without losing safety | FAILED for the proposed feasibility gate | SODA-MPC, switched safe control, reachability fallback | abrupt true trajectories violate the fast bound while another fast latent trajectory keeps the LP feasible; future change is also invisible until observed | V6 robust reset rate 0% in all abrupt regimes; certified-under-robust-model rate 0% | monitor small; estimator dominates | 13 | REJECT | neither safe mode activation nor robust reset is certified from observations |
| M7 | History-local trajectory search region | ball around history-derived primitive parameter | optimal/safe primitive lies in a smaller local region | FAILED: no containment premise derived | SM-NMPC search-domain reduction, parametric optimization sensitivity | optimal path can switch discontinuously across corridor homotopy classes | no empirical advantage over random B3 | n/a | 13 | REJECT | history of motion does not identify the task-optimal control parameter |
| M8 | Deterministic motion-primitive cover | \(\epsilon\)-net of low-dimensional primitive parameters | safe candidate existence and finite resolution | CONDITIONAL and standard | resolution-complete/optimal kinodynamic motion primitives | narrow corridor requires clearance and Lipschitz assumptions; cover size explodes with dimension | B3 random already matches B5 recall on exact subset | exponential in primitive dimension | 18 | REJECT THEORY | established completeness/resolution theory; no better construction found |
| M9 | Coarse-to-fine exact trajectory sampling | 1k proposals, top-20 exact verification | retain safe candidate under 50 ms | empirical only; no deterministic cover | SC/GS/DualGuard/BR-MPPI, safe sampling MPC | 28/100k coarse safe-candidate misses; no deterministic coverage | 99.9715% coarse recall; exact subsets 100%; P99 33.80/46.19 ms | 20 Hz on bounded subsets | 20 | KEEP ENGINEERING | strong real-time implementation, history does not beat random |
| M10 | Terminal recoverability | terminal backup-invariant set and shifted sequence | recursive feasibility | standard theorem valid if a concrete invariant set exists; construction missing | robust MPC terminal sets, backup CBF, adaptive MPSC, Safe Beyond Horizon | finite-horizon safe trajectory can end in a dead end | interface only | unknown | 15 | REJECT CURRENT CANDIDATE | strongest unresolved safety gap, but no new computable terminal set was found |
| M11 | Maximum certified progress | \(\gamma^*(x)=\max\) progress over certified candidates | avoid arbitrary fixed CLF rate and freeze | definition-level optimality only | lexicographic MPC, reference governors, progress-constrained safe control | \(\gamma^*=0\) in traps and candidate-set dependence | freeze not measured closed-loop for B5 | extra selection only | 14 | REJECT THEORY | does not guarantee non-freezing or reachability |
| M12 | Candidate-cover energy bound | Lipschitz cost over an \(\epsilon\)-cover | \(J(U_Q^*)-J(U^*)\le L_J\epsilon\) | VALID conditional lemma | resolution-optimal motion planning, generic Lipschitz covering | telemetry cost/feasible optimizer can be nonsmooth at clipping and active-set changes | candidate-set energy ranking implemented; no global comparison | tied to cover explosion | 16 | REJECT THEORY | standard approximation argument and assumptions not verified globally |
| M13 | History-only uncertainty lower bound | observationally equivalent bounded-jerk futures | no history-measurable predictor can beat the global future jerk term \(\bar j\tau^3/6\) | VALID | robust-control indistinguishability/worst-case reasoning | not a positive control method; acceleration branch requires a separate second-order model | analytic unit checks | O(1) | 18 | KEEP CLAIM BOUNDARY | prevents false safety claims but is insufficient as standalone paper core |

## Surviving Engineering Stack

```text
physical motion-vector preprocessing
-> optional bounded set-membership state estimate
-> coarse-to-fine candidate generation
-> true clipped-physics rollout
-> exact continuous-time hold verifier
-> lexicographic progress then energy ranking
-> Candidate A as current fallback baseline
```

The stack is scientifically defensible only with explicit boundaries:

- finite-horizon trajectory safety, not recursive feasibility;
- candidate-set energy ordering, not mission-global optimality;
- empirical safe-candidate recall, not deterministic coverage;
- independently valid present-state and future-motion bounds, not LP-feasibility
  confidence;
- uncertified fallback remains nonzero in Candidate A hard scenarios.

## Strongest Unresolved Mathematical Gap

The unresolved object is a computable multi-obstacle controlled-invariant or
recoverable subset for the actual sampled 3D dynamics and perception model that
admits a certified action at every cycle. Standard viability, reachability,
backup-CBF, and terminal-MPC families solve this class in principle. No
non-equivalent closed-form or lower-complexity construction has been found for
this repository.

## Final Method Decision

No candidate reaches 22/30. The highest-value retained method remains Candidate
A as an engineering baseline, augmented by exact hold verification and optional
set-membership motion estimation. This is not a defensible new theory method.
