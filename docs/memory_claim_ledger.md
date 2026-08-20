# Memory Claim Ledger

| Claim | Status | Authoritative evidence | Boundary |
|---|---|---|---|
| The implemented hidden update is contractive for fixed inputs in FP32/FP64 | SUPPORTED | analytic Lipschitz construction and regression tests | Hidden state only; not physical estimation error. |
| The analytic observer preserves interval containment | SUPPORTED CONDITIONAL | theorem, 100k bounded-state slice, zero under-bounds | Correct association, fresh trusted base, declared sensor/residual bounds, ideal arithmetic. |
| Dropout widens the set by physical propagation | SUPPORTED | radii .1000/.1545/.2260/.3265/.6625 at 0/1/2/3/5 misses | Width can rapidly become unusable. |
| Reset restores containment | SUPPORTED CONDITIONAL | fresh per-track epoch rule and tests | Only under a newer independent same-track base certificate. Unconditional reset containment in the adversarial slice is .07777. |
| Future orthotope contains the obstacle trajectory | SUPPORTED CONDITIONAL | exact constant-jerk integration derivation | Requires valid current state and future residual bounds. |
| Directional orthotope support is no wider than its enclosing sphere | SUPPORTED | support-function/Cauchy-Schwarz proof | Same set and fixed support direction. |
| Robust directional HOCBF covers all interval corners | SUPPORTED CONDITIONAL | theorem, corner tests, 100k random HOCBF draws | Valid tube, fixed direction, regularity and actuator assumptions. |
| The sampled-data hold margin preserves the directional condition | SUPPORTED CONDITIONAL | jerk-aware margin, exact-clearance and randomized hold tests | Valid jerk/state/input bounds and positive post-solve reserve. |
| Smaller uncertainty expands the robust safe-action set | SUPPORTED | set-support monotonicity | Fixed nominal state, direction and actuator set. |
| Smaller uncertainty cannot increase the same-QP optimum | SUPPORTED | feasible-set inclusion | Pointwise result only; no mission-energy theorem. |
| A recurrent model is needed for best estimation | CONTRADICTED | ego L16 MLP MAE .5303; best H128 Physics-GRU .5892 | Dataset-specific negative result. |
| A recurrent tube is tighter at matched coverage | NOT SUPPORTED | ego L16 q95 width 8.3024; GRU 9.3619; Physics-GRU 9.9987; contractive 10.2082 | Marginal 1 s held-out coverage, not repeated-time safety. |
| Learned boxes provide deterministic feedback-valid certification | CONTRADICTED | all learned/Kalman/IMM closed-loop rows have uncertified rate 1.0 and nonzero under-bounds | Development max-residual boxes only; validation is reused for checkpoint selection and no coverage claim is made. |
| The deterministic interval gives useful closed-loop control | CONTRADICTED | analytic observer: success 0, fallback 1, freeze 1 | It contains truth in the bounded slice but is too broad operationally. |
| Candidate A produced a fully certified collision-free trace | NOT SUPPORTED | zero collisions but 2.5204% infeasible/uncertified fallback | Oracle-state controlled simulation only. |
| Memory reduces intervention at equal deterministic safety | NOT SUPPORTED | no learned candidate has a valid deterministic box; analytic candidate freezes | Uncertified intervention differences are not a same-safety comparison. |
| Memory lowers mission energy at equal deterministic safety | NOT SUPPORTED | same mismatch; success-conditioned energy is not causal | Pointwise QP monotonicity does not imply mission energy. |
| Multi-obstacle association is solved | NOT SUPPORTED | easy random Hungarian accuracy 1.0, but 0.08 m identity swap passes the innovation gate | Association ambiguity must inflate/union/fallback. |
| The observer/tube/HOCBF chain is a novel theorem contribution | CONTRADICTED | hostile prior-art audit, novelty 9/30 | Matias–Silvestre 2026 and prior sampled-data/observer robust CBF work cover the theorem-level object. |
| The route is ready for formal 500k | CONTRADICTED | theory gate and empirical consequence gate both fail | Formal 500k was not started. |
