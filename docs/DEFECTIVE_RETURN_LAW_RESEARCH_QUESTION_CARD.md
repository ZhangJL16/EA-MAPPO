# Research Question Card: Defective return law under executed-interface shift

Question: Can a learned model of the **defective resource-to-safe-recharge distribution** remain decision-reliable when a nominal policy is composed with a safety filter, whereas ordinary finite mean-energy prediction cannot represent the induced failure mass?

Type: exploratory mechanism study leading to a confirmatory method study.

Hypothesis: For a fixed physical state \(x\), executed interface \(I\), and future execution disturbance \(\omega\), the stopped variable

\[
Y_I(x,\omega)=
\begin{cases}
E_I(x,\omega),&S_I=1,\\
+\infty,&S_I=0,
\end{cases}
\qquad
S_I=\mathbf1\{\tau_C\le H,\ N_{\rm contact}=0\}.
\]

has two separately learnable components: structural safe-return probability \(p_I(x)=P(S_I=1\mid x)\) and the finite conditional CDF \(F_I(b\mid x,S_I=1)\). The feasibility probability is exactly

\[
P(Y_I\le b\mid x)=p_I(x)F_I(b\mid x,S_I=1).
\]

A success-only or finite-capped MSE estimate of \(E[Y\mid x]\) generally cannot recover this probability because it either conditions away \(S=0\) or assigns an arbitrary finite cap to an infinite failure atom. Interface composition can change both factors.

Why it matters: Return decisions are irreversible. Dangerous underestimation strands the agent, while conservative overestimation abandons task utility. The completed deterministic study shows HOCBF improves collision-free return but simultaneously reduces arrival and increases energy; a scalar “expected energy saved” score conflates those effects.

Current evidence:

- ER-20260906-interface-return-01 — 150-scene/310-anchor paired simulator experiment. HOCBF-minus-raw: zero-contact arrival +13.17 pp [8.67,18.00], arrival −3.50 pp [−6.33,−1.00], energy +3.58 [1.10,6.46]. Development evidence, one SAC seed.
- ER-20260905-dual-viability-01 — 3,410 old-interface rollouts. Returnability was predominantly state-dependent under the frozen candidate set; direct action-critic route not supported.
- Independent-review literature synthesis — Recovery RL, Back to Base, consumption MDPs, RC-PPO/RAPCPO and conformal/OPE methods occupy the broad primitives. The plausible gap is the executed-interface killed resource law plus irreversible boundary decision, not safe return alone.

Missing evidence:

- A physically declared future disturbance law producing repeated outcomes at fixed anchor/interface.
- Nondegenerate within-anchor variation in both structural success and finite energy.
- Scene-held-out comparison of mean-only, binary-only, and factorized defective-law predictors.
- Fresh-scene confirmation and multiple policy/filter seeds or families.
- A theorem that yields an algorithm/data-collection consequence beyond the identity \(P(Y\le b)=pF\).

What would support it:

- Meaningful within-anchor outcome variation under keyed physical disturbances.
- Mean-only prediction has acceptable finite-energy MAE but materially worse budget Brier/dangerous false-safe error than the factorized model.
- Explicit interface conditioning or executed-action features reduce composition-shift error on leave-one-interface/pair evaluation.
- A non-identifiability construction shows finite-success energy samples can agree while failure mass—and therefore return decisions—differs.

What would falsify it:

- Repeated physical disturbances leave almost every anchor/interface outcome deterministic and finite-energy variation negligible.
- A properly calibrated simple mean/capped baseline matches the factorized model on decision risk and risk–coverage.
- Interface effects disappear on fresh scenes or under another policy/filter pair.
- The theoretical result reduces to a tautological factorization without changing estimators, data support conditions, or decision guarantees.

Minimal next action: collect repeated matched raw/HOCBF return rollouts under one predeclared bounded, temporally correlated post-controller actuation-error law. Preserve anchor/scene clustering and disturbance-key pairing. Do not fit or promote a predictor until the user requests analysis after collection.

Decision: run experiment.
