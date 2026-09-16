# Derivation Package: Two bottlenecks of resource-enabled experiment enlargement

## Target

Two-page-equivalent theorem abstraction draft: remove battery, robot and route
names; characterize when a nested experiment family preserves its maximal
asymptotic allocation coefficient but strictly improves finite-budget minimax
risk. Do not infer strictness from cheaper bundled KL measurements alone.

## Status

**COHERENT AFTER REFRAMING / EXTRA ASSUMPTION.** Exact strictness and allocation
criteria are proved below. They combine standard finite minimax and LP duality;
the criteria are not claimed as new foundational theorems. The resource-realizable
robust separation remains the candidate scientific contribution.

## Invariant Object

A known feasibility variable selects nested safe experiment systems
\(\mathcal E_1\subseteq\mathcal E_2\) over the SAME finite hypothesis class
\(\Theta\). Old actions, feedback, durations and initial condition embed exactly.
Each hypothesis has the same execution gain \(\rho_\theta\) in both systems.
An experiment may be a committed excursion; a finite-horizon policy may instead
be an arbitrary safe feedback tree in a known finite control protocol.

## Assumptions

Finite actions and feedback alphabets, bounded rewards, positive integer
durations, finite T, common legal safe policy protocol and perfect recall.
Randomized policies are allowed. Known-model comparator equality is imposed
separately; nested experiments alone do not imply it. For the allocation claim,
confusing alternatives are unchanged, KLs are finite, and both programs below
are feasible with finite attained values and LP dual optima. Its interpretation
as an attainable log-regret coefficient uses the previously proved subclass
theorem; it is not automatic for arbitrary controlled sensing.

## Notation

Let \(\mathcal A_j(T)\) be the safe policy class and
\(R_\theta^a=T\rho_\theta-E_\theta^a\sum_{t<T}Y_t\).
For a prior \(\pi\in\Delta(\Theta)\), define

\[
V_j(T,\pi)=\max_{a\in\mathcal A_j(T)}E_\pi^a\sum_{t<T}Y_t,
\quad b_j(\pi)=T\pi\!\cdot\!\rho-V_j(T,\pi),
\quad v_j=\inf_a\max_\theta R_\theta^a.
\]

Write \(g_T(\pi)=V_2(T,\pi)-V_1(T,\pi)\ge0\), and
\(\mathcal L_1=\arg\max_\pi b_1(\pi)\), the old least-favorable prior set.

## Derivation Strategy

Use TWO distinct bottlenecks: an asymptotic KL-price supporting face and a
finite-budget least-favorable-prior face. Neither determines the other.

## Derivation Map

Finite minimax duality -> exact risk-gap identity -> strictness test.
Allocation LP duality -> unchanged-coefficient test. Combine the tests and
instantiate them with the existing certificates, not new games or horizons.

## Main Derivation

### Proposition 1: exact finite-budget strictness

\[
\boxed{v_1-v_2=
\min_{\pi\in\Delta(\Theta)}
\{v_1-b_1(\pi)+g_T(\pi)\}.} \tag{1}
\]

Consequently,

\[
\boxed{v_2<v_1\ \Longleftrightarrow\
g_T(\pi)>0\quad\text{for EVERY }\pi\in\mathcal L_1.} \tag{2}
\]

Proof. Finite histories give finitely many deterministic policy trees;
randomization convexifies their risk vectors. Finite minimax yields
\(v_j=\max_\pi b_j(\pi)\). Since \(b_2=b_1-g_T\), subtracting gives (1).
Both summands are nonnegative and continuous on the compact simplex. Their
minimum is zero exactly when some old least-favorable prior has zero improvement.
Thus a benefit at ONE prior, or at a prior merely used for a lower certificate,
is insufficient. This proof does not require finding a unique least-favorable prior.

For a quantitative sufficient condition, set
\(K_\epsilon=\{\pi:v_1-b_1(\pi)\le\epsilon\}\).
For epsilon>0, if \(g_T\ge\delta>0\) throughout \(K_\epsilon\), then
\(v_1-v_2\ge\min\{\epsilon,\delta\}\).

One usable certificate avoids solving the new optimal learner: for a new
committed experiment e of duration \(\ell_e\le T\), with feedback Z and
posterior \(\pi_Z\), compute

\[
\beta_e(T,\pi)=E_\pi[Y_e+V_1(T-\ell_e,\pi_Z)]-V_1(T,\pi). \tag{3}
\]

Execute e, then an old Bayes continuation: \(g_T\ge\max(0,\beta_e)\).
Positive backups covering all old least-favorable priors imply (2). A finite
adaptive experiment tree can replace e, using its actual stopping duration
tau<=T and \(V_1(T-\tau,\pi_Z)\) inside the expectation. This is a decision-value condition,
not a KL-sum condition; no claim is made that a SINGLE-experiment backup covers
the existing example.

### Proposition 2: exact asymptotic allocation neutrality

For hypothesis theta let \(c_{\theta e}=\rho_\theta\ell_e-E_\theta Y_e\ge0\),
\(D_{\theta v,e}=\mathrm{KL}(Q_{\theta e}\Vert Q_{ve})\), and let
\(\mathrm{Alt}_\theta\) be its unchanged confusing alternatives. Define

\[
C_{\theta j}=\min_{\eta\ge0}\sum_{e\in\mathcal E_j}c_{\theta e}\eta_e
\quad\text{s.t.}\quad
\sum_eD_{\theta v,e}\eta_e\ge1\ (v\in\mathrm{Alt}_\theta).
\]

Its dual maximizes \(\sum_v y_v\) subject to
\(y\ge0,\ \sum_vy_vD_{\theta v,e}\le c_{\theta e}\).
Therefore \(C_{\theta2}=C_{\theta1}\) exactly when SOME old optimal dual price
y remains feasible for every new experiment. Necessity follows by taking a
new dual optimum; sufficiency follows from weak duality and nesting.

For \(C_j^{\max}=\max_\theta C_{\theta j}\),

\[
\boxed{C_2^{\max}=C_1^{\max}\ \Longleftrightarrow\
\exists\theta\in\arg\max_v C_{v1}
\text{ whose old optimal dual price survives all new experiments}.} \tag{4}
\]

Finite Theta and \(C_{\theta2}\le C_{\theta1}\) prove the max-level statement.
This is maximum FIXED-INSTANCE allocation cost, not an interchange-of-limits
claim about asymptotic minimax regret.

### Theorem: asymptotically neutral, finitely active enlargement

Under the assumptions, (4) together with (2) is a necessary-and-sufficient
criterion for \(C_2^{\max}=C_1^{\max}\) AND \(v_2<v_1\) at the specified T.
Execution equivalence is an independent requirement. The new experiments can
be neutral on the asymptotic price bottleneck yet active on every finite-budget
least-favorable-prior bottleneck. Cheaper travel and additive KL do NOT replace
the latter condition.

### Corollary: the existing open resource family

For \(\Theta_z=\{(u,r),(v,r),(v,w)\}\),
\(\|z-(.05,.05,.30,.95)\|_\infty<10^{-4}\), q remains the maximal-C hypothesis.
Its dual puts price \((3r-u)/\mathrm{kl}(u,v)\) on the q-versus-A constraint
and zero on q-versus-B. Both A-only and AB satisfy that price constraint with
equality: B has zero q-versus-A KL and
\(c_{q,AB}=4r-(u+r)=3r-u=c_{q,A}\). Thus (4) is structural, not numeric.

At T=12 the old Bayes lower certificate l and the single randomized adaptive
high witness with worst risk U imply \(v_1\ge l\) and \(b_2(\pi)\le U\) for
EVERY prior. Equation (1) gives \(v_1-v_2\ge l-U\), hence the proved uniform
gap \(>.038509475\) on this box. It also implies (2) without falsely calling
the certificate's chosen prior least-favorable. All old-horizon and terminal
qualifications remain as in the source derivation.

## Remarks and Interpretation

Resource augmentation may preserve the worst asymptotic information PRICE while
improving the finite posterior decision VALUE. Subadditive acquisition cost and
KL additivity do not encode posterior-dependent optimal continuation, horizon
fit, or coverage of the minimax bottleneck. The relevant new contribution, if
novelty survives review, is their robust resource-realizable separation.

## Boundaries and Non-Claims

This abstraction fits controlled sensing; it does not prove formulation
irreducibility. Equations (1)-(4) use standard minimax/Bellman/LP duality and
must not be sold as a new universal learning principle. It is not a graph-only
local criterion, an efficient general solver, or a result at T=4096. Physical
realizability and exact execution equality still require separate route proofs.
No new experiments, parameters sampled, horizons enumerated, or learner changes.

## Open Risks

Independent proof audit and priority of the FULL separation remain open.
Established neighbors: [structured allocation / OSSB](https://arxiv.org/abs/1711.00400),
[controlled sensing](https://arxiv.org/abs/1205.0858), and
[finite-horizon sensing/exploitation](https://arxiv.org/abs/1705.05960).
The abstract duality criteria alone do not clear that overlap. See the existing
prior-art audit and `ROBUST_RESOURCE_BUNDLING_SEPARATION_20260916.md`.

Verification scope: the 9 existing anchor/robustness mathematical tests pass;
frozen benchmark source hashes are unchanged. The new abstract equivalences
have the handwritten proofs above, not a new machine-verified theorem claim.
