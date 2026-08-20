# Explicit Memory Observer Theory

## Status

CONDITIONALLY SOUND IN IDEAL REAL ARITHMETIC; NOT A FLOATING-POINT OR PHYSICAL-DEPLOYMENT CERTIFICATE.

The sound object is not a “certified GRU”. It is an analytic interval observer whose nominal jerk may be supplied by a recurrent network.

## Explicit memory state

For one explicitly identified obstacle track with immutable identity \(\iota\), the
aggregate observer state (the public interval state, recurrent hidden state, and
step diagnostics together) is

\[
m_k=(\iota,\hat p_k,\hat v_k,\hat a_k,r^p_k,r^v_k,r^a_k,h_k,\nu_k,c_k,\ell_k,\Lambda_k),
\]

where the first six motion terms are nominal motion and componentwise interval
radii, \(h_k\) is recurrent residual memory, \(\nu_k\) is innovation, \(c_k\)
records consistency/reset mode, and \(\ell_k\) is the count of prediction-only
holds since the most recent measurement. The map \(\Lambda_k(\iota)\) stores
the greatest accepted certificate epoch for each track and is certification
metadata rather than a physical state. The implementation stores these fields
across `IntervalObserverState`, `ObserverStep`, and `PhysicsMemoryObserver`; no
latent variable is itself treated as a certificate.

## Physical model

Let \(x_k=[p_k;v_k;a_k]\in\mathbb R^9\). With sample period \(\Delta\),

\[
x_{k+1}=Ax_k+G(\hat j_k+d_k),\qquad y_k=Cx_k+n_k,
\]

where \(\hat j_k\) is the recurrent nominal jerk, \(|d_k|\preceq \bar d_k\), and \(|n_k|\preceq\bar n\). The matrices are

\[
A=\begin{bmatrix}I&\Delta I&\frac12\Delta^2I\\0&I&\Delta I\\0&0&I\end{bmatrix},\quad
G=\begin{bmatrix}\frac16\Delta^3I\\\frac12\Delta^2I\\\Delta I\end{bmatrix},\quad
C=[I\;0\;0].
\]

## Recurrent nominal

The real-valued parameterization assumes \(0<\eta\le1\) and
\(\|W_h\|_2\le\rho<1\), and is

\[
h_{k+1}=(1-\eta)h_k+\eta\tanh(W_hh_k+W_z\phi_k+b),
\]

\[
\hat j_k=j_{\max}\tanh(W_oh_{k+1}+b_o),
\]

with \(\|W_h\|_2\le\rho<1\). The feature vector may contain innovation, ego-compensated range flow, prior safe action, and track consistency. It does not contain a `safe` label.

## Lemma 1: hidden-state incremental contraction

For equal input \(\phi_k\),

\[
\|h_{k+1}-h'_{k+1}\|_2
\le [(1-\eta)+\eta\rho]\|h_k-h'_k\|_2.
\]

Because tanh is 1-Lipschitz and \(\rho<1\), the coefficient is below one. The
implementation supports only FP32/FP64 for this contraction claim and rejects
parameters whose real-valued factor is closer than \(10^{-6}\) to one. This
lemma is not a physical observer-error theorem, and no FP16/INT8 claim is made.

## Analytic prediction and correction

If \(|x_k-\hat x_k|\preceq r_k\), define

\[
\hat x^-_{k+1}=A\hat x_k+G\hat j_k,
\qquad
r^-_{k+1}=|A|r_k+|G|\bar d_k.
\]

For a measurement update with gain \(K\),

\[
\hat x^+_{k+1}=\hat x^-_{k+1}+K(y_{k+1}-C\hat x^-_{k+1}),
\]

\[
r^+_{k+1}=|I-KC|r^-_{k+1}+|K|\bar n.
\]

For dropout, use \(K=0\), so the radius grows only by physical propagation.

## Theorem 1: interval observer containment

Assume the prior interval contains the true state, the true-minus-nominal jerk
residual and sensor noise satisfy the declared componentwise bounds, and the
measurement carries the same immutable track identity. Then both the prediction
and correction intervals contain the true state in ideal real arithmetic.

### Proof

Prediction error is

\[
e^-_{k+1}=Ae_k+Gd_k.
\]

Taking componentwise absolute values and applying the triangle inequality gives

\[
|e^-_{k+1}|\preceq|A|r_k+|G|\bar d_k=r^-_{k+1}.
\]

Correction error is

\[
e^+_{k+1}=(I-KC)e^-_{k+1}-Kn_{k+1}.
\]

The same componentwise argument gives

\[
|e^+_{k+1}|\preceq|I-KC|r^-_{k+1}+|K|\bar n=r^+_{k+1}.
\]

Thus containment is invariant by induction. \(\square\)

## Reset proposition

If innovation violates the predicted measurement interval, at least one premise
in the conjunction of prior containment, residual bound, sensor bound, and
same-track association has failed. The test does not identify which premise
failed. Non-violation proves none of those premises. Resetting to center
\((y,0,0)\) and radii
\((\bar n,\bar v_{base},\bar a_{base})\) restores containment only when the
caller supplies a **fresh** `TrustedBaseIntervalCertificate` bound to the exact track,
attesting bounded measurement noise and the physical base velocity and
acceleration bounds. Freshness is represented by a strictly increasing
measurement epoch. The implementation maintains the greatest accepted epoch
for each track even while the public interval is invalid or another track is
observed; a certificate whose epoch is no newer than that per-track ledger is
rejected. An invalid reset therefore cannot erase freshness history and replay
an older certificate. The object records these premises; it does not
empirically prove them.

In the controlled benchmark, the reset velocity radius is not the previously
assumed 12 m/s. It is a componentwise finite-horizon envelope

\[
\bar v_{base}=|v_0|+\bar a_{scenario}N\Delta,
\]

with \(\bar a_{scenario}=6\,\mathrm{m/s^2}\), rollout length \(N\), and hold
\(\Delta\). This broad envelope is mechanically checked against every simulated
state. It is intentionally conservative; its operational infeasibility is an
experimental result rather than a hidden premise violation.

An association-uncertain, missing-ID, or mismatched-ID measurement is therefore
never exposed as a deterministic same-track interval. Certification can resume
only from a fresh matching trusted base certificate. Certified innovation checks
use zero mathematical tolerance: a numerical grey zone is not silently called
contained.

## Numerical boundary

The proofs use exact real arithmetic. The NumPy implementation does not perform
outward-rounded interval arithmetic and therefore is a numerical realization of
the theorem, not a formal floating-point certificate. Extremely large states can
lose sub-ULP motion. All closed-loop claims below remain conditional on arithmetic
error being negligible relative to declared margins; no real-UAV physical safety
claim is made.

## Observability boundary

A single radial LiDAR range derivative constrains only the radial projection of relative velocity. Tangential velocity is unidentifiable without temporal angular geometry, primitive tracking, or an additional model. A valid tube must therefore preserve a nonzero tangential radius. No recurrent architecture can remove this information-theoretic limitation from the same observations.
