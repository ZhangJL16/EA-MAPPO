# Memory Uncertainty Tube Derivation

## Target

Map the observer's current componentwise state interval to a future obstacle-position tube without using a neural uncertainty head.

## Assumptions

- Current position, velocity, and acceleration errors satisfy componentwise bounds \(r^p,r^v,r^a\).
- Future true-minus-nominal jerk is measurable and integrable and satisfies
  \(|d(\tau)|\preceq\bar d\) almost everywhere.
- The same obstacle association remains valid over the prediction segment.

## Finite-horizon tube

For horizon \(\tau\ge0\), repeated integration and the triangle inequality give

\[
|p(\tau)-\hat p(\tau)|
\preceq r^p+\tau r^v+\frac12\tau^2r^a+\frac16\tau^3\bar d
=:r^p(\tau).
\]

The tube is the orthotope

\[
\mathcal E_p(\tau)=\{e:|e|\preceq r^p(\tau)\}.
\]

Each component is nondecreasing in \(\tau\) for nonnegative radii and jerk bounds. Consecutive dropout frames therefore widen the tube automatically.

## Directional support

For any direction \(n\), the orthotope support is

\[
\sigma_{\mathcal E_p}(n)=\max_{e\in\mathcal E_p}n^Te=|n|^Tr^p.
\]

The smallest origin-centered Euclidean ball containing the same orthotope has radius \(R=\|r^p\|_2\). For unit \(n\), Cauchy-Schwarz gives

\[
|n|^Tr^p\le\|r^p\|_2=R.
\]

For nonzero \(r^p\), equality holds only when \(|n|\) is proportional to \(r^p\); when \(r^p=0\), both sides are zero for every unit direction. Thus preserving anisotropy never requires more directional clearance than replacing the interval by its enclosing ball, and is strictly tighter in generic nondegenerate directions.

## Calibration boundary

The deterministic tube is valid only when the residual bound is physically
valid. A split-conformal residual may replace \(\bar d\), but the resulting claim
is marginal, finite-sample, and exchangeability-conditional. It is not a
deterministic safety guarantee. The displayed derivation is in ideal real
arithmetic; the current implementation does not add outward-rounding error.

For the controlled learned/Kalman/IMM baselines, each coordinate radius is the
maximum absolute residual on the full development validation split. That split
was also used for checkpoint selection, so this is deliberately **not** called
split conformal, model-selection-independent, or held-out calibration. The
held-out test split is not used to configure the closed-loop radii. The boxes
are descriptive development envelopes only; no marginal, conditional,
repeated-time, distribution-free, or deterministic safety guarantee is attached
to them.
