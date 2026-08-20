# Memory Proof Skeleton

## Dependency DAG

```text
bounded same-track base interval + bounded residual/noise
    -> interval prediction/correction containment (T1)
    -> finite-horizon position orthotope (L2)
    -> exact directional support (L3)
    -> robust directional HOCBF lower bound (T2)
    -> sampled-data hold strengthening (T4)
    -> conditional directional collision separation

uncertainty-set inclusion
    -> support-function monotonicity
    -> safe-action-set inclusion (T3)
    -> pointwise infimum monotonicity (C1)
```

Hidden-state contraction is a separate lemma. It does not imply T1 and is not a
premise of the deterministic interval certificate unless a valid physical
true-minus-nominal residual bound is supplied independently.

## Typed symbols

| Symbol | Type and domain | Meaning |
|---|---|---|
| `track_id` / \(\iota\) | nonempty immutable string/identity | associated physical primitive |
| \(x=[p;v;a]\) | \(\mathbb R^9\) | true obstacle state |
| \(r=[r^p;r^v;r^a]\) | \(\mathbb R_{\ge0}^9\) | componentwise error radius |
| \(d_k\) | \(\mathbb R^3\) | true-minus-nominal jerk residual |
| \(\bar d\) | \(\mathbb R_{\ge0}^3\) | deterministic residual bound |
| \(\bar j_{true}\) | \(\mathbb R_{\ge0}^3\) | deterministic total true jerk bound |
| \(n\) | unit vector in \(\mathbb R^3\) | fixed support direction over one hold |
| \(d\) | finite scalar \(\ge0\) | combined physical clearance radius |
| \(\mathcal E_q\) | nonempty compact subset of \(\mathbb R^3\) | state-error set |
| \(\mathcal U_{act}\) | actuator-feasible subset of \(\mathbb R^3\) | admissible UAV acceleration |

## Canonical quantified claims

### T1 — interval containment

For every sample `k`, if the prior same-track interval contains the exact real
state and the declared componentwise residual and sensor bounds hold, analytic
prediction and correction contain the exact real successor. Dropout is the
special case `K=0`.

### L2 — future tube

For every finite horizon `tau >= 0`, if the current state errors and future
integrable jerk residual obey their bounds almost everywhere, the true position
lies in the displayed orthotope.

### T2 — output-feedback directional safety

For every continuous enforcement interval with fixed unit `n`, exact UAV state,
valid obstacle intervals, `k1,k2>0`, exact nonnegative robust `h,psi1`, and
almost-everywhere tightened `psi2>=0`, the true directional barrier stays
nonnegative in ideal real arithmetic.

### T3/C1 — monotonicity

For fixed nominal state, gains, direction, objective, and actuator set, nested
uncertainty sets induce reverse-nested robust safe-action sets and a
nonincreasing extended-real objective infimum. This is pointwise only.

### T4 — sampled-data hold

For every ZOH hold of length `Delta`, if a finite `L_k` bounds
`abs(dot(psi2))` throughout the hold, enforcing `psi2(t_k) >= L_k Delta`
implies `psi2(t)>=0` throughout that hold.

## Assumption ledger

| Assumption | Used by | Runtime evidence |
|---|---|---|
| same immutable track identity | T1–T4 | explicit `track_id`; mismatch fails closed |
| fresh trusted base sensor/velocity/acceleration bounds | T1 reset | explicit caller certificate with strictly newer measurement epoch; controlled runner uses and audits a finite-horizon velocity envelope, while physical deployment remains an assumption |
| deterministic residual bound | T1/L2 | configured analytic bound; not learned from conformal data |
| total true jerk bound | T4 | separate controlled-scenario bound |
| current timestamp alignment | T2/T4 | propagation every control hold; truth-containment diagnostic |
| exact nonnegative `h,psi1` | T2 | exact runtime predicate, no negative tolerance |
| feasible actuator constraint | T2/T4 | strengthened QP, convergence plus post-solve positive-slack check; fallback is uncertified |
| exact real arithmetic | all analytic claims | assumption only; no outward-rounded implementation |

## Claim boundaries

- Empirical maximum-residual intervals are uncertified baselines, not T1
  certificates.
- Innovation non-violation does not prove association or residual validity.
- Hidden contraction does not prove physical estimation contraction.
- Pointwise QP-value monotonicity does not prove lower mission energy.
- The directional theorem is sufficient, not necessary, for spherical safety.
