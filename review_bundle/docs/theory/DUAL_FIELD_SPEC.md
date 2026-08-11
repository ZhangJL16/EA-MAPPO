# Heterogeneous Collision and Recovery-Energy Proposal Fields

Status: coherent training-behavior specification after reframe; proposal core implemented and unit-tested; no trained/calibrated field or benefit claim.

## Claim boundary

The fields may rank several training-time behavior proposals that already lie inside one independently verified Generator support. Evaluation remains the ordinary one-sample affine–tanh Generator-SAC policy. A deployed `K`-sample selector would be an order-statistic/composite policy whose density and entropy objective are not the current SAC derivation, so it is excluded.

The behavior ranker cannot enlarge `C_run`, authorize a command, change `kappa`, modify certificate state or hashes, or run when Generator authority is unavailable. Critic replay uses the selected executed action; actor and temperature updates continue to use fresh samples from the base `pi_theta`. Thus collection uses an off-policy behavior `beta != pi_theta`, but no new SAC convergence theorem is claimed.

## Goal-independent features

With existing environment normalization, define

```text
phi_s(x) = [normalized position (3),
            normalized velocity (3),
            normalized station-minus-position (3),
            normalized lidar distances (32),
            lidar validity mask (32)].                       # 73 dimensions

phi_B(x,a) = [phi_s(x), a/a_max].                            # 76 dimensions
phi_E(x)   = [normalized position, normalized velocity,
              normalized station-minus-position].           # 9 dimensions
```

The first energy model is per-scenario. Cross-map generalization is not claimed without a separate non-certificate map encoder.

Forbidden neural inputs are the task goal/delta, current battery energy, verified return requirement or margin, recovery level/cell ID, certificate validity/hash/version/authority flags, corridor encoding, and `c,G`. The ranker may use `c,G` outside the networks only to map latent proposals to physical actions.

## Distinct estimands and operators

The local action-conditioned collision proposal target is

```text
B*(x,a) = inf_{y in Tube(x,a)} signed_distance(y, obstacles union unknown).
```

The tube includes the current swept segment and successor stopping tube. A cell-level geometry slack is not an adequate sole label because it is mostly action independent. The existing `-1` unsafe sentinel, if used initially, must be named a clipped-margin surrogate rather than a true signed distance.

The global recovery-energy proposal operator is undiscounted:

```text
(T^kappa E)(x) = 0,                                      x in G_ch,
(T^kappa E)(x) = sup_{(d,x+) in Gamma_kappa(x)} d+E(x+), otherwise.
```

The initial synthetic target is the atlas cell's verified `energy_upper`, interpreted as a learned estimate of the outward upper recursion, not the exact covered joint `J_cov^kappa`, current battery, or discounted task return. A Bellman-residual auxiliary loss uses no SAC discount and fixes terminal targets to zero.

Each field returns a mean and positive scale. Held-out empirical calibration defines

```text
L_B = mu_B - q_B s_B,
U_E = mu_E + q_E s_E.
```

These are proposal scores only. Empirical coverage is not certificate evidence.

## Training-only proposal operator

At a collection step with executable Generator authority:

1. Draw `K` latent candidates from the actor or warm-up Gaussian using a dedicated proposal RNG.
2. Map all candidates with the same verified `a_i=c+G tanh(u_i)`.
3. Score collision lower confidence `L_B(x,a_i)`.
4. Form the nominal successor `(p+,v+)` for each candidate and score `U_E(x_i+)`.
5. Retain the top fixed `ceil(K/2)` collision candidates, then select minimum recovery-energy score; break ties by original index.
6. On missing, stale, incompatible, adversarial, or nonfinite predictions, execute candidate zero.
7. When Generator authority is unavailable, bypass the ranker and preserve the existing backup/charger/fail-closed branch.

Shadow proposals must use a separate RNG so they do not advance the main actor stream or alter candidate-zero trajectories.

## Why one scalar coupled field is insufficient

There is no theorem that two separate networks must beat a shared encoder with two heads. The defensible structural claim is narrower: one scalar with monotone readouts cannot generally preserve two unrelated rankings. Three items can be chosen so that the energy ordering is neither the collision ordering nor its reverse; exhaustive scalar orders then violate at least one readout ordering. This motivates distinct outputs, operators, losses, boundary conditions, and calibration—not necessarily distinct encoders.

### Proposition H0 — strict scalar-order loss

Consider three already-certified proposals `a,b,c`. Let collision desirability be strict with

```text
B(a) < B(b) < B(c),
```

and let recovery-energy desirability `R=-E` be strict with

```text
R(b) < R(a) < R(c).
```

There is no scalar representation `h:{a,b,c}->R` and pair of strictly monotone scalar readouts `f_B,f_R:R->R` that reproduce both strict orders.

Proof. A strictly monotone readout either preserves the strict order induced by `h` or reverses it. Consequently, the two readout orders must be identical when `f_B` and `f_R` have the same orientation, or exact reverses when their orientations differ. The order `(b,a,c)` is neither `(a,b,c)` nor its reverse `(c,b,a)`, giving a contradiction. This three-proposal construction proves information loss for the stated one-dimensional monotone bottleneck. It does not rule out a vector representation, a nonmonotone task-specific decoder, or a shared encoder with separate heads. QED.

### Proposition H1 — finite-rank recovery residual accumulation

Let `E*_nu` be the exact solution of the finite selected-DAG rectangular recursion with terminal value zero,

```text
E*_nu = d_bar_nu + max_{mu in Child(nu)} E*_mu
```

at positive-rank nodes. Suppose a learned estimate has terminal error at most `epsilon_0` and uniform absolute recursion residual at most `epsilon`:

```text
|Ehat_nu - (d_bar_nu + max_mu Ehat_mu)| <= epsilon.
```

Then every node satisfies

```text
|Ehat_nu-E*_nu| <= epsilon_0 + r(nu) epsilon.
```

Proof. At rank zero this is the terminal premise. At a positive-rank node, the reverse triangle inequality for maxima gives

```text
|max_mu Ehat_mu - max_mu E*_mu|
    <= max_mu |Ehat_mu-E*_mu|.
```

Add the local residual `epsilon` and apply strong induction, using `r(mu)<=r(nu)-1` for every child. QED.

The bound is a learning-error statement for the rectangular recovery target, not a certificate and not an equality for `J_cov^kappa`. It explains why an error budget for the global recovery estimator depends on remaining recovery depth, whereas a false-positive collision margin can invalidate the very next swept tube.

A one-step recovery residual error can accumulate over the remaining path, while a false-positive local collision margin can fail immediately. These different error mechanisms require separate reported calibration budgets.

## Matched falsification design

Use identical candidate matrices, labels, grouped calibration split, optimizer steps, seeds, online interactions, and `K` for:

1. Candidate zero.
2. `K` computed but candidate zero selected (compute control).
3. Uniform random selection from the same `K`.
4. Collision only.
5. Recovery energy only.
6. Separate heterogeneous networks.
7. Shared physical encoder with separate heads.
8. Parameter-matched rank-one scalar bottleneck.
9. Oracle-label ranking, marked nondeployable.
10. Direct atlas/cell lookup, analytic station-distance, and spatially held-out-cell controls for the recovery-energy target.

Match learned-field parameter counts within 1%. Use blocked spatial splits and at least one held-out map before claiming generalization rather than atlas memorization. Report calibration/miscoverage, pairwise ranking, selected oracle margins/energy, next-state Generator availability, interventions, throughput, stranding, complete charge cycles, and compute. Gradient cosine is diagnostic only.

Drop the separate-network claim if a shared two-head model matches it. Drop dual-field benefit if it does not beat the best single field and compute-matched random selection. If gains vanish at `K=1`, call them multi-proposal exploration gains. If field-off actor evaluation does not improve, report collection improvement only. Reject the integration if any field value reaches a verifier predicate, certificate identity, or non-Generator authority branch.

## Required software gates before behavior-changing use

- feature leakage invariance under goal, battery/margin, mode, corridor, hash, and cell-ID perturbations;
- terminal-zero, finite-chain undiscounted sum, cycle rejection, and accumulated residual tests;
- zonotope containment, adversarial/NaN candidate-zero fallback, authority bypass, and certificate immutability tests;
- shadow RNG invariance, frozen field parameters during SAC updates, evaluation-off default, and certificate-module import-boundary tests;
- matched scalar-ordering counterexample and equal-candidate control tests.

Offline data, toy operators, and true shadow mode are appropriate now. Behavior-changing selection is not paper evidence until the primary learned Generator residual has a matched non-saturated benefit study.
