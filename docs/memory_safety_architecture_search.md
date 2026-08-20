# Memory Safety Architecture Search

## Selection rule

An architecture is retained only if it improves a valid uncertainty-control chain, not merely prediction error. Every neural nominal is evaluated against classical observers and against an exact-state oracle controller.

| ID | Architecture | Explicit physical state | Certified uncertainty | Theory status | Current decision |
|---|---|---|---|---|---|
| A | Current measurement only | position only | fixed base bound | valid but broad | keep baseline; no novelty |
| B | Raw L16 MLP | none | conformal only | no feedback-valid deterministic bound; MAE 0.8985 | reject |
| C | Ego-compensated L16 MLP | hand-built motion geometry | conformal only | best estimator (MAE 0.5303), but radial identifiability only | keep engineering baseline; reject theory |
| D | Vanilla RNN | latent | conformal only | MAE 0.5964, weaker than C | reject |
| E | GRU | latent | conformal only | MAE 0.5982, q95 undercoverage 0.949 | reject |
| F | LSTM | latent | conformal only | MAE 0.5895, weaker than C | reject |
| G | TCN | finite latent history | conformal only | MAE 0.6165, weaker than C | reject |
| H | State-space model | latent dynamical state | conformal only | MAE 0.9195 and wide q95 tube 25.5142 | reject |
| I | Physics-GRU | position/velocity/acceleration nominal | conformal only | MAE 0.6683; KalmanNet/observer overlap | reject |
| J | Physics-GRU + analytic tube | explicit nominal state | analytic interval | adding a nominal cannot shrink a fixed independently declared residual bound | reject: no certified tightening |
| K | J + innovation reset | J plus mode/reset state | analytic interval/base reset | sound only with a fresh independent base certificate; standard hybrid fallback | reject novelty; retain safety rule |
| L | Explicit physical memory + small recurrent residual | full explicit state and diagnostics | analytic orthotope | interpretable, but analytic observer freezes in controlled loop and recurrent center gives no certified radius reduction | reject contribution |
| M | Contractive L | same as L | analytic orthotope | MAE 0.8362; hidden contraction does not imply physical contraction | reject |

## Hidden-size ablation

Artifact: `artifacts/memory_hidden_size_ablation_5k_20260820_001651/summary.json`.
The same 5,000 matched trajectories were reused; no new training-data budget
was introduced.

| Family | H=16 | H=32 | H=64 | H=128 | H=128 P99 inference |
|---|---:|---:|---:|---:|---:|
| Physics-GRU joint MAE | 0.8269 | 0.6734 | 0.6059 | 0.5892 | 0.0051 ms |
| Contractive memory joint MAE | 0.8905 | 0.8393 | 0.7997 | 0.7181 | 0.6527 ms |

Scaling the recurrent model does not overturn the selection result. The best
Physics-GRU remains worse than ego L16 MLP (`0.5303`), and the contractive
family remains worse still.

## Tracking and delay diagnostics

Artifact:
`artifacts/memory_counterexamples_tracking_delay_200k_20260820_002409/summary.json`.

- Hungarian tracking was exercised for 2/4/8/16/32 obstacles. Easy random
  scenes achieved identity accuracy `1.0`, but this is not a certificate and
  does not override the crossing identity-swap counterexample.
- P99 assignment latency was `0.0029/0.0018/0.0024/0.0042/0.0136` ms and the
  numeric per-track memory footprint was `352/704/1408/2816/5632` bytes.
- A `0.3 s` delayed measurement at velocity `[25,-8,3] m/s` is not contained
  by a naive `0.1 m` radius. Timestamp propagation or delay inflation is a
  necessary assumption, not an optional implementation detail.

## Classical baselines

- Constant-velocity Kalman filter.
- Constant-acceleration Kalman filter.
- EKF, which is algebraically identical to the linear KF on the benchmark's linear position-measurement model and must be labelled as such.
- IMM over constant-velocity and constant-acceleration modes.
- Sliding-window polynomial MHE.

## Pre-registered rejection criteria

Reject a learned candidate if any of the following occurs:

- lower MAE but invalid interval/tube coverage;
- interval width is no tighter than the physical observer or IMM at matched coverage;
- abrupt-change under-bound persists beyond the declared reset delay;
- collision, intersample violation, or uncertified fallback is worse than the classical robust observer;
- P99 end-to-end compute exceeds 50 ms;
- the claimed theorem reduces to an existing observer-CBF or neural-tube theorem without a stronger conclusion.

## Architecture-level negative results already established

1. Replacing an MLP with GRU/LSTM/TCN changes prediction machinery but not the certificate object.
2. Contracting hidden dynamics only bound sensitivity to hidden initialization/input perturbations. They do not bound the unknown physical-model residual.
3. A neural uncertainty head is not a deterministic certificate. It requires distributional calibration and yields only assumption-conditional probabilistic coverage.
4. Reset logic does not recover information that LiDAR never observed. Unobservable tangential components must retain a worst-case bound.
5. A recurrent nominal can change the tube center but, without an independently
   smaller bound on true-minus-nominal jerk, it cannot shrink the deterministic
   tube radius or enlarge the certified safe-action set.
6. IMM+GRU, set-valued recurrent, and anisotropic-ellipsoid variants change the
   estimator or set representation but remain observer-error propagation plus
   observer/measurement-robust CBF. They do not create a different guarantee.

## Search closure

All pre-registered families A–M are now either empirical baselines or rejected
theory candidates. UKF is not separately run because the benchmark dynamics and
position observation are linear, making its nominal filtering object equivalent
to the linear Kalman baseline for this protocol. Anisotropic ellipsoids are
mathematically useful but their support-function tightening is the same known
set-inclusion object as the tested orthotope. No family reaches the theory gate.
