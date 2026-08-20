# Controlled Experiments: History-Conditioned Safe Trajectories

## Scope and provenance

No 500k formal run was started. The new controlled artifact is:

- artifacts/history_safe_trajectory_100k_20260819_v3/summary.json
- 100,000 initial states;
- 1,000 coarse candidate sequences per state;
- 100,000,000 total coarse sequences;
- 50 ordinary exact-reference states with 1,000 candidates each;
- 5 hard exact-reference states with 10,000 candidates each;
- horizon 5 and 20 exact survivors;
- trajectory-code SHA-256 9010c2a187093d43b328b761c1dde9a87a3530d25718902fc1420244833fc456;
- GPU NVIDIA GeForce RTX 5060.

The causal history/motion diagnostic is artifacts/history_motion_ablation_10k_20260819/summary.json with 10,000 cases per history-length/regime, lengths 2/4/8/16, constant-velocity and abrupt-change regimes, 0.10 m range noise, and 5% dropout.

The benchmark uses synthetic static-sphere states. It does not claim closed-loop task performance for B3–B5.

## Baselines already established

The strongest same-stream closed-loop evidence remains artifacts/uav_theory_adversarial_1002_20260819/summary.json:

| Method | Rollouts | Success | Collision rollouts | Unsafe fallback steps | Path ratio | Mean energy | Energy vs standard | Worst-rollout P99 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B1 sampled-data HOCBF | 334 | 99.70% | 0 | 305 | 1.0360 | 4.8466 | reference | 56.46 ms |
| B2 Candidate A energy-aware HOCBF | 334 | 99.70% | 0 | 119 | 1.0383 | 4.6659 | -3.817% | 53.78 ms |
| Rejected lexicographic pointwise route | 334 | 20.66% | 0 | 15 | 3.4560 | 15.9561 | +235.74% | 145.24 ms |

B0 frozen SAC no filter was measured on a separate 125-rollout stream: success 100%, collision-rollout rate 91.2%, path ratio 1.0001, and energy 3.7430. It is not used for same-stream energy deltas against the 334-rollout table.

## 100k coarse survey

| Metric | Result |
|---|---:|
| Initial states | 100,000 |
| Candidates per state | 1,000 |
| Total coarse sequences | 100,000,000 |
| States with endpoint-safe progress candidate | 98,196 |
| States retained by top-20 pipeline | 98,168 |
| State-level coarse safe recall | 99.9715% |
| Missed positive states | 28 |
| Mean safe candidate fraction | 92.1648% |
| Mean first-action diversity | 1.1854 |
| Mean amortized coarse latency/state | 0.7817 ms |
| P95 amortized coarse latency/state | 0.8332 ms |
| P99 amortized coarse latency/state | 0.8745 ms |

This survey uses endpoint-safe progress as a cheap reference, not the continuous-time quartic certificate. Exact claims come from the smaller reference subsets.

## Exact-reference comparison

### Ordinary subset: 50 states, 1,000 candidates/state

| Method | Definition | Safe-state recall | Compute |
|---|---|---:|---:|
| B3 safe-MPPI-style random exact | Random SAC perturbations, exact certificate | 100% of 49 positive states | approximately full exact cost |
| B4 mixture exact all | P1–P8 mixture, exact all 1,000 | 100% by reference definition | mean 1,435.76 ms |
| B5 mixture coarse-to-fine | Mixture, top 20, true physics, exact certificate | 100% of 49 positive states | mean 31.20 ms |

B5 produced 49 certified selections and one uncertified fallback. Mean survivors were 19.66 certified and 18.08 certified-progress trajectories. Candidate-level pruning removed 95.18% of safe candidates, but state-level false pruning was 0/49. Keeping one safe candidate does not preserve the complete safe set.

### Hard subset: 5 states, 10,000 candidates/state

| Metric | Result |
|---|---:|
| Reference-positive states | 5/5 |
| B3 random-exact recall | 100% |
| B4 mixture-exact recall | 100% |
| B5 coarse-to-fine recall | 100% |
| B5 fallback | 0/5 |
| Mean certified survivors | 20.0 |
| Mean progress survivors | 16.4 |
| B4 exact-all mean | 14,134.60 ms |
| B5 mean | 41.73 ms |
| B5 P95 | 45.38 ms |
| B5 P99 | 46.19 ms |
| B5 deadline misses | 0/5 |

The hard subset is too small for a paper claim. It is a falsification check showing that the first infeasible-state design was invalid and that the revised difficult-but-escapable design can exercise recall.

## Coarse-to-fine latency

### Ordinary B5 cycle

| Stage | Mean | P95 | P99 |
|---|---:|---:|---:|
| Proposal | 0.974 ms | 1.131 ms | 1.242 ms |
| Coarse screen | 0.436 ms | 0.499 ms | 0.568 ms |
| Physics for 20 survivors | 1.140 ms | 1.469 ms | 1.857 ms |
| Exact certificate for 20 survivors | 28.648 ms | 30.742 ms | 31.111 ms |
| Total | 31.198 ms | 33.261 ms | 33.796 ms |

Reducing exact survivors from 50 to 20 changed P99 from 81.27 ms in v1 to 33.80 ms in v3 without reducing state-level recall on the exact subset.

### Backend scaling

| Candidates | NumPy mean / P99 | Torch CPU mean / P99 | CUDA E2E mean / P99 |
|---:|---:|---:|---:|
| 1,000 | 3.759 / 4.118 ms | 4.685 / 5.315 ms | 37.060 / 211.266 ms |
| 10,000 | 37.576 / 46.615 ms | 32.854 / 37.845 ms | 20.954 / 23.048 ms |
| 100,000 | 686.627 ms | 206.768 ms | 151.296 ms |

The 100k backend row has one repeat, so it is a throughput observation, not a valid percentile estimate. CUDA loses at 1k because transfer/startup dominates and wins at 10k. End-to-end values include transfer.

## History and motion-vector ablation

### Constant obstacle velocity

| Input | L | Velocity MAE | Hazard recall | Hazard precision |
|---|---:|---:|---:|---:|
| Current state, zero obstacle-velocity prior | — | 1.5004 | 59.55% | 72.93% |
| Raw LiDAR history | 16 | 2.4815 | 98.80% | 46.96% |
| History + ego compensation | 2 | 1.1192 | 79.74% | 72.51% |
| History + ego compensation | 4 | 0.3734 | 93.00% | 92.69% |
| History + ego compensation | 8 | 0.1259 | 97.72% | 96.47% |
| History + ego compensation | 16 | 0.0440 | 98.50% | 99.70% |

History helps when constant motion is valid because multiple causal frames average range noise. Physical ego compensation is decisive: raw L16 MAE is 2.4815, while compensated L16 is 0.0440.

### Abrupt obstacle velocity change

| Input | L | Velocity MAE | Hazard recall | Hazard precision |
|---|---:|---:|---:|---:|
| Current-state zero prior | — | 1.4886 | 62.90% | 73.42% |
| History + ego compensation | 2 | 1.1219 | 81.51% | 74.16% |
| History + ego compensation | 4 | 0.6877 | 84.92% | 88.13% |
| History + ego compensation | 8 | 1.5535 | 65.19% | 66.45% |
| History + ego compensation | 16 | 1.8471 | 59.13% | 62.03% |

Long history is harmful after abrupt motion. L16 becomes worse than the current-state prior in MAE. Therefore “longer history is better” is rejected; a residual-based causal window selector remains an unverified design possibility.

## Safety, progress, and energy interpretation

- Trajectory safety: 54/54 selected non-fallback sequences passed every continuous-time interval certificate. This is a certificate result, not a closed-loop collision rate.
- Uncertified fallbacks: 1/50 ordinary states and 0/5 hard states. The fallback was not executed.
- Freeze rate: not measured for B5.
- Path ratio: not measured for B5.
- Energy vs B1/B2: not measured on a comparable B5 closed-loop stream.
- Recursive feasibility: terminal-backup interface exists; no concrete backup set was instantiated.

## Method tree

| Method | Why tried | Theory | Result | Keep / reject | Reason |
|---|---|---|---|---|---|
| B0 frozen SAC | Performance lower bound | none | 91.2% collision rollouts | KEEP baseline | Shows safety necessity |
| B1 sampled-data HOCBF | Current certified baseline | sampled-data pointwise CBF | 0 collisions, 99.7% success | KEEP baseline | Best established safety reference |
| B2 Candidate A | Reduce B1 energy | pointwise objective only | -3.817% energy | KEEP baseline | Best closed-loop method |
| B3 random safe-MPPI-style | Test structured proposal need | finite candidate certificate | 100% subset recall | KEEP baseline | History advantage absent |
| B4 mixture + exact all | High-cost recall reference | exact finite-horizon safety | 1.44–14.13 s/state | KEEP oracle only | Too slow |
| B5 mixture coarse-to-fine | Recover recall under 50 ms | exact safety after prune | P99 33.80/46.19 ms | KEEP engineering candidate | Real-time state-level result |
| Learned proposal | Proposal compression | verifier-independent safety | not trained | REJECT for now | B3 already matches recall |
| Motion-vector history | Remove ego confounding | causal physical derivation | L16 MAE 2.4815 to 0.0440 | KEEP feature | Strong synthetic mechanism |
| Fixed long history | Average frames | none | fails under abrupt change | REJECT | Stale history |
| Approximate tube | Cheap safety transfer | conditional triangle theorem | no certified adaptive epsilon | REJECT as novelty | Central object missing |
| Certified coarse rejection | Safe pruning | valid upper-clearance theorem | unit-tested | KEEP utility | Correct but elementary |
| Terminal backup | Recursive feasibility | standard shifted-sequence proof | interface only | REJECT as novelty | Prior art |
| CLF progress | Avoid freeze | conditional no-freeze result | state-level only | KEEP secondary | Closed-loop evidence missing |
| Energy ranking | Avoid pointwise objective | candidate-set optimality | implemented | KEEP secondary | No closed-loop delta |

## Experimental verdict

B5 is a useful engineering prototype: it reduces exact-all verification by roughly two orders of magnitude and meets 50 ms in the bounded subsets. It does not outperform random B3 on safe-state recall, and history is not part of the static-obstacle B5 benchmark. The history diagnostic validates ego compensation under smooth motion and falsifies fixed long histories under abrupt changes. These results do not support a paper-core theoretical method.
