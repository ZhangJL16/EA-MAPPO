# JSEB Navigation Repair Runtime Profile

Date: 2026-08-27

## Measured wall time

- R1 started at 2026-08-27 01:29:21 CST.
- `checkpoint_transition_500000.zip` was written at 11:53:15 CST.
- Exact 500,000-transition training wall time: 37,434.9 s (10.40 h).
- Aggregate training throughput: 13.36 environment transitions/s.
- Vector-step throughput with 8 environments: 1.67 calls/s.
- TensorBoard `time/fps` decreased from 17 early in training to 13 near 500k.
- The prior 500-task fixed navigation evaluation executed 600,409 additional environment transitions and took 6,422.7 s (1.78 h) with six workers.

## Root causes

1. Training uses `DummyVecEnv`, so the eight expensive environments are stepped sequentially in one Python process. `num_envs=8` gives batched data accounting and batched policy inference, not eight-way CPU simulation.
2. Each policy transition contains four 0.05 s physics substeps. HOCBF and LiDAR are recomputed at every substep, so 500k policy transitions induce up to 2 million safety-filter calls and 2 million LiDAR updates.
3. A short isolated benchmark with the formal 128 x 8 LiDAR and 24 obstacles measured 39.5 ms per transition. HOCBF consumed 84.2% of environment-step time and LiDAR updates consumed 8.1%.
4. The HOCBF solver itself is not the main cost. In a detailed profile, constraint construction took 3.72 s and the QP solver only 0.136 s over 100 transitions. The filter builds constraints for every LiDAR hit before selecting top-16 constraints.
5. The sampled scene produced about 264 candidate point-obstacle constraints per HOCBF call but only 0.2 active constraints on average. Every LiDAR hit is converted into a Python `SphericalObstacle`, all constraints are built and sorted, then most are dropped.
6. SAC performs approximately one gradient update per transition (`gradient_steps=-1` with eight environments). About 495k updates with batch size 256 process roughly 127 million replay samples. Separate actor and critic structured-LiDAR feature extractors repeatedly run the convolutional encoder.
7. Final evaluation is inherently large: the fixed 500-task set may require more transitions than training itself. It is parallel, but remains CPU/HOCBF bound; sampled GPU utilization during evaluation was about 3%.
8. Evaluation workers also create many BLAS/PyTorch threads. Small matrix operations do not benefit much, and thread oversubscription plus synchronous worker stepping reduces scaling.

## Approximate training-time decomposition

Using the 200-transition isolated benchmark as an order-of-magnitude estimate:

| Component | Approximate contribution |
|---|---:|
| Environment simulation | 5.5 h |
| HOCBF within environment | 4.6 h |
| LiDAR update within environment | 0.4 h |
| SAC updates, neural inference, callbacks and checkpointing | 4.9 h |
| Total measured training | 10.4 h |

The decomposition is approximate because the microbenchmark ran while final evaluation workers were active. The dominance ordering is robust across repeated samples.

## Recommended optimization order

1. Replace training `DummyVecEnv` with a tested subprocess/vector pool so HOCBF environments execute on separate physical CPU cores. Preserve exact transition accounting and replay semantics.
2. Vectorize HOCBF constraint construction and top-k scoring. Avoid allocating one Python dataclass per LiDAR hit. This can preserve exact safety semantics.
3. Preselect candidates conservatively before full HOCBF construction, or aggregate rays by physical obstacle identity using a proved conservative rule. Do not simply subsample rays because that can weaken the hard safety layer.
4. Set `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, and PyTorch worker threads to one for subprocess environments; benchmark six versus eight evaluation workers.
5. Cache/reuse geometry that is invariant across the four physics substeps where mathematically valid. Do not reduce the 20 Hz HOCBF rate without a sampled-data safety argument.
6. Only after environment optimization, profile SAC kernels. The mandated one-update-per-transition ratio should not be changed silently; batching more vector collection steps per training call is a separate controlled optimization.

## Interpretation

- The run is slow but not hung.
- Eight training environments are not overloading the machine; they are currently serial inside `DummyVecEnv`.
- The 1024-ray scan alone is not the dominant cost. The expensive operation is treating hundreds of LiDAR hits as separate HOCBF obstacle constraints four times per policy transition.
- Phase-end-only evaluation already removed repeated 50k evaluations. The remaining final Gate still costs about 1.8 h because it is a 500-task, long-distance test containing roughly 600k simulation transitions.

## Instrumentation changelog

| File | Change type | Purpose |
|---|---|---|
| `profile_output/jseb_runtime_20260827/profile_env_step.py` | created | Isolated timing wrappers for environment step, LiDAR and HOCBF calls |
| `profile_output/jseb_runtime_20260827/env_step_profile.json` | created | 200-transition benchmark output |
| `profile_output/jseb_runtime_20260827/env_step_profile.txt` | created | Console copy of the 200-transition benchmark |
| `profile_output/jseb_runtime_20260827/env_step_profile_detailed.json` | created | HOCBF build/solver timing output |
| `profile_output/jseb_runtime_20260827/env_step_profile_candidates.json` | created | Candidate/active constraint audit |
| `profile_output/jseb_runtime_20260827/REPORT.md` | created | This report |

No production source file was modified for profiling.
