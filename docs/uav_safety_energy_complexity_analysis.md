# UAV Safety + Energy Joint Action Filter: Complexity Analysis

## Scope

The control dimension is fixed at (n=3). Let (m) be the number of retained obstacle constraints, (K) the top-K size, (N_f) the number of polygon facets, and (I) the number of coordinate sweeps used by the prototype QP solver. Timing numbers below are controlled diagnostics, not final hardware claims.

## Solver Classes

| Method | Optimization class | Build complexity | Solve complexity | Memory | Important caveat |
| --- | --- | --- | --- | --- | --- |
| Single barrier projection | closed form | (O(n)) | (O(n^3)) if metric inverse is not cached; (O(n^2)) when cached | (O(n^2)) | actuator or velocity boundaries can invalidate the one-row formula |
| Full obstacle-wise HOCBF | convex QP with polygonal physical limits | (O(mn)) | prototype Hildreth: (O(Imn)) after (O(n^3)) inverse | (O(mn)) | infeasible cases can use many sweeps |
| Top-K HOCBF | constraint selection plus QP | (O(mn+m\log m)) with sorting | (O(IKn)) | (O(mn)) before pruning | selecting after building all rows does not remove the (O(m)) front end |
| Aggregate HOCBF-QP | soft-min reduction plus small QP | (O(mn)) | (O(I(N_f+1)n)) | (O(mn)) | gradient cancellation can destroy control authority |
| Aggregate closed form | soft-min plus one-row projection | (O(mn)) | (O(n^2)) with cached inverse | (O(mn)) | current compute-only version omits actuator constraints |
| Exact horizontal norm HOCBF | SOCP-representable | (O(mn)) | solver dependent; interior-point Newton systems | solver dependent | stronger geometric fidelity but heavier runtime stack |
| Candidate A | strictly convex QP | same as chosen collision representation | same order as B2/B3 | same | energy changes Hessian but not constraint class |
| Candidate B | Candidate A QP plus one fixed-scale linear term | same plus one energy-model backward pass | same QP order | same plus model activations | gradient inference, not QP solve, dominates its incremental latency |
| Candidate C | convex QCQP / possible SOCP | collision build plus energy derivatives | conic solver dependent | solver dependent | not selected; learned bound invalidates deterministic claim |

For the current implementation, the actuator disk is conservatively represented by 32 facets plus two vertical rows. Optional one-step velocity limits add another 34 rows. These physical rows are constant in count even when obstacle count grows.

## 20 Hz Deadline

The safety deadline is 50 ms. Total safety-layer latency must include:

1. 3D LiDAR ray processing;
2. hit-to-primitive or point extraction;
3. barrier construction and danger ranking;
4. solver runtime;
5. post-solve diagnostics.

Solver time alone is not an acceptable real-time claim.

## Preliminary Constraint-Scaling Diagnostic

Artifact: `artifacts/uav_safety_filter_parallel_smoke_20260819_015609/compute_scaling.json`.

This diagnostic used only two repeats per point on the current CPU and excludes LiDAR ray casting. It is useful for locating bottlenecks but too small for stable P95/P99 claims.

| (m) | Method | Mean total ms | P95 total ms | Mean solver ms | Deadline misses |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | full QP | 0.350 | 0.504 | 0.189 | 0% |
| 1 | top-K-16 QP | 0.230 | 0.282 | 0.123 | 0% |
| 1 | aggregate QP | 0.327 | 0.444 | 0.175 | 0% |
| 1 | aggregate closed form | 0.125 | 0.162 | 0.054 | 0% |
| 64 | full QP | 1.963 | 1.968 | 0.229 | 0% |
| 64 | top-K-16 QP | 1.901 | 1.903 | 0.126 | 0% |
| 64 | aggregate QP | 1.653 | 1.654 | 0.111 | 0% |
| 64 | aggregate closed form | 1.517 | 1.560 | 0.031 | 0% |
| 256 | full QP | 7.248 | 7.621 | 0.536 | 0% |
| 256 | top-K-16 QP | 7.362 | 7.464 | 0.177 | 0% |
| 256 | aggregate QP | 6.174 | 6.496 | 0.143 | 0% |
| 256 | aggregate closed form | 5.869 | 5.934 | 0.042 | 0% |
| 1024 | full QP | 25.969 | 26.798 | 1.579 | 0% |
| 1024 | top-K-16 QP | 26.543 | 27.590 | 0.224 | 0% |
| 1024 | aggregate QP | 21.703 | 22.070 | 0.199 | 0% |
| 1024 | aggregate closed form | 22.707 | 22.972 | 0.065 | 0% |

## Bottleneck Finding

At 1,024 rows, top-K reduces mean solver time from about 1.58 ms to 0.22 ms but does not reduce total time because all 1,024 constraints are still constructed and sorted. The observed total is about 26.5 ms. Therefore the primary dense-LiDAR optimization target is obstacle extraction and vectorized row construction, not merely a smaller QP.

Aggregate methods also retain (O(m)) barrier construction. Their constant-size solve is useful, but the current benefit is only about 4--5 ms at 1,024 rows, and aggregate safety semantics are weaker in symmetric scenes.

## Thirty-Repeat Constraint-Scaling Result

Artifact: `artifacts/uav_safety_filter_1000_controlled_20260819_022302/compute_scaling.json`.

The final controlled scaling diagnostic uses 30 repeats per constraint count.
It still excludes the separate 1,024-ray LiDAR measurement because the goal is
to isolate row construction, selection, aggregation, and solving. All methods
met the 50 ms deadline in this feasible synthetic scaling set.

| m | Method | Mean | Median | P90 | P95 | P99 | Max | Mean solve |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | full QP | 0.178 | 0.156 | 0.184 | 0.316 | 0.480 | 0.509 | 0.102 |
| 1 | top-K-16 QP | 0.178 | 0.163 | 0.211 | 0.258 | 0.320 | 0.332 | 0.092 |
| 1 | aggregate QP | 0.201 | 0.187 | 0.241 | 0.257 | 0.271 | 0.272 | 0.092 |
| 64 | full QP | 1.804 | 1.779 | 2.039 | 2.109 | 2.208 | 2.230 | 0.203 |
| 64 | top-K-16 QP | 2.027 | 1.935 | 2.431 | 2.454 | 2.892 | 3.070 | 0.131 |
| 64 | aggregate QP | 1.617 | 1.550 | 1.812 | 1.906 | 2.292 | 2.419 | 0.102 |
| 256 | full QP | 6.534 | 6.461 | 7.208 | 7.499 | 7.639 | 7.694 | 0.485 |
| 256 | top-K-16 QP | 7.015 | 6.985 | 7.537 | 7.562 | 7.629 | 7.655 | 0.175 |
| 256 | aggregate QP | 5.583 | 5.619 | 5.927 | 6.024 | 6.502 | 6.682 | 0.124 |
| 1024 | full QP | 27.899 | 27.357 | 29.148 | 31.931 | 36.489 | 37.806 | 1.566 |
| 1024 | top-K-16 QP | 29.665 | 29.413 | 30.859 | 31.050 | 33.279 | 34.142 | 0.232 |
| 1024 | aggregate QP | 23.868 | 23.021 | 25.912 | 29.355 | 33.162 | 33.709 | 0.199 |
| 1024 | aggregate closed form | 23.047 | 22.984 | 23.831 | 24.740 | 25.536 | 25.801 | 0.068 |

All values are milliseconds. Top-K substantially reduces solver time but is
slower in end-to-end row-level time because the current prototype still builds
and sorts every row. Aggregate construction is faster, but the rollout study
shows that its weaker geometry can be catastrophically unsafe; compute alone
does not justify selecting it.

## Top-K Priority Stress Test

Artifact: `artifacts/uav_safety_filter_1000_controlled_20260819_022302/top_k_priority.json`.

In 100 trials per count with K=8 and deliberately mixed near-receding and
far-closing obstacles, TTC and nominal HOCBF slack had zero trials with an
omitted violated row at m=32, 64, 128, and 256. Distance-only omission rates
were 30%, 26%, 23%, and 23%; closing-speed-only rates were 10%, 15%, 20%, and
18%. The result supports HOCBF slack or TTC-aware selection and rejects pure
distance or closing speed as the primary selector. This is a constructed
selection stress test, not a forward-invariance proof after row dropping.

## Required Runtime Protocol

Before any real-time claim:

1. use at least 30 repeats per constraint count after warm-up;
2. report mean, median, P90, P95, P99, and max;
3. separate LiDAR, extraction, build, solve, and total times;
4. record raw points, candidate primitives, and retained rows;
5. run on a pinned CPU configuration with thread counts and hardware recorded;
6. report misses against 50 ms;
7. test infeasible cases separately because solver iteration counts differ sharply.

## Current Compute Conclusion

**20 HZ COMPUTE FEASIBILITY IS EMPIRICALLY SUPPORTED FOR THIS CPU PROTOTYPE, NOT A HARD REAL-TIME GUARANTEE.** The 30-repeat 1,024-row stress test stayed below 50 ms, while the 125-scene rollout benchmark exposed rare deadline misses from infeasible solver tails and framework cold starts. The selected sampled-data energy-aware filter at weight 0.1 had a sub-0.1% miss rate in controlled rollouts. Hardware scheduling, perception uncertainty, and worst-case infeasibility remain outside a deterministic timing guarantee.
