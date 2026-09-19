# Public Source Location Finding: budget-reading DEV pilot

## Material Passport

Mode: run. Evidence: actual local GPU training and paired simulation evaluation.
Status: first bounded prototype completed; not paper-score reproduction or confirmation.
No FPL/UAV data, exact teachers, test/OOD roots or private CONFIRM used.

## Protocol

3 seeds × 5 models; 600 updates each. Train outer=64, inner=128; mixed H=[4, 8]. Native DAD reference uses fixed H=8.
Evaluation: 512 independent rollouts/checkpoint/H, L=4096; latent/noise/contrastive draws paired across methods and training seeds. Final checkpoint only.
H=6/12 are developer interpolation/extrapolation probes, not final unseen confirmation.

## sPCE lower-bound estimate (nats, mean ± SD across 3 training seeds)

| Method | H=4 | H=6 | H=8 | H=12 |
|---|---:|---:|---:|---:|
| pool | 2.6573 ± 0.1444 | 3.6859 ± 0.4144 | 4.1748 ± 0.4353 | 5.1472 ± 0.7146 |
| fixed_attention | 2.6350 ± 0.1696 | 3.6319 ± 0.4369 | 4.1578 ± 0.3679 | 5.0480 ± 0.7505 |
| budget_attention | 2.6265 ± 0.0769 | 3.5919 ± 0.2698 | 4.0879 ± 0.3090 | 5.0022 ± 0.6421 |
| pool_wide | 2.5744 ± 0.1081 | 3.4047 ± 0.1570 | 4.1162 ± 0.2283 | 4.8425 ± 0.5515 |
| dad_native | 2.3492 ± 0.0720 | 3.3225 ± 0.0961 | 4.0324 ± 0.0398 | 4.7141 ± 0.5792 |

## Paired candidate-minus-control lower-bound differences

Positive favors budget-conditioned reading. Each cell lists all three training-seed differences.

| Control | H | seed differences | mean | conditional paired MC SE |
|---|---:|---|---:|---:|
| budget_attention - pool/normal | 4 | -0.0512, 0.0452, -0.0863 | -0.0308 | 0.0213 |
| budget_attention - pool/normal | 6 | -0.0759, 0.0443, -0.2503 | -0.0940 | 0.0289 |
| budget_attention - pool/normal | 8 | 0.0475, -0.0319, -0.2764 | -0.0869 | 0.0306 |
| budget_attention - pool/normal | 12 | -0.1050, -0.0884, -0.2415 | -0.1450 | 0.0403 |
| budget_attention - fixed_attention/normal | 4 | -0.0018, 0.0811, -0.1046 | -0.0084 | 0.0214 |
| budget_attention - fixed_attention/normal | 6 | 0.0155, 0.1031, -0.2385 | -0.0400 | 0.0254 |
| budget_attention - fixed_attention/normal | 8 | 0.0359, -0.0647, -0.1807 | -0.0698 | 0.0269 |
| budget_attention - fixed_attention/normal | 12 | -0.1033, 0.0847, -0.1188 | -0.0458 | 0.0327 |
| budget_attention - pool_wide/normal | 4 | 0.1462, -0.1487, 0.1590 | 0.0521 | 0.0425 |
| budget_attention - pool_wide/normal | 6 | 0.3434, -0.2851, 0.5033 | 0.1872 | 0.0537 |
| budget_attention - pool_wide/normal | 8 | 0.0823, -0.6114, 0.4442 | -0.0283 | 0.0626 |
| budget_attention - pool_wide/normal | 12 | 0.5739, -1.1641, 1.0691 | 0.1596 | 0.0649 |
| budget_attention - budget_attention/query_constant_4 | 4 | -0.0048, -0.0070, -0.0090 | -0.0069 | 0.0025 |
| budget_attention - budget_attention/query_constant_4 | 6 | -0.0049, 0.0075, 0.0035 | 0.0021 | 0.0031 |
| budget_attention - budget_attention/query_constant_4 | 8 | -0.0038, -0.0140, 0.0356 | 0.0059 | 0.0047 |
| budget_attention - budget_attention/query_constant_4 | 12 | 0.0441, -0.0350, 0.0315 | 0.0135 | 0.0115 |
| budget_attention - budget_attention/query_zero | 4 | 0.0013, 0.0016, 0.0013 | 0.0014 | 0.0008 |
| budget_attention - budget_attention/query_zero | 6 | -0.0015, -0.0111, 0.0000 | -0.0042 | 0.0045 |
| budget_attention - budget_attention/query_zero | 8 | 0.0063, -0.0635, 0.0163 | -0.0136 | 0.0077 |
| budget_attention - budget_attention/query_zero | 12 | 0.0520, -0.0535, -0.0078 | -0.0031 | 0.0135 |

## Estimator interval diagnostic (means, NOT a confidence interval)

| Method | H | lower | upper | upper − lower |
|---|---:|---:|---:|---:|
| pool | 4 | 2.6573 | 2.6836 | 0.0263 |
| pool | 6 | 3.6859 | 3.7692 | 0.0833 |
| pool | 8 | 4.1748 | 4.3401 | 0.1653 |
| pool | 12 | 5.1472 | 5.6758 | 0.5286 |
| fixed_attention | 4 | 2.6350 | 2.6556 | 0.0206 |
| fixed_attention | 6 | 3.6319 | 3.6987 | 0.0668 |
| fixed_attention | 8 | 4.1578 | 4.3131 | 0.1553 |
| fixed_attention | 12 | 5.0480 | 5.5117 | 0.4637 |
| budget_attention | 4 | 2.6265 | 2.6495 | 0.0230 |
| budget_attention | 6 | 3.5919 | 3.6469 | 0.0549 |
| budget_attention | 8 | 4.0879 | 4.2099 | 0.1220 |
| budget_attention | 12 | 5.0022 | 5.4906 | 0.4884 |
| pool_wide | 4 | 2.5744 | 2.5924 | 0.0180 |
| pool_wide | 6 | 3.4047 | 3.4618 | 0.0571 |
| pool_wide | 8 | 4.1162 | 4.2919 | 0.1756 |
| pool_wide | 12 | 4.8425 | 5.2222 | 0.3796 |
| dad_native | 4 | 2.3492 | 2.3623 | 0.0132 |
| dad_native | 6 | 3.3225 | 3.3715 | 0.0490 |
| dad_native | 8 | 4.0324 | 4.1426 | 0.1102 |
| dad_native | 12 | 4.7141 | 5.0893 | 0.3752 |

## Costs

| Method | parameters | total train seconds (3 seeds) | CPU ms/decision | GPU ms/decision |
|---|---:|---:|---:|---:|
| pool | 1332 | 28.73 | 0.0179 | 0.1704 |
| fixed_attention | 1348 | 32.13 | 0.0279 | 0.2178 |
| budget_attention | 1364 | 28.23 | 0.0314 | 0.2340 |
| pool_wide | 1372 | 27.11 | 0.0186 | 0.1710 |
| dad_native | 1330 | 34.44 | 0.0168 | 0.1589 |

Total synchronized training wall time on GPU: 150.63s (0.0418 device-hours, not GPU kernel-active hours).
Training includes random-input transfer and optimization; excludes checkpoint serialization and evaluation. Native fixed-H control uses a different horizon workload; see exact simulated measurement/likelihood counts in summary.json.
The first invalid evaluation briefly overlapped the tail of training; training seconds are accounting, not isolated speed benchmarks. Corrected inference latencies were measured after training completed.
Latency: batch=1, four public history pairs, remaining=4, 20 warm-ups + 100 synchronized calls, one CPU thread. Not whole-deployment latency or hardware-independent FLOP equivalence.

## Interpretation limits

This small optimization budget tests implementation and an initial mechanism hypothesis, not converged optimal performance. Three training repeats do not support a population superiority claim. The 512 rollouts are not 512 independently trained models. Training-seed SD and conditional MC SE are distinct; steps/horizons are not independent replicates. No p-values or best-seed selection.
sPCE is a finite-inner lower bound and sNMC an upper bound in expectation, not exact EIG. The lower bound has ceiling log(4097)=8.3180; a large lower/upper gap weakens rankings as claims about true EIG.
Fixed-query attention and wider pooling are essential controls. A tiny positive number or query-intervention sensitivity alone does not establish a useful new mechanism. Any gains must survive stronger runs and independent tasks.
Step-DAD, original full T=30 training, per-H specialists and external-task transfer were NOT completed. No novelty claim and no Spotlight rating follows from this run.

## Corrected implementation error

An initial evaluation omitted loading trained weights and evaluated initialized models. Those original evaluation files are retained with INVALID_EVALUATION_NOTICE.json and excluded here. All trained checkpoints were migrated byte-for-byte to a fresh corrected evaluation directory; a checkpoint loading regression test now checks exact tensor identity and changed predictions. No retraining, changed objective, seed replacement or performance-based selection occurred. See evaluation_manifest.json for both source bindings and checkpoint hashes.

Sources: [DAD paper](https://proceedings.mlr.press/v139/foster21a.html), [pinned author code](https://github.com/ae-foster/dad/tree/4b1008174e1531d1f14601d83cef481c0f586f36).
